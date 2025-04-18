from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import base64

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# from .framework import GeSCF
from framework_ import GeSCF
# from .framework import GeSCF
from utils import calculate_metric, show_mask_new
from ultralytics import YOLO
import torchvision.ops as ops
import torch

model = YOLO("./pretrained_weight/yolov10x.pt")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model = YOLO("./pretrained_weight/yolov10x.pt")
model = GeSCF(dataset="Random", feature_facet="key", feature_layer=17, embedding_layer=32)


def read_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to OpenCV format"""
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

def is_overlapping(new_box, existing_boxes, iou_threshold=0.7):
    x1, y1, x2, y2 = new_box
    new_area = (x2 - x1) * (y2 - y1)

    for ex in existing_boxes:
        ex1, ey1, ex2, ey2 = ex
        inter_x1 = max(x1, ex1)
        inter_y1 = max(y1, ey1)
        inter_x2 = min(x2, ex2)
        inter_y2 = min(y2, ey2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = new_area + (ex2 - ex1) * (ey2 - ey1) - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        if iou > iou_threshold:
            return True
    return False

# Apply NMS on final set of boxes
def apply_global_nms(bboxes, iou_threshold=0.5):
    if not bboxes:
        return []

    boxes_tensor = torch.tensor([box for _, box in bboxes], dtype=torch.float)
    scores = torch.tensor([1.0] * len(bboxes))  # Dummy scores or use real confidences
    indices = ops.nms(boxes_tensor, scores, iou_threshold)

    return [bboxes[i] for i in indices]

def merge_bounding_boxes(boxes):
            
    def do_boxes_overlap(box1, box2):
        """Checks if two bounding boxes overlap (including touching)."""
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        return not (x2 < x1_ or x2_ < x1 or y2 < y1_ or y2_ < y1)
    
    def is_box_inside(inner, outer):
        """Check if one box is fully inside another"""
        x1, y1, x2, y2 = inner
        ox1, oy1, ox2, oy2 = outer
        return x1 >= ox1 and y1 >= oy1 and x2 <= ox2 and y2 <= oy2

    print(15)
    # Step 1: Ensure all boxes are in (x1, y1, x2, y2) format
    labels = [label for label, box in boxes if len(box) == 4]
    # boxes = [box for label, box in boxes if len(box) == 4]
    
    # Separate beds from other boxes
    bed_boxes = [box for label, box in boxes if label == 'bed']
    other_boxes = [(label, box) for label, box in boxes if label != 'bed']
    
    # Split other boxes into inside-bed and outside-bed
    inside_boxes = []
    outside_boxes = []
    print(16)
    for label, box in other_boxes:
        if any(is_box_inside(box, bed) for bed in bed_boxes):
            inside_boxes.append((label, box))
        else:
            outside_boxes.append((label, box))
    
    # boxes = [box for box in boxes if box is not None]  # Remove invalid entries

    print(17)
    # Build adjacency list for merging
    boxes_only = [box for _, box in inside_boxes]
    adjacency_list = {i: set() for i in range(len(boxes_only))}
    for i in range(len(boxes_only)):
        for j in range(i + 1, len(boxes_only)):
            if do_boxes_overlap(boxes_only[i], boxes_only[j]):
                adjacency_list[i].add(j)
                adjacency_list[j].add(i)
    print(18)
    # Step 3: Find connected components (clusters of overlapping boxes)
    visited = set()
    merged_boxes = []

    def dfs(node, cluster):
        """Depth-First Search (DFS) to find all connected bounding boxes."""
        if node in visited:
            return
        visited.add(node)
        cluster.append(boxes_only[node])
        for neighbor in adjacency_list[node]:
            dfs(neighbor, cluster)

    for i in range(len(boxes_only)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)

            # Step 4: Merge the bounding boxes in the found cluster
            merged_x1 = min(b[0] for b in cluster)
            merged_y1 = min(b[1] for b in cluster)
            merged_x2 = max(b[2] for b in cluster)
            merged_y2 = max(b[3] for b in cluster)

            merged_boxes.append(('merged', (merged_x1, merged_y1, merged_x2, merged_y2)))
    print(19)
    final_boxes = []
    final_boxes.extend([('bed', box) for box in bed_boxes])
    final_boxes.extend(merged_boxes)
    final_boxes.extend([(label, box) for label, box in outside_boxes])

    return final_boxes

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def detect_bounding_boxes_in_mask(final_change_mask, img_t1):
    """
    Detects bounding boxes for the masked parts in the second image (img_t1).
    """
    # Convert the mask to binary (thresholding)
    _, binary_mask = cv2.threshold(final_change_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    original_h, original_w, _ = img_t1.shape
    print('w, h', original_w, original_h)
    # Draw bounding boxes around the detected contours
    processed_crop_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours based on area (adjust as needed)
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            # if h > original_h * 0.7 or w > original_w * 0.7:
            #     continue    
            if h < original_h * 0.05 or w < original_w * 0.05:
                continue        
            print(12)
            
            x1, y1, x2, y2 = x, y, x + w, y + h
            if is_overlapping((x1, y1, x2, y2), processed_crop_boxes):
                continue
        
            padding = 400           
            crop_x1 = max(0, x - padding)
            crop_y1 = max(0, y - padding)
            crop_x2 = min(original_w, x + w + padding)
            crop_y2 = min(original_h, y + h + padding)                     
            cropped_object = img_t1[crop_y1: crop_y2, crop_x1:crop_x2].copy()
            print(13)
            yolo_result = yolo_model(cropped_object, verbose=False)
            found_detection = False
            print(14)
            
            for r in yolo_result:    
                                            
                for box in r.boxes:
                    xx1, yy1, xx2, yy2 = map(int, box.xyxy[0])
                    # if yy2 - yy1 < original_h * 0.025 or xx2 - xx1 < original_w * 0.025:
                    #     continue        
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = yolo_model.names[cls]

                    # Map back to original image coordinates
                    xx1, xx2 = max(0, x - padding) + xx1, max(0, x - padding) + xx2
                    yy1, yy2 = max(0, y - padding) + yy1, max(0, y - padding) + yy2

                    bboxes.append((label, (xx1, yy1, xx2, yy2)))   
            
            bboxes.append(('None', (x, y, x + w, y + h)))
            
            # Draw the bounding box on the second image (img_t1)
            # cv2.rectangle(img_t1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness=2
            
    # Remove overlapping boxes using IOU
    iou_threshold = 0.5
    filtered_bboxes = []

    for label, box in bboxes:
        keep = True
        for _, fbox in filtered_bboxes:
            if iou(box, fbox) > iou_threshold:
                keep = False
                break
        if keep:
            filtered_bboxes.append((label, box))
            
    merged_bboxes = merge_bounding_boxes(filtered_bboxes)
    print(20)
    image_list = []
    for label, mbbox in merged_bboxes:

        x1, y1, x2, y2 = mbbox
        # cv2.rectangle(img_t1, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, thickness=2
        image_list.append(img_t1[y1:y2, x1:x2])
        
    # cv2.imwrite('result.jpg', img_t1)
    return image_list


def inference(img_clean, img_dirty):
    print(2)
    final_change_mask = model(img_clean, img_dirty)
    print(3)
    mask_uint8 = final_change_mask.astype(np.uint8) * 255
    final_change_mask = cv2.resize(mask_uint8, (img_dirty.shape[1], img_dirty.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    image_list = detect_bounding_boxes_in_mask(final_change_mask, img_dirty)
    
    return image_list

     
@app.post("/compare")
async def compare_images(image1: UploadFile=File(...), image2: UploadFile=File(...)):
    try:
        # Read images into OpenCV format
        img1 = read_image(await image1.read())
        img2 = read_image(await image2.read())
        
        print(1)
        if img1 is None or img2 is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)
                
        image_list = inference(img1, img2)
        
        image_byte_list = []
        label_list = []

        for i in range(len(image_list)):
            image = image_list[i]
            lab = "lab_list[i]"
            if image is None:
                continue

            _, img_encoded = cv2.imencode(".png", image)
            image_byte_list.append(base64.b64encode(img_encoded).decode('utf-8'))
            label_list.append(lab)
            
        return JSONResponse(content={"images": image_byte_list, "labels": label_list})
        # return JSONResponse(content={"images": "ok", "labels": "ok"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
