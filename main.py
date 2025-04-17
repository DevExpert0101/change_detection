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

def merge_bounding_boxes(boxes):
            
    def do_boxes_overlap(box1, box2):
        """Checks if two bounding boxes overlap (including touching)."""
        x1, y1, w1, h1 = box1
        x1_, y1_, w2, h2 = box2
        return not (x1 + w1 < x1_ or x1_ + w2 < x1 or y1 + h1 < y1_ or y1_ + h2 < y1)

    # Step 1: Ensure all boxes are in (x1, y1, x2, y2) format
    # labels = [label for label, box in boxes if len(box) == 4]
    # boxes = [box for label, box in boxes if len(box) == 4]
    # boxes = [box for box in boxes if box is not None]  # Remove invalid entries

    # Step 2: Build adjacency list (graph of overlapping bounding boxes)
    adjacency_list = {i: set() for i in range(len(boxes))}
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if do_boxes_overlap(boxes[i], boxes[j]):
                adjacency_list[i].add(j)
                adjacency_list[j].add(i)

    # Step 3: Find connected components (clusters of overlapping boxes)
    visited = set()
    merged_boxes = []

    def dfs(node, cluster):
        """Depth-First Search (DFS) to find all connected bounding boxes."""
        if node in visited:
            return
        visited.add(node)
        cluster.append(boxes[node])
        for neighbor in adjacency_list[node]:
            dfs(neighbor, cluster)

    for i in range(len(boxes)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)

            # Step 4: Merge the bounding boxes in the found cluster
            merged_x1 = min(b[0] for b in cluster)
            merged_y1 = min(b[1] for b in cluster)
            merged_x2 = max(b[2] for b in cluster)
            merged_y2 = max(b[3] for b in cluster)

            merged_boxes.append((merged_x1, merged_y1, merged_x2, merged_y2))

    return merged_boxes

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
    # Draw bounding boxes around the detected contours
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours based on area (adjust as needed)
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            # if h > original_h * 0.7 or w > original_w * 0.7:
            #     continue    
            if h < original_h * 0.05 or w < original_w * 0.05:
                continue        
            bboxes.append((x, y, w, h))
            # Draw the bounding box on the second image (img_t1)
            # cv2.rectangle(img_t1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness=2
    merged_bboxes = merge_bounding_boxes(bboxes)
    
    
    image_list = []
    for mbbox in merged_bboxes:
        x, y, w, h = mbbox
        cv2.rectangle(img_t1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness=2
        image_list.append(img_t1[y:y+h, x:x+w])
        
    cv2.imwrite('result.jpg', img_t1)
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
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        # Read images into OpenCV format
        img1 = read_image(await image1.read())
        img2 = read_image(await image2.read())
        
        print(1)
        if img1 is None or img2 is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)
                
        image_list= inference(img1, img2)
        
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
