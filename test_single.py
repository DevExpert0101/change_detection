"""
test on standard SCD datasets and ChangeVPR (or own image pairs)
"""
import os 
import numpy as np
import cv2
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import matplotlib.pyplot as plt

# from .framework import GeSCF
from framework_ import GeSCF
# from .framework import GeSCF
from utils import calculate_metric, show_mask_new
from ultralytics import YOLO

model = YOLO("./pretrained_weight/yolov10x.pt")

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
    # Draw bounding boxes around the detected contours
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours based on area (adjust as needed)
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
            # Draw the bounding box on the second image (img_t1)
            # cv2.rectangle(img_t1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness=2
    merged_bboxes = merge_bounding_boxes(bboxes)
    
    original_h, original_w, _ = img_t1.shape
    for mbbox in merged_bboxes:
        x, y, w, h = mbbox
        if h > original_h * 0.7 or w > original_w * 0.7:
            continue
        cv2.rectangle(img_t1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness=2
        
    return img_t1

def test_single_image(dataset, img_t0_path, img_t1_path, gt_path=None, save_img=False):
    model = GeSCF(dataset=dataset, feature_facet='key', feature_layer=17, embedding_layer=32)
    
    # load image pairs
    img_t0 = cv2.imread(img_t0_path)
    rgb_img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    img_t1 = cv2.imread(img_t1_path)
    rgb_img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
           
    # inference
    final_change_mask = model(img_t0_path, img_t1_path)    
    
    # Detect objects in the change mask
    
    
    if gt_path:
        gt = cv2.imread(gt_path, 0) / 255.
        precision, recall = calculate_metric(gt, final_change_mask)
        f1score = 2 * (precision * recall) / (precision + recall)
    
    mask_uint8 = final_change_mask.astype(np.uint8) * 255
    final_change_mask = cv2.resize(mask_uint8, (rgb_img_t1.shape[1], rgb_img_t1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    img_t1_with_objects = detect_bounding_boxes_in_mask(final_change_mask, img_t1)
    # visualization
    fig = plt.figure(figsize=(18,6))
    fig.add_subplot(141)
    plt.title('img t0')
    plt.imshow(rgb_img_t0)
    plt.axis('off')

    fig.add_subplot(142)
    plt.title('img t1')
    plt.imshow(rgb_img_t1)
    plt.axis('off')
    
    fig.add_subplot(143)
    plt.title('final change mask')
    plt.imshow(rgb_img_t1)
    show_mask_new(final_change_mask.astype(np.float32), plt.gca())
    plt.axis('off')
    
    if gt_path:
        fig.add_subplot(144)
        plt.title('GT')
        plt.imshow(rgb_img_t0)
        show_mask_new(gt.astype(np.float32), plt.gca())
        plt.axis('off')
    
    plt.show()
    
    del model
    if gt_path:
        logging.info(f'Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1score*100:.1f}')
        return precision, recall, f1score
    
    return img_t1_with_objects

if __name__ == '__main__':
    
    dataset = "Random" # 'VL_CMU_CD', 'TSUNAMI', 'ChangeSim', 'ChangeVPR', 'Remote_Sensing', 'Random'
    split = "Lucia/Nordland" # one of SF-XL/St or Lucia/Nordland

    img_t0_path = "../images/clean.jpg"
    img_t1_path = "../images/dirty.jpg"
    gt_path = None

    img_t1_with_objects = test_single_image(dataset, img_t0_path, img_t1_path, gt_path)
    
    cv2.imwrite('img_t1_with_detected_objects.jpg', img_t1_with_objects)