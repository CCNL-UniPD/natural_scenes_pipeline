"""
Description: This script integrates the Grounding DINO model with the Segment Anything Model (SAM) to process and analyze images from the MSCOCO dataset. 
             The script loads annotations, predicts bounding boxes using DINO, and refines these results with SAM to generate masks for the detected objects.
             It then computes various properties such as hull area, total area, and object numerosity for each image, and stores the results in a pickle file for further analysis.

Main Features:
1. Loads bounding box annotations from Grounding DINO predictions for MSCOCO images.
2. Uses the Segment Anything Model (SAM) to generate segmentation masks for detected objects.
3. Computes and stores additional metrics like total mask area, convex hull area, and object numerosity.
4. Saves the enhanced results to a pickle file for easy reuse and further analysis.

Author: Kuinan
Date: 2024-09-01
Version: 1.0
"""

import torch
import pickle
from GroundingDINO.groundingdino.util import box_ops
from tqdm.auto import tqdm
# segment anything
from segment_anything import build_sam, SamPredictor 
from GroundingDINO.groundingdino.util.inference import load_image
import json
import os
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import convex_hull_image

# Load SAM
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam_checkpoint = "...\sam_vit_h_4b8939.pth" # you need to download this sam model.
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Define directories
im_dir = 'image dir for MSCOCO'

with open('your mscoco annotation dir', 'rb') as file:
    train17 = json.load(file)
with open('/Data/DINO_result_mscoco.pkl', 'rb') as file:
    rr = pickle.load(file)
# Create a mapping from image_id to annotations
image_annotations = {}
for ann in train17['annotations']:
    if ann['image_id'] not in image_annotations:
        image_annotations[ann['image_id']] = []
    image_annotations[ann['image_id']].append(ann)

# Mapping between id and filename
image_filename = {}
for im in train17['images']:
    if im['id'] not in image_filename:
        image_filename[im['id']] = im['file_name']


resultdir = '/Data/SAM_result_mscoco.pkl'
existing_rr = {}

for image_id, dino_result in tqdm(rr.items()):
    image_path = os.path.join(im_dir, image_filename[image_id])
    image_source, image_dino = load_image(image_path)
    filtered_boxes = [box for box, logit in zip(dino_result['boxes'], dino_result['logits']) if logit > 0.22]
    filtered_boxes = dino_result['boxes']
    tensor_boxes = torch.tensor(filtered_boxes)

    if tensor_boxes.nelement() == 0:
        continue

    # Set image
    sam_predictor.set_image(image_source)
    # Box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(tensor_boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
    #total_area = sum(masks.sum(dim=[2,3])).item()
    masks = masks.squeeze(dim=1).cpu().numpy()

    # Combine all masks
    combined_mask = np.any(masks, axis=0)
    total_area = combined_mask.sum()
    filled_combined_mask = binary_fill_holes(combined_mask)
    hull_mask = convex_hull_image(filled_combined_mask)
    hull_area = hull_mask.sum()

    image_size = H * W
    numerosity = len(filtered_boxes)
    existing_rr[image_id] = dino_result
    existing_rr[image_id]['hull_area'] = hull_area
    existing_rr[image_id]['total_area'] = total_area
    existing_rr[image_id]['image_size'] = image_size
    existing_rr[image_id]['numerosity'] = numerosity

# Save any remaining results
with open(resultdir, 'wb') as file:
    pickle.dump(existing_rr, file)
