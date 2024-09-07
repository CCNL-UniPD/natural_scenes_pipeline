import torch
import pickle
from GroundingDINO.groundingdino.util import box_ops
from tqdm.auto import tqdm
# segment anything
from segment_anything import build_sam, SamPredictor 
from GroundingDINO.groundingdino.util.inference import load_image
import numpy as np
import os
from scipy.ndimage import binary_fill_holes
from skimage.morphology import convex_hull_image

# Load SAM
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam_checkpoint = "...\sam_vit_h_4b8939.pth" # you need to download this sam model.
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Define directories
image_folder = '.../pascal/VOCdevkit/VOC2012/JPEGImages'

#with open(r"E:\gcloud_file\pascal_gemini.pkl", 'rb') as file:
    #rr = pickle.load(file)

with open('/Data/DINO_result_pascal.pkl', 'rb') as file:
    rr = pickle.load(file)

# Retrieve absolute paths and filenames without extensions
im_paths = {os.path.splitext(file)[0]: os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')}

resultdir = '/Data/SAM_result_pascal.pkl'

existing_rr = {}
#THRESHOLD = 0.10
# Process images and save results every 1000 images
for idx, (image_id, dino_result) in enumerate(tqdm(rr.items())):
    image_path = im_paths[image_id]
    image_source, image_dino = load_image(image_path)
    #filtered_boxes = [box for box, logit in zip(dino_result['boxes'], dino_result['logits']) if logit > THRESHOLD]
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