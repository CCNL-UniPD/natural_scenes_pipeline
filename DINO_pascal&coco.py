"""
Description: This script loads the Grounding DINO model and performs object detection on the PASCAL VOC/MSCOCO dataset.
             It evaluates bounding boxes based on a set of textual prompts and computes the Intersection over Union (IoU) 
             for filtering the detected boxes. The results are saved in a pickle file for further analysis.
Author: Kuinan
Date: 2024-09-01
Version: 1.0
"""

from func import compute_iou
import pickle
from tqdm.auto import tqdm
import json
import os

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import predict, load_image

# diffusers
import torch
from huggingface_hub import hf_hub_download

# G-dino model load
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
print('G-dino loaded')

# define directories
image_folder = '.../pascal/VOCdevkit/VOC2012/JPEGImages' # modify with your path, either DINO or COCO
prompt_dir = '/Data/postprocess_gemini_pascal.pkl' # remember to use the pos-processed prompts for DINO

with open(prompt_dir, 'r') as f:
    prompts = json.load(f) 

# Retrieve absolute paths and filenames without extensions
im_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg')]
filenames = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.endswith('.jpg')]

filtered_results = {}
for im_path, filename in tqdm(zip(im_paths, filenames), total=len(im_paths)):
    image_source, transformed_frame = load_image(im_path)

    prompt_set = prompts.get(filename, [])
    
    if not prompt_set:
        continue

    all_boxes = []
    all_phrases = []
    all_logits = []

    for prompt in prompt_set:
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=transformed_frame,
            caption=prompt,
            box_threshold=0.05,
            text_threshold=0.25)

        all_boxes.extend([box.tolist() for box in boxes])  # Convert each tensor to list
        all_logits.extend(logits.tolist())  # Convert tensor to list
        all_phrases.extend(phrases)

    # Convert lists to tensors for processing
    stack_boxes = torch.tensor(all_boxes)
    stack_logits = torch.tensor(all_logits)

    # Compute the IOU matrix
    iou_matrix = compute_iou(stack_boxes)
    iou_threshold = 0.95
    keep = torch.ones(len(all_boxes), dtype=torch.bool)

    for i in range(len(all_boxes) - 1):
        for j in range(i + 1, len(all_boxes)):
            if iou_matrix[i, j] > iou_threshold:
                if stack_logits[i] > stack_logits[j]:
                    keep[j] = False
                else:
                    keep[i] = False

    # Filter the results
    filtered_boxes = [all_boxes[idx] for idx in torch.where(keep)[0]]
    filtered_logits = [all_logits[idx] for idx in torch.where(keep)[0]]
    filtered_phrases = [all_phrases[idx] for idx in torch.where(keep)[0]]

    # Store the results
    filtered_results[filename] = {
        'boxes': filtered_boxes,
        'logits': filtered_logits,
        'phrases': filtered_phrases
    }
with open('/Data/DINO_result_pascal.pkl', 'wb') as f:
    pickle.dump(filtered_results, f)