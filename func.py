from typing import Tuple
import numpy as np
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from pycocotools import mask as maskUtils
import torch
import cv2
import http.client
import typing
import urllib.request
from vertexai.generative_models import Image as VImage


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes

def load_image_from_url_V(image_url: str) -> VImage:
    image_bytes = get_image_bytes_from_url(image_url)
    return VImage.from_bytes(image_bytes)

def calculate_iou(box1, box2):
    """ Assumes box format [x1, y1, x2, y2] """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

def compute_iou(boxes):
    """
    Compute the intersection over union of a set of boxes with itself.
    The boxes are in (cx, cy, w, h) format.

    Arguments:
    boxes -- (N, 4) tensor of boxes

    Returns:
    iou_matrix -- (N, N) tensor of the intersection over union values
    """
    x1 = boxes[:, 0] - 0.5 * boxes[:, 2]
    y1 = boxes[:, 1] - 0.5 * boxes[:, 3]
    x2 = boxes[:, 0] + 0.5 * boxes[:, 2]
    y2 = boxes[:, 1] + 0.5 * boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)

    xi1 = torch.maximum(x1[:, None], x1)
    yi1 = torch.maximum(y1[:, None], y1)
    xi2 = torch.minimum(x2[:, None], x2)
    yi2 = torch.minimum(y2[:, None], y2)

    inter_area = torch.maximum(xi2 - xi1, torch.tensor(0.)) * torch.maximum(yi2 - yi1, torch.tensor(0.))

    union_area = area[:, None] + area - inter_area

    iou_matrix = inter_area / union_area

    return iou_matrix

def str_catid(num_id, category_mapping):
    """
    Retrieves the category name for a given category ID using a pre-built mapping.

    Parameters:
    num_id (int): The category ID.
    category_mapping (dict): A dictionary mapping category IDs to category names.

    Returns:
    str: The name of the category corresponding to the given ID.

    Raises:
    ValueError: If the category ID is not found in the mapping.
    """
    try:
        return category_mapping[num_id]
    except KeyError:
        raise ValueError("Category ID does not exist in the mapping.")
    
# Assuming a simple mapping function from predicted phrases to category IDs
def phrase_to_cat_id(phrase, category_mapping):
    for cat_id, name in category_mapping.items():
        if name == phrase:
            return cat_id
    return None

def encode_to_rle(binary_mask):
    # Assuming binary_mask is a NumPy array with shape (H, W)
    # Ensure it's a Fortran-style contiguous array
    if not binary_mask.flags['F_CONTIGUOUS']:
        binary_mask = np.asfortranarray(binary_mask)

    # Encode the mask into RLE
    rle_encoded = maskUtils.encode(binary_mask)
    # print(type(rle_encoded[0]['counts'])) the result is bytes, which is serializable
    return rle_encoded

def decode_from_rle(rle_encoded):
    # Decode RLE back to mask
    binary_mask_decoded = maskUtils.decode(rle_encoded)
    return binary_mask_decoded

def load_image_from_url(url: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Fetch the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    image_source = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Process the image
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    
    return image, image_transformed

def show_mask(masks, image, random_color=True):
    # Convert masks to numpy if they're not already (assuming masks is a tensor)
    masks = masks.cpu().numpy()

    # Convert the input image to an RGBA PIL image
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")

    # Iterate over each mask
    for mask in masks:
        # Determine the color for the mask
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)  # Random color with alpha
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])  # Fixed color with alpha
        
        h, w = mask.shape[-2:]  # Height and width of the mask
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # Apply color to the mask

        # Convert the colored mask to an RGBA PIL image
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

        # Composite the mask with the current image
        annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)

    # Return the final composited image as a numpy array
    return np.array(annotated_frame_pil)

def add_caption(image, text, position=(20, 20), font_size=35):
    draw = ImageDraw.Draw(image)
    # Load the default PIL font
    font = ImageFont.load_default()
    # Get the size of the text to create a black background
    text_size = draw.textsize(text, font=font)
    # Calculate the position for the black rectangle background
    background_position = (position[0], position[1], position[0] + text_size[0], position[1] + text_size[1])
    # Draw the black rectangle background
    draw.rectangle(background_position, fill="black")
    # Draw the text on the image
    draw.text(position, text, font=font, fill="white")
    return image

def bbox_saliency_intersection(bbox_cxcywh, saliency_map):
    # Assuming saliency_map is a numpy array, convert it to a torch tensor
    saliency_map_tensor = torch.from_numpy(saliency_map)

    # Image dimensions
    height, width = saliency_map_tensor.shape

    # Unbind normalized bbox coordinates
    cx, cy, w, h = bbox_cxcywh.unbind()

    # Scale coordinates to absolute pixel coordinates by creating new tensors
    cx = cx * width
    cy = cy * height
    w = w * width
    h = h * height

    # Convert bbox from cx, cy, w, h to x_min, y_min, x_max, y_max
    x1 = torch.clamp((cx - w / 2).int(), 0, width)
    y1 = torch.clamp((cy - h / 2).int(), 0, height)
    x2 = torch.clamp((cx + w / 2).int(), 0, width)
    y2 = torch.clamp((cy + h / 2).int(), 0, height)

    # Ensure coordinates are sliced correctly if y2 or x2 are equal to y1 or x1 respectively
    if y1 == y2 or x1 == x2:
        return torch.tensor(0.0)  # Return 0.0 to indicate no overlap if dimensions collapse

    # Extract the bounding box area from the saliency map
    bbox_area = saliency_map_tensor[y1:y2, x1:x2]
    salient_pixels = torch.sum(bbox_area == 255)

    total_pixels = bbox_area.numel()
    if total_pixels == 0:
        return torch.tensor(0.0)  # Prevent division by zero

    # Calculate the ratio of salient pixels to total pixels in the bbox area
    saliency_ratio = salient_pixels.float() / total_pixels

    return saliency_ratio