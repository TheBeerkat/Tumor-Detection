import numpy as np
import cv2
from tqdm import tqdm
from roi_loader import load_inbreast_mask
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def crop_tumor_regions(image, mask, min_size=200, padding_factor=0.2):
    """
    Identifies and crops all isolated tumor regions from the original image based on the binary mask.
    The crop size depends on the tumor size: small tumors get a tight crop, while large tumors include padding.
    
    @image : Original image as a numpy array
    @mask : Binary mask with tumor regions as a numpy array (same shape as image)
    @min_size : Minimum size (in pixels) for a tumor to be considered significant
    @padding_factor : Fraction of tumor size added as padding for larger tumors
    return: List of cropped images containing separate significant tumor regions and their corresponding masks
    """
    cropped_images = []
    cropped_masks = []
    image_height, image_width = image.shape[:2]
    print(image_height, image_width)
    
    # Label connected components in the mask
    labeled_mask = label(mask)
    
    # Process each connected component (tumor region)
    for region in regionprops(labeled_mask):
        if region.area >= min_size:  # Consider only significant tumors
            # Get bounding box coordinates
            y_min, x_min, y_max, x_max = region.bbox
            
            # Calculate padding based on tumor size
            padding = int(padding_factor * max(y_max - y_min, x_max - x_min))
            
            # Apply padding, ensuring we stay within image bounds
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(image_height, y_max + padding)
            x_max = min(image_width, x_max + padding)
            
            # Crop the image and mask
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_mask = (labeled_mask[y_min:y_max, x_min:x_max] == region.label).astype(np.uint8)
            
            cropped_images.append(cropped_image)
            cropped_masks.append(cropped_mask)
    
    return cropped_images, cropped_masks

