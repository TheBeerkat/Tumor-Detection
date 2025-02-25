import cv2
import numpy as np
from typing import Callable, List

def create_preprocessing_pipeline(config: dict) -> List[Callable]:
    """Create processing pipeline based on config"""
    pipeline = []
    
    if config.get("contrast_enhance", True):
        pipeline.append(contrast_enhance)
    
    if config.get("noise_reduction", True):
        pipeline.append(noise_reduction)
    
    if config.get("intensity_normalization", True):
        pipeline.append(intensity_normalization)
        
    return pipeline

def contrast_enhance(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE"""
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )
    return clahe.apply(image)

def noise_reduction(image: np.ndarray) -> np.ndarray:
    """Reduce noise using median blur"""
    return cv2.medianBlur(image, 5)

def intensity_normalization(image: np.ndarray) -> np.ndarray:
    """Normalize pixel intensities"""
    image_float = image.astype(float)
    mean = np.mean(image_float)
    std = np.std(image_float)
    normalized = (image_float - mean) / (std + 1e-8)
    return np.uint8(255 * (normalized - normalized.min()) / (normalized.max() - normalized.min())) 