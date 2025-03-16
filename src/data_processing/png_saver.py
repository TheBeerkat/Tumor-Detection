import png
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
from tqdm import tqdm

def save_as_png(
    loader: Generator[Tuple[np.ndarray, str], None, None],
    output_dir: Path,
    quality: int = 9
) -> None:
    """
    Save images as PNG files with configurable quality.
    Works with both DICOM images and ROI masks from the combined generator.
    
    Args:
        dicom_loader: Generator yielding (image_data, filename) tuples
        output_dir: Directory to save PNG files
        quality: PNG compression level (0-9)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress bar
    pbar = tqdm(desc="Saving PNG files", unit="images")
    
    for image, filename in loader:
        try:
            output_path = output_dir / f"{filename}.png"
            
            # Log what type of image we're saving based on filename suffix
            image_type = "ROI mask" if filename.endswith("_roi") else "DICOM image"
            
            with open(output_path, 'wb') as png_file:
                writer = png.Writer(
                    width=image.shape[1],
                    height=image.shape[0],
                    greyscale=True,
                    compression=quality
                )
                writer.write(png_file, image.tolist() if not isinstance(image[0], list) else image)
            
            pbar.update(1)
            print(f"Saved {image_type}: {output_path}")
            
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")
            continue
            
    pbar.close() 