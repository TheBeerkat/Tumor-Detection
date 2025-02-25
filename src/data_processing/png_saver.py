import png
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
from tqdm import tqdm

def save_as_png(
    dicom_loader: Generator[Tuple[np.ndarray, str], None, None],
    output_dir: Path,
    quality: int = 9
) -> None:
    """Save DICOM images as PNG files with configurable quality"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress bar
    pbar = tqdm(desc="Saving PNG files", unit="images")
    
    for image, filename in dicom_loader:
        try:
            output_path = output_dir / f"{filename}.png"
            with open(output_path, 'wb') as png_file:
                writer = png.Writer(
                    width=image.shape[1],
                    height=image.shape[0],
                    greyscale=True,
                    compression=quality
                )
                writer.write(png_file, image)
            pbar.update(1)
            
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")
            continue
            
    pbar.close() 