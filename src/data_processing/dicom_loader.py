import pydicom as pyd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Generator, Tuple, List, Callable

def load_dicom_batch(
    input_dir: Path,
    processing_pipeline: List[Callable] = None
) -> Generator[Tuple[np.ndarray, str], None, None]:
    """Load DICOM files with optional processing pipeline."""
    dicom_files = [f for f in input_dir.glob("*.dcm") if f.is_file()]
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    for dcm_path in tqdm(dicom_files, desc="Processing DICOM files"):
        try:
            dcm = pyd.dcmread(dcm_path)
            image_2d = dcm.pixel_array.astype(float)
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 256
            image_2d_scaled = np.uint8(image_2d_scaled)

            if processing_pipeline:
                for process in processing_pipeline:
                    image_2d_scaled = process(image_2d_scaled)
                    
            yield image_2d_scaled, dcm_path.stem
            
        except Exception as e:
            print(f"Error processing {dcm_path.name}: {str(e)}")
            continue 