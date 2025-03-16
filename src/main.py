from pathlib import Path
from data_processing.dicom_loader import load_dicom_batch
from data_processing.roi_loader import load_inbreast_mask
from data_processing.png_saver import save_as_png
from image_processing.preprocessing import create_preprocessing_pipeline
from utils.metadata_handler import create_metadata_file
from utils.file_io import validate_input_dir, ensure_dir_exists

import numpy as np

DEFAULT_CONFIG = {
    "contrast_enhance": True,
    "noise_reduction": True,
    "intensity_normalization": True,
    "output_quality": 9  # PNG compression level
}

BASE_DIR = Path(__file__).parent.parent

def process_pipeline(input_dir: Path, output_dir: Path, config: dict = None) -> None:
    """Main processing pipeline"""
    config = config or DEFAULT_CONFIG
    
    # Create processing pipeline
    processing_steps = create_preprocessing_pipeline(config)
    
    # Get DICOM files generator
    dicom_generator = load_dicom_batch(input_dir / "AllDICOMs", processing_steps)
    
    # Create a generator that yields both DICOM and ROI images
    def combined_generator():
        for image, filename in dicom_generator:
            filename = filename.split("_")[0]
            # Yield the DICOM image first
            yield image, f"{filename}_image"
            
            # Try to find matching ROI file (.xml file with same base name)
            roi_path = input_dir / "AllXML" / f"{filename}.xml"
            if roi_path.exists():
                try:
                    # Load the ROI mask with the same dimensions as the image
                    roi_mask = load_inbreast_mask(roi_path, imshape=image.shape)
                    # Convert to uint8 format compatible with PNG saver
                    roi_mask = (roi_mask * 255).astype('uint8')
                    # Yield the ROI mask
                    yield roi_mask, f"{filename}_roi"
                except Exception as e:
                    print(f"Error processing ROI file {roi_path}: {str(e)}")
            else:
                # If no ROI file exists, create an empty mask of zeros
                empty_mask = np.zeros(image.shape, dtype='uint8')
                yield empty_mask, f"{filename}_roi"
    
    # Save all images (both DICOM and ROI) as PNG
    save_as_png(
        loader=combined_generator(),
        output_dir=output_dir,
        quality=config["output_quality"]
    )
    
    # Create metadata
    # create_metadata_file(output_dir)

if __name__ == "__main__":
    input_path = validate_input_dir(BASE_DIR / "data/INbreast_2012/INbreast Release 1.0")
    output_path = ensure_dir_exists(BASE_DIR / "data/processed/")
    
    process_pipeline(input_path, output_path)