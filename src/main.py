from pathlib import Path
from data_processing.dicom_loader import load_dicom_batch
from data_processing.png_saver import save_as_png
from image_processing.preprocessing import create_preprocessing_pipeline
from utils.metadata_handler import create_metadata_file
from utils.file_io import validate_input_dir, ensure_dir_exists

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
    
    # Process and save images
    save_as_png(
        dicom_loader=load_dicom_batch(input_dir, processing_steps),
        output_dir=output_dir,
        quality=config["output_quality"]
    )
    
    # Create metadata
    create_metadata_file(output_dir)

if __name__ == "__main__":
    input_path = validate_input_dir(BASE_DIR / "data/INbreast Release 1.0/AllDICOMs")
    output_path = ensure_dir_exists(BASE_DIR / "data/processed/images")
    
    process_pipeline(input_path, output_path) 