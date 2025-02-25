from pathlib import Path
from typing import Union

def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if necessary"""
    path = Path(path) if isinstance(path, str) else path
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_input_dir(path: Union[str, Path]) -> Path:
    """Validate that input directory exists and contains files"""
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():
        raise FileNotFoundError(f"Input directory {path} does not exist")
    if not any(path.iterdir()):
        raise ValueError(f"Input directory {path} is empty")
    return path 