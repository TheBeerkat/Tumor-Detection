import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def create_metadata_file(output_dir: Path) -> Path:
    """Create metadata pickle file from processed images"""
    files = [f.name for f in output_dir.glob("*.png")]
    
    df = pd.DataFrame([parse_filename(f) for f in files])
    df.columns = ['file_no', 'patient_id', 'mg', 'side', 'view', 'end']
    
    metadata = process_patient_groups(df)
    output_path = output_dir / "all_png_metadata.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return output_path

def parse_filename(filename: str) -> list:
    """Parse PNG filename into components"""
    return filename.split("_")[:5] + [filename.split("_")[-1].split(".")[0]]

def process_patient_groups(df: pd.DataFrame) -> list:
    """Process patient groups to create metadata entries"""
    metadata = []
    for patient_id, group in tqdm(df.groupby('patient_id'), desc="Processing patient groups"):
        if len(group) % 4 != 0:
            continue
            
        element = {}
        for idx, row in group.iterrows():
            if idx % 4 == 0:
                element = {'horizontal_flip': 'NO'}
                
            view_key = f"{row['side']}-{row['view'] if row['view'] == 'CC' else 'MLO'}"
            element[view_key] = [f"{row['file_no']}_{patient_id}_MG_{row['side']}_{row['view']}_ANON"]
            
            if idx % 4 == 3 and len(element) == 5:
                metadata.append(element)
                
    return metadata 