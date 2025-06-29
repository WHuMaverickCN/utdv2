import os
import pandas as pd
from glob import glob
from typing import List, Optional

class DatLoader:
    """
    A class to load data from a CSV file and a directory containing images.
    """
    
    def __init__(self, csv_path: str, images_dir_path: str):
        """
        Initialize the DataLoader with paths to CSV and image directory.
        
        Args:
            csv_path: Path to the CSV file
            images_dir_path: Path to the directory containing images
        """
        self.csv_path = csv_path
        self.images_dir_path = images_dir_path
        
        # Validate paths
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isdir(images_dir_path):
            raise FileNotFoundError(f"Images directory not found: {images_dir_path}")
        
        # Read CSV data
        self.data_df = pd.read_csv(csv_path)
        
        # Get list of image files
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} image files in {images_dir_path}")
    def _get_image_files(self) -> List[str]:
        """Get all image files in the images directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(self.images_dir_path, f"*{ext}")))
            image_files.extend(glob(os.path.join(self.images_dir_path, f"*{ext.upper()}")))
        
        return sorted(image_files)
    
temp = DatLoader("/data/gyx/cqc_p2_cd701_raw/datasets_A4_0326/location/03-26_routex_Location/CD701_000052__2024-03-26_11-49-02.csv",\
                 "/data/gyx/cqc_p2_cd701_raw/datasets_A4_0326/vision/03-26_routex/CD701_000052__2024-03-26_11-49-02")