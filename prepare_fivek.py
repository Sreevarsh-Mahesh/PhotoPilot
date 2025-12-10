#!/usr/bin/env python3
"""
MIT-Adobe FiveK Dataset Preparation Script

This script downloads the MIT-Adobe FiveK dataset from Kaggle, extracts EXIF data
from RAW files, converts them to RGB JPEGs, and creates a CSV with binned camera settings.

Usage: python prepare_fivek.py
"""

import os
import sys
import csv
import json
import shutil
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from fractions import Fraction

import kaggle
import rawpy
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FiveKProcessor:
    """Main class for processing the MIT-Adobe FiveK dataset."""
    
    def __init__(self, output_dir: str = "./fivek_data", max_images: int = 1000):
        """
        Initialize the FiveK processor.
        
        Args:
            output_dir: Directory to store processed data
            max_images: Maximum number of images to process
        """
        self.output_dir = Path(output_dir)
        self.max_images = max_images
        self.jpg_dir = self.output_dir / "jpg"
        self.raw_dir = self.output_dir / "raw"
        self.csv_path = self.output_dir / "camera_settings.csv"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.jpg_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file
        self._init_csv()
    
    def _init_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'ap_class', 'iso_class', 'shut_class'])
    
    def download_dataset(self) -> bool:
        """
        Download the MIT-Adobe FiveK dataset from Kaggle.
        Note: This downloads the full dataset (~28GB), but we'll only process max_images.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Downloading MIT-Adobe FiveK dataset from Kaggle...")
            logger.warning(f"Note: Full dataset is ~28GB, but only {self.max_images} images will be processed")
            
            # Check for kaggle.json in local kaggle folder first
            local_kaggle_json = Path("kaggle/kaggle.json")
            standard_kaggle_json = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
            
            if local_kaggle_json.exists():
                logger.info("Found kaggle.json in local kaggle folder, copying to standard location...")
                # Create ~/.kaggle directory if it doesn't exist
                os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
                # Copy the file
                shutil.copy2(local_kaggle_json, standard_kaggle_json)
                # Set proper permissions
                os.chmod(standard_kaggle_json, 0o600)
                logger.info("kaggle.json copied to ~/.kaggle/")
            elif not standard_kaggle_json.exists():
                logger.error("Kaggle API not configured. Please set up kaggle.json in ~/.kaggle/ or in local kaggle/ folder")
                return False
            
            # Check if dataset already exists
            raw_photos_path = self.output_dir / "MIT-Adobe FiveK" / "raw_photos"
            if raw_photos_path.exists() and any(raw_photos_path.glob("*.CR2")):
                logger.info(f"Found existing dataset at {raw_photos_path}")
                logger.info("Skipping download. Use --skip_download flag next time to skip this check.")
                return True
            
            # Download the dataset
            logger.info("Starting download (this may take a while for the full dataset)...")
            kaggle.api.dataset_download_files(
                'mit-adobe-fivek',
                path=str(self.output_dir),
                unzip=True
            )
            
            logger.info("Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.error("\nTroubleshooting steps:")
            logger.error("1. Make sure you have accepted the dataset terms on Kaggle:")
            logger.error("   https://www.kaggle.com/datasets/mit-adobe-fivek")
            logger.error("2. Verify your Kaggle API credentials are set up correctly")
            logger.error("3. Check if the dataset identifier is correct")
            logger.error("\nAlternative: You can manually download the dataset and place RAW files in:")
            logger.error(f"   {self.output_dir / 'MIT-Adobe FiveK' / 'raw_photos'}")
            logger.error("   Then run with --skip_download flag")
            return False
    
    def find_raw_files(self) -> List[Path]:
        """
        Find all RAW (.CR2) files in the dataset.
        
        Returns:
            List of Path objects for RAW files
        """
        raw_files = []
        
        # Look for RAW files in common locations
        search_paths = [
            self.output_dir / "MIT-Adobe FiveK" / "raw_photos",
            self.output_dir / "raw_photos",
            self.output_dir
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                raw_files.extend(search_path.glob("*.CR2"))
                raw_files.extend(search_path.glob("*.cr2"))
                raw_files.extend(search_path.glob("*.NEF"))
                raw_files.extend(search_path.glob("*.nef"))
                raw_files.extend(search_path.glob("*.ARW"))
                raw_files.extend(search_path.glob("*.arw"))
        
        # Limit to max_images
        raw_files = raw_files[:self.max_images]
        logger.info(f"Found {len(raw_files)} RAW files to process")
        
        return raw_files
    
    def extract_exif_data(self, raw_file: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        """
        Extract EXIF data from RAW file.
        
        Args:
            raw_file: Path to RAW file
            
        Returns:
            Tuple of (aperture, iso, shutter_speed) or (None, None, None) if failed
        """
        try:
            with rawpy.imread(str(raw_file)) as raw:
                # Extract EXIF data
                exif = raw.extract_exif()
                
                aperture = None
                iso = None
                shutter_speed = None
                
                # Extract aperture (FNumber)
                if 'FNumber' in exif:
                    aperture = float(exif['FNumber'])
                
                # Extract ISO
                if 'ISOSpeedRatings' in exif:
                    iso = int(exif['ISOSpeedRatings'])
                
                # Extract shutter speed (ExposureTime)
                if 'ExposureTime' in exif:
                    exposure_time = exif['ExposureTime']
                    if isinstance(exposure_time, (int, float)):
                        shutter_speed = float(exposure_time)
                    elif isinstance(exposure_time, Fraction):
                        shutter_speed = float(exposure_time)
                
                return aperture, iso, shutter_speed
                
        except Exception as e:
            logger.warning(f"Error extracting EXIF from {raw_file.name}: {e}")
            return None, None, None
    
    def bin_aperture(self, aperture: float) -> int:
        """
        Bin aperture value into 3 classes.
        
        Args:
            aperture: Aperture value (F-number)
            
        Returns:
            Class: 0=<5.6, 1=5.6-11, 2=>11
        """
        if aperture is None:
            return -1
        
        if aperture < 5.6:
            return 0
        elif aperture <= 11:
            return 1
        else:
            return 2
    
    def bin_iso(self, iso: int) -> int:
        """
        Bin ISO value into 3 classes.
        
        Args:
            iso: ISO value
            
        Returns:
            Class: 0<=200, 1=400-800, 2=>800
        """
        if iso is None:
            return -1
        
        if iso <= 200:
            return 0
        elif iso <= 800:
            return 1
        else:
            return 2
    
    def bin_shutter_speed(self, shutter_speed: float) -> int:
        """
        Bin shutter speed value into 3 classes.
        
        Args:
            shutter_speed: Shutter speed in seconds
            
        Returns:
            Class: 0<1/250, 1=1/250-1/30, 2=>1/30
        """
        if shutter_speed is None:
            return -1
        
        # Convert to 1/x format for easier comparison
        if shutter_speed < 1/250:
            return 0
        elif shutter_speed <= 1/30:
            return 1
        else:
            return 2
    
    def convert_raw_to_jpeg(self, raw_file: Path, output_path: Path) -> bool:
        """
        Convert RAW file to RGB JPEG.
        
        Args:
            raw_file: Path to input RAW file
            output_path: Path to output JPEG file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with rawpy.imread(str(raw_file)) as raw:
                # Process RAW image
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=8
                )
                
                # Convert to PIL Image
                image = Image.fromarray(rgb)
                
                # Save as JPEG
                image.save(output_path, 'JPEG', quality=95)
                
                return True
                
        except Exception as e:
            logger.warning(f"Error converting {raw_file.name} to JPEG: {e}")
            return False
    
    def process_images(self):
        """Main processing function."""
        # Find RAW files
        raw_files = self.find_raw_files()
        
        if not raw_files:
            logger.error("No RAW files found in the dataset!")
            return
        
        # Process each image
        successful_conversions = 0
        csv_data = []
        
        for raw_file in tqdm(raw_files, desc="Processing images"):
            try:
                # Extract EXIF data
                aperture, iso, shutter_speed = self.extract_exif_data(raw_file)
                
                # Bin the settings
                ap_class = self.bin_aperture(aperture)
                iso_class = self.bin_iso(iso)
                shut_class = self.bin_shutter_speed(shutter_speed)
                
                # Convert to JPEG
                jpeg_filename = raw_file.stem + '.jpg'
                jpeg_path = self.jpg_dir / jpeg_filename
                
                if self.convert_raw_to_jpeg(raw_file, jpeg_path):
                    successful_conversions += 1
                    
                    # Add to CSV data
                    csv_data.append([
                        str(jpeg_path),
                        ap_class,
                        iso_class,
                        shut_class
                    ])
                    
                    # Log progress
                    if successful_conversions % 100 == 0:
                        logger.info(f"Processed {successful_conversions} images...")
                
            except Exception as e:
                logger.warning(f"Error processing {raw_file.name}: {e}")
                continue
        
        # Save CSV data
        self._save_csv_data(csv_data)
        
        logger.info(f"Processing complete! Successfully converted {successful_conversions} images")
        logger.info(f"CSV saved to: {self.csv_path}")
        logger.info(f"JPEGs saved to: {self.jpg_dir}")
    
    def _save_csv_data(self, csv_data: List[List]):
        """Save CSV data to file."""
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'ap_class', 'iso_class', 'shut_class'])
            writer.writerows(csv_data)
    
    def copy_to_training_dir(self, training_dir: str = "./fivek-cnn/data"):
        """
        Copy processed data to training directory.
        
        Args:
            training_dir: Directory where training expects data
        """
        training_path = Path(training_dir)
        training_path.mkdir(parents=True, exist_ok=True)
        training_jpg_dir = training_path / "jpg"
        training_jpg_dir.mkdir(exist_ok=True)
        
        # Copy CSV file (update paths to be relative)
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            # Update paths to be relative to training directory
            df['image_path'] = df['image_path'].apply(
                lambda x: os.path.join('jpg', os.path.basename(x))
            )
            labels_csv = training_path / "labels.csv"
            df.to_csv(labels_csv, index=False)
            logger.info(f"Copied labels CSV to {labels_csv}")
        
        # Copy JPEG images
        jpg_files = list(self.jpg_dir.glob("*.jpg"))
        for jpg_file in tqdm(jpg_files, desc="Copying images"):
            dest_file = training_jpg_dir / jpg_file.name
            if not dest_file.exists():
                shutil.copy2(jpg_file, dest_file)
        
        logger.info(f"Copied {len(jpg_files)} images to {training_jpg_dir}")
        logger.info(f"Training data ready at {training_path}")
    
    def print_summary(self):
        """Print processing summary."""
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            logger.info("\n=== Processing Summary ===")
            logger.info(f"Total images processed: {len(df)}")
            logger.info(f"Aperture class distribution:")
            logger.info(df['ap_class'].value_counts().sort_index())
            logger.info(f"ISO class distribution:")
            logger.info(df['iso_class'].value_counts().sort_index())
            logger.info(f"Shutter speed class distribution:")
            logger.info(df['shut_class'].value_counts().sort_index())


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare MIT-Adobe FiveK dataset')
    parser.add_argument('--max_images', type=int, default=200,
                       help='Maximum number of images to process (default: 200)')
    parser.add_argument('--output_dir', type=str, default='./fivek_data',
                       help='Output directory for processed data (default: ./fivek_data)')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip dataset download (use if already downloaded)')
    parser.add_argument('--copy_to_training', action='store_true',
                       help='Copy processed data to training directory (fivek-cnn/data)')
    parser.add_argument('--test_mode', action='store_true',
                       help='Create a small test dataset (10 synthetic images) for quick testing')
    
    args = parser.parse_args()
    
    logger.info("Starting MIT-Adobe FiveK dataset preparation...")
    logger.info(f"Will process up to {args.max_images} images")
    logger.info(f"Note: Full dataset download is ~28GB, but only {args.max_images} images will be processed")
    
    # Initialize processor
    processor = FiveKProcessor(max_images=args.max_images, output_dir=args.output_dir)
    
    # Download dataset (unless skipped)
    if not args.skip_download:
        if not processor.download_dataset():
            logger.error("Failed to download dataset. Exiting.")
            sys.exit(1)
    else:
        logger.info("Skipping download (using existing data)")
    
    # Process images
    processor.process_images()
    
    # Print summary
    processor.print_summary()
    
    # Copy to training directory if requested
    if args.copy_to_training:
        logger.info("\nCopying data to training directory...")
        processor.copy_to_training_dir()
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
