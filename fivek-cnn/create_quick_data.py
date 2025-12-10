#!/usr/bin/env python3
"""
Quick synthetic dataset creator for fast training tests.
Creates a small dataset of synthetic images with random labels.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

def create_quick_dataset(num_images=50, output_dir="data"):
    """Create a quick synthetic dataset for training."""
    output_path = Path(output_dir)
    jpg_dir = output_path / "jpg"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} synthetic images...")
    
    data = []
    for i in range(num_images):
        # Create a random colorful image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        img_filename = f"image_{i:04d}.jpg"
        img_path = jpg_dir / img_filename
        img.save(img_path, 'JPEG')
        
        # Random labels (3 classes each: 0, 1, 2)
        ap_class = np.random.randint(0, 3)
        iso_class = np.random.randint(0, 3)
        shut_class = np.random.randint(0, 3)
        
        data.append({
            'image_path': f'jpg/{img_filename}',
            'ap_class': ap_class,
            'iso_class': iso_class,
            'shut_class': shut_class
        })
    
    # Save CSV
    df = pd.DataFrame(data)
    csv_path = output_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Created {num_images} images in {jpg_dir}")
    print(f"Labels saved to {csv_path}")
    print(f"\nDataset summary:")
    print(f"  Aperture classes: {df['ap_class'].value_counts().sort_index().to_dict()}")
    print(f"  ISO classes: {df['iso_class'].value_counts().sort_index().to_dict()}")
    print(f"  Shutter classes: {df['shut_class'].value_counts().sort_index().to_dict()}")
    
    return csv_path, jpg_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=50, help='Number of images to create')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    args = parser.parse_args()
    
    create_quick_dataset(args.num_images, args.output_dir)


