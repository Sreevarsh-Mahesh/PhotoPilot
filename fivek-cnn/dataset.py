"""
PyTorch Dataset for MIT-Adobe FiveK Camera Settings Classification

This module provides a custom PyTorch Dataset class for loading images and their
corresponding camera settings labels from the processed FiveK dataset.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FiveKDataset(Dataset):
    """
    Custom PyTorch Dataset for FiveK camera settings classification.
    
    Loads images and their corresponding camera settings labels (aperture, ISO, shutter speed)
    from the processed FiveK dataset CSV file.
    """
    
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels
            image_dir (str): Directory containing the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Filter out rows with missing labels (-1)
        self.data = self.data[
            (self.data['ap_class'] != -1) & 
            (self.data['iso_class'] != -1) & 
            (self.data['shut_class'] != -1)
        ].reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.data)} samples with valid labels")
        logger.info(f"Aperture class distribution: {self.data['ap_class'].value_counts().sort_index().to_dict()}")
        logger.info(f"ISO class distribution: {self.data['iso_class'].value_counts().sort_index().to_dict()}")
        logger.info(f"Shutter class distribution: {self.data['shut_class'].value_counts().sort_index().to_dict()}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, labels_dict) where labels_dict contains:
                - 'aperture': aperture class (0, 1, or 2)
                - 'iso': ISO class (0, 1, or 2)  
                - 'shutter': shutter class (0, 1, or 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and labels
        row = self.data.iloc[idx]
        image_path = row['image_path']
        aperture_class = int(row['ap_class'])
        iso_class = int(row['iso_class'])
        shutter_class = int(row['shut_class'])
        
        # Load image
        try:
            # Handle both absolute and relative paths
            if os.path.isabs(image_path):
                full_image_path = image_path
            else:
                full_image_path = os.path.join(self.image_dir, os.path.basename(image_path))
            
            image = Image.open(full_image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Create labels dictionary
            labels = {
                'aperture': torch.tensor(aperture_class, dtype=torch.long),
                'iso': torch.tensor(iso_class, dtype=torch.long),
                'shutter': torch.tensor(shutter_class, dtype=torch.long)
            }
            
            return image, labels
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a dummy sample if image loading fails
            dummy_image = Image.new('RGB', (128, 128), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            labels = {
                'aperture': torch.tensor(0, dtype=torch.long),
                'iso': torch.tensor(0, dtype=torch.long),
                'shutter': torch.tensor(0, dtype=torch.long)
            }
            
            return dummy_image, labels


def get_transforms():
    """
    Get the standard transforms for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(csv_file, image_dir, batch_size=32, train_split=0.8, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        csv_file (str): Path to the CSV file
        image_dir (str): Directory containing images
        batch_size (int): Batch size for data loaders
        train_split (float): Fraction of data to use for training
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create full dataset
    full_dataset = FiveKDataset(csv_file, image_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    # Create subset datasets
    train_dataset = FiveKDataset(csv_file, image_dir, transform=train_transform)
    val_dataset = FiveKDataset(csv_file, image_dir, transform=val_transform)
    
    # Create subset samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy data
    print("Testing FiveKDataset...")
    
    # Create a dummy CSV for testing
    import pandas as pd
    dummy_data = {
        'image_path': ['test1.jpg', 'test2.jpg'],
        'ap_class': [0, 1],
        'iso_class': [1, 2],
        'shut_class': [2, 0]
    }
    pd.DataFrame(dummy_data).to_csv('test_labels.csv', index=False)
    
    # Test dataset creation
    dataset = FiveKDataset('test_labels.csv', '.', transform=get_transforms()[0])
    print(f"Dataset length: {len(dataset)}")
    
    # Test data loading
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample image shape: {sample[0].shape}")
        print(f"Sample labels: {sample[1]}")
    
    # Clean up
    os.remove('test_labels.csv')
    print("Dataset test completed!")
