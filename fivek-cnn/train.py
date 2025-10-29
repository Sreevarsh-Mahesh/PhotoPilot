"""
Training script for Camera Settings CNN

This script trains a CNN model to predict camera settings (aperture, ISO, shutter speed)
from images using the processed MIT-Adobe FiveK dataset.
"""

import os
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset import create_data_loaders
from model import create_model, count_parameters, get_class_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def calculate_accuracy(predictions, targets):
    """
    Calculate accuracy for each head.
    
    Args:
        predictions (dict): Model predictions
        targets (dict): Ground truth labels
        
    Returns:
        dict: Accuracy for each head
    """
    accuracies = {}
    
    for head in ['aperture', 'iso', 'shutter']:
        pred_classes = torch.argmax(predictions[head], dim=1)
        correct = (pred_classes == targets[head]).float()
        accuracies[head] = correct.mean().item()
    
    return accuracies


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        dict: Training metrics
    """
    model.train()
    total_loss = 0.0
    total_accuracies = {'aperture': 0.0, 'iso': 0.0, 'shutter': 0.0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        losses = criterion(predictions, targets)
        total_loss += losses['total'].item()
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Calculate accuracy
        accuracies = calculate_accuracy(predictions, targets)
        for head in total_accuracies:
            total_accuracies[head] += accuracies[head]
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{losses['total'].item():.4f}",
            'Aperture': f"{accuracies['aperture']:.3f}",
            'ISO': f"{accuracies['iso']:.3f}",
            'Shutter': f"{accuracies['shutter']:.3f}"
        })
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_accuracies = {k: v / num_batches for k, v in total_accuracies.items()}
    
    return {
        'loss': avg_loss,
        'accuracies': avg_accuracies
    }


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_accuracies = {'aperture': 0.0, 'iso': 0.0, 'shutter': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        for images, targets in pbar:
            # Move to device
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            losses = criterion(predictions, targets)
            total_loss += losses['total'].item()
            
            # Calculate accuracy
            accuracies = calculate_accuracy(predictions, targets)
            for head in total_accuracies:
                total_accuracies[head] += accuracies[head]
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Aperture': f"{accuracies['aperture']:.3f}",
                'ISO': f"{accuracies['iso']:.3f}",
                'Shutter': f"{accuracies['shutter']:.3f}"
            })
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_accuracies = {k: v / num_batches for k, v in total_accuracies.items()}
    
    return {
        'loss': avg_loss,
        'accuracies': avg_accuracies
    }


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['val_loss']


def train_model(csv_file, image_dir, epochs=5, batch_size=32, learning_rate=0.001, 
                train_split=0.8, save_dir='./checkpoints'):
    """
    Train the camera settings CNN model.
    
    Args:
        csv_file (str): Path to CSV file with labels
        image_dir (str): Directory containing images
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        train_split (float): Fraction of data for training
        save_dir (str): Directory to save checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        csv_file=csv_file,
        image_dir=image_dir,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=4
    )
    
    # Create model and criterion
    logger.info("Creating model...")
    model, criterion = create_model(device=device)
    
    # Print model info
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = []
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Acc - Aperture: {train_metrics['accuracies']['aperture']:.3f}, "
                   f"ISO: {train_metrics['accuracies']['iso']:.3f}, "
                   f"Shutter: {train_metrics['accuracies']['shutter']:.3f}")
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Acc - Aperture: {val_metrics['accuracies']['aperture']:.3f}, "
                   f"ISO: {val_metrics['accuracies']['iso']:.3f}, "
                   f"Shutter: {val_metrics['accuracies']['shutter']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, val_metrics['loss'], checkpoint_path)
        
        # Record history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_acc': train_metrics['accuracies'],
            'val_acc': val_metrics['accuracies']
        })
        
        # Check for early stopping
        if early_stopping(val_metrics['loss']):
            logger.info("Early stopping triggered!")
            break
    
    # Save training history
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {os.path.join(save_dir, 'best_model.pth')}")
    
    return model, train_history


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Camera Settings CNN')
    parser.add_argument('--csv', type=str, default='data/labels.csv',
                       help='Path to CSV file with labels')
    parser.add_argument('--image_dir', type=str, default='data/jpg',
                       help='Directory containing images')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    if not os.path.exists(args.image_dir):
        logger.error(f"Image directory not found: {args.image_dir}")
        return
    
    # Train model
    model, history = train_model(
        csv_file=args.csv,
        image_dir=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_split=args.train_split,
        save_dir=args.save_dir
    )
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    if history:
        final_epoch = history[-1]
        logger.info(f"Final Training Loss: {final_epoch['train_loss']:.4f}")
        logger.info(f"Final Validation Loss: {final_epoch['val_loss']:.4f}")
        logger.info(f"Final Training Accuracy:")
        for head, acc in final_epoch['train_acc'].items():
            logger.info(f"  {head.capitalize()}: {acc:.3f}")
        logger.info(f"Final Validation Accuracy:")
        for head, acc in final_epoch['val_acc'].items():
            logger.info(f"  {head.capitalize()}: {acc:.3f}")


if __name__ == "__main__":
    main()
