"""
Inference script for Camera Settings CNN

This script loads a trained model and predicts camera settings for new images.
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from model import create_model, get_class_names

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraSettingsPredictor:
    """Camera settings predictor using trained CNN model."""
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on
        """
        self.device = torch.device(device)
        self.model = None
        self.class_names = get_class_names()
        self.transform = self._get_transform()
        
        # Load model
        self._load_model(model_path)
    
    def _get_transform(self):
        """Get the image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load the trained model."""
        try:
            # Create model
            self.model, _ = create_model(device=self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model state dict")
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict(self, image_path, return_probabilities=False):
        """
        Predict camera settings for an image.
        
        Args:
            image_path (str): Path to the image file
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Predicted camera settings and optionally probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Get predicted classes
        predicted_classes = {}
        probabilities = {}
        
        for head in ['aperture', 'iso', 'shutter']:
            # Get class probabilities
            probs = torch.softmax(predictions[head], dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            
            predicted_classes[head] = predicted_class
            probabilities[head] = probs.cpu().numpy()[0]
        
        result = {
            'predictions': predicted_classes,
            'class_names': {
                head: self.class_names[head][predicted_classes[head]]
                for head in predicted_classes
            }
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict camera settings for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probabilities)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def format_predictions(predictions, show_probabilities=False):
    """
    Format predictions for display.
    
    Args:
        predictions (dict): Prediction results
        show_probabilities (bool): Whether to show class probabilities
        
    Returns:
        str: Formatted prediction string
    """
    lines = []
    lines.append("Predicted Optimal Settings:")
    lines.append("=" * 40)
    
    for head in ['aperture', 'iso', 'shutter']:
        class_name = predictions['class_names'][head]
        lines.append(f"  {head.capitalize()}: {class_name}")
        
        if show_probabilities and 'probabilities' in predictions:
            probs = predictions['probabilities'][head]
            lines.append(f"    Probabilities: {[f'{p:.3f}' for p in probs]}")
    
    return "\n".join(lines)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Predict Camera Settings from Images')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run inference on (cpu, cuda, or auto)')
    parser.add_argument('--probabilities', action='store_true',
                       help='Show class probabilities')
    parser.add_argument('--batch', nargs='+', type=str,
                       help='Process multiple images')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.error("Please train the model first using train.py")
        return
    
    # Create predictor
    try:
        predictor = CameraSettingsPredictor(args.model, device=device)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return
    
    # Process images
    if args.batch:
        # Batch processing
        logger.info(f"Processing {len(args.batch)} images...")
        results = predictor.predict_batch(args.batch, return_probabilities=args.probabilities)
        
        for i, result in enumerate(results):
            if 'error' in result:
                logger.error(f"Image {i+1}: {result['error']}")
            else:
                print(f"\nImage {i+1}: {result['image_path']}")
                print(format_predictions(result, show_probabilities=args.probabilities))
    
    else:
        # Single image processing
        if not os.path.exists(args.image):
            logger.error(f"Image file not found: {args.image}")
            return
        
        logger.info(f"Processing image: {args.image}")
        
        try:
            predictions = predictor.predict(args.image, return_probabilities=args.probabilities)
            print(format_predictions(predictions, show_probabilities=args.probabilities))
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")


def demo():
    """Demo function with example usage."""
    print("Camera Settings CNN - Demo")
    print("=" * 40)
    
    # Example usage
    example_commands = [
        "python infer.py --image data/jpg/0001.jpg",
        "python infer.py --image your_photo.jpg --probabilities",
        "python infer.py --batch data/jpg/0001.jpg data/jpg/0002.jpg data/jpg/0003.jpg"
    ]
    
    print("Example usage:")
    for cmd in example_commands:
        print(f"  {cmd}")
    
    print("\nClass meanings:")
    class_names = get_class_names()
    for setting, classes in class_names.items():
        print(f"  {setting.capitalize()}:")
        for i, class_name in enumerate(classes):
            print(f"    {i}: {class_name}")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        demo()
    else:
        main()
