"""
CNN Model for Camera Settings Classification

This module implements a VGG16-based CNN architecture for predicting camera settings
(aperture, ISO, shutter speed) from images. The model uses a pretrained VGG16 backbone
with frozen weights and custom classification heads.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


class CameraSettingsCNN(nn.Module):
    """
    CNN model for camera settings classification.
    
    Architecture:
    - Pretrained VGG16 backbone (frozen)
    - Shared fully connected layer (256 units)
    - Three separate heads for aperture, ISO, and shutter speed classification
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of classes for each head (default: 3)
            dropout_rate (float): Dropout rate for regularization
        """
        super(CameraSettingsCNN, self).__init__()
        
        # Load pretrained VGG16
        self.backbone = models.vgg16(pretrained=True)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get the number of input features for the classifier
        num_features = self.backbone.classifier[0].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Individual classification heads
        self.aperture_head = nn.Linear(256, num_classes)
        self.iso_head = nn.Linear(256, num_classes)
        self.shutter_head = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized CameraSettingsCNN with {num_classes} classes per head")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _initialize_weights(self):
        """Initialize weights for the custom layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 128, 128)
            
        Returns:
            dict: Dictionary containing predictions for each head
                - 'aperture': aperture class predictions
                - 'iso': ISO class predictions  
                - 'shutter': shutter speed class predictions
        """
        # Extract features using pretrained backbone
        features = self.backbone(x)
        
        # Apply shared fully connected layers
        shared_features = self.shared_fc(features)
        
        # Get predictions from each head
        aperture_pred = self.aperture_head(shared_features)
        iso_pred = self.iso_head(shared_features)
        shutter_pred = self.shutter_head(shared_features)
        
        return {
            'aperture': aperture_pred,
            'iso': iso_pred,
            'shutter': shutter_pred
        }
    
    def get_feature_extractor(self):
        """
        Get the feature extraction part of the model (backbone + shared FC).
        
        Returns:
            nn.Module: Feature extraction module
        """
        return nn.Sequential(
            self.backbone,
            self.shared_fc
        )


class CameraSettingsLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Computes CrossEntropyLoss for each head and returns the sum.
    """
    
    def __init__(self, class_weights=None):
        """
        Initialize the loss function.
        
        Args:
            class_weights (dict): Optional class weights for each head
        """
        super(CameraSettingsLoss, self).__init__()
        
        self.class_weights = class_weights or {}
        
        # Create loss functions for each head
        self.aperture_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.get('aperture', None)
        )
        self.iso_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.get('iso', None)
        )
        self.shutter_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.get('shutter', None)
        )
    
    def forward(self, predictions, targets):
        """
        Compute the combined loss.
        
        Args:
            predictions (dict): Model predictions
            targets (dict): Ground truth labels
            
        Returns:
            dict: Dictionary containing individual and total losses
        """
        # Compute individual losses
        aperture_loss = self.aperture_loss(predictions['aperture'], targets['aperture'])
        iso_loss = self.iso_loss(predictions['iso'], targets['iso'])
        shutter_loss = self.shutter_loss(predictions['shutter'], targets['shutter'])
        
        # Total loss is the sum of all losses
        total_loss = aperture_loss + iso_loss + shutter_loss
        
        return {
            'total': total_loss,
            'aperture': aperture_loss,
            'iso': iso_loss,
            'shutter': shutter_loss
        }


def create_model(num_classes=3, dropout_rate=0.5, device='cpu'):
    """
    Create and initialize the camera settings CNN model.
    
    Args:
        num_classes (int): Number of classes for each head
        dropout_rate (float): Dropout rate for regularization
        device (str): Device to place the model on
        
    Returns:
        tuple: (model, criterion) - the model and loss function
    """
    # Create model
    model = CameraSettingsCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Create loss function
    criterion = CameraSettingsLoss()
    criterion = criterion.to(device)
    
    logger.info(f"Created model on device: {device}")
    
    return model, criterion


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): The model to count parameters for
        
    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def get_class_names():
    """
    Get human-readable class names for each camera setting.
    
    Returns:
        dict: Dictionary mapping setting names to class names
    """
    return {
        'aperture': ['Low (<f/5.6)', 'Medium (f/5.6-f/11)', 'High (>f/11)'],
        'iso': ['Low (≤200)', 'Medium (400-800)', 'High (>800)'],
        'shutter': ['Fast (<1/250s)', 'Medium (1/250-1/30s)', 'Slow (>1/30s)']
    }


if __name__ == "__main__":
    # Test the model
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing CameraSettingsCNN...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, criterion = create_model(device=device)
    
    # Print model info
    param_counts = count_parameters(model)
    print(f"Model parameters: {param_counts}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128).to(device)
    
    with torch.no_grad():
        predictions = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shapes:")
        for head, pred in predictions.items():
            print(f"  {head}: {pred.shape}")
    
    # Test loss computation
    targets = {
        'aperture': torch.randint(0, 3, (batch_size,)).to(device),
        'iso': torch.randint(0, 3, (batch_size,)).to(device),
        'shutter': torch.randint(0, 3, (batch_size,)).to(device)
    }
    
    losses = criterion(predictions, targets)
    print(f"Losses: {losses}")
    
    print("Model test completed!")
