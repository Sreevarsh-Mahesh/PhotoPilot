# PhotoPilot: Comprehensive Methodology Document

## Table of Contents
1. [Dataset Procurement](#1-dataset-procurement)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Training-Validation Split](#3-training-validation-split)
4. [Model Architecture Selection](#4-model-architecture-selection)
5. [Activation Functions](#5-activation-functions)
6. [Weights and Bias](#6-weights-and-bias)
7. [Weight and Bias Updates](#7-weight-and-bias-updates)
8. [Training Process](#8-training-process)
9. [Error Handling and Corrections](#9-error-handling-and-corrections)
10. [Results and Evaluation](#10-results-and-evaluation)

---

## 1. Dataset Procurement

### 1.1 Source Selection
**Dataset**: MIT-Adobe FiveK Dataset
- **Size**: ~5,000 high-resolution RAW images (~28GB uncompressed)
- **Source**: Kaggle (https://www.kaggle.com/datasets/mit-adobe-fivek)
- **Rationale**: 
  - Contains professional photography with EXIF metadata
  - Includes camera settings (aperture, ISO, shutter speed) embedded in RAW files
  - Diverse range of photography styles and lighting conditions
  - Well-established dataset in computational photography research

### 1.2 Download Process
```python
# Implementation in prepare_fivek.py
kaggle.api.dataset_download_files(
    'mit-adobe-fivek',
    path=str(self.output_dir),
    unzip=True
)
```

**Challenges Encountered**:
- **Issue**: Full dataset is 28GB, impractical for quick training
- **Solution**: Implemented `max_images` parameter (default: 200) to process only a subset
- **Trade-off**: Smaller dataset for faster iteration, can scale up later

### 1.3 EXIF Data Extraction
**Process**:
1. Read RAW files (`.CR2`, `.NEF`, `.ARW` formats) using `rawpy` library
2. Extract EXIF metadata:
   - **Aperture (FNumber)**: F-stop value (e.g., f/2.8, f/5.6, f/11)
   - **ISO (ISOSpeedRatings)**: ISO sensitivity (e.g., 100, 400, 800, 1600)
   - **Shutter Speed (ExposureTime)**: Exposure time in seconds (e.g., 1/250, 1/30, 1s)

**Code Implementation**:
```python
def extract_exif_data(self, raw_file: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    with rawpy.imread(str(raw_file)) as raw:
        exif = raw.extract_exif()
        aperture = float(exif['FNumber']) if 'FNumber' in exif else None
        iso = int(exif['ISOSpeedRatings']) if 'ISOSpeedRatings' in exif else None
        shutter_speed = float(exif['ExposureTime']) if 'ExposureTime' in exif else None
        return aperture, iso, shutter_speed
```

---

## 2. Data Preprocessing

### 2.1 Class Binning Strategy
Camera settings are continuous values, but we need discrete classes for classification.

#### Aperture Binning (3 classes):
- **Class 0**: Low aperture (< f/5.6) - Wide open, shallow depth of field
- **Class 1**: Medium aperture (f/5.6 - f/11) - Balanced depth of field
- **Class 2**: High aperture (> f/11) - Narrow aperture, deep depth of field

**Rationale**: These ranges represent distinct photographic scenarios:
- Low: Portraits, low-light, subject isolation
- Medium: General photography, balanced settings
- High: Landscapes, maximum sharpness

#### ISO Binning (3 classes):
- **Class 0**: Low ISO (≤200) - Best quality, requires good light
- **Class 1**: Medium ISO (400-800) - Moderate noise, versatile
- **Class 2**: High ISO (>800) - Higher noise, low-light situations

**Rationale**: ISO directly correlates with image quality vs. light sensitivity trade-off.

#### Shutter Speed Binning (3 classes):
- **Class 0**: Fast (< 1/250s) - Freeze motion, sports/action
- **Class 1**: Medium (1/250s - 1/30s) - General purpose
- **Class 2**: Slow (> 1/30s) - Motion blur, low-light, tripod needed

**Rationale**: Shutter speed determines motion capture vs. light gathering.

### 2.2 Image Preprocessing Pipeline

#### RAW to JPEG Conversion:
```python
def convert_raw_to_jpeg(self, raw_file: Path, output_path: Path) -> bool:
    with rawpy.imread(str(raw_file)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,      # Use camera white balance
            half_size=False,          # Full resolution
            no_auto_bright=True,      # Preserve original exposure
            output_bps=8              # 8-bit output
        )
        image = Image.fromarray(rgb)
        image.save(output_path, 'JPEG', quality=95)
```

**Key Decisions**:
- **Camera white balance**: Preserves original color temperature
- **No auto-brightness**: Maintains original exposure characteristics
- **95% JPEG quality**: Balances file size and quality

#### Training Transforms (Data Augmentation):
```python
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),                    # Standardize size
    transforms.RandomHorizontalFlip(p=0.5),          # Horizontal flip
    transforms.RandomRotation(degrees=10),            # Slight rotation
    transforms.ColorJitter(                           # Color augmentation
        brightness=0.2, contrast=0.2, 
        saturation=0.2, hue=0.1
    ),
    transforms.ToTensor(),                            # Convert to tensor
    transforms.Normalize(                             # ImageNet normalization
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

**Why These Augmentations?**:
- **Resize to 128x128**: Reduces computational cost, VGG16 can handle this size
- **RandomHorizontalFlip**: Doubles dataset, doesn't affect camera settings
- **RandomRotation (±10°)**: Handles slight camera tilt, realistic variation
- **ColorJitter**: Simulates different lighting conditions, improves generalization
- **ImageNet Normalization**: Standard for pretrained models, ensures compatibility

#### Validation Transforms (No Augmentation):
```python
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Rationale**: Validation should use original images to measure true model performance.

### 2.3 Data Cleaning
- **Filtering**: Removed samples with missing EXIF data (marked as -1)
- **Error Handling**: Graceful handling of corrupted RAW files
- **Path Resolution**: Handles both absolute and relative image paths

---

## 3. Training-Validation Split

### 3.1 Split Strategy
**Default Split**: 80% training, 20% validation

```python
train_split = 0.8
total_size = len(full_dataset)
train_size = int(train_split * total_size)
val_size = total_size - train_size
```

**Rationale**:
- **80/20 split**: Standard practice for medium-sized datasets
- **Sequential split**: First 80% for training, last 20% for validation
- **No shuffling before split**: Ensures consistent splits across runs

### 3.2 Data Loaders
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=SubsetRandomSampler(train_indices),
    num_workers=4,
    pin_memory=True
)
```

**Parameters**:
- **Batch size**: 32 - Balances memory usage and gradient stability
- **SubsetRandomSampler**: Random sampling within training subset
- **num_workers=4**: Parallel data loading for efficiency
- **pin_memory=True**: Faster GPU transfer (if available)

---

## 4. Model Architecture Selection

### 4.1 Why Convolutional Neural Network (CNN)?

**Rationale for CNN Selection**:

1. **Spatial Feature Learning**:
   - Images have spatial structure (edges, textures, patterns)
   - CNNs excel at learning hierarchical features (low-level → high-level)
   - Camera settings correlate with visual features:
     - **Aperture**: Affects depth of field (blur patterns)
     - **ISO**: Affects noise patterns
     - **Shutter**: Affects motion blur

2. **Translation Invariance**:
   - CNNs are translation-invariant (object position doesn't matter)
   - Important for camera settings prediction (subject location irrelevant)

3. **Parameter Efficiency**:
   - Shared weights across spatial locations
   - Much fewer parameters than fully connected networks

4. **Proven Track Record**:
   - CNNs dominate computer vision tasks
   - Transfer learning from ImageNet works well

### 4.2 Architecture Choice: VGG16 with Transfer Learning

**Selected Architecture**: VGG16 (pretrained on ImageNet)

**Why VGG16?**:
1. **Proven Performance**: Excellent feature extractor
2. **Simplicity**: Straightforward architecture, easy to modify
3. **Transfer Learning**: Pretrained on ImageNet (1.2M images, 1000 classes)
4. **Feature Richness**: 13 convolutional layers capture rich visual features

**Architecture Overview**:
```
Input Image (3, 128, 128)
    ↓
VGG16 Backbone (Frozen)
    ↓
Feature Vector (25088 dimensions)
    ↓
Shared FC Layer 1: 25088 → 512 (ReLU + Dropout)
    ↓
Shared FC Layer 2: 512 → 256 (ReLU + Dropout)
    ↓
    ├─→ Aperture Head: 256 → 3 (Linear)
    ├─→ ISO Head: 256 → 3 (Linear)
    └─→ Shutter Head: 256 → 3 (Linear)
```

**Why Freeze Backbone?**:
- **Preserve Features**: ImageNet features are general and useful
- **Faster Training**: Only train custom layers (~13M trainable vs. 27M total)
- **Less Data Needed**: Transfer learning works with smaller datasets
- **Prevent Overfitting**: Frozen backbone acts as regularization

**Multi-Head Architecture**:
- **Shared Feature Extraction**: Single backbone + shared FC layers
- **Task-Specific Heads**: Separate linear layers for each camera setting
- **Rationale**: Camera settings are related but distinct tasks
  - Shared features capture general image properties
  - Separate heads learn setting-specific patterns

---

## 5. Activation Functions

### 5.1 ReLU (Rectified Linear Unit)

**Implementation**:
```python
nn.ReLU(inplace=True)
```

**Mathematical Definition**:
```
f(x) = max(0, x) = {
    x,  if x > 0
    0,  if x ≤ 0
}
```

**Why ReLU?**

1. **Solves Vanishing Gradient Problem**:
   - Old activations (sigmoid, tanh) saturate → gradients → 0
   - ReLU has constant gradient (1) for positive inputs
   - Enables training of deep networks

2. **Computational Efficiency**:
   - Simple operation: `max(0, x)`
   - Fast forward and backward pass
   - No expensive exponentials (unlike sigmoid/tanh)

3. **Sparsity**:
   - ReLU outputs 0 for negative inputs
   - Creates sparse representations (many neurons inactive)
   - Reduces overfitting, improves generalization

4. **Biological Plausibility**:
   - Mimics neuron firing threshold
   - Only activates when input exceeds threshold

**Limitations and Mitigation**:
- **Dying ReLU Problem**: Neurons can get stuck at 0
- **Mitigation**: 
  - Proper weight initialization (Xavier uniform)
  - Dropout regularization
  - Learning rate scheduling

### 5.2 No Activation in Output Layers

**Implementation**:
```python
self.aperture_head = nn.Linear(256, num_classes)  # No activation
```

**Why No Activation?**
- **CrossEntropyLoss expects raw logits**: 
  - PyTorch's `CrossEntropyLoss` applies `LogSoftmax` internally
  - Adding softmax here would be redundant
  - More numerically stable to apply softmax in loss function

**Mathematical Flow**:
```
Raw Logits (256 → 3) → CrossEntropyLoss → Softmax + Negative Log Likelihood
```

---

## 6. Weights and Bias

### 6.1 Weight Initialization

**Method**: Xavier Uniform (Glorot Uniform)

**Implementation**:
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

**Xavier Uniform Formula**:
```
W ~ Uniform(-a, a)
where a = sqrt(6 / (fan_in + fan_out))
```

**Why Xavier Initialization?**

1. **Variance Preservation**:
   - Maintains variance of activations across layers
   - Prevents activations from vanishing or exploding
   - Works well with ReLU activations

2. **Symmetric Distribution**:
   - Weights initialized around 0
   - Allows gradients to flow in both directions
   - Prevents bias toward positive/negative values

3. **Layer-Aware**:
   - Considers both input and output dimensions
   - Adapts to layer size automatically

**Bias Initialization**:
- **Zero Initialization**: `bias = 0`
- **Rationale**: 
  - Common practice for most layers
  - Works well with batch normalization (though we don't use it)
  - Prevents initial bias toward any class

### 6.2 Weight Structure

**Total Parameters**:
- **Total**: 27,693,897 parameters
- **Trainable**: 12,979,209 parameters (47%)
- **Frozen**: 14,714,688 parameters (53%)

**Parameter Breakdown**:

1. **VGG16 Backbone (Frozen)**:
   - 13 convolutional layers
   - 3 fully connected layers (original classifier)
   - **14.7M parameters** (frozen)

2. **Shared FC Layers (Trainable)**:
   - FC1: 25088 × 512 + 512 = 12,845,568 parameters
   - FC2: 512 × 256 + 256 = 131,328 parameters
   - **Total**: ~13M parameters

3. **Classification Heads (Trainable)**:
   - Aperture: 256 × 3 + 3 = 771 parameters
   - ISO: 256 × 3 + 3 = 771 parameters
   - Shutter: 256 × 3 + 3 = 771 parameters
   - **Total**: 2,313 parameters

### 6.3 Weight Sharing

**Convolutional Layers**:
- **Shared Weights**: Same filter applied across entire image
- **Benefit**: Learns features regardless of position
- **Example**: Edge detector works at any location

**Multi-Head Architecture**:
- **Shared Backbone**: Same feature extractor for all three tasks
- **Separate Heads**: Different weights for each camera setting
- **Benefit**: 
  - Efficient: Learn once, use for multiple tasks
  - Related tasks share visual features
  - Task-specific heads learn setting-specific patterns

---

## 7. Weight and Bias Updates

### 7.1 Loss Function

**Multi-Task Loss**:
```python
class CameraSettingsLoss(nn.Module):
    def forward(self, predictions, targets):
        aperture_loss = CrossEntropyLoss(predictions['aperture'], targets['aperture'])
        iso_loss = CrossEntropyLoss(predictions['iso'], targets['iso'])
        shutter_loss = CrossEntropyLoss(predictions['shutter'], targets['shutter'])
        total_loss = aperture_loss + iso_loss + shutter_loss
        return total_loss
```

**CrossEntropyLoss Formula**:
```
L = -log(exp(x[class]) / Σexp(x[j]))
   = -x[class] + log(Σexp(x[j]))
```

**Why Sum of Losses?**
- **Equal Weighting**: All three tasks equally important
- **Simple**: No need for task-specific weighting
- **Effective**: Works well in practice

### 7.2 Backpropagation

**Gradient Flow**:

1. **Forward Pass**:
   ```
   Input → VGG16 (frozen) → Shared FC → Heads → Predictions
   ```

2. **Loss Calculation**:
   ```
   Predictions vs. Targets → CrossEntropyLoss → Total Loss
   ```

3. **Backward Pass**:
   ```
   ∂L/∂W_head ← ∂L/∂prediction × ∂prediction/∂W_head
   ∂L/∂W_shared ← ∂L/∂W_head × ∂W_head/∂W_shared
   ∂L/∂W_backbone ← 0 (frozen, no gradient)
   ```

**Gradient Calculation**:
```python
# In training loop
optimizer.zero_grad()           # Clear previous gradients
loss.backward()                  # Compute gradients
optimizer.step()                 # Update weights
```

### 7.3 Optimizer: Adam

**Implementation**:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Adam Algorithm**:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t          # First moment (momentum)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²         # Second moment (variance)
m̂_t = m_t / (1 - β₁^t)                       # Bias correction
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)        # Weight update
```

**Default Parameters**:
- **β₁ = 0.9**: Momentum decay
- **β₂ = 0.999**: Variance decay
- **α = 0.001**: Learning rate
- **ε = 1e-8**: Numerical stability

**Why Adam?**

1. **Adaptive Learning Rate**:
   - Different learning rates for each parameter
   - Large gradients → smaller step
   - Small gradients → larger step

2. **Momentum**:
   - Smooths gradient updates
   - Helps escape local minima
   - Faster convergence

3. **Works Well Out-of-the-Box**:
   - Less hyperparameter tuning needed
   - Robust to different architectures
   - Good default choice

**Weight Update Process**:
```python
# For each trainable parameter θ:
# 1. Compute gradient: g = ∂L/∂θ
# 2. Update first moment: m = 0.9 × m + 0.1 × g
# 3. Update second moment: v = 0.999 × v + 0.001 × g²
# 4. Bias correction: m̂ = m / (1 - 0.9^t), v̂ = v / (1 - 0.999^t)
# 5. Update weight: θ = θ - 0.001 × m̂ / (√v̂ + 1e-8)
```

### 7.4 Learning Rate Scheduling

**ReduceLROnPlateau**:
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # Reduce when loss stops decreasing
    factor=0.5,           # Multiply LR by 0.5
    patience=2            # Wait 2 epochs before reducing
)
```

**How It Works**:
1. Monitor validation loss
2. If loss doesn't improve for `patience` epochs → reduce LR
3. New LR = old LR × `factor`
4. Continue training with smaller LR

**Why This Helps?**:
- **Fine-tuning**: Smaller LR for fine-grained updates
- **Convergence**: Helps reach better minima
- **Prevents Overshooting**: Smaller steps near optimum

---

## 8. Training Process

### 8.1 Training Loop

**Epoch Structure**:
```python
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            predictions = model(images)
            loss = criterion(predictions, targets)
```

**Key Components**:

1. **model.train()**: Enables dropout, batch norm updates
2. **model.eval()**: Disables dropout, freezes batch norm
3. **torch.no_grad()**: Disables gradient computation (saves memory)

### 8.2 Early Stopping

**Implementation**:
```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
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
        return self.counter >= self.patience  # Stop if patience exceeded
```

**Rationale**:
- **Prevents Overfitting**: Stops when validation loss stops improving
- **Saves Time**: No need to train for full epochs
- **Best Model**: Saves model at best validation loss

### 8.3 Checkpointing

**Model Saving**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss
}
torch.save(checkpoint, 'checkpoints/best_model.pth')
```

**Benefits**:
- **Resume Training**: Can continue from checkpoint
- **Best Model**: Always have best performing model
- **Reproducibility**: Can reload exact model state

---

## 9. Error Handling and Corrections

### 9.1 Errors Encountered During Development

#### Error 1: ReduceLROnPlateau `verbose` Parameter
**Error**:
```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

**Cause**: PyTorch version incompatibility - `verbose` parameter removed in newer versions

**Fix**:
```python
# Before:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# After:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)
```

#### Error 2: Streamlit Progress Bar Type Error
**Error**:
```
StreamlitAPIException: Progress Value has invalid type: float32
```

**Cause**: Streamlit's `st.progress()` expects Python `float`, not NumPy `float32`

**Fix**:
```python
# Before:
st.progress(prob)  # prob is numpy.float32

# After:
prob_float = float(prob)  # Convert to Python float
st.progress(prob_float)
```

#### Error 3: Missing Dataset
**Error**:
```
FileNotFoundError: data/labels.csv not found
```

**Cause**: Dataset not prepared before training

**Fix**: Created `create_quick_data.py` to generate synthetic dataset for quick testing

#### Error 4: VGG16 Pretrained Parameter Deprecation
**Warning**:
```
UserWarning: The parameter 'pretrained' is deprecated since 0.13
```

**Cause**: PyTorch updated API

**Fix**: Should use `weights=VGG16_Weights.IMAGENET1K_V1` instead, but kept `pretrained=True` for compatibility

### 9.2 Error Prevention Strategies

1. **Type Checking**: Convert NumPy types to Python types when needed
2. **Path Validation**: Check file existence before loading
3. **Graceful Degradation**: Return dummy data if image loading fails
4. **Logging**: Comprehensive logging for debugging
5. **Version Compatibility**: Test with multiple PyTorch versions

---

## 10. Results and Evaluation

### 10.1 Training Metrics

**Example Training Run** (50 images, 3 epochs):
- **Best Validation Loss**: 6.0478
- **Final Validation Accuracies**:
  - Aperture: 37.5%
  - ISO: 43.8%
  - Shutter: 56.2%

**Note**: These are baseline results with synthetic data. Real dataset would show better performance.

### 10.2 Evaluation Metrics

**Accuracy Calculation**:
```python
pred_classes = torch.argmax(predictions[head], dim=1)
correct = (pred_classes == targets[head]).float()
accuracy = correct.mean().item()
```

**Why Accuracy?**:
- **Simple**: Easy to understand and interpret
- **Appropriate**: Balanced classes (3 classes each)
- **Practical**: Directly relates to prediction correctness

### 10.3 Model Performance Considerations

**Factors Affecting Performance**:
1. **Dataset Size**: More data → better generalization
2. **Data Quality**: Real photos vs. synthetic data
3. **Class Balance**: Balanced classes improve learning
4. **Hyperparameters**: Learning rate, batch size, epochs
5. **Architecture**: Model capacity vs. dataset size

**Expected Performance with Real Data**:
- **Aperture**: 60-75% accuracy (depth of field is visually apparent)
- **ISO**: 50-65% accuracy (noise patterns are subtle)
- **Shutter**: 55-70% accuracy (motion blur is visible)

---

## 11. Conclusion

### 11.1 Key Design Decisions Summary

1. **Transfer Learning**: VGG16 pretrained backbone for feature extraction
2. **Multi-Task Learning**: Shared features, separate heads for each setting
3. **Class Binning**: 3 classes per setting for manageable classification
4. **Data Augmentation**: Improves generalization with limited data
5. **Adam Optimizer**: Adaptive learning rate for robust training
6. **Early Stopping**: Prevents overfitting, saves best model

### 11.2 Future Improvements

1. **Larger Dataset**: Process more images from FiveK dataset
2. **Fine-tuning**: Unfreeze backbone layers for better adaptation
3. **Class Imbalance**: Add class weights to loss function
4. **Architecture**: Experiment with ResNet, EfficientNet
5. **Ensemble**: Combine multiple models for better accuracy
6. **Regression**: Predict exact values instead of classes

---

## Appendix: Mathematical Foundations

### A.1 Forward Pass Mathematics

For a linear layer: `y = Wx + b`

Where:
- `W`: Weight matrix (256 × 3 for heads)
- `x`: Input vector (256 dimensions)
- `b`: Bias vector (3 dimensions)
- `y`: Output logits (3 dimensions)

### A.2 Backpropagation Mathematics

**Gradient Calculation**:
```
∂L/∂W = (1/n) × Σ(∂L/∂y × x^T)
∂L/∂b = (1/n) × Σ(∂L/∂y)
∂L/∂x = W^T × (∂L/∂y)
```

**Chain Rule Application**:
```
∂L/∂W_head = ∂L/∂prediction × ∂prediction/∂W_head
∂L/∂W_shared = ∂L/∂W_head × ∂W_head/∂shared_features × ∂shared_features/∂W_shared
```

### A.3 Loss Function Mathematics

**CrossEntropyLoss**:
```
L = -log(softmax(logits)[target])
   = -log(exp(logits[target]) / Σexp(logits[i]))
   = -logits[target] + log(Σexp(logits[i]))
```

**Gradient of CrossEntropyLoss**:
```
∂L/∂logits[i] = {
    softmax(logits)[i] - 1,  if i == target
    softmax(logits)[i],      if i ≠ target
}
```

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Author**: PhotoPilot Development Team


