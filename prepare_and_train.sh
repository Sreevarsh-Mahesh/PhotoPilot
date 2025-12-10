#!/bin/bash
# Script to prepare small dataset and train the model

set -e

echo "=========================================="
echo "PhotoPilot - Dataset Preparation & Training"
echo "=========================================="
echo ""

# Default values
MAX_IMAGES=${1:-200}  # Use first argument or default to 200
EPOCHS=${2:-5}        # Use second argument or default to 5

echo "Configuration:"
echo "  - Max images to process: $MAX_IMAGES"
echo "  - Training epochs: $EPOCHS"
echo ""
echo "Note: The full MIT-Adobe FiveK dataset is ~28GB."
echo "      We will download it but only process $MAX_IMAGES images."
echo ""

# Step 1: Prepare dataset
echo "Step 1: Preparing dataset..."
python prepare_fivek.py --max_images $MAX_IMAGES --copy_to_training

# Check if data was created
if [ ! -f "fivek-cnn/data/labels.csv" ]; then
    echo "Error: Data preparation failed. labels.csv not found."
    exit 1
fi

echo ""
echo "Step 2: Starting training..."
cd fivek-cnn
python train.py --csv data/labels.csv --image_dir data/jpg --epochs $EPOCHS

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="


