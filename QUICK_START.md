# Quick Start Guide - Small Dataset Training

## Important Note About Dataset Size

The MIT-Adobe FiveK dataset on Kaggle is ~28GB. Unfortunately, the Kaggle API downloads the entire dataset even if you only want to process a small subset.

**However**, the modified script will:
- Download the full dataset (one-time, ~28GB)
- Process only a small number of images (default: 200 images)
- Use only ~2-3GB of disk space for the processed images

## Quick Start Options

### Option 1: Prepare Data Only (200 images)
```bash
python prepare_fivek.py --max_images 200 --copy_to_training
```

### Option 2: Prepare Data with Custom Size
```bash
# Process only 100 images
python prepare_fivek.py --max_images 100 --copy_to_training

# Process 500 images
python prepare_fivek.py --max_images 500 --copy_to_training
```

### Option 3: All-in-One (Prepare + Train)
```bash
# Prepare 200 images and train for 5 epochs
./prepare_and_train.sh 200 5

# Prepare 100 images and train for 10 epochs
./prepare_and_train.sh 100 10
```

### Option 4: Manual Steps
```bash
# 1. Prepare dataset (200 images)
python prepare_fivek.py --max_images 200 --copy_to_training

# 2. Train model
cd fivek-cnn
python train.py --csv data/labels.csv --image_dir data/jpg --epochs 5
```

## If Dataset Already Downloaded

If you've already downloaded the dataset, use the `--skip_download` flag:

```bash
python prepare_fivek.py --max_images 200 --skip_download --copy_to_training
```

## Training Parameters

You can customize training:

```bash
cd fivek-cnn
python train.py \
    --csv data/labels.csv \
    --image_dir data/jpg \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.0001
```

## Recommended Settings for Small Datasets

- **100-200 images**: Good for quick testing, epochs=5-10
- **300-500 images**: Better for actual training, epochs=10-15
- **1000+ images**: Full training, epochs=15-20


