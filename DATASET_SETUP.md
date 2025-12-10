# Dataset Setup Instructions

## Current Status

The script has been updated to work with smaller datasets (default: 200 images). However, the Kaggle API download requires proper setup.

## Option 1: Fix Kaggle API Setup (Recommended)

1. **Get Kaggle API credentials:**
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token" to download `kaggle.json`

2. **Set up credentials:**
   ```bash
   # Option A: Place in project kaggle/ folder
   mkdir -p kaggle
   # Copy your kaggle.json to kaggle/kaggle.json
   
   # Option B: Place in standard location
   mkdir -p ~/.kaggle
   # Copy your kaggle.json to ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Accept dataset terms:**
   - Visit: https://www.kaggle.com/datasets/mit-adobe-fivek
   - Click "New Notebook" or "Download" to accept terms

4. **Run preparation:**
   ```bash
   # Process 100 images (smaller, faster)
   python prepare_fivek.py --max_images 100 --copy_to_training
   ```

## Option 2: Manual Download (If Kaggle API doesn't work)

1. **Download dataset manually:**
   - Go to: https://www.kaggle.com/datasets/mit-adobe-fivek
   - Download the dataset (you can stop after getting some RAW files)
   - Extract to `fivek_data/MIT-Adobe FiveK/raw_photos/`

2. **Process with skip_download:**
   ```bash
   python prepare_fivek.py --max_images 100 --skip_download --copy_to_training
   ```

## Option 3: Use Alternative Smaller Dataset

If you have access to a smaller photography dataset with EXIF data, you can:
1. Place RAW files in `fivek_data/raw_photos/`
2. Run: `python prepare_fivek.py --max_images 100 --skip_download --copy_to_training`

## Quick Test (After Setup)

Once data is prepared, train the model:

```bash
cd fivek-cnn
python train.py --csv data/labels.csv --image_dir data/jpg --epochs 5
```

## Recommended Settings

- **For quick testing**: 50-100 images, 5 epochs
- **For actual training**: 200-500 images, 10-15 epochs
- **For best results**: 1000+ images, 15-20 epochs

## Notes

- The full dataset is ~28GB, but processing only 100-200 images uses ~2-3GB
- Processing time: ~5-10 minutes for 100 images, ~20-30 minutes for 500 images
- Training time: ~5-10 minutes for 100 images (5 epochs), ~30-60 minutes for 500 images (10 epochs)


