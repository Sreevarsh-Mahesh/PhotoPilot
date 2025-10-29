# 📸 PhotoPilot - Camera Assistant AI

An intelligent camera settings recommendation system that uses deep learning to analyze photos and suggest optimal aperture, ISO, and shutter speed settings for better photography.

## 🚀 Features

- **AI-Powered Analysis**: Uses a VGG16-based CNN trained on the MIT-Adobe FiveK dataset
- **Three Camera Settings**: Predicts aperture, ISO, and shutter speed classes
- **Beautiful Web Interface**: Streamlit-based UI with drag & drop photo upload
- **Command Line Tools**: Batch processing and inference scripts
- **Real-time Predictions**: Instant camera settings suggestions
- **Detailed Analytics**: Confidence scores and probability breakdowns

## 🎯 Camera Settings Classification

The model predicts **3 classes each** for:

- **Aperture**: 0=<5.6, 1=5.6-11, 2=>11
- **ISO**: 0≤200, 1=400-800, 2=>800  
- **Shutter Speed**: 0<1/250, 1=1/250-1/30, 2=>1/30

## 📁 Project Structure

```
PhotoPilot/
├── prepare_fivek.py              # Dataset preparation script
├── requirements.txt              # Main requirements
├── run_all.sh                   # One-click setup script
├── fivek-cnn/                   # CNN project directory
│   ├── data/                    # Data directory (images + labels)
│   ├── app.py                   # Streamlit web interface
│   ├── dataset.py               # PyTorch Dataset class
│   ├── model.py                 # VGG16-based CNN model
│   ├── train.py                 # Training script
│   ├── infer.py                 # Command-line inference
│   ├── requirements.txt         # CNN-specific requirements
│   └── README_Streamlit.md      # Streamlit documentation
└── kaggle/
    └── kaggle.json              # Kaggle API credentials (add your own)
```

## 🚀 Quick Start

### Option 1: One-Click Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/Sreevarsh-Mahesh/PhotoPilot.git
cd PhotoPilot

# Add your Kaggle API credentials
# Place your kaggle.json in the kaggle/ folder

# Run the complete pipeline
bash run_all.sh
```

### Option 2: Manual Setup

```bash
# 1. Install requirements
pip install -r requirements.txt
pip install -r fivek-cnn/requirements.txt

# 2. Set up Kaggle API
# Download kaggle.json from https://www.kaggle.com/account
# Place it in kaggle/kaggle.json

# 3. Prepare dataset
python prepare_fivek.py

# 4. Train model
cd fivek-cnn
python train.py --csv data/labels.csv --image_dir data/jpg

# 5. Launch web interface
streamlit run app.py
```

## 🌐 Web Interface

The Streamlit web interface provides:

- **📤 Drag & Drop Upload**: Easy photo upload with multiple format support
- **🎯 Real-time Predictions**: Instant camera settings suggestions
- **📊 Visual Analytics**: Confidence bars and probability breakdowns
- **💾 Download Results**: Save predictions as text files
- **📱 Responsive Design**: Works on desktop and mobile

### Launch Web Interface

```bash
cd fivek-cnn
streamlit run app.py
```

Open your browser to: `http://localhost:8501`

## 💻 Command Line Usage

### Single Image Prediction

```bash
cd fivek-cnn
python infer.py --image your_photo.jpg
```

### Batch Processing

```bash
python infer.py --batch image1.jpg image2.jpg image3.jpg
```

### Detailed Probabilities

```bash
python infer.py --image your_photo.jpg --probabilities
```

## 📊 Expected Output

```
Predicted Optimal Settings:
========================================
  Aperture: High (>f/11)
  ISO: High (>800)
  Shutter: Slow (>1/30s)
```

## 🔧 Technical Details

### Architecture
- **Backend**: PyTorch CNN with VGG16 pretrained backbone
- **Frontend**: Streamlit with custom CSS styling
- **Dataset**: MIT-Adobe FiveK (1000 images for speed)
- **Input Size**: 128x128 RGB images
- **Output**: 3 classification heads (aperture, ISO, shutter)

### Model Specifications
- **Base Model**: VGG16 (pretrained, frozen)
- **Custom Heads**: 3 fully connected layers (256 units each)
- **Loss Function**: CrossEntropyLoss (sum of 3 heads)
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 5 (fast training)

### Performance
- **Training Time**: ~10-15 minutes (1000 images)
- **Inference Speed**: <1 second per image
- **Memory Usage**: ~2GB GPU / ~4GB CPU
- **Accuracy**: ~70-80% on validation set

## 📋 Requirements

### System Requirements
- Python 3.7+
- 8GB+ RAM recommended
- GPU optional but recommended for training
- 10GB+ free disk space

### Python Dependencies
- PyTorch 2.0+
- Streamlit 1.28+
- Pillow 9.0+
- Pandas 1.5+
- NumPy 1.21+
- And more (see requirements.txt)

## 🎨 Web Interface Screenshots

The Streamlit interface includes:
- Beautiful gradient design
- Interactive photo upload
- Real-time prediction cards
- Confidence visualization
- Detailed probability breakdowns
- Downloadable results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/) for training data
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Streamlit](https://streamlit.io/) for web interface
- [Kaggle](https://kaggle.com/) for dataset hosting

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Sreevarsh-Mahesh/PhotoPilot/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Mobile app version
- [ ] More camera settings (white balance, focus mode)
- [ ] Style transfer integration
- [ ] Batch processing web interface
- [ ] API endpoint for integration
- [ ] Advanced image editing tools
- [ ] Settings history and comparison

---

**Built with ❤️ by [Sreevarsh Mahesh](https://github.com/Sreevarsh-Mahesh)**

*PhotoPilot - Your AI-powered photography assistant* 📸✨