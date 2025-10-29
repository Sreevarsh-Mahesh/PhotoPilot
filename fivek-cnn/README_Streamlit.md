# 📸 Camera Assistant AI - Streamlit UI

A beautiful web interface for uploading photos and getting optimal camera settings suggestions using our trained CNN model.

## 🚀 Quick Start

### 1. Run the Streamlit App
```bash
cd fivek-cnn
streamlit run app.py
```

### 2. Open in Browser
The app will automatically open in your browser at `http://localhost:8501`

## ✨ Features

### 🎨 Beautiful UI
- **Modern gradient design** with professional styling
- **Responsive layout** that works on desktop and mobile
- **Interactive elements** with smooth animations
- **Color-coded predictions** for easy understanding

### 📤 Easy Photo Upload
- **Drag & drop interface** for uploading photos
- **Multiple format support**: JPG, PNG, BMP, TIFF
- **Real-time image preview** before prediction
- **Image information display** (size, format, etc.)

### 🎯 Smart Predictions
- **Three camera settings**: Aperture, ISO, Shutter Speed
- **Confidence scores** with visual progress bars
- **Detailed probability breakdown** for each setting
- **Human-readable class names** (e.g., "High (>f/11)")

### 📊 Detailed Analysis
- **Probability visualization** for all classes
- **Confidence indicators** for each prediction
- **Downloadable results** as text file
- **Class meaning explanations** in sidebar

### ⚙️ Advanced Features
- **GPU/CPU automatic detection**
- **Model status monitoring**
- **Error handling** with user-friendly messages
- **Sidebar with settings** and information

## 🎨 UI Components

### Main Interface
- **Header**: Gradient title with app branding
- **Upload Area**: Drag & drop zone with file type validation
- **Prediction Cards**: Beautiful cards showing camera settings
- **Confidence Bars**: Visual representation of prediction confidence

### Sidebar
- **Model Status**: Shows if model is loaded and device being used
- **Class Meanings**: Explains what each class represents
- **About Section**: Information about the AI model

### Prediction Display
- **Setting Cards**: Individual cards for each camera setting
- **Confidence Visualization**: Progress bars showing prediction confidence
- **Probability Breakdown**: Detailed view of all class probabilities
- **Download Option**: Save results as text file

## 🔧 Technical Details

### Architecture
- **Backend**: PyTorch CNN model (VGG16-based)
- **Frontend**: Streamlit with custom CSS
- **Image Processing**: PIL with torchvision transforms
- **Device Support**: Automatic GPU/CPU detection

### Supported Formats
- **Input**: JPG, JPEG, PNG, BMP, TIFF
- **Output**: Human-readable text with probabilities
- **Download**: Plain text file with results

### Performance
- **Fast inference** with optimized model loading
- **Memory efficient** image processing
- **Real-time predictions** with progress indicators
- **Error handling** for invalid images

## 📱 Usage Examples

### Basic Usage
1. Open the app in your browser
2. Upload a photo using drag & drop or file picker
3. View the predicted camera settings
4. Check confidence scores and detailed probabilities
5. Download results if needed

### Advanced Usage
1. Use the sidebar to understand class meanings
2. Check model status and device information
3. View detailed probability breakdowns
4. Compare different photos for different scenarios

## 🎯 Expected Output

When you upload a photo, you'll see:

```
📸 Predicted Optimal Settings
========================================
  Aperture: High (>f/11)     [Confidence: 85%]
  ISO: Medium (400-800)      [Confidence: 72%]
  Shutter: Slow (>1/30s)     [Confidence: 91%]
```

## 🚀 Running the App

### Local Development
```bash
# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Production Deployment
```bash
# Run with specific host and port
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🔧 Customization

### Styling
- Modify the CSS in the `st.markdown()` sections
- Change colors, fonts, and layout as needed
- Add custom components for specific needs

### Model Integration
- Update model path in `load_model()` method
- Modify preprocessing in `preprocess_image()`
- Add new prediction heads if needed

### UI Components
- Add new sidebar sections
- Create custom prediction displays
- Implement additional file formats

## 🐛 Troubleshooting

### Common Issues
1. **Model not loading**: Check if `checkpoints/best_model.pth` exists
2. **Image upload fails**: Ensure image format is supported
3. **GPU not detected**: Check CUDA installation
4. **App won't start**: Verify all requirements are installed

### Error Messages
- **"Model file not found"**: Train the model first using `train.py`
- **"Error preprocessing image"**: Check image format and size
- **"Error during prediction"**: Verify model is loaded correctly

## 📈 Performance Tips

1. **Use GPU** for faster inference
2. **Resize large images** before uploading
3. **Close other apps** to free up memory
4. **Use supported formats** (JPG recommended)

## 🔮 Future Enhancements

- **Batch processing** for multiple images
- **Image editing** tools for better predictions
- **Settings history** to track previous predictions
- **Export options** for different formats
- **Mobile app** version
- **API endpoint** for integration

---

**Enjoy using Camera Assistant AI! 📸✨**
