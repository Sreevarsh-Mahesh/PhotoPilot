"""
Streamlit UI for Camera Settings Prediction

A user-friendly web interface for uploading photos and getting optimal camera settings
suggestions using the trained CNN model.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_model, get_class_names
from dataset import get_transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PhotoPilot",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .setting-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #fff;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        height: 20px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CameraSettingsApp:
    """Main Streamlit app for camera settings prediction."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = get_class_names()
        self.transform = None
        
    def load_model(self, model_path=None):
        """Load the trained model."""
        try:
            if self.model is None:
                # Determine device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Create model
                self.model, _ = create_model(device=self.device)
                
                # Determine model path (handle both local and deployed scenarios)
                if model_path is None:
                    # Get the directory where this script is located
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(script_dir, "checkpoints", "best_model.pth")
                
                # Load checkpoint
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    self.model.eval()
                    
                    # Get transforms
                    _, self.transform = get_transforms()
                    
                    logger.info(f"Model loaded successfully on {self.device}")
                    return True
                else:
                    st.error(f"Model file not found: {model_path}")
                    return False
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for inference."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image):
        """Predict camera settings for an image."""
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            if image_tensor is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Get predicted classes and probabilities
            results = {}
            
            for head in ['aperture', 'iso', 'shutter']:
                # Get class probabilities
                probs = torch.softmax(predictions[head], dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                probabilities = probs.cpu().numpy()[0]
                
                results[head] = {
                    'class': predicted_class,
                    'class_name': self.class_names[head][predicted_class],
                    'probabilities': probabilities,
                    'confidence': float(probabilities[predicted_class])
                }
            
            return results
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    
    def display_prediction(self, results):
        """Display prediction results in a nice format."""
        if results is None:
            return
        
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 1rem;">📸 Predicted Optimal Settings</h2>', unsafe_allow_html=True)
        
        for head in ['aperture', 'iso', 'shutter']:
            result = results[head]
            
            st.markdown(f'<div class="setting-item">', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: white; margin: 0;">{head.capitalize()}: {result["class_name"]}</h3>', unsafe_allow_html=True)
            
            # Display confidence as text only (removed progress bar)
            confidence = result['confidence']
            st.markdown(f'<p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_probabilities(self, results):
        """Display detailed probability breakdown."""
        if results is None:
            return
        
        st.markdown("### 📊 Detailed Probabilities")
        
        for head in ['aperture', 'iso', 'shutter']:
            result = results[head]
            probs = result['probabilities']
            
            st.markdown(f"**{head.capitalize()}:**")
            
            # Create columns for each class
            cols = st.columns(3)
            
            for i, (class_name, prob) in enumerate(zip(self.class_names[head], probs)):
                with cols[i]:
                    # Convert numpy float32 to Python float for Streamlit progress bar
                    prob_float = float(prob)
                    # Highlight the predicted class
                    if i == result['class']:
                        st.markdown(f"**{class_name}**")
                        st.progress(prob_float)
                        st.markdown(f"**{prob_float:.1%}**")
                    else:
                        st.markdown(class_name)
                        st.progress(prob_float)
                        st.markdown(f"{prob_float:.1%}")

def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">📸 PhotoPilot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload a photo to get optimal camera settings suggestions</p>', unsafe_allow_html=True)
    
    # Initialize app
    app = CameraSettingsApp()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model status
        model_loaded = app.load_model()
        if model_loaded:
            st.success("✅ Model loaded successfully")
            if torch.cuda.is_available():
                st.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.info("💻 Using CPU")
        else:
            st.error("❌ Model not loaded")
            st.stop()
        
        st.markdown("---")
        
        # Show class meanings
        st.markdown("### 📚 Class Meanings")
        
        for setting, classes in app.class_names.items():
            with st.expander(f"{setting.capitalize()} Classes"):
                for i, class_name in enumerate(classes):
                    st.markdown(f"**{i}:** {class_name}")
        
        st.markdown("---")
        
        # Show model info
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI model predicts optimal camera settings based on image analysis:
        
        - **Aperture**: Controls depth of field
        - **ISO**: Controls light sensitivity  
        - **Shutter Speed**: Controls motion blur
        
        The model was trained on the MIT-Adobe FiveK dataset.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Photo")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a photo to get camera settings suggestions"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown(f"**Image Info:**")
            st.markdown(f"- Size: {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"- Mode: {image.mode}")
            st.markdown(f"- Format: {uploaded_file.type}")
    
    with col2:
        st.markdown("### 🎯 Predictions")
        
        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                results = app.predict(image)
            
            if results is not None:
                # Display predictions
                app.display_prediction(results)
                
                # Show detailed probabilities
                with st.expander("📊 View Detailed Probabilities"):
                    app.display_probabilities(results)
                
                # Download results
                st.markdown("### 💾 Download Results")
                
                # Create results text
                results_text = "Camera Settings Prediction Results\n"
                results_text += "=" * 40 + "\n\n"
                
                for head in ['aperture', 'iso', 'shutter']:
                    result = results[head]
                    results_text += f"{head.capitalize()}: {result['class_name']} (Confidence: {result['confidence']:.1%})\n"
                
                results_text += f"\nDetailed Probabilities:\n"
                for head in ['aperture', 'iso', 'shutter']:
                    result = results[head]
                    results_text += f"\n{head.capitalize()}:\n"
                    for i, (class_name, prob) in enumerate(zip(app.class_names[head], result['probabilities'])):
                        results_text += f"  {i}: {class_name} - {prob:.1%}\n"
                
                st.download_button(
                    label="📥 Download Results as Text",
                    data=results_text,
                    file_name="camera_settings_prediction.txt",
                    mime="text/plain"
                )
        else:
            # Show placeholder
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            st.markdown("👆 Upload a photo to get started!")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>PhotoPilot - Powered by PyTorch & Streamlit</p>
        <p>Built with the MIT-Adobe FiveK dataset</p>
        <p style="margin-top: 1rem;">Built by <strong>Sreevarsh Mahesh Gandhi</strong></p>
        <p style="margin-top: 0.5rem;">
            <a href="https://www.linkedin.com/in/sreevarsh-mahesh-gandhi-53b19024a/" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">
                🔗 LinkedIn
            </a>
            <a href="https://github.com/Sreevarsh-Mahesh" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">
                💻 GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
