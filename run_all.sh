#!/bin/bash

# Camera Assistant ANN Project - One-Click Run Script
# This script automates the entire pipeline from dataset preparation to model training and inference

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        print_status "Checking GPU availability..."
        if nvidia-smi >/dev/null 2>&1; then
            print_success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
            return 0
        else
            print_warning "nvidia-smi found but GPU not accessible"
            return 1
        fi
    else
        print_warning "nvidia-smi not found, assuming CPU-only"
        return 1
    fi
}

# Function to setup Kaggle API
setup_kaggle() {
    print_status "Setting up Kaggle API..."
    
    if [ -f "kaggle/kaggle.json" ]; then
        print_success "Found kaggle.json in local kaggle/ folder"
        mkdir -p ~/.kaggle
        cp kaggle/kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
        print_success "Kaggle API configured successfully"
    elif [ -f ~/.kaggle/kaggle.json ]; then
        print_success "Kaggle API already configured"
    else
        print_error "Kaggle API not found!"
        print_status "Please download kaggle.json from https://www.kaggle.com/account and place it in:"
        print_status "  - kaggle/kaggle.json (preferred), or"
        print_status "  - ~/.kaggle/kaggle.json"
        exit 1
    fi
}

# Function to install requirements
install_requirements() {
    print_status "Installing requirements..."
    
    # Check if pip exists
    if ! command_exists pip && ! command_exists pip3; then
        print_error "pip not found! Please install Python and pip first."
        exit 1
    fi
    
    # Use pip3 if available, otherwise pip
    PIP_CMD="pip"
    if command_exists pip3; then
        PIP_CMD="pip3"
    fi
    
    # Install main requirements
    print_status "Installing main requirements..."
    $PIP_CMD install -r requirements.txt
    
    # Install CNN requirements
    print_status "Installing CNN requirements..."
    $PIP_CMD install -r fivek-cnn/requirements.txt
    
    print_success "Requirements installed successfully"
}

# Function to prepare dataset
prepare_dataset() {
    print_status "Preparing MIT-Adobe FiveK dataset..."
    
    if [ ! -f "prepare_fivek.py" ]; then
        print_error "prepare_fivek.py not found!"
        exit 1
    fi
    
    # Run dataset preparation
    python prepare_fivek.py
    
    if [ $? -eq 0 ]; then
        print_success "Dataset preparation completed"
    else
        print_error "Dataset preparation failed!"
        exit 1
    fi
}

# Function to setup CNN data
setup_cnn_data() {
    print_status "Setting up CNN data structure..."
    
    # Create CNN data directory
    mkdir -p fivek-cnn/data
    
    # Copy CSV file
    if [ -f "fivek_data/camera_settings.csv" ]; then
        cp fivek_data/camera_settings.csv fivek-cnn/data/labels.csv
        print_success "Labels CSV copied to fivek-cnn/data/labels.csv"
    else
        print_error "Camera settings CSV not found! Run dataset preparation first."
        exit 1
    fi
    
    # Create symlink to images (or copy if symlink fails)
    if [ -d "fivek_data/jpg" ]; then
        if ln -sf "$(pwd)/fivek_data/jpg" fivek-cnn/data/jpg 2>/dev/null; then
            print_success "Image directory linked to fivek-cnn/data/jpg"
        else
            print_status "Creating copy of image directory..."
            cp -r fivek_data/jpg fivek-cnn/data/
            print_success "Image directory copied to fivek-cnn/data/jpg"
        fi
    else
        print_error "Image directory not found! Run dataset preparation first."
        exit 1
    fi
}

# Function to train model
train_model() {
    print_status "Training CNN model..."
    
    cd fivek-cnn
    
    if [ ! -f "train.py" ]; then
        print_error "train.py not found in fivek-cnn directory!"
        exit 1
    fi
    
    # Run training
    python train.py --csv data/labels.csv --image_dir data/jpg --epochs 5 --batch_size 32
    
    if [ $? -eq 0 ]; then
        print_success "Model training completed"
    else
        print_error "Model training failed!"
        exit 1
    fi
    
    cd ..
}

# Function to test inference
test_inference() {
    print_status "Testing inference..."
    
    cd fivek-cnn
    
    if [ ! -f "infer.py" ]; then
        print_error "infer.py not found in fivek-cnn directory!"
        exit 1
    fi
    
    # Find a sample image
    SAMPLE_IMAGE=""
    if [ -d "data/jpg" ]; then
        SAMPLE_IMAGE=$(find data/jpg -name "*.jpg" | head -1)
    fi
    
    if [ -n "$SAMPLE_IMAGE" ]; then
        print_status "Testing inference on: $SAMPLE_IMAGE"
        python infer.py --image "$SAMPLE_IMAGE"
        
        if [ $? -eq 0 ]; then
            print_success "Inference test completed"
        else
            print_warning "Inference test failed, but model may still be usable"
        fi
    else
        print_warning "No sample images found for inference test"
    fi
    
    cd ..
}

# Function to start Streamlit app
start_streamlit() {
    print_status "Starting Streamlit web interface..."
    
    cd fivek-cnn
    
    if [ ! -f "app.py" ]; then
        print_error "app.py not found in fivek-cnn directory!"
        exit 1
    fi
    
    print_success "Streamlit app is ready!"
    print_status "To start the web interface, run:"
    print_status "  cd fivek-cnn"
    print_status "  streamlit run app.py"
    print_status ""
    print_status "The app will open at: http://localhost:8501"
    
    cd ..
}

# Function to print final instructions
print_final_instructions() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}🎉 Camera Assistant ANN Project Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Your model is ready! Here's how to use it:"
    echo ""
    echo "1. Test inference on a sample image:"
    echo "   cd fivek-cnn"
    echo "   python infer.py --image data/jpg/your_image.jpg"
    echo ""
    echo "2. Test on your own photos:"
    echo "   python infer.py --image /path/to/your/photo.jpg"
    echo ""
    echo "3. Batch process multiple images:"
    echo "   python infer.py --batch image1.jpg image2.jpg image3.jpg"
    echo ""
    echo "4. Show detailed probabilities:"
    echo "   python infer.py --image your_photo.jpg --probabilities"
    echo ""
    echo "5. Launch Streamlit web interface:"
    echo "   cd fivek-cnn"
    echo "   streamlit run app.py"
    echo "   (Opens at http://localhost:8501)"
    echo ""
    echo "Model files:"
    echo "  - fivek-cnn/checkpoints/best_model.pth (best model)"
    echo "  - fivek-cnn/checkpoints/training_history.csv (training logs)"
    echo ""
    echo "Web Interface Features:"
    echo "  - 📤 Drag & drop photo upload"
    echo "  - 🎯 Real-time camera settings prediction"
    echo "  - 📊 Detailed probability visualization"
    echo "  - 💾 Download results as text file"
    echo "  - 🎨 Beautiful, responsive UI"
    echo ""
    echo "Expected output format:"
    echo "  Predicted Optimal Settings:"
    echo "  ========================================"
    echo "    Aperture: High (>f/11)"
    echo "    ISO: High (>800)"
    echo "    Shutter: Slow (>1/30s)"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo -e "${BLUE}Camera Assistant ANN Project${NC}"
    echo "=========================================="
    echo ""
    
    # Check Python
    if ! command_exists python && ! command_exists python3; then
        print_error "Python not found! Please install Python 3.7+ first."
        exit 1
    fi
    
    # Check GPU
    check_gpu
    
    # Setup Kaggle API
    setup_kaggle
    
    # Install requirements
    install_requirements
    
    # Prepare dataset
    prepare_dataset
    
    # Setup CNN data
    setup_cnn_data
    
    # Train model
    train_model
    
    # Test inference
    test_inference
    
    # Start Streamlit app
    start_streamlit
    
    # Print final instructions
    print_final_instructions
}

# Run main function
main "$@"
