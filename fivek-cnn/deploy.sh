#!/bin/bash

# PhotoPilot Deployment Setup Script
# This script helps prepare your app for deployment

set -e

echo "🚀 PhotoPilot Deployment Setup"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}Git repository not initialized.${NC}"
    read -p "Do you want to initialize git? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git init
        echo -e "${GREEN}✓ Git initialized${NC}"
    fi
fi

# Check model file size
MODEL_FILE="checkpoints/best_model.pth"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo -e "Model file size: ${YELLOW}$MODEL_SIZE${NC}"
    echo ""
    
    # Check if over 100MB
    SIZE_BYTES=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE")
    if [ $SIZE_BYTES -gt 104857600 ]; then
        echo -e "${YELLOW}⚠️  Model file is larger than 100MB${NC}"
        echo "You'll need to use Git LFS or external storage."
        echo ""
        
        # Check if git-lfs is installed
        if ! command -v git-lfs &> /dev/null; then
            echo -e "${RED}Git LFS is not installed${NC}"
            echo "Install with: brew install git-lfs"
            echo ""
            read -p "Do you want to install Git LFS now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                brew install git-lfs
                git lfs install
                echo -e "${GREEN}✓ Git LFS installed${NC}"
            fi
        else
            echo -e "${GREEN}✓ Git LFS is installed${NC}"
            
            # Setup LFS tracking
            if [ ! -f .gitattributes ]; then
                git lfs track "checkpoints/*.pth"
                echo -e "${GREEN}✓ Git LFS tracking configured${NC}"
            fi
        fi
    fi
else
    echo -e "${RED}⚠️  Model file not found: $MODEL_FILE${NC}"
    echo "Make sure to train the model first!"
fi

echo ""
echo "📝 Deployment Options:"
echo "1. Streamlit Cloud (Free, easiest)"
echo "2. Hugging Face Spaces (Free, with GPU option)"
echo "3. Docker (Self-host or cloud)"
echo "4. Railway (Paid, simple)"
echo ""

read -p "Which deployment option? (1-4): " DEPLOY_OPTION

case $DEPLOY_OPTION in
    1)
        echo ""
        echo -e "${GREEN}Preparing for Streamlit Cloud...${NC}"
        echo ""
        echo "Steps to deploy:"
        echo "1. Push your code to GitHub"
        echo "2. Go to https://share.streamlit.io"
        echo "3. Sign in and click 'New app'"
        echo "4. Select your repository and branch"
        echo "5. Set main file path: fivek-cnn/app.py"
        echo ""
        
        # Check if remote exists
        if git remote get-url origin &> /dev/null; then
            echo "Git remote already configured:"
            git remote get-url origin
        else
            read -p "Enter your GitHub repository URL: " REPO_URL
            git remote add origin "$REPO_URL"
            echo -e "${GREEN}✓ Remote added${NC}"
        fi
        
        echo ""
        read -p "Ready to commit and push? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add .
            git commit -m "Prepare for Streamlit Cloud deployment"
            git push -u origin main
            echo -e "${GREEN}✓ Pushed to GitHub${NC}"
            echo ""
            echo "Now go to: https://share.streamlit.io"
        fi
        ;;
        
    2)
        echo ""
        echo -e "${GREEN}Preparing for Hugging Face Spaces...${NC}"
        echo ""
        echo "Steps to deploy:"
        echo "1. Create account at https://huggingface.co"
        echo "2. Create a new Space (SDK: Streamlit)"
        echo "3. Clone the space repository"
        echo "4. Copy your files to the space"
        echo "5. Push to Hugging Face"
        echo ""
        echo "See DEPLOYMENT.md for detailed instructions"
        ;;
        
    3)
        echo ""
        echo -e "${GREEN}Preparing Docker deployment...${NC}"
        echo ""
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Docker is not installed${NC}"
            echo "Install from: https://www.docker.com/products/docker-desktop"
        else
            echo -e "${GREEN}✓ Docker is installed${NC}"
            echo ""
            read -p "Build Docker image now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker build -t photopilot .
                echo -e "${GREEN}✓ Docker image built${NC}"
                echo ""
                echo "Test locally with:"
                echo "  docker run -p 8501:8501 photopilot"
                echo ""
                echo "Or use docker-compose:"
                echo "  docker-compose up"
            fi
        fi
        ;;
        
    4)
        echo ""
        echo -e "${GREEN}Preparing for Railway...${NC}"
        echo ""
        echo "Steps to deploy:"
        echo "1. Go to https://railway.app"
        echo "2. Sign up/login"
        echo "3. New Project → Deploy from GitHub"
        echo "4. Select your repository"
        echo "5. Railway auto-detects Streamlit"
        echo ""
        echo "Cost: ~$5-10/month"
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ Setup complete!${NC}"
echo ""
echo "For detailed instructions, see: DEPLOYMENT.md"
echo ""
