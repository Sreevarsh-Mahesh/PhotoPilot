# PhotoPilot Deployment Guide

This guide covers multiple deployment options for the PhotoPilot camera settings prediction app.

## Model File Note

The trained model (`checkpoints/best_model.pth`) is **205MB**, which exceeds GitHub's 100MB file size limit. You'll need to use **Git LFS** (Large File Storage) or alternative hosting for the model file.

## Option 1: Streamlit Cloud (Recommended - FREE)

Streamlit Cloud is the easiest way to deploy Streamlit apps for free.

### Prerequisites
1. GitHub account
2. Git LFS installed (for large model file)

### Setup Git LFS

```bash
# Install Git LFS (macOS)
brew install git-lfs

# Initialize Git LFS in your repo
cd /Users/sreevarshmaheshgandhi/Desktop/PHOTOPILOT/PhotoPilot
git lfs install

# Track the model file
cd fivek-cnn
git lfs track "checkpoints/*.pth"

# Add and commit
git add .gitattributes
git add checkpoints/best_model.pth
git commit -m "Add model file with Git LFS"
git push
```

### Deploy to Streamlit Cloud

1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Sign in with GitHub**

4. **Click "New app"**

5. **Configure deployment:**
   - Repository: `Sreevarsh-Mahesh/PhotoPilot`
   - Branch: `main`
   - Main file path: `fivek-cnn/app.py`

6. **Click "Deploy!"**

The app will be live at: `https://[your-username]-photopilot.streamlit.app`

### Troubleshooting Streamlit Cloud

- **If model file is too large:** Consider using external storage (see Option 4)
- **Memory issues:** Streamlit Cloud free tier has 1GB RAM limit
- **Slow loading:** Model is loaded on first request (cold start)

## Option 2: Hugging Face Spaces (FREE with GPU)

Hugging Face Spaces offers free hosting with optional GPU support.

### Setup

1. **Create account at [huggingface.co](https://huggingface.co)**

2. **Create a new Space:**
   - SDK: Streamlit
   - Hardware: CPU (free) or GPU (paid)

3. **Clone the Space locally:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/photopilot
   ```

4. **Copy your files:**
   ```bash
   cp -r fivek-cnn/* photopilot/
   ```

5. **Push to Hugging Face:**
   ```bash
   cd photopilot
   git add .
   git commit -m "Initial deployment"
   git push
   ```

The app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/photopilot`

## Option 3: Docker Deployment

For AWS, Azure, GCP, or self-hosting.

### Dockerfile

A `Dockerfile` has been created for you. Build and run:

```bash
# Build the image
docker build -t photopilot .

# Run locally
docker run -p 8501:8501 photopilot

# Push to Docker Hub
docker tag photopilot YOUR_USERNAME/photopilot
docker push YOUR_USERNAME/photopilot
```

### Deploy to Cloud Platforms

**AWS ECS/Fargate:**
```bash
aws ecr create-repository --repository-name photopilot
docker tag photopilot:latest AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/photopilot
docker push AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/photopilot
```

**Google Cloud Run:**
```bash
gcloud run deploy photopilot --image gcr.io/PROJECT_ID/photopilot --platform managed
```

**Azure Container Instances:**
```bash
az container create --resource-group myResourceGroup --name photopilot \
  --image YOUR_USERNAME/photopilot --dns-name-label photopilot --ports 8501
```

## Option 4: Model Hosting Alternatives

If the model file is too large for Git/deployment platform:

### A. Google Drive
```python
# In app.py, add model download function
import gdown

def download_model():
    url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
    output = 'checkpoints/best_model.pth'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
```

### B. AWS S3
```python
import boto3

def download_model_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket', 'best_model.pth', 'checkpoints/best_model.pth')
```

### C. Hugging Face Hub
```python
from huggingface_hub import hf_hub_download

def download_model():
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/photopilot-model",
        filename="best_model.pth",
        cache_dir="checkpoints"
    )
```

## Option 5: Railway.app (Simple, Paid)

Railway offers simple deployment with automatic builds.

1. **Go to [railway.app](https://railway.app)**
2. **Connect GitHub repository**
3. **Railway will auto-detect Streamlit**
4. **Deploy automatically**

Cost: ~$5-10/month depending on usage

## Performance Optimization

### For Production Deployment:

1. **Reduce model size:**
   ```python
   # Use model quantization
   import torch.quantization
   model_quantized = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **Add model caching:**
   ```python
   @st.cache_resource
   def load_model():
       # Model loading code
   ```

3. **Use CPU-optimized PyTorch:**
   ```txt
   torch==2.0.0+cpu
   torchvision==0.15.0+cpu
   ```

## Environment Variables

For production, set these environment variables:

```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
```

## Monitoring

Add basic monitoring to your app:

```python
import streamlit as st

# Track usage
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1
```

## Security Considerations

1. **Rate limiting:** Consider adding rate limiting for API abuse
2. **File validation:** Already implemented in app.py
3. **HTTPS:** Most platforms provide this automatically
4. **API keys:** Store in secrets (Streamlit uses `.streamlit/secrets.toml`)

## Cost Estimates

- **Streamlit Cloud:** FREE (with limits)
- **Hugging Face Spaces:** FREE (CPU) or $0.60/hour (GPU)
- **AWS ECS:** ~$10-30/month
- **Google Cloud Run:** Pay per request, ~$5-15/month
- **Railway:** ~$5-10/month

## Recommended Approach

**For this project, I recommend:**

1. **Start with Streamlit Cloud** (free, easy)
2. **Use Git LFS** for the model file
3. **If model too large:** Upload to Google Drive and download on startup
4. **If need GPU:** Use Hugging Face Spaces

## Next Steps

1. Choose your deployment platform
2. Set up Git LFS if needed
3. Push to GitHub
4. Deploy!

Need help with any specific platform? Let me know!
