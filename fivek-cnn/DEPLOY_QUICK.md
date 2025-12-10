# 🚀 Quick Deploy

Choose your deployment method and follow the steps below.

## 🎯 Fastest Option: Streamlit Cloud (FREE)

**Steps:**
1. Install Git LFS (for the 205MB model file):
   ```bash
   brew install git-lfs
   git lfs install
   git lfs track "checkpoints/*.pth"
   ```

2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Deploy PhotoPilot"
   git push origin main
   ```

3. Deploy:
   - Go to **[share.streamlit.io](https://share.streamlit.io)**
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repo: `Sreevarsh-Mahesh/PhotoPilot`
   - Branch: `main`
   - Main file: `fivek-cnn/app.py`
   - Click **"Deploy"**

Your app will be live at: `https://your-username-photopilot.streamlit.app`

---

## 🤗 Alternative: Hugging Face Spaces (FREE + GPU)

**Steps:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space → SDK: Streamlit
3. Clone and push:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/photopilot
   cd photopilot
   # Copy your files here
   git add .
   git commit -m "Deploy PhotoPilot"
   git push
   ```

---

## 🐳 Docker Deployment

**Local Testing:**
```bash
docker build -t photopilot .
docker run -p 8501:8501 photopilot
```

**Or use Docker Compose:**
```bash
docker-compose up
```

**Deploy to Cloud:**
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**

See `DEPLOYMENT.md` for detailed cloud instructions.

---

## 🛤️ Railway (Simple Paid Option)

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Deploy automatically
4. Cost: ~$5-10/month

---

## ⚡ Quick Setup Script

Run the interactive setup script:
```bash
cd fivek-cnn
./deploy.sh
```

This will guide you through:
- Git LFS setup
- Model file handling
- Platform-specific configuration
- Deployment steps

---

## 📦 What's Included

- ✅ `Dockerfile` - Containerized deployment
- ✅ `docker-compose.yml` - Local development
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `packages.txt` - System dependencies
- ✅ `.gitattributes` - Git LFS configuration
- ✅ `deploy.sh` - Interactive setup script
- ✅ `DEPLOYMENT.md` - Comprehensive guide

---

## ⚠️ Important Notes

### Model File (205MB)
The trained model exceeds GitHub's 100MB limit. Options:

1. **Git LFS** (Recommended)
   ```bash
   brew install git-lfs
   git lfs install
   git lfs track "checkpoints/*.pth"
   ```

2. **External Storage**
   - Google Drive
   - AWS S3
   - Hugging Face Hub
   
   See `DEPLOYMENT.md` for code examples.

### Memory Requirements
- Minimum: 1GB RAM
- Recommended: 2GB+ RAM
- Free tiers may have limitations

---

## 🔧 Configuration

### Environment Variables
```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
```

### Streamlit Secrets
For external model storage, create `.streamlit/secrets.toml`:
```toml
[model]
url = "https://your-storage.com/model.pth"
```

---

## 📊 Platform Comparison

| Platform | Cost | GPU | Easy Setup | Best For |
|----------|------|-----|------------|----------|
| Streamlit Cloud | FREE | ❌ | ⭐⭐⭐⭐⭐ | Quick demos |
| HF Spaces | FREE | ✅ | ⭐⭐⭐⭐ | ML projects |
| Railway | $5-10/mo | ❌ | ⭐⭐⭐⭐ | Simple apps |
| Docker + Cloud | Varies | ✅ | ⭐⭐⭐ | Production |

---

## 🆘 Troubleshooting

**Model file too large:**
```bash
# Use Git LFS
git lfs install
git lfs track "checkpoints/*.pth"
```

**Out of memory:**
- Use CPU-optimized PyTorch
- Enable model quantization
- Upgrade to paid tier

**Slow startup:**
- Normal for first request (cold start)
- Consider model caching
- Use smaller model or quantization

---

## 📚 Full Documentation

See **`DEPLOYMENT.md`** for:
- Detailed platform guides
- Model hosting alternatives
- Performance optimization
- Security best practices
- Cost estimates
- Monitoring setup

---

## 🎉 Ready to Deploy?

Run the setup script:
```bash
./deploy.sh
```

Or follow the steps for your chosen platform above!

Need help? Check `DEPLOYMENT.md` or open an issue on GitHub.
