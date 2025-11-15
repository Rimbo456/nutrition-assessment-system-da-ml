# Setup Instructions for GPU Training

## Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (GTX 1060 or better recommended)
- NVIDIA GPU drivers installed

## Installation Steps

### 1. Check NVIDIA GPU and CUDA availability

```powershell
# Check if you have NVIDIA GPU
nvidia-smi
```

If this command works, note your CUDA version (e.g., CUDA 11.8 or 12.1).

### 2. Create virtual environment

```powershell
cd D:\Temp\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA support

**For CUDA 11.8 (recommended - most compatible):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (NOT recommended for training):**
```powershell
pip install torch torchvision
```

### 4. Install other dependencies

```powershell
pip install -r requirements.txt
```

Note: This will skip torch/torchvision since you already installed them.

### 5. Verify CUDA is available

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output (GPU available):**
```
CUDA available: True
CUDA version: 11.8
Device count: 1
Device name: NVIDIA GeForce RTX 3060
```

**If you see (CUDA not available):**
```
CUDA available: False
CUDA version: None
Device count: 0
Device name: N/A
```

This means you need to reinstall PyTorch with CUDA support (see step 3).

### 6. Regenerate manifests (first time only)

```powershell
cd segmentation
python regenerate_manifests.py
```

### 7. Run training

```powershell
python train.py
```

## Troubleshooting

### Issue 1: "RuntimeError: GET was unable to find an engine to execute this computation"

**Root Cause:** PyTorch installation is corrupted or version mismatch.

**Solution:** Reinstall PyTorch completely:
```powershell
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Reinstall with CUDA 11.8 (if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR CPU-only (if no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python check_installation.py
```

### Issue 2: "CUDA not available" but I have NVIDIA GPU

**Solution:** You installed CPU-only PyTorch. Reinstall with CUDA:
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: "RuntimeError: CUDA out of memory"

**Solution 1:** Reduce batch size in `config.py`:
```python
BATCH_SIZE = 8  # or 4
```

**Solution 2:** Enable gradient accumulation in `config.py`:
```python
ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
```

### Issue 4: Training very slow on GPU

**Check if model is actually on GPU:**
```python
# In train.py, after model creation:
print(f"Model device: {next(model.parameters()).device}")
```

Should print: `Model device: cuda:0`

### Issue 5: "FileNotFoundError: [Errno 2] No such file or directory"

**Solution:** Regenerate manifests with relative paths:
```powershell
python regenerate_manifests.py
```

## Quick Diagnostic

**Run this before training to check your setup:**
```powershell
python check_installation.py
```

This will:
- ✅ Check PyTorch installation
- ✅ Test CUDA availability
- ✅ Test tensor operations
- ✅ Check all dependencies
- ✅ Provide specific fix commands if issues found

## Performance Expectations

### GPU Training (NVIDIA RTX 3060)
- Batch size: 16
- Time per epoch: ~2-3 minutes
- Total training time (100 epochs): ~4-5 hours
- Memory usage: ~8-10 GB VRAM

### CPU Training (NOT recommended)
- Batch size: 4 (max)
- Time per epoch: ~30-40 minutes
- Total training time (100 epochs): ~50-60 hours
- Memory usage: ~8-12 GB RAM

## Alternative: Google Colab (Free GPU)

If you don't have a local GPU, use Google Colab:

1. Upload project to Google Drive
2. Create new Colab notebook
3. Mount Drive and navigate to project:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/your-project-folder/segmentation
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

5. Run training:
```python
!python train.py
```

Colab provides free T4 GPU (16GB VRAM) with ~15 hours runtime limit.

## Recommended Setup

**For best performance:**
1. NVIDIA GPU with 8GB+ VRAM (RTX 3060, 3070, or better)
2. PyTorch with CUDA 11.8
3. Batch size 16-32
4. Mixed precision training (enabled by default in config)

**Minimal setup:**
1. NVIDIA GPU with 6GB+ VRAM (GTX 1060 6GB)
2. PyTorch with CUDA 11.8
3. Batch size 8
4. Mixed precision training

**If no GPU:**
- Use Google Colab (free)
- Or use CPU with very small config (batch=1, epochs=5, image_size=256)
