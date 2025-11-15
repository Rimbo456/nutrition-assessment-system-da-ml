# Quick Start Guide - Fix Installation Issues

## ⚠️ Current Error: "GET was unable to find an engine"

This error means PyTorch is **corrupted or wrong version installed**.

## 🔧 Quick Fix (Copy-Paste These Commands)

### Step 1: Open PowerShell in project directory
```powershell
cd D:\Temp\project\segmentation
.\.venv\Scripts\Activate.ps1
```

### Step 2: Completely remove PyTorch
```powershell
pip uninstall torch torchvision torchaudio -y
```

### Step 3: Check if you have GPU
```powershell
nvidia-smi
```

**If nvidia-smi works (shows GPU info):**
```powershell
# Install CUDA 11.8 version (most compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**If nvidia-smi doesn't work (no GPU):**
```powershell
# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install other dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Verify installation
```powershell
python check_installation.py
```

**Expected output (GPU):**
```
✅ PyTorch imported successfully
   Version: 2.x.x+cu118
✅ CUDA available: True
✅ CPU tensor operations: OK
✅ CPU backward pass: OK
✅ CUDA tensor operations: OK
```

**Expected output (CPU):**
```
✅ PyTorch imported successfully
   Version: 2.x.x+cpu
⚠️  CUDA available: False
✅ CPU tensor operations: OK
✅ CPU backward pass: OK
```

### Step 6: Regenerate manifests (first time only)
```powershell
python regenerate_manifests.py
```

### Step 7: Run training
```powershell
python train.py
```

## 📊 What to Expect

### With GPU (CUDA)
```
Using device: cuda (GPU: NVIDIA GeForce RTX 3060)
✓ Mixed precision training enabled (faster on GPU)
Training: 100%|████████| 265/265 [02:30<00:00]  # ~2-3 minutes per epoch
```

### With CPU
```
Using device: cpu (CUDA not available)
⚠️  Training on CPU will be very slow.
⚠️  Mixed precision disabled (CPU mode)
Training: 100%|████████| 265/265 [35:00<00:00]  # ~30-40 minutes per epoch
```

## ❌ Still Getting Errors?

Run diagnostic:
```powershell
python check_installation.py
```

It will tell you exactly what's wrong and how to fix it.

## 🚀 Pro Tips

### For faster training on CPU (if no GPU):
Edit `config.py`:
```python
BATCH_SIZE = 4        # Reduce from 16
NUM_EPOCHS = 10       # Reduce from 100
IMG_SIZE = 256        # Reduce from 512
NUM_WORKERS = 0       # Disable multiprocessing on Windows
```

### Use Google Colab (Free GPU):
1. Upload project to Google Drive
2. Open Colab: https://colab.research.google.com
3. Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/your-project/segmentation
```
4. Install & run:
```python
!pip install -r requirements.txt
!python regenerate_manifests.py
!python train.py
```

## 📚 More Help

- **Detailed setup:** See `SETUP.md`
- **Full documentation:** See `README.md` (when available)
- **Check GPU:** Run `python check_installation.py`

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] PyTorch installed (correct version for your hardware)
- [ ] `check_installation.py` passes all tests
- [ ] Manifests regenerated with relative paths
- [ ] Training starts without errors

---

**Last Updated:** 2025-11-15
