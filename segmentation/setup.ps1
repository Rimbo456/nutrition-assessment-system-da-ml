# Automatic setup script for Windows (PowerShell)
# This script will check GPU and install correct PyTorch version

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "FoodSeg103 Segmentation Training - Auto Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[1/6] Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/"
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if not exists
if (-not (Test-Path ".venv")) {
    Write-Host ""
    Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[2/6] Virtual environment already exists, skipping..." -ForegroundColor Gray
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check GPU availability
Write-Host ""
Write-Host "[4/6] Checking GPU availability..." -ForegroundColor Yellow

$hasGPU = $false
try {
    $nvidiaOutput = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasGPU = $true
        Write-Host "GPU detected!" -ForegroundColor Green
        Write-Host $nvidiaOutput
    }
} catch {
    $hasGPU = $false
}

# Install PyTorch
Write-Host ""
Write-Host "[5/6] Installing PyTorch..." -ForegroundColor Yellow

if ($hasGPU) {
    Write-Host "Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Cyan
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "WARNING: No NVIDIA GPU detected!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  1. Install NVIDIA drivers (if you have GPU)"
    Write-Host "  2. Use Google Colab for free GPU"
    Write-Host "  3. Continue with CPU only (training will be VERY slow)"
    Write-Host ""
    
    $choice = Read-Host "Choose option (1/2/3)"
    
    switch ($choice) {
        "1" {
            Write-Host "Opening NVIDIA driver download page..."
            Start-Process "https://www.nvidia.com/Download/index.aspx"
            Write-Host "Please install drivers and run this script again."
            Read-Host "Press Enter to exit"
            exit 0
        }
        "2" {
            Write-Host "Opening Google Colab..."
            Start-Process "https://colab.research.google.com"
            Write-Host "Please use Google Colab for training."
            Read-Host "Press Enter to exit"
            exit 0
        }
        "3" {
            Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
            pip install torch torchvision
        }
        default {
            Write-Host "Invalid choice. Exiting." -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}

# Install other dependencies
Write-Host ""
Write-Host "[6/6] Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Verifying installation..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run GPU check:  python check_gpu.py"
Write-Host "  2. Prepare data:   python regenerate_manifests.py"
Write-Host "  3. Start training: python train.py"
Write-Host "  4. Evaluate model: python evaluate.py"
Write-Host ""
Write-Host "For detailed help, see: README.md or SETUP.md" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Read-Host "Press Enter to exit"
