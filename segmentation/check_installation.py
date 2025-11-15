"""
Diagnostic script to check PyTorch installation and fix common issues.
"""
import sys
import subprocess


def check_torch_installation():
    """Check PyTorch installation details."""
    print("="*60)
    print("🔍 PyTorch Installation Diagnostics")
    print("="*60)
    
    # Try importing torch
    try:
        import torch
        print(f"\n✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
    except ImportError as e:
        print(f"\n❌ Failed to import PyTorch: {e}")
        print("\n💡 Fix: Install PyTorch")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # Check CUDA availability
    print(f"\n🎮 CUDA Status:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version (compiled): {torch.version.cuda if torch.version.cuda else 'N/A'}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version (runtime): {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   ⚠️  CUDA not available")
    
    # Check torchvision
    try:
        import torchvision
        print(f"\n📦 TorchVision:")
        print(f"   Version: {torchvision.__version__}")
    except ImportError:
        print(f"\n❌ TorchVision not installed")
        print("   pip install torchvision")
        return False
    
    # Test basic operations
    print(f"\n🧪 Testing basic operations...")
    
    try:
        # Test CPU tensor operations
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = x @ y
        print(f"   ✅ CPU tensor operations: OK")
    except Exception as e:
        print(f"   ❌ CPU tensor operations failed: {e}")
        return False
    
    # Test backward pass on CPU
    try:
        x = torch.randn(10, 10, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        print(f"   ✅ CPU backward pass: OK")
    except Exception as e:
        print(f"   ❌ CPU backward pass failed: {e}")
        print(f"\n🔥 This is the issue! PyTorch installation is corrupted.")
        return False
    
    # Test CUDA operations if available
    if torch.cuda.is_available():
        try:
            x = torch.randn(10, 10, device='cuda')
            y = torch.randn(10, 10, device='cuda')
            z = x @ y
            print(f"   ✅ CUDA tensor operations: OK")
        except Exception as e:
            print(f"   ⚠️  CUDA tensor operations failed: {e}")
    
    print(f"\n{'='*60}")
    return True


def check_dependencies():
    """Check other dependencies."""
    print("\n📦 Other Dependencies:")
    
    packages = [
        'numpy',
        'opencv-python',
        'albumentations',
        'segmentation-models-pytorch',
        'tqdm',
        'matplotlib',
        'scikit-learn'
    ]
    
    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def recommend_fix():
    """Recommend fix based on diagnosis."""
    print("\n" + "="*60)
    print("💊 Recommended Fix")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            # Check if nvidia-smi works
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    # Has GPU but PyTorch doesn't see it
                    print("""
🔧 You have NVIDIA GPU but PyTorch is CPU-only version.

Fix (PowerShell):
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Verify:
    python -c "import torch; print(torch.cuda.is_available())"
""")
                else:
                    # No GPU
                    print("""
⚠️  No NVIDIA GPU detected.

Options:
    1. Use Google Colab (free GPU): https://colab.research.google.com
    2. Continue with CPU (slow but works if no errors above)
    3. Get a machine with NVIDIA GPU

For CPU training, reduce config in config.py:
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    IMG_SIZE = 256
""")
            except:
                print("""
⚠️  Cannot detect GPU status.

For safe installation (works on any machine):
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
""")
        
        else:
            # Has CUDA
            print("""
✅ Your PyTorch installation looks good!

If you still get errors during training:
    1. Update GPU drivers: https://www.nvidia.com/Download/index.aspx
    2. Reinstall PyTorch:
       pip uninstall torch torchvision -y
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
""")
    
    except Exception as e:
        print(f"""
❌ PyTorch import failed: {e}

Fix:
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
""")
    
    print("="*60)


def main():
    # Check installation
    torch_ok = check_torch_installation()
    
    if not torch_ok:
        print("\n⚠️  PyTorch installation has issues!")
    
    # Check other dependencies
    deps_ok = check_dependencies()
    
    # Recommend fix
    recommend_fix()
    
    # Exit code
    if torch_ok and deps_ok:
        print("\n✅ All checks passed! You can run: python train.py")
        return 0
    else:
        print("\n❌ Some issues found. Please fix them before training.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
