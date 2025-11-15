"""
Check detailed PyTorch and CUDA compatibility.
"""
import torch
import sys

print("="*70)
print("🔍 Detailed PyTorch & CUDA Diagnostics")
print("="*70)

# PyTorch version
print(f"\n📦 PyTorch:")
print(f"   Version: {torch.__version__}")
print(f"   Install path: {torch.__file__}")

# CUDA info
print(f"\n🎮 CUDA:")
print(f"   Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   PyTorch compiled with CUDA: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      - Compute Capability: {props.major}.{props.minor}")
        print(f"      - Total Memory: {props.total_memory / 1024**3:.2f} GB")

# Test basic operations
print(f"\n🧪 Testing Operations:")

try:
    # CPU test
    x = torch.randn(10, 10, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    print("   ✅ CPU backward: OK")
except Exception as e:
    print(f"   ❌ CPU backward: FAILED - {e}")
    sys.exit(1)

if torch.cuda.is_available():
    try:
        # CUDA test
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        print("   ✅ CUDA backward: OK")
    except Exception as e:
        print(f"   ❌ CUDA backward: FAILED - {e}")
        print("\n🔥 CUDA operations are broken!")
    
    try:
        # Test AMP
        from torch.amp import autocast, GradScaler
        
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        scaler = GradScaler('cuda')
        
        with autocast('cuda'):
            y = (x ** 2).sum()
        
        scaler.scale(y).backward()
        print("   ✅ AMP (new API): OK")
    except Exception as e:
        print(f"   ⚠️  AMP (new API): {e}")
        
        # Try old API
        try:
            from torch.cuda.amp import autocast, GradScaler
            
            x = torch.randn(10, 10, device='cuda', requires_grad=True)
            scaler = GradScaler()
            
            with autocast():
                y = (x ** 2).sum()
            
            scaler.scale(y).backward()
            print("   ✅ AMP (old API): OK")
        except Exception as e:
            print(f"   ❌ AMP (old API): FAILED - {e}")
            print("\n🔥 Mixed precision training is broken!")

# Check torchvision
print(f"\n📦 TorchVision:")
try:
    import torchvision
    print(f"   Version: {torchvision.__version__}")
except:
    print("   ❌ Not installed")

# Recommendations
print("\n" + "="*70)
print("💊 Diagnosis & Fix:")
print("="*70)

if not torch.cuda.is_available():
    print("""
❌ CUDA not available but you have GPU!

Fix:
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
""")
else:
    cuda_version = torch.version.cuda
    if cuda_version:
        major = int(cuda_version.split('.')[0])
        if major >= 12:
            print(f"""
⚠️  You have CUDA {cuda_version} but might have compatibility issues.

Try reinstalling with matching version:
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
""")
        elif major == 11:
            print(f"""
✅ CUDA {cuda_version} detected.

If still getting errors, try:
    pip uninstall torch torchvision -y
    pip cache purge
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    
(Using specific older version that's more stable)
""")
    
    # Check for GET engine error specifically
    try:
        x = torch.randn(100, 100, device='cuda', requires_grad=True)
        loss = (x ** 2).sum()
        loss.backward()
        print("\n✅ GET engine test: PASSED")
    except RuntimeError as e:
        if "GET was unable to find an engine" in str(e):
            print(f"""
🔥 GET ENGINE ERROR CONFIRMED!

This is usually caused by:
1. PyTorch CUDA version mismatch
2. Corrupted PyTorch installation
3. Missing CUDA libraries

RECOMMENDED FIX:
    # Complete clean reinstall
    pip uninstall torch torchvision torchaudio -y
    pip cache purge
    
    # Install stable version with CUDA 11.8
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    
    # Verify
    python -c "import torch; x = torch.randn(10, 10, device='cuda', requires_grad=True); (x**2).sum().backward(); print('OK')"
""")
        else:
            print(f"\n⚠️  Backward test failed: {e}")

print("="*70)
