"""
Quick test for GET engine error.
Run this to verify PyTorch installation is working.
"""
import torch

print("Testing CUDA backward pass...")

try:
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        print("Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        exit(1)
    
    # Test that causes GET engine error if PyTorch is broken
    x = torch.randn(100, 100, device='cuda', requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward()
    
    print("✅ PASSED - PyTorch is working correctly!")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
except RuntimeError as e:
    if "GET was unable to find an engine" in str(e):
        print("❌ FAILED - GET engine error detected!")
        print("\nFix:")
        print("  pip uninstall torch torchvision -y")
        print("  pip cache purge")
        print("  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118")
        exit(1)
    else:
        print(f"❌ FAILED - {e}")
        exit(1)
