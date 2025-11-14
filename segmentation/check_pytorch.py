"""
Check PyTorch and CUDA setup
"""

import torch
import sys

print("=" * 80)
print("PyTorch Installation Check")
print("=" * 80)

print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ Python version: {sys.version}")

print(f"\n🖥️  CUDA availability:")
print(f"   - CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   - CUDA version: {torch.version.cuda}")
    print(f"   - Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     • Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"     • Compute Capability: {props.major}.{props.minor}")
else:
    print(f"   ⚠️  CUDA is NOT available")
    print(f"   → PyTorch is running in CPU-only mode")
    print(f"\n   To install PyTorch with CUDA support:")
    print(f"   1. Uninstall current PyTorch:")
    print(f"      pip uninstall torch torchvision")
    print(f"   2. Install CUDA-enabled PyTorch:")
    print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print(f"      (Replace cu118 with your CUDA version: cu117, cu121, etc.)")

print(f"\n💻 CPU Info:")
print(f"   - Number of CPU threads: {torch.get_num_threads()}")

# Test tensor creation
print(f"\n🧪 Quick Test:")
try:
    cpu_tensor = torch.randn(3, 3)
    print(f"   ✓ CPU tensor creation: OK")
    
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(3, 3).cuda()
        print(f"   ✓ GPU tensor creation: OK")
    else:
        print(f"   - GPU tensor creation: Skipped (CUDA not available)")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)

# Recommendation
if torch.cuda.is_available():
    print("✅ RECOMMENDATION: Your system is ready for GPU training!")
else:
    print("⚠️  RECOMMENDATION:")
    print("   - Training on CPU will be VERY SLOW for deep learning")
    print("   - Consider:")
    print("     1. Installing PyTorch with CUDA support (if you have NVIDIA GPU)")
    print("     2. Using Google Colab (free GPU)")
    print("     3. Reducing BATCH_SIZE to 2-4 if training on CPU")
    print("     4. Reducing NUM_EPOCHS to test the pipeline first")

print("=" * 80)
