"""
Training configuration for FoodSeg103 Semantic Segmentation
OPTIMIZED FOR 6GB GPU (RTX 3050, GTX 1060 6GB, etc.)
"""

import os

# Paths - relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "foodseg103")
TRAIN_MANIFEST = os.path.join(DATA_ROOT, "manifest_train.csv")
VAL_MANIFEST = os.path.join(DATA_ROOT, "manifest_val.csv")
TEST_MANIFEST = os.path.join(DATA_ROOT, "manifest_test.csv")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Model - Balanced for 6GB VRAM
MODEL_NAME = "DeepLabV3+"  # State-of-the-art for segmentation
ENCODER = "resnet34"  # Lighter than resnet50, fits in 6GB
ENCODER_WEIGHTS = "imagenet"  # Use pretrained weights

# Training - OPTIMIZED FOR 6GB GPU
NUM_CLASSES = 104  # 0=background + 103 food classes
IMG_SIZE = (512, 512)  # Full resolution
BATCH_SIZE = 8  # Reduced from 16 to fit in 6GB VRAM
NUM_EPOCHS = 100  # Enough for convergence
LEARNING_RATE = 2e-4  # Slightly lower LR for smaller batch
WEIGHT_DECAY = 1e-4  # Regularization

# Optimizer & Scheduler
OPTIMIZER = "AdamW"  # Best optimizer for deep networks
SCHEDULER = "CosineAnnealingWarmRestarts"  # Best scheduler for long training

# Loss function
LOSS = "CrossEntropy"  # Standard for semantic segmentation

# Hardware - GPU settings
DEVICE = "cuda"  # Will fallback to CPU if CUDA unavailable
NUM_WORKERS = 2  # Reduced for 6GB GPU to save system RAM
PIN_MEMORY = True  # Faster data transfer to GPU

# Logging
SAVE_EVERY = 5  # Save checkpoint every 5 epochs
LOG_EVERY = 50  # Log every 50 batches
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs

# Seed
SEED = 42

# ============================================================================
# Memory Usage Estimate for this config:
# Model (DeepLabV3+ ResNet34): ~350 MB
# Batch (8 x 512x512 RGB images): ~24 MB
# Batch (8 x 512x512 masks): ~8 MB
# Gradients + Optimizer state: ~1.5 GB
# Mixed precision overhead: ~500 MB
# TOTAL: ~2.4 GB (leaves ~3.6 GB buffer for PyTorch's memory allocator)
# ============================================================================
