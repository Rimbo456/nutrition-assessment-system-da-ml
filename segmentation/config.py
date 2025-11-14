"""
Training configuration for FoodSeg103 Semantic Segmentation
OPTIMIZED FOR GPU TRAINING - Best settings for production
"""

# Paths
DATA_ROOT = r"D:\Dev\University\KLTN\project\data\foodseg103"
TRAIN_MANIFEST = f"{DATA_ROOT}/manifest_train.csv"
VAL_MANIFEST = f"{DATA_ROOT}/manifest_val.csv"
TEST_MANIFEST = f"{DATA_ROOT}/manifest_test.csv"
CHECKPOINT_DIR = r"D:\Dev\University\KLTN\project\checkpoints"
LOG_DIR = r"D:\Dev\University\KLTN\project\logs"

# Model - Best architecture for semantic segmentation
MODEL_NAME = "DeepLabV3+"  # State-of-the-art for segmentation
ENCODER = "resnet50"  # Good balance between speed and accuracy
ENCODER_WEIGHTS = "imagenet"  # Use pretrained weights

# Training - OPTIMAL for GPU
NUM_CLASSES = 104  # 0=background + 103 food classes
IMG_SIZE = (512, 512)  # Full resolution for best results
BATCH_SIZE = 16  # Large batch for GPU (adjust based on GPU memory)
NUM_EPOCHS = 100  # Enough for convergence
LEARNING_RATE = 3e-4  # Higher LR for faster convergence with large batch
WEIGHT_DECAY = 1e-4  # Regularization

# Optimizer & Scheduler
OPTIMIZER = "AdamW"  # Best optimizer for transformers/deep networks
SCHEDULER = "CosineAnnealingWarmRestarts"  # Best scheduler for long training

# Loss function
LOSS = "CrossEntropy"  # Standard for semantic segmentation

# Hardware - GPU settings (will auto-detect)
DEVICE = "cuda"  # Will fallback to CPU if CUDA unavailable
NUM_WORKERS = 4  # Parallel data loading (set to 0 on Windows if issues)
PIN_MEMORY = True  # Faster data transfer to GPU

# Logging
SAVE_EVERY = 5  # Save checkpoint every 5 epochs
LOG_EVERY = 50  # Log every 50 batches
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs

# Seed
SEED = 42

# ============================================================================
# NOTES FOR CPU TRAINING:
# ============================================================================
# 1. Training will be MUCH slower on CPU (10-100x slower than GPU)
# 2. One epoch might take 30-60 minutes on CPU vs 1-2 minutes on GPU
# 3. This config is for TESTING the pipeline, not for actual training
# 4. For production training, use GPU or cloud services (Google Colab, AWS, etc.)
# ============================================================================
