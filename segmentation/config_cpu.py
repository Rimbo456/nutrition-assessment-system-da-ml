"""
Training configuration for CPU (reduced settings for testing)
Copy this to config.py if you want to train on CPU
"""

# Paths
DATA_ROOT = r"D:\Dev\University\KLTN\project\data\foodseg103"
TRAIN_MANIFEST = f"{DATA_ROOT}/manifest_train.csv"
VAL_MANIFEST = f"{DATA_ROOT}/manifest_val.csv"
TEST_MANIFEST = f"{DATA_ROOT}/manifest_test.csv"
CHECKPOINT_DIR = r"D:\Dev\University\KLTN\project\checkpoints"
LOG_DIR = r"D:\Dev\University\KLTN\project\logs"

# Model - Use lighter architecture for CPU
MODEL_NAME = "U-Net"  # U-Net is faster than DeepLabV3+
ENCODER = "resnet18"  # resnet18 is much faster than resnet50
ENCODER_WEIGHTS = "imagenet"

# Training - REDUCED for CPU
NUM_CLASSES = 104
IMG_SIZE = (256, 256)  # Smaller image size (instead of 512x512)
BATCH_SIZE = 2  # Small batch size for CPU
NUM_EPOCHS = 5  # Just for testing, increase to 20-30 for real training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Optimizer & Scheduler
OPTIMIZER = "Adam"  # Adam is simpler than AdamW
SCHEDULER = "StepLR"  # Simpler scheduler

# Loss function
LOSS = "CrossEntropy"

# Hardware - CPU settings
DEVICE = "cpu"
NUM_WORKERS = 0  # Must be 0 on Windows
PIN_MEMORY = False  # Don't use pin_memory on CPU

# Logging
SAVE_EVERY = 2  # Save more frequently for testing
LOG_EVERY = 5  # Log more frequently
EARLY_STOPPING_PATIENCE = 5  # Stop earlier to save time

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
