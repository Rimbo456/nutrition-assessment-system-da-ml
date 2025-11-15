"""
Training configuration for FoodSeg103 semantic segmentation
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

# Model
MODEL_NAME = "DeepLabV3+"  # Options: "U-Net", "DeepLabV3+", "FPN"
ENCODER = "resnet50"  # Options: resnet18, resnet34, resnet50, resnet101, efficientnet-b0, etc.
ENCODER_WEIGHTS = "imagenet"  # pretrained weights

# Training
NUM_CLASSES = 104  # 0=background + 103 food classes
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Optimizer & Scheduler
OPTIMIZER = "AdamW"  # Options: Adam, AdamW, SGD
SCHEDULER = "ReduceLROnPlateau"  # Options: ReduceLROnPlateau, CosineAnnealing, StepLR

# Loss function
LOSS = "CrossEntropy"  # Options: CrossEntropy, Focal, Dice

# Hardware
DEVICE = "cuda"  # cuda or cpu (will auto-detect in train.py)
NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing issues
PIN_MEMORY = False  # Set to False when using CPU

# Logging
SAVE_EVERY = 5  # save checkpoint every N epochs
LOG_EVERY = 10  # log metrics every N batches
EARLY_STOPPING_PATIENCE = 10  # stop if no improvement for N epochs

# Seed for reproducibility
SEED = 42
