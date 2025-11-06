"""
Configuration cho Food-101N Classification Model

Cấu trúc thư mục:
    - Data: KLTN/project/data/food-101N/
    - Scripts: KLTN/project/classification/
    - Checkpoints: KLTN/project/classification/checkpoints/
    - Logs: KLTN/project/classification/logs/
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent  # KLTN/project/classification/
DATA_ROOT = PROJECT_ROOT.parent / 'data' / 'food-101N'  # KLTN/project/data/food-101N/

# Data files
TRAIN_JSON = 'train_clean.json'  # hoặc 'train_all.json' nếu muốn train với noisy labels
VAL_JSON = 'val_clean.json'      # hoặc 'val_all.json'
CLASS_MAP_JSON = 'class_map.json'

# Output paths
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
LOG_DIR = PROJECT_ROOT / 'logs'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Tạo directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model architecture
MODEL_NAME = "resnet50"  # resnet50, resnet101, efficientnet_b3, vit_b_16
NUM_CLASSES = 101  # Food-101N có 101 classes
PRETRAINED = True  # Sử dụng ImageNet pretrained weights

# Image settings
IMG_SIZE = (512, 512)  # (height, width) - Resize tất cả ảnh về kích thước này
IMG_CHANNELS = 3  # RGB


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Training settings
BATCH_SIZE = 16  # Giảm xuống 8 nếu GPU memory không đủ
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Optimizer
OPTIMIZER = "AdamW"  # Options: "Adam", "AdamW", "SGD"
MOMENTUM = 0.9  # Chỉ dùng khi OPTIMIZER = "SGD"

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "CosineAnnealing", "StepLR"
SCHEDULER_PATIENCE = 5  # For ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5  # For ReduceLROnPlateau
SCHEDULER_MIN_LR = 1e-7  # Minimum learning rate

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # Stop nếu không improve sau N epochs

# Loss function
CRITERION = "CrossEntropyLoss"  # Options: "CrossEntropyLoss", "LabelSmoothingLoss", "FocalLoss"
LABEL_SMOOTHING = 0.1  # Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)

# Gradient clipping
USE_GRADIENT_CLIPPING = False
MAX_GRADIENT_NORM = 1.0


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset settings
USE_VERIFIED_ONLY = True  # Chỉ dùng verified samples (recommended)
NUM_WORKERS = 4  # Số workers cho DataLoader (tăng lên 8 nếu CPU mạnh)
PIN_MEMORY = True  # Pin memory cho GPU
PREFETCH_FACTOR = 2  # Prefetch batches

# Data augmentation (được define trong dataset.py)
# Training augmentation sẽ bao gồm:
#   - Resize to 512x512
#   - Horizontal flip (50%)
#   - Shift/Scale/Rotate
#   - Random brightness/contrast
#   - Color jitter (hue/saturation)
#   - Gaussian noise
#   - Normalize (ImageNet mean/std)

# Normalization (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# ============================================================================
# DEVICE & MIXED PRECISION
# ============================================================================

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True  # Automatic Mixed Precision (float16) - Tăng tốc training


# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================

# Logging
LOG_INTERVAL = 50  # Log mỗi N batches
TENSORBOARD_LOG = True  # Sử dụng TensorBoard

# Checkpoint
SAVE_INTERVAL = 1  # Save checkpoint mỗi N epochs
SAVE_BEST_ONLY = True  # Chỉ lưu model tốt nhất
METRIC_FOR_BEST = "accuracy"  # Metric để chọn best model: "accuracy", "loss", "top5_accuracy"

# Resume training
RESUME_FROM = None  # Path to checkpoint để resume training (None = train from scratch)


# ============================================================================
# VALIDATION & METRICS
# ============================================================================

# Metrics to track
METRICS = [
    "accuracy",       # Top-1 accuracy
    "top5_accuracy",  # Top-5 accuracy
    "loss",           # Cross-entropy loss
]

# Validation
VAL_INTERVAL = 1  # Validate mỗi N epochs


# ============================================================================
# EVALUATION & INFERENCE
# ============================================================================

# Inference
INFERENCE_BATCH_SIZE = 32  # Batch size lớn hơn cho inference (không cần backward)
SAVE_PREDICTIONS = True  # Lưu predictions vào file


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config():
    """In ra toàn bộ config"""
    print("=" * 80)
    print("CONFIGURATION - Food-101N Classification")
    print("=" * 80)
    
    print("\n📂 PATHS:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Logs: {LOG_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    print("\n📊 DATA:")
    print(f"  Train JSON: {TRAIN_JSON}")
    print(f"  Val JSON: {VAL_JSON}")
    print(f"  Use verified only: {USE_VERIFIED_ONLY}")
    print(f"  Image size: {IMG_SIZE}")
    
    print("\n🏗️  MODEL:")
    print(f"  Architecture: {MODEL_NAME}")
    print(f"  Num classes: {NUM_CLASSES}")
    print(f"  Pretrained: {PRETRAINED}")
    
    print("\n⚙️  TRAINING:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Num epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Criterion: {CRITERION}")
    if CRITERION == "LabelSmoothingLoss":
        print(f"  Label smoothing: {LABEL_SMOOTHING}")
    
    print("\n📈 SCHEDULER:")
    print(f"  Use scheduler: {USE_SCHEDULER}")
    if USE_SCHEDULER:
        print(f"  Type: {SCHEDULER_TYPE}")
        if SCHEDULER_TYPE == "ReduceLROnPlateau":
            print(f"  Patience: {SCHEDULER_PATIENCE}")
            print(f"  Factor: {SCHEDULER_FACTOR}")
    
    print("\n🛑 EARLY STOPPING:")
    print(f"  Enabled: {USE_EARLY_STOPPING}")
    if USE_EARLY_STOPPING:
        print(f"  Patience: {EARLY_STOPPING_PATIENCE} epochs")
    
    print("\n💻 DEVICE:")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed precision (AMP): {USE_AMP}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n📊 METRICS:")
    print(f"  Tracking: {', '.join(METRICS)}")
    print(f"  Best model metric: {METRIC_FOR_BEST}")
    
    print("\n🔧 DATALOADER:")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Pin memory: {PIN_MEMORY}")
    
    print("\n" + "=" * 80)


def get_config_dict():
    """Trả về config dưới dạng dictionary (để lưu vào checkpoint)"""
    return {
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
        'pretrained': PRETRAINED,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'optimizer': OPTIMIZER,
        'criterion': CRITERION,
        'label_smoothing': LABEL_SMOOTHING,
        'use_scheduler': USE_SCHEDULER,
        'scheduler_type': SCHEDULER_TYPE,
        'use_early_stopping': USE_EARLY_STOPPING,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'use_amp': USE_AMP,
        'use_verified_only': USE_VERIFIED_ONLY,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_config()
    
    print("\n💡 Usage:")
    print("  import config")
    print("  print(config.MODEL_NAME)")
    print("  print(config.BATCH_SIZE)")
