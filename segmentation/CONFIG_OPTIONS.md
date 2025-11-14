# Quick Reference: Config Options

## GPU Training (RECOMMENDED) - config.py
```python
MODEL_NAME = "DeepLabV3+"
ENCODER = "resnet50"
IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
SCHEDULER = "CosineAnnealingWarmRestarts"
NUM_WORKERS = 4
```

## CPU Training (Testing only) - config_cpu.py
```python
MODEL_NAME = "U-Net"
ENCODER = "resnet18"
IMG_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
SCHEDULER = "StepLR"
NUM_WORKERS = 0
```

## Model Options
- U-Net: Fast, good for simple segmentation
- DeepLabV3+: Best for complex scenes (recommended)
- FPN: Good balance

## Encoder Options
- resnet18: Fastest, lowest accuracy
- resnet34: Fast, good accuracy
- resnet50: Good balance (recommended)
- resnet101: Slower, higher accuracy
- efficientnet-b0: Efficient
- efficientnet-b3: Very good accuracy

## Scheduler Options
- CosineAnnealingWarmRestarts: Best for long training (recommended)
- ReduceLROnPlateau: Adaptive based on validation
- CosineAnnealing: Simple cosine schedule
- StepLR: Step decay

## Loss Options
- CrossEntropy: Standard (recommended)
- Focal: For class imbalance
- Dice: For small objects
