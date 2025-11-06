"""
Training script for FoodSeg103 Semantic Segmentation
"""

import os
import sys
import csv
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import config
from dataset import FoodSeg103Dataset


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(num_classes, encoder, encoder_weights):
    """Create segmentation model"""
    if config.MODEL_NAME == "U-Net":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    elif config.MODEL_NAME == "DeepLabV3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    elif config.MODEL_NAME == "FPN":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {config.MODEL_NAME}")
    
    return model


def get_loss_fn(class_weights=None):
    """Create loss function"""
    if config.LOSS == "CrossEntropy":
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    elif config.LOSS == "Focal":
        return smp.losses.FocalLoss(mode='multiclass', ignore_index=255)
    elif config.LOSS == "Dice":
        return smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    else:
        raise ValueError(f"Unknown loss: {config.LOSS}")


def get_metrics():
    """Create metrics for evaluation"""
    return {
        'iou': smp.utils.metrics.IoU(ignore_index=255),
        'fscore': smp.utils.metrics.Fscore(ignore_index=255),
        'accuracy': smp.utils.metrics.Accuracy(ignore_index=255)
    }


def train_epoch(model, dataloader, criterion, optimizer, metrics, device):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0.0
    metric_values = {name: 0.0 for name in metrics.keys()}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            for name, metric_fn in metrics.items():
                metric_values[name] += metric_fn(preds, masks).item()
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_EVERY == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Average metrics
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for name in metric_values:
        metric_values[name] /= num_batches
    
    return epoch_loss, metric_values


def validate_epoch(model, dataloader, criterion, metrics, device):
    """Validate for one epoch"""
    model.eval()
    
    epoch_loss = 0.0
    metric_values = {name: 0.0 for name in metrics.keys()}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            for name, metric_fn in metrics.items():
                metric_values[name] += metric_fn(preds, masks).item()
    
    # Average metrics
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for name in metric_values:
        metric_values[name] /= num_batches
    
    return epoch_loss, metric_values


def save_checkpoint(model, optimizer, epoch, best_metric, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    # Set seed
    set_seed(config.SEED)
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FoodSeg103Dataset(config.TRAIN_MANIFEST)
    val_dataset = FoodSeg103Dataset(config.VAL_MANIFEST)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Get class weights for handling class imbalance
    class_weights = train_dataset.get_class_weights()
    print(f"Computed class weights for {len(class_weights)} classes")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create model
    print(f"Creating model: {config.MODEL_NAME} with encoder {config.ENCODER}")
    model = get_model(config.NUM_CLASSES, config.ENCODER, config.ENCODER_WEIGHTS)
    model = model.to(device)
    
    # Loss function
    criterion = get_loss_fn(class_weights if config.LOSS == "CrossEntropy" else None)
    
    # Optimizer
    if config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    # Scheduler
    if config.SCHEDULER == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif config.SCHEDULER == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, verbose=True)
    elif config.SCHEDULER == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    else:
        scheduler = None
    
    # Metrics
    metrics = get_metrics()
    
    # Training loop
    best_iou = 0.0
    patience_counter = 0
    
    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"training_{timestamp}.csv")
    
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'train_fscore', 'train_accuracy',
                        'val_loss', 'val_iou', 'val_fscore', 'val_accuracy', 'lr'])
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("=" * 80)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, metrics, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, metrics, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, "
              f"F-score: {train_metrics['fscore']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"F-score: {val_metrics['fscore']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Log to CSV
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_metrics['iou'], train_metrics['fscore'], train_metrics['accuracy'],
                           val_loss, val_metrics['iou'], val_metrics['fscore'], val_metrics['accuracy'], current_lr])
        
        # Update scheduler
        if scheduler is not None:
            if config.SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(val_metrics['iou'])
            else:
                scheduler.step()
        
        # Save checkpoint
        if epoch % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics['iou'], checkpoint_path)
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_iou, best_checkpoint_path)
            print(f"New best IoU: {best_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best validation IoU: {best_iou:.4f}")
    print(f"Logs saved to: {log_file}")
    print(f"Best model saved to: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()
