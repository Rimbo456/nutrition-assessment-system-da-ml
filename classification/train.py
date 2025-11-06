"""
Training script for Food-101N Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import config
from dataset_food101n import create_dataloaders


def main():
    print("=" * 80)
    print("TRAINING Food-101N Classification Model")
    print("=" * 80)

    # Device
    device = config.DEVICE
    print(f"Using device: {device}")

    # DataLoaders
    train_loader, val_loader, class_map = create_dataloaders(
        data_dir=config.DATA_ROOT,
        train_json=config.TRAIN_JSON,
        val_json=config.VAL_JSON,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE,
        use_verified_only=config.USE_VERIFIED_ONLY
    )

    # Model
    print("\nCreating model...")
    model = models.resnet50(pretrained=config.PRETRAINED)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
            min_lr=config.SCHEDULER_MIN_LR
        )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and device.type == 'cuda' else None

    best_acc = 0.0
    start_epoch = 0
    num_epochs = config.NUM_EPOCHS

    print("\nStart training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f"{running_loss/total:.4f}",
                'acc': f"{correct/total:.4f}"
            })

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, scaler)
        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Scheduler step
        if scheduler:
            scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = config.CHECKPOINT_DIR / f"best_model.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config.get_config_dict()
            }, save_path)
            print(f"Saved best model to {save_path} (Val Acc={best_acc:.4f})")

    print("\nTraining completed!")
    print(f"Best Val Acc: {best_acc:.4f}")


def evaluate(model, val_loader, criterion, device, scaler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


if __name__ == "__main__":
    main()
