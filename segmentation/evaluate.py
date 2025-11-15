"""
Evaluate trained model on test set and visualize predictions
"""

import os
import torch
import cv2
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from dataset import FoodSeg103Dataset, get_val_transform
from train import get_model, get_metrics


def evaluate_model(checkpoint_path, test_manifest, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_manifest: Path to test manifest CSV
        device: 'cuda' or 'cpu'
    """
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading test dataset...")
    test_dataset = FoodSeg103Dataset(
        test_manifest,
        data_root=config.DATA_ROOT,
        transform=get_val_transform(config.IMG_SIZE)
    )
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == 'cuda')
    )
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}")
    model = get_model(config.NUM_CLASSES, config.ENCODER, config.ENCODER_WEIGHTS)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"✓ Best validation IoU: {checkpoint['best_metric']:.4f}")
    
    # Get metrics
    metrics = get_metrics()
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    total_loss = 0.0
    metric_values = {name: 0.0 for name in metrics.keys()}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics
            preds = torch.argmax(outputs, dim=1)
            for name, metric_fn in metrics.items():
                result = metric_fn(preds, masks)
                if isinstance(result, torch.Tensor):
                    metric_values[name] += result.item()
                else:
                    metric_values[name] += result
    
    # Average metrics
    num_batches = len(test_loader)
    for name in metric_values:
        metric_values[name] /= num_batches
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"\n📊 Metrics:")
    print(f"   - IoU: {metric_values['iou']:.4f} ({metric_values['iou']*100:.2f}%)")
    print(f"   - F-score: {metric_values['fscore']:.4f} ({metric_values['fscore']*100:.2f}%)")
    print(f"   - Accuracy: {metric_values['accuracy']:.4f} ({metric_values['accuracy']*100:.2f}%)")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    iou = metric_values['iou']
    if iou >= 0.7:
        print(f"   ✅ EXCELLENT: Model is working very well!")
    elif iou >= 0.5:
        print(f"   ✓ GOOD: Model is performing well")
    elif iou >= 0.3:
        print(f"   ⚠️  FAIR: Model needs more training")
    else:
        print(f"   ❌ POOR: Model needs significant improvement")
    
    print("=" * 80)
    
    return metric_values


def visualize_predictions(checkpoint_path, test_manifest, output_dir, num_samples=10, device='cuda'):
    """
    Visualize model predictions on test samples
    """
    print(f"\n{'='*80}")
    print("VISUALIZING PREDICTIONS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset (without transform for visualization)
    test_dataset = FoodSeg103Dataset(test_manifest, data_root=config.DATA_ROOT, transform=None)
    
    # Load model
    model = get_model(config.NUM_CLASSES, config.ENCODER, config.ENCODER_WEIGHTS)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Transform for inference
    transform = get_val_transform(config.IMG_SIZE)
    
    # Load class map
    class_map_path = os.path.join(config.DATA_ROOT, 'class_map.json')
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    # Generate random colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, (config.NUM_CLASSES, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background = black
    
    print(f"Generating {num_samples} visualizations...")
    
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    for i, idx in enumerate(tqdm(indices)):
        # Get original image and ground truth
        sample = test_dataset.samples[idx]
        image = cv2.imread(sample['image'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(sample['mask'], cv2.IMREAD_UNCHANGED)
        
        # Predict
        transformed = transform(image=image_rgb, mask=gt_mask)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # Resize prediction to original size
        h, w = image.shape[:2]
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored masks
        gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(config.NUM_CLASSES):
            gt_colored[gt_mask == class_id] = colors[class_id]
            pred_colored[pred_mask == class_id] = colors[class_id]
        
        # Create overlays
        alpha = 0.5
        gt_overlay = cv2.addWeighted(image_rgb, 1-alpha, gt_colored, alpha, 0)
        pred_overlay = cv2.addWeighted(image_rgb, 1-alpha, pred_colored, alpha, 0)
        
        # Combine into one image
        top_row = np.hstack([image_rgb, gt_overlay])
        bottom_row = np.hstack([pred_overlay, pred_colored])
        combined = np.vstack([top_row, bottom_row])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Ground Truth', (w+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Prediction', (10, h+30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Pred Mask', (w+10, h+30), font, 1, (255, 255, 255), 2)
        
        # Save
        output_path = os.path.join(output_dir, f'test_sample_{i+1:03d}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    # Paths
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    test_manifest = config.TEST_MANIFEST
    viz_output_dir = os.path.join(config.CHECKPOINT_DIR, "visualizations")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Evaluate
    if os.path.exists(checkpoint_path):
        metrics = evaluate_model(checkpoint_path, test_manifest, device)
        
        # Visualize
        print(f"\nGenerating visualization samples...")
        visualize_predictions(checkpoint_path, test_manifest, viz_output_dir, num_samples=20, device=device)
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Please train the model first by running: python train.py")
