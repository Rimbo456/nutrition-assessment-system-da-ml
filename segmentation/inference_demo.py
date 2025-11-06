"""
Demo script để hiểu Input/Output của model Semantic Segmentation
"""

import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

import config


def load_trained_model(checkpoint_path, device='cuda'):
    """Load model đã train từ checkpoint"""
    # Tạo model architecture
    if config.MODEL_NAME == "DeepLabV3+":
        model = smp.DeepLabV3Plus(
            encoder_name=config.ENCODER,
            encoder_weights=None,  # Không dùng pretrained, load từ checkpoint
            in_channels=3,
            classes=config.NUM_CLASSES
        )
    elif config.MODEL_NAME == "U-Net":
        model = smp.Unet(
            encoder_name=config.ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=config.NUM_CLASSES
        )
    elif config.MODEL_NAME == "FPN":
        model = smp.FPN(
            encoder_name=config.ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=config.NUM_CLASSES
        )
    
    # Load weights từ checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation IoU: {checkpoint['best_metric']:.4f}")
    
    return model


def predict_single_image(model, image_path, device='cuda'):
    """
    Dự đoán segmentation mask cho 1 ảnh
    
    INPUT:
        - image_path: Đường dẫn đến ảnh (JPG/PNG)
        - Ảnh có thể có kích thước bất kỳ (sẽ được resize về 512x512)
    
    OUTPUT:
        - mask: Numpy array shape (H, W) với giá trị từ 0-103
            + 0 = background (không phải đồ ăn)
            + 1-103 = các class đồ ăn khác nhau
        - confidence_map: Numpy array shape (H, W) - confidence score của prediction
    """
    # 1. ĐỌC ẢNH
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")
    
    # 2. PREPROCESSING (giống như trong training)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, config.IMG_SIZE)  # Resize về 512x512
    
    # Normalize về [0, 1] và chuyển sang tensor
    image_tensor = torch.from_numpy(image_resized).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    image_tensor = image_tensor.unsqueeze(0)  # Thêm batch dimension: (1, 3, 512, 512)
    image_tensor = image_tensor.to(device)
    
    print(f"Input tensor shape: {image_tensor.shape}")
    # INPUT: Tensor shape (1, 3, 512, 512)
    #        - 1: batch size
    #        - 3: RGB channels
    #        - 512x512: image size
    
    # 3. INFERENCE
    with torch.no_grad():
        output = model(image_tensor)
    
    print(f"Output tensor shape: {output.shape}")
    # OUTPUT: Tensor shape (1, 104, 512, 512)
    #         - 1: batch size
    #         - 104: số classes (0=background + 103 food classes)
    #         - 512x512: prediction cho mỗi pixel
    #         - Mỗi pixel có 104 giá trị (logits/scores) cho 104 classes
    
    # 4. POST-PROCESSING
    # Lấy class có score cao nhất cho mỗi pixel
    probabilities = torch.softmax(output, dim=1)  # Convert logits -> probabilities
    confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
    
    # Chuyển về numpy
    mask = predicted_classes[0].cpu().numpy()  # Shape: (512, 512)
    confidence_map = confidence_scores[0].cpu().numpy()  # Shape: (512, 512)
    
    print(f"Prediction mask shape: {mask.shape}")
    print(f"Unique classes in prediction: {np.unique(mask)}")
    print(f"Confidence range: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]")
    
    # 5. RESIZE VỀ KÍCH THƯỚC GỐC (nếu cần)
    if (original_height, original_width) != config.IMG_SIZE:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        confidence_map = cv2.resize(confidence_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    return mask, confidence_map


def visualize_prediction(image_path, mask, confidence_map, class_map_path, output_path):
    """
    Visualize kết quả prediction
    
    Tạo ảnh overlay với màu sắc cho từng class
    """
    import json
    
    # Đọc class names
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    
    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Tạo colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Tạo màu ngẫu nhiên cho mỗi class
    np.random.seed(42)
    colors = np.random.randint(0, 255, (104, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background = black
    
    for class_id in np.unique(mask):
        colored_mask[mask == class_id] = colors[class_id]
    
    # Overlay lên ảnh gốc
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    # Thêm text hiển thị classes được detect
    detected_classes = np.unique(mask)
    detected_classes = detected_classes[detected_classes > 0]  # Loại background
    
    text_img = overlay.copy()
    y_offset = 30
    for class_id in detected_classes[:10]:  # Hiển thị tối đa 10 classes
        class_name = class_map['by_id'][str(class_id)]
        pixel_count = np.sum(mask == class_id)
        percentage = (pixel_count / mask.size) * 100
        
        text = f"Class {class_id}: {class_name} ({percentage:.1f}%)"
        cv2.putText(text_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(text_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, tuple(colors[class_id].tolist()), 1, cv2.LINE_AA)
        y_offset += 25
    
    # Save
    result_bgr = cv2.cvtColor(text_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"Visualization saved to: {output_path}")


def predict_batch_images(model, image_paths, device='cuda', batch_size=8):
    """
    Dự đoán cho nhiều ảnh cùng lúc (batch processing)
    
    INPUT:
        - image_paths: List các đường dẫn ảnh
        
    OUTPUT:
        - masks: List các numpy arrays (H, W) - segmentation masks
    """
    all_masks = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        original_sizes = []
        
        # Load và preprocess batch
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_sizes.append(img.shape[:2])
            
            img_resized = cv2.resize(img, config.IMG_SIZE)
            img_tensor = torch.from_numpy(img_resized).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            batch_images.append(img_tensor)
        
        # Stack thành batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        print(f"Batch input shape: {batch_tensor.shape}")
        # INPUT: Tensor shape (N, 3, 512, 512) với N = batch_size
        
        # Inference
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        print(f"Batch output shape: {outputs.shape}")
        # OUTPUT: Tensor shape (N, 104, 512, 512)
        
        # Post-processing
        predictions = torch.argmax(outputs, dim=1)  # (N, 512, 512)
        predictions = predictions.cpu().numpy()
        
        # Resize về kích thước gốc
        for j, (mask, original_size) in enumerate(zip(predictions, original_sizes)):
            if original_size != (512, 512):
                mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
            all_masks.append(mask)
    
    return all_masks


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DEMO: Input/Output của Model Semantic Segmentation")
    print("=" * 80)
    
    # Paths
    checkpoint_path = r"D:\Dev\University\KLTN\project\checkpoints\best_model.pth"
    test_image = r"D:\Dev\University\KLTN\project\data\foodseg103\images\test\000001.jpg"
    class_map_path = r"D:\Dev\University\KLTN\project\data\foodseg103\class_map.json"
    output_vis = r"D:\Dev\University\KLTN\project\prediction_demo.jpg"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # 1. Load model
    print("\n[1] Loading trained model...")
    print("-" * 80)
    model = load_trained_model(checkpoint_path, device)
    
    # 2. Predict single image
    print("\n[2] Predicting single image...")
    print("-" * 80)
    mask, confidence_map = predict_single_image(model, test_image, device)
    
    # 3. Visualize
    print("\n[3] Creating visualization...")
    print("-" * 80)
    visualize_prediction(test_image, mask, confidence_map, class_map_path, output_vis)
    
    print("\n" + "=" * 80)
    print("SUMMARY - INPUT/OUTPUT CỦA MODEL:")
    print("=" * 80)
    print("\n📥 INPUT:")
    print("  - Tensor shape: (batch_size, 3, 512, 512)")
    print("  - batch_size: Số lượng ảnh (thường 1 khi inference, 8 khi training)")
    print("  - 3: RGB channels")
    print("  - 512x512: Kích thước ảnh sau resize")
    print("  - Giá trị: [0.0, 1.0] (normalized)")
    
    print("\n📤 OUTPUT:")
    print("  - Tensor shape: (batch_size, 104, 512, 512)")
    print("  - batch_size: Số lượng ảnh")
    print("  - 104: Số classes (0=background + 103 food classes)")
    print("  - 512x512: Prediction cho từng pixel")
    print("  - Giá trị: Logits (raw scores) cho mỗi class")
    
    print("\n🎯 PREDICTION MASK (sau post-processing):")
    print("  - Array shape: (H, W) với H, W là kích thước ảnh")
    print("  - Giá trị: 0-103 (class index)")
    print("    + 0 = background (không phải đồ ăn)")
    print("    + 1-103 = các loại đồ ăn khác nhau")
    print("  - Mỗi pixel trong ảnh được gán 1 class")
    
    print("\n💡 CÁCH SỬ DỤNG:")
    print("  1. Đọc ảnh bất kỳ")
    print("  2. Resize về 512x512")
    print("  3. Normalize về [0, 1]")
    print("  4. Chuyển thành tensor (1, 3, 512, 512)")
    print("  5. Pass vào model -> Output (1, 104, 512, 512)")
    print("  6. Lấy argmax theo dim=1 -> Mask (1, 512, 512)")
    print("  7. Resize mask về kích thước ảnh gốc (nếu cần)")
    print("  8. Visualize hoặc lưu mask")
    
    print("\n" + "=" * 80)
