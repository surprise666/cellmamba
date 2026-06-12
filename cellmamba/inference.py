"""
Inference script for CellMamba-MVP
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from configs.config import config
from models.cellmamba_mvp import build_model
from utils.dataset import CoNSePDataset


def decode_predictions(objectness_list, regression_list, strides, conf_thresh=0.3):
    """Decode predictions to boxes"""
    all_boxes = []
    all_scores = []
    
    for level_idx, (obj, reg, stride) in enumerate(zip(objectness_list, regression_list, strides)):
        B, _, H, W = obj.shape
        
        # Process each image in the batch
        for b in range(B):
            obj_b = obj[b]  # (1, H, W)
            reg_b = reg[b]  # (4, H, W)
            
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=obj.device),
                torch.arange(W, device=obj.device),
                indexing='ij'
            )
            
            # 解码 LTRB 距离 (用 softplus 确保非负)
            # 模型预测的是从网格点到box边界的距离，没有中心偏移
            l = F.softplus(reg_b[0]) * stride
            t = F.softplus(reg_b[1]) * stride
            r = F.softplus(reg_b[2]) * stride
            b = F.softplus(reg_b[3]) * stride
            
            # 中心在网格点 (gx*stride, gy*stride)
            cx = x_grid.float() * stride
            cy = y_grid.float() * stride
            
            # 解码为角点坐标
            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b
            
            # Stack to (H, W, 4)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            scores = obj_b[0]  # (H, W)
            
            # Flatten for this level
            boxes_flat = boxes.reshape(-1, 4)  # (H*W, 4)
            scores_flat = scores.reshape(-1)  # (H*W,)
            
            all_boxes.append(boxes_flat.cpu().numpy())
            all_scores.append(scores_flat.cpu().numpy())
    
    if len(all_boxes) > 0 and len(all_boxes[0]) > 0:
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
    else:
        boxes = np.zeros((0, 4))
        scores = np.zeros((0,))
    
    return boxes, scores


def run_inference(image_path, model, device, strides, conf_thresh=0.3):
    """Run inference on a single image"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    # Resize to model input size
    image_resized = cv2.resize(image, (config.PATCH_SIZE, config.PATCH_SIZE))
    
    # Normalize
    image_norm = image_resized.astype(np.float32) / 255.0
    image_norm = (image_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Decode predictions
    pred_boxes, pred_scores = decode_predictions(
        predictions['objectness'],
        predictions['regression'],
        strides,
        conf_thresh
    )
    
    # Scale boxes back to original image size
    scale_x = original_w / config.PATCH_SIZE
    scale_y = original_h / config.PATCH_SIZE
    
    pred_boxes[:, [0, 2]] *= scale_x
    pred_boxes[:, [1, 3]] *= scale_y
    
    return image, pred_boxes, pred_scores


def visualize_and_save(image, boxes, scores, output_path, conf_thresh=0.3):
    """Visualize and save results"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.imshow(image)
    
    for box, score in zip(boxes, scores):
        if score < conf_thresh:
            continue
        
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 5), f'{score:.2f}', color='yellow', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    ax.set_title(f'Detected {len(boxes)} cells')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved result to {output_path}")


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found!")
    
    strides = config.STRIDES
    
    # Test on a sample from dataset
    print("\nTesting on dataset samples...")
    dataset = CoNSePDataset(
        image_dir=config.IMAGE_DIR,
        label_dir=config.LABEL_DIR,
        patch_size=config.PATCH_SIZE
    )
    
    for idx in range(3):
        sample = dataset[idx]
        image = sample['image']
        image_name = sample['image_name']
        centers = sample['centers']
        
        # Get ground truth boxes
        gt_boxes = np.array([c['bbox'] for c in centers])
        
        # Inference
        image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
        model.eval()
        with torch.no_grad():
            predictions = model(image_tensor)
        
        pred_boxes, pred_scores = decode_predictions(
            predictions['objectness'],
            predictions['regression'],
            strides,
            conf_thresh=0.3
        )
        
        print(f"\nImage: {image_name}")
        print(f"Ground truth: {len(gt_boxes)} cells")
        print(f"Predicted: {len(pred_boxes)} cells")
        
        # Visualize
        img_vis = image.transpose(1, 2, 0)
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = (img_vis * 255).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img_vis)
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='green', linewidth=2)
            axes[0].add_patch(rect)
        axes[0].set_title(f'Ground Truth ({len(gt_boxes)} cells)')
        axes[0].axis('off')
        
        axes[1].imshow(img_vis)
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, max(0, y1 - 5), f'{score:.2f}', color='yellow', fontsize=8)
        axes[1].set_title(f'Predictions ({len(pred_boxes)} cells)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'inference_result_{idx}_{image_name}.png')
        plt.close()
        print(f"Saved result to inference_result_{idx}_{image_name}.png")


if __name__ == '__main__':
    main()
