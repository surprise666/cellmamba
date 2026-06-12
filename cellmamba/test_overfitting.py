"""
Single Image Overfitting Test
Train on 1 fixed image for 500 iterations to verify the model works
"""

import os
import matplotlib
matplotlib.use('Agg')

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from configs.config import config
from models.cellmamba_mvp import build_model
from utils.dataset import parse_consep_label
from utils.fcos_target import compute_fcos_targets
from utils.losses import CellMambaLoss


class SingleImageDataset(Dataset):
    """Dataset that returns crops from a single large image"""
    def __init__(self, image_path, label_data, patch_size=128, fixed_pos=None):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.label_data = label_data
        self.patch_size = patch_size
        self.fixed_pos = fixed_pos
        self.centers = label_data['centers']
    
    def __len__(self):
        return 500
    
    def _get_random_crop(self, max_attempts=50):
        img_h, img_w = self.image.shape[:2]
        for _ in range(max_attempts):
            x = np.random.randint(0, max(1, img_w - self.patch_size)) if img_w > self.patch_size else 0
            y = np.random.randint(0, max(1, img_h - self.patch_size)) if img_h > self.patch_size else 0
            count = sum(1 for c in self.centers 
                       if x <= c['center_x'] <= x + self.patch_size 
                       and y <= c['center_y'] <= y + self.patch_size)
            if count >= 1:
                return x, y
        return max(0, (img_w - self.patch_size) // 2), max(0, (img_h - self.patch_size) // 2)
    
    def _filter_centers(self, x, y):
        filtered = []
        for c in self.centers:
            cx, cy = c['center_x'], c['center_y']
            
            # Use original bbox directly
            real_x1, real_y1, real_x2, real_y2 = c['bbox']
                
            if x <= cx <= x + self.patch_size and y <= cy <= y + self.patch_size:
                filtered.append({
                    'center_x': cx - x,
                    'center_y': cy - y,
                    'bbox': [real_x1 - x, real_y1 - y, real_x2 - x, real_y2 - y],
                    'type': 0
                })
        return filtered
    
    def __getitem__(self, idx):
        if self.fixed_pos is not None:
            x, y = self.fixed_pos
        else:
            x, y = self._get_random_crop()
        
        patch = self.image[y:y + self.patch_size, x:x + self.patch_size]
        centers = self._filter_centers(x, y)
        
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        patch = np.transpose(patch, (2, 0, 1))
        patch = np.ascontiguousarray(patch)
        
        return {
            'image': patch,
            'centers': centers,
            'image_name': f'crop_{x}_{y}',
            'image_size': [self.patch_size, self.patch_size]
        }


def decode_predictions(objectness_list, regression_list, strides, conf_thresh=0.3):
    """Decode predictions to boxes with NMS - collect all levels first, then NMS"""
    all_boxes = []
    all_scores = []
    
    for obj, reg, stride in zip(objectness_list, regression_list, strides):
        B, _, H, W = obj.shape
        for b in range(B):
            # Apply Sigmoid
            scores = torch.sigmoid(obj[b, 0])
            reg_b = reg[b]
            
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=obj.device),
                torch.arange(W, device=obj.device),
                indexing='ij'
            )
            
            # 解码 LTRB 距离 (用 softplus 确保非负，和 losses.py 保持一致)
            l = F.softplus(reg_b[0])  # left
            t = F.softplus(reg_b[1])  # top
            r = F.softplus(reg_b[2])  # right
            b = F.softplus(reg_b[3])  # bottom
            
            # 中心点 + 宽高格式
            cx = (x_grid.float() + 0) * stride  # 默认中心在网格点
            cy = (y_grid.float() + 0) * stride
            
            # 转回 LTRB 角点格式
            x1 = (x_grid.float() - l) * stride
            y1 = (y_grid.float() - t) * stride
            x2 = (x_grid.float() + r) * stride
            y2 = (y_grid.float() + b) * stride
            
            boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
            scores_flat = scores.reshape(-1)
            
            mask = scores_flat > conf_thresh
            if mask.sum() > 0:
                all_boxes.append(boxes[mask])
                all_scores.append(scores_flat[mask])
    
    if len(all_boxes) > 0:
        # Collect ALL boxes from ALL levels first
        boxes_tensor = torch.cat(all_boxes, dim=0)
        scores_tensor = torch.cat(all_scores, dim=0)
        
        # Filter by minimum box size (remove tiny boxes)
        box_w = boxes_tensor[:, 2] - boxes_tensor[:, 0]
        box_h = boxes_tensor[:, 3] - boxes_tensor[:, 1]
        # Use a reasonable minimum box size (10 pixels) to filter noise
        size_mask = (box_w > 10) & (box_h > 10)
        boxes_tensor = boxes_tensor[size_mask]
        scores_tensor = scores_tensor[size_mask]
        
        if len(boxes_tensor) > 0:
            # Apply NMS with moderate threshold (balance between keeping boxes and removing overlaps)
            keep_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            
            final_boxes = boxes_tensor[keep_idx].cpu().numpy()
            final_scores = scores_tensor[keep_idx].cpu().numpy()
            return final_boxes, final_scores
        
    return np.zeros((0, 4)), np.zeros((0,))


def visualize_predictions(image, gt_boxes, pred_boxes, pred_scores, save_path=None):
    """Visualize ground truth and predictions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_vis = image.copy()
    img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)
    
    axes[0].imshow(img_vis)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_vis)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='green', linewidth=2)
        axes[1].add_patch(rect)
    axes[1].set_title(f'Ground Truth ({len(gt_boxes)} cells)')
    axes[1].axis('off')
    
    axes[2].imshow(img_vis)
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(x1, y1, f'{score:.2f}', color='yellow', fontsize=8)
    axes[2].set_title(f'Predictions ({len(pred_boxes)} cells)')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    plt.close()


def compute_targets(centers_list, image_sizes, strides):
    """Compute targets for a batch"""
    num_levels = len(strides)
    level_obj_targets = [[] for _ in range(num_levels)]
    level_reg_targets = [[] for _ in range(num_levels)]
    
    for centers, image_size in zip(centers_list, image_sizes):
        obj_targets, reg_targets = compute_fcos_targets(
            image_size, centers, strides, center_radius=1.5
        )
        for level_idx in range(num_levels):
            level_obj_targets[level_idx].append(obj_targets[level_idx])
            level_reg_targets[level_idx].append(reg_targets[level_idx])
    
    obj_targets_batch = []
    reg_targets_batch = []
    
    for level_idx in range(num_levels):
        obj_batch = torch.stack(level_obj_targets[level_idx], dim=0)
        reg_batch = torch.stack(level_reg_targets[level_idx], dim=0)
        obj_targets_batch.append(obj_batch)
        reg_targets_batch.append(reg_batch)
    
    return obj_targets_batch, reg_targets_batch


def run_overfitting_test():
    """Run the overfitting test on a single image with cropping"""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    image_files = sorted([f for f in os.listdir(config.IMAGE_DIR) if f.lower().endswith('.png')])
    img_name = image_files[0]
    img_path = os.path.join(config.IMAGE_DIR, img_name)
    label_path = os.path.join(config.LABEL_DIR, os.path.splitext(img_name)[0] + '.mat')
    
    label_data = parse_consep_label(label_path)
    
    # Fixed position for consistent overfitting
    FIXED_X, FIXED_Y = 200, 200
    
    test_dataset = SingleImageDataset(
        image_path=img_path,
        label_data=label_data,
        patch_size=config.PATCH_SIZE,
        fixed_pos=(FIXED_X, FIXED_Y)
    )
    
    sample = test_dataset[0]
    image = sample['image']
    centers = sample['centers']
    image_size = sample['image_size']
    
    print(f"Crop size: {image_size}")
    print(f"Cells in this crop: {len(centers)} (Green boxes)")
    
    print("Building model...")
    config.NUM_CLASSES = 1
    model = build_model(config).to(device)
    
    criterion = CellMambaLoss(
        objectness_weight=config.OBJECTNESS_WEIGHT,
        regression_weight=config.REGESSION_WEIGHT
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    strides = config.STRIDES
    
    print(f"\nStarting overfitting test on Fixed Crop ({FIXED_X}, {FIXED_Y}) for 500 iterations...")
    losses = []
    
    model.train()
    for iteration in tqdm(range(500)):
        sample = test_dataset[0]
        image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).float().to(device)
        centers = sample['centers']
        
        obj_targets, reg_targets = compute_targets([centers], [image_size], strides)
        targets = {
            'objectness_targets': obj_targets,
            'regression_targets': reg_targets
        }
        
        predictions = model(image_tensor)
        predictions['strides'] = strides
        
        loss, loss_dict = criterion(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss.item():.4f}")
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    
    print("\nRunning inference...")
    model.eval()
    
    with torch.no_grad():
        image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).float().to(device)
        predictions = model(image_tensor)
    
    pred_boxes, pred_scores = decode_predictions(
        predictions['objectness'],
        predictions['regression'],
        strides,
        conf_thresh=0.5
    )
    
    print(f"Predicted {len(pred_boxes)} cells in fixed crop (Red boxes)")
    gt_boxes = np.array([c['bbox'] for c in sample['centers']])
    
    visualize_predictions(
        sample['image'].transpose(1, 2, 0),
        gt_boxes,
        pred_boxes,
        pred_scores,
        save_path='overfitting_result.png'
    )
    print("Saved visualization to overfitting_result.png")


if __name__ == '__main__':
    run_overfitting_test()
