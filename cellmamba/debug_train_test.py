"""Debug training targets and loss computation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat
import cv2

from models.cellmamba import build_model
from configs.config import config
from utils.fcos_target import compute_fcos_targets, get_model_strides
from utils.losses import CellMambaLoss


def parse_consep_label(label_path):
    data = loadmat(label_path)
    inst_map = data['inst_map']
    
    centers = []
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]
    
    for inst_id in inst_ids:
        mask = (inst_map == inst_id)
        ys, xs = np.where(mask)
        center_y, center_x = ys.mean(), xs.mean()
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        centers.append({
            'center_y': float(center_y),
            'center_x': float(center_x),
            'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],
        })
    
    return centers


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = build_model(config).to(device)
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    # Get ACTUAL strides from model
    actual_strides = get_model_strides(model, config.PATCH_SIZE)
    print(f"\nActual strides from model: {actual_strides}")
    print(f"Config strides: {config.STRIDES}")
    
    # Create loss function
    criterion = CellMambaLoss()
    
    # Load a training sample
    train_dir = "/mnt/d/Code/Dataset/hover_net/CoNSeP/original/Train"
    img_dir = os.path.join(train_dir, "Images")
    label_dir = os.path.join(train_dir, "Labels")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    img_name = img_files[0]
    
    # Load image
    img = cv2.imread(os.path.join(img_dir, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load GT
    label_path = os.path.join(label_dir, img_name.replace('.png', '.mat'))
    gt_centers = parse_consep_label(label_path)
    
    print(f"\nImage: {img_name}, Size: {img.shape[:2]}, GT cells: {len(gt_centers)}")
    
    # Extract 256x256 patch from center
    patch_size = 256
    img_h, img_w = img.shape[:2]
    y_start = max(0, (img_h - patch_size) // 2)
    x_start = max(0, (img_w - patch_size) // 2)
    
    # Filter GT centers in patch
    patch_centers = []
    for c in gt_centers:
        cx, cy = c['center_x'], c['center_y']
        x1, y1, x2, y2 = c['bbox']
        
        if (x_start <= cx < x_start + patch_size and 
            y_start <= cy < y_start + patch_size):
            patch_centers.append({
                'center_x': cx - x_start,
                'center_y': cy - y_start,
                'bbox': [x1 - x_start, y1 - y_start, x2 - x_start, y2 - y_start],
            })
    
    print(f"GT centers in patch: {len(patch_centers)}")
    
    # Analyze GT boxes in patch
    gt_sizes = []
    for c in patch_centers:
        x1, y1, x2, y2 = c['bbox']
        size = max(x2 - x1, y2 - y1)
        gt_sizes.append(size)
    
    print(f"GT box sizes in patch: min={min(gt_sizes)}, max={max(gt_sizes)}, mean={np.mean(gt_sizes):.1f}")
    
    # Preprocess
    patch = img[y_start:y_start+patch_size, x_start:x_start+patch_size]
    patch = patch.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    patch = (patch - mean) / std
    patch = np.transpose(patch, (2, 0, 1))
    patch = np.ascontiguousarray(patch)
    
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
    
    # Compute targets using ACTUAL strides
    targets = compute_fcos_targets(
        image_size=(patch_size, patch_size),
        centers=patch_centers,
        strides=actual_strides,  # Use ACTUAL strides!
        device=device
    )
    
    print("\n" + "="*60)
    print("Target Analysis (using ACTUAL strides)")
    print("="*60)
    
    for i, (obj_t, reg_t) in enumerate(zip(targets[0], targets[1])):
        stride = actual_strides[i]
        n_pos = (obj_t > 0).sum().item()
        print(f"\nLevel {i} (stride {stride}): feat={obj_t.shape[2]}x{obj_t.shape[3]}, pos={n_pos}")
        
        if n_pos > 0:
            pos_mask = obj_t[0, 0] > 0
            reg_at_pos = reg_t[0, :, pos_mask]
            
            print(f"  LTRB range: L=[{reg_at_pos[0].min():.2f}, {reg_at_pos[0].max():.2f}], "
                  f"T=[{reg_at_pos[1].min():.2f}, {reg_at_pos[1].max():.2f}], "
                  f"R=[{reg_at_pos[2].min():.2f}, {reg_at_pos[2].max():.2f}], "
                  f"B=[{reg_at_pos[3].min():.2f}, {reg_at_pos[3].max():.2f}]")
            
            # Pixel level
            print(f"  Pixel LTRB (x{stride}): L=[{reg_at_pos[0].min().item()*stride:.1f}, {reg_at_pos[0].max().item()*stride:.1f}], etc.")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(patch_tensor)
    
    print("\n" + "="*60)
    print("Model Output Analysis")
    print("="*60)
    
    for i, (obj_p, reg_p) in enumerate(zip(outputs['objectness'], outputs['regression'])):
        stride = actual_strides[i]
        obj_sigmoid = torch.sigmoid(obj_p)
        
        n_high = (obj_sigmoid > 0.3).sum().item()
        print(f"\nLevel {i} (stride {stride}): shape={obj_p.shape[2:4]}, high_conf={n_high}")
        
        # Get predictions at positive locations
        if targets[0][i].sum() > 0:
            pos_mask = targets[0][i][0, 0] > 0  # (H, W)
            
            # Raw predictions at positive locations
            raw_pred = reg_p[0, :, pos_mask]  # (4, N_pos)
            print(f"  Raw pred LTRB: L=[{raw_pred[0].min():.3f}, {raw_pred[0].max():.3f}], "
                  f"T=[{raw_pred[1].min():.3f}, {raw_pred[1].max():.3f}], "
                  f"R=[{raw_pred[2].min():.3f}, {raw_pred[2].max():.3f}], "
                  f"B=[{raw_pred[3].min():.3f}, {raw_pred[3].max():.3f}]")
            
            # Softplus predictions
            softplus_pred = F.softplus(raw_pred)
            print(f"  Softplus LTRB: L=[{softplus_pred[0].min():.3f}, {softplus_pred[0].max():.3f}], "
                  f"T=[{softplus_pred[1].min():.3f}, {softplus_pred[1].max():.3f}], "
                  f"R=[{softplus_pred[2].min():.3f}, {softplus_pred[2].max():.3f}], "
                  f"B=[{softplus_pred[3].min():.3f}, {softplus_pred[3].max():.3f}]")
            
            # GT targets
            gt_pred = targets[1][i][0, :, pos_mask]
            print(f"  GT target LTRB: L=[{gt_pred[0].min():.3f}, {gt_pred[0].max():.3f}], "
                  f"T=[{gt_pred[1].min():.3f}, {gt_pred[1].max():.3f}], "
                  f"R=[{gt_pred[2].min():.3f}, {gt_pred[2].max():.3f}], "
                  f"B=[{gt_pred[3].min():.3f}, {gt_pred[3].max():.3f}]")
    
    # Compute loss
    stacked_targets = {
        'objectness_targets': targets[0],
        'regression_targets': targets[1]
    }
    
    loss, loss_dict = criterion(outputs, stacked_targets)
    
    print("\n" + "="*60)
    print("Loss Analysis")
    print("="*60)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Obj loss: {loss_dict['obj_loss']:.4f}")
    print(f"Reg loss: {loss_dict['reg_loss']:.4f}")
    
    # Check ratio
    print("\n" + "="*60)
    print("Prediction vs GT Ratio")
    print("="*60)
    
    for i, reg_p in enumerate(outputs['regression']):
        if targets[0][i].sum() > 0:
            pos_mask = targets[0][i][0, 0] > 0
            raw_pred = reg_p[0, :, pos_mask]
            gt_pred = targets[1][i][0, :, pos_mask]
            
            ratio = (F.softplus(raw_pred).mean() / (gt_pred.mean() + 1e-6)).item()
            print(f"Level {i} (stride {actual_strides[i]}): softplus/GT ratio = {ratio:.3f} "
                  f"(pred_mean={F.softplus(raw_pred).mean():.2f}, gt_mean={gt_pred.mean():.2f})")
            
            if ratio < 0.3:
                print(f"  WARNING: Predictions much smaller than GT!")
            elif ratio > 3.0:
                print(f"  WARNING: Predictions much larger than GT!")


if __name__ == '__main__':
    main()
