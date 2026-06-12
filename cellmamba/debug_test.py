"""Debug test inference to see actual model outputs"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
from scipy.io import loadmat

from models.cellmamba import build_model
from configs.config import config


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
    
    # Load a test image
    test_dir = "/mnt/d/Code/Dataset/hover_net/CoNSeP/original/Test"
    img_dir = os.path.join(test_dir, "Images")
    label_dir = os.path.join(test_dir, "Labels")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if not img_files:
        print(f"No images found in {img_dir}")
        return
    
    img_name = img_files[0]
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace('.png', '.mat'))
    
    print(f"\nTesting on: {img_name}")
    
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    print(f"Image size: {img_h}x{img_w}")
    
    # Load GT
    gt_centers = parse_consep_label(label_path)
    print(f"GT cells: {len(gt_centers)}")
    
    # Analyze GT boxes
    gt_sizes = []
    gt_areas = []
    for c in gt_centers:
        x1, y1, x2, y2 = c['bbox']
        size = max(x2 - x1, y2 - y1)
        area = (x2 - x1) * (y2 - y1)
        gt_sizes.append(size)
        gt_areas.append(area)
    
    gt_sizes = np.array(gt_sizes)
    gt_areas = np.array(gt_areas)
    print(f"GT size: min={gt_sizes.min():.0f}, max={gt_sizes.max():.0f}, mean={gt_sizes.mean():.1f}")
    print(f"GT area: min={gt_areas.min():.0f}, max={gt_areas.max():.0f}, mean={gt_areas.mean():.1f}")
    
    # Preprocess - ensure float32
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    img_tensor = np.ascontiguousarray(img_tensor)
    
    # Extract a 256x256 patch from center
    patch_size = 256
    center_y, center_x = img_h // 2, img_w // 2
    y_start = max(0, center_y - patch_size // 2)
    x_start = max(0, center_x - patch_size // 2)
    
    patch = img_tensor[:, y_start:y_start+patch_size, x_start:x_start+patch_size]
    
    if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
        pad_h = patch_size - patch.shape[1]
        pad_w = patch_size - patch.shape[2]
        patch = np.pad(patch, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
    
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
    
    # Get GT boxes in this patch
    patch_gt_boxes = []
    for c in gt_centers:
        cx, cy = c['center_x'], c['center_y']
        x1, y1, x2, y2 = c['bbox']
        
        # Check if box center is in patch
        if (x_start <= cx < x_start + patch_size and 
            y_start <= cy < y_start + patch_size):
            # Convert to patch coordinates
            patch_gt_boxes.append([
                x1 - x_start, y1 - y_start, 
                x2 - x_start, y2 - y_start
            ])
    
    print(f"\nGT boxes in patch: {len(patch_gt_boxes)}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(patch_tensor)
    
    # Analyze outputs
    print("\n" + "="*60)
    print("Model Output Analysis")
    print("="*60)
    
    for i, (obj, reg) in enumerate(zip(outputs['objectness'], outputs['regression'])):
        stride = outputs['strides'][i]
        obj_sigmoid = torch.sigmoid(obj)
        
        print(f"\nLevel {i} (stride {stride}):")
        print(f"  obj sigmoid: min={obj_sigmoid.min():.4f}, max={obj_sigmoid.max():.4f}, mean={obj_sigmoid.mean():.4f}")
        print(f"  reg (raw): min={reg.min():.4f}, max={reg.max():.4f}")
        
        # Check high confidence predictions
        high_conf_mask = obj_sigmoid > 0.3
        n_high = high_conf_mask.sum().item()
        print(f"  High confidence (>0.3): {n_high}")
        
        if n_high > 0:
            # Get positions and values
            y_idx, x_idx = torch.where(high_conf_mask[0, 0])
            confs = obj_sigmoid[0, 0][high_conf_mask[0, 0]]
            regs = reg[0, :, high_conf_mask[0, 0]]
            
            # Decode to pixel coordinates
            for j in range(min(5, n_high)):
                cx_pixel = x_idx[j].item() * stride
                cy_pixel = y_idx[j].item() * stride
                l = torch.nn.functional.softplus(regs[0, j]).item()
                t = torch.nn.functional.softplus(regs[1, j]).item()
                r = torch.nn.functional.softplus(regs[2, j]).item()
                b = torch.nn.functional.softplus(regs[3, j]).item()
                
                x1_pred = cx_pixel - l
                y1_pred = cy_pixel - t
                x2_pred = cx_pixel + r
                y2_pred = cy_pixel + b
                
                box_w = x2_pred - x1_pred
                box_h = y2_pred - y1_pred
                
                print(f"    [{j}] conf={confs[j]:.3f}, pos=({cx_pixel:.0f},{cy_pixel:.0f}), "
                      f"LTRB=({l:.1f},{t:.1f},{r:.1f},{b:.1f}), box=({x1_pred:.0f},{y1_pred:.0f},{x2_pred:.0f},{y2_pred:.0f}), size=({box_w:.0f}x{box_h:.0f})")
    
    # Show GT boxes in patch
    print("\n" + "="*60)
    print("GT Boxes in Patch")
    print("="*60)
    for i, box in enumerate(patch_gt_boxes[:5]):
        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)
        print(f"  [{i}] box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), size={size:.0f}")
    
    # Compare decoded predictions with GT
    print("\n" + "="*60)
    print("Matching Analysis")
    print("="*60)
    
    # Get all predictions
    all_preds = []
    for i, (obj, reg) in enumerate(zip(outputs['objectness'], outputs['regression'])):
        stride = outputs['strides'][i]
        obj_sigmoid = torch.sigmoid(obj)
        
        for b in range(obj.shape[0]):
            for gy in range(obj.shape[2]):
                for gx in range(obj.shape[3]):
                    conf = obj_sigmoid[b, 0, gy, gx].item()
                    if conf > 0.1:  # Lower threshold for analysis
                        cx_pixel = gx * stride
                        cy_pixel = gy * stride
                        
                        l = torch.nn.functional.softplus(reg[b, 0, gy, gx]).item()
                        t = torch.nn.functional.softplus(reg[b, 1, gy, gx]).item()
                        r = torch.nn.functional.softplus(reg[b, 2, gy, gx]).item()
                        bb = torch.nn.functional.softplus(reg[b, 3, gy, gx]).item()
                        
                        x1 = cx_pixel - l
                        y1 = cy_pixel - t
                        x2 = cx_pixel + r
                        y2 = cy_pixel + bb
                        
                        all_preds.append({
                            'conf': conf,
                            'box': [x1, y1, x2, y2],
                            'cx': cx_pixel, 'cy': cy_pixel,
                            'size': max(x2-x1, y2-y1)
                        })
    
    # Sort by confidence
    all_preds.sort(key=lambda x: -x['conf'])
    print(f"Total predictions (>0.1 conf): {len(all_preds)}")
    print(f"High confidence (>0.3 conf): {sum(1 for p in all_preds if p['conf'] > 0.3)}")
    
    # Show top predictions
    print("\nTop 10 predictions:")
    for i, pred in enumerate(all_preds[:10]):
        box = pred['box']
        print(f"  [{i}] conf={pred['conf']:.3f}, pos=({pred['cx']:.0f},{pred['cy']:.0f}), "
              f"box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}), size={pred['size']:.0f}")
    
    # Check how many GT boxes are covered by predictions
    print("\n" + "="*60)
    print("Coverage Analysis")
    print("="*60)
    
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        return inter_area / (union_area + 1e-6)
    
    matched_gt = set()
    for pred in all_preds:
        if pred['conf'] > 0.3:
            for gt_idx, gt_box in enumerate(patch_gt_boxes):
                if gt_idx not in matched_gt:
                    iou = compute_iou(pred['box'], gt_box)
                    if iou >= 0.5:
                        matched_gt.add(gt_idx)
                        break
    
    print(f"GT boxes in patch: {len(patch_gt_boxes)}")
    print(f"Matched (IoU>=0.5): {len(matched_gt)}")
    print(f"Coverage: {100*len(matched_gt)/len(patch_gt_boxes):.1f}%" if patch_gt_boxes else "N/A")


if __name__ == '__main__':
    main()
