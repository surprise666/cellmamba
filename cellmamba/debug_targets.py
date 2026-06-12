"""Debug script to verify target generation and model outputs"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
from scipy.io import loadmat

# Import from actual module
from utils.fcos_target import compute_fcos_targets

def parse_consep_label(label_path):
    data = loadmat(label_path)
    inst_map = data['inst_map']
    type_map = data['type_map']
    
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
            'type': 0
        })
    
    return centers


def main():
    # Find a label file
    label_dir = "/mnt/d/Code/Dataset/hover_net/CoNSeP/original/Train/Labels"
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.mat')]
    
    if not label_files:
        print(f"No .mat files found in {label_dir}")
        return
    
    # Use first label file
    label_path = os.path.join(label_dir, label_files[0])
    print(f"Analyzing: {label_files[0]}")
    
    # Parse labels
    centers = parse_consep_label(label_path)
    
    print(f"\nImage size: 256x256 (patch)")
    print(f"Number of GT cells: {len(centers)}")
    
    # Analyze GT box sizes
    box_sizes = []
    for c in centers:
        x1, y1, x2, y2 = c['bbox']
        size = max(x2 - x1, y2 - y1)
        box_sizes.append(size)
    
    if box_sizes:
        box_sizes = np.array(box_sizes)
        print(f"GT box sizes - min: {box_sizes.min():.1f}, max: {box_sizes.max():.1f}, mean: {box_sizes.mean():.1f}")
        print(f"Box size distribution:")
        for threshold in [8, 16, 24, 32, 48, 64, 128]:
            count = np.sum(box_sizes <= threshold)
            print(f"  <= {threshold}: {count} ({100*count/len(box_sizes):.1f}%)")
    
    # Simulate patch-based (256x256)
    patch_size = 256
    strides = [4, 8, 16, 32, 64]
    
    print("\n" + "="*60)
    print("Testing with PATCH SIZE 256x256 using imported compute_fcos_targets")
    print("="*60)
    
    # Use imported function
    obj_targets, reg_targets = compute_fcos_targets(
        image_size=(patch_size, patch_size), 
        centers=centers, 
        strides=strides,
        device='cpu'
    )
    
    # Count total positive locations
    total_pos = sum((obj.sum().item() for obj in obj_targets))
    print(f"\nTotal positive locations across all levels: {total_pos}")
    
    # Check each level
    for i, obj in enumerate(obj_targets):
        pos_count = obj.sum().item()
        print(f"Level {i} (stride {strides[i]}): pos={int(pos_count)}")
    
    # Check regression targets
    print("\nRegression target statistics:")
    for i, reg in enumerate(reg_targets):
        pos_mask = obj_targets[i] > 0
        if pos_mask.sum() > 0:
            reg_vals = reg[pos_mask.expand_as(reg)].reshape(4, -1)
            print(f"Level {i} (stride {strides[i]}):")
            print(f"  L: min={reg_vals[0].min():.2f}, max={reg_vals[0].max():.2f}")
            print(f"  T: min={reg_vals[1].min():.2f}, max={reg_vals[1].max():.2f}")
            print(f"  R: min={reg_vals[2].min():.2f}, max={reg_vals[2].max():.2f}")
            print(f"  B: min={reg_vals[3].min():.2f}, max={reg_vals[3].max():.2f}")


if __name__ == '__main__':
    main()
