"""
Dataset utilities for CoNSeP dataset - Paper Implementation
"""

import os
import numpy as np
from scipy.io import loadmat
import cv2
from torch.utils.data import Dataset


def get_train_val_split(image_dir, label_dir, val_ratio=0.2, seed=42):
    """Split dataset into train and validation sets by image files"""
    np.random.seed(seed)
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                        if f.lower().endswith('.png')])
    
    n_images = len(image_files)
    indices = np.arange(n_images)
    np.random.shuffle(indices)
    
    n_val = int(n_images * val_ratio)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    
    return train_indices, val_indices


def parse_consep_label(label_path):
    """
    Parse CoNSeP .mat label file
    Returns: dict with keys:
        - inst_map: instance segmentation map (H x W)
        - type_map: type/classification map (H x W)
        - centers: list of dicts for each nucleus
    """
    data = loadmat(label_path)
    
    inst_map = data['inst_map']
    type_map = data['type_map']
    
    centers = []
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]  # Remove background
    
    for inst_id in inst_ids:
        mask = (inst_map == inst_id)
        ys, xs = np.where(mask)
        center_y, center_x = ys.mean(), xs.mean()
        
        types_in_mask = type_map[mask]
        type_value = types_in_mask[0] if len(types_in_mask) > 0 else 1
        type_value = 0  # Force all to foreground
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        centers.append({
            'center_y': float(center_y),
            'center_x': float(center_x),
            'bbox': [float(x_min), float(y_min), float(x_max), float(y_max)],  # [x1, y1, x2, y2]
            'type': int(type_value)
        })
    
    return {
        'inst_map': inst_map,
        'type_map': type_map,
        'centers': centers
    }


def filter_centers_in_patch(centers, patch_x, patch_y, patch_size, margin=0):
    """Filter centers within patch bounds"""
    filtered = []
    for c in centers:
        cx, cy = c['center_x'], c['center_y']
        bbox = c['bbox']
        
        if (patch_x - margin <= cx <= patch_x + patch_size + margin and
            patch_y - margin <= cy <= patch_y + patch_size + margin):
            filtered.append({
                'center_x': cx - patch_x,
                'center_y': cy - patch_y,
                'bbox': [bbox[0] - patch_x, bbox[1] - patch_y,
                         bbox[2] - patch_x, bbox[3] - patch_y],
                'type': c['type']
            })
    return filtered


class CoNSePDataset(Dataset):
    """
    CoNSeP Dataset - Paper Implementation (256x256 patches)
    """
    def __init__(self, image_dir, label_dir, patch_size=256, 
                 crop_strategy='random', overlaps=64, transform=None,
                 min_cells_per_patch=1, num_samples_per_epoch=200,
                 image_indices=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.crop_strategy = crop_strategy
        self.overlaps = overlaps
        self.transform = transform
        self.min_cells_per_patch = min_cells_per_patch
        self.num_samples_per_epoch = num_samples_per_epoch
        
        all_image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.lower().endswith('.png')])
        
        if image_indices is not None:
            self.image_files = [all_image_files[i] for i in image_indices]
        else:
            self.image_files = all_image_files
        
        print(f"CoNSePDataset: {len(self.image_files)} images")
        
        # Pre-load labels
        self.labels_cache = {}
        for img_name in self.image_files:
            label_name = os.path.splitext(img_name)[0] + '.mat'
            label_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(label_path):
                self.labels_cache[img_name] = parse_consep_label(label_path)
        
        # Cache images
        self.images_cache = {}
        for img_name in self.image_files:
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            self.images_cache[img_name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def __len__(self):
        return self.num_samples_per_epoch
    
    def _get_random_crop(self, image, centers, img_h, img_w):
        """Get a random crop containing at least min_cells_per_patch"""
        max_attempts = 50
        for _ in range(max_attempts):
            if img_w > self.patch_size:
                x = np.random.randint(0, img_w - self.patch_size)
            else:
                x = 0
            if img_h > self.patch_size:
                y = np.random.randint(0, img_h - self.patch_size)
            else:
                y = 0
            
            filtered = filter_centers_in_patch(centers, x, y, self.patch_size)
            if len(filtered) >= self.min_cells_per_patch:
                return x, y
        
        x = max(0, (img_w - self.patch_size) // 2)
        y = max(0, (img_h - self.patch_size) // 2)
        return x, y
    
    def _extract_patch(self, image, x, y):
        """Extract a patch from image with padding if needed"""
        if y + self.patch_size > image.shape[0] or x + self.patch_size > image.shape[1]:
            # Need padding
            pad_h = max(0, self.patch_size - (image.shape[0] - y))
            pad_w = max(0, self.patch_size - (image.shape[1] - x))
            patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=image.dtype)
            h_end = min(self.patch_size, image.shape[0] - y)
            w_end = min(self.patch_size, image.shape[1] - x)
            patch[:h_end, :w_end] = image[y:y+h_end, x:x+w_end]
            return patch
        else:
            return image[y:y + self.patch_size, x:x + self.patch_size]
    
    def __getitem__(self, idx):
        img_name = np.random.choice(self.image_files)
        image = self.images_cache[img_name]
        
        img_h, img_w = image.shape[:2]
        
        label_data = self.labels_cache.get(img_name)
        centers = label_data['centers'] if label_data else []
        
        x, y = self._get_random_crop(image, centers, img_h, img_w)
        patch = self._extract_patch(image, x, y)
        filtered_centers = filter_centers_in_patch(centers, x, y, self.patch_size)
        
        # Normalize - ImageNet stats
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Convert to (C, H, W)
        patch = np.transpose(patch, (2, 0, 1))
        patch = np.ascontiguousarray(patch)
        
        return {
            'image': patch,
            'image_name': img_name,
            'centers': filtered_centers,
            'image_size': patch.shape[1:],
            'crop_pos': (x, y)
        }


class SlidingWindowDataset(Dataset):
    """
    Dataset with sliding window patches - Paper Implementation (256x256)
    """
    def __init__(self, image_dir, label_dir, patch_size=256, overlaps=64, image_indices=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.overlaps = overlaps
        
        all_image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.lower().endswith('.png')])
        
        if image_indices is not None:
            self.image_files = [all_image_files[i] for i in image_indices]
        else:
            self.image_files = all_image_files
        
        print(f"SlidingWindowDataset: {len(self.image_files)} images")
        
        # Pre-load labels
        self.labels_cache = {}
        for img_name in self.image_files:
            label_name = os.path.splitext(img_name)[0] + '.mat'
            label_path = os.path.join(self.label_dir, label_name)
            if os.path.exists(label_path):
                self.labels_cache[img_name] = parse_consep_label(label_path)
        
        # Cache images
        self.images_cache = {}
        for img_name in self.image_files:
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            self.images_cache[img_name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Build crop index
        self.crop_index = []
        for img_idx, img_name in enumerate(self.image_files):
            image = self.images_cache[img_name]
            img_h, img_w = image.shape[:2]
            
            stride = patch_size - overlaps
            y = 0
            while y < img_h:
                x = 0
                while x < img_w:
                    self.crop_index.append((img_idx, img_name, x, y))
                    x += stride
                y += stride
        
        print(f"SlidingWindowDataset: {len(self.crop_index)} total patches")
    
    def __len__(self):
        return len(self.crop_index)
    
    def __getitem__(self, idx):
        img_idx, img_name, x, y = self.crop_index[idx]
        
        image = self.images_cache[img_name]
        
        # Extract patch with padding
        patch = image[y:y + self.patch_size, x:x + self.patch_size]
        
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            pad_h = self.patch_size - patch.shape[0]
            pad_w = self.patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Filter centers
        label_data = self.labels_cache.get(img_name)
        centers = label_data['centers'] if label_data else []
        filtered_centers = filter_centers_in_patch(centers, x, y, self.patch_size)
        
        # Normalize
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        patch = np.transpose(patch, (2, 0, 1))
        patch = np.ascontiguousarray(patch)
        
        return {
            'image': patch,
            'image_name': img_name,
            'centers': filtered_centers,
            'crop_pos': (x, y),
            'image_size': patch.shape[1:]
        }
