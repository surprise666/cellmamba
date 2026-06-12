# CellMamba Configuration (Based on Paper arXiv:2512.21803)

import os
import platform
import torch


def convert_windows_to_wsl_path(win_path):
    """Convert Windows path to WSL path"""
    if platform.system() == 'Linux' or 'microsoft' in platform.release().lower():
        win_path = win_path.replace('\\', '/')
        if len(win_path) > 1 and win_path[1] == ':':
            drive = win_path[0].lower()
            return f"/mnt/{drive}/{win_path[3:]}"
    return win_path


class Config:
    # Data paths - Windows style
    _DATA_ROOT = r"D:\Code\Dataset\hover_net\CoNSeP\original\Train"
    _PRETRAINED_PATH = r"D:\Download\vssd_micro_best.pth"
    
    # Auto-convert paths based on environment
    if platform.system() == 'Linux' or 'microsoft' in platform.release().lower():
        DATA_ROOT = convert_windows_to_wsl_path(_DATA_ROOT)
        PRETRAINED_PATH = convert_windows_to_wsl_path(_PRETRAINED_PATH)
    else:
        DATA_ROOT = _DATA_ROOT
        PRETRAINED_PATH = _PRETRAINED_PATH
    
    IMAGE_DIR = f"{DATA_ROOT}/Images"
    LABEL_DIR = f"{DATA_ROOT}/Labels"
    
    # ============== Model Settings (Paper: 256x256 patches) ==============
    IMAGE_SIZE = 256  # Paper uses 256x256 patches
    PATCH_SIZE = 256
    CROP_STRATEGY = 'random'
    CROP_OVERLAPS = 32
    MIN_CELLS_PER_PATCH = 1
    
    # Backbone (Paper: base_channels=64)
    BASE_CHANNELS = 64
    
    # FPN (Paper: P2-P6, 5 levels)
    FPN_CHANNELS = 256
    
    # Detection (Paper: single-class detection on CoNSeP)
    NUM_CLASSES = 1  # Binary: foreground/background
    
    # FPN strides (Paper: P2=4, P3=8, P4=16, P5=32, P6=64)
    STRIDES = [4, 8, 16, 32, 64]
    
    # ============== Loss Function (Paper: Focal Loss + Smooth L1) ==============
    OBJECTNESS_WEIGHT = 1.0
    REGESSION_WEIGHT = 1.0
    
    # Focal Loss parameters (Paper: α=0.25, γ=2.0)
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # ============== Training Settings (Paper) ==============
    BATCH_SIZE = 8  # Reduced for 256x256 patches
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    
    # TMAC two-stage training (Paper: N=35)
    TMAC_EPOCH_THRESHOLD = 35
    
    # ============== Inference ==============
    CONF_THRESH = 0.3
    NMS_THRESH = 0.5
    MAX_DETECTIONS = 100
    
    # ============== Device ==============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()

# Print paths for debugging
print(f"Platform: {platform.system()}")
print(f"Data root: {config.DATA_ROOT}")
print(f"Image dir: {config.IMAGE_DIR}")
print(f"Label dir: {config.LABEL_DIR}")
print(f"Strides: {config.STRIDES}")
print(f"Patches: {config.PATCH_SIZE}x{config.PATCH_SIZE}")
