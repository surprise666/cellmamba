"""
CellMamba Training Script - Paper Implementation (arXiv:2512.21803)
- VSSD Backbone with NC-Mamba and MSA
- Triple-Mapping Adaptive Coupling (TMAC)
- Adaptive Mamba Head
- Training: 256x256 patches (as per paper)
- Validation: Full image sliding window evaluation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from configs.config import config
from models.cellmamba import build_model
from utils.dataset import CoNSePDataset, SlidingWindowDataset, get_train_val_split
from utils.fcos_target import compute_fcos_targets
from utils.losses import CellMambaLoss


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Custom collate function for batched training"""
    images = torch.stack([torch.from_numpy(item['image']).float() for item in batch])
    image_names = [item['image_name'] for item in batch]
    centers_list = [item['centers'] for item in batch]
    image_sizes = [item['image_size'] for item in batch]
    
    return {
        'images': images,
        'image_names': image_names,
        'centers_list': centers_list,
        'image_sizes': image_sizes
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, strides, epoch):
    """Train for one epoch"""
    model.train()
    
    # Update epoch for TMAC two-stage training
    model.set_epoch(epoch)
    
    total_loss = 0
    total_obj = 0
    total_reg = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch in pbar:
        images = batch['images'].to(device)
        centers_list = batch['centers_list']
        image_sizes = batch['image_sizes']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute targets and aggregate by level (5 levels: P2-P6)
        all_obj_targets = [[] for _ in range(len(strides))]
        all_reg_targets = [[] for _ in range(len(strides))]
        
        for i in range(images.size(0)):
            obj_targets, reg_targets = compute_fcos_targets(
                image_size=image_sizes[i],
                centers=centers_list[i],
                strides=strides,
                device=device
            )
            for lvl in range(len(strides)):
                all_obj_targets[lvl].append(obj_targets[lvl])
                all_reg_targets[lvl].append(reg_targets[lvl])
        
        # Stack targets by level
        stacked_targets = {
            'objectness_targets': [torch.stack(all_obj_targets[lvl]) for lvl in range(len(strides))],
            'regression_targets': [torch.stack(all_reg_targets[lvl]) for lvl in range(len(strides))]
        }
        
        # Compute loss
        loss, loss_dict = criterion(outputs, stacked_targets)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        avg_batch_loss = loss.item()
        total_obj += loss_dict['obj_loss'].item() if isinstance(loss_dict['obj_loss'], torch.Tensor) else loss_dict['obj_loss']
        total_reg += loss_dict['reg_loss'].item() if isinstance(loss_dict['reg_loss'], torch.Tensor) else loss_dict['reg_loss']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{avg_batch_loss:.4f}',
            'obj': f'{loss_dict["obj_loss"]:.4f}',
            'reg': f'{loss_dict["reg_loss"]:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_obj = total_obj / num_batches
    avg_reg = total_reg / num_batches
    
    return avg_loss, avg_obj, avg_reg


@torch.no_grad()
def validate(model, val_loader, criterion, device, strides, epoch=None):
    """Validate using sliding window evaluation"""
    model.eval()
    
    if epoch is not None:
        model.set_epoch(epoch)
    
    total_loss = 0
    total_obj = 0
    total_reg = 0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        images = batch['images'].to(device)
        centers_list = batch['centers_list']
        image_sizes = batch['image_sizes']
        
        # Forward pass
        outputs = model(images)
        
        # Compute targets
        all_obj_targets = [[] for _ in range(len(strides))]
        all_reg_targets = [[] for _ in range(len(strides))]
        
        for i in range(images.size(0)):
            obj_targets, reg_targets = compute_fcos_targets(
                image_size=image_sizes[i],
                centers=centers_list[i],
                strides=strides,
                device=device
            )
            for lvl in range(len(strides)):
                all_obj_targets[lvl].append(obj_targets[lvl])
                all_reg_targets[lvl].append(reg_targets[lvl])
        
        # Stack targets
        stacked_targets = {
            'objectness_targets': [torch.stack(all_obj_targets[lvl]) for lvl in range(len(strides))],
            'regression_targets': [torch.stack(all_reg_targets[lvl]) for lvl in range(len(strides))]
        }
        
        # Compute loss
        loss, loss_dict = criterion(outputs, stacked_targets)
        
        # Metrics
        total_loss += loss.item()
        total_obj += loss_dict['obj_loss'].item() if isinstance(loss_dict['obj_loss'], torch.Tensor) else loss_dict['obj_loss']
        total_reg += loss_dict['reg_loss'].item() if isinstance(loss_dict['reg_loss'], torch.Tensor) else loss_dict['reg_loss']
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_obj = total_obj / num_batches
    avg_reg = total_reg / num_batches
    
    return avg_loss, avg_obj, avg_reg


def main():
    set_seed(42)
    
    # Setup
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("CellMamba Training - Paper Implementation")
    print(f"{'='*60}")
    print(f"Patch size: {config.PATCH_SIZE}x{config.PATCH_SIZE} (paper: 256x256)")
    print(f"FPN levels: P2-P6 (strides: {config.STRIDES})")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"TMAC threshold: N={config.TMAC_EPOCH_THRESHOLD}")
    
    # Get train/val split
    train_indices, val_indices = get_train_val_split(
        config.IMAGE_DIR,
        config.LABEL_DIR,
        val_ratio=0.2
    )
    print(f"\nTrain images: {len(train_indices)}, Val images: {len(val_indices)}")
    
    # Create datasets with 256x256 patches (as per paper)
    train_dataset = SlidingWindowDataset(
        image_dir=config.IMAGE_DIR,
        label_dir=config.LABEL_DIR,
        patch_size=config.PATCH_SIZE,  # 256
        overlaps=64,  # 25% overlap for 256 patches
        image_indices=train_indices
    )
    
    val_dataset = SlidingWindowDataset(
        image_dir=config.IMAGE_DIR,
        label_dir=config.LABEL_DIR,
        patch_size=config.PATCH_SIZE,  # 256
        overlaps=64,
        image_indices=val_indices
    )
    
    print(f"Train patches: {len(train_dataset)}, Val patches: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,  # 16 for 256x256
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("\nBuilding CellMamba model...")
    model = build_model(config)
    model = model.to(device)
    
    # Get actual strides from model (to match output sizes)
    from utils.fcos_target import get_model_strides
    actual_strides = get_model_strides(model, config.PATCH_SIZE)
    print(f"Model actual strides: {actual_strides}")
    print(f"Config strides: {config.STRIDES}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Load pretrained weights if available
    if os.path.exists(config.PRETRAINED_PATH):
        print(f"Loading pretrained weights from {config.PRETRAINED_PATH}")
        model.load_pretrained(config.PRETRAINED_PATH)
    
    # Loss (Paper: Focal Loss + Smooth L1)
    criterion = CellMambaLoss(
        objectness_weight=config.OBJECTNESS_WEIGHT,
        regression_weight=config.REGESSION_WEIGHT
    )
    
    # Optimizer (Paper: SGD with lr=1e-3, wd=1e-4)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True
    )
    
    # Learning rate schedule (Paper: LinearLR + MultiStepLR)
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 70], gamma=0.1)
        ],
        milestones=[warmup_epochs]
    )
    
    # Training loop
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_obj, train_reg = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            actual_strides, epoch
        )
        print(f"Train Loss: {train_loss:.4f} | Obj: {train_obj:.4f} | Reg: {train_reg:.4f}")
        
        # Validate
        val_loss, val_obj, val_reg = validate(
            model, val_loader, criterion, device, actual_strides, epoch
        )
        print(f"Val Loss: {val_loss:.4f} | Obj: {val_obj:.4f} | Reg: {val_reg:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = 'checkpoints/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"✓ Saved best model to {save_path}")
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
