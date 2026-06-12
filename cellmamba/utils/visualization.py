"""
Helper functions for visualization and debugging
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_feature_maps(feature_maps, save_path=None, max_channels=16):
    """
    Visualize feature maps from different FPN levels
    
    Args:
        feature_maps: list of (B, C, H, W) tensors
        save_path: path to save the visualization
        max_channels: maximum number of channels to show per level
    """
    n_levels = len(feature_maps)
    fig, axes = plt.subplots(n_levels, min(max_channels, feature_maps[0].shape[1]), 
                             figsize=(max_channels * 2, n_levels * 2))
    
    if n_levels == 1:
        axes = [axes]
    
    for level_idx, feat in enumerate(feature_maps):
        # Take first batch
        feat = feat[0]  # (C, H, W)
        C = feat.shape[0]
        n_show = min(max_channels, C)
        
        for ch in range(n_show):
            ax = axes[level_idx][ch] if n_levels > 1 else axes[ch]
            ax.imshow(feat[ch].cpu().numpy(), cmap='viridis')
            ax.set_title(f'L{level_idx} Ch{ch}')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature map visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_targets(objectness_targets, regression_targets, strides, save_path=None):
    """
    Visualize FCOS targets for debugging
    
    Args:
        objectness_targets: list of (1, H, W) tensors
        regression_targets: list of (4, H, W) tensors
        strides: list of strides
        save_path: path to save
    """
    n_levels = len(objectness_targets)
    fig, axes = plt.subplots(2, n_levels, figsize=(n_levels * 4, 8))
    
    for level_idx in range(n_levels):
        obj = objectness_targets[level_idx][0].cpu().numpy()  # (H, W)
        reg = regression_targets[level_idx].cpu().numpy()  # (4, H, W)
        
        # Objectness
        axes[0, level_idx].imshow(obj, cmap='hot')
        axes[0, level_idx].set_title(f'Objectness S{strides[level_idx]}')
        axes[0, level_idx].axis('off')
        
        # Regression magnitude
        reg_mag = np.sqrt(reg[0]**2 + reg[1]**2)  # sqrt(dx^2 + dy^2)
        axes[1, level_idx].imshow(reg_mag, cmap='coolwarm')
        axes[1, level_idx].set_title(f'Reg Magn S{strides[level_idx]}')
        axes[1, level_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved target visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_curve(losses, save_path='loss_curve.png'):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to {save_path}")


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    print(f"Frozen parameters: {frozen:,}")
    print(f"Total parameters: {total + frozen:,}")
    return total


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
