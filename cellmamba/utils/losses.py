"""
Loss functions - Paper Implementation
- Focal Loss for classification (as per paper)
- Smooth L1 Loss for bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection
    
    From paper: Focal Loss [20] for classification to mitigate class imbalance
    Paper parameters: α=0.25, γ=2.0
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.clamp(min=1e-7, max=1-1e-7)
        
        # Binary cross entropy
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Focal weight
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class CellMambaLoss(nn.Module):
    """
    Total loss: Focal Loss + Smooth L1 Loss
    
    From paper: Focal Loss for classification + Smooth L1 Loss for box regression
    """
    def __init__(self, objectness_weight=1.0, regression_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.objectness_weight = objectness_weight
        self.regression_weight = regression_weight

    def forward(self, predictions, targets):
        strides = predictions['strides']
        total_obj_loss = 0.0
        total_reg_loss = 0.0
        num_levels = len(strides)

        for i in range(num_levels):
            pred_obj = predictions['objectness'][i]      # (B, 1, H, W)
            pred_reg = predictions['regression'][i]      # (B, 4, H, W)
            target_obj = targets['objectness_targets'][i]  # (B, 1, 1, H, W) or (B, 1, H, W)
            target_reg = targets['regression_targets'][i]  # (B, 1, 4, H, W) or (B, 4, H, W)

            # Handle extra dimension if present
            # target_obj: (B, 1, 1, H, W) -> squeeze dim 2 (which is 1)
            # target_reg: (B, 1, 4, H, W) -> squeeze dim 1 (which is 1)
            if target_obj.dim() == 5:
                target_obj = target_obj.squeeze(2)  # (B, 1, 1, H, W) -> (B, 1, H, W)
            if target_reg.dim() == 5:
                target_reg = target_reg.squeeze(1)  # (B, 1, 4, H, W) -> (B, 4, H, W)

            # Get shapes
            B_obj, _, H_obj, W_obj = target_obj.shape
            B_reg, C_reg, H_reg, W_reg = target_reg.shape
            
            # Sanity check
            assert B_obj == B_reg, f"Batch size mismatch: obj={B_obj}, reg={B_reg}"

            # Interpolate predictions to match target size if different
            pred_h, pred_w = pred_obj.shape[2:4]
            if pred_h != H_obj or pred_w != W_obj:
                pred_obj = F.interpolate(pred_obj, size=(H_obj, W_obj), mode='bilinear', align_corners=False)
                pred_reg = F.interpolate(pred_reg, size=(H_obj, W_obj), mode='bilinear', align_corners=False)

            # Focal loss for objectness
            total_obj_loss += self.focal_loss(pred_obj, target_obj)

            # Smooth L1 loss for regression (only on positive locations)
            # target_obj: (B, 1, H, W) -> (B, H, W)
            pos_mask = target_obj.squeeze(1) > 0  # (B, H, W)

            if pos_mask.sum() > 0:
                # Flatten spatial dimensions: (B, 4, H, W) -> (B, 4, H*W)
                pred_reg_flat = pred_reg.reshape(B_reg, 4, -1)      # (B, 4, H*W)
                target_reg_flat = target_reg.reshape(B_reg, 4, -1)  # (B, 4, H*W)
                # pos_mask: (B, H, W) -> (B, H*W)
                pos_mask_flat = pos_mask.reshape(B_reg, -1)  # (B, H*W)

                # Combine all positive samples from all batches
                all_pred_reg_pos = []
                all_target_reg_pos = []
                
                for b in range(B_reg):
                    pos_idx = pos_mask_flat[b]  # (H*W,)
                    n_pos = pos_idx.sum().item()
                    if n_pos > 0:
                        # Get all 4 channels at positive positions
                        all_pred_reg_pos.append(pred_reg_flat[b, :, pos_idx].T)  # (n_pos, 4)
                        all_target_reg_pos.append(target_reg_flat[b, :, pos_idx].T)  # (n_pos, 4)

                if len(all_pred_reg_pos) > 0:
                    pred_reg_pos = torch.cat(all_pred_reg_pos, dim=0)    # (N_pos, 4)
                    target_reg_pos = torch.cat(all_target_reg_pos, dim=0)  # (N_pos, 4)

                    # Use softplus to ensure positive predictions
                    pred_ltrb = F.softplus(pred_reg_pos)

                    # Smooth L1 loss
                    reg_loss = F.smooth_l1_loss(pred_ltrb, target_reg_pos)
                    total_reg_loss += reg_loss

        # Average over levels
        avg_obj_loss = total_obj_loss / num_levels
        avg_reg_loss = total_reg_loss / num_levels if num_levels > 0 else torch.tensor(0.0, device=pred_reg.device)

        # Weighted sum
        loss = avg_obj_loss * self.objectness_weight + avg_reg_loss * self.regression_weight

        return loss, {'obj_loss': avg_obj_loss, 'reg_loss': avg_reg_loss}
