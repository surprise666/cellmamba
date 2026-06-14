"""
Loss functions - Paper Implementation (Fixed Normalization)
- Focal Loss normalized by total positive samples
- Smooth L1 Loss for bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.clamp(min=1e-7, max=1-1e-7)
        
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        # 🚀 核心修复1：这里必须返回 sum()，绝对不能用 mean()！
        return loss.sum()


class CellMambaLoss(nn.Module):
    def __init__(self, objectness_weight=1.0, regression_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.objectness_weight = objectness_weight
        self.regression_weight = regression_weight

    def forward(self, predictions, targets):
        strides = predictions['strides']
        num_levels = len(strides)

        # 🚀 核心修复2：统计这一批次中，所有尺度下真正的"正样本(细胞)总数"
        total_num_pos = 0
        for i in range(num_levels):
            target_obj = targets['objectness_targets'][i]
            total_num_pos += (target_obj > 0).sum().item()
        
        # 防止除以 0 导致 Nan
        total_num_pos = max(1.0, total_num_pos)

        total_obj_loss = 0.0
        total_reg_loss = 0.0

        for i in range(num_levels):
            pred_obj = predictions['objectness'][i]
            pred_reg = predictions['regression'][i]
            target_obj = targets['objectness_targets'][i]
            target_reg = targets['regression_targets'][i]

            if target_obj.dim() == 5:
                target_obj = target_obj.squeeze(2)
            if target_reg.dim() == 5:
                target_reg = target_reg.squeeze(1)

            B_obj, _, H_obj, W_obj = target_obj.shape
            B_reg, C_reg, H_reg, W_reg = target_reg.shape
            
            # 尺寸对齐
            pred_h, pred_w = pred_obj.shape[2:4]
            if pred_h != H_obj or pred_w != W_obj:
                pred_obj = F.interpolate(pred_obj, size=(H_obj, W_obj), mode='bilinear', align_corners=False)
                pred_reg = F.interpolate(pred_reg, size=(H_obj, W_obj), mode='bilinear', align_corners=False)

            # 🚀 核心修复3：Focal Loss 的总和 除以 正样本数！
            obj_loss_sum = self.focal_loss(pred_obj, target_obj)
            total_obj_loss += obj_loss_sum / total_num_pos

            # Regression loss logic
            pos_mask = target_obj.squeeze(1) > 0
            if pos_mask.sum() > 0:
                pred_reg_flat = pred_reg.reshape(B_reg, 4, -1)
                target_reg_flat = target_reg.reshape(B_reg, 4, -1)
                pos_mask_flat = pos_mask.reshape(B_reg, -1)

                all_pred_reg_pos = []
                all_target_reg_pos = []
                
                for b in range(B_reg):
                    pos_idx = pos_mask_flat[b]
                    n_pos = pos_idx.sum().item()
                    if n_pos > 0:
                        all_pred_reg_pos.append(pred_reg_flat[b, :, pos_idx].T)
                        all_target_reg_pos.append(target_reg_flat[b, :, pos_idx].T)

                if len(all_pred_reg_pos) > 0:
                    pred_reg_pos = torch.cat(all_pred_reg_pos, dim=0)
                    target_reg_pos = torch.cat(all_target_reg_pos, dim=0)

                    pred_ltrb = F.softplus(pred_reg_pos)
                    # Smooth L1 默认用 mean()，这正好相当于除以了 n_pos，所以不需要改
                    reg_loss = F.smooth_l1_loss(pred_ltrb, target_reg_pos)
                    total_reg_loss += reg_loss

        avg_obj_loss = total_obj_loss / num_levels
        avg_reg_loss = total_reg_loss / num_levels if num_levels > 0 else torch.tensor(0.0, device=pred_reg.device)

        loss = avg_obj_loss * self.objectness_weight + avg_reg_loss * self.regression_weight

        return loss, {'obj_loss': avg_obj_loss, 'reg_loss': avg_reg_loss}
