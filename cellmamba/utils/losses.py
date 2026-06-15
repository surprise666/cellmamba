"""
Loss functions for CellMamba (FCOS).
- Focal Loss (sum, divided by total positives)
- IoU Loss on bounding box regression

The regression head outputs (l, t, r, b) in feature-map pixel units.
We softplus() the raw logits, then convert to (x1, y1, x2, y2) coordinates
relative to the grid center, and finally compute IoU.  This is the
mathematically correct form (the buggy version treated (l,t,r,b) directly
as (x1,y1,x2,y2) and was a no-op geometrically).
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
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.sum()


def compute_iou_loss(pred_ltrb, target_ltrb):
    """
    IoU loss on (l, t, r, b) distance vectors.
    Internally converts to (x1, y1, x2, y2) at the grid center.

    pred_ltrb, target_ltrb: (N, 4) tensors
    """
    pred_l = pred_ltrb[:, 0]
    pred_t = pred_ltrb[:, 1]
    pred_r = pred_ltrb[:, 2]
    pred_b = pred_ltrb[:, 3]

    tgt_l = target_ltrb[:, 0]
    tgt_t = target_ltrb[:, 1]
    tgt_r = target_ltrb[:, 2]
    tgt_b = target_ltrb[:, 3]

    # Predicted box in (x1, y1, x2, y2)
    pred_x1 = -pred_l
    pred_y1 = -pred_t
    pred_x2 =  pred_r
    pred_y2 =  pred_b

    # Target box in (x1, y1, x2, y2)
    tgt_x1 = -tgt_l
    tgt_y1 = -tgt_t
    tgt_x2 =  tgt_r
    tgt_y2 =  tgt_b

    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area  = (tgt_x2  - tgt_x1 ).clamp(min=0) * (tgt_y2  - tgt_y1 ).clamp(min=0)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = pred_area + tgt_area - inter
    iou = inter / (union + 1e-6)
    # -log(iou) is well-behaved (loss in (0, +inf))
    return -torch.log(iou.clamp(min=1e-6)).mean()


class CellMambaLoss(nn.Module):
    def __init__(self, objectness_weight=1.0, regression_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.objectness_weight = objectness_weight
        self.regression_weight = regression_weight

    def forward(self, predictions, targets):
        strides = predictions['strides']
        num_levels = len(strides)

        # Global positive-sample count (for focal-loss normalization)
        total_num_pos = 0
        for i in range(num_levels):
            target_obj = targets['objectness_targets'][i]
            total_num_pos += (target_obj > 0).sum().item()
        total_num_pos = max(1.0, float(total_num_pos))

        total_obj_loss = 0.0
        all_pred_reg_pos = []
        all_target_reg_pos = []

        for i in range(num_levels):
            pred_obj = predictions['objectness'][i]
            pred_reg = predictions['regression'][i]
            target_obj = targets['objectness_targets'][i]
            target_reg = targets['regression_targets'][i]

            if target_obj.dim() == 5: target_obj = target_obj.squeeze(2)
            if target_reg.dim() == 5: target_reg = target_reg.squeeze(1)

            B_reg, C_reg, H_reg, W_reg = target_reg.shape
            pred_h, pred_w = pred_obj.shape[2:4]
            if pred_h != H_reg or pred_w != W_reg:
                pred_obj = F.interpolate(pred_obj, size=(H_reg, W_reg), mode='bilinear', align_corners=False)
                pred_reg = F.interpolate(pred_reg, size=(H_reg, W_reg), mode='bilinear', align_corners=False)

            total_obj_loss += self.focal_loss(pred_obj, target_obj)

            # centerness > 0 marks a positive sample (no more 0.01 clamp!)
            pos_mask = target_obj.squeeze(1) > 0
            if pos_mask.sum() > 0:
                pred_reg_flat = pred_reg.reshape(B_reg, 4, -1)
                target_reg_flat = target_reg.reshape(B_reg, 4, -1)
                pos_mask_flat = pos_mask.reshape(B_reg, -1)

                for b in range(B_reg):
                    pos_idx = pos_mask_flat[b]
                    if pos_idx.sum() > 0:
                        all_pred_reg_pos.append(pred_reg_flat[b, :, pos_idx].T)
                        all_target_reg_pos.append(target_reg_flat[b, :, pos_idx].T)

        avg_obj_loss = total_obj_loss / total_num_pos

        if len(all_pred_reg_pos) > 0:
            pred_reg_pos   = torch.cat(all_pred_reg_pos,   dim=0)
            target_reg_pos = torch.cat(all_target_reg_pos, dim=0)

            pred_ltrb = F.softplus(pred_reg_pos)
            # 🚀 IoU Loss replaces Smooth L1 — forces the network to learn
            #    both centering AND box size/shape.
            avg_reg_loss = compute_iou_loss(pred_ltrb, target_reg_pos)
        else:
            avg_reg_loss = torch.tensor(0.0, device=predictions['objectness'][0].device)

        loss = avg_obj_loss * self.objectness_weight + avg_reg_loss * self.regression_weight
        return loss, {'obj_loss': avg_obj_loss, 'reg_loss': avg_reg_loss}
