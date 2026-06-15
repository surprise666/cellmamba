"""
FCOS target generation - Center-only Assignment (1 cell = 1 grid point).

Design rationale (kept verbatim from the consensus version):
  * 1-for-1 center-only labels. NO 3x3 sampling, NO Gaussian.
    Reason: CoNSeP cells are tiny (12-24 px). One point per cell gives
    Focal Loss a clean signal, even though max confidence tops out at
    ~0.15-0.20. We accept that and lower the test threshold to 0.08.
  * round() (not floor) for grid assignment.
    Reason: 12-px cells span only 3 px at stride 4. floor() biases the
    grid point by 0.5 px and 50% of those cells would be dropped by
    the strict positivity check.
  * >= 0 (not > 0) in the distance check.
    Reason: A center that lands exactly on a box edge gives l/r/t/b == 0.
    > 0 would silently drop that cell.

Bug fixes vs the consensus version:
  * Grid-CENTER anchor: distances are computed to (gx + 0.5), the actual
    geometric center of grid cell gx. Without the 0.5 shift the regression
    target is systematically 0.5 px too small in every direction.
  * Overlap policy: when two cells collide on the same grid point, we
    keep the one with the smaller box (harder, less redundant for IoU).
    The previous condition (reg > l) was a dead branch because by
    construction the new cell's box is always >= the existing one in
    the colliding case.
  * No 0.01 clamp: regression loss is IoU Loss (see utils/losses.py)
    which internally clamps and is well-behaved at 0. The clamp was
    polluting the target and slowing convergence.
"""

import torch
import numpy as np


def compute_fcos_targets(image_size, centers, strides, device='cuda'):
    """
    Args:
        image_size: (H, W) of the input patch
        centers:    list of dicts with 'center_x', 'center_y' and
                    'bbox': (x1, y1, x2, y2) in image-pixel coords
        strides:    list of FPN strides, e.g. [4, 8, 16, 32, 64]

    Returns:
        objectness_targets, regression_targets
        both lists of length len(strides). Each tensor is (1, C, H, W).
        objectness: 1 channel, value 1.0 at the assigned center, 0 elsewhere
        regression: 4 channels, (l, t, r, b) in feature-map units
                    measured from the grid CENTER (gx + 0.5, gy + 0.5)
    """
    H, W = image_size[0], image_size[1]

    objectness_targets = []
    regression_targets = []

    for level_idx, stride in enumerate(strides):
        feat_h = H // stride
        feat_w = W // stride

        obj_target = torch.zeros((1, 1, feat_h, feat_w), dtype=torch.float32, device=device)
        reg_target = torch.zeros((1, 4, feat_h, feat_w), dtype=torch.float32, device=device)

        for center_info in centers:
            cx = center_info['center_x']
            cy = center_info['center_y']
            x1, y1, x2, y2 = center_info['bbox']

            box_size = max(x2 - x1, y2 - y1)
            if box_size <= 0:
                continue

            # =========================================================
            # 🚀 EDA micro scale assignment (CoNSeP, 2831 cells)
            #    0-24 px:    64%  (L0, S4)
            #   24-48 px:    31%  (L1, S8)
            #   48-96 px:     5%  (L2, S16)
            #  96-192 px:  0.2%  (L3, S32)
            #     >192 px:    0%  (L4, S64)
            # =========================================================
            should_assign = False
            if   level_idx == 0 and box_size <= 24:        should_assign = True
            elif level_idx == 1 and 24  < box_size <= 48:  should_assign = True
            elif level_idx == 2 and 48  < box_size <= 96:  should_assign = True
            elif level_idx == 3 and 96  < box_size <= 192: should_assign = True
            elif level_idx == 4 and box_size > 192:        should_assign = True
            if not should_assign:
                continue

            # Map everything to feature-map coordinates
            cx_feat = cx / stride
            cy_feat = cy / stride
            x1_feat = x1 / stride
            y1_feat = y1 / stride
            x2_feat = x2 / stride
            y2_feat = y2 / stride

            # =========================================================
            # 🚀 Center-only assignment: round to nearest grid index.
            # =========================================================
            gx = int(round(cx_feat))
            gy = int(round(cy_feat))

            if not (0 <= gx < feat_w and 0 <= gy < feat_h):
                continue

            # Distances measured to the GRID CENTER (gx + 0.5, gy + 0.5)
            # - not the integer grid index. This keeps regression targets
            #   geometrically honest: the model's softplus(regression) is
            #   compared against these exact same anchors by the IoU loss.
            l = (gx + 0.5) - x1_feat
            t = (gy + 0.5) - y1_feat
            r =  x2_feat   - (gx + 0.5)
            b =  y2_feat   - (gy + 0.5)

            # >= 0 (not > 0) to keep 1-px cells whose center lands
            # exactly on the box edge. Softplus(0) ~= 0.69, which is a
            # valid regression target.
            if l < 0 or t < 0 or r < 0 or b < 0:
                continue

            # Overlap policy: rare in CoNSeP but happens with 300+ cells.
            # Keep the SMALLER box - it is the harder one to regress and
            # the larger one is more likely to be matched at decode time
            # by some neighbouring cell that didn't collide here.
            if obj_target[0, 0, gy, gx] == 0.0:
                obj_target[0, 0, gy, gx] = 1.0
                reg_target[0, 0, gy, gx] = l
                reg_target[0, 1, gy, gx] = t
                reg_target[0, 2, gy, gx] = r
                reg_target[0, 3, gy, gx] = b
            else:
                # Existing assignment present. Keep the smaller (harder)
                # box: (l+r)*(t+b) is a proxy for area at the grid center.
                cur_area = (reg_target[0, 0, gy, gx] + reg_target[0, 2, gy, gx]) * \
                           (reg_target[0, 1, gy, gx] + reg_target[0, 3, gy, gx])
                new_area = (l + r) * (t + b)
                if new_area < cur_area:
                    reg_target[0, 0, gy, gx] = l
                    reg_target[0, 1, gy, gx] = t
                    reg_target[0, 2, gy, gx] = r
                    reg_target[0, 3, gy, gx] = b

        objectness_targets.append(obj_target)
        regression_targets.append(reg_target)

    return objectness_targets, regression_targets


def get_model_strides(model, input_size=256):
    """Probe the model to get the real FPN strides from forward output."""
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        outputs = model(dummy_input)

    actual_strides = []
    for i in range(len(outputs['objectness'])):
        feat_h = outputs['objectness'][i].shape[2]
        actual_strides.append(input_size // feat_h)
    return actual_strides


def batch_encode_boxes(regression_list, strides):
    """Reserved for future use."""
    pass
