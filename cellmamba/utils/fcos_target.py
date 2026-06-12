import torch
import numpy as np

def compute_fcos_targets(image_size, centers, strides, device='cuda'):
    """
    Compute FCOS targets for multi-scale detection

    Args:
        image_size: (H, W) tuple - the actual size of the input patch
        centers: list of {'center_x', 'center_y', 'bbox': (x1, y1, x2, y2)}
        strides: list of strides for each FPN level [4, 8, 16, 32, 64]
        device: torch device

    Returns:
        objectness_targets, regression_targets (both lists of length len(strides))
        Each target is (B=1, C, H, W) matching model output shape
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

            if box_size == 0:
                continue

            # Simplified scale assignment for small cells
            # For CoNSeP cells (mean ~19px), focus on fine levels
            #
            # Strategy: assign to finest level that can represent this box
            # - Level 0 (stride 4): all boxes <= 128px
            # - Level 1 (stride 8): boxes 32-256px  
            # - Level 2 (stride 16): boxes 64-512px
            # - Level 3 (stride 32): boxes 128-1024px
            # - Level 4 (stride 64): boxes > 256px
            
            should_assign = False
            
            if level_idx == 0:  # stride 4
                if box_size <= 128:
                    should_assign = True
            elif level_idx == 1:  # stride 8
                if box_size > 32 and box_size <= 256:
                    should_assign = True
            elif level_idx == 2:  # stride 16
                if box_size > 64 and box_size <= 512:
                    should_assign = True
            elif level_idx == 3:  # stride 32
                if box_size > 128:
                    should_assign = True
            
            if not should_assign:
                continue

            # Feature map coordinates
            cx_feat = cx / stride
            cy_feat = cy / stride
            x1_feat = x1 / stride
            y1_feat = y1 / stride
            x2_feat = x2 / stride
            y2_feat = y2 / stride

            # Center point assignment
            gx = int(round(cx_feat))
            gy = int(round(cy_feat))

            if 0 <= gx < feat_w and 0 <= gy < feat_h:
                obj_target[0, 0, gy, gx] = 1.0

                # LTRB: distances from center to box edges (in feature coordinates)
                reg_target[0, 0, gy, gx] = gx - x1_feat  # left
                reg_target[0, 1, gy, gx] = gy - y1_feat  # top
                reg_target[0, 2, gy, gx] = x2_feat - gx  # right
                reg_target[0, 3, gy, gx] = y2_feat - gy  # bottom

        objectness_targets.append(obj_target)
        regression_targets.append(reg_target)

    return objectness_targets, regression_targets


def get_model_strides(model, input_size=256):
    """
    Get actual strides from model based on output sizes.
    This ensures target generation matches model output.
    """
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        outputs = model(dummy_input)
    
    actual_strides = []
    for i in range(len(outputs['objectness'])):
        feat_h = outputs['objectness'][i].shape[2]
        stride = input_size // feat_h
        actual_strides.append(stride)
    
    return actual_strides


def batch_encode_boxes(regression_list, strides):
    """Training utility - not needed for current approach"""
    pass
