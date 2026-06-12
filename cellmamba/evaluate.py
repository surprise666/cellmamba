"""
Evaluation Script for CellMamba - Paper Implementation
Sliding Window Inference with proper FPN P2-P6 support
"""

import os
import matplotlib
matplotlib.use('Agg')
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from configs.config import config
from models.cellmamba import build_model
from utils.dataset import parse_consep_label


class FullImageTestDataset(Dataset):
    """Dataset for full image evaluation"""
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.mat')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        if os.path.exists(label_path):
            label_data = parse_consep_label(label_path)
            centers = label_data['centers']
        else:
            centers = []

        img_tensor = img.astype(np.float32) / 255.0
        img_tensor = (img_tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = np.transpose(img_tensor, (2, 0, 1))

        return {'image': img_tensor, 'centers': centers, 'image_name': img_name, 'image_size': [img_h, img_w]}


def collate_fn(batch):
    return {
        'images': [item['image'] for item in batch],
        'image_names': [item['image_name'] for item in batch],
        'centers_list': [item['centers'] for item in batch],
        'image_sizes': [item['image_size'] for item in batch]
    }


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)


def decode_predictions(objectness_list, regression_list, strides, conf_thresh=0.3, nms_thresh=0.45):
    """
    Decode predictions from all FPN levels to bounding boxes
    
    Args:
        objectness_list: list of (B, 1, H, W) tensors for each level
        regression_list: list of (B, 4, H, W) tensors for each level
        strides: list of strides [4, 8, 16, 32, 64]
        conf_thresh: confidence threshold
        nms_thresh: NMS threshold
    """
    all_boxes, all_scores = [], []
    
    for level_idx, (obj, reg, stride) in enumerate(zip(objectness_list, regression_list, strides)):
        B, _, H, W = obj.shape
        for b in range(B):
            scores = torch.sigmoid(obj[b, 0])
            reg_b = reg[b]

            # Apply confidence threshold
            mask = scores > conf_thresh
            if not mask.any():
                continue

            y_grid, x_grid = torch.where(mask)
            scores_filtered = scores[mask]
            reg_filtered = reg_b[:, mask]

            # Decode LTRB distances to absolute coordinates
            l = F.softplus(reg_filtered[0]) * stride
            t = F.softplus(reg_filtered[1]) * stride
            r = F.softplus(reg_filtered[2]) * stride
            b = F.softplus(reg_filtered[3]) * stride

            # Center point in original coordinates
            cx = x_grid.float() * stride
            cy = y_grid.float() * stride

            # Convert to box corners
            x1, y1 = cx - l, cy - t
            x2, y2 = cx + r, cy + b

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            all_boxes.append(boxes)
            all_scores.append(scores_filtered)

    if len(all_boxes) > 0:
        boxes_tensor = torch.cat(all_boxes, dim=0)
        scores_tensor = torch.cat(all_scores, dim=0)
        keep_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=nms_thresh)
        return boxes_tensor[keep_idx].cpu().numpy(), scores_tensor[keep_idx].cpu().numpy()
    
    return np.zeros((0, 4)), np.zeros((0,))


def sliding_window_inference(model, image_numpy, device, strides, patch_size=256, overlaps=64, conf_thresh=0.3):
    """
    Industrial-grade sliding window inference
    
    Args:
        model: CellMamba model
        image_numpy: (C, H, W) image in numpy format
        device: torch device
        strides: list of FPN strides [4, 8, 16, 32, 64]
        patch_size: size of sliding window (paper: 256)
        overlaps: overlap between windows (paper: 25% = 64)
        conf_thresh: confidence threshold
    """
    C, H, W = image_numpy.shape
    stride_window = patch_size - overlaps

    all_boxes, all_scores = [], []

    y = 0
    while y < H:
        x = 0
        while x < W:
            # Extract patch
            crop = image_numpy[:, y:y+patch_size, x:x+patch_size]

            # Pad if needed (edge handling)
            pad_h = max(0, patch_size - crop.shape[1])
            pad_w = max(0, patch_size - crop.shape[2])
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')

            crop_tensor = torch.from_numpy(crop).unsqueeze(0).to(device).float()
            predictions = model(crop_tensor)

            obj_list = predictions['objectness']
            reg_list = predictions['regression']

            # Local NMS
            boxes, scores = decode_predictions(obj_list, reg_list, strides, conf_thresh, nms_thresh=0.45)

            if len(boxes) > 0:
                # Offset boxes back to original image coordinates
                boxes[:, 0] += x
                boxes[:, 1] += y
                boxes[:, 2] += x
                boxes[:, 3] += y

                all_boxes.append(boxes)
                all_scores.append(scores)

            x += stride_window
        y += stride_window

    if len(all_boxes) > 0:
        boxes_all = np.concatenate(all_boxes, axis=0)
        scores_all = np.concatenate(all_scores, axis=0)

        # Global NMS
        boxes_tensor = torch.from_numpy(boxes_all)
        scores_tensor = torch.from_numpy(scores_all)
        keep_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.35)

        return boxes_tensor[keep_idx].cpu().numpy(), scores_tensor[keep_idx].cpu().numpy()

    return np.zeros((0, 4)), np.zeros((0,))


def compute_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Compute precision, recall, F1 at given IoU threshold"""
    if len(gt_boxes) == 0:
        return (1.0, 0, 0, 0, 0) if len(pred_boxes) == 0 else (0.0, 0, len(pred_boxes), 0, 0)
    if len(pred_boxes) == 0:
        return 0.0, 0, 0, len(gt_boxes), len(gt_boxes)

    sort_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sort_idx]

    matched_gt = set()
    tp, fp = 0, 0
    for pred_box in pred_boxes:
        matched = False
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if compute_iou(pred_box, gt_box) >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, tp, fp, fn, len(gt_boxes)


def evaluate_model(model, dataloader, device, strides, conf_thresh=0.3):
    """Evaluate model on dataset"""
    model.eval()
    all_tp, all_fp, all_fn, all_gt, all_pred = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            image_list = batch['images']
            centers_list = batch['centers_list']

            for b in range(len(image_list)):
                image_numpy = image_list[b]

                pred_boxes, pred_scores = sliding_window_inference(
                    model, image_numpy, device, strides,
                    patch_size=config.PATCH_SIZE, overlaps=config.CROP_OVERLAPS, conf_thresh=conf_thresh
                )

                gt_boxes = np.array([c['bbox'] for c in centers_list[b]])

                f1, tp, fp, fn, num_gt = compute_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5)
                all_tp += tp
                all_fp += fp
                all_fn += fn
                all_gt += num_gt
                all_pred += len(pred_boxes)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': all_tp,
        'total_fp': all_fp,
        'total_fn': all_fn,
        'total_gt': all_gt,
        'total_pred': all_pred
    }


def visualize_results(dataset, model, device, strides, output_dir='eval_results', num_samples=3, conf_thresh=0.3):
    """Generate visualization comparing GT and predictions"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print(f"\nGenerating visualizations for {min(num_samples, len(dataset))} images...")
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        image_numpy, centers, image_name = sample['image'], sample['centers'], sample['image_name']
        img_h, img_w = sample['image_size']
        gt_boxes = np.array([c['bbox'] for c in centers])

        with torch.no_grad():
            pred_boxes, pred_scores = sliding_window_inference(
                model, image_numpy, device, strides,
                patch_size=config.PATCH_SIZE, overlaps=config.CROP_OVERLAPS, conf_thresh=conf_thresh
            )

        # Denormalize image for visualization
        img_vis = np.transpose(image_numpy, (1, 2, 0))
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground truth
        axes[0].imshow(img_vis)
        for box in gt_boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                               fill=False, color='green', linewidth=2)
            axes[0].add_patch(rect)
        axes[0].set_title(f'Ground Truth ({len(gt_boxes)} cells)', fontsize=14)
        axes[0].axis('off')

        # Predictions
        axes[1].imshow(img_vis)
        for box, score in zip(pred_boxes, pred_scores):
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                               fill=False, color='red', linewidth=2)
            axes[1].add_patch(rect)
        axes[1].set_title(f'Predictions ({len(pred_boxes)} cells, Conf > {conf_thresh})', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'vis_{image_name}')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("CellMamba Evaluation - Paper Implementation")
    print(f"{'='*60}")
    print(f"Patch size: {config.PATCH_SIZE}x{config.PATCH_SIZE}")
    print(f"FPN levels: P2-P6 (strides: {config.STRIDES})")
    
    # Load test data
    test_image_dir = config.IMAGE_DIR.replace('Train', 'Test')
    test_label_dir = config.LABEL_DIR.replace('Train', 'Test')
    
    if not os.path.exists(test_image_dir):
        print(f"Test directory not found: {test_image_dir}")
        print("Using training data for evaluation instead.")
        test_image_dir = config.IMAGE_DIR
        test_label_dir = config.LABEL_DIR
    
    dataset = FullImageTestDataset(test_image_dir, test_label_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Load model
    model = build_model(config).to(device)
    
    if os.path.exists('checkpoints/best_model.pth'):
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("Warning: No checkpoint found, using random weights!")

    # Print model output ranges
    print("\n=== DEBUG: Model Output Ranges ===")
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        for lvl, (obj, reg) in enumerate(zip(outputs['objectness'], outputs['regression'])):
            stride = outputs['strides'][lvl]
            obj_sigmoid = torch.sigmoid(obj)
            print(f"Level {lvl} (stride {stride}): obj mean={obj_sigmoid.mean():.4f}, max={obj_sigmoid.max():.4f}")

    # Evaluate
    metrics = evaluate_model(model, dataloader, device, config.STRIDES, conf_thresh=config.CONF_THRESH)

    print("\n" + "=" * 50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nTotal Ground Truth: {metrics['total_gt']}")
    print(f"Total Predictions: {metrics['total_pred']}")
    print(f"TP: {metrics['total_tp']} | FP: {metrics['total_fp']} | FN: {metrics['total_fn']}")
    print("=" * 50)

    # Visualize
    visualize_results(dataset, model, device, config.STRIDES, conf_thresh=config.CONF_THRESH)


if __name__ == '__main__':
    main()
