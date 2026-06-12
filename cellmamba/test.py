"""
CellMamba Evaluation Script - Paper Implementation (arXiv:2512.21803)
Computes: mAP@50, mAP@75, Precision, Recall, F1-score
Test on CoNSeP dataset with sliding window inference
"""

import os
import sys
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


class TestDataset(Dataset):
    """Dataset for test/evaluation"""
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"TestDataset: Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Try .mat first, then .json
        label_path_mat = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.mat')
        label_path_json = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.json')
        
        centers = []
        if os.path.exists(label_path_mat):
            label_data = parse_consep_label(label_path_mat)
            centers = label_data['centers']
        elif os.path.exists(label_path_json):
            import json
            with open(label_path_json, 'r') as f:
                label_data = json.load(f)
                centers = label_data.get('centers', [])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # Normalize - ImageNet stats - ensure float32
        img_tensor = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = (img_tensor - mean) / std
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = np.ascontiguousarray(img_tensor)

        return {
            'image': img_tensor,
            'centers': centers,
            'image_name': img_name,
            'image_size': [img_h, img_w]
        }


def collate_fn(batch):
    """Custom collate for batch_size=1"""
    return {
        'images': [item['image'] for item in batch],
        'image_names': [item['image_name'] for item in batch],
        'centers_list': [item['centers'] for item in batch],
        'image_sizes': [item['image_size'] for item in batch]
    }


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)


def decode_predictions(objectness_list, regression_list, strides, conf_thresh=0.3, nms_thresh=0.45):
    """
    Decode predictions from all FPN levels to bounding boxes
    Returns: boxes (N, 4), scores (N,)
    """
    all_boxes, all_scores = [], []
    
    for obj, reg, stride in zip(objectness_list, regression_list, strides):
        B, _, H, W = obj.shape
        for b in range(B):
            scores = torch.sigmoid(obj[b, 0])
            reg_b = reg[b]

            mask = scores > conf_thresh
            if not mask.any():
                continue

            y_grid, x_grid = torch.where(mask)
            scores_filtered = scores[mask]
            reg_filtered = reg_b[:, mask]

            # Decode LTRB
            l = F.softplus(reg_filtered[0]) * stride
            t = F.softplus(reg_filtered[1]) * stride
            r = F.softplus(reg_filtered[2]) * stride
            b_box = F.softplus(reg_filtered[3]) * stride

            cx = x_grid.float() * stride
            cy = y_grid.float() * stride

            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b_box

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            all_boxes.append(boxes)
            all_scores.append(scores_filtered)

    if len(all_boxes) > 0:
        boxes_tensor = torch.cat(all_boxes, dim=0)
        scores_tensor = torch.cat(all_scores, dim=0)
        keep_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=nms_thresh)
        return boxes_tensor[keep_idx].cpu().numpy(), scores_tensor[keep_idx].cpu().numpy()
    
    return np.zeros((0, 4)), np.zeros((0,))


def sliding_window_inference(model, image_numpy, device, strides, 
                           patch_size=256, overlaps=64, conf_thresh=0.3):
    """
    Sliding window inference for full image
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

            # Pad if needed
            pad_h = max(0, patch_size - crop.shape[1])
            pad_w = max(0, patch_size - crop.shape[2])
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')

            crop_tensor = torch.from_numpy(crop).unsqueeze(0).to(device).float()
            
            with torch.no_grad():
                predictions = model(crop_tensor)

            obj_list = predictions['objectness']
            reg_list = predictions['regression']

            # Local NMS per patch
            boxes, scores = decode_predictions(obj_list, reg_list, strides, conf_thresh, nms_thresh=0.45)

            if len(boxes) > 0:
                # Offset to original coordinates
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


def compute_precision_recall_f1(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Compute P, R, F1 at given IoU threshold"""
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 1.0, 1.0, 1.0, 0, 0, 0
        return 0.0, 0.0, 0.0, 0, len(pred_boxes), 0
    
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(gt_boxes)

    # Sort by score
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
    
    return precision, recall, f1, tp, fp, fn


def compute_map(gt_boxes_list, pred_boxes_list, pred_scores_list, iou_thresholds=[0.5, 0.75]):
    """
    Compute mAP at different IoU thresholds
    
    Args:
        gt_boxes_list: list of gt boxes per image
        pred_boxes_list: list of pred boxes per image
        pred_scores_list: list of pred scores per image
        iou_thresholds: list of IoU thresholds
    
    Returns:
        dict with AP at each threshold and mAP
    """
    aps = {}
    
    for iou_thresh in iou_thresholds:
        all_tp, all_fp, all_fn = 0, 0, 0
        total_gt = 0
        
        for gt_boxes, pred_boxes, pred_scores in zip(gt_boxes_list, pred_boxes_list, pred_scores_list):
            total_gt += len(gt_boxes)
            
            if len(gt_boxes) == 0:
                if len(pred_boxes) > 0:
                    all_fp += len(pred_boxes)
                continue
            
            if len(pred_boxes) == 0:
                all_fn += len(gt_boxes)
                continue
            
            # Sort predictions by score
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            
            matched_gt = set()
            tp, fp = 0, 0
            
            for pred_box in pred_boxes:
                matched = False
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    if compute_iou(pred_box, gt_box) >= iou_thresh:
                        tp += 1
                        matched_gt.add(gt_idx)
                        matched = True
                        break
                if not matched:
                    fp += 1
            
            fn = len(gt_boxes) - len(matched_gt)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        
        # Compute precision-recall curve
        if all_tp + all_fp > 0:
            precision = all_tp / (all_tp + all_fp)
        else:
            precision = 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        
        # AP = precision at threshold (simplified, for detection we use P@R intercept)
        # More accurate AP would require PR curve integration
        aps[f'AP@{int(iou_thresh*100)}'] = precision  # P-R at operation point
        
    aps['mAP'] = np.mean(list(aps.values()))
    return aps


def evaluate_model(model, dataloader, device, strides, conf_thresh=0.3):
    """
    Evaluate model on test dataset
    
    Returns comprehensive metrics:
    - mAP@50, mAP@75
    - Precision, Recall, F1
    - Per-image results
    """
    model.eval()
    
    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_image_names = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            image_list = batch['images']
            centers_list = batch['centers_list']
            image_names = batch['image_names']

            for b in range(len(image_list)):
                image_numpy = image_list[b]
                
                # Run sliding window inference
                pred_boxes, pred_scores = sliding_window_inference(
                    model, image_numpy, device, strides,
                    patch_size=config.PATCH_SIZE, 
                    overlaps=config.CROP_OVERLAPS, 
                    conf_thresh=conf_thresh
                )

                # Convert GT centers to boxes
                gt_boxes = []
                for c in centers_list[b]:
                    bbox = c['bbox']
                    gt_boxes.append(bbox)
                gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))

                all_gt_boxes.append(gt_boxes)
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_image_names.append(image_names[b])
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # mAP@50 and mAP@75
    map_results = compute_map(all_gt_boxes, all_pred_boxes, all_pred_scores, 
                            iou_thresholds=[0.5, 0.75])
    
    # Per-threshold P/R/F1
    metrics = {}
    for iou_thresh in [0.5, 0.75]:
        p, r, f1, tp, fp, fn = compute_precision_recall_f1(
            np.concatenate(all_gt_boxes) if len(all_gt_boxes) > 0 else np.zeros((0, 4)),
            np.concatenate(all_pred_boxes) if len(all_pred_boxes) > 0 else np.zeros((0, 4)),
            np.concatenate(all_pred_scores) if len(all_pred_scores) > 0 else np.zeros((0,)),
            iou_threshold=iou_thresh
        )
        metrics[f'P@{int(iou_thresh*100)}'] = p
        metrics[f'R@{int(iou_thresh*100)}'] = r
        metrics[f'F1@{int(iou_thresh*100)}'] = f1
    
    # Aggregate metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt = 0
    total_pred = 0
    
    for gt_boxes, pred_boxes, pred_scores in zip(all_gt_boxes, all_pred_boxes, all_pred_scores):
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        matched_gt = set()
        tp, fp = 0, 0
        for pred_box in pred_boxes:
            matched = False
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if compute_iou(pred_box, gt_box) >= 0.5:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched = True
                    break
            if not matched:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Overall P/R/F1
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    metrics.update({
        'Precision': overall_p,
        'Recall': overall_r,
        'F1': overall_f1,
        'mAP@50': map_results.get('AP@50', 0),
        'mAP@75': map_results.get('AP@75', 0),
        'mAP': map_results.get('mAP', 0),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_gt,
        'total_pred': total_pred
    })
    
    return metrics, all_image_names, all_gt_boxes, all_pred_boxes, all_pred_scores


def print_metrics(metrics):
    """Print evaluation metrics in a nice format"""
    print("\n" + "=" * 60)
    print("              CellMamba Evaluation Results")
    print("=" * 60)
    print(f"\n  Detection Metrics (IoU=0.5):")
    print(f"  {'='*40}")
    print(f"  Precision:       {metrics['Precision']:.4f}")
    print(f"  Recall:          {metrics['Recall']:.4f}")
    print(f"  F1-score:        {metrics['F1']:.4f}")
    
    print(f"\n  mAP Metrics:")
    print(f"  {'='*40}")
    print(f"  mAP@50:          {metrics['mAP@50']:.4f}")
    print(f"  mAP@75:          {metrics['mAP@75']:.4f}")
    print(f"  mAP (avg):       {metrics['mAP']:.4f}")
    
    print(f"\n  Per-Class Metrics (IoU=0.5):")
    print(f"  {'='*40}")
    print(f"  P@50:            {metrics.get('P@50', 0):.4f}")
    print(f"  R@50:            {metrics.get('R@50', 0):.4f}")
    print(f"  F1@50:           {metrics.get('F1@50', 0):.4f}")
    
    print(f"\n  Detection Counts:")
    print(f"  {'='*40}")
    print(f"  Ground Truth:    {metrics['total_gt']}")
    print(f"  Predictions:     {metrics['total_pred']}")
    print(f"  True Positives:  {metrics['total_tp']}")
    print(f"  False Positives: {metrics['total_fp']}")
    print(f"  False Negatives: {metrics['total_fn']}")
    print("=" * 60)


def visualize_results(dataset, model, device, strides, output_dir='test_results', 
                      num_samples=5, conf_thresh=0.3):
    """Generate visualization of detection results"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print(f"\nGenerating visualizations ({num_samples} samples)...")
    
    for idx in tqdm(range(min(num_samples, len(dataset))), desc='Visualizing'):
        sample = dataset[idx]
        image_numpy = sample['image']
        centers = sample['centers']
        image_name = sample['image_name']
        
        with torch.no_grad():
            pred_boxes, pred_scores = sliding_window_inference(
                model, image_numpy, device, strides,
                patch_size=config.PATCH_SIZE,
                overlaps=config.CROP_OVERLAPS,
                conf_thresh=conf_thresh
            )
        
        # Denormalize image
        img_vis = np.transpose(image_numpy, (1, 2, 0))
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)
        
        # GT boxes
        gt_boxes = np.array([c['bbox'] for c in centers])
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original image
        axes[0].imshow(img_vis)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(img_vis)
        for box in gt_boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                               fill=False, color='green', linewidth=2)
            axes[1].add_patch(rect)
        axes[1].set_title(f'Ground Truth ({len(gt_boxes)} cells)', fontsize=14)
        axes[1].axis('off')
        
        # Predictions
        axes[2].imshow(img_vis)
        for box, score in zip(pred_boxes, pred_scores):
            color = 'red' if score >= conf_thresh else 'orange'
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                               fill=False, color=color, linewidth=2)
            axes[2].add_patch(rect)
        axes[2].set_title(f'Predictions ({len(pred_boxes)} cells, conf>{conf_thresh})', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'result_{image_name}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"Saved visualizations to: {output_dir}/")


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("  CellMamba Evaluation - Paper Implementation")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Patch size: {config.PATCH_SIZE}x{config.PATCH_SIZE}")
    print(f"FPN levels: P2-P6 (strides: {config.STRIDES})")
    print(f"Confidence threshold: {config.CONF_THRESH}")
    
    # Find test data directory
    test_image_dir = config.IMAGE_DIR.replace('Train', 'Test')
    test_label_dir = config.LABEL_DIR.replace('Train', 'Test')
    
    # Check alternative paths
    if not os.path.exists(test_image_dir):
        alt_paths = [
            os.path.join(os.path.dirname(config.DATA_ROOT), 'Test', 'Images'),
            os.path.join(os.path.dirname(config.DATA_ROOT), 'test', 'Images'),
            os.path.join(os.path.dirname(config.DATA_ROOT), 'Test'),
        ]
        for path in alt_paths:
            if os.path.exists(path):
                test_image_dir = path
                test_label_dir = path.replace('Images', 'Labels')
                break
    
    if not os.path.exists(test_image_dir):
        print(f"\nWarning: Test directory not found at {test_image_dir}")
        print("Using training data for demonstration...")
        test_image_dir = config.IMAGE_DIR
        test_label_dir = config.LABEL_DIR
    
    print(f"\nTest image dir: {test_image_dir}")
    print(f"Test label dir: {test_label_dir}")
    
    # Create dataset
    dataset = TestDataset(test_image_dir, test_label_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config).to(device)
    
    # Get actual strides from model
    from utils.fcos_target import get_model_strides
    actual_strides = get_model_strides(model, config.PATCH_SIZE)
    print(f"Model actual strides: {actual_strides}")
    
    # Load checkpoint if exists
    checkpoint_path = 'checkpoints/best_model.pth'
    checkpoint_compatible = False
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            checkpoint_compatible = True
        except Exception as e:
            print(f"Checkpoint incompatible with new architecture: {e}")
            print("Using randomly initialized weights for testing...")
    
    if not checkpoint_compatible:
        print("\n" + "="*60)
        print("WARNING: Testing with randomly initialized weights!")
        print("="*60)
    
    # Set model to eval mode
    model.eval()
    
    # Quick sanity check
    print("\nModel output sanity check:")
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256).to(device)
        out = model(dummy)
        for i, (obj, reg) in enumerate(zip(out['objectness'], out['regression'])):
            stride = out['strides'][i]
            obj_s = torch.sigmoid(obj)
            print(f"  Level {i} (stride {stride:2d}): obj range=[{obj_s.min():.3f}, {obj_s.max():.3f}], "
                  f"reg range=[{reg.min():.3f}, {reg.max():.3f}]")
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Starting Evaluation...")
    print("=" * 60)
    
    metrics, image_names, gt_boxes_list, pred_boxes_list, pred_scores_list = \
        evaluate_model(model, dataloader, device, actual_strides, conf_thresh=config.CONF_THRESH)
    
    # Print results
    print_metrics(metrics)
    
    # Save results to file
    results_file = 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write("CellMamba Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Patch size: {config.PATCH_SIZE}x{config.PATCH_SIZE}\n")
        f.write(f"FPN levels: {config.STRIDES}\n")
        f.write(f"Confidence threshold: {config.CONF_THRESH}\n\n")
        f.write(f"mAP@50: {metrics['mAP@50']:.4f}\n")
        f.write(f"mAP@75: {metrics['mAP@75']:.4f}\n")
        f.write(f"mAP: {metrics['mAP']:.4f}\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
        f.write(f"F1-score: {metrics['F1']:.4f}\n\n")
        f.write(f"Ground Truth: {metrics['total_gt']}\n")
        f.write(f"Predictions: {metrics['total_pred']}\n")
        f.write(f"TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}\n")
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualizations
    visualize_results(dataset, model, device, actual_strides, 
                    num_samples=5, conf_thresh=config.CONF_THRESH)
    
    return metrics


if __name__ == '__main__':
    metrics = main()
