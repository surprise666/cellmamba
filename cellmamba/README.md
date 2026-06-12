# CellMamba-MVP

Pure regression-based cell detection using VSSD (Mamba) backbone with standard FPN.

## Project Structure

```
cellmamba/
├── configs/
│   └── config.py          # Configuration settings
├── models/
│   └── cellmamba_mvp.py   # Model architecture
├── utils/
│   ├── dataset.py         # CoNSeP dataset loader
│   ├── fcos_target.py     # FCOS target assignment
│   ├── losses.py          # Focal + GIoU losses
│   └── visualization.py   # Debug visualization tools
├── train.py               # Training script
├── test_overfitting.py    # Single image overfitting test
├── evaluate.py            # Evaluation script
└── inference.py           # Inference script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit `configs/config.py` to set your data paths:
- `DATA_ROOT`: Path to CoNSeP dataset
- `PRETRAINED_PATH`: Path to VSSD pretrained weights

### 3. Single Image Overfitting Test (Critical!)

Before training, verify the model works:

```bash
python test_overfitting.py
```

This runs 500 iterations on a single image. Expected:
- Loss should converge to near 0
- Predicted boxes should match ground truth boxes

### 4. Train the Model

```bash
python train.py
```

### 5. Evaluate

```bash
python evaluate.py
```

### 6. Run Inference

```bash
python inference.py
```

## Architecture

### Backbone: VSSD (Mamba)
- Loads pretrained weights from `vssd_micro_best.pth`
- Outputs 4 feature maps at strides: 8, 16, 32, 64
- No additional attention mechanisms

### Neck: Standard FPN
- 1x1 convolutions to unify channels to 256
- Top-down feature fusion with upsampling
- Clean implementation without adaptive coupling

### Head: Minimal Detection Head
- Objectness branch: 1 channel (foreground probability)
- Regression branch: 4 channels (dx, dy, log_w, log_h)

## Target Assignment (FCOS Style)

- Center sampling with 3x3 expansion
- Radius: 1.5 × stride
- Safety check: positive samples must be inside ground truth bbox
- All cell types forced to 0 (foreground)

## Loss Functions

- **Objectness**: Focal Loss (α=0.25, γ=2.0)
- **Regression**: GIoU Loss (computed only on positive samples)

## Dataset

CoNSeP dataset format:
- Images: PNG files
- Labels: MAT files with instance segmentation
- All cell types forced to 0 (foreground only)

## Evaluation Metrics

- Precision
- Recall
- F1 Score at IoU=0.5
- Localization recall

## Notes

- Initial implementation uses NO data augmentation
- Training patch size: 128×128
- Batch size: 8
- Learning rate: 1e-3
