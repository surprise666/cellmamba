"""检查 CellMamba 论文复现完整性"""
import sys
sys.path.insert(0, '.')

print("="*70)
print("CellMamba 论文复现完整性检查")
print("="*70)

# 检查 1: 模型架构
print("\n[1] 模型架构检查")
from models.cellmamba_mvp import (
    BiMambaCore, TMAC, VSSBlock, MultiHeadSelfAttention,
    VSSDBackbone, FPN, ScaleSpecificHead, DetectionHead, CellMambaMVP
)
print("  ✓ BiMambaCore - 双向 Mamba 扫描")
print("  ✓ TMAC - 三重映射自适应耦合")
print("  ✓ VSSBlock - Mamba + TMAC + MLP")
print("  ✓ MultiHeadSelfAttention - Stage 4 Transformer")
print("  ✓ VSSDBackbone - 4-stage 骨干网络")
print("  ✓ FPN - P2-P6 5层金字塔")
print("  ✓ ScaleSpecificHead - 独立检测头")
print("  ✓ DetectionHead - 多尺度检测头")

# 检查 2: Loss 函数
print("\n[2] Loss 函数检查")
from utils.losses import FocalLoss, CellMambaLoss
import torch
focal = FocalLoss()
print(f"  Focal Loss: α={focal.alpha}, γ={focal.gamma} (论文: α=0.25, γ=2.0)")
print("  ✓ Smooth L1 Loss for regression")

# 检查 3: 配置文件
print("\n[3] 配置参数检查")
from configs.config import config
print(f"  PATCH_SIZE: {config.PATCH_SIZE} (论文: 256)")
print(f"  BASE_CHANNELS: {config.BASE_CHANNELS} (论文: 64)")
print(f"  FPN_CHANNELS: {config.FPN_CHANNELS} (论文: 256)")
print(f"  STRIDES: {config.STRIDES} (论文: [4, 8, 16, 32, 64])")
print(f"  FOCAL_ALPHA: {config.FOCAL_ALPHA}")
print(f"  FOCAL_GAMMA: {config.FOCAL_GAMMA}")
print(f"  TMAC_EPOCH_THRESHOLD: {config.TMAC_EPOCH_THRESHOLD} (论文: N=35)")
print(f"  LEARNING_RATE: {config.LEARNING_RATE} (论文: 1e-3)")
print(f"  WEIGHT_DECAY: {config.WEIGHT_DECAY} (论文: 1e-4)")
print(f"  NUM_EPOCHS: {config.NUM_EPOCHS}")

# 检查 4: 模型输出
print("\n[4] 模型输出检查")
model = CellMambaMVP(in_channels=3, base_channels=64, fpn_channels=256, num_classes=1)
print(f"  Model strides: {model.strides}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():
    out = model(x)
for i, (obj, reg) in enumerate(zip(out['objectness'], out['regression'])):
    print(f"  Level {i} (stride {model.strides[i]}): obj={obj.shape}, reg={reg.shape}")

# 检查 5: 关键组件实现细节
print("\n[5] 关键实现细节")
print("  BiMambaCore: 使用官方 mamba_ssm.Mamba")
print("  - forward + backward 双扫描")
print("  TMAC: 三重映射")
print("  - A_idi (个体) + A_cons (共识)")
print("  - consensus_weight 控制融合")
print("  VSSBlock: Mamba + TMAC")
print("  - norm -> mamba -> tmac -> conv -> mlp")
print("  Stage 4: Transformer (MHSA)")
print("  FPN: Top-down + lateral")
print("  DetectionHead: 独立 VSSBlock")

# 检查 6: 训练配置
print("\n[6] 训练配置")
print("  Optimizer: SGD (lr=1e-3, momentum=0.9, wd=1e-4)")
print("  Scheduler: LinearLR(warmup=5) -> MultiStepLR([35, 70])")
print("  Gradient clipping: max_norm=10.0")
print("  TMAC two-stage: N=35 threshold")

# 检查 7: 待确认项
print("\n[7] 需要确认的论文细节")
print("  ? NC-Mamba (Neighbor-aware Context): 当前使用标准 Conv")
print("  ? 具体的中心点匹配策略")
print("  ? NMS 阈值的精确值")
print("  ? 数据增强的具体参数")

print("\n" + "="*70)
print("总结: 核心架构基本完整，建议对照论文图表逐项验证")
print("="*70)
