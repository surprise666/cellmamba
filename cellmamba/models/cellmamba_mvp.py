"""
CellMamba-MVP Model Architecture
- VSSD Backbone with TMAC (Triple Mapping Adaptive Coupling)
- Mixed Mamba + Transformer Stage 4
- Adaptive Scale Weights Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """Spatial Attention Module for TMAC"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class TMAC(nn.Module):
    """Triple Mapping Adaptive Coupling for dense cell separation"""
    def __init__(self, channels):
        super().__init__()
        self.spatial_attn = SpatialAttention()
        self.channels = channels
        
    def forward(self, F1, F2, consensus_weight=1.0):
        # consensus_weight: 渐进式开启，0=只用专属注意力，1=完整共识注意力
        A1_idi = self.spatial_attn(F1)
        A2_idi = self.spatial_attn(F2)
        
        # 共识注意力图
        F_cons = F1 + F2
        A_cons = self.spatial_attn(F_cons)
        
        # 渐进式融合：weight=0 时退化为专属注意力
        A1 = A1_idi * (1 - consensus_weight) + A1_idi * A_cons * consensus_weight
        A2 = A2_idi * (1 - consensus_weight) + A2_idi * A_cons * consensus_weight
        
        # 特征融合
        F1_final = F1 * A1
        F2_final = F2 * A2
        
        return torch.cat([F1_final, F2_final], dim=1)


class MultiHeadSelfAttention(nn.Module):
    """Transformer-style Multi-Head Self Attention for Stage 4"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_out = self.proj(x_out)
        
        return x_out.transpose(1, 2).reshape(B, C, H, W)


class ConvModule(nn.Module):
    """Basic conv + bn + relu module"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if x.shape[-1] == 1 and x.shape[-2] == 1:
            return self.relu(self.conv(x))
        return self.relu(self.bn(self.conv(x)))


class VSSBlock(nn.Module):
    """Vision State Space Block with TMAC integration"""
    def __init__(self, channels, d_state=16, use_tmac=True, tmac_epoch_threshold=35, current_epoch=0):
        super().__init__()
        self.channels = channels
        self.use_tmac = use_tmac
        self.tmac_epoch_threshold = tmac_epoch_threshold
        self.current_epoch = current_epoch
        
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)
        
        # TMAC: split channels into two branches
        half_channels = channels // 2
        self.tmac = TMAC(half_channels)
        
        # Spatial conv for each branch
        self.conv1 = nn.Conv2d(half_channels, half_channels, 3, 1, 1, groups=half_channels)
        self.conv2 = nn.Conv2d(half_channels, half_channels, 3, 1, 1, groups=half_channels)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        
        # Projection to align channel dims
        self.channel_proj = nn.Conv2d(channels, channels, 1) if channels % 2 != 0 else nn.Identity()
    
    def set_epoch(self, epoch):
        """Update epoch for two-stage training control"""
        self.current_epoch = epoch
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Initial projection
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        x_proj_flat = self.proj(x_norm)  # 🚀 Save flat version for residual
        x_proj = x_proj_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # TMAC: Split into two branches
        F1, F2 = x_proj[:, :C//2], x_proj[:, C//2:]
        
        # Two-stage training: disable consensus before threshold
        # 共识权重：epoch < 35 时为 0（无共识），之后渐进式增加到 1.0
        if self.training and self.current_epoch < self.tmac_epoch_threshold:
            consensus_weight = 0.0
        elif self.training:
            progress = (self.current_epoch - self.tmac_epoch_threshold) / 10.0
            consensus_weight = min(1.0, progress)
        else:
            consensus_weight = 1.0
        
        # Apply TMAC
        tmac_out = self.tmac(F1, F2, consensus_weight=consensus_weight)
        
        # Convolutions
        F1_conv = self.conv1(tmac_out[:, :C//2])
        F2_conv = self.conv2(tmac_out[:, C//2:])
        
        # Combine
        x_conv = torch.cat([F1_conv, F2_conv], dim=1)
        if x_conv.shape[1] != C:
            x_conv = self.channel_proj(x_conv)
        
        # MLP
        x_mlp = self.mlp(x_norm + x_proj_flat)  # 🚀 Use flat version
        x_mlp = x_mlp.transpose(1, 2).reshape(B, C, H, W)
        
        return x_conv + x_mlp


class VSSDBlock(nn.Module):
    """VSSD Stage with multiple VSS blocks"""
    def __init__(self, in_channels, out_channels, num_blocks=2, downsample=False, stage_idx=0):
        super().__init__()
        layers = []
        
        if downsample:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        first_in_channels = out_channels if downsample else in_channels
        # Stage 4 (index 3) uses Transformer instead of Mamba
        if stage_idx == 3:
            layers.append(MultiHeadSelfAttention(first_in_channels))
        else:
            layers.append(VSSBlock(first_in_channels))
        
        for _ in range(num_blocks - 1):
            if stage_idx == 3:
                layers.append(MultiHeadSelfAttention(out_channels))
            else:
                layers.append(VSSBlock(out_channels))
        
        self.blocks = nn.Sequential(*layers)
        self.downsample = downsample
    
    def forward(self, x):
        return self.blocks(x)


class VSSDBackbone(nn.Module):
    """VSSD Backbone with Mixed Mamba + Transformer"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1 & 2: Pure Mamba
        self.stage1 = nn.Sequential(VSSBlock(base_channels), VSSBlock(base_channels))
        self.stage2 = VSSDBlock(base_channels, base_channels * 2, num_blocks=2, downsample=True)
        
        # Stage 3: Mamba with TMAC
        self.stage3 = VSSDBlock(base_channels * 2, base_channels * 4, num_blocks=2, downsample=True, stage_idx=2)
        
        # Stage 4: Transformer (MSA)
        self.stage4 = VSSDBlock(base_channels * 4, base_channels * 8, num_blocks=2, downsample=True, stage_idx=3)
        
        # Stage 5: Continue with Mamba
        self.stage5 = VSSDBlock(base_channels * 8, base_channels * 16, num_blocks=2, downsample=True, stage_idx=0)
        
        self.out_channels = [
            base_channels * 4,   # stride 8
            base_channels * 8,  # stride 16
            base_channels * 16, # stride 32
        ]
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self, pretrained_path):
        try:
            # 🚀 Add weights_only=False for PyTorch 2.6 compatibility
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def set_epoch(self, epoch):
        """Propagate epoch to all VSSBlocks for two-stage training"""
        for module in self.modules():
            if isinstance(module, VSSBlock):
                module.set_epoch(epoch)
    
    def forward(self, x):
        outputs = []
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        outputs.append(x)  # stride 8
        
        x = self.stage4(x)
        outputs.append(x)  # stride 16
        
        x = self.stage5(x)
        outputs.append(x)  # stride 32
        
        return outputs


class FPNNeck(nn.Module):
    """Standard Feature Pyramid Network"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in in_channels_list
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[2:], 
                mode='nearest'
            )
        
        outputs = [fpn_conv(lat) for lat, fpn_conv in zip(laterals, self.fpn_convs)]
        
        return outputs


class AdaptiveScaleWeights(nn.Module):
    """Adaptive Scale Weights for Multi-level Feature Fusion"""
    def __init__(self, num_levels=3):
        super().__init__()
        self.fc = nn.Linear(num_levels, num_levels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p_features):
        B = p_features[0].shape[0]
        T = len(p_features)
        
        channel_pooled = []
        for p in p_features:
            # 1. 空间全局池化 (B, C, H, W) -> (B, C)
            spatial_pool = torch.mean(p, dim=[2, 3])
            # 2. 通道全局池化 (B, C) -> (B, 1)
            chan_pool = torch.mean(spatial_pool, dim=1, keepdim=True)
            channel_pooled.append(chan_pool)
            
        s_t = torch.cat(channel_pooled, dim=1)  # 拼接为 (B, T)
        
        # 生成动态权重 alpha
        alpha = self.sigmoid(self.fc(s_t))
        
        # 逐层施加权重
        weighted_features = []
        for i in range(T):
            w = alpha[:, i].view(B, 1, 1, 1)
            weighted_features.append(p_features[i] * w)
            
        return weighted_features


class DetectionHead(nn.Module):
    """Adaptive detection head with CellMamba block applied independently to each FPN level"""
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        # Adaptive Scale Weights
        self.scale_weights = AdaptiveScaleWeights(num_levels=3)
        
        # Objectness branch (共享权重，分别应用到各个尺度)
        self.obj_conv1 = ConvModule(in_channels, in_channels, 3, 1, 1)
        self.obj_mamba = VSSBlock(in_channels)
        self.obj_conv2 = ConvModule(in_channels, in_channels, 3, 1, 1)
        self.obj_dropout = nn.Dropout2d(0.1)
        self.objectness_output = nn.Conv2d(in_channels, num_classes, 1)
        
        # Regression branch (共享权重，分别应用到各个尺度)
        self.reg_conv1 = ConvModule(in_channels, in_channels, 3, 1, 1)
        self.reg_mamba = VSSBlock(in_channels)
        self.reg_conv2 = ConvModule(in_channels, in_channels, 3, 1, 1)
        self.reg_dropout = nn.Dropout2d(0.1)
        self.regression_output = nn.Conv2d(in_channels, 4, 1)
    
    def set_epoch(self, epoch):
        """Propagate epoch to VSSBlocks"""
        self.obj_mamba.set_epoch(epoch)
        self.reg_mamba.set_epoch(epoch)
    
    def forward(self, fpn_features):
        # 1. 动态加权 FPN 特征
        weighted_features = self.scale_weights(fpn_features)
        
        objectness_list = []
        regression_list = []
        
        # 2. 🚀 必须用 for 循环遍历各个不同分辨率的尺度，分别进行推理！
        for feat in weighted_features:
            # 分类分支
            obj = self.obj_conv1(feat)
            obj = self.obj_mamba(obj)
            obj = self.obj_conv2(obj)
            obj = self.obj_dropout(obj)
            objectness_list.append(self.objectness_output(obj))
            
            # 回归分支
            reg = self.reg_conv1(feat)
            reg = self.reg_mamba(reg)
            reg = self.reg_conv2(reg)
            reg = self.reg_dropout(reg)
            regression_list.append(self.regression_output(reg))
            
        return objectness_list, regression_list


class CellMambaMVP(nn.Module):
    """Complete CellMamba-MVP architecture with all upgrades"""
    def __init__(self, in_channels=3, base_channels=64, fpn_channels=256, num_classes=1):
        super().__init__()
        self.backbone = VSSDBackbone(in_channels, base_channels)
        self.fpn = FPNNeck(self.backbone.out_channels, fpn_channels)
        self.head = DetectionHead(fpn_channels, num_classes)
        
        self.strides = [8, 16, 32]
    
    def load_pretrained(self, pretrained_path):
        self.backbone.load_pretrained(pretrained_path)
    
    def set_epoch(self, epoch):
        """Update epoch for two-stage training"""
        self.backbone.set_epoch(epoch)
        self.head.set_epoch(epoch)
    
    def forward(self, x):
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)
        objectness, regression = self.head(fpn_features)
        
        return {
            'objectness': objectness,
            'regression': regression,
            'strides': self.strides
        }


def build_model(config):
    """Build CellMamba-MVP model"""
    model = CellMambaMVP(
        in_channels=3,
        base_channels=64,
        fpn_channels=config.FPN_CHANNELS,
        num_classes=config.NUM_CLASSES
    )
    return model
