"""
CellMamba: Adaptive Mamba for Accurate and Efficient Cell Detection
Based on paper: arXiv:2512.21803v1 [cs.CV] 25 Dec 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SS2D(nn.Module):
    """
    Selective State Space Model (SSM) - Core Mamba Operation

    From paper: Mamba uses selective state space models for efficient long-range modeling
    The discretized SSM: x_k = A @ x_{k-1} + B @ u_k, y_k = C @ x_k
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * dim)

        # Input projection (selective)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # Conv for local context
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner
        )

        # State space parameters (selective, input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # dt, B, C

        # Learnable parameters
        # A: (d_inner, d_state) - state transition matrix
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.act = nn.SiLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Input projection and split
        xz = self.in_proj(x_flat)  # (B, N, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, N, d_inner) each

        # Local conv
        x_conv = x_inner.transpose(1, 2).reshape(B, self.d_inner, H, W)
        x_conv = self.conv2d(x_conv)
        x_conv = x_conv.flatten(2).transpose(1, 2)  # (B, N, d_inner)

        # Selective SSM parameters
        x_dbl = self.x_proj(x_conv)  # (B, N, d_state*2 + 1)

        # Split: dt (input gate), B (state input), C (state output)
        dt = x_dbl[:, :, :self.d_state]  # (B, N, d_state)
        B_state = x_dbl[:, :, self.d_state:self.d_state*2]  # (B, N, d_state)
        C_state = x_dbl[:, :, self.d_state*2:]  # (B, N, d_state)

        # Discretize dt: delta_t = softplus(dt)
        dt = self.softplus(dt)  # (B, N, d_state)

        # Discretize A: A_discrete = exp(delta_t * A)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Expand dt for batch dimension
        # dt: (B, N, d_state), A: (d_inner, d_state)
        # Result: for each batch and position, compute exp(dt * A)
        # dt: (B, N, d_state) -> transpose -> (B, d_state, N)
        # A: (d_inner, d_state) -> transpose -> (d_state, d_inner)
        # Result: exp(dt @ A) -> (B, N, d_inner)
        A_t = A.t()  # (d_state, d_inner)
        A_discrete = torch.exp(torch.einsum('bnd,de->bne', dt, A_t))  # (B, N, d_inner)

        # Selective scan (simplified for 2D)
        # Gate the input based on dt
        gate = torch.sigmoid(dt.mean(dim=-1, keepdim=True))  # (B, N, 1)

        # SSM-like transformation
        ssm_out = x_conv * gate * self.D

        # Gate with z
        output = self.out_proj(ssm_out * torch.sigmoid(z))

        return output.transpose(1, 2).reshape(B, C, H, W)
        
        return output.transpose(1, 2).reshape(B, C, H, W)


class SpatialAttention(nn.Module):
    """CBAM-style Spatial Attention for TMAC"""
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
    """
    Triple-Mapping Adaptive Coupling (TMAC) Module
    
    From paper:
    - Splits channels into two branches X1, X2
    - Generates two idiosyncratic attention maps Aidi_m and one consensus attention map Acons
    - Adaptive coupling: Am = Aidi_m ⊙ Acons (gated element-wise multiplication)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.spatial_attn = SpatialAttention()
        
        # Shared weights for all three attention branches (as per paper)
        # All three use the same spatial attention structure
    
    def forward(self, F1, F2, consensus_weight=1.0):
        """
        Args:
            F1, F2: Two feature branches (after channel split)
            consensus_weight: 0 = only idiosyncratic, 1 = full coupling
        Returns:
            Fused features from both branches
        """
        # Idiopathic attention maps (one per branch)
        A_idi_1 = self.spatial_attn(F1)  # (B, 1, H, W)
        A_idi_2 = self.spatial_attn(F2)
        
        # Consensus attention map (from sum of both branches)
        F_cons = F1 + F2
        A_cons = self.spatial_attn(F_cons)
        
        # Adaptive coupling: Am = A_idi_m ⊙ Acons (equation 5)
        # During first N epochs: Acons = all-ones (disabled)
        # After epoch N: progressive coupling
        if consensus_weight < 1.0:
            # Blend between pure idiosyncratic and full coupling
            A_cons_blended = A_cons * consensus_weight + (1 - consensus_weight)
        else:
            A_cons_blended = A_cons
        
        A1 = A_idi_1 * A_cons_blended
        A2 = A_idi_2 * A_cons_blended
        
        # Feature fusion with attention (equation 6)
        # Broadcasting: (B,1,H,W) * (B,C/2,H,W) -> (B,C/2,H,W)
        F_final_1 = A1 * F1
        F_final_2 = A2 * F2
        
        return torch.cat([F_final_1, F_final_2], dim=1)


class NCMambaBlock(nn.Module):
    """
    Non-Causal Mamba Block with TMAC
    
    From paper: Sequence modeling (NC-Mamba) followed by TMAC, then FFN
    """
    def __init__(self, channels, d_state=16, use_tmac=True, tmac_epoch_threshold=35):
        super().__init__()
        self.channels = channels
        self.use_tmac = use_tmac
        self.tmac_epoch_threshold = tmac_epoch_threshold
        self.current_epoch = 0
        
        # Layer norm
        self.norm = nn.LayerNorm(channels)
        
        # Channel splitting: split into two halves
        half_channels = channels // 2
        
        # TMAC module
        if use_tmac:
            self.tmac = TMAC(half_channels)
        
        # SS2D for each branch
        self.ss2d_1 = SS2D(half_channels, d_state=d_state)
        self.ss2d_2 = SS2D(half_channels, d_state=d_state)
        
        # FFN (equation 7: after TMAC, feed into LN and FFN)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Flatten for LN
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        
        # Channel split
        x1, x2 = x_norm.chunk(2, dim=-1)
        
        # SS2D for each branch
        x1 = x1.transpose(1, 2).reshape(B, C//2, H, W)
        x2 = x2.transpose(1, 2).reshape(B, C//2, H, W)
        
        x1 = self.ss2d_1(x1)
        x2 = self.ss2d_2(x2)
        
        # TMAC
        if self.use_tmac:
            # Two-stage training (from paper)
            if self.training and self.current_epoch < self.tmac_epoch_threshold:
                consensus_weight = 0.0
            elif self.training:
                progress = (self.current_epoch - self.tmac_epoch_threshold) / 10.0
                consensus_weight = min(1.0, progress)
            else:
                consensus_weight = 1.0
            
            tmac_out = self.tmac(x1, x2, consensus_weight=consensus_weight)
            x_cat = tmac_out
        else:
            x_cat = torch.cat([x1, x2], dim=1)
        
        # FFN
        x_ffn = x_cat.flatten(2).transpose(1, 2)
        x_ffn = self.ffn(x_ffn + x_flat)  # Residual
        x_ffn = x_ffn.transpose(1, 2).reshape(B, C, H, W)
        
        return x_ffn + x  # Residual connection


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MSA) for Stage 4
    
    From paper: Final stage adopts MSA to enhance global contextual modeling
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        
        # TMAC for MSA stage
        self.use_tmac = True
        self.tmac = TMAC(channels // 2)
        self.tmac_epoch_threshold = 35
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Residual path with norm
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        
        # MSA
        x_flat = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        
        # Reshape for TMAC
        x_msa = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # TMAC (if enabled)
        if self.use_tmac:
            if self.training and self.current_epoch < self.tmac_epoch_threshold:
                consensus_weight = 0.0
            elif self.training:
                progress = (self.current_epoch - self.tmac_epoch_threshold) / 10.0
                consensus_weight = min(1.0, progress)
            else:
                consensus_weight = 1.0
            
            x1, x2 = x_msa[:, :C//2], x_msa[:, C//2:]
            tmac_out = self.tmac(x1, x2, consensus_weight=consensus_weight)
            return tmac_out + x
        else:
            return x_msa.transpose(1, 2).reshape(B, C, H, W) + x


class CellMambaBlock(nn.Module):
    """
    CellMamba Block: Combined structure of sequence modeling + TMAC
    
    From paper:
    - Stage 1-3: NC-Mamba + TMAC
    - Stage 4: MSA + TMAC
    """
    def __init__(self, channels, block_type='mamba', d_state=16, use_tmac=True):
        super().__init__()
        self.block_type = block_type
        self.channels = channels
        
        if block_type == 'mamba':
            self.block = NCMambaBlock(channels, d_state=d_state, use_tmac=use_tmac)
        else:  # MSA
            self.block = MultiHeadSelfAttention(channels)
        
        self.use_tmac = use_tmac
    
    def set_epoch(self, epoch):
        self.block.set_epoch(epoch)
    
    def forward(self, x):
        return self.block(x)


class VSSDBackbone(nn.Module):
    """
    VSSD Backbone with Mixed Mamba + Transformer
    
    From paper:
    - Stage 1: 2 blocks (Mamba)
    - Stage 2: 2 blocks (Mamba)
    - Stage 3: 8 blocks (Mamba)
    - Stage 4: 4 blocks (MSA)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        
        # Stage 1: 2 Mamba blocks (stride 4)
        self.stage1 = nn.ModuleList([
            CellMambaBlock(base_channels, 'mamba', use_tmac=True) for _ in range(2)
        ])
        
        # Stage 2: 2 Mamba blocks + downsample (stride 8)
        self.stage2_down = nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False)
        self.stage2 = nn.ModuleList([
            CellMambaBlock(base_channels * 2, 'mamba', use_tmac=True) for _ in range(2)
        ])
        
        # Stage 3: 8 Mamba blocks + downsample (stride 16)
        self.stage3_down = nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False)
        self.stage3 = nn.ModuleList([
            CellMambaBlock(base_channels * 4, 'mamba', use_tmac=True) for _ in range(8)
        ])
        
        # Stage 4: 4 MSA blocks + downsample (stride 32)
        self.stage4_down = nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False)
        self.stage4 = nn.ModuleList([
            CellMambaBlock(base_channels * 8, 'msa', use_tmac=True) for _ in range(4)
        ])
        
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
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def set_epoch(self, epoch):
        """Propagate epoch to all blocks for two-stage training"""
        for block in self.stage1:
            block.set_epoch(epoch)
        for block in self.stage2:
            block.set_epoch(epoch)
        for block in self.stage3:
            block.set_epoch(epoch)
        for block in self.stage4:
            block.set_epoch(epoch)
    
    def forward(self, x):
        outputs = []
        
        # Stem
        x = self.stem(x)  # stride 2
        
        # Stage 1
        for block in self.stage1:
            x = block(x)
        outputs.append(x)  # stride 4, C=64
        
        # Stage 2
        x = self.stage2_down(x)
        for block in self.stage2:
            x = block(x)
        outputs.append(x)  # stride 8, C=128
        
        # Stage 3
        x = self.stage3_down(x)
        for block in self.stage3:
            x = block(x)
        outputs.append(x)  # stride 16, C=256
        
        # Stage 4
        x = self.stage4_down(x)
        for block in self.stage4:
            x = block(x)
        outputs.append(x)  # stride 32, C=512
        
        return outputs  # [stride4, stride8, stride16, stride32]


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN)
    
    From paper: Outputs from stages L2–L4 are fused into five feature maps P2-P6
    - P2: stride 4 (from stage 1)
    - P3: stride 8 (from stage 2)
    - P4: stride 16 (from stage 3)
    - P5, P6: derived from P4
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # in_channels_list: [C4, C8, C16, C32] from backbone
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral connections for P2, P3, P4 (from stage 1, 2, 3)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list[:3]
        ])
        
        # Output convs for P2, P3, P4
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in range(3)
        ])
        
        # P5: downsample from P4
        self.p5_down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        
        # P6: downsample from P5
        self.p6_down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, backbone_features):
        """
        backbone_features: [stride4, stride8, stride16, stride32]
        Returns: [P2, P3, P4, P5, P6]
        """
        C2, C3, C4, C5 = backbone_features[:4]
        
        # Lateral connections
        P4_lat = self.lateral_convs[2](C4)  # stride 16 -> 256
        P3_lat = self.lateral_convs[1](C3)  # stride 8 -> 256
        P2_lat = self.lateral_convs[0](C2)  # stride 4 -> 256
        
        # Top-down pathway
        P4 = P4_lat
        P3 = P3_lat + F.interpolate(P4, size=P3_lat.shape[2:], mode='nearest')
        P2 = P2_lat + F.interpolate(P3, size=P2_lat.shape[2:], mode='nearest')
        
        # Apply output convs
        P4 = self.fpn_convs[2](P4)
        P3 = self.fpn_convs[1](P3)
        P2 = self.fpn_convs[0](P2)
        
        # P5 and P6 (from paper: derived from P4)
        P5 = self.p5_down(F.relu(P4))
        P6 = self.p6_down(F.relu(P5))
        
        return [P2, P3, P4, P5, P6]  # 5 levels


class AdaptiveMambaHead(nn.Module):
    """
    Adaptive Mamba Head
    
    From paper:
    - FPN outputs {Pi} aggregate to x with learnable weights
    - Dual pooling: spatial then channel
    - Dynamic weight mechanism via FC
    - Each level processed independently with CellMamba block
    """
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Adaptive scale weights (from paper equations)
        self.scale_fc = nn.Linear(5, 5)  # T=5 levels
        
        # CellMamba block for each classification and regression branch (per level)
        self.obj_blocks = nn.ModuleList([
            CellMambaBlock(in_channels, 'mamba', use_tmac=True) for _ in range(5)
        ])
        self.reg_blocks = nn.ModuleList([
            CellMambaBlock(in_channels, 'mamba', use_tmac=True) for _ in range(5)
        ])
        
        # Output convs (per level, independent as per paper)
        self.obj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, num_classes, 1)
            ) for _ in range(5)
        ])
        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 4, 1)  # LTRB
            ) for _ in range(5)
        ])
        
        self.sigmoid = nn.Sigmoid()
    
    def set_epoch(self, epoch):
        for block in self.obj_blocks:
            block.set_epoch(epoch)
        for block in self.reg_blocks:
            block.set_epoch(epoch)
    
    def forward(self, p_features):
        """
        p_features: [P2, P3, P4, P5, P6] from FPN
        Returns: objectness and regression for each level
        """
        T = len(p_features)
        B = p_features[0].shape[0]
        
        # Adaptive scale weights (from paper equations)
        # 1. Global pooling in spatial dimension
        spatial_pooled = []
        for p in p_features:
            pool = torch.mean(p, dim=[2, 3])  # (B, C)
            spatial_pooled.append(pool)
        
        # 2. Secondary pooling in channel dimension
        channel_pooled = []
        for sp in spatial_pooled:
            cp = torch.mean(sp, dim=1, keepdim=True)  # (B, 1)
            channel_pooled.append(cp)
        
        s_t = torch.cat(channel_pooled, dim=1)  # (B, T=5)
        
        # 3. FC to generate weights (equation: αt = Sigmoid(FC(st)))
        alpha = self.sigmoid(self.scale_fc(s_t))  # (B, 5)
        
        objectness_list = []
        regression_list = []
        
        # Process each level independently (as per paper)
        for t in range(T):
            feat = p_features[t]
            
            # Apply scale weight αt * Pi
            w = alpha[:, t].view(B, 1, 1, 1)
            feat_weighted = feat * w
            
            # Classification branch (equation 8)
            obj = self.obj_blocks[t](feat_weighted)
            obj = self.obj_convs[t](obj)
            objectness_list.append(obj)
            
            # Regression branch (equation 9)
            reg = self.reg_blocks[t](feat_weighted)
            reg = self.reg_convs[t](reg)
            regression_list.append(reg)
        
        return objectness_list, regression_list


class CellMamba(nn.Module):
    """
    Complete CellMamba Architecture
    
    From paper Figure 1:
    - Four-stage mixed Mamba-Transformer hierarchical backbone
    - FPN for multi-scale feature fusion
    - Adaptive Mamba Head for classification and box regression
    """
    def __init__(self, in_channels=3, base_channels=64, fpn_channels=256, num_classes=1):
        super().__init__()
        
        # Backbone
        self.backbone = VSSDBackbone(in_channels, base_channels)
        
        # FPN (P2-P6 as per paper)
        backbone_out_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.fpn = FPNNeck(backbone_out_channels, fpn_channels)
        
        # Adaptive Mamba Head
        self.head = AdaptiveMambaHead(fpn_channels, num_classes)
        
        # Strides: P2=stride4, P3=stride8, P4=stride16, P5=stride32, P6=stride64
        self.strides = [4, 8, 16, 32, 64]
    
    def load_pretrained(self, pretrained_path):
        self.backbone.load_pretrained(pretrained_path)
    
    def set_epoch(self, epoch):
        """Update epoch for two-stage TMAC training"""
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
    """Build CellMamba model from config"""
    model = CellMamba(
        in_channels=3,
        base_channels=config.BASE_CHANNELS,
        fpn_channels=config.FPN_CHANNELS,
        num_classes=config.NUM_CLASSES
    )
    return model
