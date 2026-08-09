"""CBAM 注意力模块 (Convolutional Block Attention Module)。

用于评估模块骨干各层后的特征精炼 (论文 III-C, 图7)。实现为标准 CBAM:
先通道注意力后空间注意力, 均为逐元素门控 (sigmoid)。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力: 全局平均池化 + 全局最大池化 → 共享 MLP → 相加 → sigmoid。

    M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    共享 MLP 用两个 1x1 卷积实现 (C → C/r → C), 中间 ReLU。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)  # 防止 C < r 时降为 0
        # 1x1 卷积等价于逐通道全连接, 对 avg/max 两路共享权重
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) → 注意力权重 (B,C,1,1)
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力: 通道维平均 + 通道维最大 → 拼接 → 7x7 卷积 → sigmoid。

    M_s(F) = σ( Conv7x7([AvgPool_c(F); MaxPool_c(F)]) )
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "空间注意力卷积核须为奇数"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维聚合: (B,C,H,W) → (B,2,H,W)
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return torch.sigmoid(attn)  # (B,1,H,W)


class CBAM(nn.Module):
    """完整 CBAM: F' = M_c(F) ⊗ F, F'' = M_s(F') ⊗ F'。

    参数:
        channels: 输入通道数
        reduction: 通道注意力压缩比 (默认 16)
        spatial_kernel: 空间注意力卷积核大小 (默认 7)
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)   # 通道门控
        x = x * self.spatial_attn(x)   # 空间门控
        return x
