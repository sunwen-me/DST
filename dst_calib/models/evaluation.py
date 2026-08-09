"""评估模块 (Evaluation Block) — 论文 III-C, 图7。

由差分图 (式11,12) 回归位姿偏差 ξ=[r,t]∈R^6:
- 骨干: 从零实现的 ResNet BasicBlock 截断骨干 (不依赖 torchvision 预训练,
  结构对应 ResNet18 的 stem + layer1..3, 通道 [64,128,256]), 每层后接 CBAM。
- 头部: BlockPoseHead 块处理 + 旋转/平移解耦回归 (III-C-3)。
提供单分支 EvaluationSB (差分图输入) 与双分支 EvaluationDB (LDP/CDP 各一支,
消融对照, 论文结论: 单分支差分图更轻且更优)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cbam import CBAM


# ------------------------------------------------------------- ResNet 基础组件

def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """3x3 卷积 (无偏置, 后接 BN)。"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock (从零实现, 与 torchvision 结构一致但完全离线):

        x → conv3x3 → BN → ReLU → conv3x3 → BN → (+shortcut) → ReLU

    stride>1 或通道变化时 shortcut 用 1x1 卷积投影。
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetCBAMBackbone(nn.Module):
    """截断 ResNet18 骨干 + 逐层 CBAM (论文 图7 特征提取部分)。

    结构: stem (7x7 s2 conv + BN + ReLU + 3x3 s2 maxpool)
          → layer1 (2×BasicBlock, 64, s1) → CBAM
          → layer2 (2×BasicBlock, 128, s2) → CBAM
          → layer3 (2×BasicBlock, 256, s2) → CBAM
    输入 (B,in_ch,256,512) → 输出特征图 (B,256,16,32)。
    输入通道数自适应 (差分图 3 通道 / 单深度投影 1 通道)。
    """

    def __init__(self, in_ch: int = 3,
                 channels: tuple = (64, 128, 256),
                 blocks: tuple = (2, 2, 2)):
        super().__init__()
        self.out_channels = channels[-1]
        # stem: 7x7 步长2 卷积 + 最大池化 (标准 ResNet 入口)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        layers = []
        prev = channels[0]
        for i, (ch, n_blk) in enumerate(zip(channels, blocks)):
            stride = 1 if i == 0 else 2  # layer1 不降采样, 其余步长 2
            layers.append(self._make_layer(prev, ch, n_blk, stride))
            layers.append(CBAM(ch))      # 每层后接 CBAM 精炼
            prev = ch
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        """堆叠 n_blocks 个 BasicBlock, 首块承担降采样/通道变换。"""
        blocks = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(x))


# ------------------------------------------------------------- 块处理 + 回归头

class BlockPoseHead(nn.Module):
    """块处理 + 位姿回归头 (论文 III-C-3, 图7 右侧)。

    特征图先自适应池化到 (n*4, n*4), 划分 n×n 网格块 (每块 4×4);
    每块经共享 3x3 卷积 + 全局平均池化压缩为向量 B_i (block_dim 维),
    全部块拼接展开为 F_p ∈ R^{n²·block_dim}, 经 FC(512)+ReLU 聚合后,
    旋转 r 与平移 t 由两个解耦 MLP 头 (FC 512→256→3) 分别回归,
    输出 ξ=[r,t]∈R^6。旋转头输出乘 0.1 缩放, 使初始预测接近单位旋转。
    """

    def __init__(self, in_ch: int, n: int = 5, block_dim: int = 128,
                 rot_scale: float = 0.1):
        super().__init__()
        self.n = n
        self.block_dim = block_dim
        self.rot_scale = rot_scale
        # 池化到 n×n 个 4×4 块
        self.pool = nn.AdaptiveAvgPool2d((n * 4, n * 4))
        # 块内共享卷积: 所有块共用同一 3x3 conv (权重共享, 逐块独立作用)
        self.block_conv = nn.Sequential(
            nn.Conv2d(in_ch, block_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(block_dim),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        # F_p (n²·block_dim) → 512 聚合
        self.fc = nn.Sequential(
            nn.Linear(n * n * block_dim, 512),
            nn.ReLU(inplace=True),
        )
        # 旋转/平移解耦头
        self.head_rot = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 3))
        self.head_trans = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 3))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, in_ch, H', W') → ξ (B, 6)。"""
        B, C = feat.shape[:2]
        n = self.n
        x = self.pool(feat)                              # (B, C, 4n, 4n)
        # 划分 n×n 个 4×4 块并展开到批维, 使共享卷积逐块独立作用(不跨块)
        x = x.reshape(B, C, n, 4, n, 4)                  # (B,C,n,4,n,4)
        x = x.permute(0, 2, 4, 1, 3, 5)                  # (B,n,n,C,4,4)
        x = x.reshape(B * n * n, C, 4, 4)                # (B·n², C, 4, 4)
        x = self.block_conv(x)                           # (B·n², block_dim, 4, 4)
        x = self.gap(x).flatten(1)                       # (B·n², block_dim) = B_i
        f_p = x.reshape(B, n * n * self.block_dim)       # F_p: 拼接展开
        h = self.fc(f_p)                                 # (B, 512)
        r = self.head_rot(h) * self.rot_scale            # 旋转向量, 缩放 0.1
        t = self.head_trans(h)                           # 平移 (米)
        return torch.cat([r, t], dim=-1)                 # ξ=[r,t] (B,6)


# ----------------------------------------------------------------- 评估模块

class EvaluationSB(nn.Module):
    """单分支评估模块 (论文主方案, 图7):

    差分图 (B,3,256,512) → ResNet+CBAM 骨干 → 特征 (B,256,16,32)
    → BlockPoseHead → ξ (B,6)。
    """

    def __init__(self, in_ch: int = 3, block_n: int = 5):
        super().__init__()
        self.backbone = ResNetCBAMBackbone(in_ch=in_ch)
        self.head = BlockPoseHead(self.backbone.out_channels, n=block_n)

    def forward(self, diff_map: torch.Tensor) -> torch.Tensor:
        """diff_map: (B,3,H,W) 差分图 (式11) → ξ (B,6)。"""
        return self.head(self.backbone(diff_map))


class EvaluationDB(nn.Module):
    """双分支评估模块 (消融对照, 论文 III-C 讨论):

    LDP、CDP 各走一条同构的单通道 ResNet+CBAM 分支, 特征在通道维拼接
    (B,512,H',W') 后经 1x1 卷积融合回 256 通道, 再过 BlockPoseHead。
    参数量约为单分支两倍 (论文结论: 单分支差分图输入更轻更优)。
    """

    def __init__(self, block_n: int = 5):
        super().__init__()
        self.branch_ldp = ResNetCBAMBackbone(in_ch=1)
        self.branch_cdp = ResNetCBAMBackbone(in_ch=1)
        c = self.branch_ldp.out_channels
        # 通道拼接后 1x1 卷积融合
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.head = BlockPoseHead(c, n=block_n)

    def forward(self, cdp: torch.Tensor, ldp: torch.Tensor) -> torch.Tensor:
        """cdp/ldp: (B,1,H,W) 深度投影图 → ξ (B,6)。"""
        f_c = self.branch_cdp(cdp)
        f_l = self.branch_ldp(ldp)
        f = self.fuse(torch.cat([f_c, f_l], dim=1))
        return self.head(f)
