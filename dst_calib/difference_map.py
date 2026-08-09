"""差分图构建 — 论文 式(11)(12)。

式(11): Δ(u,v) = LDP(u,v) - CDP(u,v)，仅在两幅深度投影同时有效
(LDP>0 且 CDP>0) 的像素计算，其余像素 Δ 记 0。
式(12): 差分图 D = ( LDP, [Δ]_+^{e_tar}, [Δ]_-^{e_tar} )，其中
    [Δ]_+ = Δ if |Δ| >  e_tar else 0   （大误差通道）
    [Δ]_- = Δ if |Δ| <= e_tar else 0   （小误差通道）
两个差分通道非零位置互斥；第一通道保留完整 LDP（含 CDP 无效处）。
numpy 版供数据准备，torch 版供训练/评估模块前向（保持可微，掩码
本身不反传但通道值对 ldp/cdp 可导）。
"""
from __future__ import annotations

import numpy as np
import torch


def build_difference_map(ldp: np.ndarray, cdp: np.ndarray, e_tar: float = 0.1
                         ) -> np.ndarray:
    """式(11)(12) numpy 版: (H,W)+(H,W) → 差分图 (3,H,W) float32。

    通道 0: 原始 LDP; 通道 1: [Δ]_+^{e_tar}; 通道 2: [Δ]_-^{e_tar}。
    Δ 仅在 LDP>0 且 CDP>0 的共同有效像素计算，其余像素两个差分通道为 0。
    """
    ldp = np.asarray(ldp, dtype=np.float32)
    cdp = np.asarray(cdp, dtype=np.float32)
    if ldp.shape != cdp.shape:
        raise ValueError(f"LDP/CDP 形状不一致: {ldp.shape} vs {cdp.shape}")

    # 式(11): 共同有效像素处的深度残差
    valid = (ldp > 0) & (cdp > 0)
    delta = np.where(valid, ldp - cdp, 0.0).astype(np.float32)

    # 式(12): 按阈值 e_tar 分裂为大/小误差两个互斥通道
    big = valid & (np.abs(delta) > e_tar)
    small = valid & (np.abs(delta) <= e_tar)
    delta_plus = np.where(big, delta, 0.0).astype(np.float32)
    delta_minus = np.where(small, delta, 0.0).astype(np.float32)

    return np.stack([ldp, delta_plus, delta_minus], axis=0)


def build_difference_map_torch(ldp: torch.Tensor, cdp: torch.Tensor,
                               e_tar: float = 0.1) -> torch.Tensor:
    """式(11)(12) torch 版: (B,H,W) → (B,3,H,W)；亦接受 (H,W) → (3,H,W)。

    与 numpy 版语义一致；掩码由 detach 的比较得到（阈值判决不可导），
    通道数值路径对 ldp/cdp 可微，供训练/评估模块前向使用。
    """
    if ldp.shape != cdp.shape:
        raise ValueError(f"LDP/CDP 形状不一致: {tuple(ldp.shape)} vs {tuple(cdp.shape)}")
    squeeze = False
    if ldp.dim() == 2:  # 单幅 (H,W) 兼容
        ldp, cdp = ldp.unsqueeze(0), cdp.unsqueeze(0)
        squeeze = True
    if ldp.dim() != 3:
        raise ValueError(f"期望 (B,H,W) 或 (H,W)，得到 {tuple(ldp.shape)}")

    zero = torch.zeros((), dtype=ldp.dtype, device=ldp.device)
    # 式(11): 仅共同有效像素
    valid = (ldp > 0) & (cdp > 0)
    delta = torch.where(valid, ldp - cdp, zero)
    # 式(12): 大/小误差互斥通道
    big = valid & (delta.abs() > e_tar)
    small = valid & (delta.abs() <= e_tar)
    delta_plus = torch.where(big, delta, zero)
    delta_minus = torch.where(small, delta, zero)

    out = torch.stack([ldp, delta_plus, delta_minus], dim=1)  # (B,3,H,W)
    return out.squeeze(0) if squeeze else out
