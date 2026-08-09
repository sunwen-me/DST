"""Chamfer 距离 (论文式(17)) — torch 可微实现, 分块暴力 + cKDTree 大点云加速。

式(17):
    L_CD = α/|P̂| Σ_{p̂∈P̂} min_{q∈Q} ||p̂-q||²
         + β/|Q| Σ_{q∈Q} min_{p̂∈P̂} ||q-p̂||²          (默认 α=β=0.5)

实现要点:
- 暴力路径: 对 P 每 chunk 行与全 Q 用展开式 ||a-b||² = ||a||²+||b||²-2a·b
  求平方距离并取行最小, 全程无 sqrt —— 在零距离处梯度亦良定 (∂d²/∂a = 2(a-b)),
  且避免构造完整 N×M 矩阵。
- CPU 大点云路径: 当两输入均在 CPU 且 |P|·|Q| > 4e6 时, 用 scipy.spatial.cKDTree
  在 detach 后的 numpy 数据上查最近邻 *索引*, 再用 torch 按索引 gather 计算
  平方距离 —— 梯度仍经距离项 ||a - b_idx||² 回传到 P 与 Q。
- 设备/精度跟随输入张量 (CPU/CUDA, float32/float64 均可)。
"""
from __future__ import annotations

import numpy as np
import torch

try:  # scipy 仅用于 CPU 大点云加速; 缺失时自动退回分块暴力
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None
    _HAS_SCIPY = False

# CPU 上启用 KDTree 最近邻检索的点对数阈值 (|P|*|Q|)
_KDTREE_MIN_PAIRS = 4.0e6


def _nn_sq_dists(A: torch.Tensor, B: torch.Tensor, chunk: int = 4096,
                 use_kdtree: bool | None = None) -> torch.Tensor:
    """对 A (N,3) 中每点求到 B (M,3) 的最近邻平方距离, 返回 (N,) 可微张量。

    use_kdtree:
        None  → 自动选择: CPU 且 N*M > 4e6 且 scipy 可用时走 KDTree, 否则暴力;
        True  → 强制 KDTree (测试用);
        False → 强制分块暴力 (测试用)。

    KDTree 只用来确定最近邻索引 (对 detach 数据), 平方距离用 torch 重新计算,
    因此梯度对 A、B 中带 requires_grad 的一方均可回传。
    """
    N, M = int(A.shape[0]), int(B.shape[0])
    if use_kdtree is None:
        use_kdtree = (_HAS_SCIPY and A.device.type == "cpu"
                      and B.device.type == "cpu"
                      and float(N) * float(M) > _KDTREE_MIN_PAIRS)
    if use_kdtree:
        if not _HAS_SCIPY:  # pragma: no cover
            raise RuntimeError("scipy 不可用, 无法使用 cKDTree 最近邻路径")
        tree = cKDTree(B.detach().cpu().numpy())
        try:
            _, idx = tree.query(A.detach().cpu().numpy(), k=1, workers=-1)
        except TypeError:  # pragma: no cover — 旧版 scipy 无 workers 参数
            _, idx = tree.query(A.detach().cpu().numpy(), k=1)
        idx_t = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=A.device)
        diff = A - B.index_select(0, idx_t)  # 可微: 对 A 与 B 均回传
        return (diff * diff).sum(dim=-1)
    # ---------------- 分块暴力路径 ----------------
    chunk = max(int(chunk), 1)
    b_sq = (B * B).sum(dim=-1)  # (M,)
    out = []
    for i in range(0, N, chunk):
        a = A[i:i + chunk]  # (c,3)
        a_sq = (a * a).sum(dim=-1, keepdim=True)  # (c,1)
        # ||a-b||² = ||a||² + ||b||² - 2 a·b  (无 sqrt, 零距离处梯度良定)
        d2 = a_sq + b_sq.unsqueeze(0) - 2.0 * (a @ B.transpose(0, 1))  # (c,M)
        d2 = d2.clamp_min(0.0)  # 浮点舍入可能产生微小负值
        out.append(d2.min(dim=1).values)
    return torch.cat(out, dim=0)


def chamfer_distance(P: torch.Tensor, Q: torch.Tensor,
                     alpha: float = 0.5, beta: float = 0.5,
                     chunk: int = 4096) -> torch.Tensor:
    """式(17) 标准双向 Chamfer 距离。

    L_CD = α/|P| Σ_{p∈P} min_{q∈Q} ||p-q||² + β/|Q| Σ_{q∈Q} min_{p∈P} ||q-p||²

    参数:
        P: (N,3) torch 张量;  Q: (M,3) torch 张量 (设备/精度跟随输入)。
        alpha, beta: 双向权重, 论文默认 α=β=0.5。
        chunk: 暴力路径的分块行数, 避免一次构造 N×M 矩阵。
    返回: 标量张量, 可微 (梯度经最近对平方距离回传)。
    空点云时按约定返回 0。
    """
    if P.numel() == 0 or Q.numel() == 0:
        # 空点云: 返回 0, 同时保持梯度图与设备/精度
        return P.sum() * 0.0 + Q.sum() * 0.0
    d2_pq = _nn_sq_dists(P, Q, chunk=chunk)
    d2_qp = _nn_sq_dists(Q, P, chunk=chunk)
    return alpha * d2_pq.mean() + beta * d2_qp.mean()


def truncated_chamfer(P: torch.Tensor, Q: torch.Tensor,
                      alpha: float = 0.5, beta: float = 0.5,
                      trunc: float = 1.0, chunk: int = 4096) -> torch.Tensor:
    """截断版式(17): 两个方向的最近对平方距离均以 trunc² 为上限 clamp 后再平均。

    超过截断阈值的点对 (离群点、非重叠区域) 只贡献常数 trunc², 其梯度为 0,
    从而提高对离群点的稳健性。自监督位姿优化默认使用本函数。
    """
    if P.numel() == 0 or Q.numel() == 0:
        return P.sum() * 0.0 + Q.sum() * 0.0
    t2 = float(trunc) ** 2
    d2_pq = _nn_sq_dists(P, Q, chunk=chunk).clamp_max(t2)
    d2_qp = _nn_sq_dists(Q, P, chunk=chunk).clamp_max(t2)
    return alpha * d2_pq.mean() + beta * d2_qp.mean()
