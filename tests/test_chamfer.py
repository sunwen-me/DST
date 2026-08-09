"""式(17) Chamfer 距离单元测试 + 式(19) 位姿估计器损失冒烟测试。

覆盖:
- 对称性 (α=β 时交换 P/Q 不变)、零距离、与 numpy 暴力参考一致、分块无关性;
- 可微性: 小规模 float64 gradcheck + float32 反传梯度存在且有限;
- 截断版对离群点稳健;
- cKDTree 路径与暴力路径数值一致 (1e-5), 自动切换阈值生效, KDTree 路径梯度可回传。
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

# 保证从任意 cwd 运行 pytest 都能 import dst_calib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dst_calib.chamfer import _nn_sq_dists, chamfer_distance, truncated_chamfer  # noqa: E402


def _numpy_chamfer(P: np.ndarray, Q: np.ndarray, alpha: float, beta: float) -> float:
    """O(N·M) numpy 暴力参考实现 (式17)。"""
    d2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    return float(alpha * d2.min(axis=1).mean() + beta * d2.min(axis=0).mean())


# ---------------------------------------------------------------- 基本性质

def test_zero_distance():
    """P 与自身的 Chamfer 距离应为 0 (float32 允许 mm 展开的舍入噪声 ~1e-7)。"""
    torch.manual_seed(0)
    P = torch.rand(64, 3)
    assert float(chamfer_distance(P, P.clone())) < 1e-6
    P64 = torch.rand(64, 3, dtype=torch.float64)
    assert float(chamfer_distance(P64, P64.clone())) < 1e-14


def test_symmetry_when_alpha_eq_beta():
    """α=β=0.5 时 L_CD(P,Q) = L_CD(Q,P)。"""
    torch.manual_seed(1)
    P, Q = torch.rand(50, 3), torch.rand(80, 3)
    a = chamfer_distance(P, Q, 0.5, 0.5)
    b = chamfer_distance(Q, P, 0.5, 0.5)
    assert torch.allclose(a, b, atol=1e-6)


def test_matches_numpy_reference():
    """与 numpy 暴力实现一致 (含非对称权重 α≠β)。"""
    rng = np.random.default_rng(2)
    P = rng.standard_normal((37, 3))
    Q = rng.standard_normal((53, 3))
    ref = _numpy_chamfer(P, Q, 0.3, 0.7)
    got = float(chamfer_distance(torch.from_numpy(P), torch.from_numpy(Q), 0.3, 0.7))
    assert abs(got - ref) < 1e-9


def test_chunk_size_irrelevant():
    """分块大小不影响结果。"""
    torch.manual_seed(3)
    P = torch.rand(101, 3, dtype=torch.float64)
    Q = torch.rand(77, 3, dtype=torch.float64)
    a = chamfer_distance(P, Q, chunk=7)
    b = chamfer_distance(P, Q, chunk=100000)
    assert torch.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------- 可微性

def test_gradcheck_small():
    """小规模 float64 gradcheck: 梯度对 P、Q 均正确。"""
    torch.manual_seed(4)
    P = torch.rand(6, 3, dtype=torch.float64, requires_grad=True)
    Q = torch.rand(9, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p, q: chamfer_distance(p, q), (P, Q), eps=1e-6, atol=1e-5)


def test_grad_flows_float32():
    """float32 常规反传: 梯度存在、有限且非零。"""
    torch.manual_seed(5)
    P = torch.rand(200, 3, requires_grad=True)
    Q = torch.rand(150, 3)
    loss = chamfer_distance(P, Q)
    loss.backward()
    assert P.grad is not None
    assert torch.isfinite(P.grad).all()
    assert float(P.grad.abs().sum()) > 0


# ---------------------------------------------------------------- 截断版

def test_truncated_robust_to_outlier():
    """远处离群点使普通 Chamfer 爆炸, 截断版贡献被限制在 trunc²/|Q| 内。"""
    torch.manual_seed(6)
    base = torch.rand(100, 3)
    P = base + 1e-3 * torch.randn(100, 3)
    Q_out = torch.cat([base, torch.tensor([[100.0, 100.0, 100.0]])])  # 加一个离群点
    plain = float(chamfer_distance(P, Q_out))
    trunc_val = float(truncated_chamfer(P, Q_out, trunc=0.5))
    assert plain > 50.0            # 离群点主导普通 Chamfer
    assert trunc_val < 0.01        # 截断后离群点最多贡献 β·trunc²/|Q| ≈ 1.2e-3
    assert trunc_val <= plain


def test_truncated_equals_plain_for_large_trunc():
    """截断阈值远大于所有距离时退化为标准 Chamfer。"""
    torch.manual_seed(7)
    P, Q = torch.rand(40, 3), torch.rand(30, 3)
    a = chamfer_distance(P, Q)
    b = truncated_chamfer(P, Q, trunc=1e6)
    assert torch.allclose(a, b, atol=1e-7)


def test_truncated_grad_zero_beyond_trunc():
    """完全被截断的点对不产生梯度 (离群点梯度为 0)。"""
    P = torch.tensor([[100.0, 0.0, 0.0]], requires_grad=True)  # 唯一点即离群点
    Q = torch.zeros(5, 3)
    loss = truncated_chamfer(P, Q, trunc=0.5)
    loss.backward()
    assert float(P.grad.abs().sum()) == 0.0


# ---------------------------------------------------------------- KDTree 路径

def test_kdtree_matches_bruteforce():
    """强制 KDTree vs 强制暴力: 最近邻平方距离逐点一致 (1e-5)。"""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(8)
    A = torch.from_numpy(rng.standard_normal((500, 3)))
    B = torch.from_numpy(rng.standard_normal((400, 3)))
    d_kd = _nn_sq_dists(A, B, use_kdtree=True)
    d_bf = _nn_sq_dists(A, B, use_kdtree=False)
    assert torch.allclose(d_kd, d_bf, atol=1e-5)


def test_kdtree_autopath_large_cloud():
    """N*M = 2500*2000 = 5e6 > 4e6 → CPU 自动走 KDTree, 结果与暴力一致 (1e-5)。"""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(9)
    P = torch.from_numpy(rng.standard_normal((2500, 3)))
    Q = torch.from_numpy(rng.standard_normal((2000, 3)))
    auto_val = chamfer_distance(P, Q)  # 自动选择 → KDTree 路径
    ref = (0.5 * _nn_sq_dists(P, Q, use_kdtree=False).mean()
           + 0.5 * _nn_sq_dists(Q, P, use_kdtree=False).mean())
    assert torch.allclose(auto_val, ref, atol=1e-5)


def test_kdtree_grad_flows():
    """KDTree 路径梯度仍经距离项回传到 A 与 B。"""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(10)
    A = torch.from_numpy(rng.standard_normal((50, 3))).requires_grad_(True)
    B = torch.from_numpy(rng.standard_normal((40, 3))).requires_grad_(True)
    d = _nn_sq_dists(A, B, use_kdtree=True)
    d.mean().backward()
    for g in (A.grad, B.grad):
        assert g is not None
        assert torch.isfinite(g).all()
    assert float(A.grad.abs().sum()) > 0


# ---------------------------------------------------------------- 式(19) 冒烟

def test_pose_estimator_loss_smoke():
    """pose_estimator_loss: 分项正确、可反传 (式19 集成 chamfer)。"""
    from dst_calib.losses import pose_estimator_loss

    torch.manual_seed(11)
    P = torch.rand(120, 3, dtype=torch.float64)
    Q = P.clone()  # ξ=0 时 P̂=P=Q → L_CD≈0
    xi = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    xi_init = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # 仅平移差 0.1m
    total, parts = pose_estimator_loss(xi, P, Q, xi_init=xi_init, trunc=1.0)
    assert set(parts) == {"cd", "tini", "eva"}
    assert parts["cd"] < 1e-8
    assert abs(parts["tini"] - 0.1) < 1e-6  # L_t_ini = ||t_init - t||_2
    assert parts["eva"] == 0.0              # 无评估模块 → 0
    total.backward()
    assert xi.grad is not None and torch.isfinite(xi.grad).all()
