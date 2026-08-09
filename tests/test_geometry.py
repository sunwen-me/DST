"""geometry.py 单元测试 + 双侧增广 (augmentation) 自洽性测试 (pytest)。

覆盖: so3 exp/log 往返、SE(3) 逆、欧拉角转换、quat_mean_rotations 的
slerp 中点性质、可微性 (se3_from_rt → transform_points 反传), 以及
double_sided_sample 的几何自洽性 (式(3)(4): 用 T_gt_virtual 变换投影前的
LDP 三维点后与 CDP 三维点 Chamfer≈0)。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证从任意目录运行 pytest 都能找到 dst_calib 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from dst_calib import geometry as geo


def _rand_rotvec(rng: np.random.Generator, max_angle: float = np.pi - 0.2) -> np.ndarray:
    """随机旋转向量: 随机轴 × [0.01, max_angle) 均匀角度。"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0.01, max_angle)


def _chamfer_sq(P: np.ndarray, Q: np.ndarray) -> float:
    """对称平方 Chamfer 距离 (numpy 暴力版, 仅测试用, 对应式(17) α=β=0.5)。"""
    d2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    return 0.5 * float(d2.min(axis=1).mean()) + 0.5 * float(d2.min(axis=0).mean())


# ------------------------------------------------------------------ so3 / se3

def test_so3_exp_log_roundtrip():
    """so3_exp/so3_log 往返: r → R → r' ≈ r (单个与 batch)。"""
    rng = np.random.default_rng(0)
    rs = np.stack([_rand_rotvec(rng) for _ in range(32)])
    r = torch.from_numpy(rs).to(torch.float64)
    r_back = geo.so3_log(geo.so3_exp(r))
    assert torch.allclose(r_back, r, atol=1e-6)
    # 单个 (无 batch 维)
    r1 = r[0]
    assert torch.allclose(geo.so3_log(geo.so3_exp(r1)), r1, atol=1e-6)


def test_so3_exp_is_rotation():
    """so3_exp 输出正交且 det=+1。"""
    rng = np.random.default_rng(1)
    r = torch.from_numpy(np.stack([_rand_rotvec(rng) for _ in range(8)]))
    R = geo.so3_exp(r.to(torch.float64))
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-8)
    assert torch.allclose(torch.linalg.det(R), torch.ones(8, dtype=torch.float64), atol=1e-8)


def test_so3_small_angle():
    """小角度分支: r→0 时 exp≈I+hat(r), log(exp(r))≈r。"""
    r = torch.tensor([1e-8, -2e-8, 5e-9], dtype=torch.float64)
    R = geo.so3_exp(r)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-7)
    assert torch.allclose(geo.so3_log(R), r, atol=1e-10)


def test_se3_roundtrip_and_inverse():
    """se3_from_rt/rt_from_se3 往返; se3_inverse: T·T^{-1}=I (torch 与 numpy)。"""
    rng = np.random.default_rng(2)
    r = torch.from_numpy(_rand_rotvec(rng)).to(torch.float64)
    t = torch.from_numpy(rng.normal(size=3)).to(torch.float64)
    T = geo.se3_from_rt(r, t)
    r2, t2 = geo.rt_from_se3(T)
    assert torch.allclose(r2, r, atol=1e-6) and torch.allclose(t2, t, atol=1e-9)
    eye4 = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(T @ geo.se3_inverse(T), eye4, atol=1e-9)
    # numpy 版
    Tn = geo.np_se3_from_rt(_rand_rotvec(rng), rng.normal(size=3))
    assert np.allclose(Tn @ geo.np_se3_inverse(Tn), np.eye(4), atol=1e-10)
    rn, tn = geo.np_rt_from_se3(Tn)
    assert np.allclose(geo.np_se3_from_rt(rn, tn), Tn, atol=1e-8)


# ---------------------------------------------------------------------- 欧拉角

def test_euler_zyx_roundtrip():
    """R_from_euler_zyx ↔ euler_zyx_from_R 往返 (pitch 处于非奇异区)。"""
    rng = np.random.default_rng(3)
    for _ in range(20):
        yaw = rng.uniform(-np.pi + 0.05, np.pi - 0.05)
        pitch = rng.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1)
        roll = rng.uniform(-np.pi + 0.05, np.pi - 0.05)
        R = geo.R_from_euler_zyx(yaw, pitch, roll)
        e = geo.euler_zyx_from_R(R)
        assert np.allclose(e, [yaw, pitch, roll], atol=1e-9)


def test_euler_torch_numpy_consistent():
    """torch euler_zyx_from_r 与 numpy euler_zyx_from_R 对同一旋转结果一致。"""
    rng = np.random.default_rng(4)
    for _ in range(10):
        R = geo.np_so3_exp(_rand_rotvec(rng, max_angle=1.2))
        r = geo.np_so3_log(R)
        e_torch = geo.euler_zyx_from_r(torch.from_numpy(r).to(torch.float64))
        e_np = geo.euler_zyx_from_R(R)
        assert np.allclose(e_torch.numpy(), e_np, atol=1e-6)


# ------------------------------------------------------------------- 旋转平均

def test_quat_mean_two_rotations_is_slerp_midpoint():
    """两旋转等权平均 = slerp 中点: R_mid = R1·exp(0.5·log(R1^T R2)) (式23 旋转分量)。"""
    rng = np.random.default_rng(5)
    for _ in range(10):
        R1 = geo.np_so3_exp(_rand_rotvec(rng, max_angle=1.5))
        R2 = geo.np_so3_exp(_rand_rotvec(rng, max_angle=1.5))
        R_mid = R1 @ geo.np_so3_exp(0.5 * geo.np_so3_log(R1.T @ R2))
        R_mean = geo.quat_mean_rotations([R1, R2])
        assert geo.rotation_error_deg(R_mean, R_mid) < 1e-3


def test_quat_mean_weighted_degenerate():
    """权重 [1,0] 时应精确返回第一个旋转; 加权结果仍为合法旋转。"""
    rng = np.random.default_rng(6)
    R1 = geo.np_so3_exp(_rand_rotvec(rng, max_angle=1.0))
    R2 = geo.np_so3_exp(_rand_rotvec(rng, max_angle=1.0))
    R_mean = geo.quat_mean_rotations([R1, R2], weights=[1.0, 0.0])
    assert geo.rotation_error_deg(R_mean, R1) < 1e-5
    R_w = geo.quat_mean_rotations([R1, R2], weights=[0.7, 0.3])
    assert np.allclose(R_w @ R_w.T, np.eye(3), atol=1e-9)
    assert geo.rotation_error_deg(R_w, R1) < geo.rotation_error_deg(R_w, R2)


# --------------------------------------------------------------------- 可微性

def test_se3_transform_points_differentiable():
    """requires_grad 经 se3_from_rt → transform_points 反传, 梯度非零且有限。"""
    torch.manual_seed(0)
    r = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float32, requires_grad=True)
    t = torch.tensor([0.5, 0.1, -0.2], dtype=torch.float32, requires_grad=True)
    pts = torch.randn(64, 3)
    T = geo.se3_from_rt(r, t)
    out = geo.transform_points(T, pts)
    loss = (out ** 2).sum()
    loss.backward()
    for g in (r.grad, t.grad):
        assert g is not None
        assert torch.isfinite(g).all()
        assert float(g.abs().sum()) > 0.0


# ----------------------------------------------------- 双侧增广 (III-A, 式2-4)

def test_sample_perturbation_bounds_and_reproducible():
    """扰动幅值符合论文 IV: 每轴 rot=5°·w, trans=0.5m·w; rng 可复现。"""
    from dst_calib.augmentation import sample_perturbation
    rot_amp = np.deg2rad(5.0) * np.array([0.6, 0.2, 0.2])
    trans_amp = 0.5 * np.array([0.6, 0.2, 0.2])
    rng = np.random.default_rng(7)
    xs = np.stack([sample_perturbation(rng=rng) for _ in range(500)])
    assert xs.shape == (500, 6)
    assert (np.abs(xs[:, :3]) <= rot_amp + 1e-12).all()
    assert (np.abs(xs[:, 3:]) <= trans_amp + 1e-12).all()
    assert (xs.std(axis=0) > 0).all()  # 各轴均非退化
    a = sample_perturbation(rng=np.random.default_rng(11))
    b = sample_perturbation(rng=np.random.default_rng(11))
    assert np.allclose(a, b)


def test_double_sided_sample_self_consistency():
    """双侧增广自洽性 (式3-4): T_gt_virtual 把投影前 LDP 三维点精确变换到
    CDP 三维点 (同一物理点集 → 逐点重合, Chamfer≈0); xi_gt 与 T_gt 一致。"""
    pytest.importorskip("dst_calib.projection")
    pytest.importorskip("dst_calib.difference_map")
    from dst_calib.augmentation import double_sided_sample
    from dst_calib.projection import virtual_camera_intrinsics

    rng = np.random.default_rng(42)
    size = (256, 512)
    K = virtual_camera_intrinsics(600.0, size)

    # 真值外参: 名义轴变换 (x_c=-y_l, y_c=-z_l, z_c=x_l) + 小偏置
    T_gt_cam_lidar = np.eye(4)
    T_gt_cam_lidar[:3, :3] = (np.array([[0.0, -1.0, 0.0],
                                        [0.0, 0.0, -1.0],
                                        [1.0, 0.0, 0.0]])
                              @ geo.np_so3_exp(np.array([0.02, -0.01, 0.03])))
    T_gt_cam_lidar[:3, 3] = [0.05, -0.02, 0.08]

    # 同一批物理点: 相机视锥内随机采样 (扰动后大部分仍在视场内)
    n = 1200
    z = rng.uniform(1.5, 5.0, n)
    x = rng.uniform(-0.30, 0.30, n) * z   # W/(2f)≈0.43, 留出扰动余量
    y = rng.uniform(-0.15, 0.15, n) * z   # H/(2f)≈0.21
    pts_cam = np.stack([x, y, z], axis=1)
    pts_lidar = geo.np_transform(geo.np_se3_inverse(T_gt_cam_lidar), pts_cam)

    out = double_sided_sample(pts_lidar.astype(np.float32),
                              pts_cam.astype(np.float32),
                              T_gt_cam_lidar, K, size, None, rng)

    # 契约键与形状
    for key in ("ldp", "cdp", "diff_map", "xi_gt", "T_gt"):
        assert key in out, f"缺少契约键 {key}"
    H, W = size
    assert out["ldp"].shape == (H, W) and out["cdp"].shape == (H, W)
    assert out["diff_map"].shape == (3, H, W)
    assert out["xi_gt"].shape == (6,)
    assert (out["ldp"] > 0).sum() > 100 and (out["cdp"] > 0).sum() > 100

    # xi_gt ↔ T_gt 一致 (式4)
    T_from_xi = geo.np_se3_from_rt(np.asarray(out["xi_gt"][:3], float),
                                   np.asarray(out["xi_gt"][3:], float))
    assert np.allclose(T_from_xi, out["T_gt"], atol=1e-5)

    # 式(3)(4) 代数一致: T_gt = T_cam · T_lidar^{-1}
    assert np.allclose(out["T_gt"],
                       out["T_cam"] @ geo.np_se3_inverse(out["T_lidar"]), atol=1e-9)

    # 核心自洽性: 投影前 LDP 三维点经 T_gt 变换后与 CDP 三维点重合
    aligned = geo.np_transform(out["T_gt"], out["points_ldp_3d"].astype(np.float64))
    P_c = out["points_cdp_3d"].astype(np.float64)
    assert aligned.shape == P_c.shape
    assert np.allclose(aligned, P_c, atol=1e-4)   # 同一物理点集 → 逐点对应
    assert _chamfer_sq(aligned, P_c) < 1e-8       # Chamfer≈0
