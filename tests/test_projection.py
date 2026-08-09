"""projection.py 与 difference_map.py 单元测试 (pytest)。

覆盖: 内参构造、投影/反投影往返一致性、z-buffer 最小深度、范围/出界过滤、
LDP/CDP 生成、Gemini 深度反投影、式(11)(12) 差分图语义与 torch/numpy 一致性。
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dst_calib.difference_map import build_difference_map, build_difference_map_torch
from dst_calib.geometry import np_se3_from_rt, np_transform
from dst_calib.projection import (
    backproject_depth,
    depth_cloud_from_gemini,
    generate_cdp,
    generate_ldp,
    project_points_to_depth,
    virtual_camera_intrinsics,
)


# ---------------------------------------------------------------- 内参

def test_virtual_camera_intrinsics():
    # 论文 IV: (H,W)=(256,512), f=600, cx=W/2, cy=H/2
    K = virtual_camera_intrinsics()
    assert K.shape == (3, 3)
    assert K[0, 0] == 600.0 and K[1, 1] == 600.0
    assert K[0, 2] == 256.0 and K[1, 2] == 128.0
    assert K[2, 2] == 1.0 and K[0, 1] == 0.0

    K2 = virtual_camera_intrinsics(f=80.0, size=(64, 96))
    assert K2[0, 2] == 48.0 and K2[1, 2] == 32.0


# ---------------------------------------------------- 投影/反投影往返

def _pixel_ray_points(rng, K, H, W, n):
    """在随机互异像素中心射线上取随机深度构造点云，保证投影索引精确。"""
    idx = rng.choice(H * W, size=n, replace=False)
    v = idx // W
    u = idx % W
    z = rng.uniform(0.5, 10.0, size=n).astype(np.float32)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, u, v, z


def test_project_backproject_roundtrip():
    rng = np.random.default_rng(0)
    H, W = 64, 96
    K = virtual_camera_intrinsics(f=80.0, size=(H, W))
    n = 200
    pts, u, v, z = _pixel_ray_points(rng, K, H, W, n)

    depth = project_points_to_depth(pts, K, (H, W))
    assert depth.shape == (H, W) and depth.dtype == np.float32
    # 每个点落在预期像素且深度一致
    assert np.count_nonzero(depth) == n
    assert np.allclose(depth[v, u], z, atol=1e-4)

    pts2, pix = backproject_depth(depth, K)
    assert pts2.shape == (n, 3) and pix.shape == (n, 2)
    assert pix.dtype.kind == "i" and pts2.dtype == np.float32
    # 按线性像素索引对齐后逐点比较（往返一致性）
    order_in = np.argsort(v * W + u)
    order_out = np.argsort(pix[:, 1] * W + pix[:, 0])
    assert np.array_equal(pix[order_out, 0], u[order_in])
    assert np.array_equal(pix[order_out, 1], v[order_in])
    assert np.allclose(pts2[order_out], pts[order_in], atol=1e-3)

    # 再投影闭环: 反投影点再投影应得到同一幅深度图
    depth2 = project_points_to_depth(pts2, K, (H, W))
    assert np.allclose(depth2, depth, atol=1e-4)


def test_zbuffer_min_depth():
    H, W = 64, 96
    K = virtual_camera_intrinsics(f=80.0, size=(H, W))
    # 两点均投影到主点像素，z-buffer 应保留较近者
    pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    depth = project_points_to_depth(pts, K, (H, W))
    cu, cv = int(K[0, 2]), int(K[1, 2])
    assert depth[cv, cu] == np.float32(2.0)
    assert np.count_nonzero(depth) == 1


def test_projection_filtering():
    H, W = 64, 96
    K = virtual_camera_intrinsics(f=80.0, size=(H, W))
    pts = np.array([
        [0.0, 0.0, 0.01],     # 小于 min_depth
        [0.0, 0.0, 100.0],    # 大于 max_depth
        [0.0, 0.0, -3.0],     # 相机后方
        [1000.0, 0.0, 2.0],   # 投影出界
    ], dtype=np.float32)
    depth = project_points_to_depth(pts, K, (H, W))
    assert np.count_nonzero(depth) == 0

    empty = project_points_to_depth(np.zeros((0, 3), np.float32), K, (H, W))
    assert empty.shape == (H, W) and np.count_nonzero(empty) == 0


def test_backproject_range_filter():
    H, W = 8, 8
    K = virtual_camera_intrinsics(f=10.0, size=(H, W))
    depth = np.zeros((H, W), dtype=np.float32)
    depth[1, 1] = 0.01   # 太近, 应被跳过
    depth[2, 2] = 2.0    # 有效
    depth[3, 3] = 90.0   # 太远, 应被跳过
    pts, pix = backproject_depth(depth, K, min_depth=0.05, max_depth=60.0)
    assert pts.shape == (1, 3)
    assert pix.tolist() == [[2, 2]]
    assert np.isclose(pts[0, 2], 2.0)


# ------------------------------------------------------------ LDP / CDP

def test_generate_ldp_matches_manual():
    rng = np.random.default_rng(1)
    H, W = 64, 96
    K = virtual_camera_intrinsics(f=80.0, size=(H, W))
    # 随机外参 (小旋转 + 平移)
    T = np_se3_from_rt(np.array([0.05, -0.1, 0.2]), np.array([0.1, -0.05, 0.3]))
    pts_lidar = rng.uniform(-3, 3, size=(500, 3)).astype(np.float32)
    pts_lidar[:, 2] += 5.0  # 大致在相机前方

    ldp = generate_ldp(pts_lidar, T, K, (H, W))
    manual = project_points_to_depth(
        np_transform(T, pts_lidar).astype(np.float32), K, (H, W))
    assert np.allclose(ldp, manual, atol=1e-5)

    # 空点云 → 全零
    assert np.count_nonzero(generate_ldp(np.zeros((0, 3), np.float32), T, K, (H, W))) == 0


def test_generate_cdp_identity_view():
    rng = np.random.default_rng(2)
    H, W = 64, 96
    K = virtual_camera_intrinsics(f=80.0, size=(H, W))
    pts, _, _, _ = _pixel_ray_points(rng, K, H, W, 100)
    # T_view_cam = I 时 CDP 即原视角投影
    cdp = generate_cdp(pts, np.eye(4), K, (H, W))
    assert np.allclose(cdp, project_points_to_depth(pts, K, (H, W)), atol=1e-5)


# --------------------------------------------------- Gemini 深度反投影

def test_depth_cloud_from_gemini():
    H, W = 8, 10
    K = np.array([[50.0, 0, 5.0], [0, 50.0, 4.0], [0, 0, 1.0]])
    depth = np.full((H, W), 1.0, dtype=np.float32)
    depth[0, 0] = 0.1    # 低于 d_min=0.25
    depth[0, 2] = 7.0    # 高于 d_max=6.0
    depth[2, 4] = 0.0    # 无效

    pts = depth_cloud_from_gemini(depth, K, d_min=0.25, d_max=6.0, stride=1)
    assert pts.shape == (H * W - 3, 3) and pts.dtype == np.float32
    assert np.all(pts[:, 2] >= 0.25) and np.all(pts[:, 2] <= 6.0)

    # stride=2: 网格 4x5=20，(0,0)/(0,2)/(2,4) 都在网格上且无效 → 17 点
    pts2 = depth_cloud_from_gemini(depth, K, stride=2)
    assert pts2.shape == (17, 3)
    # 亚采样保持原像素坐标几何: 重投影 u = fx*x/z+cx 应为 stride 倍数整数
    u = K[0, 0] * pts2[:, 0] / pts2[:, 2] + K[0, 2]
    v = K[1, 1] * pts2[:, 1] / pts2[:, 2] + K[1, 2]
    assert np.allclose(np.round(u / 2) * 2, u, atol=1e-4)
    assert np.allclose(np.round(v / 2) * 2, v, atol=1e-4)


# ------------------------------------------------ 差分图 式(11)(12)

def _toy_ldp_cdp():
    ldp = np.zeros((4, 5), dtype=np.float32)
    cdp = np.zeros((4, 5), dtype=np.float32)
    ldp[0, 0], cdp[0, 0] = 2.0, 1.5    # Δ=0.5 → 边界 |Δ|=e_tar → [Δ]_-
    ldp[1, 1], cdp[1, 1] = 5.0, 3.0    # Δ=2.0 → [Δ]_+
    ldp[2, 2], cdp[2, 2] = 5.0, 0.0    # CDP 无效 → 两差分通道为 0
    ldp[3, 3], cdp[3, 3] = 0.0, 2.0    # LDP 无效 → 两差分通道为 0
    ldp[3, 4], cdp[3, 4] = 1.0, 4.0    # Δ=-3.0 → [Δ]_+ (负值大误差)
    return ldp, cdp


def test_difference_map_semantics():
    ldp, cdp = _toy_ldp_cdp()
    e_tar = 0.5
    D = build_difference_map(ldp, cdp, e_tar=e_tar)
    assert D.shape == (3, 4, 5) and D.dtype == np.float32

    # 通道 0 保留完整 LDP（含 CDP 无效处）
    assert np.array_equal(D[0], ldp)

    # 式(12): |Δ|<=e_tar → [Δ]_-；|Δ|>e_tar → [Δ]_+（保留符号）
    assert D[2][0, 0] == np.float32(0.5) and D[1][0, 0] == 0.0
    assert D[1][1, 1] == np.float32(2.0) and D[2][1, 1] == 0.0
    assert D[1][3, 4] == np.float32(-3.0) and D[2][3, 4] == 0.0

    # 式(11): 非共同有效像素两差分通道为 0
    assert D[1][2, 2] == 0.0 and D[2][2, 2] == 0.0
    assert D[1][3, 3] == 0.0 and D[2][3, 3] == 0.0

    # 通道互斥: [Δ]_+ 与 [Δ]_- 非零位置不相交
    assert not np.any((D[1] != 0) & (D[2] != 0))

    # 两通道之和在共同有效像素处重构 Δ
    valid = (ldp > 0) & (cdp > 0)
    assert np.allclose((D[1] + D[2])[valid], (ldp - cdp)[valid], atol=1e-6)


def test_difference_map_torch_matches_numpy():
    rng = np.random.default_rng(3)
    B, H, W = 3, 16, 24
    ldp = rng.uniform(0, 8, size=(B, H, W)).astype(np.float32)
    cdp = rng.uniform(0, 8, size=(B, H, W)).astype(np.float32)
    # 随机撒无效(0)像素
    ldp[rng.random((B, H, W)) < 0.3] = 0.0
    cdp[rng.random((B, H, W)) < 0.3] = 0.0

    out = build_difference_map_torch(torch.from_numpy(ldp), torch.from_numpy(cdp),
                                     e_tar=0.1)
    assert out.shape == (B, 3, H, W)
    for b in range(B):
        ref = build_difference_map(ldp[b], cdp[b], e_tar=0.1)
        assert np.allclose(out[b].numpy(), ref, atol=1e-6)

    # 单幅 (H,W) 兼容
    out2 = build_difference_map_torch(torch.from_numpy(ldp[0]),
                                      torch.from_numpy(cdp[0]), e_tar=0.1)
    assert out2.shape == (3, H, W)
    assert np.allclose(out2.numpy(), build_difference_map(ldp[0], cdp[0], 0.1),
                       atol=1e-6)


def test_difference_map_torch_differentiable():
    # 数值路径可微: 对有效像素 d(out)/d(ldp) 应有梯度
    ldp = torch.tensor([[[1.0, 2.0], [0.0, 3.0]]], requires_grad=True)
    cdp = torch.tensor([[[0.9, 0.5], [1.0, 0.0]]])
    out = build_difference_map_torch(ldp, cdp, e_tar=0.1)
    out.sum().backward()
    g = ldp.grad
    # (0,0,0): LDP 通道 + [Δ]_- → 2；(0,0,1): LDP 通道 + [Δ]_+ → 2
    assert g[0, 0, 0] == 2.0 and g[0, 0, 1] == 2.0
    # (0,1,0): LDP=0 无效但 LDP 通道保留原值 → 1；(0,1,1): CDP 无效 → 1
    assert g[0, 1, 0] == 1.0 and g[0, 1, 1] == 1.0


# ------------------------------------------- 综合: LDP/CDP → 差分图

def test_ldp_cdp_difference_pipeline():
    """完美外参下 LDP 与 CDP 应几乎一致 → [Δ]_+ 通道近乎全零。"""
    rng = np.random.default_rng(4)
    H, W = 64, 96
    K = virtual_camera_intrinsics(f=80.0, size=(H, W))
    pts_cam, _, _, _ = _pixel_ray_points(rng, K, H, W, 300)

    # LiDAR 点 = 相机点经 T_lidar_cam 逆变换（构造一致场景）
    T_cam_lidar = np_se3_from_rt(np.array([0.02, 0.3, -0.1]),
                                 np.array([0.2, -0.1, 0.05]))
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    pts_lidar = np_transform(T_lidar_cam, pts_cam).astype(np.float32)

    ldp = generate_ldp(pts_lidar, T_cam_lidar, K, (H, W))
    cdp = generate_cdp(pts_cam, np.eye(4), K, (H, W))
    D = build_difference_map(ldp, cdp, e_tar=0.1)

    valid = (ldp > 0) & (cdp > 0)
    assert valid.sum() > 250  # 变换往返后绝大多数点仍落回原像素
    assert np.count_nonzero(D[1]) == 0  # 无大误差
    assert np.allclose(D[2][valid], (ldp - cdp)[valid], atol=1e-6)
