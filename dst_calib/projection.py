"""针孔投影 / 反投影工具 — LDP 与 CDP 生成 (论文 III-A/III-B)。

提供:
- 虚拟相机内参 (论文 IV 实现细节: (H,W)=(256,512), f=600, cx=W/2, cy=H/2)
- 点云 → 深度图 (z-buffer 取最小深度)
- 深度图 → 点云 (反投影)
- LDP (LiDAR Depth Projection) / CDP (Camera Depth Projection)，供
  式(5) 锚点提取与式(11)(12) 差分图构建使用
- Gemini335 度量深度图 → 相机系点云

全部为 numpy 实现（数据准备路径），深度图 float32、单位米、0 表示无效像素。
"""
from __future__ import annotations

import numpy as np

from .geometry import np_transform


def virtual_camera_intrinsics(f: float = 600.0, size: tuple = (256, 512)) -> np.ndarray:
    """构造虚拟相机内参 K (3x3)。

    论文 IV 实现细节: 投影尺寸 (H,W)=(256,512)，焦距 f=600，主点位于图像中心
    cx=W/2, cy=H/2。size=(H,W)。
    """
    H, W = int(size[0]), int(size[1])
    K = np.array([
        [f, 0.0, W / 2.0],
        [0.0, f, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def project_points_to_depth(points_cam: np.ndarray, K: np.ndarray, size: tuple,
                            min_depth: float = 0.05, max_depth: float = 60.0) -> np.ndarray:
    """相机系点云 (N,3) → 深度图 (H,W) float32（标准针孔投影, 论文 III-A）。

    步骤（完全向量化）:
    1) 过滤 z ∈ [min_depth, max_depth] 之外的点（含相机后方 z<=0）;
    2) u = fx*x/z + cx, v = fy*y/z + cy，四舍五入到最近像素并过滤出界像素;
    3) z-buffer: 同一像素落多点时取最小深度（np.minimum.at 无缓冲累积）。
    空像素为 0。
    """
    H, W = int(size[0]), int(size[1])
    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    depth_flat = np.full(H * W, np.inf, dtype=np.float32)

    if pts.shape[0] > 0:
        z = pts[:, 2]
        # 深度范围过滤（z>0 兜底防止除零）
        valid = (z > 0) & (z >= min_depth) & (z <= max_depth)
        pts = pts[valid]
        if pts.shape[0] > 0:
            z = pts[:, 2]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            u = fx * pts[:, 0] / z + cx
            v = fy * pts[:, 1] / z + cy
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if np.any(inb):
                lin = vi[inb] * W + ui[inb]
                # z-buffer: 同像素取最小深度（无缓冲 ufunc.at 保证重复索引正确）
                np.minimum.at(depth_flat, lin, z[inb])

    depth = depth_flat.reshape(H, W)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def backproject_depth(depth: np.ndarray, K: np.ndarray,
                      min_depth: float = 0.05, max_depth: float = 60.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """深度图 → (points (M,3) 相机系, pixels (M,2) 整数 (u,v))。

    针孔模型逆映射: x=(u-cx)z/fx, y=(v-cy)z/fy。仅保留
    z ∈ [min_depth, max_depth] 的有效像素（0 即无效）。
    """
    d = np.asarray(depth, dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = (d > 0) & (d >= min_depth) & (d <= max_depth)
    v_idx, u_idx = np.nonzero(valid)
    z = d[v_idx, u_idx]
    x = (u_idx.astype(np.float32) - cx) * z / fx
    y = (v_idx.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pixels = np.stack([u_idx, v_idx], axis=1).astype(np.int32)  # (u,v) 列序
    return points, pixels


def generate_ldp(points_lidar: np.ndarray, T_cam_lidar: np.ndarray,
                 K: np.ndarray, size: tuple) -> np.ndarray:
    """LiDAR Depth Projection (论文 III-A): LiDAR 点经外参 T_cam_lidar 变换到
    相机系后做针孔投影生成深度图。为式(5)锚点与式(11) Δ 的 LDP 输入。
    """
    pts = np.asarray(points_lidar, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        H, W = int(size[0]), int(size[1])
        return np.zeros((H, W), dtype=np.float32)
    pts_cam = np_transform(np.asarray(T_cam_lidar, dtype=np.float64), pts)
    return project_points_to_depth(pts_cam, K, size)


def generate_cdp(points_cam_depth: np.ndarray, T_view_cam: np.ndarray,
                 K: np.ndarray, size: tuple) -> np.ndarray:
    """Camera Depth Projection (论文 III-A): 相机深度点云经视角变换 T_view_cam
    后投影为深度图。T_view_cam=I 时即原相机视角；训练时双侧增广(式3)用
    扰动后的虚拟视角。
    """
    pts = np.asarray(points_cam_depth, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        H, W = int(size[0]), int(size[1])
        return np.zeros((H, W), dtype=np.float32)
    pts_view = np_transform(np.asarray(T_view_cam, dtype=np.float64), pts)
    return project_points_to_depth(pts_view, K, size)


def depth_cloud_from_gemini(depth: np.ndarray, K_color: np.ndarray,
                            d_min: float = 0.25, d_max: float = 6.0,
                            stride: int = 1) -> np.ndarray:
    """Gemini335 已对齐彩色的度量深度图 → 相机系点云 (N,3) float32。

    仅保留 d ∈ [d_min, d_max] 的像素（Gemini 335 可靠深度范围），
    stride 亚采样降低点数（保持原始像素坐标反投影，不改变几何）。
    """
    d = np.asarray(depth, dtype=np.float32)
    H, W = d.shape
    stride = max(1, int(stride))
    # 亚采样网格上的原始像素坐标
    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    sub = d[np.ix_(vs, us)]
    valid = (sub > 0) & (sub >= d_min) & (sub <= d_max)
    iv, iu = np.nonzero(valid)
    v_idx = vs[iv].astype(np.float32)
    u_idx = us[iu].astype(np.float32)
    z = sub[iv, iu]

    fx, fy = float(K_color[0, 0]), float(K_color[1, 1])
    cx, cy = float(K_color[0, 2]), float(K_color[1, 2])
    x = (u_idx - cx) * z / fx
    y = (v_idx - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)
