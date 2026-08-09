"""合成房间场景生成器 — tests/test_synthetic.py 与 tests/test_full_paper_form.py 共用。

从 test_synthetic.py 抽出的公共部分 (契约 A5: 慢测复用场景生成器), 行为与原实现
逐比特一致 (默认参数下采样点数与 rng 调用次序完全不变, 保证原端到端测试不回归):

- 合成房间 (地面 + 四面墙 + 3 个不同尺寸箱体, 表面均匀采样);
- Mid-360 LiDAR 模拟: 以 LiDAR 原点做法向可见性剔除 + 0.01 m 高斯噪声;
- Gemini 335 深度模拟: 稠密采样经真值外参变换后 z-buffer 投影 (640x480,
  fx=fy=450) + 0.005 m 高斯噪声;
- 真值外参 make_T_gt: R0 组合 15° 轴角扰动, |t| ≈ 0.27 m。

make_frame 增加 dens_scale 关键字 (默认 1.0 即原行为), 供慢测在多场景时
适度降密度控制时长。
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from dst_calib.config import load_config
from dst_calib.geometry import np_so3_exp, np_transform
from dst_calib.projection import project_points_to_depth
from dst_calib.self_supervised import Frame

# ---------------------------------------------------------------- 场景几何参数

FLOOR_Z = -0.8            # 地面高度 (LiDAR 系, LiDAR 位于地面上方 0.8 m)
ROOM_X = (-1.0, 4.5)      # 房间 x 范围 (LiDAR 系, x 前)
ROOM_Y = (-3.5, 3.5)      # 房间 y 范围 (y 左)
WALL_TOP = 1.8            # 墙顶高度

# 相机内参 (任务要求: 640x480, fx=fy=450)
CAM_HW = (480, 640)
K_CAM = np.array([[450.0, 0.0, 320.0],
                  [0.0, 450.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

# 采样密度 (点/m²): LiDAR 侧较稀 (模拟累积后的 Mid-360), 相机侧稠密避免 z-buffer 漏光
DENS_LIDAR_ROOM = 1200.0
DENS_LIDAR_BOX = 3000.0
DENS_CAM_ROOM = 9000.0
DENS_CAM_BOX = 24000.0

NOISE_LIDAR = 0.01        # LiDAR 点噪声 σ (m, 每轴)
NOISE_DEPTH = 0.005       # 相机深度噪声 σ (m)

# 3 个不同尺寸箱体 (sx, sy, sz)
BOX_SIZES = [(0.6, 0.45, 0.9), (0.4, 0.4, 0.5), (0.9, 0.35, 0.6)]

# 每帧箱体摆位 (cx, cy, yaw_deg) —— 移动箱体模拟不同视角/内容
FRAME_BOX_POSES = [
    [(2.2, -1.2, 20.0), (3.2, 1.0, -15.0), (2.8, -0.1, 45.0)],
    [(3.4, -0.6, -30.0), (2.0, 1.4, 10.0), (2.5, 0.5, 0.0)],
    [(2.6, 1.6, 50.0), (2.9, -1.5, 0.0), (3.6, 0.4, -25.0)],
    [(2.0, 0.2, -10.0), (3.5, -1.2, 35.0), (3.1, 1.3, 15.0)],
]


# ---------------------------------------------------------------- 场景构造

def room_faces() -> list:
    """房间 5 个面 (地面 + 前/后/左/右墙), 法向指向室内。

    每个面 = (origin, edge_u, edge_v, unit_normal)。
    """
    x0, x1 = ROOM_X
    y0, y1 = ROOM_Y
    z0, z1 = FLOOR_Z, WALL_TOP
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    A = np.array
    return [
        # 地面 (法向 +z)
        (A([x0, y0, z0]), A([dx, 0, 0]), A([0, dy, 0]), A([0.0, 0.0, 1.0])),
        # 前墙 x=x1 (法向 -x)
        (A([x1, y0, z0]), A([0, dy, 0]), A([0, 0, dz]), A([-1.0, 0.0, 0.0])),
        # 后墙 x=x0 (法向 +x)
        (A([x0, y0, z0]), A([0, dy, 0]), A([0, 0, dz]), A([1.0, 0.0, 0.0])),
        # 左墙 y=y1 (法向 -y)
        (A([x0, y1, z0]), A([dx, 0, 0]), A([0, 0, dz]), A([0.0, -1.0, 0.0])),
        # 右墙 y=y0 (法向 +y)
        (A([x0, y0, z0]), A([dx, 0, 0]), A([0, 0, dz]), A([0.0, 1.0, 0.0])),
    ]


def box_faces(cx: float, cy: float, size: tuple, yaw_deg: float) -> list:
    """立于地面的箱体 5 个面 (顶 + 四侧, 底面不可见故省略), 绕 z 转 yaw。"""
    sx, sy, sz = size
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    ex = np.array([c, s, 0.0]) * sx        # 箱体局部 x 边
    ey = np.array([-s, c, 0.0]) * sy       # 箱体局部 y 边
    ez = np.array([0.0, 0.0, sz])
    exn = np.array([c, s, 0.0])            # 单位法向
    eyn = np.array([-s, c, 0.0])
    p0 = np.array([cx, cy, FLOOR_Z]) - ex * 0.5 - ey * 0.5  # 底面角点
    return [
        (p0 + ez, ex, ey, np.array([0.0, 0.0, 1.0])),   # 顶面
        (p0, ey, ez, -exn),                              # -x 侧
        (p0 + ex, ey, ez, exn),                          # +x 侧
        (p0, ex, ez, -eyn),                              # -y 侧
        (p0 + ey, ex, ez, eyn),                          # +y 侧
    ]


def sample_faces(faces: list, density: float, rng: np.random.Generator,
                 viewpoint: np.ndarray = None) -> np.ndarray:
    """在各面上均匀随机采样 (点数 = 面积×密度)。

    viewpoint 非 None 时做背面剔除: 仅保留法向朝向观察点的面
    (近似单视点可见性; 物体间遮挡由相机侧 z-buffer 处理,
    LiDAR 侧保留 —— 与相机 z-buffer 漏光点同落在真实几何上, 两侧一致)。
    """
    pts = []
    for origin, eu, ev, n in faces:
        if viewpoint is not None:
            center = origin + 0.5 * eu + 0.5 * ev
            if float(np.dot(n, viewpoint - center)) <= 0.0:
                continue
        area = float(np.linalg.norm(np.cross(eu, ev)))
        cnt = max(1, int(round(area * density)))
        u = rng.random(cnt)[:, None]
        v = rng.random(cnt)[:, None]
        pts.append(origin + u * eu + v * ev)
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


def make_T_gt() -> np.ndarray:
    """真值外参: R = R0 · exp(15° 轴角扰动), |t| ≈ 0.27 m。"""
    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    axis = np.array([0.4, -0.5, 0.76])
    axis = axis / np.linalg.norm(axis)
    R_pert = np_so3_exp(axis * math.radians(15.0))     # 15° ∈ [10°, 20°]
    t_gt = np.array([0.15, -0.05, 0.22])               # |t| ≈ 0.27 ∈ [0.1, 0.3]
    T = np.eye(4)
    T[:3, :3] = R0 @ R_pert
    T[:3, 3] = t_gt
    return T


def make_frame(box_poses: list, T_gt: np.ndarray, stamp: float,
               rng: np.random.Generator, dens_scale: float = 1.0,
               box_sizes: list = None) -> Frame:
    """生成一帧: LiDAR 表面采样点云 + 相机 z-buffer 深度图。

    默认参数下与原 test_synthetic._make_frame 行为逐比特一致;
    dens_scale<1.0 按比例降低两侧采样密度; box_sizes 可覆盖默认 3 箱体尺寸表
    (慢测用更多不同深度的箱体增强平移视差可观测性)。
    """
    room = room_faces()
    boxes = []
    for (cx, cy, yaw), size in zip(box_poses, box_sizes or BOX_SIZES):
        boxes.extend(box_faces(cx, cy, size, yaw))

    # --- LiDAR 点云 (LiDAR 系): 以 LiDAR 原点做法向可见性剔除 + 0.01 m 噪声
    origin_lidar = np.zeros(3)
    pts_l = np.concatenate([
        sample_faces(room, DENS_LIDAR_ROOM * dens_scale, rng, viewpoint=origin_lidar),
        sample_faces(boxes, DENS_LIDAR_BOX * dens_scale, rng, viewpoint=origin_lidar),
    ], axis=0)
    pts_l = pts_l + rng.normal(0.0, NOISE_LIDAR, pts_l.shape)

    # --- 相机深度图: 稠密采样 → 变换到相机系 → z-buffer 投影 + 0.005 m 噪声
    cam_center_w = -T_gt[:3, :3].T @ T_gt[:3, 3]       # 相机光心在 LiDAR 系的位置
    dense = np.concatenate([
        sample_faces(room, DENS_CAM_ROOM * dens_scale, rng, viewpoint=cam_center_w),
        sample_faces(boxes, DENS_CAM_BOX * dens_scale, rng, viewpoint=cam_center_w),
    ], axis=0)
    p_cam = np_transform(T_gt, dense)
    depth = project_points_to_depth(p_cam, K_CAM, CAM_HW,
                                    min_depth=0.05, max_depth=10.0)
    m = depth > 0
    depth[m] = depth[m] + rng.normal(0.0, NOISE_DEPTH, int(m.sum())).astype(np.float32)

    return Frame(points_lidar=pts_l.astype(np.float32), depth=depth,
                 K=K_CAM.copy(), rgb=None, stamp=stamp)


# ---------------------------------------------------------------- 精简配置

def make_reduced_cfg():
    """基于 default.yaml 的精简配置: 缩减迭代数控制时长, 代码路径不变。

    (原 test_synthetic._make_cfg, 供端到端测试与慢测的 selfsup/both 路径共用。)
    """
    cfg = load_config()
    # 合成房间对角最远约 6.1 m, 放宽相机量程避免边角裁剪不一致 (真实 Gemini 为 6 m)
    cfg.sensors.camera.depth_max = 7.0
    ss = cfg.self_supervised
    ss.seed = 0
    # 粗搜索: 8×45° yaw × 3 pitch, 每起点 30 次 Adam (足够选出正确朝向盆地)
    ss.coarse.yaw_grid = 8
    ss.coarse.pitch_grid_deg = [-15.0, 0.0, 15.0]
    ss.coarse.iters_per_start = 30
    ss.coarse.max_points = 3000
    # 三阶段由粗到细 (迭代数缩减, 总数 270 ≥ 30 批, 满足论文 IV)
    ss.stages = [
        SimpleNamespace(voxel=0.30, iters=60, lr=0.010, trunc=2.0, max_points=6000),
        SimpleNamespace(voxel=0.12, iters=90, lr=0.004, trunc=0.8, max_points=10000),
        SimpleNamespace(voxel=0.05, iters=120, lr=0.0015, trunc=0.3, max_points=15000),
    ]
    ss.per_frame_finetune_iters = 25
    ss.per_frame_lr = 0.0008
    ss.batch_frames = 2
    return cfg
