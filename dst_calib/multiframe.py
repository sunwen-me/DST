"""多帧优化 (论文 III-E, 式(20)-(23))。

对每帧独立微调得到的候选外参 {T_i} 打分:
- 全监督 式(20): s_i = exp(-(a·||e'_i - e_i||_2 + ||t'_i - t_i||_2));
- 自监督 式(21): s_i = exp(-L_CD(T_i·P_i, Q_i));
再按 式(22) k=⌈x·n⌉ 取分数最高的 k 个, 按 式(23) 权重
w_j = s_{π(j)} / Σ_{i=1..k} s_{π(i)} 加权平均 (平移线性加权,
旋转用四元数特征向量法 geometry.quat_mean_rotations 加权) 得最终 T*。
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .geometry import (
    euler_zyx_from_R,
    np_so3_exp,
    np_transform,
    quat_mean_rotations,
)
from .chamfer import chamfer_distance


def score_self_supervised(T_list: list, frames: list, cfg) -> np.ndarray:
    """式(21): s_i = exp(-L_CD(T_i·P_i, Q_i))。

    L_CD 为式(17) 标准(未截断)双向 Chamfer —— 与式(21) 严格一致, 使误标定
    帧的得分不因截断而饱和, 保持式(23) 权重的区分度。用末段参数
    (voxel=stages[-1].voxel, max_points 同末段) 的下采样点云计算。
    无效帧 (相机点数<500 或共视点不足) 得分置 0 并告警。
    """
    # 延迟导入避免与 self_supervised 的循环依赖
    from .self_supervised import (
        MIN_CAMERA_POINTS,
        MIN_STAGE_POINTS,
        _cam_depth_range,
        _device,
        _ss,
        _to_t,
        camera_cloud,
        fov_crop_points,
        voxel_downsample,
    )

    ss = _ss(cfg)
    last = list(getattr(ss, "stages"))[-1]
    voxel = float(last.voxel)
    max_points = int(getattr(last, "max_points", 20000))
    alpha = float(getattr(ss, "chamfer_alpha", 0.5))
    beta = float(getattr(ss, "chamfer_beta", 0.5))
    margin = float(getattr(ss, "fov_margin_deg", 8.0))
    _, depth_max = _cam_depth_range(cfg)
    device = _device()
    rng = np.random.default_rng(int(getattr(ss, "seed", 0)) + 3000)

    scores = np.zeros(len(T_list), dtype=np.float64)
    for i, (T_i, fr) in enumerate(zip(T_list, frames)):
        T_i = np.asarray(T_i, dtype=np.float64).reshape(4, 4)
        Q = camera_cloud(fr, cfg)
        if Q.shape[0] < MIN_CAMERA_POINTS:
            print(f"[multiframe] 警告: 帧 {i} 相机深度点数 {Q.shape[0]} < "
                  f"{MIN_CAMERA_POINTS}, 得分置 0")
            continue
        P = fov_crop_points(fr.points_lidar, T_i, fr.K, fr.depth.shape, margin, depth_max)
        P = voxel_downsample(P, voxel, max_points, rng)
        Q = voxel_downsample(Q, voxel, max_points, rng)
        if P.shape[0] < MIN_STAGE_POINTS or Q.shape[0] < MIN_STAGE_POINTS:
            print(f"[multiframe] 警告: 帧 {i} 共视点不足 (P={P.shape[0]}, "
                  f"Q={Q.shape[0]}), 得分置 0")
            continue
        with torch.no_grad():
            cd = chamfer_distance(
                _to_t(np_transform(T_i, P.astype(np.float64)), device),
                _to_t(Q, device), alpha=alpha, beta=beta)
        scores[i] = math.exp(-float(cd))  # 式(21)
    return scores


def score_full_supervised(T_list: list, xi_eva_list: list, a: float = 0.1) -> np.ndarray:
    """式(20): s_i = exp(-(a·||e'_i - e_i||_2 + ||t'_i - t_i||_2))。

    e'_i, t'_i 取自候选外参 T_i; e_i, t_i 取自评估模块输出 ξ_eva,i = [r, t]。
    工程细节: 旋转差不取两组绝对欧拉角相减, 而取相对旋转 R'_i·R_iᵀ 的
    ZYX 欧拉角向量范数 —— 小差异下与 ||e'_i-e_i|| 一阶等价 (语义同式(20)),
    且在任意安装姿态下良态: 绝对欧拉角在 pitch=±90° 万向节锁附近病态
    (本项目名义前视安装 R0 恰在 pitch=-90°), 微小真实旋转差会被放大数百倍,
    使式(22) 选帧排序被逐帧各向异性噪声支配; 相对旋转近恒等, 欧拉提取
    远离奇异, 各角天然落在主值区间, 无需包角。a=0.1 (论文 IV)。
    """
    scores = np.zeros(len(T_list), dtype=np.float64)
    for i, (T_i, xi_eva) in enumerate(zip(T_list, xi_eva_list)):
        T_i = np.asarray(T_i, dtype=np.float64).reshape(4, 4)
        xi_eva = np.asarray(xi_eva, dtype=np.float64).reshape(6)
        R_pred = T_i[:3, :3]
        R_eva = np_so3_exp(xi_eva[:3])
        de = float(np.linalg.norm(euler_zyx_from_R(R_pred @ R_eva.T)))  # 相对空间
        dt = float(np.linalg.norm(T_i[:3, 3] - xi_eva[3:]))
        scores[i] = math.exp(-(float(a) * de + dt))  # 式(20)
    return scores


def select_and_average(T_list: list, scores: np.ndarray, x: float = 0.3,
                       weighting: str = "score") -> np.ndarray:
    """式(22)(23): 按分数降序取 k=⌈x·n⌉ 个候选, 加权平均得 T*。

    式(22): k = ⌈x·n⌉;
    式(23): w_j = s_{π(j)} / Σ_{i=1..k} s_{π(i)} (weighting=="uniform" 时等权)。
    平移线性加权平均; 旋转用 quat_mean_rotations 四元数加权平均。返回 (4,4)。
    """
    n = len(T_list)
    if n == 0:
        raise RuntimeError("多帧加权失败: 候选外参列表为空 — 上游优化未产生任何有效帧结果。")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.shape[0] != n:
        raise ValueError(f"scores 长度 {s.shape[0]} 与 T_list 长度 {n} 不一致")
    k = int(math.ceil(float(x) * n))  # 式(22)
    k = max(1, min(k, n))
    order = np.argsort(-s)  # 分数降序 π
    top = order[:k]
    if weighting == "score" and float(s[top].sum()) > 0:
        w = s[top] / float(s[top].sum())  # 式(23)
    else:
        # uniform 或全零分数退化: 等权
        w = np.full(k, 1.0 / k)
    R_list = [np.asarray(T_list[j], dtype=np.float64)[:3, :3] for j in top]
    t_arr = np.stack([np.asarray(T_list[j], dtype=np.float64)[:3, 3] for j in top])
    R_star = quat_mean_rotations(R_list, weights=w)
    t_star = (w[:, None] * t_arr).sum(axis=0)
    T_star = np.eye(4)
    T_star[:3, :3] = R_star
    T_star[:3, 3] = t_star
    print(f"[multiframe] n={n}, k={k}, 选中帧={top.tolist()}, "
          f"权重={np.round(w, 3).tolist()}")
    return T_star
