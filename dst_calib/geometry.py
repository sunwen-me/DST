"""SE(3)/SO(3) 几何工具 — torch 可微 + numpy 版本。

约定: T_cam_lidar @ p_lidar = p_cam；旋转向量 r 为轴角 (so3)，欧拉角为 ZYX 内旋
(yaw-pitch-roll)，单位弧度除非注明。
"""
from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-8


# ---------------------------------------------------------------- torch (可微)

def so3_exp(r: torch.Tensor) -> torch.Tensor:
    """轴角 → 旋转矩阵 (Rodrigues)。r: (...,3) → (...,3,3)，小角度泰勒展开保稳定。"""
    theta = r.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (...,1)
    small = theta < 1e-6
    k = r / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz, ky], dim=-1),
        torch.stack([kz, zero, -kx], dim=-1),
        torch.stack([-ky, kx, zero], dim=-1),
    ], dim=-2)  # (...,3,3)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(K.shape)
    th = theta.unsqueeze(-1)  # (...,1,1)
    sin_t, cos_t = torch.sin(th), torch.cos(th)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    # 小角度: R ≈ I + hat(r)
    rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]
    Hat = torch.stack([
        torch.stack([zero, -rz, ry], dim=-1),
        torch.stack([rz, zero, -rx], dim=-1),
        torch.stack([-ry, rx, zero], dim=-1),
    ], dim=-2)
    R_small = eye + Hat
    return torch.where(small.unsqueeze(-1).expand(R.shape), R_small, R)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 → 轴角。R: (...,3,3) → (...,3)。含 θ≈0 与 θ≈π 两个退化分支
    (与 np_so3_log 等价; 无近 π 分支时 w→0 会把 180° 旋转静默坍缩成 ≈0)。"""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)  # (...)
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)  # (...,3) = 2·sinθ·axis
    scale = theta / (2.0 * torch.sin(theta)).clamp_min(_EPS)
    r = w * scale.unsqueeze(-1)
    # ---- θ≈π 退化分支: w→0, 改由 (R+I)/2 ≈ n·nᵀ 的最大对角元所在列提取轴 ----
    # 对称化消去 sinθ·K 反对称污染项, 轴误差从 O(π-θ) 降到 O((π-θ)²)
    eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
    A = ((R + R.transpose(-1, -2)) * 0.5 + eye) * 0.5
    diagA = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(0.0)     # (...,3) ≈ n_i²
    k = diagA.argmax(dim=-1)                                       # (...)
    col = torch.gather(A, -1, k[..., None, None].expand(*A.shape[:-1], 1)).squeeze(-1)
    d = torch.gather(diagA, -1, k[..., None]).squeeze(-1).clamp_min(1e-12).sqrt()
    axis = col / d.unsqueeze(-1)
    axis = axis / torch.sqrt((axis * axis).sum(dim=-1, keepdim=True) + 1e-12)
    # θ 在 π 附近改用 asin(|w|/2)=asin(sinθ) 补偿 acos 在 -1 附近的精度损失
    w_norm = torch.sqrt((w * w).sum(dim=-1) + 1e-12)               # ≈ 2·sinθ
    theta_pi = float(np.pi) - torch.asin((0.5 * w_norm).clamp(0.0, 1.0 - 1e-7))
    # 符号与 w 对齐 (θ<π 时 w 给出真方向; θ=π 时 ±axis 等价, 取 +)
    dot = (axis * w).sum(dim=-1, keepdim=True)
    sgn = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    r_pi = axis * sgn * theta_pi.unsqueeze(-1)
    near_pi = (theta > float(np.pi) - 1e-3).unsqueeze(-1)
    small = (theta < 1e-6).unsqueeze(-1)
    return torch.where(small, 0.5 * w, torch.where(near_pi, r_pi, r))


def se3_from_rt(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(r (...,3), t (...,3)) → T (...,4,4)。"""
    R = so3_exp(r)
    batch = R.shape[:-2]
    T = torch.zeros(*batch, 4, 4, dtype=r.dtype, device=r.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def rt_from_se3(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """T (...,4,4) → (r (...,3), t (...,3))。"""
    return so3_log(T[..., :3, :3]), T[..., :3, 3]


def transform_points(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """T (4,4) 或 (B,4,4)，pts (N,3) 或 (B,N,3) → 变换后同形状。可微。"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    return pts @ R.transpose(-1, -2) + t.unsqueeze(-2)


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """SE(3) 逆。"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    Ti = torch.zeros_like(T)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = (-Rt @ t.unsqueeze(-1)).squeeze(-1)
    Ti[..., 3, 3] = 1.0
    return Ti


def euler_zyx_from_r(r: torch.Tensor) -> torch.Tensor:
    """旋转向量 → 欧拉角 [yaw, pitch, roll] (ZYX)。用于式(18)(20)的 e。可微。"""
    R = so3_exp(r)
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1 + 1e-7, 1 - 1e-7))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return torch.stack([yaw, pitch, roll], dim=-1)


# ---------------------------------------------------------------------- numpy

def np_so3_exp(r: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(r))
    if theta < 1e-10:
        return np.eye(3) + _np_hat(r)
    k = r / theta
    K = _np_hat(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def np_so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    if abs(np.pi - theta) < 1e-6:  # 接近 π 的退化情形
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0, None))
        # 选最大分量定符号
        i = int(np.argmax(axis))
        if axis[i] > 0:
            axis = A[:, i] / (axis[i] + _EPS)
            axis = axis / (np.linalg.norm(axis) + _EPS)
        return axis * theta
    return w * theta / (2.0 * np.sin(theta))


def _np_hat(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)


def np_se3_from_rt(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np_so3_exp(np.asarray(r, dtype=float))
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def np_rt_from_se3(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np_so3_log(T[:3, :3]), T[:3, 3].copy()


def np_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def np_se3_inverse(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def euler_zyx_from_R(R: np.ndarray) -> np.ndarray:
    """→ [yaw, pitch, roll] (弧度, ZYX)。"""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([yaw, pitch, roll])


def R_from_euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    """R → 单位四元数 (w,x,y,z)。"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_mean_rotations(R_list: list, weights=None) -> np.ndarray:
    """加权旋转平均 (四元数特征向量法, Markley)。用于式(23)旋转分量。"""
    n = len(R_list)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    M = np.zeros((4, 4))
    q0 = _quat_from_R(R_list[0])
    for wi, R in zip(w, R_list):
        q = _quat_from_R(R)
        if q @ q0 < 0:  # 半球对齐
            q = -q
        M += wi * np.outer(q, q)
    vals, vecs = np.linalg.eigh(M)
    return _R_from_quat(vecs[:, -1])


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两旋转矩阵测地距离 (度)。"""
    cos = np.clip((np.trace(R_a.T @ R_b) - 1) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def euler_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """式(24) e_r: 欧拉角向量差的模 (度)。"""
    e = euler_zyx_from_R(R_est) - euler_zyx_from_R(R_gt)
    e = (e + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.linalg.norm(e)))
