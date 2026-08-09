"""在线动态标定的纯 NumPy 状态管理。

ROS2 订阅、CUDA 工作进程和 TF 发布位于 ``scripts/online_calib_node.py``；
本模块只负责候选外参的质量门控、突变确认和 SE(3) 平滑，因而可以脱离
ROS2 做确定性单元测试。

在线窗口内仍遵循论文假设：LiDAR 与相机之间的外参保持不变。窗口之间允许
外参因震动、碰撞或主动调整发生变化；大幅跳变需要连续候选确认后才重锚定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """两个旋转矩阵的 SO(3) 测地距离，单位度。"""
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    c = np.clip((np.trace(R_a.T @ R_b) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转 ROS 顺序单位四元数 ``[x,y,z,w]``。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
            q = np.array([
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s,
            ])
        elif i == 1:
            s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
            q = np.array([
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s,
            ])
        else:
            s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
            q = np.array([
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
                (R[1, 0] - R[0, 1]) / s,
            ])
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("旋转矩阵无法转换为有效四元数")
    return q / n


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """ROS 顺序四元数 ``[x,y,z,w]`` 转旋转矩阵。"""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("四元数范数为零或非有限")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_se3(T_old: np.ndarray, T_new: np.ndarray, alpha: float) -> np.ndarray:
    """平移线性插值、旋转 SLERP；``alpha=1`` 完全采用新外参。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    A = np.asarray(T_old, dtype=np.float64).reshape(4, 4)
    B = np.asarray(T_new, dtype=np.float64).reshape(4, 4)
    qa = matrix_to_quaternion_xyzw(A[:3, :3])
    qb = matrix_to_quaternion_xyzw(B[:3, :3])
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = qa + alpha * (qb - qa)
        q /= np.linalg.norm(q)
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        q = (np.sin((1.0 - alpha) * theta) / sin_theta) * qa
        q += (np.sin(alpha * theta) / sin_theta) * qb
        q /= np.linalg.norm(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_matrix(q)
    T[:3, 3] = (1.0 - alpha) * A[:3, 3] + alpha * B[:3, 3]
    return T


def validate_se3(T: np.ndarray, atol: float = 1e-3) -> tuple[bool, str]:
    """检查有限性、底行、正交性和 ``det(R)=+1``。"""
    try:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return False, "候选外参不是 4x4 矩阵"
    if not np.all(np.isfinite(T)):
        return False, "候选外参包含 NaN/Inf"
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        return False, "候选外参底行不是 [0,0,0,1]"
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        return False, "候选旋转矩阵不正交"
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        return False, "候选旋转矩阵行列式不是 +1"
    return True, "ok"


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    transform: Optional[np.ndarray]
    rotation_jump_deg: float
    translation_jump_m: float
    final_cd: float
    reanchored: bool = False


class OnlineExtrinsicGate:
    """在线候选门控器。

    小变化立即按 ``smoothing_alpha`` 平滑；超过单次跳变阈值的候选不直接发布，
    只有连续 ``reanchor_confirmations`` 个候选彼此一致时才认为外参确实发生变化。
    """

    def __init__(
        self,
        *,
        max_final_cd: float = 0.15,
        max_rotation_jump_deg: float = 5.0,
        max_translation_jump_m: float = 0.30,
        smoothing_alpha: float = 0.35,
        reanchor_confirmations: int = 2,
        confirmation_rotation_deg: float = 1.0,
        confirmation_translation_m: float = 0.05,
    ):
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha 必须在 (0,1] 内")
        if reanchor_confirmations < 1:
            raise ValueError("reanchor_confirmations 必须 >=1")
        self.max_final_cd = float(max_final_cd)
        self.max_rotation_jump_deg = float(max_rotation_jump_deg)
        self.max_translation_jump_m = float(max_translation_jump_m)
        self.smoothing_alpha = float(smoothing_alpha)
        self.reanchor_confirmations = int(reanchor_confirmations)
        self.confirmation_rotation_deg = float(confirmation_rotation_deg)
        self.confirmation_translation_m = float(confirmation_translation_m)
        self._current: Optional[np.ndarray] = None
        self._pending: Optional[np.ndarray] = None
        self._pending_count = 0

    @property
    def current(self) -> Optional[np.ndarray]:
        return None if self._current is None else self._current.copy()

    @property
    def pending_count(self) -> int:
        return self._pending_count

    def reset(self) -> None:
        self._current = None
        self._pending = None
        self._pending_count = 0

    def _decision(
        self,
        accepted: bool,
        reason: str,
        rot: float,
        trans: float,
        cd: float,
        *,
        reanchored: bool = False,
    ) -> GateDecision:
        return GateDecision(
            accepted=accepted,
            reason=reason,
            transform=self.current,
            rotation_jump_deg=float(rot),
            translation_jump_m=float(trans),
            final_cd=float(cd),
            reanchored=reanchored,
        )

    def update(self, T_candidate: np.ndarray, final_cd: float) -> GateDecision:
        ok, why = validate_se3(T_candidate)
        cd = float(final_cd)
        if not ok:
            return self._decision(False, why, np.inf, np.inf, cd)
        if not np.isfinite(cd):
            return self._decision(False, "final_cd 非有限", np.inf, np.inf, cd)
        if self.max_final_cd > 0.0 and cd > self.max_final_cd:
            return self._decision(
                False,
                f"final_cd={cd:.5f} 超过阈值 {self.max_final_cd:.5f}",
                np.inf,
                np.inf,
                cd,
            )

        candidate = np.asarray(T_candidate, dtype=np.float64).reshape(4, 4).copy()
        if self._current is None:
            self._current = candidate
            self._pending = None
            self._pending_count = 0
            return self._decision(True, "首个有效候选", 0.0, 0.0, cd)

        rot = rotation_distance_deg(self._current[:3, :3], candidate[:3, :3])
        trans = float(np.linalg.norm(self._current[:3, 3] - candidate[:3, 3]))
        ordinary = (
            rot <= self.max_rotation_jump_deg
            and trans <= self.max_translation_jump_m
        )
        if ordinary:
            self._current = interpolate_se3(
                self._current, candidate, self.smoothing_alpha)
            self._pending = None
            self._pending_count = 0
            return self._decision(True, "候选通过并平滑", rot, trans, cd)

        if self._pending is None:
            self._pending = candidate
            self._pending_count = 1
        else:
            pending_rot = rotation_distance_deg(
                self._pending[:3, :3], candidate[:3, :3])
            pending_trans = float(np.linalg.norm(
                self._pending[:3, 3] - candidate[:3, 3]))
            if (
                pending_rot <= self.confirmation_rotation_deg
                and pending_trans <= self.confirmation_translation_m
            ):
                n = self._pending_count
                self._pending = interpolate_se3(
                    self._pending, candidate, 1.0 / float(n + 1))
                self._pending_count += 1
            else:
                self._pending = candidate
                self._pending_count = 1

        if self._pending_count >= self.reanchor_confirmations:
            self._current = self._pending.copy()
            self._pending = None
            count = self._pending_count
            self._pending_count = 0
            return self._decision(
                True,
                f"连续 {count} 个一致候选，确认外参突变并重锚定",
                rot,
                trans,
                cd,
                reanchored=True,
            )

        return self._decision(
            False,
            f"候选跳变 {rot:.2f}°/{trans:.3f}m，"
            f"等待确认 {self._pending_count}/{self.reanchor_confirmations}",
            rot,
            trans,
            cd,
        )
