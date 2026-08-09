"""评估模块与位姿估计器损失 (论文式(13)-(19))。

记号 (与论文 III-C/III-D 一致):
    (R_cam, t_cam)     — 相机侧扰动位姿 (双侧增广式(3) 中的 T_cam 分量);
    (R_lidar, t_lidar) — LiDAR 侧扰动位姿;
    (R, t)             — 网络预测的旋转/平移;
    ξ = [r, t] ∈ R^6   — 位姿向量, r 为 so(3) 旋转向量, t 为平移 (米)。
"""
from __future__ import annotations

import numpy as np
import torch

from .chamfer import truncated_chamfer
from .geometry import se3_from_rt, so3_exp, transform_points

_EPS = 1e-12  # 零向量处 sqrt 的梯度保护


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """||x||_2, 加 _EPS 防止零向量处梯度 NaN (数值影响可忽略)。"""
    return torch.sqrt((x * x).sum(dim=dim) + _EPS)


def _to_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    """把 numpy/list/tensor 输入统一为与 ref 同 dtype/device 的张量。"""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(np.asarray(x), dtype=ref.dtype, device=ref.device)


def rotation_loss(R_cam: torch.Tensor, R_pred_lidar: torch.Tensor) -> torch.Tensor:
    """式(13): L_rgt = ||R_cam · (R·R_lidar)^{-1} - I||_{1,1} (逐元素绝对值之和)。

    参数:
        R_cam: (...,3,3) 相机侧扰动旋转。
        R_pred_lidar: (...,3,3) 组合旋转 R·R_lidar —— 预测旋转 R 与 LiDAR 侧
            扰动旋转 R_lidar 的乘积, 由调用方 (如 eva_total_loss) 先算好传入。
            旋转矩阵的逆即转置。
    批量输入时对 batch 维取平均。
    """
    M = R_cam @ R_pred_lidar.transpose(-1, -2)
    I = torch.eye(3, dtype=M.dtype, device=M.device).expand_as(M)
    return (M - I).abs().sum(dim=(-2, -1)).mean()


def translation_loss(t_cam: torch.Tensor, t_lidar: torch.Tensor,
                     t_pred: torch.Tensor) -> torch.Tensor:
    """式(14): L_tgt = ||t_cam - (t_lidar + t)||_2。批量时对 batch 取平均。"""
    return _safe_norm(t_cam - (t_lidar + t_pred)).mean()


def cloud_loss(P: torch.Tensor, R_pred: torch.Tensor, t_pred: torch.Tensor,
               R_cam: torch.Tensor, t_cam: torch.Tensor,
               R_lidar: torch.Tensor, t_lidar: torch.Tensor) -> torch.Tensor:
    """式(15): L_cloud = Σ_{p∈P} ||R(R_lidar·p + t_lidar) + t - (R_cam·p + t_cam)||_2。

    P: (...,N,3)。按论文对点 *求和*; 批量输入时再对 batch 取平均。
    """
    p_l = P @ R_lidar.transpose(-1, -2) + t_lidar.unsqueeze(-2)   # R_lidar·p + t_lidar
    lhs = p_l @ R_pred.transpose(-1, -2) + t_pred.unsqueeze(-2)   # R(·) + t
    rhs = P @ R_cam.transpose(-1, -2) + t_cam.unsqueeze(-2)       # R_cam·p + t_cam
    return _safe_norm(lhs - rhs).sum(dim=-1).mean()


def eva_total_loss(R_pred: torch.Tensor, t_pred: torch.Tensor,
                   R_cam: torch.Tensor, t_cam: torch.Tensor,
                   R_lidar: torch.Tensor, t_lidar: torch.Tensor,
                   P: torch.Tensor) -> torch.Tensor:
    """式(16): L_eva = L_rgt + L_tgt + L_cloud (评估模块训练总损失)。

    参数即式(13)-(15)所需全部量: 预测 (R,t)、相机/LiDAR 侧扰动位姿、点云 P。
    """
    l_rgt = rotation_loss(R_cam, R_pred @ R_lidar)          # 式(13)
    l_tgt = translation_loss(t_cam, t_lidar, t_pred)        # 式(14)
    l_cloud = cloud_loss(P, R_pred, t_pred,
                         R_cam, t_cam, R_lidar, t_lidar)    # 式(15)
    return l_rgt + l_tgt + l_cloud


def _euler_zyx_from_R(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 → ZYX 欧拉角 [yaw, pitch, roll] (可微; 提取式与
    geometry.euler_zyx_from_r 一致, 此处直接吃 R 免去轴角往返)。"""
    pitch = torch.asin((-R[..., 2, 0]).clamp(-1 + 1e-7, 1 - 1e-7))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return torch.stack([yaw, pitch, roll], dim=-1)


def eva_score_loss(xi_pred: torch.Tensor, xi_eva: torch.Tensor,
                   a: float = 0.1) -> torch.Tensor:
    """式(18): L'_eva = a·||e_eva - e||_2 + ||t_eva - t||_2。

    e 为 ZYX 欧拉角向量; t 为 ξ 的平移部分。a=0.1 (论文尺度参数 a=0.1/1)。
    工程细节: 旋转差取相对旋转 R_eva·R_predᵀ 的欧拉角向量 —— 小差异下与
    e_eva-e 一阶等价 (语义同式(18)), 且在任意安装姿态下良态: 绝对欧拉角在
    pitch=±90° 万向节锁附近病态 (本项目名义前视安装恰在此处), 差值被放大
    数百倍且 asin 截断处梯度爆炸, 会把优化钉死在先验上无法被 L_CD 精化;
    相对旋转近恒等, 欧拉提取处处良态, 各角落在主值区间, 天然免去包角。
    批量时对 batch 取平均。
    """
    R_pred = so3_exp(xi_pred[..., :3])
    R_eva = so3_exp(xi_eva[..., :3])
    de = _euler_zyx_from_R(R_eva @ R_pred.transpose(-1, -2))  # 相对空间
    dt = xi_eva[..., 3:] - xi_pred[..., 3:]
    return (a * _safe_norm(de) + _safe_norm(dt)).mean()


def pose_estimator_loss(xi: torch.Tensor, P: torch.Tensor, Q: torch.Tensor,
                        xi_init=None, xi_eva=None,
                        a: float = 0.1, trunc: float = 1.0
                        ) -> tuple[torch.Tensor, dict]:
    """式(19): L_pe = L_t_ini + L_CD + L'_eva。

    分项:
    - L_CD: P 经 T=se3_from_rt(ξ) 变换为 P̂ 后与 Q 的截断 Chamfer 距离
      (式17, α=β=0.5, 截断阈值 trunc), 自监督主项。
    - L_t_ini = ||t_init - t||_2 (与式14同形): 仅约束 ξ 的平移部分, 把优化
      拉在初值附近; xi_init=None (无初值) 时该项为 0。
    - L'_eva: 式(18) 评估模块引导项; xi_eva=None (无评估模块) 时为 0。

    参数:
        xi: (6,) torch 张量 (通常 requires_grad), [r, t]。
        P:  (N,3) LiDAR 系点云 (torch);  Q: (M,3) 相机系目标点云 (torch)。
        xi_init / xi_eva: (6,) numpy 或 torch, 可为 None。
    返回:
        (total, {'cd': float, 'tini': float, 'eva': float}) —— 分项为
        detach 后的 float, 便于日志记录。
    """
    xi = torch.as_tensor(xi)
    T = se3_from_rt(xi[..., :3], xi[..., 3:])       # ξ → SE(3), 可微
    P_hat = transform_points(T, P)                   # P̂ = T·P
    l_cd = truncated_chamfer(P_hat, Q, trunc=trunc)  # 式(17) 截断版

    zero = xi.new_zeros(())
    l_tini = zero
    if xi_init is not None:
        xi_init_t = _to_tensor_like(xi_init, xi)
        l_tini = _safe_norm(xi_init_t[..., 3:] - xi[..., 3:]).mean()

    l_eva = zero
    if xi_eva is not None:
        xi_eva_t = _to_tensor_like(xi_eva, xi)
        l_eva = eva_score_loss(xi, xi_eva_t, a=a)    # 式(18)

    total = l_tini + l_cd + l_eva                    # 式(19)
    parts = {"cd": float(l_cd.detach()),
             "tini": float(l_tini.detach()),
             "eva": float(l_eva.detach())}
    return total, parts
