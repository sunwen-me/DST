"""评估模块推理 (附录 A1) — fast/both 模式的全监督快速标定路径 (论文 III-C/III-E)。

用训练好的评估模块 (train.py 断点) 做 RegNet 式迭代精化: 每轮以当前外参估计
T_est 生成 LDP, 真实相机深度点云按原视角生成 CDP, 差分图 (式11,12) 前向得 ξ,
更新 T_est ← se3(ξ)·T_est; 逐帧得到候选 {T_i} 后按 式(20) 全监督打分,
multiframe.select_and_average (式22,23) 加权融合出 T*。

坐标约定推导 (与 augmentation.py 模块 docstring 严格一致):
训练时 LDP 在 T_lidar 位姿投影、CDP 在 T_cam 视角投影, 网络学习的是把 LDP
投影系三维点变换到 CDP 投影系的 T_gt_virtual = T_cam·T_lidar^{-1} (式4)。
推理时 LDP 用当前估计 T_est 投影 (承担 T_lidar 角色); CDP 用真实相机深度
点云按原视角投影 (ΔT_cam=I), 其投影系即真实外参 T_true 定义的相机系
(承担 T_cam 角色)。故网络输出 ξ ≈ T_true·T_est^{-1}, 更新规则为左乘:
    T_est ← se3(ξ) @ T_est。

全程 torch.no_grad() (纯推理, 无反传); 数据准备 numpy, 网络前向 torch。
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from . import multiframe
from .difference_map import build_difference_map
from .geometry import np_rt_from_se3, np_se3_from_rt
from .models import EvaluationDB, EvaluationSB
from .projection import generate_cdp, generate_ldp, virtual_camera_intrinsics
from .self_supervised import MIN_CAMERA_POINTS, _xi_str, camera_cloud

__all__ = ["load_eva", "eva_refine", "eva_calibrate_frames"]

# 迭代精化的收敛提前停判据 (契约 A1): 本轮增量 ||r|| < 0.05° 且 ||t|| < 1 mm
CONV_ROT_RAD = math.radians(0.05)
CONV_TRANS_M = 1.0e-3

# 虚拟相机参数不一致的警告按参数组合只打印一次 (逐帧调用时避免刷屏)
_VC_WARNED: set = set()


# ------------------------------------------------------------------ 配置解析

def _cfg_get(cfg, section: str, key: str, default):
    """cfg.<section>.<key>, 任一级缺失时返回默认值。"""
    return getattr(getattr(cfg, section, SimpleNamespace()), key, default)


def _vc_get(vc, key: str, default):
    """断点 virtual_camera 兼容 dict 与 SimpleNamespace 两种存法。"""
    if isinstance(vc, dict):
        return vc.get(key, default)
    return getattr(vc, key, default)


def _arch_of(model) -> str:
    """arch 优先从断点 meta 读 (load_eva 附着于 model), 无则按模型类型推断。"""
    meta = getattr(model, "_dst_eva_meta", None)
    if isinstance(meta, dict) and "arch" in meta:
        return str(meta["arch"]).lower()
    return "db" if isinstance(model, EvaluationDB) else "sb"


def _resolve_virtual_camera(model, cfg) -> tuple[np.ndarray, tuple]:
    """解析虚拟相机参数: 优先断点里的 virtual_camera, 无则用 cfg (契约 A1)。

    断点值与 cfg 不一致时打印警告并以断点为准 —— 网络权重是在断点参数的
    投影分辨率/焦距下训练的, 换参数会破坏其输入分布。返回 (K (3,3), size (H,W))。
    """
    vcc = getattr(cfg, "virtual_camera", SimpleNamespace())
    cfg_h = int(getattr(vcc, "height", 256))
    cfg_w = int(getattr(vcc, "width", 512))
    cfg_f = float(getattr(vcc, "focal", 600.0))
    meta = getattr(model, "_dst_eva_meta", None)
    vc = meta.get("virtual_camera") if isinstance(meta, dict) else None
    if vc is not None:
        h = int(_vc_get(vc, "height", cfg_h))
        w = int(_vc_get(vc, "width", cfg_w))
        f = float(_vc_get(vc, "focal", cfg_f))
        if (h, w, f) != (cfg_h, cfg_w, cfg_f):
            key = (h, w, f, cfg_h, cfg_w, cfg_f)
            if key not in _VC_WARNED:
                _VC_WARNED.add(key)
                print(f"[eva_infer] 警告: 断点虚拟相机 (H={h}, W={w}, f={f:g}) 与配置 "
                      f"(H={cfg_h}, W={cfg_w}, f={cfg_f:g}) 不一致, 以断点为准")
    else:
        h, w, f = cfg_h, cfg_w, cfg_f
    size = (h, w)
    return virtual_camera_intrinsics(f, size), size


# ------------------------------------------------------------------ 断点加载

def load_eva(ckpt_path: str, device) -> tuple[torch.nn.Module, dict]:
    """加载 train.py 断点, 返回 (eval() 模式评估模块, meta dict) (契约 A1)。

    断点键: 'arch' (sb|db, 必需), 'model' (state_dict, 必需),
    'virtual_camera' (可选, {'height','width','focal'}, 附录 A2)。
    meta 至少含 arch; 断点存有 virtual_camera 时一并放入 meta, 并附着在
    model._dst_eva_meta 上, 供 eva_refine 解析虚拟相机参数与 arch
    (与 cfg 不一致时的警告见 _resolve_virtual_camera)。
    """
    device = torch.device(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise KeyError(f"断点 {ckpt_path} 缺少键 'model' (期望 train.py 保存的 dict 断点)")
    if "arch" not in state:
        raise KeyError(f"断点 {ckpt_path} 缺少键 'arch' (应为 sb 或 db)")
    arch = str(state["arch"]).lower()
    if arch == "sb":
        model: torch.nn.Module = EvaluationSB()
    elif arch == "db":
        model = EvaluationDB()
    else:
        raise ValueError(f"断点 arch={arch!r} 未知 (应为 sb 或 db)")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    meta: dict = {"arch": arch}
    for k in ("epoch", "loss", "virtual_camera"):
        if k in state:
            meta[k] = state[k]
    model._dst_eva_meta = meta  # 附着 meta (arch/virtual_camera), 供推理端读取
    extra = "".join(
        f", {k}={meta[k]}" for k in ("epoch",) if k in meta)
    print(f"[eva_infer] 加载断点 {ckpt_path}: arch={arch}{extra}, "
          f"virtual_camera={'断点' if 'virtual_camera' in meta else '配置'}")
    return model, meta


# ------------------------------------------------------------------ 单步前向

def _eva_step(points_lidar: np.ndarray, cdp: np.ndarray, T_est: np.ndarray,
              model, arch: str, K: np.ndarray, size: tuple,
              e_tar: float, device: torch.device) -> np.ndarray:
    """单次评估前向: 以 T_est 生成 LDP, 与缓存的 CDP 组网络输入 → ξ (6,) numpy。

    SB 输入差分图 (1,3,H,W) (式11,12); DB 输入 (cdp, ldp) 各 (1,1,H,W)。
    调用方须处于 torch.no_grad() 上下文。
    """
    ldp = generate_ldp(points_lidar, T_est, K, size)  # T_lidar 角色 (模块 docstring)
    if arch == "db":
        cdp_t = torch.from_numpy(np.ascontiguousarray(cdp)).float() \
            .reshape(1, 1, *size).to(device)
        ldp_t = torch.from_numpy(np.ascontiguousarray(ldp)).float() \
            .reshape(1, 1, *size).to(device)
        xi = model(cdp_t, ldp_t)
    else:
        diff = build_difference_map(ldp, cdp, e_tar=e_tar)  # 式(11)(12)
        xi = model(torch.from_numpy(np.ascontiguousarray(diff)).float()
                   .unsqueeze(0).to(device))
    return xi.detach().reshape(-1).cpu().numpy().astype(np.float64)


def _refine_from_cdp(points_lidar: np.ndarray, cdp: np.ndarray, T0: np.ndarray,
                     model, arch: str, K: np.ndarray, size: tuple,
                     e_tar: float, device: torch.device,
                     n_iters: int) -> tuple[np.ndarray, np.ndarray]:
    """迭代精化内核 (CDP 已备好): 每轮 ξ=网络(LDP(T_est), CDP), T_est←se3(ξ)·T_est。

    收敛提前停 (契约 A1): 本轮增量 ||r||<0.05° 且 ||t||<1mm。
    返回 (T_refined (4,4), xi_last (6,) — 最后一轮网络输出)。
    """
    T = np.asarray(T0, dtype=np.float64).reshape(4, 4).copy()
    xi_last = np.zeros(6, dtype=np.float64)
    with torch.no_grad():
        for _ in range(max(int(n_iters), 0)):
            xi_last = _eva_step(points_lidar, cdp, T, model, arch, K, size,
                                e_tar, device)
            T = np_se3_from_rt(xi_last[:3], xi_last[3:]) @ T  # ξ≈T_true·T_est^{-1}, 左乘
            if (np.linalg.norm(xi_last[:3]) < CONV_ROT_RAD
                    and np.linalg.norm(xi_last[3:]) < CONV_TRANS_M):
                break  # 增量已小于 0.05°/1mm, 提前停
    return T, xi_last


# ------------------------------------------------------------------ 单帧精化

def eva_refine(frame, T_est: np.ndarray, model, cfg, device,
               n_iters: int = 3,
               cam_cloud: Optional[np.ndarray] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """单帧迭代精化 (RegNet 式, 契约 A1)。每轮:

        LDP = generate_ldp(points_lidar, T_est, K_virtual, size)   # T_lidar 角色
        CDP = generate_cdp(cam_cloud, I, K_virtual, size)          # ΔT_cam=I
        D   = build_difference_map(LDP, CDP, e_tar)                # 式(11)(12)
        ξ   = model(D) (SB) 或 model(cdp, ldp) (DB)                # ≈ T_true·T_est^{-1}
        T_est ← se3(ξ) @ T_est                                     # 左乘更新

    CDP 与 T_est 无关 (真实相机原视角), 循环外只投影一次; 收敛判据
    ||r||<0.05° 且 ||t||<1mm 提前停。虚拟相机参数优先取断点 (见
    _resolve_virtual_camera), e_tar 取 cfg.difference_map.e_tar。

    参数:
        frame: self_supervised.Frame (points_lidar/depth/K/...)。
        T_est: (4,4) 当前外参估计 T_cam_lidar。
        model: load_eva 返回的评估模块 (或同构模型)。
        cam_cloud: 契约外可选缓存参数 — (M,3) 相机深度点云; 调用方
            (eva_calibrate_frames) 每帧只反投影一次并传入, None 时内部用
            camera_cloud(frame, cfg) 反投影 (depth_cloud_from_gemini)。
    返回:
        (T_refined (4,4) float64, xi_last (6,) float64 — 最后一轮网络输出)。
    """
    device = torch.device(device)
    arch = _arch_of(model)
    K, size = _resolve_virtual_camera(model, cfg)
    e_tar = float(_cfg_get(cfg, "difference_map", "e_tar", 0.1))
    if cam_cloud is None:
        cam_cloud = camera_cloud(frame, cfg)  # 每帧只需反投影一次 (调用方可缓存)
    cdp = generate_cdp(cam_cloud, np.eye(4), K, size)  # 原视角 ΔT_cam=I
    model.eval()
    return _refine_from_cdp(frame.points_lidar, cdp, T_est, model, arch,
                            K, size, e_tar, device, n_iters)


# ------------------------------------------------------------------ 多帧标定

def eva_calibrate_frames(frames: list, T_init: np.ndarray, model, cfg, device,
                         n_iters: int = 3) -> dict:
    """快速全监督标定 (契约 A1): 逐帧精化 → 式(20) 打分 → 式(22)(23) 融合。

    流程:
      1. 每帧相机深度点云只反投影一次 (缓存), 从 T_init 出发 eva_refine 得
         候选 T_i (相机点数 < MIN_CAMERA_POINTS 的无效帧候选取 T_init);
      2. 式(20) 打分: 对每个 T_i 以其为输入位姿再前向一次得评估结果
         T'_i = se3(ξ'_i)·T_i, s_i = exp(-(a·||e'_i - e_i||₂ + ||t'_i - t_i||₂)),
         a = cfg.loss.a (0.1), e/e' 为 ZYX 欧拉角向量 (复用
         multiframe.score_full_supervised, 其 xi_eva 参数取 T'_i 的 [r,t]);
         无效帧得分置 0;
      3. multiframe.select_and_average(T_list, scores,
         x=cfg.multiframe.selection_ratio) 加权融合得 T*。

    返回 dict:
        'T_cam_lidar': (4,4) T*;
        'per_frame_T': [len(frames) 个 (4,4)] 逐帧候选;
        'scores':      (n,) float64 式(20) 得分;
        'xi_eva':      (6,) T* 的 ξ=[r,t] — 供 both 模式作式(18)(19) 先验。
    """
    device = torch.device(device)
    T_init = np.asarray(T_init, dtype=np.float64).reshape(4, 4)
    arch = _arch_of(model)
    K, size = _resolve_virtual_camera(model, cfg)
    e_tar = float(_cfg_get(cfg, "difference_map", "e_tar", 0.1))
    a = float(_cfg_get(cfg, "loss", "a", 0.1))
    model.eval()

    T_list: list[np.ndarray] = []
    xi_eva_list: list[np.ndarray] = []
    valid = np.zeros(len(frames), dtype=bool)
    with torch.no_grad():
        for i, fr in enumerate(frames):
            Q = camera_cloud(fr, cfg)  # 每帧只反投影一次, 精化与打分共用
            if Q.shape[0] < MIN_CAMERA_POINTS:
                print(f"[eva_infer] 警告: 帧 {i} 相机深度点数 {Q.shape[0]} < "
                      f"{MIN_CAMERA_POINTS}, 候选取 T_init, 得分置 0")
                T_list.append(T_init.copy())
                xi_eva_list.append(np.concatenate(np_rt_from_se3(T_init)))
                continue
            cdp = generate_cdp(Q, np.eye(4), K, size)  # 原视角, 该帧内不变
            T_i, _ = _refine_from_cdp(fr.points_lidar, cdp, T_init, model,
                                      arch, K, size, e_tar, device, n_iters)
            # 式(20) 评估结果: 以 T_i 为输入位姿再前向一次得 T'_i
            xi_p = _eva_step(fr.points_lidar, cdp, T_i, model, arch, K, size,
                             e_tar, device)
            T_prime = np_se3_from_rt(xi_p[:3], xi_p[3:]) @ T_i
            T_list.append(T_i)
            xi_eva_list.append(np.concatenate(np_rt_from_se3(T_prime)))
            valid[i] = True
    if not valid.any():
        print("[eva_infer] 警告: 所有帧相机深度点云均无效, 结果退化为 T_init")

    scores = multiframe.score_full_supervised(T_list, xi_eva_list, a=a)  # 式(20)
    scores[~valid] = 0.0
    T_star = multiframe.select_and_average(
        T_list, scores,
        x=float(_cfg_get(cfg, "multiframe", "selection_ratio", 0.3)),
        weighting=str(_cfg_get(cfg, "multiframe", "weighting", "score")))  # 式(22)(23)

    r_star, t_star = np_rt_from_se3(T_star)
    xi_eva = np.concatenate([r_star, t_star])
    print(f"[eva_infer] 快速标定完成 ({int(valid.sum())}/{len(frames)} 帧有效): "
          f"{_xi_str(xi_eva)}")
    return {
        "T_cam_lidar": T_star,
        "per_frame_T": T_list,
        "scores": scores,
        "xi_eva": xi_eva,
    }
