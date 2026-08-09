"""自监督一键标定核心 (论文 III-C-4 位姿估计器 + III-D-2 全自监督标定)。

流程 (arXiv 2601.01188):
1. `load_frames` 读取 capture_data.py 保存的 frame_*.npz;
2. `coarse_search_init` 多起点粗搜索 (工程补充: Mid-360 为 360° 视场,
   论文假设已有粗初值, 此处用 yaw×pitch 网格 + 单向截断 Chamfer 自动求初值);
3. `optimize_pose` 用 SimplePoseEstimator(常数输入 MLP) + Adam 做由粗到细的
   自监督优化, 损失为截断 Chamfer 距离 式(17), 可选叠加 式(18) L'_eva 与
   初值平移约束 L_t_ini, 合成 式(19) L_pe = L_t_ini + L_CD + L'_eva;
4. `calibrate_self_supervised` 一键入口, 串联粗搜索 → 优化 → 多帧加权
   (multiframe, 式(20)-(23)) 得到最终外参 T*。

约定: T_cam_lidar @ p_lidar = p_cam; ξ=[r,t]∈R^6, r 为 so(3) 轴角。
全程 torch float32, CPU/CUDA 自适应。
"""
from __future__ import annotations

import copy
import glob
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from .geometry import (
    euler_zyx_from_R,
    np_rt_from_se3,
    np_se3_from_rt,
    np_so3_log,
    np_transform,
    R_from_euler_zyx,
    se3_from_rt,
    transform_points,
)
from .projection import depth_cloud_from_gemini
from .chamfer import truncated_chamfer
from .losses import eva_score_loss
from .models import SimplePoseEstimator

# 相机深度点云的最小有效点数 (稳健性: 低于此值的帧告警并跳过)
MIN_CAMERA_POINTS = 500
# 裁剪/下采样后参与优化所需的最少点数 (工程阈值)
MIN_STAGE_POINTS = 30


# ------------------------------------------------------------------ 数据结构

@dataclass
class Frame:
    """单帧采集数据 (契约见 INTERFACES.md)。"""
    points_lidar: np.ndarray          # (N,3) LiDAR 系, float32
    depth: np.ndarray                 # (H,W) 米, 已对齐彩色, float32
    K: np.ndarray                     # (3,3) 彩色内参
    rgb: Optional[np.ndarray]         # (H,W,3) uint8, 可为 None
    stamp: float


def load_frames(data_dir: str) -> list[Frame]:
    """读取 capture_data.py 保存的 frame_*.npz (键: points_lidar, depth, K, rgb, stamp)。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"数据目录 {data_dir} 中未找到 frame_*.npz — 请先运行 scripts/capture_data.py 采集数据。")
    frames: list[Frame] = []
    for p in paths:
        with np.load(p, allow_pickle=True) as f:
            pts = np.asarray(f["points_lidar"], dtype=np.float32).reshape(-1, 3)
            depth = np.asarray(f["depth"], dtype=np.float32)
            K = np.asarray(f["K"], dtype=np.float64).reshape(3, 3)
            rgb = None
            if "rgb" in f.files:
                r = f["rgb"]
                # 兼容存入 None(object 标量) 或空数组的情形
                if r is not None and getattr(r, "size", 0) > 0 and r.ndim == 3:
                    rgb = np.asarray(r, dtype=np.uint8)
            stamp = float(np.asarray(f["stamp"]).reshape(-1)[0]) if "stamp" in f.files else 0.0
        frames.append(Frame(points_lidar=pts, depth=depth, K=K, rgb=rgb, stamp=stamp))
    print(f"[load_frames] 从 {data_dir} 读取 {len(frames)} 帧")
    return frames


# ------------------------------------------------------------------ 通用工具

def _device() -> torch.device:
    """优先 CUDA。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ss(cfg):
    """取 self_supervised 子配置 (兼容传入整体 cfg 或其子命名空间)。"""
    return getattr(cfg, "self_supervised", cfg)


def _cam_depth_range(cfg) -> tuple[float, float]:
    """Gemini335 可靠深度范围, 缺省 [0.25, 6.0] m。"""
    cam = getattr(getattr(cfg, "sensors", SimpleNamespace()), "camera", SimpleNamespace())
    return float(getattr(cam, "depth_min", 0.25)), float(getattr(cam, "depth_max", 6.0))


def voxel_downsample(points: np.ndarray, voxel: float,
                     max_points: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """numpy 体素下采样: 坐标//voxel 取唯一格子, 每格取质心; 超过 max_points 时随机截取。

    自实现, 不依赖 open3d。返回 (M,3) float32。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    if voxel is not None and voxel > 0:
        keys = np.floor(pts / float(voxel)).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inv).astype(np.float64)
        sums = np.stack([np.bincount(inv, weights=pts[:, k]) for k in range(3)], axis=1)
        pts = sums / counts[:, None]
    if max_points is not None and pts.shape[0] > int(max_points):
        rng = rng if rng is not None else np.random.default_rng(0)
        sel = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[sel]
    return np.ascontiguousarray(pts, dtype=np.float32)


def fov_crop_points(points_lidar: np.ndarray, T_cam_lidar: np.ndarray,
                    K: np.ndarray, img_hw: tuple, margin_deg: float,
                    depth_max: float, z_min: float = 0.05) -> np.ndarray:
    """按相机视场 (由真实 K 与深度图尺寸推 hfov/vfov, 外扩 margin_deg) 裁剪 LiDAR 点。

    用当前外参估计把 LiDAR 点变换到相机系后按
    z>z_min、|x|/z<tan(hfov/2+m)、|y|/z<tan(vfov/2+m)、z<depth_max+0.5 保留;
    返回 **LiDAR 系** 的子集 (后续优化仍以 ξ 变换它们)。
    """
    pts = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts.astype(np.float32)
    pc = np_transform(np.asarray(T_cam_lidar, dtype=np.float64), pts)
    H, W = int(img_hw[0]), int(img_hw[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    m = math.radians(float(margin_deg))
    tan_h = math.tan(min(math.atan((W * 0.5) / max(fx, 1e-6)) + m, math.radians(89.0)))
    tan_v = math.tan(min(math.atan((H * 0.5) / max(fy, 1e-6)) + m, math.radians(89.0)))
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    mask = ((z > z_min) & (z < float(depth_max) + 0.5)
            & (np.abs(x) < z * tan_h) & (np.abs(y) < z * tan_v))
    return np.ascontiguousarray(pts[mask], dtype=np.float32)


def camera_cloud(frame: Frame, cfg) -> np.ndarray:
    """由帧的对齐深度图生成相机系点云 Q_i (projection.depth_cloud_from_gemini)。"""
    d_min, d_max = _cam_depth_range(cfg)
    stride = int(getattr(_ss(cfg), "depth_stride", 2))  # 工程缺省 2, 之后还有体素下采样
    return depth_cloud_from_gemini(frame.depth, frame.K, d_min=d_min, d_max=d_max,
                                   stride=max(stride, 1))


def _to_t(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)


def _one_sided_truncated_cd(P: torch.Tensor, Q: torch.Tensor,
                            trunc: float = 2.0, chunk: int = 2048) -> torch.Tensor:
    """单向截断 Chamfer (P→Q, 式(17) 仅 α 项, 截断至 trunc²): 粗搜索评分用, 快。"""
    t2 = float(trunc) * float(trunc)
    total = P.new_zeros(())
    n = P.shape[0]
    for s in range(0, n, chunk):
        d = torch.cdist(P[s:s + chunk], Q)          # (c,M) 欧氏距离
        m = (d * d).min(dim=1).values               # 最近对平方距离
        total = total + torch.clamp(m, max=t2).sum()
    return total / max(n, 1)


def _xi_str(xi: np.ndarray) -> str:
    """ξ → 可读字符串 (欧拉角 deg + 平移 m)。"""
    T = np_se3_from_rt(xi[:3], xi[3:])
    e = np.degrees(euler_zyx_from_R(T[:3, :3]))
    t = T[:3, 3]
    return (f"euler_zyx=[{e[0]:+7.2f},{e[1]:+7.2f},{e[2]:+7.2f}] deg, "
            f"t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}] m")


# ------------------------------------------------------------------ 粗搜索初值

def coarse_search_init(frames: list, cfg) -> np.ndarray:
    """多起点粗搜索初值 (工程补充, 论文 III-D-2 假设已有粗初值)。

    基础对齐 R0 把 LiDAR 系 (x前 y左 z上) 转到相机光学系 (x右 y下 z前);
    候选 R = R0 @ Rz(yaw) @ Ry(pitch), yaw 均分 2π, pitch 取配置列表;
    每候选: 大体素下采样 + 视锥裁剪(候选位姿下的相机视场) 后,
    SimplePoseEstimator + Adam 少量迭代单向截断 Chamfer(P→Q) + 平移软约束,
    取终值最优者。若 cfg.init_T 非 None 则直接转 ξ 返回。

    修复说明 (集成测试发现的退化问题):
    1) 裁剪从 "z>0 半空间" 改为按候选位姿的相机视锥裁剪 —— 半空间内大量
       视场外的点没有对应, 会把正确候选的评分抬高, 掩盖真实对齐质量;
    2) 增加平移软约束 (t_bound/t_penalty): 单向截断 Chamfer 存在
       "整体平移塌缩到相机点云体积内" 的退化最优 (错误朝向也能把墙/地面
       平面平移贴到相机点云的平面上获得低分)。相机与雷达共装同一支架,
       粗搜索阶段平移不应远离 0, 超出 t_bound 的范数按二次惩罚;
    3) 终值评分改用对称截断 Chamfer (trunc=score_trunc, 缺省 0.5m):
       单向 P→Q 分辨不了 "平面贴平面" 的错误朝向 (它不惩罚相机点云中
       未被解释的结构, 如箱体/墙角); 对称评分的 Q→P 项使正确朝向
       (全部相机结构都有 LiDAR 对应) 得分显著低于塌缩解。
       优化迭代仍用单向 (快, 且塌缩解交由评分排除)。
    """
    init_T = getattr(cfg, "init_T", None)
    if init_T is not None:
        r, t = np_rt_from_se3(np.asarray(init_T, dtype=np.float64).reshape(4, 4))
        xi = np.concatenate([r, t]).astype(np.float64)
        print(f"[coarse] 使用外部先验 init_T: {_xi_str(xi)}")
        return xi

    ss = _ss(cfg)
    coarse = getattr(ss, "coarse", SimpleNamespace())
    yaw_grid = int(getattr(coarse, "yaw_grid", 12))
    pitch_list = list(getattr(coarse, "pitch_grid_deg", [-20, 0, 20]))
    voxel = float(getattr(coarse, "voxel", 0.4))
    iters = int(getattr(coarse, "iters_per_start", 40))
    lr = float(getattr(coarse, "lr", 0.03))
    max_points = int(getattr(coarse, "max_points", 4000))
    trunc = float(getattr(coarse, "trunc", 2.0))
    max_frames = int(getattr(coarse, "max_frames", 3))
    # 平移软约束 (修复: 防止单向 Chamfer 的平移塌缩退化解, 见函数 docstring)
    t_bound = float(getattr(coarse, "t_bound", 0.5))
    t_penalty = float(getattr(coarse, "t_penalty", 10.0))
    # 终值评分用的对称截断 Chamfer 截断距离 (比优化用 trunc 更紧, 提高分辨力)
    score_trunc = float(getattr(coarse, "score_trunc", 0.5))
    # 视锥裁剪余量: 粗搜索候选朝向误差大, 余量取得比精优化阶段更宽
    margin = float(getattr(coarse, "fov_margin_deg",
                           float(getattr(ss, "fov_margin_deg", 8.0)) + 4.0))
    _, depth_max = _cam_depth_range(cfg)
    device = _device()
    rng = np.random.default_rng(int(getattr(ss, "seed", 0)))

    # 选少量帧参与粗搜索 (均匀抽取), 预先做大体素下采样
    n = len(frames)
    sel = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
    P_ds, Q_t, metas = [], [], []
    for i in sel:
        fr = frames[i]
        P = voxel_downsample(fr.points_lidar, voxel, None, rng)
        Q = voxel_downsample(camera_cloud(fr, cfg), voxel, max_points, rng)
        if Q.shape[0] < MIN_STAGE_POINTS:
            print(f"[coarse] 警告: 帧 {i} 相机点云过少 ({Q.shape[0]}), 粗搜索跳过该帧")
            continue
        P_ds.append(P)
        Q_t.append(_to_t(Q, device))
        metas.append((fr.K, fr.depth.shape))  # 视锥裁剪需要各帧真实内参与图像尺寸
    if not P_ds:
        raise RuntimeError("粗搜索失败: 所有帧的相机深度点云均无效 — 请检查深度话题/量程配置。")

    # 基础对齐: LiDAR x前 y左 z上 → 相机光学 x右 y下 z前
    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    best_xi, best_score = None, float("inf")
    yaws = np.arange(yaw_grid) * (2.0 * np.pi / max(yaw_grid, 1))
    for yaw in yaws:
        for pitch_deg in pitch_list:
            Rc = R0 @ R_from_euler_zyx(float(yaw), 0.0, 0.0) \
                    @ R_from_euler_zyx(0.0, math.radians(float(pitch_deg)), 0.0)
            xi_cand = np.concatenate([np_so3_log(Rc), np.zeros(3)])
            # 修复: 视锥裁剪 (原为 z>0 半空间) —— 只保留候选位姿下相机视场内的
            # LiDAR 点, 视场外无对应的点不再抬高正确候选的评分。
            # 可见点不足的帧逐帧跳过 (与 _prepare_stage_data 语义一致),
            # 仅当该候选下没有任何有效帧时才跳过整个候选 —— 避免单个退化帧
            # (近距遮挡/吸光材质/盲区) 一票否决包括正确候选在内的全部候选。
            T_cand = np_se3_from_rt(xi_cand[:3], xi_cand[3:])
            P_cand, Q_cand = [], []
            for P, Q, (K_i, shape_i) in zip(P_ds, Q_t, metas):
                Pm = fov_crop_points(P, T_cand, K_i, shape_i, margin, depth_max)
                if Pm.shape[0] < MIN_STAGE_POINTS:
                    continue
                if Pm.shape[0] > max_points:
                    Pm = Pm[rng.choice(Pm.shape[0], size=max_points, replace=False)]
                P_cand.append(_to_t(Pm, device))
                Q_cand.append(Q)
            if not P_cand:
                print(f"[coarse] yaw={math.degrees(yaw):6.1f} pitch={pitch_deg:5.1f}: "
                      f"所有帧可见点均不足, 跳过")
                continue

            def _t_pen(xi):
                """平移软约束: 超出 t_bound 的范数按二次惩罚。"""
                return t_penalty * torch.relu(torch.norm(xi[3:]) - t_bound) ** 2

            # 少量 Adam 迭代 (正常反传), 单向截断 Chamfer P→Q + 平移软约束
            model = SimplePoseEstimator(init_xi=xi_cand.astype(np.float32)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(iters):
                xi = model()
                T = se3_from_rt(xi[:3], xi[3:])
                loss = torch.stack([
                    _one_sided_truncated_cd(transform_points(T, P), Q, trunc=trunc)
                    for P, Q in zip(P_cand, Q_cand)]).mean() + _t_pen(xi)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # 终值评分 (no_grad 重算): 对称截断 Chamfer (紧截断) + 平移惩罚。
            # Q→P 项惩罚未被 LiDAR 解释的相机结构, 排除 "平面贴平面" 塌缩解
            with torch.no_grad():
                xi = model()
                T = se3_from_rt(xi[:3], xi[3:])
                score = float(torch.stack([
                    truncated_chamfer(transform_points(T, P), Q,
                                      trunc=score_trunc)
                    for P, Q in zip(P_cand, Q_cand)]).mean() + _t_pen(xi))
            print(f"[coarse] yaw={math.degrees(yaw):6.1f} pitch={pitch_deg:5.1f} "
                  f"→ score={score:.4f}")
            if score < best_score:
                best_score = score
                best_xi = xi.detach().cpu().numpy().astype(np.float64)
    if best_xi is None:
        raise RuntimeError(
            "粗搜索失败: 所有候选朝向下 LiDAR 可见点均不足 — "
            "请确认相机与雷达确有共视区域, 或提供 --init_yaml 先验外参。")
    print(f"[coarse] 最优初值 score={best_score:.4f}, {_xi_str(best_xi)}")
    return best_xi


# ------------------------------------------------------------------ 自监督优化

def _prepare_stage_data(frames, Q_raw, T_cur, stage, cfg, rng, device):
    """按当前外参重裁剪 + 阶段体素下采样, 返回 (帧索引列表, P张量列表, Q张量列表)。"""
    ss = _ss(cfg)
    margin = float(getattr(ss, "fov_margin_deg", 8.0))
    _, depth_max = _cam_depth_range(cfg)
    voxel = float(stage.voxel)
    max_points = int(getattr(stage, "max_points", 20000))
    idx, P_t, Q_t = [], [], []
    for i, fr in enumerate(frames):
        if Q_raw[i] is None:
            continue
        P = fov_crop_points(fr.points_lidar, T_cur, fr.K, fr.depth.shape, margin, depth_max)
        P = voxel_downsample(P, voxel, max_points, rng)
        Q = voxel_downsample(Q_raw[i], voxel, max_points, rng)
        if P.shape[0] < MIN_STAGE_POINTS or Q.shape[0] < MIN_STAGE_POINTS:
            print(f"[optimize] 警告: 帧 {i} 裁剪/下采样后点数不足 "
                  f"(P={P.shape[0]}, Q={Q.shape[0]}), 本阶段跳过该帧")
            continue
        idx.append(i)
        P_t.append(_to_t(P, device))
        Q_t.append(_to_t(Q, device))
    return idx, P_t, Q_t


def _pe_loss(xi, T, P, Q, stage, cfg, xi_eva_t, t_ini_t, a, tini_w):
    """式(19) L_pe = L_t_ini + L_CD + L'_eva (无对应模块时相应项为 0)。

    L_CD: 式(17) 截断 Chamfer (α=β=0.5, trunc=stage.trunc);
    L'_eva: 式(18) a·||e_eva-e||_2 + ||t_eva-t||_2 (e 为 ZYX 欧拉角) ——
        复用 losses.eva_score_loss (旋转差取相对旋转的欧拉角: 万向节锁安全、
        免包角且含零向量安全范数, 见其 docstring);
    L_t_ini: 初值平移约束 ||t - t_ini||_2 (仅当外部给定先验时启用)。
    """
    ss = _ss(cfg)
    alpha = float(getattr(ss, "chamfer_alpha", 0.5))
    beta = float(getattr(ss, "chamfer_beta", 0.5))
    cd = torch.stack([
        truncated_chamfer(transform_points(T, Pi), Qi, alpha=alpha, beta=beta,
                          trunc=float(stage.trunc))
        for Pi, Qi in zip(P, Q)]).mean()
    parts = {"cd": float(cd.detach())}
    loss = cd
    if xi_eva_t is not None:  # 式(18)
        l_eva = eva_score_loss(xi, xi_eva_t, a=a)
        loss = loss + l_eva
        parts["eva"] = float(l_eva.detach())
    if t_ini_t is not None and tini_w > 0:  # L_t_ini
        l_tini = tini_w * torch.norm(xi[3:] - t_ini_t)
        loss = loss + l_tini
        parts["tini"] = float(l_tini.detach())
    return loss, parts


def optimize_pose(frames: list, xi_init: np.ndarray, cfg,
                  xi_eva: Optional[np.ndarray] = None,
                  log: Optional[list] = None) -> tuple[np.ndarray, list]:
    """自监督位姿优化 (论文 III-C-4 / III-D-2, 式(17)(19))。

    SimplePoseEstimator(init_xi) + Adam, 逐阶段 (cfg.self_supervised.stages)
    由粗到细: 每阶段开始用当前 ξ 重裁剪 LiDAR 点到相机视场并按阶段体素下采样,
    每迭代随机取 batch_frames 帧, loss = mean_i truncated_chamfer(T·P_i, Q_i)
    (+可选 式(18) L'_eva, +可选 L_t_ini) → 反传。总迭代数 ≥30 批 (论文 IV)。
    末段后对每帧深拷贝模型独立微调得到 T_i (多帧候选)。

    返回 (最终全局 ξ (6,), 每帧 T_i 列表 [len(frames) 个 4x4])。
    log 为可选外部列表, 逐迭代追加 {'stage','iter','loss',...} (契约外的兼容扩展)。
    """
    ss = _ss(cfg)
    device = _device()
    seed = int(getattr(ss, "seed", 0))
    torch.manual_seed(seed)
    stages = list(getattr(ss, "stages"))
    batch_frames = int(getattr(ss, "batch_frames", 2))
    history = log if log is not None else []

    total_iters = sum(int(st.iters) for st in stages)
    if total_iters < 30:
        print(f"[optimize] 警告: 总迭代数 {total_iters} < 30 (论文要求 ≥30 批)")

    # 每帧相机点云只算一次; 无效帧 (点数<500) 告警并置 None
    Q_raw: list[Optional[np.ndarray]] = []
    for i, fr in enumerate(frames):
        Q = camera_cloud(fr, cfg)
        if Q.shape[0] < MIN_CAMERA_POINTS:
            print(f"[optimize] 警告: 帧 {i} 相机深度点数 {Q.shape[0]} < {MIN_CAMERA_POINTS}, 跳过该帧")
            Q_raw.append(None)
        else:
            Q_raw.append(Q)
    if all(q is None for q in Q_raw):
        raise RuntimeError(
            "所有帧的相机深度点云均无效 (<500 点) — 诊断: 请检查 1) 深度话题是否有数据; "
            "2) depth_unit (mm/m) 判别是否正确; 3) depth_min/depth_max 量程配置。")

    model = SimplePoseEstimator(init_xi=np.asarray(xi_init, dtype=np.float32)).to(device)
    a = float(getattr(getattr(cfg, "loss", SimpleNamespace()), "a", 0.1))
    xi_eva_t = _to_t(np.asarray(xi_eva, dtype=np.float32), device) if xi_eva is not None else None
    # 初值平移约束: t_ini_weight>0 时启用。selfsup 缺省关闭 (粗搜索初值平移
    # 不可靠); both 模式由 calibrate._calibrate_with_eva 按契约 A3 (式19
    # 全三项) 在配置未显式给出该键时注入 1.0
    tini_w = float(getattr(ss, "t_ini_weight", 0.0))
    t_ini_t = _to_t(np.asarray(xi_init, dtype=np.float32)[3:], device) if tini_w > 0 else None

    for si, stage in enumerate(stages):
        rng = np.random.default_rng(seed + si + 1)
        with torch.no_grad():
            xi_cur = model().detach().cpu().numpy().astype(np.float64)
        T_cur = np_se3_from_rt(xi_cur[:3], xi_cur[3:])
        idx, P_t, Q_t = _prepare_stage_data(frames, Q_raw, T_cur, stage, cfg, rng, device)
        if not idx:
            raise RuntimeError(
                f"阶段 {si}: 当前外参估计下没有帧有足够的共视点 — 诊断: 初值可能严重错误, "
                "或相机与雷达无共视区域; 可尝试增大 fov_margin_deg 或提供先验外参。")
        opt = torch.optim.Adam(model.parameters(), lr=float(stage.lr))
        nb = min(batch_frames, len(idx))
        print(f"[optimize] 阶段 {si}: voxel={stage.voxel} trunc={stage.trunc} "
              f"lr={stage.lr} iters={stage.iters}, 有效帧 {len(idx)}/{len(frames)}")
        for it in range(int(stage.iters)):
            b = rng.choice(len(idx), size=nb, replace=False)
            xi = model()
            T = se3_from_rt(xi[:3], xi[3:])
            loss, parts = _pe_loss(xi, T, [P_t[j] for j in b], [Q_t[j] for j in b],
                                   stage, cfg, xi_eva_t, t_ini_t, a, tini_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append({"stage": si, "iter": it, "loss": float(loss.detach()), **parts})
        with torch.no_grad():
            xi_cur = model().detach().cpu().numpy().astype(np.float64)
        print(f"[optimize] 阶段 {si} 结束: loss={history[-1]['loss']:.4f}, {_xi_str(xi_cur)}")

    # ---- 末段后 per-frame finetune (论文 III-E 多帧候选 T_i) ----
    with torch.no_grad():
        xi_fin = model().detach().cpu().numpy().astype(np.float64)
    T_fin = np_se3_from_rt(xi_fin[:3], xi_fin[3:])
    last = stages[-1]
    rng = np.random.default_rng(seed + 1000)
    idx, P_t, Q_t = _prepare_stage_data(frames, Q_raw, T_fin, last, cfg, rng, device)
    pos = {i: j for j, i in enumerate(idx)}
    ft_iters = int(getattr(ss, "per_frame_finetune_iters", 40))
    ft_lr = float(getattr(ss, "per_frame_lr", 0.0008))
    per_frame_T: list[np.ndarray] = []
    for i in range(len(frames)):
        if i not in pos:
            # 无效帧/共视不足: 退回全局解 (保持与 frames 对齐)
            per_frame_T.append(T_fin.copy())
            continue
        j = pos[i]
        m_i = copy.deepcopy(model)
        opt = torch.optim.Adam(m_i.parameters(), lr=ft_lr)
        for _ in range(ft_iters):
            xi = m_i()
            T = se3_from_rt(xi[:3], xi[3:])
            loss, _ = _pe_loss(xi, T, [P_t[j]], [Q_t[j]], last, cfg,
                               xi_eva_t, t_ini_t, a, tini_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            xi_i = m_i().detach().cpu().numpy().astype(np.float64)
        per_frame_T.append(np_se3_from_rt(xi_i[:3], xi_i[3:]))
    print(f"[optimize] per-frame finetune 完成 ({len(idx)} 帧参与), 全局 {_xi_str(xi_fin)}")
    return xi_fin, per_frame_T


# ------------------------------------------------------------------ 一键入口

def _mean_final_cd(frames: list, T_star: np.ndarray, cfg) -> float:
    """用最终 T* 在全部有效帧上求截断 Chamfer 均值 (末段体素/截断参数)。"""
    ss = _ss(cfg)
    last = list(getattr(ss, "stages"))[-1]
    alpha = float(getattr(ss, "chamfer_alpha", 0.5))
    beta = float(getattr(ss, "chamfer_beta", 0.5))
    margin = float(getattr(ss, "fov_margin_deg", 8.0))
    _, depth_max = _cam_depth_range(cfg)
    device = _device()
    rng = np.random.default_rng(int(getattr(ss, "seed", 0)) + 2000)
    vals = []
    for i, fr in enumerate(frames):
        Q = camera_cloud(fr, cfg)
        if Q.shape[0] < MIN_CAMERA_POINTS:
            continue
        P = fov_crop_points(fr.points_lidar, T_star, fr.K, fr.depth.shape, margin, depth_max)
        P = voxel_downsample(P, float(last.voxel), int(last.max_points), rng)
        Q = voxel_downsample(Q, float(last.voxel), int(last.max_points), rng)
        if P.shape[0] < MIN_STAGE_POINTS or Q.shape[0] < MIN_STAGE_POINTS:
            continue
        with torch.no_grad():
            cd = truncated_chamfer(_to_t(np_transform(T_star, P.astype(np.float64)), device),
                                   _to_t(Q, device), alpha=alpha, beta=beta,
                                   trunc=float(last.trunc))
        vals.append(float(cd))
    return float(np.mean(vals)) if vals else float("nan")


def calibrate_self_supervised(frames: list, cfg) -> dict:
    """一键全自监督标定 (论文 III-D-2 + III-E)。

    coarse_search_init (若无 init_T 先验则网格粗搜) → optimize_pose (式(17)(19))
    → multiframe.score_self_supervised (式(21)) + select_and_average (式(22)(23))
    → 返回 {'T_cam_lidar','per_frame_T','scores','final_cd','log',...}。
    """
    ss = _ss(cfg)
    seed = int(getattr(ss, "seed", 0))
    torch.manual_seed(seed)
    device = _device()
    print(f"[calibrate] device={device}, seed={seed}, 帧数={len(frames)}")

    # 稳健性: 预筛除相机深度点数不足 500 的帧
    valid_idx, skipped = [], []
    for i, fr in enumerate(frames):
        n_q = camera_cloud(fr, cfg).shape[0]
        if n_q < MIN_CAMERA_POINTS:
            print(f"[calibrate] 警告: 帧 {i} 相机深度点数 {n_q} < {MIN_CAMERA_POINTS}, 跳过")
            skipped.append(i)
        else:
            valid_idx.append(i)
    if not valid_idx:
        raise RuntimeError(
            "全部帧无效: 每帧相机深度点云都少于 500 点。诊断建议: "
            "1) 确认深度话题 (depth_registration:=true) 有数据且与彩色对齐; "
            "2) 检查 depth_unit 配置 (mm/m 自动判别可能失败); "
            "3) 检查场景距离是否落在 depth_min~depth_max 量程内; "
            "4) 重新采集数据。")
    vframes = [frames[i] for i in valid_idx]

    log: list = []
    xi0 = coarse_search_init(vframes, cfg)
    xi, per_frame_T = optimize_pose(vframes, xi0, cfg, log=log)

    from . import multiframe  # 延迟导入避免循环依赖
    scores = multiframe.score_self_supervised(per_frame_T, vframes, cfg)
    mf = getattr(cfg, "multiframe", SimpleNamespace())
    T_star = multiframe.select_and_average(
        per_frame_T, scores,
        x=float(getattr(mf, "selection_ratio", 0.3)),
        weighting=str(getattr(mf, "weighting", "score")))
    final_cd = _mean_final_cd(vframes, T_star, cfg)

    r_star, t_star = np_rt_from_se3(T_star)
    print(f"[calibrate] 完成: final_cd={final_cd:.5f}, "
          f"{_xi_str(np.concatenate([r_star, t_star]))}")
    return {
        "T_cam_lidar": T_star,
        "per_frame_T": per_frame_T,
        "scores": scores,
        "final_cd": final_cd,
        "log": log,
        "frame_indices": valid_idx,
        "skipped_frames": skipped,
        "xi_global": xi,
    }
