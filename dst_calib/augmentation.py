"""双侧数据增广 (论文 III-A, 式(2)-(4)) — 评估模块训练数据的在线生成。

核心思想: 对同一帧数据同时扰动"相机视角"与"LiDAR 外参"两侧:
  式(2): 在 [±rot_range, ±trans_range] 内按轴权重采样扰动位姿 ΔT;
  式(3): T_cam = ΔT_cam · T_gt_cam_lidar,   T_lidar = ΔT_lidar · T_cam;
  式(4): 监督真值 T_gt_virtual = T_cam · T_lidar^{-1}  (代数上恒等于 ΔT_lidar^{-1})。

坐标约定推导 (与 projection.generate_ldp / generate_cdp 的语义严格一致):

设某物理点在原 LiDAR 系坐标为 p_l, 在原相机系坐标为 p_c = T_gt · p_l。

1. 虚拟相机系 c' (CDP 投影系): 定义点坐标 p_c' = T_cam · p_l, 即把 T_cam 直接
   解释为 "LiDAR 系 → 虚拟相机系" 的外参。对输入的相机深度点云 (坐标即 p_c) 有
       p_c' = T_cam · T_gt^{-1} · p_c = ΔT_cam · p_c,
   故 generate_cdp 的视角变换取 T_view_cam = ΔT_cam。等价的物理图像: 虚拟相机
   机体相对原相机被主动移动了 ΔT_cam^{-1} (相机中心被扰动), 固定场景点的坐标
   随之做机体运动的逆变换 (即左乘 ΔT_cam)。
   [推导说明: 若改用 "p_c' = ΔT_cam^{-1}·p_c" 的约定, 则 CDP 投影系的基变为
   ΔT_cam^{-1}·T_gt, 与式(4) 的 T_gt_virtual = T_cam·T_lidar^{-1} 联立后
   LDP→CDP 的真实对齐变换是 ΔT_cam^{-1}·ΔT_lidar·ΔT_cam (共轭项不消),
   破坏 "用 T_gt_virtual 变换 LDP 点云可与 CDP 点云精确对齐" 的自洽性;
   因此采用本约定。由于 ΔT_cam 关于 0 对称采样, 两种约定在分布意义上等价。]

2. LDP 投影系 l' : LiDAR 点直接用带误差的外参 T_lidar 投影,
       p_l' = T_lidar · p_l,
   对应 generate_ldp(points_lidar, T_lidar, K, size)。

3. 自洽性 (tests/test_geometry.py 验证): 对同一物理点,
       T_gt_virtual · p_l' = (T_cam · T_lidar^{-1}) · T_lidar · p_l
                           = T_cam · p_l = p_c',
   即投影前的 LDP 三维点经 T_gt_virtual 变换后与 CDP 三维点严格重合
   (同一物理点集时 Chamfer 距离恰为 0)。

4. 基准外参精度要求: 标签公式 T_gt_virtual = ΔT_lidar^{-1} 数值上与传入的
   基准外参无关, 但第 3 条自洽性隐含假设输入相机深度点满足 p_c = T_gt·p_l ——
   真实数据的 CDP 点坐标由真值外参 T_true 固定 (p_c = T_true·p_l), 传入
   基准 T_base ≠ T_true 时, LDP/CDP 三维点的真实对齐变换是
       ΔT_cam · (T_true·T_base^{-1}) · ΔT_cam^{-1} · ΔT_lidar^{-1}
   而非标签 ΔT_lidar^{-1}: 全部训练对被注入共轭化的系统偏差
   E = T_true·T_base^{-1}, 评估模块迭代精化的不动点随之偏向 T_base
   (数值验证: T_base=T_true 时残差 ~1e-16 m, 基准偏 3.3°/6cm 时同一物理
   点集经标签变换后失配达 ~0.2 m)。因此训练必须用尽量接近真值的基准外参
   (session 内 extrinsic.yaml 或 --init_yaml, 见 train.py 附录 A2);
   仅凭名义外参训练时, 真实安装偏离名义值的量会整体成为 eva 路径
   (fast/both 模式) 的系统偏差。
"""
from __future__ import annotations

import numpy as np

from .geometry import (
    np_rt_from_se3,
    np_se3_from_rt,
    np_se3_inverse,
    np_transform,
)

__all__ = ["sample_perturbation", "double_sided_sample"]


def _cfg_get(cfg, paths, default):
    """从嵌套 SimpleNamespace/dict 中按点号路径依次尝试取值, 全部失败返回默认值。

    paths 为路径元组, 例如 ("augmentation.rot_range_deg", "rot_range_deg"),
    既支持传入完整配置命名空间, 也支持直接传入 augmentation 子命名空间。
    """
    if cfg is None:
        return default
    for path in paths:
        node = cfg
        ok = True
        for name in path.split("."):
            if isinstance(node, dict):
                if name in node:
                    node = node[name]
                else:
                    ok = False
                    break
            elif hasattr(node, name):
                node = getattr(node, name)
            else:
                ok = False
                break
        if ok and node is not None:
            return node
    return default


def sample_perturbation(rot_range_deg: float = 5.0,
                        trans_range_m: float = 0.5,
                        axis_weights=(0.6, 0.2, 0.2),
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """式(2): 采样一个位姿扰动 ξ = [r, t] ∈ R^6。

    论文 IV 实现细节: 总范围 [±5°, ±0.5 m], 轴权重 [0.6, 0.2, 0.2], 即每轴幅值
    rot_i = rot_range_deg·w_i, trans_i = trans_range_m·w_i —— 默认为
    [±3°, ±1°, ±1°] 与 [±0.3 m, ±0.1 m, ±0.1 m], 各轴独立均匀采样。

    参数:
        rot_range_deg: 旋转总幅值 (度), 逐轴乘以权重后为该轴幅值。
        trans_range_m: 平移总幅值 (米)。
        axis_weights: 3 个轴的权重 (依次作用于 x/y/z 轴分量)。
        rng: numpy 随机数发生器 (np.random.Generator); None 时新建默认发生器。

    返回:
        (6,) float64, 前 3 维为 so(3) 旋转向量 (弧度), 后 3 维为平移 (米)。
    """
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(axis_weights, dtype=float).reshape(3)
    rot_amp = np.deg2rad(float(rot_range_deg)) * w    # 每轴旋转幅值 (弧度)
    trans_amp = float(trans_range_m) * w              # 每轴平移幅值 (米)
    r = rng.uniform(-rot_amp, rot_amp)
    t = rng.uniform(-trans_amp, trans_amp)
    return np.concatenate([r, t])


def double_sided_sample(points_lidar: np.ndarray,
                        cam_depth_points: np.ndarray,
                        T_gt_cam_lidar: np.ndarray,
                        K: np.ndarray,
                        size: tuple,
                        cfg=None,
                        rng: np.random.Generator | None = None) -> dict:
    """双侧增广采样一个训练样本 (论文 III-A, 式(2)-(4); 坐标约定见模块 docstring)。

    流程:
      1. 采样 ΔT_cam, ΔT_lidar (式2, sample_perturbation);
      2. T_cam = ΔT_cam·T_gt_cam_lidar, T_lidar = ΔT_lidar·T_cam (式3);
      3. CDP: 相机深度点云左乘 T_view_cam = T_cam·T_gt^{-1} = ΔT_cam 后投影
         (generate_cdp), 即在虚拟相机 c' 视角成像;
         LDP: LiDAR 点云用带误差外参 T_lidar 投影 (generate_ldp);
      4. 差分图 D = (LDP, [Δ]_+^{e_tar}, [Δ]_-^{e_tar}) (式11-12,
         build_difference_map);
      5. 监督真值 T_gt_virtual = T_cam·T_lidar^{-1} (式4, = ΔT_lidar^{-1}),
         满足: T_gt_virtual 把 LDP 投影系三维点精确变换到 CDP 投影系。

    参数:
        points_lidar: (N,3) 原 LiDAR 系点云。
        cam_depth_points: (M,3) 原相机系深度点云 (depth_cloud_from_gemini 输出)。
        T_gt_cam_lidar: (4,4) 基准外参 —— 标签公式与其无关, 但训练对的几何
            一致性要求它尽量接近真值 (偏差成为系统偏差, 见模块 docstring 第 4 条)。
        K: (3,3) 虚拟相机内参 (projection.virtual_camera_intrinsics)。
        size: (H,W) 虚拟相机投影尺寸。
        cfg: 配置 (完整命名空间或 augmentation 子命名空间), None 用默认超参。
        rng: np.random.Generator, None 时新建。

    返回 dict:
        'ldp', 'cdp':  (H,W) float32 深度投影图;
        'diff_map':    (3,H,W) float32 差分图 (式12);
        'xi_gt':       (6,) float32 监督位姿向量 [r, t];
        'T_gt':        (4,4) float64 = T_cam·T_lidar^{-1} (式4);
        额外调试键 (供测试/可视化, 不属于最小契约):
        'T_cam', 'T_lidar':  (4,4) 式(3) 两侧位姿;
        'points_ldp_3d':     (N,3) float32 LDP 投影前三维点 (LDP 投影系);
        'points_cdp_3d':     (M,3) float32 CDP 投影前三维点 (CDP 投影系),
                             满足 T_gt · points_ldp_3d 与 points_cdp_3d 对齐。
    """
    # 延迟导入: 使 sample_perturbation 不依赖 projection/difference_map 也可用
    from .difference_map import build_difference_map
    from .projection import generate_cdp, generate_ldp

    if rng is None:
        rng = np.random.default_rng()

    rot_range = float(_cfg_get(cfg, ("augmentation.rot_range_deg", "rot_range_deg"), 5.0))
    trans_range = float(_cfg_get(cfg, ("augmentation.trans_range_m", "trans_range_m"), 0.5))
    axis_w = _cfg_get(cfg, ("augmentation.axis_weights", "axis_weights"), (0.6, 0.2, 0.2))
    e_tar = float(_cfg_get(cfg, ("difference_map.e_tar", "e_tar"), 0.1))

    # 式(2): 两侧独立采样扰动
    xi_cam = sample_perturbation(rot_range, trans_range, axis_w, rng)
    xi_lidar = sample_perturbation(rot_range, trans_range, axis_w, rng)
    dT_cam = np_se3_from_rt(xi_cam[:3], xi_cam[3:])
    dT_lidar = np_se3_from_rt(xi_lidar[:3], xi_lidar[3:])

    # 式(3)
    T_gt_cam_lidar = np.asarray(T_gt_cam_lidar, dtype=float).reshape(4, 4)
    T_cam = dT_cam @ T_gt_cam_lidar
    T_lidar = dT_lidar @ T_cam

    pts_l = np.asarray(points_lidar, dtype=np.float32).reshape(-1, 3)
    pts_c = np.asarray(cam_depth_points, dtype=np.float32).reshape(-1, 3)

    # LDP: 带误差外参 T_lidar 投影; CDP: 视角变换 ΔT_cam (= T_cam·T_gt^{-1}) 投影
    ldp = generate_ldp(pts_l, T_lidar, K, size)
    cdp = generate_cdp(pts_c, dT_cam, K, size)

    # 式(11)(12): 差分图
    diff_map = build_difference_map(ldp, cdp, e_tar=e_tar)

    # 式(4): 监督真值 (数值上等于 ΔT_lidar^{-1})
    T_gt_virtual = T_cam @ np_se3_inverse(T_lidar)
    r_gt, t_gt = np_rt_from_se3(T_gt_virtual)
    xi_gt = np.concatenate([r_gt, t_gt]).astype(np.float32)

    return {
        "ldp": ldp,
        "cdp": cdp,
        "diff_map": diff_map,
        "xi_gt": xi_gt,
        "T_gt": T_gt_virtual,
        # ---- 以下为额外调试键 ----
        "T_cam": T_cam,
        "T_lidar": T_lidar,
        "points_ldp_3d": np_transform(T_lidar, pts_l).astype(np.float32),
        "points_cdp_3d": np_transform(dT_cam, pts_c).astype(np.float32),
    }
