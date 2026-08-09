"""离线标定 CLI (论文 III 全流程一键入口, 三种使用形态见附录 A3)。

用法:
    python -m dst_calib.calibrate --data_dir <采集目录> \
        [--config config/default.yaml] [--output <结果目录>] \
        [--init_yaml 先验外参.yaml] [--mode auto|selfsup|fast|both] \
        [--eva_ckpt runs/train_eva/best.pt]

模式 (论文 III-C 双路径):
- selfsup: load_frames → calibrate_self_supervised (粗搜索 + 自监督优化
  式(17)(19) + 多帧加权 式(21)-(23));
- fast: 跳过粗搜索/自监督, T_init(--init_yaml 或名义外参) →
  eva_infer.eva_calibrate_frames 逐帧精化 + 式(20) 打分, 秒级出结果;
- both (推荐): fast 结果作初值与 ξ_eva 先验 → optimize_pose (式(19) 全三项:
  L_t_ini + L_CD + L'_eva) → 逐帧微调 → SB 再精化 PE 候选 → 式(20)
  全监督打分 (异常回退式(21)) → 多帧加权，并以相同 Chamfer 目标做 fast
  非退化保护;
- auto (缺省): eva_ckpt 断点存在→both, 否则 selfsup。

输出: extrinsic.yaml (T_cam_lidar / T_lidar_cam / euler_zyx_deg / translation_m /
final_cd / per_frame_scores / mode / eva_used / eva_ckpt_sha8 / 配置摘要 / 时间戳)
→ 每帧渲染 LiDAR 投影叠加图 overlay_%03d.png → loss_curve.png。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import yaml

from .config import DEFAULT_CONFIG, load_config
from .geometry import euler_zyx_from_R, np_rt_from_se3, np_se3_inverse, np_transform
from .self_supervised import (
    MIN_CAMERA_POINTS,
    _device,
    _mean_final_cd,
    calibrate_self_supervised,
    camera_cloud,
    load_frames,
    optimize_pose,
)

# DST 仓库根目录: --eva_ckpt / inference.eva_ckpt 的相对路径以此为基准 (附录 A3)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ 输入输出

def _load_init_yaml(path: str) -> np.ndarray:
    """读取先验外参 yaml: 顶层 4x4 列表, 或含 'T_cam_lidar' 键的字典。"""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    arr = d.get("T_cam_lidar", d) if isinstance(d, dict) else d
    T = np.asarray(arr, dtype=np.float64).reshape(4, 4)
    # 合法性检查: 旋转块应近似正交且 det=+1 (反射矩阵同样满足 R·Rᵀ=I,
    # 常见错误来源: 手写外参时行/列写反或某轴符号写错), 底行应为 [0,0,0,1]
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-2):
        raise ValueError(f"先验外参 {path} 的旋转块不正交 — 请检查文件内容。")
    if float(np.linalg.det(R)) < 0.0:
        raise ValueError(
            f"先验外参 {path} 的旋转块 det<0 (是反射而非旋转) — "
            "常见原因: 行/列写反或某轴符号写错, 请检查文件内容。")
    if not np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"先验外参 {path} 的第 4 行应为 [0,0,0,1] — 请检查文件内容。")
    return T


def _sha256_8(path: str) -> str:
    """文件 sha256 前 8 位 (hashlib, 分块读避免大断点占内存) — 写入输出 yaml 溯源。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _resolve_inference(args, cfg) -> tuple[str, str, int]:
    """解析推理配置 (附录 A3): CLI 优先 → config.inference → 内置缺省。

    返回 (mode, eva_ckpt绝对路径, eva_iters)。mode 已完成 auto 语义展开:
    断点文件存在→both, 否则 selfsup (打印所选模式与原因)。
    """
    inf = getattr(cfg, "inference", SimpleNamespace())
    mode = str(args.mode or getattr(inf, "mode", "auto")).lower()
    ckpt = str(args.eva_ckpt or getattr(inf, "eva_ckpt", "runs/train_eva/best.pt"))
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(_REPO_ROOT, ckpt)  # 相对路径以 DST 根为基准
    eva_iters = int(getattr(inf, "eva_iters", 3))
    if mode not in ("auto", "selfsup", "fast", "both"):
        raise ValueError(f"未知 mode: {mode} (应为 auto/selfsup/fast/both)")
    if mode == "auto":  # auto 语义: 断点存在→both, 否则 selfsup
        if os.path.isfile(ckpt):
            mode = "both"
            print(f"[calibrate] mode=auto → both (评估模块断点存在: {ckpt})")
        else:
            mode = "selfsup"
            print(f"[calibrate] mode=auto → selfsup (评估模块断点不存在: {ckpt})")
    else:
        print(f"[calibrate] mode={mode}")
    if mode in ("fast", "both") and not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"mode={mode} 需要评估模块断点, 但 {ckpt} 不存在 — "
            "请先运行 scripts/train_eval_module.sh (或 python -m dst_calib.train) "
            "训练评估模块, 或改用 --mode selfsup。")
    return mode, ckpt, eva_iters


def _apply_ckpt_virtual_camera(cfg, meta: dict) -> None:
    """断点存有 virtual_camera 时覆盖 cfg 对应参数, 保证投影尺寸/焦距与训练一致。

    契约 A1「不一致时打印警告并以断点为准」的警告必须在本函数比较后打印:
    覆盖发生在任何投影之前, 此后 cfg 与断点恒相等, 下游
    eva_infer._resolve_virtual_camera 的比较不可能再发现差异。
    同时把覆盖值同步进 cfg._raw['virtual_camera'] —— 输出 yaml 的
    config.virtual_camera 段 (_config_summary 取自 _raw) 才能记录实际
    投影所用参数, 溯源不失真。"""
    vc = meta.get("virtual_camera") if isinstance(meta, dict) else None
    if not isinstance(vc, dict) or not hasattr(cfg, "virtual_camera"):
        return
    vcc = cfg.virtual_camera
    cfg_hwf = (int(getattr(vcc, "height", 256)), int(getattr(vcc, "width", 512)),
               float(getattr(vcc, "focal", 600.0)))
    ck_hwf = (int(vc.get("height", cfg_hwf[0]) or cfg_hwf[0]),
              int(vc.get("width", cfg_hwf[1]) or cfg_hwf[1]),
              float(vc.get("focal", cfg_hwf[2]) or cfg_hwf[2]))
    if ck_hwf != cfg_hwf:
        print(f"[calibrate] 警告: 断点虚拟相机 (H={ck_hwf[0]}, W={ck_hwf[1]}, "
              f"f={ck_hwf[2]:g}) 与配置 (H={cfg_hwf[0]}, W={cfg_hwf[1]}, "
              f"f={cfg_hwf[2]:g}) 不一致, 以断点为准")
    vcc.height, vcc.width, vcc.focal = ck_hwf
    raw = getattr(cfg, "_raw", None)
    if isinstance(raw, dict) and isinstance(raw.setdefault("virtual_camera", {}), dict):
        raw["virtual_camera"].update(
            {"height": ck_hwf[0], "width": ck_hwf[1], "focal": ck_hwf[2]})


def _filter_valid_frames(frames: list, cfg) -> tuple[list, list, list]:
    """稳健性预筛 (与 calibrate_self_supervised 同准则): 相机深度点数
    < MIN_CAMERA_POINTS 的帧跳过。返回 (有效帧列表, 有效索引, 跳过索引)。"""
    valid_idx, skipped = [], []
    for i, fr in enumerate(frames):
        n_q = camera_cloud(fr, cfg).shape[0]
        if n_q < MIN_CAMERA_POINTS:
            print(f"[calibrate] 警告: 帧 {i} 相机深度点数 {n_q} < "
                  f"{MIN_CAMERA_POINTS}, 跳过")
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
    return [frames[i] for i in valid_idx], valid_idx, skipped


def _calibrate_with_eva(frames: list, cfg, mode: str, ckpt_path: str,
                        eva_iters: int) -> dict:
    """fast / both 模式 (附录 A3): 评估模块 (全监督路径) 参与的标定。

    fast: T_init (--init_yaml 先验或 train.nominal_T_cam_lidar 名义外参) →
      eva_infer.eva_calibrate_frames 逐帧精化 + 式(20) 打分 + 式(22)(23) 融合,
      直接出结果 (秒级);
    both: fast 结果作 T_init 与 ξ_eva → optimize_pose(式(19) 全三项:
      L_t_ini + L_CD + L'_eva, L_t_ini 权重在配置未显式给出时按契约 A3
      以 1.0 启用) → 逐帧微调 (复用 optimize_pose 现有 per-frame finetune 路径)
      → 以 PE 的逐帧结果为先验再运行 SB 精化 → 式(20)
      score_full_supervised 打分。打分异常时回退式(21)
      score_self_supervised；若组合结果在相同的末段截断 Chamfer 指标上劣于
      fast 结果，则回退 fast，保证 both 不因 PE 局部极小值而退化。
    """
    # 延迟导入: eva_infer 为并行开发模块 (契约 A1), selfsup 路径不依赖它;
    # multiframe 与 self_supervised 存在既有的延迟导入约定, 保持一致
    from . import eva_infer, multiframe
    from .train import nominal_T_cam_lidar

    device = _device()
    model, meta = eva_infer.load_eva(ckpt_path, device)
    _apply_ckpt_virtual_camera(cfg, meta)
    vframes, valid_idx, skipped = _filter_valid_frames(frames, cfg)

    # T_init 来源: --init_yaml 先验 > 名义外参 (契约 A3)
    init_T = getattr(cfg, "init_T", None)
    if init_T is not None:
        T_init = np.asarray(init_T, dtype=np.float64).reshape(4, 4)
        print("[calibrate] fast T_init 来源: --init_yaml 先验外参")
    else:
        T_init = nominal_T_cam_lidar()
        print("[calibrate] fast T_init 来源: 名义外参 (train.nominal_T_cam_lidar)")

    print(f"[calibrate] fast: 评估模块逐帧迭代精化 (n_iters={eva_iters}) ...")
    fast_res = eva_infer.eva_calibrate_frames(vframes, T_init, model, cfg, device,
                                              n_iters=eva_iters)
    T_eva = np.asarray(fast_res["T_cam_lidar"], dtype=np.float64).reshape(4, 4)

    if mode == "fast":
        return {
            "T_cam_lidar": T_eva,
            "per_frame_T": [np.asarray(t, dtype=np.float64).reshape(4, 4)
                            for t in fast_res["per_frame_T"]],
            "scores": np.asarray(fast_res["scores"], dtype=np.float64),
            "final_cd": _mean_final_cd(vframes, T_eva, cfg),
            "log": [],
            "frame_indices": valid_idx,
            "skipped_frames": skipped,
            "score_method": "full_supervised",  # 式(20), eva_calibrate_frames 内部
        }

    # ---- both: fast 结果作初值与 ξ_eva 先验, 进入自监督优化 (式(19)) ----
    xi_eva = np.asarray(fast_res["xi_eva"], dtype=np.float64).reshape(6)
    r0, t0 = np_rt_from_se3(T_eva)
    xi_init = np.concatenate([r0, t0])
    # 契约 A3: both 模式式(19) 三项齐活 —— optimize_pose 的 L_t_ini 由
    # self_supervised.t_ini_weight 门控, selfsup 缺省关闭 (粗搜索初值平移
    # 不可靠); 此处 xi_init 为 fast 结果 T_eva 的 [r,t], 平移可作约束,
    # 配置未显式给出该键时以权重 1.0 启用 (显式配置含 0 时以配置为准)
    ss_ns = getattr(cfg, "self_supervised", None)
    if ss_ns is not None and not hasattr(ss_ns, "t_ini_weight"):
        ss_ns.t_ini_weight = 1.0
    print("[calibrate] both: 以 fast 结果为初值与 ξ_eva 先验, 自监督优化 (式19)")
    log: list = []
    xi_fin, pe_per_frame_T = optimize_pose(
        vframes, xi_init, cfg, xi_eva=xi_eva, log=log)

    # PE+SB*: PE 的逐帧候选只是 SB 的先验，必须把 SB 精化后的 T_i 用于最终
    # 融合。旧实现虽然计算了 T_ref，却仅把它用于式(20)打分，最终平均的仍是
    # pe_per_frame_T，等价于丢弃 SB 的修正，导致 both 可显著劣于 fast。
    #
    # both_eva_iters 与 fast 的 eva_iters 分开配置: tiny/欠训练断点可能为避免
    # 从较粗初值反复累积偏置而把 fast 设为 2 次，但 PE 已把候选送入更小邻域，
    # 最终 SB 至少 3 次精化可稳定收敛。旧配置没有该键时也采用此安全缺省。
    inf = getattr(cfg, "inference", SimpleNamespace())
    both_eva_iters = int(getattr(inf, "both_eva_iters", max(eva_iters, 3)))
    print(f"[calibrate] both: 以 PE 候选为先验执行 SB 最终精化 "
          f"(n_iters={both_eva_iters})")

    # 式(20) 打分: 对精化候选 T_i 再前向一次得到 T'_i，计算二者一致性。
    a = float(getattr(getattr(cfg, "loss", SimpleNamespace()), "a", 0.1))
    try:
        per_frame_T = []
        for fr, T_pe in zip(vframes, pe_per_frame_T):
            T_ref, _ = eva_infer.eva_refine(
                fr, np.asarray(T_pe, dtype=np.float64).reshape(4, 4),
                model, cfg, device, n_iters=both_eva_iters)
            per_frame_T.append(
                np.asarray(T_ref, dtype=np.float64).reshape(4, 4))

        xi_eva_list = []
        for fr, T_i in zip(vframes, per_frame_T):
            T_prime, _ = eva_infer.eva_refine(
                fr, np.asarray(T_i, dtype=np.float64).reshape(4, 4),
                model, cfg, device, n_iters=1)
            r_i, t_i = np_rt_from_se3(
                np.asarray(T_prime, dtype=np.float64).reshape(4, 4))
            xi_eva_list.append(np.concatenate([r_i, t_i]))
        scores = multiframe.score_full_supervised(per_frame_T, xi_eva_list, a=a)
        score_method = "full_supervised_after_pe_sb"
    except Exception as exc:  # 打分异常回退式(21), 输出 yaml 的 score_method 注明
        print(f"[calibrate] 警告: 式(20) 全监督打分失败 ({exc!r}), "
              "回退式(21) 自监督打分")
        per_frame_T = pe_per_frame_T
        scores = multiframe.score_self_supervised(per_frame_T, vframes, cfg)
        score_method = f"self_supervised_fallback ({exc})"

    mf = getattr(cfg, "multiframe", SimpleNamespace())
    T_star = multiframe.select_and_average(
        per_frame_T, scores,
        x=float(getattr(mf, "selection_ratio", 0.3)),
        weighting=str(getattr(mf, "weighting", "score")))
    final_cd = _mean_final_cd(vframes, T_star, cfg)

    # 非退化保护: both 从 fast 出发，最终至少不应在其自监督主目标上更差。
    # 两个值由 _mean_final_cd 使用同一批帧、同一体素/截断配置和确定性采样计算，
    # 因而可以直接比较。该保护只在数值非有限或严格退化时触发。
    fast_cd = _mean_final_cd(vframes, T_eva, cfg)
    if not np.isfinite(final_cd) or (
            np.isfinite(fast_cd) and final_cd > fast_cd):
        print(f"[calibrate] both 非退化保护: 组合结果 CD={final_cd:.5f} "
              f"劣于 fast CD={fast_cd:.5f}, 回退 fast 结果")
        T_star = T_eva
        per_frame_T = [
            np.asarray(t, dtype=np.float64).reshape(4, 4)
            for t in fast_res["per_frame_T"]
        ]
        scores = np.asarray(fast_res["scores"], dtype=np.float64)
        final_cd = fast_cd
        score_method = f"{score_method}_fast_guard"

    return {
        "T_cam_lidar": T_star,
        "per_frame_T": per_frame_T,
        "scores": scores,
        "final_cd": final_cd,
        "log": log,
        "frame_indices": valid_idx,
        "skipped_frames": skipped,
        "xi_global": xi_fin,
        "score_method": score_method,
    }


def _config_summary(cfg) -> dict:
    """从 cfg._raw 摘取与标定相关的配置段落写入结果, 便于溯源。"""
    raw = getattr(cfg, "_raw", None)
    if not isinstance(raw, dict):
        return {}
    keys = ("self_supervised", "multiframe", "loss", "inference", "online",
            "sensors", "virtual_camera", "difference_map")
    return {k: raw[k] for k in keys if k in raw}


def _write_extrinsic_yaml(path: str, result: dict, cfg, mode: str,
                          eva_used: bool, eva_ckpt_sha8: str | None) -> None:
    """写 extrinsic.yaml (全部转为原生 Python 类型; 附录 A3 新增
    mode / eva_used / eva_ckpt_sha8, 以及 both 回退时的 score_method 注明)。"""
    T = np.asarray(result["T_cam_lidar"], dtype=np.float64)
    out = {
        "T_cam_lidar": [[float(v) for v in row] for row in T],
        "T_lidar_cam": [[float(v) for v in row] for row in np_se3_inverse(T)],
        "euler_zyx_deg": [float(v) for v in np.degrees(euler_zyx_from_R(T[:3, :3]))],
        "translation_m": [float(v) for v in T[:3, 3]],
        "final_cd": float(result["final_cd"]),
        "per_frame_scores": [float(s) for s in np.asarray(result["scores"]).reshape(-1)],
        "frame_indices": [int(i) for i in result.get("frame_indices", [])],
        "skipped_frames": [int(i) for i in result.get("skipped_frames", [])],
        "mode": str(mode),                       # 实际执行的模式 (auto 已展开)
        "eva_used": bool(eva_used),              # 评估模块是否参与
        "eva_ckpt_sha8": eva_ckpt_sha8,          # 断点 sha256 前 8 位 (未用为 null)
        "config": _config_summary(cfg),
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
    }
    if "score_method" in result:  # 多帧打分方式 (both 回退式(21) 时在此注明)
        out["score_method"] = str(result["score_method"])
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None)
    print(f"[calibrate] 外参已写入 {path}")


# ------------------------------------------------------------------ 可视化

def _render_overlays(frames: list, T_cam_lidar: np.ndarray, out_dir: str, cfg) -> None:
    """每帧渲染 overlay_%03d.png: LiDAR 点经 T* 投影到 RGB, 深度伪彩 (JET)。

    无 RGB 的帧跳过; cv2 不可用时整体跳过并告警。
    """
    try:
        import cv2
    except ImportError:
        print("[calibrate] 警告: 未安装 cv2, 跳过 overlay 渲染")
        return
    cam = getattr(getattr(cfg, "sensors", SimpleNamespace()), "camera", SimpleNamespace())
    depth_max = float(getattr(cam, "depth_max", 6.0))
    T = np.asarray(T_cam_lidar, dtype=np.float64)
    n_drawn = 0
    for i, fr in enumerate(frames):
        if fr.rgb is None:
            continue
        img = cv2.cvtColor(np.ascontiguousarray(fr.rgb), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        pc = np_transform(T, fr.points_lidar.astype(np.float64))
        z = pc[:, 2]
        mask = (z > 0.05) & (z < depth_max + 0.5)
        pc = pc[mask]
        z = z[mask]
        if pc.shape[0] == 0:
            print(f"[calibrate] 警告: 帧 {i} 无 LiDAR 点落入相机前方, overlay 为原图")
        else:
            fx, fy = float(fr.K[0, 0]), float(fr.K[1, 1])
            cx, cy = float(fr.K[0, 2]), float(fr.K[1, 2])
            u = np.round(fx * pc[:, 0] / z + cx).astype(np.int64)
            v = np.round(fy * pc[:, 1] / z + cy).astype(np.int64)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z = u[inb], v[inb], z[inb]
            # 点太多时抽稀, 控制绘制耗时
            if u.shape[0] > 60000:
                sel = np.random.default_rng(0).choice(u.shape[0], 60000, replace=False)
                u, v, z = u[sel], v[sel], z[sel]
            # 深度 → JET 伪彩 (近红远蓝: 反相归一化)
            norm = np.clip(z / depth_max, 0.0, 1.0)
            u8 = (255.0 * (1.0 - norm)).astype(np.uint8).reshape(-1, 1)
            colors = cv2.applyColorMap(u8, cv2.COLORMAP_JET).reshape(-1, 3)
            for uu, vv, c in zip(u, v, colors):
                cv2.circle(img, (int(uu), int(vv)), 1,
                           (int(c[0]), int(c[1]), int(c[2])), -1)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{i:03d}.png"), img)
        n_drawn += 1
    print(f"[calibrate] 渲染 {n_drawn} 张 overlay 到 {out_dir}")


def _plot_loss_curve(log: list, path: str) -> None:
    """matplotlib 绘制优化损失曲线 (阶段边界用竖线标出)。"""
    if not log:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[calibrate] 警告: 未安装 matplotlib, 跳过 loss 曲线")
        return
    losses = [rec["loss"] for rec in log]
    stages = [rec.get("stage", 0) for rec in log]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, lw=1.0)
    for j in range(1, len(stages)):
        if stages[j] != stages[j - 1]:
            ax.axvline(j, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("L_pe")
    ax.set_yscale("log")
    ax.set_title("DST-Calib self-supervised optimization")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[calibrate] loss 曲线已保存 {path}")


# ------------------------------------------------------------------ 主入口

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="DST-Calib 离线标定: 无标定板 LiDAR-相机外参自监督标定")
    parser.add_argument("--data_dir", required=True, help="capture_data.py 采集目录")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件")
    parser.add_argument("--output", default=None,
                        help="结果目录 (缺省 runs/calib_<时间戳>)")
    parser.add_argument("--init_yaml", default=None, help="先验外参 yaml (可选)")
    parser.add_argument("--mode", default=None,
                        choices=["auto", "selfsup", "fast", "both"],
                        help="标定模式 (附录 A3, 缺省取配置 inference.mode=auto): "
                             "auto=断点存在→both 否则 selfsup; selfsup=纯自监督; "
                             "fast=仅评估模块, 秒级; both=双路径全激活 (推荐)")
    parser.add_argument("--eva_ckpt", default=None,
                        help="评估模块断点路径 (缺省取配置 inference.eva_ckpt, "
                             "默认 runs/train_eva/best.pt, 相对路径以 DST 根解析)")
    parser.add_argument("--no_artifacts", action="store_true",
                        help="只写 extrinsic.yaml，跳过 loss 曲线与 overlay；"
                             "供在线滑窗工作进程使用")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.init_T = _load_init_yaml(args.init_yaml) if args.init_yaml else None
    mode, eva_ckpt, eva_iters = _resolve_inference(args, cfg)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = getattr(getattr(cfg, "output", SimpleNamespace()), "runs_dir", "runs")
    out_dir = args.output or os.path.join(runs_dir, f"calib_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    frames = load_frames(args.data_dir)
    if mode == "selfsup":
        result = calibrate_self_supervised(frames, cfg)
        eva_used, eva_ckpt_sha8 = False, None
    else:  # fast / both: 评估模块参与, 记录断点哈希便于溯源
        result = _calibrate_with_eva(frames, cfg, mode, eva_ckpt, eva_iters)
        eva_used, eva_ckpt_sha8 = True, _sha256_8(eva_ckpt)

    T = np.asarray(result["T_cam_lidar"], dtype=np.float64)
    _write_extrinsic_yaml(os.path.join(out_dir, "extrinsic.yaml"), result, cfg,
                          mode, eva_used, eva_ckpt_sha8)
    if not args.no_artifacts:
        _plot_loss_curve(result.get("log", []), os.path.join(out_dir, "loss_curve.png"))
        _render_overlays(frames, T, out_dir, cfg)

    e_deg = np.degrees(euler_zyx_from_R(T[:3, :3]))
    print("=" * 60)
    print(f"[calibrate] 标定完成 (mode={mode}, eva_used={eva_used})")
    print(f"  euler_zyx_deg = [{e_deg[0]:+8.3f}, {e_deg[1]:+8.3f}, {e_deg[2]:+8.3f}]")
    print(f"  translation_m = [{T[0, 3]:+7.4f}, {T[1, 3]:+7.4f}, {T[2, 3]:+7.4f}]")
    print(f"  final_cd      = {result['final_cd']:.5f}")
    print(f"  结果目录       = {os.path.abspath(out_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
