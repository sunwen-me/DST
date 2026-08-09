#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DST-Calib 标定结果可视化 (conda python: numpy + cv2 + yaml)。

对采集目录中的每帧:
  1. 用外参 T_cam_lidar (p_cam = T @ p_lidar) 把 LiDAR 点投影到彩色图上,
     按深度伪彩着色 —— 即论文 LDP (LiDAR Depth Projection) 的 RGB 叠加视图;
  2. 对相机深度图取 Canny 边缘, 与投影 LiDAR 点叠加成第二块面板 ——
     外参正确时 LiDAR 深度不连续处应贴合深度边缘;
  3. 统计投影 LiDAR 深度与 Gemini 深度的一致性 |LDP - depth|
     (对应论文式(11) Δ 的绝对值在有效像素上的分布), 输出逐帧与总体中位数。

用法:
  /home/sw/Software/anaconda3/envs/dstcalib/bin/python visualize_result.py \
      --data_dir runs/<ts>/data --extrinsic_yaml runs/<ts>/extrinsic.yaml \
      --out runs/<ts>/vis
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml

MAX_DRAW_POINTS = 60000   # 叠加图最多绘制的点数 (超出则随机抽样)


def load_extrinsic(path: str) -> np.ndarray:
    """extrinsic.yaml → T_cam_lidar (4,4)。字段契约见 INTERFACES.md calibrate 节。"""
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    if "T_cam_lidar" not in y:
        raise KeyError(f"{path} 中缺少 T_cam_lidar 字段")
    T = np.asarray(y["T_cam_lidar"], dtype=np.float64).reshape(4, 4)
    return T


def project_lidar(points: np.ndarray, T: np.ndarray, K: np.ndarray,
                  hw: tuple[int, int], min_depth: float = 0.05,
                  max_depth: float = 60.0):
    """LiDAR 点 → 像素坐标与相机系深度 (针孔投影, 同 projection.generate_ldp 语义)。

    返回 (uv (M,2) int, z (M,) float): 仅保留 z∈(min_depth,max_depth) 且落在
    图像内的点。"""
    H, W = hw
    pc = points @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    m = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    pc = pc[m]
    z = z[m]
    u = np.round(K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u[inb], v[inb]], axis=1), z[inb]


def depth_colors_bgr(z: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """深度 → JET 伪彩 (BGR, 供 cv2 绘制): 近=红/暖, 远=蓝/冷。"""
    if z.size == 0:  # cv2 5.x 对空输入返回 None, 需前置拦截
        return np.zeros((0, 3), np.uint8)
    t = np.clip((z - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    u8 = ((1.0 - t) * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET).reshape(-1, 3)


def splat_points(base_bgr: np.ndarray, uv: np.ndarray, colors_bgr: np.ndarray,
                 radius: int = 1, alpha: float = 0.85) -> np.ndarray:
    """把彩色点画到图上: 先写 1 像素层再膨胀成 (2r+1) 方点, 与底图 alpha 混合。"""
    if uv.shape[0] == 0:
        return base_bgr.copy()
    layer = np.zeros_like(base_bgr)
    layer[uv[:, 1], uv[:, 0]] = colors_bgr
    if radius > 0:
        k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        layer = cv2.dilate(layer, k)
    mask = layer.any(axis=2)
    out = base_bgr.copy()
    out[mask] = ((1.0 - alpha) * out[mask] + alpha * layer[mask]).astype(np.uint8)
    return out


def depth_edges(depth: np.ndarray) -> np.ndarray:
    """深度图 → Canny 边缘 (uint8 0/255)。仅在有效像素范围内归一化。"""
    valid = depth > 0
    if valid.sum() < 100:
        return np.zeros(depth.shape, np.uint8)
    lo = float(np.percentile(depth[valid], 2))
    hi = float(np.percentile(depth[valid], 98))
    dn = np.zeros(depth.shape, np.float32)
    dn[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    u8 = (dn * 255).astype(np.uint8)
    u8 = cv2.medianBlur(u8, 5)     # 压制散粒噪声, 避免碎边缘
    return cv2.Canny(u8, 40, 120)


def process_frame(npz_path: str, T: np.ndarray, out_dir: str, idx: int):
    """单帧: 生成两联对比图 overlay_%03d.png, 返回 (误差数组, 命中率, 点数)。"""
    d = np.load(npz_path)
    pts = np.asarray(d["points_lidar"], dtype=np.float64)
    depth = np.asarray(d["depth"], dtype=np.float32)
    K = np.asarray(d["K"], dtype=np.float64).reshape(3, 3)
    H, W = depth.shape
    if "rgb" in d.files and d["rgb"].ndim == 3:
        rgb = np.asarray(d["rgb"], dtype=np.uint8)
    else:
        rgb = np.zeros((H, W, 3), np.uint8)

    uv, z = project_lidar(pts, T, K, (H, W))
    # 一致性统计: 投影点落到有效深度像素处的 |z_lidar - depth| (式(11) Δ 的 |·|)
    dv = depth[uv[:, 1], uv[:, 0]]
    hit = dv > 0
    err = np.abs(z[hit] - dv[hit])

    # 绘制抽样
    if uv.shape[0] > MAX_DRAW_POINTS:
        sel = np.random.default_rng(0).choice(uv.shape[0], MAX_DRAW_POINTS,
                                              replace=False)
        uv_draw, z_draw = uv[sel], z[sel]
    else:
        uv_draw, z_draw = uv, z
    valid_d = depth[depth > 0]
    d_min = float(np.percentile(valid_d, 2)) if valid_d.size else 0.25
    d_max = float(np.percentile(valid_d, 98)) if valid_d.size else 6.0
    colors = depth_colors_bgr(z_draw, d_min, d_max)

    # 面板 A: RGB + LiDAR 投影 (深度伪彩)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    panel_a = splat_points(bgr, uv_draw, colors, radius=1)

    # 面板 B: 深度 Canny 边缘(绿) + LiDAR 投影 —— 对齐好坏一目了然
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    panel_b = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.35).astype(np.uint8)
    edges = depth_edges(depth)
    panel_b[edges > 0] = (0, 255, 0)
    panel_b = splat_points(panel_b, uv_draw, colors, radius=0, alpha=1.0)

    canvas = np.concatenate([panel_a, panel_b], axis=1)
    med = float(np.median(err)) if err.size else float("nan")
    cv2.putText(canvas, f"frame {idx:03d}  pts={uv.shape[0]}  med|LDP-D|={med:.3f}m",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    out_path = os.path.join(out_dir, f"overlay_{idx:03d}.png")
    cv2.imwrite(out_path, canvas)

    hit_ratio = float(hit.mean()) if uv.shape[0] else 0.0
    return err, hit_ratio, uv.shape[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DST-Calib 标定结果可视化与一致性统计")
    ap.add_argument("--data_dir", required=True, help="capture_data.py 输出目录")
    ap.add_argument("--extrinsic_yaml", required=True,
                    help="calibrate 输出的 extrinsic.yaml")
    ap.add_argument("--out", default=None,
                    help="输出目录 (默认 <extrinsic 所在目录>/vis)")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.extrinsic_yaml)), "vis")
    os.makedirs(out_dir, exist_ok=True)

    T = load_extrinsic(args.extrinsic_yaml)
    files = sorted(glob.glob(os.path.join(args.data_dir, "frame_*.npz")))
    if not files:
        print(f"[错误] {args.data_dir} 中没有 frame_*.npz", file=sys.stderr)
        return 1

    all_err = []
    lines = []
    for i, f in enumerate(files):
        err, hit_ratio, n_proj = process_frame(f, T, out_dir, i)
        med = float(np.median(err)) if err.size else float("nan")
        all_err.append(err)
        line = (f"帧 {i:03d}: 视场内投影点={n_proj}, 落在有效深度像素={hit_ratio * 100:.1f}%, "
                f"|LDP-depth| 中位数={med:.4f} m")
        print(line)
        lines.append(line)

    cat = np.concatenate([e for e in all_err if e.size]) if all_err else np.array([])
    if cat.size:
        summary = (f"总体: 有效对比点={cat.size}, |LDP-depth| 中位数={np.median(cat):.4f} m, "
                   f"均值={cat.mean():.4f} m, <0.10m 占比={float((cat < 0.10).mean()) * 100:.1f}%")
    else:
        summary = "总体: 无有效对比点 (投影点均未落在有效深度像素上, 外参可能严重错误)"
    print(summary)
    lines.append(summary)
    with open(os.path.join(out_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"叠加图与统计已写入: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
