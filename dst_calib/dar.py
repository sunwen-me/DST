"""深度锚点精化 DAR (Depth Anchor Refinement) — 论文 III-B, 式(5)-(10), 算法1。

单目网络输出的归一化相对深度 d^C ∈ [0,1] 缺乏度量尺度且与真实深度呈未知的
单调非线性关系。DAR 利用 LiDAR 深度投影 (LDP) 提供的稀疏度量深度 d^L 作为锚点，
拟合一条"单调近线性"(单调 + 离散凸) 的分段线性映射 f: d^C → d^L (式6-7)，
把整幅归一化深度图校正为度量深度图。

流程 (算法1):
  1) 式(5): 在 LDP 与 CDP(归一化) 共同有效像素处提取锚点集合 A={(d_i^C, d_i^L)};
  2) 式(8)-(10): 动态规划选出满足单调性与离散凸性约束的最长锚点子序列 S;
  3) 式(6)(7): 用 S 做分段线性重映射, 得到度量深度图。

Gemini335 有真实度量深度时推理不需要 DAR; 训练侧双侧增广与纯 RGB 退化模式需要。
"""
from __future__ import annotations

import warnings

import numpy as np

from .projection import generate_ldp

__all__ = [
    "extract_anchors",
    "select_anchors_monotone",
    "piecewise_linear_remap",
    "refine_depth",
]


# ------------------------------------------------------------------ 式(5) 锚点提取

def extract_anchors(ldp: np.ndarray, cdp_norm: np.ndarray) -> np.ndarray:
    """式(5): 提取锚点集合 A = {(d_i^C, d_i^L)}。

    在 LiDAR 深度投影 LDP 与归一化相机深度 CDP 同时有效 (>0) 的像素处取
    (d^C, d^L) 对; d^C ∈ [0,1] 为归一化相机深度, d^L > 0 为 LiDAR 度量深度。

    参数:
        ldp:      (H,W) float, LiDAR 深度投影, 0 表示无效像素。
        cdp_norm: (H,W) float, 归一化相机深度, 0 表示无效像素。
    返回:
        (K,2) float64, [:,0]=d^C, [:,1]=d^L, 按 d^C 升序排列。
    """
    ldp = np.asarray(ldp, dtype=np.float64)
    cdp = np.asarray(cdp_norm, dtype=np.float64)
    if ldp.shape != cdp.shape:
        raise ValueError(f"LDP 与 CDP 形状不一致: {ldp.shape} vs {cdp.shape}")
    # 共同有效: 两幅图皆 >0 且有限
    valid = (ldp > 0) & (cdp > 0) & np.isfinite(ldp) & np.isfinite(cdp)
    d_c = cdp[valid]
    d_l = ldp[valid]
    anchors = np.stack([d_c, d_l], axis=1)
    # 按 d^C 升序返回 (稳定排序保证确定性)
    order = np.argsort(anchors[:, 0], kind="stable")
    return anchors[order]


# ------------------------------------------------- 式(8)-(10) 单调近线性锚点选择

def _dedup_by_dc(anchors: np.ndarray) -> np.ndarray:
    """d^C 相同的锚点只保留一个, d^L 取该组中位数 (保证 d^C 严格递增, 式9)。

    实现: 按 (d^C, d^L) 字典序排序后各组组内已按 d^L 有序，
    组中位数 = 组内中间元素(偶数个取中间两元素均值), 全向量化。
    """
    order = np.lexsort((anchors[:, 1], anchors[:, 0]))
    a = anchors[order]
    d_c = a[:, 0]
    d_l = a[:, 1]
    uniq_c, starts = np.unique(d_c, return_index=True)
    counts = np.diff(np.append(starts, len(a)))
    # 组内中位数: 中间一或两个元素的均值 (组内 d_l 已升序)
    mid1 = starts + (counts - 1) // 2
    mid2 = starts + counts // 2
    med_l = 0.5 * (d_l[mid1] + d_l[mid2])
    return np.stack([uniq_c, med_l], axis=1)


def _quantile_subsample(anchors: np.ndarray, max_anchors: int) -> np.ndarray:
    """按 d^C 分位数抽样至 ≤ max_anchors 点 (anchors 已按 d^C 升序)。

    在排序序列上取等间隔秩位置, 等价于对 d^C 经验分布做等分位抽样。
    """
    n = anchors.shape[0]
    if n <= max_anchors:
        return anchors
    idx = np.unique(np.round(np.linspace(0, n - 1, max_anchors)).astype(np.int64))
    return anchors[idx]


def select_anchors_monotone(anchors: np.ndarray, max_anchors: int = 400) -> np.ndarray:
    """式(8)-(10): 单调近线性锚点选择 (动态规划求最长合法子序列)。

    约束 (式9): 子序列 S 须满足
      - d^C 严格递增 (去重后排序保证候选集内成立);
      - d^L 非递减;
      - 割线斜率 s_k = (d_{k+1}^L - d_k^L)/(d_{k+1}^C - d_k^C) 非递减
        (离散凸性, 式8-9)。
    目标 (式10): max |S| s.t. 式(9)。

    DP: 状态 dp[i][j] = 以 (i,j) 为最后两点的最长合法子序列长度 (i<j),
    转移枚举 h<i, 要求 slope(h,i) ≤ slope(i,j) 且 d^L 单调约束成立。
    朴素转移为 O(n^3); 此处按 d^C 排序后, 对每个中间点 i 将前驱按斜率
    排序取前缀最大值 + 二分查找阈值, 降为 O(n^2 log n)。

    参数:
        anchors:     (N,2) 锚点, [:,0]=d^C, [:,1]=d^L。
        max_anchors: DP 前先按 d^C 分位数抽样到 ≤ 此数
                     (config: dar.max_anchors, 默认 400)。
    返回:
        (K,2) float64 选中锚点, 按 d^C 升序; 无合法点对时返回单点,
        输入为空时返回 (0,2)。
    """
    a = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    # 过滤非法值: d^C>0, d^L>0, 有限
    keep = np.isfinite(a).all(axis=1) & (a[:, 0] > 0) & (a[:, 1] > 0)
    a = a[keep]
    if a.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # 去重 (d^C 相同取 d^L 中位数) → 按 d^C 严格递增排序
    a = _dedup_by_dc(a)
    # 分位数抽样, 控制 DP 规模
    a = _quantile_subsample(a, int(max_anchors))
    n = a.shape[0]
    if n == 1:
        return a

    d_c = a[:, 0]
    d_l = a[:, 1]

    # 割线斜率矩阵 S[p,q] = (d_q^L - d_p^L)/(d_q^C - d_p^C), 仅用上三角 (式8)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (d_l[None, :] - d_l[:, None]) / (d_c[None, :] - d_c[:, None])

    # 合法点对: j>i 且 d^L 非递减 (d^C 严格递增由排序+去重保证)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_pair = upper & (d_l[None, :] >= d_l[:, None])

    # dp[i,j]: 以 (i,j) 结尾的最长长度; 非法对为 -inf; parent[i,j] = 最优前驱 h
    dp = np.where(valid_pair, 2.0, -np.inf)
    parent = np.full((n, n), -1, dtype=np.int64)

    # 按中间点 i 升序转移: dp[h,i] (h<i) 在处理 i 前已全部定型
    for i in range(1, n - 1):
        hs = np.nonzero(valid_pair[:i, i])[0]          # 合法前驱 h
        js = i + 1 + np.nonzero(valid_pair[i, i + 1:])[0]  # 合法后继 j
        if hs.size == 0 or js.size == 0:
            continue
        # 前驱按 slope(h,i) 升序, 做前缀最大 dp 及其 argmax
        s_hi = slope[hs, i]
        v_hi = dp[hs, i]
        order = np.argsort(s_hi, kind="stable")
        s_sorted = s_hi[order]
        v_sorted = v_hi[order]
        h_sorted = hs[order]
        pref_val = np.maximum.accumulate(v_sorted)
        # 前缀 argmax (取最早达到当前最大值的位置), 全向量化
        prev = np.concatenate(([-np.inf], pref_val[:-1]))
        new_max = v_sorted > prev
        pref_arg = np.maximum.accumulate(
            np.where(new_max, np.arange(v_sorted.size), 0))
        # 对每个 j: 阈值 slope(i,j), 允许 slope(h,i) ≤ slope(i,j) (式9 含等号)
        thr = slope[i, js]
        pos = np.searchsorted(s_sorted, thr, side="right") - 1
        ok = pos >= 0
        if not np.any(ok):
            continue
        js_ok = js[ok]
        pos_ok = pos[ok]
        cand = pref_val[pos_ok] + 1.0
        better = cand > dp[i, js_ok]
        if np.any(better):
            js_upd = js_ok[better]
            dp[i, js_upd] = cand[better]
            parent[i, js_upd] = h_sorted[pref_arg[pos_ok[better]]]

    best = dp.max()
    if not np.isfinite(best):
        # 无任何合法点对 (如 d^L 全程递减): 退化为单点, 取中位序号保持稳定
        return a[[n // 2]]

    # 回溯最长子序列
    i, j = np.unravel_index(int(np.argmax(dp)), dp.shape)
    seq = [j, i]
    h = int(parent[i, j])
    while h >= 0:
        seq.append(h)
        i, j = h, i
        h = int(parent[i, j])
    seq.reverse()
    return a[np.asarray(seq, dtype=np.int64)]


# ------------------------------------------------------- 式(6)(7) 分段线性重映射

def piecewise_linear_remap(depth_norm: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """式(6)(7): 用选中锚点做分段线性映射 f: [0,1] → R+, 逐像素校正深度图。

    f(d^C) =
      d_0^L,                                                d^C ≤ d_0^C      (式7)
      d_{i-1}^L + (d_i^L - d_{i-1}^L)/(d_i^C - d_{i-1}^C)
                  · (d^C - d_{i-1}^C),      d_{i-1}^C < d^C ≤ d_i^C          (式6)
      d_{K-1}^L,                                            d^C > d_{K-1}^C  (式7)

    np.interp 内段线性插值、区间外自动钳位到端点常值, 与式(6)(7) 完全一致。
    无效像素 (=0) 保持 0。

    参数:
        depth_norm: (H,W) 归一化深度图, 0 为无效。
        anchors:    (K,2) 选中锚点 (K≥1), [:,0]=d^C, [:,1]=d^L。
    返回:
        (H,W) float32 度量深度图。
    """
    a = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] == 0:
        raise ValueError("piecewise_linear_remap: 锚点为空, 无法重映射")
    # 保证按 d^C 升序 (np.interp 要求 xp 非降)
    a = a[np.argsort(a[:, 0], kind="stable")]

    d = np.asarray(depth_norm, dtype=np.float64)
    out = np.zeros(d.shape, dtype=np.float32)
    valid = (d > 0) & np.isfinite(d)
    if np.any(valid):
        out[valid] = np.interp(d[valid], a[:, 0], a[:, 1]).astype(np.float32)
    return out


# ------------------------------------------------------------- 完整 DAR 流程

def refine_depth(mono_depth_norm: np.ndarray, points_lidar: np.ndarray,
                 T_cam_lidar: np.ndarray, K: np.ndarray,
                 min_anchors: int = 8, max_anchors: int = 400) -> np.ndarray:
    """完整 DAR 流程 (论文 III-B, 算法1): 归一化单目深度 → 度量深度。

    步骤: LDP 生成 (projection.generate_ldp) → 锚点提取 (式5) →
    单调近线性选择 (式8-10) → 分段线性重映射 (式6-7)。

    退化路径: 合法锚点数 < min_anchors 时, 退化为中位数尺度因子的线性映射
    scale = median(d^L)/median(d^C) (基于全部原始锚点), 并发出 UserWarning;
    完全无锚点时 scale=1 (返回未校正深度) 并警告。

    参数:
        mono_depth_norm: (H,W) 归一化单目深度, d^C ∈ [0,1], 0 为无效。
        points_lidar:    (N,3) LiDAR 系点云。
        T_cam_lidar:     (4,4) 外参, p_cam = T_cam_lidar @ p_lidar。
        K:               (3,3) 相机内参。
        min_anchors:     退化阈值 (config: dar.min_anchors, 默认 8)。
        max_anchors:     DP 抽样上限 (config: dar.max_anchors, 默认 400)。
    返回:
        (H,W) float32 度量深度图, 无效像素保持 0。
    """
    depth = np.asarray(mono_depth_norm, dtype=np.float32)
    size = depth.shape

    # 1) LDP 生成
    ldp = generate_ldp(points_lidar, T_cam_lidar, K, size)

    # 2) 式(5) 锚点提取
    anchors = extract_anchors(ldp, depth)

    # 3) 式(8)-(10) 单调近线性选择
    selected = select_anchors_monotone(anchors, max_anchors=max_anchors)

    if selected.shape[0] < int(min_anchors):
        # 退化路径: 中位数尺度因子线性映射
        if anchors.shape[0] == 0:
            warnings.warn(
                "DAR: LDP 与 CDP 无共同有效像素, 无法估计尺度, 返回未校正深度 (scale=1)",
                UserWarning, stacklevel=2)
            scale = 1.0
        else:
            med_c = float(np.median(anchors[:, 0]))
            med_l = float(np.median(anchors[:, 1]))
            scale = med_l / max(med_c, 1e-12)
            warnings.warn(
                f"DAR: 合法锚点数 {selected.shape[0]} < min_anchors={min_anchors}, "
                f"退化为中位数尺度因子线性映射 (scale={scale:.4f})",
                UserWarning, stacklevel=2)
        out = np.zeros_like(depth, dtype=np.float32)
        m = depth > 0
        out[m] = depth[m] * np.float32(scale)
        return out

    # 4) 式(6)(7) 分段线性重映射
    return piecewise_linear_remap(depth, selected)
