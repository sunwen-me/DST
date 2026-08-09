"""dar.py 单元测试 — 论文 III-B 深度锚点精化 (式5-10, 算法1)。

覆盖:
- 式(5)  锚点提取: 共同有效像素、按 d^C 升序;
- 式(8)-(10) 单调近线性选择: 无噪凸函数全选、噪声+离群下约束满足与重映射精度、
  d^C 去重取中位数、分位数抽样上限、全反单调退化为单点;
- 式(6)(7) 分段线性重映射: 内段线性、外段端点常值、无效像素保持 0;
- refine_depth 完整流程 (合成场景) 与锚点不足时的中位数尺度退化路径。
"""
import numpy as np
import pytest

from dst_calib.dar import (
    extract_anchors,
    piecewise_linear_remap,
    refine_depth,
    select_anchors_monotone,
)
from dst_calib.geometry import np_se3_from_rt, np_se3_inverse, np_transform


# ------------------------------------------------------------------ 工具

def _slopes(sel: np.ndarray) -> np.ndarray:
    """式(8) 割线斜率序列。"""
    return np.diff(sel[:, 1]) / np.diff(sel[:, 0])


def _assert_monotone_convex(sel: np.ndarray) -> None:
    """断言式(9) 约束: d^C 严格递增、d^L 非递减、割线斜率非递减。"""
    assert np.all(np.diff(sel[:, 0]) > 0), "d^C 必须严格递增"
    assert np.all(np.diff(sel[:, 1]) >= -1e-12), "d^L 必须非递减"
    s = _slopes(sel)
    if s.size >= 2:
        assert np.all(np.diff(s) >= -1e-9), "割线斜率必须非递减 (离散凸性)"


def _convex_g(d_c):
    """测试用已知单调凸函数 d^L = g(d^C)。"""
    return 1.0 + 5.0 * np.asarray(d_c) ** 2


def _make_synthetic_scene(depth_fn, H=120, W=160, f=200.0, stride=4, margin=8):
    """构造合成场景: 网格像素上归一化深度 d^C(u) + 与之对应的 LiDAR 点云。

    d^C 仅随列号 u 变化; 每个网格像素放一个深度 z=depth_fn(d^C) 的相机系点,
    再用 T_cam_lidar 的逆变换回 LiDAR 系。像素坐标加 0.25 偏移, 使投影
    无论四舍五入还是向下取整都落回原像素。
    返回 (depth_norm, points_lidar, T_cam_lidar, K)。
    """
    K = np.array([[f, 0.0, W / 2.0],
                  [0.0, f, H / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    depth_norm = np.zeros((H, W), dtype=np.float32)
    pts_cam = []
    for v in range(margin, H - margin, stride):
        for u in range(margin, W - margin, stride):
            d_c = 0.05 + 0.9 * u / (W - 1)
            z = float(depth_fn(d_c))
            depth_norm[v, u] = d_c
            x = (u + 0.25 - W / 2.0) * z / f
            y = (v + 0.25 - H / 2.0) * z / f
            pts_cam.append([x, y, z])
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    T = np_se3_from_rt(np.array([0.1, -0.05, 0.2]), np.array([0.1, -0.2, 0.3]))
    pts_lidar = np_transform(np_se3_inverse(T), pts_cam).astype(np.float32)
    return depth_norm, pts_lidar, T, K


# ------------------------------------------------------------------ 式(5)

def test_extract_anchors_basic():
    """共同有效像素处取 (d^C,d^L) 对, 单侧有效不取, 按 d^C 升序。"""
    ldp = np.zeros((4, 4), dtype=np.float32)
    cdp = np.zeros((4, 4), dtype=np.float32)
    ldp[0, 0], cdp[0, 0] = 2.0, 0.4
    ldp[1, 2], cdp[1, 2] = 1.0, 0.2
    ldp[3, 3], cdp[3, 3] = 5.0, 0.9
    ldp[2, 2] = 3.0   # cdp 无效 → 不取
    cdp[0, 1] = 0.5   # ldp 无效 → 不取
    a = extract_anchors(ldp, cdp)
    assert a.shape == (3, 2)
    np.testing.assert_allclose(
        a, [[0.2, 1.0], [0.4, 2.0], [0.9, 5.0]], atol=1e-6)


def test_extract_anchors_empty():
    a = extract_anchors(np.zeros((5, 5)), np.zeros((5, 5)))
    assert a.shape == (0, 2)


def test_extract_anchors_shape_mismatch():
    with pytest.raises(ValueError):
        extract_anchors(np.zeros((4, 4)), np.zeros((4, 5)))


# ------------------------------------------------------------- 式(8)-(10)

def test_select_exact_convex_keeps_all():
    """无噪严格凸增函数: 全部锚点满足式(9), 最长子序列 = 全集。"""
    d_c = np.linspace(0.05, 0.95, 50)
    d_l = _convex_g(d_c)
    anchors = np.stack([d_c, d_l], axis=1)
    # 打乱行序, 验证内部排序
    rng = np.random.default_rng(1)
    sel = select_anchors_monotone(anchors[rng.permutation(50)])
    assert sel.shape[0] == 50
    _assert_monotone_convex(sel)
    np.testing.assert_allclose(sel[:, 0], d_c)
    np.testing.assert_allclose(sel[:, 1], d_l)


def test_select_noise_and_outliers():
    """凸函数 + 小噪声 + 25 个大离群点: 选择结果满足式(9),
    且分段线性重映射逼近真值 g。"""
    rng = np.random.default_rng(0)
    n = 200
    d_c = np.sort(rng.uniform(0.05, 0.95, n))
    d_l = _convex_g(d_c) + rng.normal(0.0, 0.01, n)
    out_idx = rng.choice(n, 25, replace=False)
    d_l[out_idx] += rng.uniform(1.0, 4.0, 25) * rng.choice([-1.0, 1.0], 25)
    d_l = np.clip(d_l, 0.05, None)
    sel = select_anchors_monotone(np.stack([d_c, d_l], axis=1))

    _assert_monotone_convex(sel)
    assert sel.shape[0] >= 20, "应选出可观的一致内点子集"

    # 重映射误差: 与真值曲线比较
    q = np.linspace(0.1, 0.9, 33)
    img = q.reshape(1, -1).astype(np.float32)
    remap = piecewise_linear_remap(img, sel)[0].astype(np.float64)
    gt = _convex_g(q)
    assert np.mean(np.abs(remap - gt)) < 0.25
    assert np.max(np.abs(remap - gt)) < 0.8


def test_select_dedup_same_dc_takes_median():
    """d^C 相同的点只保留一个, d^L 取中位数。"""
    anchors = np.array([[0.5, 1.0], [0.5, 3.0], [0.5, 2.0], [0.7, 4.0]])
    sel = select_anchors_monotone(anchors)
    np.testing.assert_allclose(sel, [[0.5, 2.0], [0.7, 4.0]])


def test_select_all_decreasing_returns_single():
    """d^L 全程严格递减 (无任何合法点对): 退化为单点。"""
    d_c = np.linspace(0.1, 0.9, 30)
    d_l = 5.0 - 4.0 * d_c
    sel = select_anchors_monotone(np.stack([d_c, d_l], axis=1))
    assert sel.shape == (1, 2)


def test_select_max_anchors_subsample():
    """n 远超上限时先分位数抽样到 ≤ max_anchors, 凸数据抽样后仍全选。"""
    d_c = np.linspace(0.01, 0.99, 3000)
    d_l = 0.5 + 3.0 * d_c + 4.0 * d_c ** 3  # 凸增
    sel = select_anchors_monotone(np.stack([d_c, d_l], axis=1), max_anchors=400)
    assert sel.shape[0] == 400
    _assert_monotone_convex(sel)


def test_select_empty_and_single():
    assert select_anchors_monotone(np.zeros((0, 2))).shape == (0, 2)
    sel = select_anchors_monotone(np.array([[0.3, 2.0]]))
    np.testing.assert_allclose(sel, [[0.3, 2.0]])


# --------------------------------------------------------------- 式(6)(7)

def test_piecewise_linear_remap_exact():
    """内段线性插值 (式6), 范围外端点常值 (式7), 无效像素保持 0。"""
    anchors = np.array([[0.2, 1.0], [0.5, 2.0], [0.8, 4.0]])
    img = np.array([[0.0, 0.1, 0.35, 0.65, 0.9]], dtype=np.float32)
    out = piecewise_linear_remap(img, anchors)
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        out, [[0.0, 1.0, 1.5, 3.0, 4.0]], atol=1e-6)


def test_piecewise_linear_remap_single_anchor():
    """单锚点: 所有有效像素映射为常值 d_0^L。"""
    img = np.array([[0.0, 0.2, 0.9]], dtype=np.float32)
    out = piecewise_linear_remap(img, np.array([[0.5, 2.0]]))
    np.testing.assert_allclose(out, [[0.0, 2.0, 2.0]], atol=1e-6)


def test_piecewise_linear_remap_empty_anchors_raises():
    with pytest.raises(ValueError):
        piecewise_linear_remap(np.ones((2, 2)), np.zeros((0, 2)))


# ------------------------------------------------------------ refine_depth

def test_refine_depth_full_pipeline():
    """合成场景端到端: 归一化深度经 DAR 校正后应逼近度量真值 g(d^C)。"""
    depth_norm, pts_lidar, T, K = _make_synthetic_scene(_convex_g)
    out = refine_depth(depth_norm, pts_lidar, T, K)

    assert out.shape == depth_norm.shape
    assert out.dtype == np.float32
    # 无效像素保持 0
    assert np.all(out[depth_norm == 0] == 0)

    gt = _convex_g(depth_norm.astype(np.float64))
    mask = (out > 0) & (depth_norm > 0)
    n_valid = int((depth_norm > 0).sum())
    assert mask.sum() >= 0.9 * n_valid, "绝大多数有效像素应被校正"
    err = np.abs(out[mask].astype(np.float64) - gt[mask])
    assert err.max() < 2e-2, f"重映射最大误差过大: {err.max():.4f}"


def test_refine_depth_degenerate_median_scale():
    """全'离群'情形 (d^L 随 d^C 递减, 无合法凸子序列): 触发退化路径,
    发出 UserWarning 并按中位数尺度因子线性映射。"""
    anti = lambda d_c: 5.0 - 4.0 * d_c  # 反单调 → 选择结果为单点 < min_anchors
    depth_norm, pts_lidar, T, K = _make_synthetic_scene(anti)
    with pytest.warns(UserWarning, match="DAR"):
        out = refine_depth(depth_norm, pts_lidar, T, K)

    mask = depth_norm > 0
    d_c = depth_norm[mask].astype(np.float64)
    z_gt = anti(d_c)
    # 期望尺度 = median(d^L)/median(d^C) (锚点即全部网格像素)
    scale = np.median(z_gt) / np.median(d_c)
    np.testing.assert_allclose(
        out[mask].astype(np.float64), d_c * scale, rtol=1e-3)
    assert np.all(out[~mask] == 0)


def test_refine_depth_no_anchors_scale_one():
    """无任何锚点 (点云为空 → LDP 全 0): 警告并返回未校正深度 (scale=1)。"""
    depth_norm = np.zeros((32, 48), dtype=np.float32)
    depth_norm[10:20, 10:30] = 0.5
    K = np.array([[100.0, 0, 24.0], [0, 100.0, 16.0], [0, 0, 1.0]])
    with pytest.warns(UserWarning, match="DAR"):
        out = refine_depth(depth_norm, np.zeros((0, 3), np.float32),
                           np.eye(4), K)
    np.testing.assert_allclose(out, depth_norm)
