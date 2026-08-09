"""端到端合成场景测试 (INTERFACES.md tests/test_synthetic.py 契约)。

场景生成器已抽到 tests/synthetic_scene.py (契约 A5: 与 test_full_paper_form 共用),
本文件保留端到端流程与断言, 默认参数下数据与抽取前逐比特一致:

- 合成房间场景 (地面 + 四面墙 + 3 个不同尺寸箱体, 表面均匀采样), 模拟
  Mid-360 LiDAR (法向可见性剔除 + 0.01 m 噪声, x前 y左 z上) 与 Gemini 335
  深度 (真值外参变换 + z-buffer 投影, 640x480 fx=fy=450, + 0.005 m 噪声);
- 真值 T_gt: R0 组合 15° 轴角扰动 (10-20° 区间), |t| ≈ 0.27 m (0.1-0.3 m);
- 4 帧不同内容: 每帧移动/旋转箱体位置模拟不同场景。

跑完整 calibrate_self_supervised (粗搜索 coarse_search_init → 三阶段由粗到细
自监督优化 → 逐帧微调 → 多帧加权平均), 按式(24) 断言
geometry.euler_error_deg < 1.0 且平移误差 < 0.10 m。
迭代数相对 default.yaml 适当缩减以控制时长, 但代码路径完整 (无捷径)。
"""
from __future__ import annotations

import numpy as np
import pytest

from dst_calib.geometry import euler_error_deg, np_se3_inverse, np_transform
from dst_calib.self_supervised import calibrate_self_supervised, camera_cloud

from synthetic_scene import (
    CAM_HW,
    FRAME_BOX_POSES,
    make_frame,
    make_reduced_cfg,
    make_T_gt,
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def synthetic_data():
    """构造 4 帧合成数据 + 真值外参 + 精简配置 (模块级缓存)。"""
    rng = np.random.default_rng(2026)
    T_gt = make_T_gt()
    frames = [
        make_frame(poses, T_gt, stamp=float(i), rng=rng)
        for i, poses in enumerate(FRAME_BOX_POSES)
    ]
    return frames, T_gt, make_reduced_cfg()


# ---------------------------------------------------------------- 测试

def test_synthetic_scene_consistency(synthetic_data):
    """数据自检: 相机深度点云反变换回 LiDAR 系后应与 LiDAR 点云重合 (噪声量级)。"""
    from scipy.spatial import cKDTree

    frames, T_gt, cfg = synthetic_data
    fr = frames[0]
    assert fr.depth.shape == CAM_HW
    assert (fr.depth > 0).sum() > 50000, "深度图有效像素过少 — 稠密采样密度不足"

    Q = camera_cloud(fr, cfg)                       # 相机系点云
    assert Q.shape[0] > 10000
    Q_l = np_transform(np_se3_inverse(T_gt), Q.astype(np.float64))
    d, _ = cKDTree(fr.points_lidar.astype(np.float64)).query(Q_l[::7], k=1)
    med = float(np.median(d))
    print(f"[synthetic] 相机点云↔LiDAR点云最近邻中位距离 = {med:.4f} m")
    assert med < 0.05, "合成两侧几何不一致 — 场景生成有误"


def test_synthetic_end_to_end(synthetic_data):
    """完整流程 (粗搜索→三阶段→逐帧微调→多帧优化), 式(24): e_r<1°, e_t<0.10 m。"""
    frames, T_gt, cfg = synthetic_data
    res = calibrate_self_supervised(frames, cfg)

    T_est = np.asarray(res["T_cam_lidar"], dtype=np.float64)
    e_r = euler_error_deg(T_est[:3, :3], T_gt[:3, :3])
    e_t = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    print(f"[synthetic] SYNTH_ERR rot={e_r:.4f} deg, trans={e_t:.4f} m, "
          f"final_cd={res['final_cd']:.5f}")

    # 结构契约检查
    assert len(res["per_frame_T"]) == len(frames)
    assert len(res["scores"]) == len(frames)
    assert np.isfinite(res["final_cd"])
    assert res["T_cam_lidar"].shape == (4, 4)
    # 式(24) 精度指标
    assert e_r < 1.0, f"旋转误差 {e_r:.3f}° ≥ 1°"
    assert e_t < 0.10, f"平移误差 {e_t:.3f} m ≥ 0.10 m"
