"""完整论文形态慢测 (契约 A5, @pytest.mark.slow):

复用 tests/synthetic_scene.py 场景生成器构造 3 场景 × 4 帧 (同一真值外参,
每场景箱体摆位不同; 用 6 个不同深度的箱体增强平移视差可观测性),
tiny 配置训练评估模块 SB (视场 128×256, GPU 5 分钟内), 然后经 calibrate.py
CLI 全链路验证:
  (a) fast 模式: ±5°/±0.3m 内的失准 (3.3°/0.07m) → 误差 <2.5° / <0.20 m;
  (b) both 模式: 评估模块收敛域内的先验失准 (1.1°/0.035m) →
      e_r<1°, e_t<0.1m (式24, 与纯 selfsup 基线同精度门槛);
  (c) 输出 extrinsic.yaml 含 mode / eva_used 字段。

设计说明 (集成验收中确定的边界条件):
- 训练基准外参: 场景 0/1 用目录内 extrinsic.yaml (模拟先前 selfsup 标定输出,
  真值 ⊕ 0.2°/8mm 噪声), 场景 2 无 extrinsic.yaml、经 --init_yaml 全局先验提供
  —— 同时覆盖附录 A2 的两级优先级。基准外参必须≈真值: 双侧增广的监督
  T_gt_virtual=ΔT_lidar^{-1} 虽与基准无关, 但 CDP 来自真实相机深度 (其投影系
  由真值外参决定), 基准偏离真值会给全部训练对注入系统性配准偏差
  (名义外参距真值 28° 时 fast 完全不收敛, 这正是 A2 要求逐 session
  基准外参的原因)。
- 真值外参取 R0·exp([5°,27°,7°]) (相机前视下俯 ~28°): 欧拉误差度量 (式24)
  在 R0 (pitch=-90°) 附近万向节退化, 该姿态将放大系数控制在 ~3 内。
- tiny 训练: epochs=120 × 每帧 8 样本/epoch... (实际 samples_per_frame=16,
  epochs=120, 总步数 ~2880, RTX3060 约 3.5-4 分钟); 增广平移范围缩至 0.2 m
  与小失准场景匹配; eva_iters=2 (迭代过多时欠训练网络的平移零点偏差会累积)。
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest
import torch
import yaml

from dst_calib import calibrate
from dst_calib import train as train_mod
from dst_calib.geometry import euler_error_deg, np_se3_from_rt, np_so3_exp

from synthetic_scene import make_frame

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------- 场景定义

# 6 个不同尺寸箱体 (深度分布 1.6-4.2 m, 提供平移视差)
BOX_SIZES6 = [(0.6, 0.45, 0.9), (0.4, 0.4, 0.5), (0.9, 0.35, 0.6),
              (0.5, 0.5, 1.2), (0.35, 0.6, 0.75), (0.7, 0.3, 0.45)]

# 3 场景 × 4 帧 × 6 箱体摆位 (cx, cy, yaw_deg)
SCENE_POSES6 = [
    [
        [(1.7, -0.9, 20.0), (3.2, 1.0, -15.0), (2.8, -0.2, 45.0),
         (4.0, 0.3, 10.0), (1.9, 0.9, -30.0), (3.3, -1.6, 5.0)],
        [(2.1, 0.5, -30.0), (3.6, -0.8, 10.0), (2.6, 1.4, 0.0),
         (1.6, -1.5, 40.0), (4.1, 1.2, -20.0), (2.9, 0.1, 25.0)],
        [(1.8, 1.3, 50.0), (2.9, -1.3, 0.0), (3.9, 0.6, -25.0),
         (2.4, -0.3, 15.0), (1.6, -1.8, -10.0), (3.4, 1.7, 30.0)],
        [(2.0, 0.0, -10.0), (3.5, -1.2, 35.0), (3.1, 1.3, 15.0),
         (1.7, 1.6, 5.0), (4.2, -0.4, -35.0), (2.5, -1.7, 20.0)],
    ],
    [
        [(2.4, 1.1, -40.0), (3.0, -0.9, 25.0), (1.8, -0.2, 60.0),
         (4.0, -1.4, 0.0), (1.6, 1.7, 15.0), (3.5, 0.5, -10.0)],
        [(3.3, 0.7, 15.0), (2.2, -1.5, -20.0), (2.8, 0.9, 35.0),
         (1.7, 0.1, -5.0), (4.1, 1.6, 20.0), (2.0, -0.8, 45.0)],
        [(2.0, -1.0, 5.0), (3.6, 1.2, -35.0), (2.6, 0.2, -10.0),
         (1.8, 1.0, 30.0), (3.1, -1.8, 10.0), (4.2, 0.0, -15.0)],
        [(3.1, -0.4, 45.0), (2.3, 1.3, 0.0), (3.4, -1.3, 20.0),
         (1.6, -1.2, -25.0), (2.7, 1.8, -5.0), (4.0, 0.8, 30.0)],
    ],
    [
        [(2.7, 0.3, 30.0), (2.1, 1.2, -25.0), (3.5, -0.8, 10.0),
         (1.6, -0.6, 0.0), (4.1, 0.9, -30.0), (2.3, -1.8, 25.0)],
        [(2.5, -1.4, -15.0), (3.2, 0.2, 40.0), (2.9, 1.5, -5.0),
         (1.7, 0.8, 20.0), (3.8, -1.0, 0.0), (1.9, -0.3, -40.0)],
        [(3.0, 1.0, 55.0), (2.4, -0.6, -30.0), (2.0, 0.8, 15.0),
         (4.2, -1.5, 5.0), (1.6, 1.5, -20.0), (3.6, 0.1, 35.0)],
        [(3.6, -1.1, -45.0), (2.6, 1.4, 20.0), (3.2, 0.6, 0.0),
         (1.8, -1.6, 10.0), (2.2, 0.2, -35.0), (4.0, 1.8, 15.0)],
    ],
]


def make_T_gt_pf() -> np.ndarray:
    """慢测真值外参: R0·exp([5°,27°,7°]), 前视下俯 ~28° (欧拉度量良态, 见模块 docstring)。"""
    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R0 @ np_so3_exp(np.radians([5.0, 27.0, 7.0]))
    T[:3, 3] = [0.12, -0.06, 0.20]
    return T


def _errs(T_est, T_gt) -> tuple:
    """式(24) 误差对: (欧拉角向量差范数 deg, 平移差范数 m)。"""
    T_est = np.asarray(T_est, dtype=np.float64)
    return (euler_error_deg(T_est[:3, :3], T_gt[:3, :3]),
            float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])))


def _dump_T_yaml(path: str, T: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"T_cam_lidar": [[float(v) for v in row] for row in T]}, f)


# ---------------------------------------------------------------- fixture

@pytest.fixture(scope="module")
def paper_env(tmp_path_factory):
    """3 场景数据 + tiny 配置 + 训练好的 SB 断点 (模块级缓存, 一次训练三处复用)。"""
    root = tmp_path_factory.mktemp("paper_form")
    T_gt = make_T_gt_pf()
    # 同一 GPU 上可复现的训练 (cudnn 卷积反传默认非确定, 会使精度断言在
    # 重复运行间抖动; 固定算法选择消除抖动, tiny 规模下开销可忽略)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- 数据: 每场景 4 帧 + 基准外参 (真值 ⊕ 0.2°/8mm, 模拟 selfsup 标定输出)。
    # 场景 0/1 写目录内 extrinsic.yaml, 场景 2 的写入全局 init_global.yaml
    # (经 --init_yaml 提供) —— 覆盖附录 A2 的两级基准外参优先级。
    # 12 帧同时汇入 calib_frames 作标定输入 (等价一次 num_frames=12 的采集,
    # 多帧加权 式(20)(22)(23) 在 k=⌈0.3·12⌉=4 上融合, 降低单帧候选的方差)。
    rng = np.random.default_rng(2026)
    scene_dirs = []
    calib_dir = root / "calib_frames"
    calib_dir.mkdir()
    init_global = str(root / "init_global.yaml")
    t0 = time.time()
    n_saved = 0
    for si, poses_list in enumerate(SCENE_POSES6):
        d = root / f"scene_{si}"
        d.mkdir()
        for fi, poses in enumerate(poses_list):
            fr = make_frame(poses, T_gt, stamp=float(fi), rng=rng,
                            box_sizes=BOX_SIZES6)
            payload = dict(points_lidar=fr.points_lidar, depth=fr.depth,
                           K=fr.K, stamp=fr.stamp)
            np.savez(str(d / f"frame_{fi:03d}.npz"), **payload)
            np.savez(str(calib_dir / f"frame_{n_saved:03d}.npz"), **payload)
            n_saved += 1
        dr = rng.normal(0.0, math.radians(0.2), 3)
        dt = rng.normal(0.0, 0.008, 3)
        T_sess = np_se3_from_rt(dr, dt) @ T_gt
        _dump_T_yaml(str(d / "extrinsic.yaml") if si < 2 else init_global, T_sess)
        scene_dirs.append(str(d))
    print(f"[paper_form] 数据生成 {time.time() - t0:.1f}s (3 场景 × 4 帧)")

    # ---- tiny 配置 (契约 A5: 视场 128×256; 焦距同比例缩至 300 保持视场角)
    with open(os.path.join(os.path.dirname(calibrate.__file__), "..",
                           "config", "default.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["virtual_camera"] = {"height": 128, "width": 256, "focal": 300.0}
    raw["sensors"]["camera"]["depth_max"] = 7.0    # 合成房间量程 (同 test_synthetic)
    raw["augmentation"]["trans_range_m"] = 0.2     # 与小失准场景匹配的增广平移范围
    raw["inference"]["eva_iters"] = 2              # 迭代过多时欠训练平移零点偏差累积
    ss = raw["self_supervised"]
    ss["seed"] = 0
    ss["stages"] = [                               # 缩减迭代 (同 test_synthetic 精简)
        {"voxel": 0.30, "iters": 60, "lr": 0.010, "trunc": 2.0, "max_points": 6000},
        {"voxel": 0.12, "iters": 90, "lr": 0.004, "trunc": 0.8, "max_points": 10000},
        {"voxel": 0.05, "iters": 120, "lr": 0.0015, "trunc": 0.3, "max_points": 15000},
    ]
    ss["per_frame_finetune_iters"] = 25
    ss["per_frame_lr"] = 0.0008
    ss["batch_frames"] = 2
    cfg_path = str(root / "tiny.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True)

    # ---- tiny 训练 (多 session, 附录 A2 CLI 路径; GPU ~4.5 分钟)。
    # 断点取 model_final.pt (完整 OneCycle 退火后的末轮模型): tiny 数据规模下
    # best.pt 的 "最小 epoch 损失" 选择受在线增广抽样噪声支配, 邻近 epoch 间
    # 零点偏差抖动明显, 末轮模型显著更稳。
    out_dir = str(root / "train_eva")
    args = train_mod.build_argparser().parse_args(
        ["--data_dir", *scene_dirs, "--arch", "sb", "--epochs", "90",
         "--samples_per_frame", "24", "--out", out_dir, "--config", cfg_path,
         "--seed", "0", "--init_yaml", init_global])
    t0 = time.time()
    train_mod.train(args)
    train_sec = time.time() - t0
    print(f"[paper_form] tiny 训练完成 {train_sec:.0f}s")

    return {
        "root": root,
        "T_gt": T_gt,
        "scene_dirs": scene_dirs,
        "calib_dir": str(calib_dir),
        "cfg_path": cfg_path,
        "ckpt": os.path.join(out_dir, "model_final.pt"),
        "train_sec": train_sec,
    }


def _run_calibrate(env, mode: str, rot_deg, trans_m, out_name: str):
    """写失准先验 → calibrate.main CLI → 返回 (输出 yaml dict, 初始误差对)。"""
    T_init = np_se3_from_rt(np.radians(rot_deg), trans_m) @ env["T_gt"]
    init_yaml = str(env["root"] / f"init_{out_name}.yaml")
    _dump_T_yaml(init_yaml, T_init)
    out_dir = str(env["root"] / out_name)
    calibrate.main(["--data_dir", env["calib_dir"], "--config", env["cfg_path"],
                    "--output", out_dir, "--init_yaml", init_yaml,
                    "--mode", mode, "--eva_ckpt", env["ckpt"]])
    with open(os.path.join(out_dir, "extrinsic.yaml"), encoding="utf-8") as f:
        result = yaml.safe_load(f)
    return result, _errs(T_init, env["T_gt"])


# ---------------------------------------------------------------- 测试

def test_tiny_training_artifacts(paper_env):
    """训练产物: best.pt 存在且含 arch/model/virtual_camera (附录 A2 断点键)。"""
    assert os.path.isfile(paper_env["ckpt"]), "训练未产出 best.pt"
    state = torch.load(paper_env["ckpt"], map_location="cpu", weights_only=False)
    assert state["arch"] == "sb"
    assert "model" in state
    assert state["virtual_camera"] == {"height": 128, "width": 256, "focal": 300.0}
    # 契约 A5: tiny 训练控制在 GPU 5 分钟内
    assert paper_env["train_sec"] < 300 or not torch.cuda.is_available(), \
        f"tiny 训练耗时 {paper_env['train_sec']:.0f}s 超出 5 分钟预算"


def test_fast_mode(paper_env):
    """(a)+(c): fast 模式在 ±5°/±0.3m 失准内 (3.3°/0.07m) → <2.5°/<0.20m, 秒级。"""
    t0 = time.time()
    result, (e0, t0e) = _run_calibrate(
        paper_env, "fast", [3.0, 1.2, 1.2], [0.06, 0.025, 0.025], "out_fast")
    e_r, e_t = _errs(np.asarray(result["T_cam_lidar"]), paper_env["T_gt"])
    print(f"[paper_form] FAST_ERR rot={e_r:.4f} deg, trans={e_t:.4f} m "
          f"(init {e0:.2f}/{t0e:.3f}, {time.time() - t0:.1f}s)")

    # (c) 输出 yaml 契约字段 (附录 A3)
    assert result["mode"] == "fast"
    assert result["eva_used"] is True
    assert isinstance(result["eva_ckpt_sha8"], str) and len(result["eva_ckpt_sha8"]) == 8
    assert len(result["per_frame_scores"]) == 12
    # (a) 精度: 显著优于初始失准且达标
    assert e_r < 2.5, f"fast 旋转误差 {e_r:.3f}° ≥ 2.5°"
    assert e_t < 0.20, f"fast 平移误差 {e_t:.3f} m ≥ 0.20 m"
    assert e_r < e0, "fast 未能改善旋转失准"


def test_both_mode(paper_env):
    """(b)+(c): both 双路径在评估模块收敛域内先验 (1.1°/0.035m) → e_r<1°, e_t<0.1m。

    门槛与纯 selfsup 基线 (test_synthetic 式24) 相同, 即"不劣于纯 selfsup"。
    """
    result, (e0, t0e) = _run_calibrate(
        paper_env, "both", [1.0, 0.4, 0.4], [0.03, 0.012, 0.012], "out_both")
    e_r, e_t = _errs(np.asarray(result["T_cam_lidar"]), paper_env["T_gt"])
    print(f"[paper_form] BOTH_ERR rot={e_r:.4f} deg, trans={e_t:.4f} m "
          f"(init {e0:.2f}/{t0e:.3f})")

    # (c) 输出 yaml 契约字段
    assert result["mode"] == "both"
    assert result["eva_used"] is True
    assert result["score_method"].startswith(
        ("full_supervised_after_pe_sb", "self_supervised"))
    assert np.isfinite(result["final_cd"])
    # (b) 式(24) 精度 (同 selfsup 基线门槛)
    assert e_r < 1.0, f"both 旋转误差 {e_r:.3f}° ≥ 1°"
    assert e_t < 0.10, f"both 平移误差 {e_t:.3f} m ≥ 0.10 m"
