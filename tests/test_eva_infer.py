"""eva_infer 单元测试 (契约 A5: 合成帧 + 随机初始化 SB 模型跑通形状/收敛判据/打分,
不要求精度; 断点 round-trip save→load_eva)。

复用 tests/synthetic_scene.py 的场景生成器 (低密度) 构造合成帧;
用确定性的零输出桩模型验证收敛提前停与式(20)打分的代数不动点
(ξ≡0 ⇒ T 不变、T'_i=T_i ⇒ s_i=exp(0)=1), 用随机初始化 EvaluationSB
验证真实模型路径的形状与数值有效性。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dst_calib.config import load_config
from dst_calib.eva_infer import eva_calibrate_frames, eva_refine, load_eva
from dst_calib.geometry import np_rt_from_se3, np_se3_from_rt
from dst_calib.models import EvaluationSB
from dst_calib.self_supervised import Frame

from synthetic_scene import FRAME_BOX_POSES, make_frame, make_T_gt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 工具与桩模型

class _ZeroXiModel(torch.nn.Module):
    """确定性桩模型: 恒输出 ξ=0 (等价 "已收敛"), 并计数前向次数。

    ‖ξ‖=0 < 收敛阈值 ⇒ eva_refine 首轮即提前停;
    式(20) 打分时 T'_i = se3(0)·T_i = T_i ⇒ s_i = exp(0) = 1。
    """

    def __init__(self):
        super().__init__()
        self.n_calls = 0

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        return torch.zeros(1, 6)


def _assert_valid_se3(T: np.ndarray):
    """T 为合法 SE(3): 旋转块正交且 det=+1, 底行 [0,0,0,1], 全元素有限。"""
    T = np.asarray(T, dtype=np.float64)
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) > 0.0
    assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])


def _perturbed(T: np.ndarray, rot_deg=(2.0, 1.0, -1.0), trans=(0.1, -0.05, 0.08)):
    """T 左乘一个小扰动 (模拟带误差的外参估计)。"""
    r = np.radians(np.asarray(rot_deg, dtype=np.float64))
    dT = np_se3_from_rt(r, np.asarray(trans, dtype=np.float64))
    return dT @ np.asarray(T, dtype=np.float64)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def eva_env():
    """低密度合成帧 ×2 + 真值外参 + 小虚拟相机配置 (模块级缓存, 控制时长)。"""
    rng = np.random.default_rng(7)
    T_gt = make_T_gt()
    frames = [make_frame(FRAME_BOX_POSES[i], T_gt, stamp=float(i), rng=rng,
                         dens_scale=0.25) for i in range(2)]
    cfg = load_config()
    cfg.sensors.camera.depth_max = 7.0        # 合成房间量程 (与 test_synthetic 一致)
    cfg.virtual_camera.height = 64            # 缩小虚拟相机, 加快投影与前向
    cfg.virtual_camera.width = 128
    cfg.virtual_camera.focal = 150.0
    return frames, T_gt, cfg


@pytest.fixture(scope="module")
def sb_model():
    """随机初始化 SB 模型 (契约 A5: 不要求精度, 只验证形状/流程)。"""
    torch.manual_seed(0)
    return EvaluationSB().to(DEVICE).eval()


# ---------------------------------------------------------------- 断点 round-trip

def test_load_eva_roundtrip(tmp_path):
    """save → load_eva: arch/virtual_camera 元数据齐全, 权重逐位一致 (同输入同输出)。"""
    torch.manual_seed(1)
    model = EvaluationSB().eval()
    vc = {"height": 64, "width": 128, "focal": 150.0}
    ckpt = tmp_path / "best.pt"
    torch.save({"arch": "sb", "model": model.state_dict(),
                "virtual_camera": vc, "epoch": 3, "loss": 0.5}, ckpt)

    loaded, meta = load_eva(str(ckpt), torch.device("cpu"))
    assert isinstance(loaded, EvaluationSB)
    assert not loaded.training                       # eval() 模式
    assert meta["arch"] == "sb"
    assert meta["virtual_camera"] == vc              # A2 断点键透传
    assert meta["epoch"] == 3

    x = torch.randn(1, 3, 64, 128, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = model(x)
        out = loaded(x)
    assert torch.allclose(ref, out), "round-trip 后前向输出不一致 — 权重加载有误"


def test_load_eva_missing_keys(tmp_path):
    """断点缺 model / arch 键时应显式报错 (KeyError), 未知 arch 报 ValueError。"""
    p1 = tmp_path / "no_model.pt"
    torch.save({"arch": "sb"}, p1)
    with pytest.raises(KeyError):
        load_eva(str(p1), torch.device("cpu"))

    p2 = tmp_path / "no_arch.pt"
    torch.save({"model": EvaluationSB().state_dict()}, p2)
    with pytest.raises(KeyError):
        load_eva(str(p2), torch.device("cpu"))

    p3 = tmp_path / "bad_arch.pt"
    torch.save({"arch": "xx", "model": {}}, p3)
    with pytest.raises(ValueError):
        load_eva(str(p3), torch.device("cpu"))


def test_load_eva_virtual_camera_mismatch_warns(tmp_path, eva_env, capsys):
    """断点 virtual_camera 与 cfg 不一致时打印警告并以断点为准 (契约 A1)。"""
    frames, T_gt, cfg = eva_env
    torch.manual_seed(3)
    model = EvaluationSB()
    ckpt = tmp_path / "vc.pt"
    # 与 cfg (64,128,150) 不同且全局唯一的参数组合 (警告按组合去重打印)
    torch.save({"arch": "sb", "model": model.state_dict(),
                "virtual_camera": {"height": 96, "width": 192, "focal": 288.0}},
               ckpt)
    loaded, _ = load_eva(str(ckpt), DEVICE)
    T, xi = eva_refine(frames[0], T_gt, loaded, cfg, DEVICE, n_iters=1)
    _assert_valid_se3(T)
    assert "不一致" in capsys.readouterr().out


# ---------------------------------------------------------------- 单帧精化

def test_eva_refine_shapes(eva_env, sb_model):
    """随机 SB 模型: 输出 (4,4) 合法 SE(3) 与 (6,) 有限 ξ (契约 A1 签名/形状)。"""
    frames, T_gt, cfg = eva_env
    T0 = _perturbed(T_gt)
    T, xi = eva_refine(frames[0], T0, sb_model, cfg, DEVICE, n_iters=2)
    _assert_valid_se3(T)
    assert xi.shape == (6,)
    assert xi.dtype == np.float64
    assert np.all(np.isfinite(xi))
    # 更新规则为左乘 ξ 增量: 随机模型输出有界 (旋转头 ×0.1), T 不应飞出合理范围
    assert np.linalg.norm(T[:3, 3] - T0[:3, 3]) < 5.0


def test_eva_refine_early_stop(eva_env):
    """收敛判据: ‖ξ‖ 低于阈值时提前停 — 零输出桩模型应只前向 1 次且 T 不变。"""
    frames, T_gt, cfg = eva_env
    stub = _ZeroXiModel()
    T0 = _perturbed(T_gt)
    T, xi = eva_refine(frames[0], T0, stub, cfg, DEVICE, n_iters=5)
    assert stub.n_calls == 1, "ξ=0 已满足收敛判据, 不应继续迭代"
    assert np.allclose(T, T0), "ξ=0 时 T_est 应保持不变 (se3(0)=I 左乘)"
    assert np.allclose(xi, 0.0)


# ---------------------------------------------------------------- 多帧标定与打分

def test_eva_calibrate_frames_structure(eva_env, sb_model):
    """随机 SB 模型: 返回 dict 结构齐全, 分数 ∈ (0,1], T*/ξ_eva 自洽 (契约 A1)。"""
    frames, T_gt, cfg = eva_env
    T0 = _perturbed(T_gt)
    res = eva_calibrate_frames(frames, T0, sb_model, cfg, DEVICE, n_iters=2)

    assert set(res.keys()) >= {"T_cam_lidar", "per_frame_T", "scores", "xi_eva"}
    _assert_valid_se3(res["T_cam_lidar"])
    assert len(res["per_frame_T"]) == len(frames)
    for T_i in res["per_frame_T"]:
        _assert_valid_se3(T_i)
    s = np.asarray(res["scores"], dtype=np.float64)
    assert s.shape == (len(frames),)
    assert np.all(np.isfinite(s)) and np.all(s > 0.0) and np.all(s <= 1.0)
    xi_eva = np.asarray(res["xi_eva"], dtype=np.float64)
    assert xi_eva.shape == (6,)
    # xi_eva 是 T* 的 [r,t] (供 both 模式作先验)
    assert np.allclose(np_se3_from_rt(xi_eva[:3], xi_eva[3:]),
                       res["T_cam_lidar"], atol=1e-8)


def test_eva_calibrate_frames_scoring_fixed_point(eva_env):
    """零输出桩的代数不动点: T_i=T_init, T'_i=T_i ⇒ 式(20) s_i=1, T*=T_init。"""
    frames, T_gt, cfg = eva_env
    stub = _ZeroXiModel()
    T0 = _perturbed(T_gt)
    res = eva_calibrate_frames(frames, T0, stub, cfg, DEVICE, n_iters=3)
    for T_i in res["per_frame_T"]:
        assert np.allclose(T_i, T0)
    assert np.allclose(res["scores"], 1.0), "T'_i=T_i 时式(20)应给满分 exp(0)=1"
    assert np.allclose(res["T_cam_lidar"], T0, atol=1e-8)
    r0, t0 = np_rt_from_se3(T0)
    assert np.allclose(res["xi_eva"], np.concatenate([r0, t0]), atol=1e-8)


def test_eva_calibrate_frames_invalid_frame(eva_env):
    """相机深度点数不足的帧: 候选取 T_init 且得分置 0 (稳健性契约)。"""
    frames, T_gt, cfg = eva_env
    bad = Frame(points_lidar=frames[0].points_lidar,
                depth=np.zeros_like(frames[0].depth),   # 全无效深度
                K=frames[0].K.copy(), rgb=None, stamp=9.0)
    stub = _ZeroXiModel()
    T0 = _perturbed(T_gt)
    res = eva_calibrate_frames([frames[0], bad], T0, stub, cfg, DEVICE, n_iters=2)
    s = np.asarray(res["scores"])
    assert s[0] == pytest.approx(1.0)
    assert s[1] == 0.0, "无效帧得分应置 0"
    assert np.allclose(res["per_frame_T"][1], T0), "无效帧候选应取 T_init"
    # 融合结果只由有效帧决定 (k=ceil(0.3·2)=1 取最高分)
    assert np.allclose(res["T_cam_lidar"], T0, atol=1e-8)
