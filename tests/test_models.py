"""模型单元测试 (论文 III-C, 图7): CBAM / 评估模块 / 位姿估计器。"""
import numpy as np
import pytest
import torch

from dst_calib.models import (CBAM, BlockPoseHead, EvaluationDB, EvaluationSB,
                              SimplePoseEstimator, StandardPoseEstimator)


def _n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------- CBAM

def test_cbam_preserves_shape():
    """CBAM 为逐元素门控, 输出形状必须与输入一致。"""
    x = torch.randn(2, 32, 16, 24)
    out = CBAM(32)(x)
    assert out.shape == x.shape


def test_cbam_output_bounded_by_input():
    """两级 sigmoid 门控 → |输出| 逐元素不超过 |输入|。"""
    x = torch.randn(1, 8, 10, 10)
    out = CBAM(8, reduction=4)(x)
    assert torch.all(out.abs() <= x.abs() + 1e-6)


def test_cbam_small_channels():
    """通道数小于 reduction 时不应崩溃 (hidden 至少为 1)。"""
    x = torch.randn(1, 4, 8, 8)
    out = CBAM(4, reduction=16)(x)
    assert out.shape == x.shape


# ------------------------------------------------------------ 评估模块

def test_block_pose_head_shape():
    head = BlockPoseHead(in_ch=256, n=5)
    feat = torch.randn(2, 256, 16, 32)
    xi = head(feat)
    assert xi.shape == (2, 6)


def test_evaluation_sb_forward():
    """单分支: 差分图 (B,3,256,512) → ξ (B,6)。"""
    model = EvaluationSB().eval()
    x = torch.randn(2, 3, 256, 512)
    with torch.no_grad():
        xi = model(x)
    assert xi.shape == (2, 6)
    assert torch.isfinite(xi).all()


def test_evaluation_db_forward():
    """双分支: CDP/LDP 各 (B,1,256,512) → ξ (B,6)。"""
    model = EvaluationDB().eval()
    cdp = torch.randn(2, 1, 256, 512)
    ldp = torch.randn(2, 1, 256, 512)
    with torch.no_grad():
        xi = model(cdp, ldp)
    assert xi.shape == (2, 6)
    assert torch.isfinite(xi).all()


def test_sb_fewer_params_than_db():
    """论文结论: 单分支差分图输入比双分支更轻。"""
    assert _n_params(EvaluationSB()) < _n_params(EvaluationDB())


def test_evaluation_sb_rotation_head_scaled():
    """旋转头输出乘 0.1 → 初始旋转预测应明显小 (接近单位阵)。"""
    torch.manual_seed(0)
    model = EvaluationSB().eval()
    x = torch.randn(1, 3, 256, 512)
    with torch.no_grad():
        xi = model(x)
    # 旋转向量前 3 维: 由 0.1 缩放, 初始网络输出下应远小于 1 rad
    assert xi[0, :3].abs().max().item() < 0.5


def test_evaluation_sb_backward():
    """训练路径需可反传。"""
    model = EvaluationSB()
    x = torch.randn(1, 3, 256, 512)
    xi = model(x)
    xi.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------- 位姿估计器

def test_simple_pose_estimator_shape_and_init():
    """末层零初始化 → 初始 forward 输出恰为 init_xi; 形状 (6,)。"""
    init = np.array([0.1, -0.2, 0.05, 0.3, 0.0, -0.1], dtype=np.float32)
    model = SimplePoseEstimator(init_xi=init)
    xi = model()
    assert xi.shape == (6,)
    np.testing.assert_allclose(xi.detach().numpy(), init, atol=1e-6)
    # init_xi / zero_input 为 buffer, 不在可训练参数中
    param_names = dict(model.named_parameters()).keys()
    assert "init_xi" not in param_names and "zero_input" not in param_names
    buf_names = dict(model.named_buffers()).keys()
    assert "init_xi" in buf_names and "zero_input" in buf_names


def test_simple_pose_estimator_default_init_zero():
    model = SimplePoseEstimator()
    xi = model()
    np.testing.assert_allclose(xi.detach().numpy(), np.zeros(6), atol=1e-6)


def test_simple_pose_estimator_convergence():
    """外层 Adam 优化: 50 步内拟合目标 ξ 至 1e-2 (论文 III-C-4 自动优化器)。"""
    torch.manual_seed(0)
    target = torch.tensor([0.05, -0.03, 0.08, 0.2, -0.15, 0.1])
    model = SimplePoseEstimator()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(50):
        opt.zero_grad()
        loss = ((model() - target) ** 2).sum()
        loss.backward()
        opt.step()
    err = (model().detach() - target).abs().max().item()
    assert err < 1e-2, f"50 步后最大分量误差 {err:.4f} >= 1e-2"


def test_simple_pose_estimator_convergence_with_init():
    """带初值偏置时同样能收敛到目标 (增量学习)。"""
    torch.manual_seed(1)
    init = np.array([0.0, 0.0, 1.5, 0.1, 0.0, 0.0], dtype=np.float32)  # 粗搜索初值
    target = torch.tensor([0.02, -0.01, 1.55, 0.15, -0.05, 0.02])
    model = SimplePoseEstimator(init_xi=init)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(50):
        opt.zero_grad()
        loss = ((model() - target) ** 2).sum()
        loss.backward()
        opt.step()
    err = (model().detach() - target).abs().max().item()
    assert err < 1e-2


def test_simple_pose_estimator_init_xi_no_grad():
    """init_xi 是常量偏置: 优化后 buffer 本身不变。"""
    init = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    model = SimplePoseEstimator(init_xi=init)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        model().sum().backward()
        opt.step()
    np.testing.assert_allclose(model.init_xi.numpy(), init, atol=0)


def test_standard_pose_estimator_forward():
    """标准估计器与 EvaluationSB 同构: 差分图 → ξ (B,6)。"""
    model = StandardPoseEstimator().eval()
    assert isinstance(model, EvaluationSB)
    x = torch.randn(1, 3, 256, 512)
    with torch.no_grad():
        xi = model(x)
    assert xi.shape == (1, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
