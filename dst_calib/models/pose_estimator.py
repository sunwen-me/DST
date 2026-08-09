"""位姿估计器 (Pose Estimation Block) — 论文 III-C-4。

两种实现:
- SimplePoseEstimator: 论文的轻量方案。常数零向量输入的 MLP, 其参数被外层
  Adam 迭代更新, 网络本身充当"自动优化器"——每次 forward 输出当前 ξ 估计,
  以自监督损失 (式19: L_t_ini + L_CD + L'_eva) 反传更新, 实现无监督标定。
- StandardPoseEstimator: 标准方案, 复用 EvaluationSB 结构, 由差分图直接
  回归 ξ (需训练, 论文 III-D-1)。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .evaluation import EvaluationSB


class SimplePoseEstimator(nn.Module):
    """轻量位姿估计器 (论文 III-C-4 的自动优化器实现)。

    结构: 固定零向量 buffer (维度 16) → MLP(hidden) → 6 维输出,
    输出按旋转/平移分别缩放 (rot 0.3 / trans 1.0) 后加上常量初值偏置:

        ξ = MLP(0) * scale + ξ_init

    ξ_init 注册为 buffer, 不参与梯度; 末层零初始化使初始输出恰为 ξ_init。
    外层用 Adam 对本模块参数迭代 (self_supervised.optimize_pose),
    等价于以 MLP 参数为优化变量的位姿求解。

    参数:
        hidden: 隐层宽度序列, 默认 (64, 64)
        init_xi: 初始位姿 ξ (numpy (6,)) 或 None (零向量)
    """

    IN_DIM = 16  # 固定零输入维度

    def __init__(self, hidden: tuple = (64, 64),
                 init_xi: Optional[np.ndarray] = None):
        super().__init__()
        # 固定零向量输入 (buffer: 随模块迁移设备, 不参与梯度)
        self.register_buffer("zero_input", torch.zeros(self.IN_DIM))
        # 初值偏置 (常量, 不参与梯度)
        if init_xi is None:
            init = torch.zeros(6, dtype=torch.float32)
        else:
            init = torch.as_tensor(np.asarray(init_xi, dtype=np.float32).reshape(6))
        self.register_buffer("init_xi", init)
        # 输出缩放: 旋转增量 0.3 / 平移增量 1.0, 平衡两者的有效学习率
        self.register_buffer(
            "out_scale",
            torch.tensor([0.3, 0.3, 0.3, 1.0, 1.0, 1.0], dtype=torch.float32))

        # MLP: 16 → hidden... → 6
        layers = []
        prev = self.IN_DIM
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        last = nn.Linear(prev, 6)
        # 末层零初始化 → 初始 forward() 输出恰等于 init_xi
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)

    def forward(self) -> torch.Tensor:
        """→ ξ (6,) 当前位姿估计 (可微, 梯度只流向 MLP 参数)。"""
        return self.mlp(self.zero_input) * self.out_scale + self.init_xi

    @torch.no_grad()
    def current_xi(self) -> np.ndarray:
        """便捷读取: 当前 ξ 的 numpy 拷贝 (6,)。"""
        return self.forward().detach().cpu().numpy()


class StandardPoseEstimator(EvaluationSB):
    """标准位姿估计器: 与 EvaluationSB 同构 (直接子类化),

    输入差分图 (B,3,H,W) 回归 ξ (B,6)。区别仅在用途:
    评估模块估计"位姿偏差"用于打分 (式18), 本模块直接回归外参增量
    (需按 III-D-1 以式(16)监督训练)。
    """

    def __init__(self, in_ch: int = 3, block_n: int = 5):
        super().__init__(in_ch=in_ch, block_n=block_n)
