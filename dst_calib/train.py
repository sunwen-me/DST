"""评估模块训练 (论文 III-D-1, 可选路径; 训练超参见论文 IV 与 config/default.yaml)。

数据: capture_data.py 采集的帧目录 (load_frames 读取) + 在线双侧增广
(augmentation.double_sided_sample, 论文 III-A, 式(2)-(4))。
监督: 式(16) L_eva = L_rgt(式13) + L_tgt(式14) + L_cloud(式15)。
优化: AdamW(lr=5e-4, weight_decay=1e-4) + OneCycleLR, 200 epochs, batch 8。
断点: 每个 epoch 保存 <out>/checkpoint.pt, 重启自动恢复; 最优模型存 best.pt。

CLI:
    python -m dst_calib.train --data_dir <采集目录>... --arch sb|db \
        [--epochs N] [--out runs/train_eva] [--config config/default.yaml]

多 session (附录 A2): --data_dir 可给多个目录, 用 ConcatDataset 组合;
每目录的基准外参优先级: 目录内 extrinsic.yaml (calibrate.py 输出)
> --init_yaml 全局覆盖 > 名义外参。
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .augmentation import double_sided_sample
from .config import load_config
from .geometry import so3_exp
from .losses import cloud_loss, rotation_loss, translation_loss
from .models import EvaluationDB, EvaluationSB
from .projection import depth_cloud_from_gemini, virtual_camera_intrinsics
from .self_supervised import load_frames

__all__ = ["DoubleSidedDataset", "eva_loss_batch", "nominal_T_cam_lidar",
           "train", "main"]


# ---------------------------------------------------------------------- 数据集

def nominal_T_cam_lidar() -> np.ndarray:
    """名义外参: 前视相机安装在 LiDAR (x前 y左 z上) 附近时的标准轴变换,
    x_c = -y_l, y_c = -z_l, z_c = x_l, 平移取 0。

    注意: 双侧增广的标签公式 T_gt_virtual = ΔT_lidar^{-1} 虽与基准外参无关,
    但训练对的几何一致性要求基准外参 ≈ 真值 —— 真实 CDP 点坐标由真值外参
    固定, 基准偏离真值的量会整体成为 eva 路径的系统偏差 (见 augmentation
    模块 docstring 第 4 条)。名义外参仅是无任何先验时的最后回退; 正式训练
    应先对各 session 跑一次 selfsup 标定生成 extrinsic.yaml, 或用
    --init_yaml 提供精确先验。
    也是 calibrate.py fast 模式 (附录 A3) 无 --init_yaml 时的 T_init 来源。
    """
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [1.0, 0.0, 0.0]])
    return T


# 向后兼容别名 (原私有名, 契约 A3 要求公共名后保留旧引用不破坏)
_nominal_T_cam_lidar = nominal_T_cam_lidar


def _load_init_T(path: str) -> np.ndarray:
    """从 yaml 读取 4x4 名义外参 (键 T_cam_lidar, 与 calibrate.py 输出格式一致)。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "T_cam_lidar" in data:
        return np.asarray(data["T_cam_lidar"], dtype=float).reshape(4, 4)
    raise KeyError(f"{path} 中未找到键 'T_cam_lidar' (期望 4x4 矩阵)")


def _resolve_session_T(data_dir: str, init_yaml: str | None):
    """确定单个 session 的基准外参 (附录 A2)。

    优先级: 目录内 extrinsic.yaml (calibrate.py 输出) > --init_yaml 全局覆盖
    > None (名义外参, 由 DoubleSidedDataset 内部取 _nominal_T_cam_lidar)。
    返回 (T 或 None, 来源描述字符串)。
    """
    ext = Path(data_dir) / "extrinsic.yaml"
    if ext.exists():
        return _load_init_T(str(ext)), f"目录内 {ext}"
    if init_yaml:
        return _load_init_T(init_yaml), f"--init_yaml {init_yaml}"
    # 回退名义外参: 真实安装偏离名义值时, 全部训练对将带共轭化的系统性配准
    # 偏差 (基准≠真值, 见 augmentation 模块 docstring 第 4 条), eva 路径的
    # 精化不动点随之偏向名义外参 —— 醒目告警而非静默回退
    print(f"[警告] session {data_dir}: 未找到 extrinsic.yaml 且未给 --init_yaml, "
          "回退名义外参。基准外参偏离真值的量会成为评估模块 (fast/both) 的"
          "系统偏差 —— 建议先对该 session 跑一次 selfsup 标定 "
          "(one_click_calib.sh 或 dst_calib.calibrate) 生成 extrinsic.yaml。")
    return None, "名义外参"


class DoubleSidedDataset(Dataset):
    """帧目录 + 在线双侧增广 (论文 III-A) 的训练数据集。

    每个样本: 随机取一帧 → double_sided_sample 生成
      diff (3,H,W) 差分图(式12) / cdp,ldp (1,H,W) / T_cam,T_lidar (式3) /
      xi_gt (式4) / P (cloud_points,3) —— 式(15) L_cloud 所用的 LiDAR 下采样点云。
    增广在线随机进行, 名义 epoch 长度 = 帧数 × samples_per_frame。
    """

    def __init__(self, data_dir: str, cfg, T_gt_cam_lidar: np.ndarray | None = None,
                 samples_per_frame: int = 32, cloud_points: int = 2048,
                 seed: int = 0, stride: int = 2):
        self.cfg = cfg
        self.frames = load_frames(data_dir)
        if len(self.frames) == 0:
            raise RuntimeError(f"{data_dir} 中没有可用的 frame_*.npz")
        vc = cfg.virtual_camera
        self.size = (int(vc.height), int(vc.width))
        self.K = virtual_camera_intrinsics(float(vc.focal), self.size)
        self.T_gt = (nominal_T_cam_lidar() if T_gt_cam_lidar is None
                     else np.asarray(T_gt_cam_lidar, dtype=float).reshape(4, 4))
        cam = cfg.sensors.camera
        # 每帧的相机深度点云只需反投影一次
        self.cam_clouds = [
            depth_cloud_from_gemini(f.depth, f.K, d_min=float(cam.depth_min),
                                    d_max=float(cam.depth_max), stride=stride)
            for f in self.frames
        ]
        bad = [i for i, (f, c) in enumerate(zip(self.frames, self.cam_clouds))
               if np.asarray(f.points_lidar).reshape(-1, 3).shape[0] < 10
               or np.asarray(c).reshape(-1, 3).shape[0] < 10]
        if bad:
            raise ValueError(f"帧 {bad} 的 LiDAR 点云或相机深度点云过少, 无法训练")
        self.samples_per_frame = int(samples_per_frame)
        self.cloud_points = int(cloud_points)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """多 worker 时为各 worker 设定独立随机流 (见 _worker_init_fn)。"""
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frames) * self.samples_per_frame

    def __getitem__(self, idx: int) -> dict:
        fi = idx % len(self.frames)
        fr = self.frames[fi]
        out = double_sided_sample(fr.points_lidar, self.cam_clouds[fi],
                                  self.T_gt, self.K, self.size, self.cfg, self.rng)
        # 式(15) 用的 LiDAR 点下采样到固定大小, 便于批组装
        pts = np.asarray(fr.points_lidar, dtype=np.float32).reshape(-1, 3)
        choice = self.rng.choice(pts.shape[0], size=self.cloud_points,
                                 replace=pts.shape[0] < self.cloud_points)
        return {
            "diff": torch.from_numpy(np.ascontiguousarray(out["diff_map"])).float(),
            "cdp": torch.from_numpy(out["cdp"]).float().unsqueeze(0),
            "ldp": torch.from_numpy(out["ldp"]).float().unsqueeze(0),
            "T_cam": torch.from_numpy(out["T_cam"]).float(),
            "T_lidar": torch.from_numpy(out["T_lidar"]).float(),
            "xi_gt": torch.from_numpy(out["xi_gt"]).float(),
            "P": torch.from_numpy(pts[choice]).float(),
        }


def _iter_double_sided(dataset):
    """向下遍历数据集树, 产出所有 DoubleSidedDataset 叶子。

    多 session 时 DataLoader 拿到的是 ConcatDataset (info.dataset 不再是
    DoubleSidedDataset), 需递归其 .datasets; 单目录裸数据集路径保持不变。
    """
    if isinstance(dataset, DoubleSidedDataset):
        yield dataset
    elif isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            yield from _iter_double_sided(sub)


def _worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 初始化: 各 worker、各 epoch 使用互不相同的随机流。

    每个子集的 base_seed 在构造时已按 session 错开, 叠加 worker 偏移后
    保证 (session, worker) 两两独立; 再混入 info.seed —— DataLoader 每个
    epoch 重建 worker (默认 persistent_workers=False) 时从主进程 generator
    重新派生, 逐 epoch 不同 —— 否则固定种子会让每个 epoch 重播完全相同的
    增广随机流 (200 epoch 实际只见 1 个 epoch 量的扰动集合), 且与
    num_workers=0 (主进程 rng 持续推进) 行为不一致。整体仍由
    torch.manual_seed 决定, 跨运行可复现。
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    for ds in _iter_double_sided(info.dataset):
        ds.reseed(ds.base_seed + 1000 * (worker_id + 1) + info.seed % (2 ** 31))


# ------------------------------------------------------------------------ 损失

def eva_loss_batch(xi_pred: torch.Tensor, T_cam: torch.Tensor,
                   T_lidar: torch.Tensor, P: torch.Tensor):
    """式(16): L_eva = L_rgt + L_tgt + L_cloud, 逐样本计算后对 batch 取均值。

    xi_pred: (B,6) 网络输出 [r,t]; T_cam/T_lidar: (B,4,4) 式(3) 两侧位姿;
    P: (B,M,3) 原 LiDAR 系下采样点云。
    组成:
      L_rgt   式(13) = ||R_cam·(R·R_lidar)^{-1} - I||_{1,1}
      L_tgt   式(14) = ||t_cam - (t_lidar + t)||_2
      L_cloud 式(15) = Σ_p ||R(R_lidar·p + t_lidar) + t - (R_cam·p + t_cam)||_2
    返回 (total, {'rgt','tgt','cloud'} 标量 float 便于日志)。
    """
    B = xi_pred.shape[0]
    l_r = xi_pred.new_zeros(())
    l_t = xi_pred.new_zeros(())
    l_c = xi_pred.new_zeros(())
    for b in range(B):
        R_pred = so3_exp(xi_pred[b, :3])
        t_pred = xi_pred[b, 3:]
        R_cam, t_cam = T_cam[b, :3, :3], T_cam[b, :3, 3]
        R_lid, t_lid = T_lidar[b, :3, :3], T_lidar[b, :3, 3]
        l_r = l_r + rotation_loss(R_cam, R_pred @ R_lid)          # 式(13)
        l_t = l_t + translation_loss(t_cam, t_lid, t_pred)        # 式(14)
        l_c = l_c + cloud_loss(P[b], R_pred, t_pred,
                               R_cam, t_cam, R_lid, t_lid)        # 式(15)
    l_r, l_t, l_c = l_r / B, l_t / B, l_c / B
    total = l_r + l_t + l_c                                       # 式(16)
    return total, {"rgt": float(l_r.detach()), "tgt": float(l_t.detach()),
                   "cloud": float(l_c.detach())}


# -------------------------------------------------------------------- 训练主体

def _build_model(arch: str) -> nn.Module:
    if arch == "sb":
        return EvaluationSB()
    if arch == "db":
        return EvaluationDB()
    raise ValueError(f"未知 arch: {arch} (应为 sb 或 db)")


def _atomic_save(obj, path: Path) -> None:
    """先写临时文件再重命名, 避免中断产生损坏的断点文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(args) -> Path:
    """完整训练流程 (论文 III-D-1)。返回输出目录。"""
    cfg = load_config(args.config)
    tr = cfg.train
    lr = float(args.lr if args.lr is not None else tr.lr)
    wd = float(args.weight_decay if args.weight_decay is not None else tr.weight_decay)
    epochs = int(args.epochs if args.epochs is not None else tr.epochs)
    batch_size = int(args.batch_size if args.batch_size is not None else tr.batch_size)
    arch = (args.arch if args.arch is not None else tr.arch).lower()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[警告] 未检测到 CUDA GPU, 将在 CPU 上训练 —— 可以运行但会非常慢。")

    # ---- 多 session 数据集 (附录 A2): 每目录独立基准外参 + 错开的子集 seed
    data_dirs = args.data_dir if isinstance(args.data_dir, (list, tuple)) else [args.data_dir]
    subsets, sources = [], []
    for i, d in enumerate(data_dirs):
        T_i, src = _resolve_session_T(d, args.init_yaml)
        subsets.append(DoubleSidedDataset(d, cfg, T_i,
                                          samples_per_frame=args.samples_per_frame,
                                          cloud_points=args.cloud_points,
                                          seed=args.seed + 7919 * i))
        sources.append(src)
    # 单目录保持裸数据集 (旧路径不回归); 多目录用 ConcatDataset 组合
    dataset = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False,
                        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None)

    model = _build_model(arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    total_steps=total_steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"

    # ---- 断点恢复: 显式 --resume 优先, 否则自动检测输出目录下的 checkpoint.pt
    start_epoch, best_loss = 0, math.inf
    resume_path = Path(args.resume) if args.resume else (ckpt_path if ckpt_path.exists() else None)
    if resume_path is not None and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("arch", arch) != arch:
            raise RuntimeError(f"断点 arch={state.get('arch')} 与当前 --arch {arch} 不一致")
        model.load_state_dict(state["model"])
        if "optim" not in state:
            # best.pt / model_final.pt 是仅含权重的模型断点 (无 optim/sched),
            # 不能续训 —— 明确报错而非裸 KeyError('optim')
            raise RuntimeError(
                f"断点 {resume_path} 缺少优化器状态 (键 'optim') —— 它是仅含"
                "权重的模型断点 (best.pt / model_final.pt), 无法用于 --resume "
                "续训; 请改用同目录的 checkpoint.pt (每个 epoch 保存的完整断点)。")
        optimizer.load_state_dict(state["optim"])
        try:
            scheduler.load_state_dict(state["sched"])
        except Exception as exc:  # 步数配置变化时调度器状态可能不兼容
            print(f"[警告] 调度器状态恢复失败 ({exc}), 使用新调度器继续。")
        start_epoch = int(state.get("epoch", 0))
        best_loss = float(state.get("best_loss", math.inf))
        print(f"[恢复] 从 {resume_path} 继续: epoch {start_epoch}, best={best_loss:.4f}")

    print(f"[训练] arch={arch} device={device.type} epochs={epochs} "
          f"batch={batch_size} lr={lr:g} wd={wd:g} "
          f"sessions={len(subsets)} 总帧数={sum(len(ds.frames) for ds in subsets)} "
          f"每epoch样本={len(dataset)}")
    for i, (d, ds, src) in enumerate(zip(data_dirs, subsets, sources)):
        print(f"[数据] session {i}: {d} 帧数={len(ds.frames)} 基准外参={src}")

    # 断点记录虚拟相机参数 (附录 A2; eva_infer.load_eva 校验用, 加载端 .get 容错)
    vc = cfg.virtual_camera
    virtual_camera = {"height": int(vc.height), "width": int(vc.width),
                      "focal": float(vc.focal)}

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        sums = {"total": 0.0, "rgt": 0.0, "tgt": 0.0, "cloud": 0.0}
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if arch == "sb":
                xi_pred = model(batch["diff"])
            else:
                xi_pred = model(batch["cdp"], batch["ldp"])
            if xi_pred.dim() == 1:
                xi_pred = xi_pred.unsqueeze(0)
            loss, parts = eva_loss_batch(xi_pred, batch["T_cam"],
                                         batch["T_lidar"], batch["P"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # 工程稳定性
            optimizer.step()
            try:
                scheduler.step()
            except ValueError:
                pass  # 恢复训练时步数可能已达 OneCycle 总步数上限
            sums["total"] += float(loss.detach())
            for k in ("rgt", "tgt", "cloud"):
                sums[k] += parts[k]
            n_batches += 1
        means = {k: v / max(1, n_batches) for k, v in sums.items()}
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch + 1}/{epochs}] L_eva={means['total']:.4f} "
              f"(rgt {means['rgt']:.4f} | tgt {means['tgt']:.4f} | "
              f"cloud {means['cloud']:.4f}) lr={cur_lr:.2e} "
              f"{time.time() - t0:.1f}s")

        if means["total"] < best_loss:
            best_loss = means["total"]
            _atomic_save({"arch": arch, "model": model.state_dict(),
                          "loss": best_loss, "epoch": epoch + 1,
                          "virtual_camera": virtual_camera}, out / "best.pt")

        _atomic_save({"epoch": epoch + 1, "arch": arch,
                      "model": model.state_dict(),
                      "optim": optimizer.state_dict(),
                      "sched": scheduler.state_dict(),
                      "best_loss": best_loss,
                      "virtual_camera": virtual_camera}, ckpt_path)

    _atomic_save({"arch": arch, "model": model.state_dict(),
                  "epoch": epochs,
                  "virtual_camera": virtual_camera}, out / "model_final.pt")
    print(f"[完成] 最优 L_eva={best_loss:.4f}; 模型保存在 {out}")
    return out


# ------------------------------------------------------------------------- CLI

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dst_calib.train",
        description="DST-Calib 评估模块训练 (论文 III-D-1, 双侧增广 + 式16 损失)")
    p.add_argument("--data_dir", required=True, nargs="+",
                   help="capture_data.py 采集的帧目录, 可给多个 session; "
                        "目录内 extrinsic.yaml 优先作该 session 基准外参")
    p.add_argument("--arch", choices=["sb", "db"], default=None,
                   help="评估模块结构: sb 单分支差分图 / db 双分支 (默认取配置)")
    p.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认取配置 200)")
    p.add_argument("--out", default="runs/train_eva", help="输出目录 (断点/模型)")
    p.add_argument("--config", default=None, help="配置 yaml (默认 config/default.yaml)")
    p.add_argument("--batch_size", type=int, default=None, help="批大小 (默认取配置 8)")
    p.add_argument("--lr", type=float, default=None, help="学习率 (默认取配置 5e-4)")
    p.add_argument("--weight_decay", type=float, default=None, help="权重衰减 (默认 1e-4)")
    p.add_argument("--samples_per_frame", type=int, default=32,
                   help="每帧每 epoch 的在线增广样本数")
    p.add_argument("--cloud_points", type=int, default=2048,
                   help="式(15) L_cloud 使用的 LiDAR 下采样点数")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--resume", default=None, help="断点文件路径 (默认自动找 <out>/checkpoint.pt)")
    p.add_argument("--init_yaml", default=None,
                   help="全局基准外参 yaml (键 T_cam_lidar 4x4); 目录内 extrinsic.yaml "
                        "优先于它, 两者皆无时用名义外参 (标准轴变换)")
    p.add_argument("--device", default=None, help="cuda / cpu (默认自动)")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
