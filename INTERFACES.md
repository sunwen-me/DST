# DST-Calib 复现 — 模块接口规范（权威契约）

复现论文: arXiv 2601.01188 "DST-Calib: A Dual-Path, Self-Supervised, Target-Free
LiDAR-Camera Extrinsic Calibration Network" (Huang et al.)

目标硬件: Livox Mid-360 (通过 livox_ros_driver2, PointCloud2) + Orbbec Gemini 335
(通过 OrbbecSDK_ROS2, 深度已对齐彩色)。

**所有实现模块必须严格遵守本文件的函数签名与语义。** 论文公式编号以正文为准。

## 全局约定

- 外参记号: `T_cam_lidar` 是 4x4 SE(3)，满足 `p_cam = T_cam_lidar @ p_lidar`
  （LiDAR 坐标系 → 相机光学坐标系, x右 y下 z前）。
- 位姿向量 ξ = [r, t] ∈ R^6：r 为 so(3) 旋转向量(轴角)，t 为平移 (米)。
- 深度图: float32, 单位米, 0 表示无效/空像素。
- 点云: float32 (N,3)。
- torch 代码必须支持 CPU 与 CUDA，dtype=float32；所有需要反传的路径保持可微。
- numpy/torch 混用规则: 数据准备用 numpy；优化/网络用 torch。
- 虚拟相机 (论文 IV 实现细节): 投影尺寸 (H,W)=(256,512)，焦距 f=600，
  cx=W/2, cy=H/2。真实相机路径用 Gemini335 的 CameraInfo 内参。

## dst_calib/geometry.py （已完整实现，勿改动签名）

见文件本身。提供:
- `so3_exp(r)` / `so3_log(R)`：torch 可微 Rodrigues（支持 batch）
- `se3_from_rt(r, t)` / `rt_from_se3(T)`：torch
- `transform_points(T, pts)`：torch 可微
- `np_se3_from_rt`, `np_rt_from_se3`, `np_transform`, `euler_zyx_from_R`,
  `R_from_euler_zyx`, `quat_mean_rotations`（numpy 工具）

## dst_calib/projection.py

```python
def virtual_camera_intrinsics(f: float = 600.0, size: tuple = (256, 512)) -> np.ndarray
    """返回 3x3 K，cx=W/2, cy=H/2。size=(H,W)。"""

def project_points_to_depth(points_cam: np.ndarray, K: np.ndarray, size: tuple,
                            min_depth: float = 0.05, max_depth: float = 60.0) -> np.ndarray
    """点云(相机系, N,3) → 深度图 (H,W) float32。z-buffer 取最小深度，空像素为 0。"""

def backproject_depth(depth: np.ndarray, K: np.ndarray,
                      min_depth: float = 0.05, max_depth: float = 60.0)
    -> tuple[np.ndarray, np.ndarray]
    """深度图 → (points (M,3) 相机系, pixels (M,2) 整数 uv)。跳过无效像素。"""

def generate_ldp(points_lidar: np.ndarray, T_cam_lidar: np.ndarray,
                 K: np.ndarray, size: tuple) -> np.ndarray
    """LiDAR Depth Projection: 把 LiDAR 点变换到相机系后投影成深度图。"""

def generate_cdp(points_cam_depth: np.ndarray, T_view_cam: np.ndarray,
                 K: np.ndarray, size: tuple) -> np.ndarray
    """Camera Depth Projection: 相机深度点云经视角变换 T_view_cam 后投影。
    T_view_cam=I 时即原视角。"""

def depth_cloud_from_gemini(depth: np.ndarray, K_color: np.ndarray,
                            d_min: float = 0.25, d_max: float = 6.0,
                            stride: int = 1) -> np.ndarray
    """Gemini335 已对齐彩色的度量深度图 → 相机系点云 (N,3)。stride 亚采样。"""
```

## dst_calib/dar.py — 深度锚点精化 (论文 III-B, 式5-10, 算法1)

单目估计深度(归一化 d^C∈[0,1]) 用 LiDAR 度量深度校正。Gemini335 有真实深度时
推理不需要 DAR，但训练侧双侧增广、以及纯 RGB 退化模式需要。

```python
def extract_anchors(ldp: np.ndarray, cdp_norm: np.ndarray) -> np.ndarray
    """式(5): 在两幅深度投影同时有效的像素处取 (d^C, d^L) 对，返回 (K,2)
    [:,0]=归一化相机深度, [:,1]=LiDAR 度量深度。"""

def select_anchors_monotone(anchors: np.ndarray) -> np.ndarray
    """式(8)-(10) 单调近线性锚点选择: 先按 d^C 升序排序，再用 O(n^2) 动态规划求
    最长子序列 S 满足: d^C 严格递增、d^L 非递减、且相邻割线斜率
    s_k=(dL_{k+1}-dL_k)/(dC_{k+1}-dC_k) 非递减(离散凸性, 式9)。
    返回选中的 (K,2) 锚点。DP 状态: dp[j][i]=以 (i,j) 为最后两点的最长长度，
    转移需 slope(i,j) >= slope(h,i)。n 大时先对 anchors 做分位数抽样(≤400点)。"""

def piecewise_linear_remap(depth_norm: np.ndarray, anchors: np.ndarray) -> np.ndarray
    """式(6)(7): 用选中锚点做分段线性映射 f:[0,1]→R+，逐像素校正整幅归一化深度图。
    锚点范围外投影到端点常数值。无效(0)像素保持 0。"""

def refine_depth(mono_depth_norm: np.ndarray, points_lidar: np.ndarray,
                 T_cam_lidar: np.ndarray, K: np.ndarray) -> np.ndarray
    """完整 DAR 流程: LDP 生成 → 锚点提取 → 单调选择 → 分段线性重映射。
    返回度量深度图。"""
```

## dst_calib/difference_map.py — 式(11)(12)

```python
def build_difference_map(ldp: np.ndarray, cdp: np.ndarray, e_tar: float = 0.1)
    -> np.ndarray
    """D(u,v) = ( LDP, [Δ]_+^{e_tar}, [Δ]_-^{e_tar} ), Δ = LDP - CDP (式11)。
    [Δ]_+ = Δ if |Δ|>e_tar else 0;  [Δ]_- = Δ if |Δ|<=e_tar else 0 (式12)。
    仅在两者皆有效的像素计算 Δ；LDP 通道保留原值。返回 (3,H,W) float32。"""

def build_difference_map_torch(ldp: torch.Tensor, cdp: torch.Tensor,
                               e_tar: float = 0.1) -> torch.Tensor
    """同上，torch 版 (B,3,H,W)，供训练/评估模块前向使用。"""
```

## dst_calib/chamfer.py — 式(17)

```python
def chamfer_distance(P: torch.Tensor, Q: torch.Tensor,
                     alpha: float = 0.5, beta: float = 0.5,
                     chunk: int = 4096) -> torch.Tensor
    """L_CD = α/|P| Σ_{p∈P} min_{q∈Q} ||p-q||² + β/|Q| Σ_{q∈Q} min_{p∈P} ||q-p||²。
    可微(梯度经最近对距离回传)。分块计算避免 NxM 大矩阵；
    CPU 且点数大时可用 scipy cKDTree 求最近邻索引(detach)再以 torch 计算距离。"""

def truncated_chamfer(P, Q, alpha=0.5, beta=0.5, trunc: float = 1.0, chunk=4096)
    """截断版: 距离超过 trunc 的最近对贡献按 trunc² 截断，提高离群稳健性。
    自监督优化默认用它。"""
```

## dst_calib/losses.py — 式(13)-(19)

```python
def rotation_loss(R_cam, R_lidar_or_pred) -> torch.Tensor   # 式(13) ||R_cam(R R_lidar)^-1 - I||_1,1
def translation_loss(t_cam, t_lidar, t_pred) -> torch.Tensor  # 式(14) ||t_cam-(t_lidar+t)||_2
def cloud_loss(P, R_pred, t_pred, R_cam, t_cam, R_lidar, t_lidar) -> torch.Tensor  # 式(15)
def eva_total_loss(...) -> torch.Tensor                     # 式(16) 三项之和
def eva_score_loss(xi_pred, xi_eva, a: float = 0.1) -> torch.Tensor
    # 式(18) L'_eva = a*||e_eva - e||_2 + ||t_eva - t||_2, e 为欧拉角(从旋转向量转)
    # (实现: 旋转差取相对旋转 R_eva·R_predᵀ 的欧拉角 — 小差异下一阶等价,
    #  任意安装姿态下万向节锁安全; 式(20) 同理)
def pose_estimator_loss(xi, P, Q, xi_init=None, xi_eva=None,
                        a=0.1, trunc=1.0) -> tuple[torch.Tensor, dict]
    # 式(19) L_pe = L_t_ini + L_CD + L'_eva；xi_init/xi_eva 为 None 时对应项为 0
    # 返回 (total, {'cd':..,'tini':..,'eva':..}) 便于日志
```

## dst_calib/models/cbam.py
标准 CBAM (通道注意力 + 空间注意力)，`CBAM(channels, reduction=16, spatial_kernel=7)`。

## dst_calib/models/evaluation.py — 评估模块 (论文 III-C, 图7)

```python
class BlockPoseHead(nn.Module):
    """块处理+位姿回归 (III-C-3): 特征图划分 n×n=5×5 网格块，每块经卷积压缩为
    向量 B_i，拼接展开为 F_p，过全连接聚合，旋转/平移解耦两个 MLP 头，
    输出 ξ=[r,t]∈R^6。"""

class EvaluationSB(nn.Module):
    """单分支: 输入差分图 (B,3,H,W) → ResNet18 截断骨干(至 layer3 或 layer4，
    通道自适应) + CBAM → BlockPoseHead → ξ。forward(diff_map) -> xi (B,6)。"""

class EvaluationDB(nn.Module):
    """双分支对照: CDP、LDP 各一条 ResNet18+CBAM 分支，特征拼接后 BlockPoseHead。
    forward(cdp, ldp) -> xi (B,6)。（消融对照用）"""
```
ResNet 从零构建（不依赖 torchvision 预训练权重下载），输入 1 或 3 通道自适应。

## dst_calib/models/pose_estimator.py — 位姿估计器 (III-C-4)

```python
class SimplePoseEstimator(nn.Module):
    """论文的轻量实现: 常数零向量输入的 MLP，作为自动优化器。
    __init__(hidden=(64,64), init_xi: Optional[np.ndarray]=None)
    forward() -> xi (6,)  ：MLP(0向量) + init_xi(常量偏置)。
    参数被外层 Adam 迭代更新，实现全自监督标定。"""

class StandardPoseEstimator(nn.Module):
    """标准实现: 与 EvaluationSB 同构，输入差分图回归 ξ。"""
```

## dst_calib/self_supervised.py — 一键标定核心 (III-C-4 + III-D-2)

```python
@dataclass
class Frame:
    points_lidar: np.ndarray   # (N,3) LiDAR 系
    depth: np.ndarray          # (H,W) 米, 对齐彩色
    K: np.ndarray              # (3,3) 彩色内参
    rgb: Optional[np.ndarray]  # (H,W,3) uint8, 可为 None
    stamp: float

def load_frames(data_dir: str) -> list[Frame]
    """读取 capture_data.py 保存的 frame_*.npz。"""

def coarse_search_init(frames, cfg) -> np.ndarray
    """多起点粗搜索: Mid-360 为 360° 雷达而相机视场有限，先在 yaw 网格
    (默认 12×30°) × 少量 pitch 上，用大体素(0.4m)下采样点云做少量迭代
    截断 Chamfer 评分，返回最优初始 ξ (6,)。若 cfg 提供 init_T 则直接返回其 ξ。"""

def optimize_pose(frames, xi_init, cfg, xi_eva=None) -> tuple[np.ndarray, list]
    """自监督位姿优化: SimplePoseEstimator(init_xi=xi_init) + Adam。
    由粗到细阶段 (体素 0.4/0.15/0.05m, 迭代 cfg.iters 各阶段)，每次迭代:
      随机取一帧(或小批帧) → ξ=model() → T=se3_from_rt →
      L_pe = truncated_chamfer(T·P_i^下采样, Q_i^下采样) (+可选 L_tini, L'_eva) → 反传。
    LiDAR 点云先按相机视场+距离裁剪(用当前 ξ, 每阶段更新一次裁剪)。
    ≥30 批 (论文 IV)。返回 (最终 ξ, 每帧最终 T 列表用于多帧优化)。
    注: 每帧 T 相同(全局 ξ)——按论文多帧输出取各帧独立微调结果:
    最后阶段对每帧单独 fine-tune 少量迭代得到 T_i。"""

def calibrate_self_supervised(frames, cfg) -> dict
    """一键入口: coarse_search_init → optimize_pose → multiframe.optimize →
    返回 {'T_cam_lidar': 4x4, 'per_frame_T': [...], 'scores': [...],
          'final_cd': float, 'log': [...]}"""
```

## dst_calib/multiframe.py — 式(20)-(23)

```python
def score_self_supervised(T_list, frames, cfg) -> np.ndarray
    """式(21): s_i = exp(-L_CD(T_i·P_i, Q_i))。用下采样点云。"""

def score_full_supervised(T_list, xi_eva_list, a=0.1) -> np.ndarray
    """式(20): s_i = exp(-(a||e'_i - e_i||_2 + ||t'_i - t_i||_2))。
    (实现: 旋转差取相对旋转 R'_i·R_iᵀ 的欧拉角范数 — 小差异下一阶等价,
    万向节锁安全, 见 losses.eva_score_loss 注记。)"""

def select_and_average(T_list, scores, x: float = 0.3,
                       weighting: str = "score") -> np.ndarray
    """式(22)(23): 按分数降序取 k=ceil(x·n) 个；平移加权平均，
    旋转用 quat_mean_rotations 加权平均。返回 T* (4,4)。"""
```

## dst_calib/augmentation.py — 双侧数据增广 (III-A, 式2-4) — 训练用

```python
def sample_perturbation(rot_range_deg=5.0, trans_range_m=0.5,
                        axis_weights=(0.6,0.2,0.2), rng=None) -> np.ndarray
    """论文 IV: 默认 [±5°, ±0.5m]，轴权重 [0.6,0.2,0.2] →
    实际每轴 [±3°,±1°,±1°], [±0.3m,±0.1m,±0.1m]。返回 ξ 扰动 (6,)。"""

def double_sided_sample(points_lidar, cam_depth_points, T_gt_cam_lidar, K, size,
                        cfg, rng=None) -> dict
    """式(3)(4): T_cam=ΔT_cam·T_gt, T_lidar=ΔT_lidar·T_cam →
    CDP 在 T_cam 视角生成、LDP 在 T_lidar 位姿生成，
    监督目标 T_gt_virtual = T_cam·(T_lidar)^-1 (式4)。
    返回 {'ldp','cdp','diff_map','xi_gt','T_gt'}。"""
```

## dst_calib/train.py — 评估模块训练 (III-D-1, 可选)
AdamW lr=5e-4 wd=1e-4, OneCycle, 200 epochs, batch 8。数据: 帧目录 + 双侧增广。
损失 式(16)。CLI: `python -m dst_calib.train --data_dir ... --arch sb|db`。

## dst_calib/calibrate.py — 离线标定 CLI

```python
# python -m dst_calib.calibrate --data_dir <采集目录> --config config/default.yaml
#        --output <结果目录> [--init_yaml 先验外参.yaml]
# 流程: load_frames → calibrate_self_supervised → 写 extrinsic.yaml
#       (T_cam_lidar, T_lidar_cam, euler_zyx_deg, translation_m, final_cd, 时间戳)
#       → 对每帧渲染投影叠加图 overlay_*.png (LiDAR点按深度着色投到RGB)
```

## scripts/（系统 python + rclpy，不进 venv 也能跑；标定用 venv python）

- `capture_data.py`: rclpy 节点。参数: --out, --num_frames (默认12), --accumulate_sec
  (默认3.0), --interval_sec (默认1.0), 话题名从 config/default.yaml 读。
  订阅 Livox PointCloud2(累积一窗口)、Gemini 深度(16UC1 mm 或 32FC1 m, 自动判别,
  --depth_unit 覆盖)、CameraInfo、彩色图。每帧存 frame_%03d.npz
  (points_lidar float32, depth float32米, K, rgb, stamp)。QoS: 传感器数据
  best_effort。彩色/深度用 message_filters 近似同步。
- `visualize_result.py`: --data_dir --extrinsic_yaml → overlay 图 + 3D 一致性统计。
- `one_click_calib.sh`: 见 README。检查网络(主机须 192.168.1.5 网段可达雷达
  192.168.1.12, 否则打印修复命令并退出)、生成 livox 配置(host ip 自动填)、
  启动 livox_ros_driver2 (msg_MID360_launch 改 PointCloud2 输出) 与
  OrbbecSDK_ROS2 (depth_registration:=true)、等话题就绪、capture、calibrate、
  可视化、清理进程。所有产物进 runs/<时间戳>/。
- `setup_drivers.sh`: 克隆并编译 OrbbecSDK_ROS2 到 /home/sw/DST/ros2_ws。
- `online_calib_node.py`: 系统 Python ROS2 在线节点。近似同步单条 LiDAR
  PointCloud2 与 Gemini 对齐深度，维护滑窗；异步调用 conda CUDA
  `dst_calib.calibrate --no_artifacts`，主线程持续收流。候选经过
  `dst_calib.online.OnlineExtrinsicGate` 的 SE(3)/CD 门控、SLERP 平滑与大跳变
  连续确认后，发布 `/dst_calib/extrinsic` 和动态 TF
  `camera_color_optical_frame <- livox_frame`。窗口内外参固定，窗口间可重锚定。
- `online_calib.sh`: 在线一键入口；默认 fast，可选 both/selfsup，支持
  `--skip-drivers`、先验、窗口大小/步长和窗口留存。

## dst_calib/online.py — 在线结果状态管理

纯 NumPy、无 ROS 依赖。主要契约：

```python
def validate_se3(T) -> tuple[bool, str]
def interpolate_se3(T_old, T_new, alpha) -> np.ndarray

class OnlineExtrinsicGate:
    def update(T_candidate, final_cd) -> GateDecision
    @property
    def current() -> np.ndarray | None
    def reset() -> None
```

小变化平滑接受；非有限/高 CD 候选拒绝；超过单次跳变阈值的候选须连续一致才
重锚定，避免单个坏窗口污染运行中的 TF。

## config/default.yaml
见文件。所有超参集中于此，模块通过 `dst_calib.config.load_config()` 读取
（返回嵌套 SimpleNamespace，实现于 config.py，已提供）。

## tests/
- `test_geometry.py`, `test_dar.py`, `test_projection.py`, `test_chamfer.py`,
  `test_models.py`: 单元测试 (pytest)。
- `test_online.py`: 在线候选 SE(3) 校验、四元数/SLERP、小变化平滑、
  高 CD 拒绝与动态突变确认。
- `test_synthetic.py`: 端到端: 合成房间场景(带遮挡的墙/箱体) → 模拟 Mid-360
  稀疏采样 + 相机深度图(z-buffer) → 已知 T_gt, 从粗搜索开始跑
  calibrate_self_supervised → 断言 e_r<1° 且 e_t<0.1m (式24 指标)。

---

# 附录 A — 完整论文形态（评估模块训练 + 快速标定）接口契约

目标: 论文三种使用形态全部可用 (III-C):
`selfsup`(现状) / `fast`(仅全监督路径, 秒级) / `both`(双路径全激活, 推荐)。

## A1. dst_calib/eva_infer.py （新模块）

```python
def load_eva(ckpt_path: str, device) -> tuple[torch.nn.Module, dict]
    """加载 train.py 断点 (键: arch, model, 可选 virtual_camera)。返回
    (eval() 模式 model, meta dict)。meta 至少含 arch; 若断点存有
    virtual_camera 则校验/覆盖 cfg 的虚拟相机参数 (不一致时打印警告并以断点为准)。"""

def eva_refine(frame, T_est: np.ndarray, model, cfg, device,
               n_iters: int = 3) -> tuple[np.ndarray, np.ndarray]
    """单帧迭代精化 (RegNet 式)。每轮:
      LDP = generate_ldp(points_lidar, T_est, K_virtual, size)      # T_lidar 角色
      CDP = generate_cdp(cam_cloud, I, K_virtual, size)             # 真实相机即 ΔT_cam=I
      D   = build_difference_map(LDP, CDP, e_tar)
      ξ   = model(D)  (SB) 或 model(cdp,ldp) (DB)
      T_est ← se3(ξ) @ T_est        # 网络输出近似 T_true·T_est^{-1} (训练约定
                                    # T_gt_virtual = T_cam·T_lidar^{-1}, 见 augmentation)
    收敛判据: ‖ξ‖ 足够小提前停。返回 (T_refined, xi_last)。
    cam_cloud 用 depth_cloud_from_gemini(frame.depth, frame.K, ...) 反投影,
    per-frame 计算一次并缓存于调用方。"""

def eva_calibrate_frames(frames, T_init: np.ndarray, model, cfg, device,
                         n_iters: int = 3) -> dict
    """逐帧 eva_refine → T_list;
    式(20) 打分: 对每个 T_i, 以 T_i 为输入位姿再前向一次得 T'_i (=评估结果),
    s_i = exp(-(a·||e'_i - e_i||₂ + ||t'_i - t_i||₂)), a=0.1;
    multiframe.select_and_average(T_list, scores, x) → T*。
    返回 {'T_cam_lidar','per_frame_T','scores','xi_eva'}  (xi_eva: T* 的 ξ,
    供 both 模式作先验)。"""
```

## A2. train.py 多场景扩展（向后兼容）

- `--data_dir` 改为 `nargs='+'`：多个采集目录; 每目录若存在
  `extrinsic.yaml`(calibrate.py 输出) 或 `--init_yaml` 指定, 用作该 session 的
  基准外参, 否则退回名义外参。实现: `MultiSessionDataset = ConcatDataset([
  DoubleSidedDataset(dir_i, cfg, T_i, ...)])`, 各子集独立 seed。
- 断点新增键 `virtual_camera = {"height","width","focal"}` (加载端 .get 容错)。
- 其余训练逻辑/超参不变 (式16, AdamW 5e-4, OneCycle, 200ep 默认)。

## A3. calibrate.py 模式扩展

- 新参数: `--mode {auto,selfsup,fast,both}` 默认 auto; `--eva_ckpt PATH` 默认
  `runs/train_eva/best.pt`(相对 DST 根)。auto 语义: 断点存在→both, 否则 selfsup。
- fast: coarse/自监督全跳过; T_init 来源 --init_yaml 或名义外参;
  eva_calibrate_frames 直接出结果 (秒级)。
- both: 先 fast 得 xi_eva 与 T_eva → 以 T_eva 为初值 optimize_pose(...,
  xi_eva=xi_eva)(式19 全三项) → 逐帧微调 → 以 PE 候选为先验执行 SB 精化
  → 多帧融合用式(20)打分；失败回退式(21)，若最终截断 Chamfer 劣于 fast
  则触发非退化保护并回退 fast。SB 最终精化轮数由
  `inference.both_eva_iters` 控制（旧配置缺省为 `max(eva_iters, 3)`）。
- 输出 yaml 增加字段: mode, eva_ckpt_sha8 (断点 sha256 前 8 位),
  eva_used: true/false。
- 与 one_click_calib.sh 打通: 环境变量 CALIB_MODE (默认 auto) 传 --mode。

## A4. scripts/train_eval_module.sh （新, 一键训练）

用法: `./scripts/train_eval_module.sh <session_dir>... [-- 额外train参数]`
无参数时默认收集 `runs/*/frames`(存在 extrinsic.yaml 的 run 目录)。
流程: 逐 session 校验帧数 → 打印数据摘要 → conda python -m dst_calib.train
--data_dir <全部session> → 训练完打印 best.pt 路径与验证损失, 提示
one_click_calib.sh 下次将自动进入 both 模式。GPU 检查同 one_click。

## A5. 测试契约

- tests/test_eva_infer.py: 合成帧 + 随机初始化 SB 模型跑通形状/收敛判据/打分
  (不要求精度); 断点 round-trip (save→load_eva)。
- tests/test_full_paper_form.py (慢, 标记 @pytest.mark.slow):
  复用 test_synthetic 场景生成器 → 3 场景×4帧, 名义外参+双侧增广训练 tiny SB
  (缩小: epochs≈8, samples_per_frame≈8, 视场 128×256) →
  (a) fast 模式在 ±5°/±0.3m 失准内把误差降到 <2.5°/<0.20m;
  (b) both 模式最终 e_r<1°, e_t<0.1m (不劣于纯 selfsup 基线);
  (c) 输出 yaml 含 mode/eva_used 字段。
- 全量 pytest 必须通过 (68 旧例不回归)。
