# DST-Calib 复现 — Livox Mid-360 + Orbbec Gemini 335 无标定板外参标定

`both` 双路径的 PE→SB 修复、配置与验证记录见
[BOTH_PATH_FIX.md](BOTH_PATH_FIX.md)。
在线动态滑窗、TF 发布与安全门控的实现记录见
[ONLINE_CALIBRATION.md](ONLINE_CALIBRATION.md)。

复现论文 **arXiv 2601.01188** "DST-Calib: A Dual-Path, Self-Supervised,
Target-Free LiDAR-Camera Extrinsic Calibration Network" (Huang et al.)，
在 Livox Mid-360 (livox_ros_driver2, ROS2 lyrical) 与 Orbbec Gemini 335
(OrbbecSDK_ROS2 v2, 深度已对齐彩色) 上实现一键无标定板外参标定。

外参约定: `T_cam_lidar` 为 4x4 SE(3)，满足 `p_cam = T_cam_lidar @ p_lidar`
(相机光学系: x 右、y 下、z 前)。权威接口契约见 [INTERFACES.md](INTERFACES.md)。

---

## 1. 论文方法一页概述

DST-Calib 的核心思想: 把 LiDAR 点云与相机深度都投影成**深度图**，在图像域
比较二者的差异来估计/精化外参，全程不需要棋盘格等人工标定物。

| 论文模块 | 思想 | 本项目实现 |
|---|---|---|
| **双侧数据增广** (III-A, 式2-4) | 训练时对相机侧与 LiDAR 侧**各自**施加随机位姿扰动 ΔT_cam、ΔT_lidar (±5°/±0.5m, 轴权重 0.6/0.2/0.2)，CDP 在 T_cam 视角渲染、LDP 在 T_lidar 位姿渲染，监督目标为二者相对位姿 T_cam·T_lidar⁻¹ (式4)，比单侧扰动覆盖更大误差空间 | `dst_calib/augmentation.py` |
| **DAR 深度锚点精化** (III-B, 式5-10, 算法1) | 单目网络输出的归一化深度 d^C∈[0,1] 没有度量尺度；在 LDP 与 CDP 同时有效的像素处取 (d^C, d^L) 锚点对 (式5)，用动态规划选出"单调近线性"子集 (式8-10: d^C 严格增、d^L 非减、割线斜率非减即离散凸)，再分段线性映射把整幅归一化深度校正为米制深度 (式6-7)。Gemini 335 有真实度量深度，推理不需要 DAR，训练侧增广与纯 RGB 退化模式需要 | `dst_calib/dar.py` |
| **差分图** (式11-12) | D(u,v) = (LDP, [Δ]₊^{e_tar}, [Δ]₋^{e_tar})，Δ = LDP − CDP；按阈值 e_tar 把差异拆成"大误差"与"小误差"两通道，显式暴露未对齐区域 | `dst_calib/difference_map.py` |
| **评估模块** (III-C, 式13-16, 图7) | ResNet18 截断骨干 + CBAM 注意力 + 5×5 块处理位姿回归头，输入差分图回归位姿偏差 ξ=[r,t]∈R⁶ (监督训练, 可选) | `dst_calib/models/evaluation.py`, `models/cbam.py`, `dst_calib/train.py` |
| **自监督位姿估计器** (III-C-4, 式17-19) | 轻量 MLP (常数零输入) 输出 ξ 作为"可学习的外参变量"，用**截断 Chamfer 距离** (式17) 作自监督损失，Adam 由粗到细迭代 (体素 0.4/0.15/0.05m)，无需任何标注即可在线标定 | `dst_calib/models/pose_estimator.py`, `dst_calib/chamfer.py`, `dst_calib/losses.py`, `dst_calib/self_supervised.py` |
| **多帧优化** (III-D-2, 式20-23) | 各帧独立微调得候选 T_i，按自监督评分 s_i = exp(−L_CD) (式21) 排序取前 x 比例 (式22)，平移加权平均、旋转四元数加权平均 (式23) 得最终 T* | `dst_calib/multiframe.py` |
| 工程补充: 粗搜索 | Mid-360 是 360° 雷达而相机视场有限，论文假设有粗初值；本项目在 12×30° yaw × 3 档 pitch 网格上用大体素截断 Chamfer 打分选初值 | `self_supervised.coarse_search_init` |

几何/投影基础设施: `dst_calib/geometry.py` (可微 SE(3))、`dst_calib/projection.py`
(虚拟相机 256×512/f=600 与真实内参两条投影路径)。

---

## 2. 硬件连接

### Livox Mid-360 (网线)
1. 雷达网线接主机网口 `enp6s0` (或经交换机)，雷达供电 (9-27V)。
2. Mid-360 出厂 IP 为 `192.168.1.12`，**要求主机在 192.168.1.x 网段**
   (驱动配置里 host_ip 推荐 `192.168.1.5`)。若主机不在该网段:
   ```bash
   sudo ip addr add 192.168.1.5/24 dev enp6s0     # 附加一个静态地址即可, 不影响原地址
   ping -c1 192.168.1.12                          # 通了即可
   ```
   一键脚本会自动检测并在不通时打印上述修复命令 (不会自动改网络)。

### Orbbec Gemini 335 (USB)
1. 用原装线接 **USB 3.0** 口 (蓝色口；USB2 带宽不足会掉帧/降分辨率)。
2. 首次使用需安装 udev 规则 (见下节 `setup_drivers.sh` 提示)，装完重新插拔。

### 安装位姿
两传感器刚性固连 (共同支架)，相机视场与雷达前向有明显重叠。粗搜索能处理任意
yaw 安装角，但平移量不宜超过 ±0.5m 量级。

---

## 3. 安装

```text
/home/sw/DST                 本仓库
/opt/ros/lyrical             ROS2 (系统 python3 提供 rclpy/numpy/cv2/yaml)
/home/sw/Super-LIO/install   livox_ros_driver2 (已编译好, 直接复用)
/home/sw/Software/anaconda3/envs/dstcalib   标定计算环境 (torch/scipy/cv2, 已建好)
```

两套解释器严格分离: **采集**用系统 python3 (rclpy)，**标定**用 conda `dstcalib`
(torch)。`scripts/capture_data.py` 不 import torch/dst_calib。

安装 Orbbec 相机驱动 (一次性):

```bash
bash /home/sw/DST/scripts/setup_drivers.sh
# 脚本会: 克隆 OrbbecSDK_ROS2 (v2-main) → 提示手动装 udev(需 sudo) →
#         rosdep(可选) → colcon build --symlink-install → 自检 launch 文件
```

脚本提示的 udev 步骤需手动执行 (需要 sudo):

```bash
cd /home/sw/DST/ros2_ws/src/OrbbecSDK_ROS2/orbbec_camera/scripts
sudo bash install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 3.1 换机开发与 Git

在另一台电脑上先安装 Git，然后克隆项目仓库：

```bash
git clone https://github.com/<你的账号>/<仓库名>.git DST
cd DST
```

ROS2、Livox 工作区和标定用 Python 环境不随 Git 仓库提交。若它们不在本机的默认位置，运行脚本前设置对应路径：

```bash
export ROS_SETUP=/opt/ros/lyrical/setup.bash
export LIVOX_INSTALL=/path/to/Super-LIO/install/setup.bash
export CONDA_PY=/path/to/conda/envs/dstcalib/bin/python
export DST_CALIB_PYTHON="$CONDA_PY"
bash scripts/setup_drivers.sh
```

`setup_drivers.sh` 会重新获取 Orbbec 驱动并自动应用本项目的 lyrical 兼容补丁。编译目录、采集结果、模型断点和第三方源码不会进入根仓库。

日常开发使用以下流程：

```bash
git pull --ff-only
# 修改代码并运行测试
git add dst_calib scripts tests config README.md
git commit -m "描述本次修改"
git push
```

---

## 4. 一键标定

```bash
bash /home/sw/DST/scripts/one_click_calib.sh                # 全默认: 12 帧
bash /home/sw/DST/scripts/one_click_calib.sh --frames 8     # 少采几帧
bash /home/sw/DST/scripts/one_click_calib.sh --init prior.yaml   # 有先验外参, 跳过粗搜索
bash /home/sw/DST/scripts/one_click_calib.sh --skip-drivers      # 驱动已在别的终端跑着
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--frames N` | 12 (config `capture.num_frames`) | 采集帧数 |
| `--out DIR` | `runs/<时间戳>` | 产物目录 |
| `--skip-drivers` | 关 | 跳过网络自检与驱动启动，直接用现有话题采集 |
| `--init YAML` | 无 | 含 `T_cam_lidar` 的先验外参，跳过 yaw 粗搜索直接精化 |

环境变量 `CALIB_MODE` 选择标定模式并传给 `dst_calib.calibrate --mode`
(默认 `auto`: 存在断点 `runs/train_eva/best.pt` 则走 both 双路径, 否则纯自监督):

```bash
CALIB_MODE=fast bash /home/sw/DST/scripts/one_click_calib.sh   # 秒级标定 (需已训练评估模块)
```

三种模式的含义、评估模块的一键训练与模式选择建议见第 8 节。

脚本内部步骤: 网络自检 (ping 192.168.1.12) → 生成 livox JSON (host_ip 自动填
本机 192.168.1.x) → 后台启动 `livox_mid360_calib.launch.py` (PointCloud2 输出)
与 `orbbec_camera gemini_330_series.launch.py depth_registration:=true
enable_frame_sync:=true` → 等话题就绪 → 采集 → `python -m dst_calib.calibrate` →
打印外参摘要 + 可视化 → 清理驱动进程 (trap，Ctrl-C 也会清理)。

这一入口是用于建训练集/单次验收的**离线静态采集**，采集期间保持设备完全静止，
全程约 1-2 分钟。设备运动时请使用第 9 节的在线动态入口。

---

## 5. 采集场景建议

- **距离 1-5 m、结构丰富**: 桌椅、货架、门框、箱体、墙角等多平面/多深度层次；
  Gemini 335 可靠深度约 0.25-6 m，太远的结构相机看不到。
- **避免纯平墙/大面积玻璃/强反光**: 单一平面使 Chamfer 损失对平移不敏感,
  玻璃与强光会打坏结构光深度。
- **静止采集**: 整个 12 帧过程中传感器与场景都不要动 (Mid-360 每帧累积 3s
  非重复扫描点云，动了会拖影)。
- 12 帧默认即可；多帧优化 (式22-23) 会自动剔除低分帧。想更稳可在几个略不同的
  朝向各跑一次取均值。

---

## 6. 输出说明

每次运行产物在 `runs/<时间戳>/`:

```text
runs/20260726_153000/
├── data/frame_000.npz ...   原始帧 (points_lidar, depth, K, rgb, stamp)
├── extrinsic.yaml           标定结果
├── overlay_000.png ...      calibrate 生成的投影叠加图
├── vis/overlay_*.png        visualize_result.py 双面板对比图 + stats.txt
└── logs/                    livox/orbbec 驱动日志
```

`extrinsic.yaml` 字段:

| 字段 | 含义 |
|---|---|
| `T_cam_lidar` | 4×4，`p_cam = T @ p_lidar` (最终多帧加权结果 T*) |
| `T_lidar_cam` | 上者的逆 |
| `euler_zyx_deg` | ZYX 欧拉角 [yaw, pitch, roll] (度) |
| `translation_m` | 平移 [x, y, z] (米，相机系) |
| `final_cd` | 最终截断 Chamfer 距离 (米级，越小越好) |
| `mode` / `eva_ckpt_sha8` / `eva_used` | 所走标定模式、评估模块断点 sha256 前 8 位、评估模块是否参与 (第 8 节；旧版结果无这些字段) |
| 时间戳 | 标定完成时间 |

**overlay 图怎么看**: 左面板是 LiDAR 点按深度伪彩投到 RGB 上 —— 外参正确时
点云轮廓应与图像物体轮廓严格贴合 (门框点贴门框、箱沿点贴箱沿)；右面板绿色为
相机深度 Canny 边缘，LiDAR 点在深度突变处应压着绿边。若整体平移/旋转错位,
说明标定失败或设备中途移动。`vis/stats.txt` 给出投影 LiDAR 深度与相机深度差
|LDP−depth| 的中位数：室内场景正常应在 **0.02-0.05 m** 量级；>0.15 m 建议重标。

单独重跑可视化:

```bash
/home/sw/Software/anaconda3/envs/dstcalib/bin/python scripts/visualize_result.py \
    --data_dir runs/<ts>/data --extrinsic_yaml runs/<ts>/extrinsic.yaml --out runs/<ts>/vis
```

---

## 7. 离线重算与训练

对已采好的数据离线重新标定 (可改 `config/default.yaml` 超参):

```bash
cd /home/sw/DST
PYTHONPATH=/home/sw/DST /home/sw/Software/anaconda3/envs/dstcalib/bin/python \
    -m dst_calib.calibrate --data_dir runs/<ts>/data \
    --config config/default.yaml --output runs/<ts>_recalib \
    [--init_yaml runs/<ts>/extrinsic.yaml]
```

评估模块监督训练 (可选, 论文 III-D-1, 用双侧增广自造监督对):

```bash
PYTHONPATH=/home/sw/DST /home/sw/Software/anaconda3/envs/dstcalib/bin/python \
    -m dst_calib.train --data_dir runs/<ts>/data --arch sb   # sb=单分支差分图, db=双分支消融
```

多场景数据上的一键训练 (推荐入口 `scripts/train_eval_module.sh`) 见第 8 节。

单元/端到端测试 (合成场景, 断言 e_r<1°, e_t<0.1m, 式24):

```bash
cd /home/sw/DST && PYTHONPATH=/home/sw/DST \
  /home/sw/Software/anaconda3/envs/dstcalib/bin/python -m pytest tests/ -x
```

---

## 8. 完整论文形态（评估模块训练）

论文的完整形态是**双路径**: 除前述默认的自监督路径外，还有一条监督训练的
**评估模块**路径 (III-C, ResNet18+CBAM+块位姿头, 输入差分图回归位姿偏差)——
训练一次之后可反复用于**秒级**标定。`dst_calib.calibrate --mode`
(一键脚本经环境变量 `CALIB_MODE` 传入) 支持:

| 模式 | 路径 | 需训练? | 单次耗时 | 说明 |
|---|---|---|---|---|
| `selfsup` | 纯自监督由粗到细优化 | 否 | 分钟级 | 无断点时的默认路径，即前几节流程 |
| `fast` | 仅评估模块迭代精化 | 是 | 秒级 | 从名义/先验外参出发逐帧精化，式(20) 打分融合 |
| `both` | fast 先验 + PE 自监督优化 + SB 最终精化 | 是 | 分钟级 | 式19 全三项；以相同 Chamfer 指标做 fast 非退化保护 |
| `auto` (默认) | 自动选择 | — | — | 存在 `runs/train_eva/best.pt` → both，否则 selfsup |

`inference.eva_iters` 控制 fast 从粗初值开始的迭代次数；
`inference.both_eva_iters` 控制 PE 候选进入小邻域后，SB 最终精化的迭代次数。
两者默认均为 3，可分别调整。

### 完整工作流

1. **多场景采集 + 自监督标定**: 在 3-5 个结构不同的场景 (换房间/换朝向) 各跑一次
   `bash scripts/one_click_calib.sh`，每次自动产出 `runs/<ts>/data/frame_*.npz`
   与 `runs/<ts>/extrinsic.yaml` (该 session 训练时的基准外参)。建议总帧数 ≥30。
   单次采集内保持静止；session 之间安装位姿可微调 (各 session 用各自的
   extrinsic.yaml 作基准)。
2. **一键训练评估模块**:
   ```bash
   bash /home/sw/DST/scripts/train_eval_module.sh                     # 自动收集所有合格 run
   bash /home/sw/DST/scripts/train_eval_module.sh runs/A runs/B       # 或显式指定 session
   bash /home/sw/DST/scripts/train_eval_module.sh -- --epochs 100 --num_workers 4  # -- 后透传给 dst_calib.train
   ```
   自动收集规则: `runs/*/` 下同时含 `data/frame_*.npz` 与 `extrinsic.yaml`
   (即标定成功的 run)。产物 `runs/train_eva/best.pt`，中断后重跑自动断点续训。
3. **fast / both 标定**: 此后 `one_click_calib.sh` 默认 (`CALIB_MODE=auto`)
   检测到断点自动进入 both；需要秒级重标时:
   ```bash
   CALIB_MODE=fast bash scripts/one_click_calib.sh
   ```
   结果 yaml 以 `mode` / `eva_ckpt_sha8` / `eva_used` 字段标明实际所走路径。

### 预期训练时长 (RTX 3060)

估算式: **总时长 ≈ epochs × 总帧数 × samples_per_frame × t_样本**，
RTX 3060 上 t_样本 ≈ 0.02–0.06 s (含在线双侧增广的 CPU 数据生成；
`-- --num_workers 4` 时接近下限)。默认 epochs=200、samples_per_frame=32:

| 数据量 | 每 epoch 样本 | 预计总时长 |
|---|---|---|
| 3 场景 × 12 帧 = 36 帧 | 1152 | 约 1.5–4 小时 |
| 5 场景 × 12 帧 = 60 帧 | 1920 | 约 2.5–6.5 小时 |

赶时间可 `-- --epochs 100` 约减半 (精度略降)；CPU 训练慢 10–20 倍，不推荐。
训练脚本启动前会按上式打印本次数据量对应的估算。

### 模式选择建议

- **只标一次 / 尚未训练** → `selfsup` (auto 无断点时即它)，零准备开箱即用。
- **频繁重标** (设备常拆装、多套同型支架、产线批量) → 训练一次后用 `fast`
  秒级出结果。注意 fast 的可靠区约为训练增广覆盖的 ±5°/±0.5m：安装位姿相对
  名义/先验外参偏差超出此范围时评估模块外推失效，应回 `both`/`selfsup`。
- **追求最高精度且已有断点** → `both` (推荐，auto 有断点时即它)：fast 结果作
  初值与式(19)监督项，经 PE 自监督优化后再由 SB 精化并多帧融合；若组合结果在
  相同 Chamfer 指标上劣于 fast，会自动回退 fast。该模式跳过粗搜索，对对称场景
  的 yaw 分支歧义更稳健。
- 换了传感器/支架或场景风格与训练数据差异过大时，补采 1-2 个新场景重跑
  `train_eval_module.sh` (自动断点续训) 即可。

---

## 9. 在线动态标定

论文中的 online 含义是：传感器平台可以沿路线运动，PE 在标定窗口上在线自监督
优化；每个窗口内假设 LiDAR–Camera 外参固定，窗口之间允许因振动、碰撞或主动
调整发生变化。本工程对应入口：

```bash
# 默认启动驱动并以 fast 模式持续滑窗更新
bash /home/sw/DST/scripts/online_calib.sh

# 驱动已在其他终端运行
bash /home/sw/DST/scripts/online_calib.sh --skip-drivers

# 首窗口提供一个粗先验；先验在候选通过质量门控前不会发布
bash /home/sw/DST/scripts/online_calib.sh --init-yaml prior.yaml

# 低频运行完整 PE+SB 双路径
bash /home/sw/DST/scripts/online_calib.sh --mode both
```

`fast`/`both` 需要先训练好 `runs/train_eva/best.pt`。持续在线使用推荐 `fast`：
ROS2 系统 Python 负责同步收流和 TF，conda CUDA 子进程异步处理最新滑窗；工作
进程运行时不会停止接收新帧。`both` 和 `selfsup` 也可在线滑窗运行，但当前 PE
优化迭代较多，更新频率明显更低。

### 在线数据与发布

- 直接同步每条 `/livox/lidar` PointCloud2 与
  `/camera/depth/image_raw`，不再做离线入口的 3 秒裸累积。
- 默认 12 帧滑窗，每新增 4 帧提交最新窗口，最大同步差 50 ms；配置在
  `config/default.yaml: online`。
- Gemini 335 提供米制度量深度，在线路径不调用单目深度模型或 DAR。
- 接受后的 `T_cam_lidar` 发布到 `/dst_calib/extrinsic`，并以动态 TF 发布：
  `camera_color_optical_frame <- livox_frame`。
- 状态 JSON 发布到 `/dst_calib/status`。
- 最新平滑结果写入 `runs/online_<时间戳>/latest_extrinsic.yaml`；每次原始候选、
  工作日志保存在 `results/update_*`。

查看状态：

```bash
ros2 topic echo /dst_calib/status
ros2 topic echo /dst_calib/extrinsic
ros2 run tf2_ros tf2_echo camera_color_optical_frame livox_frame
```

手动立即提交最新完整窗口，或清空当前发布状态：

```bash
ros2 service call /dst_calib_online/trigger std_srvs/srv/Trigger '{}'
ros2 service call /dst_calib_online/reset_filter std_srvs/srv/Trigger '{}'
```

### 动态安全机制

候选首先检查 SE(3) 合法性与 `final_cd`。普通小变化按
`online.smoothing_alpha` 做平移线性插值与旋转 SLERP；超过单次旋转/平移跳变
阈值时不直接污染 TF，默认要连续两个彼此一致的窗口才确认外参确实改变并重锚定。

这里消除的是原先最严重的“3 秒点云对一张深度图”运动拖影。单条 Mid-360
PointCloud2 内仍有约一个扫描周期的点时间跨度；低速移动通常可直接使用，高速或
强角运动应把 `sensors.lidar.topic` 改成一个已经完成点级运动补偿、且仍表达在
LiDAR 坐标系中的 PointCloud2 话题。世界坐标系地图点云不能直接代替该输入。

---

## 10. 常见问题 (FAQ)

**Q: ping 不通 192.168.1.12 / 没有 `/livox/lidar` 话题？**
主机不在 192.168.1.x 网段 (常见: 路由器给的是 192.168.1.153 之外的段或雷达直连
无 DHCP)。执行 `sudo ip addr add 192.168.1.5/24 dev enp6s0` 后重试；再查网线、
雷达供电 (启动约 10s)、`runs/<ts>/logs/livox.log`。若之前跑过别的 livox 驱动
实例，先 `pkill -f livox_ros_driver2_node` (端口 5610x 被占会启动失败)。

**Q: 相机无 `/camera/*` 话题？**
① udev 规则没装 (`setup_drivers.sh` 有提示，装后重新插拔)；② 插在 USB2 口，
换蓝色 USB3 口；③ `lsusb | grep -i orbbec` 确认枚举；④ 看
`runs/<ts>/logs/orbbec.log`；⑤ 确认编译过驱动 (`ros2 pkg prefix orbbec_camera`)。

**Q: 深度单位不对 / 深度有效像素比例极低？**
Gemini 335 深度话题通常是 `16UC1` (毫米)，`capture_data.py` 按 encoding 自动
换算成米；若驱动配置成 `32FC1` 也能自动识别。异常时用
`--depth_unit mm|m` 强制指定。有效像素低 (<30%): 场景太远 (>6m)、强光直射、
黑色吸光/玻璃面过多——换场景。另必须带 `depth_registration:=true`，否则深度
未对齐彩色，分辨率不一致时 capture 会给中文警告。

**Q: 标定精度不达标 (overlay 明显错位 / final_cd 大 / |LDP−depth| 中位数 >0.15m)？**
按序排查: ① 采集时设备被碰动 → 重采；② 场景纯平墙 → 换结构丰富场景 (第 5 节)；
③ 粗搜索选错 yaw 分支 (360° 雷达对称场景可能歧义) → 用 `--init` 给一个手量的
粗外参 (角度精度 ±10° 即可) 再跑；④ 帧数太少 → `--frames 16`；⑤ 反复失败时
检查 `data/frame_*.npz` 单帧点数 (应 >10 万) 与深度有效比例。

**Q: `--skip-drivers` 什么时候用？**
驱动已在别的终端长期运行 (例如同时跑 SLAM)，或话题名与默认不同 (先改
`config/default.yaml` 的 `sensors.*` 话题再跑) 时；此时脚本不 ping、不生成
livox 配置、不启动也不清理任何驱动进程。

**Q: 换了话题名/量程怎么办？**
所有话题名、深度量程、采集与优化超参都集中在 `config/default.yaml`，
capture 与 calibrate 都从它读取，改一处即可。
