# 在线动态标定实现记录

记录日期：2026-07-28

适用硬件：Livox Mid-360 + Orbbec Gemini 335。

## 1. “动态”的边界

本实现支持传感器平台运动过程中持续接收同步数据，并按滑动窗口更新
LiDAR–Camera 外参：

- 同一窗口内假设刚性外参不变；
- 窗口之间允许因振动、碰撞或主动调整而改变；
- 标定计算期间 ROS2 主进程继续收流，不暂停传感器；
- 外参候选通过质量门控后持续发布 TF。

Gemini 335 直接提供米制度量深度，因此推理阶段不使用单目深度模型或 DAR。

## 2. 进程架构

本机 ROS2 使用系统 Python 3.14，CUDA PyTorch 环境使用 conda Python 3.12，
二者的 `rclpy` ABI 不兼容。因此使用异步双进程：

```text
Mid-360 PointCloud2 ─┐
                     ├─ ApproximateTimeSynchronizer
Gemini 对齐深度 ─────┘
                              │
                              ▼
                    系统 Python ROS2 主进程
                    - 数据质量预筛
                    - 12 帧滑动窗口
                    - 持续接收新帧
                              │
                     最新窗口文件 IPC
                              ▼
                    conda CUDA 标定工作进程
                    fast / both / selfsup
                              │
                              ▼
                    候选门控、平滑、重锚定
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
          /dst_calib/extrinsic    camera <- lidar TF
```

工作进程忙时，主进程继续更新内存窗口；完成后提交当时最新的完整窗口。

## 3. 与离线采集的区别

旧的 `capture_data.py` 为提高固态雷达密度，会把 3 秒内的 LiDAR 消息直接叠加，
然后配一张深度图，因此要求设备静止。

在线节点直接使用单条 PointCloud2 与深度图近似同步，不做 3 秒叠加。默认参数：

```yaml
online:
  window_size: 12
  window_stride: 4
  sync_slop_sec: 0.05
```

单条 PointCloud2 内仍有约一个扫描周期的时间跨度。高速或强角运动时，应输入已经
完成点级运动补偿、但仍表达在 LiDAR 坐标系中的 PointCloud2；世界坐标系地图点云
不能直接使用。

## 4. 在线模式

| 模式 | 用途 | 在线更新特性 |
|---|---|---|
| `fast` | 持续动态标定，推荐 | SB 逐帧精化与多帧融合，更新最快 |
| `both` | 低频高精度复核 | fast → PE → SB，当前迭代较多 |
| `selfsup` | 没有 SB 断点时 | 无需训练，但粗搜索和 PE 较慢 |

`fast` 和 `both` 需要 `runs/train_eva/best.pt`。每次接受的外参会成为下一窗口的
先验，使在线跟踪保持在评估模块的收敛邻域内。

## 5. 发布安全

`dst_calib.online.OnlineExtrinsicGate` 执行：

1. 检查候选是否为有效 SE(3)；
2. 拒绝非有限或超过 `online.max_final_cd` 的结果；
3. 普通小变化使用平移线性插值和旋转 SLERP；
4. 大于单次跳变阈值的候选暂不发布；
5. 连续多个大跳变候选彼此一致时，确认机械外参确实变化并重锚定。

这能避免单个退化场景或同步异常立即污染运行中的 TF，同时允许真实的外参突变。

## 6. 运行

```bash
bash /home/sw/DST/scripts/online_calib.sh
bash /home/sw/DST/scripts/online_calib.sh --skip-drivers
bash /home/sw/DST/scripts/online_calib.sh --init-yaml prior.yaml
bash /home/sw/DST/scripts/online_calib.sh --mode both
```

输出：

```text
runs/online_<时间戳>/
├── latest_extrinsic.yaml
├── results/update_000001/extrinsic.yaml
├── results/update_000001/worker.log
└── logs/livox.log、orbbec.log
```

默认不保留体积较大的窗口帧；调试时使用 `--save-windows`。

## 7. ROS2 接口

| 接口 | 类型 | 含义 |
|---|---|---|
| `/dst_calib/extrinsic` | `geometry_msgs/TransformStamped` | 已门控和平滑的外参 |
| `/dst_calib/status` | `std_msgs/String` | JSON 状态与拒绝原因 |
| `/tf` | TF | `camera_color_optical_frame <- livox_frame` |
| `~/trigger` | `std_srvs/Trigger` | 立即提交最新完整窗口 |
| `~/reset_filter` | `std_srvs/Trigger` | 清空已发布状态并重新获取 |

`T_cam_lidar` 满足：

```text
p_camera = T_cam_lidar @ p_lidar
```

所以 TF 的父坐标系是 camera，子坐标系是 lidar。

## 8. 涉及文件

| 文件 | 内容 |
|---|---|
| `scripts/online_calib_node.py` | ROS2 同步、滑窗、异步工作进程、发布与持久化 |
| `scripts/online_calib.sh` | 驱动与在线节点一键启动 |
| `dst_calib/online.py` | SE(3) 校验、SLERP、候选门控与动态重锚定 |
| `dst_calib/calibrate.py` | 新增 `--no_artifacts` 在线轻量输出 |
| `config/default.yaml` | 新增 `online` 配置段 |
| `tests/test_online.py` | 纯算法单元测试 |
| `tests/test_online_ros.py` | ROS2 滑窗到结果发布烟雾测试 |

## 9. 验证

- 在线门控测试：`7 passed`；
- ROS2 系统 Python 烟雾测试：`1 passed`；
- 真实 CUDA 工作进程使用 12 帧合成窗口成功生成仅含 `extrinsic.yaml` 的结果；
- 全量非慢速回归：`83 passed, 1 skipped, 3 deselected`。

ROS2 烟雾测试在 conda 测试中因 ABI 不兼容按设计跳过，并使用系统 Python 单独
执行通过。
