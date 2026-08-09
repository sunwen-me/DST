# -*- coding: utf-8 -*-
"""DST-Calib 专用 Livox Mid-360 launch (PointCloud2 输出)。

与 Super-LIO 的 launch_ROS2/msg_MID360_launch.py 同构, 差异:
  - xfer_format = 0 → 输出标准 sensor_msgs/PointCloud2 (capture_data.py 需要);
  - user_config_path 由环境变量 LIVOX_CFG 传入 (one_click_calib.sh 在
    /home/sw/DST/runs/<时间戳>/ 下运行时生成, host_ip 自动填当前主机
    192.168.1.x 地址), 未设置时用默认路径; 若默认路径不存在则按
    MID360_config.json 模板即时生成一份, 便于单独调试本 launch。

用法:
  source /opt/ros/lyrical/setup.bash
  source /home/sw/Super-LIO/install/setup.bash
  LIVOX_CFG=/path/to/cfg.json ros2 launch /home/sw/DST/scripts/livox_mid360_calib.launch.py
"""
import json
import os
from pathlib import Path
import subprocess

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

# 默认配置 JSON 路径 (LIVOX_CFG 未设置时)
DEFAULT_CFG = str(Path(__file__).resolve().parents[1] / "runs" / "livox_calib_config.json")
LIDAR_IP = "192.168.1.12"
FALLBACK_HOST_IP = "192.168.1.5"


def _detect_host_ip() -> str:
    """探测本机 192.168.1.x 地址 (Mid-360 要求主机与雷达同网段)。"""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"],
                             capture_output=True, text=True, timeout=5).stdout
        for tok in out.split():
            if tok.startswith("192.168.1.") and "/" in tok:
                return tok.split("/")[0]
    except Exception:
        pass
    return FALLBACK_HOST_IP


def _make_config(path: str, host_ip: str) -> None:
    """按 livox_ros_driver2 的 MID360_config.json 模板生成配置。"""
    cfg = {
        "lidar_summary_info": {"lidar_type": 8},
        "MID360": {
            "lidar_net_info": {
                "cmd_data_port": 56100,
                "push_msg_port": 56200,
                "point_data_port": 56300,
                "imu_data_port": 56400,
                "log_data_port": 56500,
            },
            "host_net_info": {
                "cmd_data_ip": host_ip, "cmd_data_port": 56101,
                "push_msg_ip": host_ip, "push_msg_port": 56201,
                "point_data_ip": host_ip, "point_data_port": 56301,
                "imu_data_ip": host_ip, "imu_data_port": 56401,
                "log_data_ip": "", "log_data_port": 56501,
            },
        },
        "lidar_configs": [{
            "ip": LIDAR_IP,
            "pcl_data_type": 1,
            "pattern_mode": 0,
            "extrinsic_parameter": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                                    "x": 0, "y": 0, "z": 0},
        }],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def generate_launch_description():
    user_config_path = os.environ.get("LIVOX_CFG", DEFAULT_CFG)
    if not os.path.isfile(user_config_path):
        # 单独调试本 launch 时的兜底: 即时生成默认配置
        _make_config(user_config_path, _detect_host_ip())

    livox_params = [
        {"xfer_format": 0},        # 0 = PointCloud2 (PointXYZRTL) —— 标定必需
        {"multi_topic": 0},        # 单话题 /livox/lidar
        {"data_src": 0},           # 0 = 实时雷达
        {"publish_freq": 10.0},
        {"output_data_type": 0},
        {"frame_id": "livox_frame"},
        {"lvx_file_path": "/tmp/livox_calib.lvx"},   # 未使用, 占位
        {"user_config_path": user_config_path},
        {"cmdline_input_bd_code": "livox0000000001"},
    ]

    return LaunchDescription([
        LogInfo(msg=f"[dst_calib] livox user_config_path = {user_config_path}"),
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=livox_params,
        ),
    ])
