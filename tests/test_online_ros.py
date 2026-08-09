"""系统 Python 下的 ROS2 在线节点烟雾测试。

conda Python 与本机 ROS2 rclpy ABI 不同，因此常规 conda pytest 会自动 skip；
发布前另用系统 ``python3 -m pytest tests/test_online_ros.py`` 验证。
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from sensor_msgs_py import point_cloud2  # noqa: E402
from std_msgs.msg import Header  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dst_online_node_script", ROOT / "scripts" / "online_calib_node.py")
online_node = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = online_node
SPEC.loader.exec_module(online_node)


def _args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        mode="fast",
        config=str(ROOT / "config" / "default.yaml"),
        eva_ckpt=None,
        init_yaml=None,
        out=str(tmp_path),
        window_size=2,
        window_stride=1,
        sync_slop=0.05,
        publish_rate=10.0,
        worker_timeout=10.0,
        worker_python="/bin/true",
        camera_frame="camera_test",
        lidar_frame="lidar_test",
        save_windows=False,
    )


def _messages(stamp_sec=1):
    header = Header()
    header.stamp.sec = int(stamp_sec)
    header.frame_id = "lidar_test"
    points = [
        (1.0, -0.5, 1.0), (1.0, 0.0, 1.0), (1.0, 0.5, 1.0),
        (2.0, -0.5, 1.0), (2.0, 0.0, 1.0), (2.0, 0.5, 1.0),
    ]
    cloud = point_cloud2.create_cloud_xyz32(header, points)

    depth = np.full((8, 8), 1000, dtype=np.uint16)
    image = Image()
    image.header.stamp.sec = int(stamp_sec)
    image.height, image.width = depth.shape
    image.encoding = "16UC1"
    image.is_bigendian = False
    image.step = depth.shape[1] * depth.dtype.itemsize
    image.data = depth.tobytes()
    return cloud, image


def test_ros_node_window_to_filtered_yaml(tmp_path):
    with open(ROOT / "config" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["online"]["min_lidar_points"] = 3
    cfg["online"]["min_depth_valid_ratio"] = 0.01
    cfg["online"]["max_final_cd"] = 1.0

    rclpy.init(args=[])
    node = online_node.OnlineCalibNode(cfg, _args(tmp_path), tmp_path)
    try:
        info = CameraInfo()
        info.k = [100.0, 0.0, 4.0, 0.0, 100.0, 4.0, 0.0, 0.0, 1.0]
        node._on_info(info)

        def fake_worker(update_id, _frames, _prior):
            node._result_queue.put({
                "ok": True,
                "update_id": update_id,
                "result": {
                    "T_cam_lidar": np.eye(4).tolist(),
                    "T_lidar_cam": np.eye(4).tolist(),
                    "final_cd": 0.01,
                    "mode": "fast",
                    "eva_used": True,
                },
                "result_path": str(tmp_path / "candidate.yaml"),
                "window_dir": str(tmp_path / "missing_window"),
            })

        node._run_worker = fake_worker
        node._on_synced(*_messages(1))
        node._on_synced(*_messages(2))
        deadline = time.monotonic() + 2.0
        while node._result_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        node._poll_worker()

        latest = tmp_path / "latest_extrinsic.yaml"
        assert latest.is_file()
        result = yaml.safe_load(latest.read_text(encoding="utf-8"))
        assert result["online"]["accepted"] is True
        assert result["online"]["camera_frame"] == "camera_test"
        assert np.allclose(result["T_cam_lidar"], np.eye(4))
        assert node.gate.current is not None
    finally:
        node.shutdown_worker()
        node.destroy_node()
        rclpy.shutdown()
