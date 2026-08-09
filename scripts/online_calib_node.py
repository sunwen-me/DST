#!/usr/bin/env python3
"""DST-Calib ROS2 在线动态标定节点。

架构:

* 系统 Python 进程负责 ROS2 同步收流、滑窗、质量门控、平滑和 TF 发布；
* CUDA conda Python 作为异步工作进程处理窗口，优化期间主进程继续接收新帧；
* 每个窗口直接使用一条 Mid-360 ``PointCloud2``，不做静止场景专用的 3 秒裸累积；
* 窗口内假设外参固定，窗口间允许外参变化；大跳变须连续候选确认。

``T_cam_lidar`` 满足 ``p_camera = T_cam_lidar @ p_lidar``，因此发布的 TF
父坐标系为 camera、子坐标系为 lidar。
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from typing import Optional

import message_filters
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DST_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DST_ROOT) not in sys.path:
    sys.path.insert(0, str(DST_ROOT))

from capture_data import _decode_image, _depth_to_meters  # noqa: E402
from dst_calib.online import (  # noqa: E402
    OnlineExtrinsicGate,
    matrix_to_quaternion_xyzw,
    validate_se3,
)


@dataclass(frozen=True)
class SyncedFrame:
    points_lidar: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    stamp: float
    sync_delta_sec: float


def _stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _load_T(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    value = raw.get("T_cam_lidar", raw) if isinstance(raw, dict) else raw
    T = np.asarray(value, dtype=np.float64).reshape(4, 4)
    ok, why = validate_se3(T)
    if not ok:
        raise ValueError(f"初始外参 {path} 无效: {why}")
    return T


def _inverse_se3(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _euler_zyx_deg(R: np.ndarray) -> list[float]:
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return [float(v) for v in np.degrees([yaw, pitch, roll])]


class OnlineCalibNode(Node):
    def __init__(self, cfg: dict, args: argparse.Namespace, run_dir: Path):
        super().__init__("dst_calib_online")
        self.cfg = cfg
        self.args = args
        self.run_dir = run_dir
        self.windows_dir = run_dir / "windows"
        self.results_dir = run_dir / "results"
        self.windows_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        sensors = cfg["sensors"]
        lidar_cfg = sensors["lidar"]
        camera_cfg = sensors["camera"]
        online_cfg = cfg.get("online", {})

        self.lidar_topic = str(lidar_cfg["topic"])
        self.depth_topic = str(camera_cfg["depth_topic"])
        self.info_topic = str(camera_cfg["info_topic"])
        self.depth_unit = str(camera_cfg.get("depth_unit", "auto"))
        self.window_size = int(args.window_size)
        self.window_stride = int(args.window_stride)
        self.min_lidar_points = int(online_cfg.get("min_lidar_points", 1000))
        self.max_lidar_points = int(online_cfg.get("max_lidar_points", 120000))
        self.min_depth_valid_ratio = float(
            online_cfg.get("min_depth_valid_ratio", 0.05))
        self.camera_frame = str(args.camera_frame)
        self.lidar_frame = str(args.lidar_frame)
        self.worker_timeout_sec = float(args.worker_timeout)
        self.save_windows = bool(args.save_windows)

        self.frames: deque[SyncedFrame] = deque(maxlen=self.window_size)
        self.K: Optional[np.ndarray] = None
        self._accepted_frame_count = 0
        self._last_trigger_frame_count = 0
        self._update_id = 0
        self._force_trigger = False
        self._worker_busy = False
        self._worker_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._result_queue: queue.Queue = queue.Queue()
        self._initial_prior = _load_T(args.init_yaml)
        self._last_source_stamp = 0.0

        self.gate = OnlineExtrinsicGate(
            max_final_cd=float(online_cfg.get("max_final_cd", 0.15)),
            max_rotation_jump_deg=float(
                online_cfg.get("max_rotation_jump_deg", 5.0)),
            max_translation_jump_m=float(
                online_cfg.get("max_translation_jump_m", 0.30)),
            smoothing_alpha=float(online_cfg.get("smoothing_alpha", 0.35)),
            reanchor_confirmations=int(
                online_cfg.get("reanchor_confirmations", 2)),
            confirmation_rotation_deg=float(
                online_cfg.get("confirmation_rotation_deg", 1.0)),
            confirmation_translation_m=float(
                online_cfg.get("confirmation_translation_m", 0.05)),
        )

        qos = qos_profile_sensor_data
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos)
        self._cloud_sub = message_filters.Subscriber(
            self, PointCloud2, self.lidar_topic, qos_profile=qos)
        self._depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._cloud_sub, self._depth_sub],
            queue_size=int(online_cfg.get("sync_queue_size", 40)),
            slop=float(args.sync_slop),
        )
        self._sync.registerCallback(self._on_synced)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.extrinsic_pub = self.create_publisher(
            TransformStamped, str(online_cfg.get(
                "extrinsic_topic", "/dst_calib/extrinsic")), 10)
        self.status_pub = self.create_publisher(
            String, str(online_cfg.get(
                "status_topic", "/dst_calib/status")), 10)
        self.create_service(Trigger, "~/trigger", self._on_trigger)
        self.create_service(Trigger, "~/reset_filter", self._on_reset_filter)
        self.create_timer(0.1, self._poll_worker)
        self.create_timer(
            1.0 / max(float(args.publish_rate), 0.1), self._publish_current)

        self.get_logger().info(
            "在线动态标定已启动: "
            f"mode={args.mode}, window={self.window_size}, "
            f"stride={self.window_stride}, sync_slop={args.sync_slop:.3f}s")
        self.get_logger().info(
            f"同步话题: {self.lidar_topic} + {self.depth_topic}; "
            f"CameraInfo: {self.info_topic}")
        self.get_logger().info(
            f"TF 语义: {self.camera_frame} <- {self.lidar_frame}; "
            f"结果目录: {self.run_dir}")
        if self._initial_prior is not None:
            self.get_logger().info(
                f"首窗口使用先验外参: {args.init_yaml}（验证通过前不发布）")
        self._publish_status("starting", "等待 CameraInfo 与同步帧")

    def _publish_status(self, state: str, message: str, **extra) -> None:
        payload = {
            "state": state,
            "message": message,
            "mode": self.args.mode,
            "window_frames": len(self.frames),
            "window_size": self.window_size,
            "worker_busy": self._worker_busy,
            "accepted_frames": self._accepted_frame_count,
            "update_id": self._update_id,
            **extra,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.status_pub.publish(msg)

    def _on_info(self, msg: CameraInfo) -> None:
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] > 0.0 and K[1, 1] > 0.0:
            self.K = K

    def _on_synced(self, cloud_msg: PointCloud2, depth_msg: Image) -> None:
        if self.K is None:
            self.get_logger().warning(
                "已收到点云/深度同步对，但 CameraInfo 尚未就绪", throttle_duration_sec=5.0)
            return
        try:
            raw = point_cloud2.read_points(
                cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            if len(raw) == 0:
                return
            xyz = np.stack([
                np.asarray(raw["x"], dtype=np.float32),
                np.asarray(raw["y"], dtype=np.float32),
                np.asarray(raw["z"], dtype=np.float32),
            ], axis=-1)
            ranges = np.linalg.norm(xyz, axis=1)
            lidar_cfg = self.cfg["sensors"]["lidar"]
            keep = (
                np.isfinite(ranges)
                & (ranges > float(lidar_cfg.get("min_range", 0.1)))
                & (ranges < float(lidar_cfg.get("max_range", 40.0)))
            )
            xyz = xyz[keep]
            if xyz.shape[0] < self.min_lidar_points:
                self.get_logger().warning(
                    f"同步帧 LiDAR 点数 {xyz.shape[0]} < {self.min_lidar_points}，跳过",
                    throttle_duration_sec=3.0)
                return
            if self.max_lidar_points > 0 and xyz.shape[0] > self.max_lidar_points:
                step = int(np.ceil(xyz.shape[0] / self.max_lidar_points))
                xyz = xyz[::step][:self.max_lidar_points]

            depth_raw, encoding = _decode_image(depth_msg)
            depth = _depth_to_meters(depth_raw, encoding, self.depth_unit)
            valid_ratio = float((depth > 0.0).mean())
            if valid_ratio < self.min_depth_valid_ratio:
                self.get_logger().warning(
                    f"同步帧有效深度比例 {100.0 * valid_ratio:.1f}% < "
                    f"{100.0 * self.min_depth_valid_ratio:.1f}%，跳过",
                    throttle_duration_sec=3.0)
                return

            cloud_stamp = _stamp_sec(cloud_msg.header.stamp)
            depth_stamp = _stamp_sec(depth_msg.header.stamp)
            delta = abs(cloud_stamp - depth_stamp)
            frame = SyncedFrame(
                points_lidar=np.ascontiguousarray(xyz, dtype=np.float32),
                depth=np.ascontiguousarray(depth, dtype=np.float32),
                K=self.K.copy(),
                stamp=0.5 * (cloud_stamp + depth_stamp),
                sync_delta_sec=delta,
            )
            self.frames.append(frame)
            self._accepted_frame_count += 1
            self._last_source_stamp = frame.stamp
            if self._accepted_frame_count % max(self.window_stride, 1) == 0:
                self.get_logger().info(
                    f"在线窗口 {len(self.frames)}/{self.window_size}: "
                    f"点数={xyz.shape[0]}, 深度有效={100.0 * valid_ratio:.1f}%, "
                    f"|Δt|={1000.0 * delta:.1f}ms")
            self._maybe_start_worker()
        except Exception as exc:
            self.get_logger().error(f"同步帧解码失败: {exc!r}")

    def _maybe_start_worker(self) -> None:
        if len(self.frames) < self.window_size or self._worker_busy:
            return
        due = (
            self._accepted_frame_count - self._last_trigger_frame_count
            >= self.window_stride
        )
        if not due and not self._force_trigger:
            return
        self._force_trigger = False
        self._last_trigger_frame_count = self._accepted_frame_count
        self._update_id += 1
        update_id = self._update_id
        snapshot = list(self.frames)
        prior = self.gate.current
        if prior is None:
            prior = None if self._initial_prior is None else self._initial_prior.copy()
        self._worker_busy = True
        self._publish_status(
            "calibrating", f"提交滑窗 {update_id} 到 CUDA 工作进程")
        thread = threading.Thread(
            target=self._run_worker,
            args=(update_id, snapshot, prior),
            name=f"dst-online-worker-{update_id}",
            daemon=True,
        )
        thread.start()

    def _run_worker(
        self,
        update_id: int,
        frames: list[SyncedFrame],
        prior: Optional[np.ndarray],
    ) -> None:
        window_dir = self.windows_dir / f"window_{update_id:06d}"
        result_dir = self.results_dir / f"update_{update_id:06d}"
        try:
            window_dir.mkdir(parents=True, exist_ok=False)
            result_dir.mkdir(parents=True, exist_ok=False)
            for i, frame in enumerate(frames):
                np.savez(
                    window_dir / f"frame_{i:03d}.npz",
                    points_lidar=frame.points_lidar,
                    depth=frame.depth,
                    K=frame.K,
                    stamp=np.float64(frame.stamp),
                )
            prior_path = None
            if prior is not None:
                prior_path = window_dir / "prior.yaml"
                with open(prior_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump({
                        "T_cam_lidar": prior.tolist(),
                    }, f, sort_keys=False)

            cmd = [
                self.args.worker_python,
                "-m", "dst_calib.calibrate",
                "--data_dir", str(window_dir),
                "--config", str(Path(self.args.config).resolve()),
                "--output", str(result_dir),
                "--mode", self.args.mode,
                "--no_artifacts",
            ]
            if self.args.eva_ckpt:
                cmd += ["--eva_ckpt", str(Path(self.args.eva_ckpt).resolve())]
            if prior_path is not None:
                cmd += ["--init_yaml", str(prior_path)]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(DST_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            log_path = result_dir / "worker.log"
            with open(log_path, "w", encoding="utf-8") as log:
                with self._process_lock:
                    self._worker_process = subprocess.Popen(
                        cmd, cwd=str(DST_ROOT), env=env,
                        stdout=log, stderr=subprocess.STDOUT, text=True)
                try:
                    returncode = self._worker_process.wait(
                        timeout=self.worker_timeout_sec)
                except subprocess.TimeoutExpired:
                    self._worker_process.terminate()
                    try:
                        self._worker_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._worker_process.kill()
                    raise RuntimeError(
                        f"工作进程超过 {self.worker_timeout_sec:.0f}s")
                finally:
                    with self._process_lock:
                        self._worker_process = None
            if returncode != 0:
                tail = ""
                try:
                    tail = "\n".join(
                        log_path.read_text(encoding="utf-8").splitlines()[-12:])
                except OSError:
                    pass
                raise RuntimeError(
                    f"标定工作进程退出码 {returncode}\n{tail}")
            result_path = result_dir / "extrinsic.yaml"
            if not result_path.is_file():
                raise RuntimeError("工作进程成功退出但未生成 extrinsic.yaml")
            with open(result_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
            self._result_queue.put({
                "ok": True,
                "update_id": update_id,
                "result": result,
                "result_path": str(result_path),
                "window_dir": str(window_dir),
            })
        except Exception as exc:
            self._result_queue.put({
                "ok": False,
                "update_id": update_id,
                "error": str(exc),
                "window_dir": str(window_dir),
            })

    def _write_latest(self, result: dict, decision, update_id: int) -> None:
        T = np.asarray(decision.transform, dtype=np.float64).reshape(4, 4)
        raw_T = np.asarray(result["T_cam_lidar"], dtype=np.float64).reshape(4, 4)
        out = dict(result)
        out["candidate_T_cam_lidar"] = raw_T.tolist()
        out["candidate_final_cd"] = float(result["final_cd"])
        out["T_cam_lidar"] = T.tolist()
        out["T_lidar_cam"] = _inverse_se3(T).tolist()
        out["euler_zyx_deg"] = _euler_zyx_deg(T[:3, :3])
        out["translation_m"] = [float(v) for v in T[:3, 3]]
        out["online"] = {
            "update_id": int(update_id),
            "source_stamp": float(self._last_source_stamp),
            "accepted": True,
            "reason": decision.reason,
            "reanchored": bool(decision.reanchored),
            "rotation_jump_deg": float(decision.rotation_jump_deg),
            "translation_jump_m": float(decision.translation_jump_m),
            "window_size": int(self.window_size),
            "window_stride": int(self.window_stride),
            "camera_frame": self.camera_frame,
            "lidar_frame": self.lidar_frame,
            "filtered": True,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
        }
        latest = self.run_dir / "latest_extrinsic.yaml"
        temporary = self.run_dir / ".latest_extrinsic.yaml.tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
        os.replace(temporary, latest)

    def _poll_worker(self) -> None:
        try:
            item = self._result_queue.get_nowait()
        except queue.Empty:
            return
        self._worker_busy = False
        window_dir = Path(item["window_dir"])
        if not self.save_windows and window_dir.is_dir():
            shutil.rmtree(window_dir)
        if not item["ok"]:
            message = f"滑窗 {item['update_id']} 标定失败: {item['error']}"
            self.get_logger().error(message)
            self._publish_status("worker_error", message)
            self._maybe_start_worker()
            return

        result = item["result"]
        try:
            candidate = np.asarray(
                result["T_cam_lidar"], dtype=np.float64).reshape(4, 4)
            final_cd = float(result["final_cd"])
            decision = self.gate.update(candidate, final_cd)
        except Exception as exc:
            self.get_logger().error(f"滑窗结果解析失败: {exc!r}")
            self._publish_status(
                "result_error", f"滑窗结果解析失败: {exc!r}")
            self._maybe_start_worker()
            return

        if decision.accepted:
            self._write_latest(result, decision, item["update_id"])
            self.get_logger().info(
                f"外参更新 {item['update_id']} 已接受: {decision.reason}; "
                f"CD={decision.final_cd:.5f}, "
                f"jump={decision.rotation_jump_deg:.2f}°/"
                f"{decision.translation_jump_m:.3f}m")
            self._publish_status(
                "tracking",
                decision.reason,
                final_cd=decision.final_cd,
                rotation_jump_deg=decision.rotation_jump_deg,
                translation_jump_m=decision.translation_jump_m,
                reanchored=decision.reanchored,
                result_path=item["result_path"],
            )
            self._publish_current()
        else:
            self.get_logger().warning(
                f"外参候选 {item['update_id']} 被拒绝: {decision.reason}")
            self._publish_status(
                "candidate_rejected",
                decision.reason,
                final_cd=decision.final_cd,
                rotation_jump_deg=decision.rotation_jump_deg,
                translation_jump_m=decision.translation_jump_m,
                pending_confirmations=self.gate.pending_count,
            )
        self._maybe_start_worker()

    def _make_transform(self, T: np.ndarray) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.child_frame_id = self.lidar_frame
        msg.transform.translation.x = float(T[0, 3])
        msg.transform.translation.y = float(T[1, 3])
        msg.transform.translation.z = float(T[2, 3])
        q = matrix_to_quaternion_xyzw(T[:3, :3])
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        return msg

    def _publish_current(self) -> None:
        T = self.gate.current
        if T is None:
            return
        msg = self._make_transform(T)
        self.tf_broadcaster.sendTransform(msg)
        self.extrinsic_pub.publish(msg)

    def _on_trigger(self, _request, response):
        if len(self.frames) < self.window_size:
            response.success = False
            response.message = (
                f"窗口未满: {len(self.frames)}/{self.window_size}")
            return response
        was_busy = self._worker_busy
        self._force_trigger = True
        self._maybe_start_worker()
        response.success = True
        response.message = (
            "工作进程忙，已排队使用最新窗口"
            if was_busy else "已提交最新窗口")
        return response

    def _on_reset_filter(self, _request, response):
        self.gate.reset()
        response.success = True
        response.message = "已清空已发布外参与突变确认状态，等待下一个有效窗口"
        self._publish_status("filter_reset", response.message)
        return response

    def shutdown_worker(self) -> None:
        with self._process_lock:
            proc = self._worker_process
        if proc is not None and proc.poll() is None:
            self.get_logger().warning("终止仍在运行的标定工作进程")
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DST-Calib ROS2 在线动态标定")
    parser.add_argument(
        "--config", default=str(DST_ROOT / "config" / "default.yaml"))
    parser.add_argument("--mode", choices=["fast", "both", "selfsup", "auto"],
                        default=None,
                        help="在线推荐 fast；both/selfsup 会显著降低更新频率")
    parser.add_argument("--eva-ckpt", default=None)
    parser.add_argument("--init-yaml", default=None,
                        help="首窗口的粗外参先验；验证通过前不会发布")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--sync-slop", type=float, default=None)
    parser.add_argument("--publish-rate", type=float, default=None)
    parser.add_argument("--worker-timeout", type=float, default=None)
    parser.add_argument("--worker-python", default=None)
    parser.add_argument("--camera-frame", default=None)
    parser.add_argument("--lidar-frame", default=None)
    parser.add_argument("--save-windows", action="store_true")
    return parser


def _resolve_args(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    online = cfg.get("online", {})
    args.mode = args.mode or str(online.get("mode", "fast"))
    args.eva_ckpt = args.eva_ckpt or cfg.get(
        "inference", {}).get("eva_ckpt")
    if args.eva_ckpt and not os.path.isabs(args.eva_ckpt):
        args.eva_ckpt = str(DST_ROOT / args.eva_ckpt)
    args.window_size = args.window_size or int(online.get("window_size", 12))
    args.window_stride = args.window_stride or int(
        online.get("window_stride", 4))
    args.sync_slop = (
        args.sync_slop if args.sync_slop is not None
        else float(online.get("sync_slop_sec", 0.05))
    )
    args.publish_rate = (
        args.publish_rate if args.publish_rate is not None
        else float(online.get("publish_rate_hz", 10.0))
    )
    args.worker_timeout = (
        args.worker_timeout if args.worker_timeout is not None
        else float(online.get("worker_timeout_sec", 900.0))
    )
    args.worker_python = (
        args.worker_python
        or os.environ.get("DST_CALIB_PYTHON")
        or online.get("worker_python")
        or sys.executable
    )
    if not os.path.isabs(args.worker_python):
        args.worker_python = shutil.which(args.worker_python) or args.worker_python
    args.camera_frame = args.camera_frame or str(
        online.get("camera_frame", "camera_color_optical_frame"))
    args.lidar_frame = args.lidar_frame or str(
        online.get("lidar_frame", "livox_frame"))
    if args.window_size < 2:
        raise ValueError("--window-size 必须 >=2")
    if args.window_stride < 1:
        raise ValueError("--window-stride 必须 >=1")
    if not os.path.isfile(args.worker_python):
        raise FileNotFoundError(f"CUDA 工作解释器不存在: {args.worker_python}")
    if args.mode in ("fast", "both") and (
        not args.eva_ckpt or not os.path.isfile(args.eva_ckpt)
    ):
        raise FileNotFoundError(
            f"在线 mode={args.mode} 需要评估模块断点: {args.eva_ckpt}；"
            "请先训练 runs/train_eva/best.pt，或临时使用 --mode selfsup")
    return args


def main(argv=None) -> int:
    parser = _build_parser()
    args, ros_args = parser.parse_known_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = _resolve_args(args, cfg)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out or DST_ROOT / "runs" / f"online_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=ros_args)
    node = OnlineCalibNode(cfg, args, run_dir)
    code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f"在线节点异常退出: {exc!r}")
        code = 1
    finally:
        node.shutdown_worker()
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
