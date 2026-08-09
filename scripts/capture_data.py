#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DST-Calib 数据采集节点 (论文 arXiv 2601.01188 复现, 数据准备阶段)。

rclpy 节点: 订阅 Livox Mid-360 PointCloud2 (livox_ros_driver2, xfer_format=0)
与 Orbbec Gemini 335 的彩色 / 对齐深度 / CameraInfo。每帧把 accumulate_sec
时间窗内累积的全部 LiDAR 点 (Mid-360 非重复扫描, 累积提密度) 与最近一组
近似同步的彩色-深度对保存为 frame_%03d.npz, 键与 INTERFACES.md 的 Frame
数据类一一对应:

    points_lidar : (N,3) float32, LiDAR 系
    depth        : (H,W) float32, 米, 0=无效, 已对齐彩色
    K            : (3,3) 彩色内参 (CameraInfo.k)
    rgb          : (H,W,3) uint8, RGB 顺序
    stamp        : float, 彩色帧时间戳(秒)

注意: 本脚本用系统 python3 (rclpy+numpy+yaml) 运行, 绝不 import torch /
dst_calib —— 标定计算由 conda 环境的 `python -m dst_calib.calibrate` 完成,
两套解释器严格分离 (见 INTERFACES.md scripts 节)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
import message_filters


# ------------------------------------------------------------------ 工具函数

def _stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time → 秒 (float)。"""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# encoding → (numpy dtype, 通道数)
_ENC_TABLE = {
    "16uc1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
    "32fc1": (np.float32, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgba8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "8uc1": (np.uint8, 1),
}


def _decode_image(msg: Image) -> tuple[np.ndarray, str]:
    """sensor_msgs/Image → numpy 数组 (不依赖 cv_bridge)。

    返回 (数组, 小写 encoding)。按 msg.step 处理行填充, 按 is_bigendian 处理
    字节序 (x86 主机上 Orbbec/livox 均为小端, 此处仅作稳健兜底)。
    """
    enc = msg.encoding.lower()
    if enc not in _ENC_TABLE:
        raise ValueError(f"不支持的图像编码: {msg.encoding}")
    dtype, ch = _ENC_TABLE[enc]
    bpp = np.dtype(dtype).itemsize * ch  # 每像素字节数
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    need = msg.height * msg.step
    if buf.size < need:
        raise ValueError(f"图像数据长度不足: {buf.size} < {need} (encoding={enc})")
    # 去掉行尾填充后拷贝为连续内存, 再按 dtype 重解释
    img = buf[:need].reshape(msg.height, msg.step)[:, : msg.width * bpp].copy()
    img = img.view(dtype)
    if msg.is_bigendian and np.dtype(dtype).itemsize > 1:
        img = img.byteswap()
    if ch > 1:
        img = img.reshape(msg.height, msg.width, ch)
    else:
        img = img.reshape(msg.height, msg.width)
    return img, enc


def _depth_to_meters(img: np.ndarray, enc: str, unit: str) -> np.ndarray:
    """深度图 → float32 米。16UC1 视为毫米(/1000), 32FC1 视为米;
    unit='auto' 时按 encoding/dtype 自动判别, 'mm'/'m' 强制覆盖。
    非有限值与负值置 0 (契约: 0 = 无效像素)。"""
    if unit == "mm":
        d = img.astype(np.float32) / 1000.0
    elif unit == "m":
        d = img.astype(np.float32)
    else:  # auto
        if img.dtype == np.uint16:
            d = img.astype(np.float32) / 1000.0   # 16UC1: 毫米
        elif img.dtype == np.float32:
            d = img.astype(np.float32)            # 32FC1: 米
        else:
            raise ValueError(f"无法自动判别深度单位 (encoding={enc}, dtype={img.dtype}); "
                             f"请用 --depth_unit mm|m 指定")
    d = np.ascontiguousarray(d)
    bad = ~np.isfinite(d)
    if bad.any():
        d[bad] = 0.0
    d[d < 0] = 0.0
    return d


def _to_rgb(img: np.ndarray, enc: str) -> np.ndarray:
    """彩色图 → (H,W,3) uint8 RGB。bgr8 → 通道反转; 灰度 → 三通道复制。"""
    if img.ndim == 2:
        rgb = np.repeat(img[:, :, None], 3, axis=2)
    elif enc == "rgb8":
        rgb = img
    elif enc == "bgr8":
        rgb = img[:, :, ::-1]
    elif enc == "rgba8":
        rgb = img[:, :, :3]
    elif enc == "bgra8":
        rgb = img[:, :, [2, 1, 0]]
    else:
        rgb = img[:, :, :3]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


# ------------------------------------------------------------------ 采集节点

class CaptureNode(Node):
    """订阅 LiDAR/彩色/深度/内参, 按帧累积并保存 npz。

    QoS 统一用 sensor data (best_effort) —— best_effort 订阅端可兼容
    reliable 发布端, 反之不成立, 故对 livox 与 orbbec 均安全。
    彩色/深度用 message_filters.ApproximateTimeSynchronizer 近似同步。
    """

    def __init__(self, cfg: dict, args: argparse.Namespace):
        super().__init__("dst_calib_capture")
        sensors = cfg.get("sensors", {})
        lid = sensors.get("lidar", {})
        cam = sensors.get("camera", {})
        self.lidar_topic = str(lid.get("topic", "/livox/lidar"))
        self.color_topic = str(cam.get("color_topic", "/camera/color/image_raw"))
        self.depth_topic = str(cam.get("depth_topic", "/camera/depth/image_raw"))
        self.info_topic = str(cam.get("info_topic", "/camera/color/camera_info"))
        self.min_range = float(lid.get("min_range", 0.1))
        self.max_range = float(lid.get("max_range", 40.0))
        self.args = args

        # 状态
        self.collecting = False                # 是否处于累积窗口内
        self.cloud_buf: list[np.ndarray] = []  # 累积的 (Ni,3) float32
        self.lidar_msgs = 0
        self.color_msgs = 0
        self.depth_msgs = 0
        self.K: np.ndarray | None = None       # (3,3)
        self.synced = None                     # (color_msg, depth_msg, 单调时刻)

        qos = qos_profile_sensor_data
        self.create_subscription(PointCloud2, self.lidar_topic, self._on_cloud, qos)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos)
        self._sub_color = message_filters.Subscriber(self, Image, self.color_topic,
                                                     qos_profile=qos)
        self._sub_depth = message_filters.Subscriber(self, Image, self.depth_topic,
                                                     qos_profile=qos)
        # 额外注册裸回调, 仅用于就绪诊断计数
        self._sub_color.registerCallback(self._on_color_raw)
        self._sub_depth.registerCallback(self._on_depth_raw)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_color, self._sub_depth], queue_size=30, slop=0.08)
        self._sync.registerCallback(self._on_synced)

    # ---------------------------------------------------------------- 回调

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.lidar_msgs += 1
        if not self.collecting:
            return
        arr = point_cloud2.read_points(msg, field_names=("x", "y", "z"),
                                       skip_nans=True)
        if len(arr) == 0:
            return
        xyz = np.stack([np.asarray(arr["x"], dtype=np.float32),
                        np.asarray(arr["y"], dtype=np.float32),
                        np.asarray(arr["z"], dtype=np.float32)], axis=-1)
        # 距离滤波: 去掉 (0,0,0) 无效回波与超量程点
        r = np.linalg.norm(xyz, axis=1)
        m = np.isfinite(r) & (r > self.min_range) & (r < self.max_range)
        if m.any():
            self.cloud_buf.append(xyz[m])

    def _on_info(self, msg: CameraInfo) -> None:
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] > 0:
            self.K = K
        elif self.K is None:
            self.get_logger().warning("收到 CameraInfo 但 fx=0, 忽略 (驱动尚未就绪?)")

    def _on_color_raw(self, _msg: Image) -> None:
        self.color_msgs += 1

    def _on_depth_raw(self, _msg: Image) -> None:
        self.depth_msgs += 1

    def _on_synced(self, color_msg: Image, depth_msg: Image) -> None:
        self.synced = (color_msg, depth_msg, time.monotonic())

    # ---------------------------------------------------------------- 流程

    def wait_ready(self, timeout: float = 30.0) -> None:
        """等所有话题都收到至少一条消息 (含彩深同步对), 超时报中文错误。"""
        self.get_logger().info(
            f"等待话题就绪 (超时 {timeout:.0f}s): {self.lidar_topic}, "
            f"{self.color_topic}, {self.depth_topic}, {self.info_topic}")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.lidar_msgs > 0 and self.K is not None and self.synced is not None:
                self.get_logger().info("所有话题就绪, 开始采集。")
                return
        missing = []
        if self.lidar_msgs == 0:
            missing.append(f"LiDAR 点云 {self.lidar_topic} —— 检查 Mid-360 网线/主机 IP "
                           f"(须 192.168.1.x)/livox_ros_driver2 是否启动且 xfer_format=0")
        if self.color_msgs == 0:
            missing.append(f"彩色图 {self.color_topic} —— 检查 Gemini 335 USB3 连接与 "
                           f"orbbec_camera 驱动")
        if self.depth_msgs == 0:
            missing.append(f"深度图 {self.depth_topic} —— 检查驱动参数 depth_registration:=true")
        if self.K is None:
            missing.append(f"相机内参 {self.info_topic} —— CameraInfo 未收到或 fx=0")
        if not missing and self.synced is None:
            missing.append("彩色/深度时间同步失败 —— 两话题都有数据但时间戳差距过大, "
                           "请给 orbbec 驱动加 enable_frame_sync:=true")
        raise RuntimeError("等待话题超时(%.0fs), 缺失/异常:\n  - %s"
                           % (timeout, "\n  - ".join(missing)))

    def capture_all(self) -> None:
        """采集 num_frames 帧: 每帧累积 accumulate_sec, 帧间隔 interval_sec。"""
        out = Path(self.args.out)
        out.mkdir(parents=True, exist_ok=True)
        n = int(self.args.num_frames)
        for i in range(n):
            # 1) 累积窗口: 收集窗口内所有 LiDAR 消息的 xyz
            self.cloud_buf = []
            self.collecting = True
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.args.accumulate_sec:
                rclpy.spin_once(self, timeout_sec=0.05)
            self.collecting = False
            if not self.cloud_buf:
                raise RuntimeError(
                    f"帧 {i}: 累积窗口 {self.args.accumulate_sec}s 内未收到任何 LiDAR 点 "
                    f"—— 驱动可能中途断开, 检查 {self.lidar_topic}")
            pts = np.concatenate(self.cloud_buf, axis=0).astype(np.float32)

            # 2) 取最近一组同步彩深对
            if self.synced is None:
                raise RuntimeError(f"帧 {i}: 无同步的彩色/深度对")
            color_msg, depth_msg, sync_t = self.synced
            if time.monotonic() - sync_t > 2.0 * self.args.accumulate_sec + 2.0:
                self.get_logger().warning(f"帧 {i}: 彩深同步对已过时 "
                                          f"{time.monotonic() - sync_t:.1f}s, 相机可能掉流")
            depth_img, denc = _decode_image(depth_msg)
            depth = _depth_to_meters(depth_img, denc, self.args.depth_unit)
            color_img, cenc = _decode_image(color_msg)
            rgb = _to_rgb(color_img, cenc)
            if depth.shape[:2] != rgb.shape[:2]:
                # 硬失败: 整条流水线契约要求深度已对齐彩色且 K 为彩色内参;
                # 继续保存会让 calibrate 用彩色 K 反投影未对齐深度 (结果无效)
                # 并使 visualize_result.py 崩溃
                raise RuntimeError(
                    f"帧 {i}: 深度分辨率 {depth.shape[:2]} 与彩色 {rgb.shape[:2]} "
                    f"不一致 —— 深度未对齐彩色, 请确认 orbbec 驱动加了 "
                    f"depth_registration:=true 后重新采集")

            # 3) 保存 (键与 INTERFACES.md Frame 契约一致)
            stamp = _stamp_to_sec(color_msg.header.stamp)
            path = out / f"frame_{i:03d}.npz"
            np.savez_compressed(path,
                                points_lidar=pts,
                                depth=depth.astype(np.float32),
                                K=self.K.astype(np.float64),
                                rgb=rgb,
                                stamp=np.float64(stamp))
            valid_pct = float((depth > 0).mean()) * 100.0
            self.get_logger().info(
                f"帧 {i + 1}/{n}: LiDAR 点数={pts.shape[0]}, "
                f"深度有效像素={valid_pct:.1f}% ({denc}), 彩色={cenc} → {path.name}")

            # 4) 帧间隔 (继续 spin 保持订阅活跃)
            if i + 1 < n:
                t1 = time.monotonic()
                while time.monotonic() - t1 < self.args.interval_sec:
                    rclpy.spin_once(self, timeout_sec=0.05)


# ------------------------------------------------------------------ 入口

def _parse_args(argv=None) -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    ap = argparse.ArgumentParser(
        description="DST-Calib 数据采集 (Livox Mid-360 + Orbbec Gemini 335)")
    ap.add_argument("--out", required=True, help="输出目录, 保存 frame_%%03d.npz")
    ap.add_argument("--num_frames", type=int, default=None,
                    help="采集帧数 (默认取 config capture.num_frames=12)")
    ap.add_argument("--accumulate_sec", type=float, default=None,
                    help="每帧 LiDAR 累积时长秒 (默认 config=3.0)")
    ap.add_argument("--interval_sec", type=float, default=None,
                    help="帧间隔秒 (默认 config=1.0)")
    ap.add_argument("--depth_unit", choices=["auto", "mm", "m"], default=None,
                    help="深度单位: auto 按 encoding 判别 (16UC1→mm, 32FC1→m)")
    ap.add_argument("--config", default=str(default_cfg),
                    help="config/default.yaml 路径 (读话题名/量程/采集默认值)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cap = cfg.get("capture", {})
    cam = cfg.get("sensors", {}).get("camera", {})
    if args.num_frames is None:
        args.num_frames = int(cap.get("num_frames", 12))
    if args.accumulate_sec is None:
        args.accumulate_sec = float(cap.get("accumulate_sec", 3.0))
    if args.interval_sec is None:
        args.interval_sec = float(cap.get("interval_sec", 1.0))
    if args.depth_unit is None:
        args.depth_unit = str(cam.get("depth_unit", "auto"))

    rclpy.init()
    node = CaptureNode(cfg, args)
    code = 0
    try:
        node.wait_ready(30.0)
        node.capture_all()
        node.get_logger().info(f"采集完成: {args.num_frames} 帧 → {args.out}")
    except KeyboardInterrupt:
        print("[中断] 用户取消采集", file=sys.stderr)
        code = 130
    except (RuntimeError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
