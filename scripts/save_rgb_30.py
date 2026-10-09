#!/usr/bin/env python3
"""Save 30 RGB frames from Isaac Sim, one per second.

Isaac Sim must already be playing, with the ROS 2 Camera graph publishing /rgb.

Run:
    source /opt/ros/humble/setup.bash
    python3 /home/dt/openvla/shenle_projects/scripts/save_rgb_30.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


OUT_DIR = Path('/home/dt/openvla/shenle_projects/scripts/rgb_capture')
FRAME_COUNT = 30
INTERVAL_SEC = 1.0


def image_to_bgr(msg: Image) -> np.ndarray:
    """Turn a sensor_msgs/Image into a BGR array for cv2.imwrite."""
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(encoding)
    if channels is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    row_bytes = int(msg.step) if msg.step else width * channels
    pixels = raw.reshape(height, row_bytes)[:, : width * channels].reshape(height, width, channels)
    if encoding == 'rgb8':
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == 'rgba8':
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if encoding == 'bgra8':
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    return pixels


class RgbSaver(Node):
    """Keep the newest /rgb frame and write one image each second."""

    def __init__(self):
        super().__init__('save_rgb_30')
        self.declare_parameter('image_topic', '/rgb')
        topic = str(self.get_parameter('image_topic').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, topic, self._on_image, qos)

        self.latest = None
        self.saved = 0
        self.last_save = None
        self.waiting_logs = 0
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        self.create_timer(0.05, self._maybe_save)
        self.get_logger().info(
            f'Saving {FRAME_COUNT} frames from {topic}, one every {INTERVAL_SEC:.0f}s, into {OUT_DIR}'
        )

    def _on_image(self, msg: Image):
        self.latest = msg

    def _maybe_save(self):
        if self.saved >= FRAME_COUNT:
            return
        now = time.monotonic()
        if self.latest is None:
            self.waiting_logs += 1
            if self.waiting_logs % 20 == 1:
                self.get_logger().info('Still waiting for /rgb. Is Isaac Sim playing?')
            return
        if self.last_save is not None and now - self.last_save < INTERVAL_SEC:
            return

        msg = self.latest
        frame = image_to_bgr(msg)
        path = OUT_DIR / f'{self.saved:03d}.png'
        if not cv2.imwrite(str(path), frame):
            self.get_logger().error(f'Failed to write {path}')
            return

        self.saved += 1
        self.last_save = now
        self.get_logger().info(
            f'{self.saved}/{FRAME_COUNT} {path.name}  {msg.width}x{msg.height} {msg.encoding}'
        )
        if self.saved >= FRAME_COUNT:
            self.get_logger().info(f'Done. {FRAME_COUNT} images in {OUT_DIR}')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RgbSaver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
