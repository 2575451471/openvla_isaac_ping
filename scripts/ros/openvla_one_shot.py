#!/usr/bin/env python3
"""Send one /rgb frame to the local OpenVLA server and print the 7-vector.

Isaac Sim must be playing and publishing /rgb.
OpenVLA must already be listening on http://127.0.0.1:8000/act.

This process does not publish /joint_command. The 7 numbers are end-effector
deltas, not joint angles.

Run:
    source /opt/ros/humble/setup.bash
    python3 /home/dt/openvla/shenle_projects/scripts/ros/openvla_one_shot.py
"""

import json
from base64 import b64decode, b64encode

import numpy as np
import requests
import rclpy
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


LABELS = ('dx_m', 'dy_m', 'dz_m', 'droll_rad', 'dpitch_rad', 'dyaw_rad', 'gripper')


def dumps_numpy(payload: dict) -> str:
    """Encode numpy arrays the same way json_numpy does, without importing it."""

    def default(obj):
        if isinstance(obj, np.ndarray):
            array = np.ascontiguousarray(obj)
            return {
                '__numpy__': b64encode(array.tobytes()).decode('ascii'),
                'dtype': dtype_to_descr(array.dtype),
                'shape': array.shape,
            }
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    return json.dumps(payload, default=default)


def loads_numpy(text: str):
    def hook(dct):
        if '__numpy__' not in dct:
            return dct
        values = np.frombuffer(b64decode(dct['__numpy__']), descr_to_dtype(dct['dtype']))
        shape = dct.get('shape')
        return values.reshape(shape) if shape else values[0]

    return json.loads(text, object_hook=hook)


def image_to_rgb(msg: Image) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(encoding)
    if channels is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    row_bytes = int(msg.step) if msg.step else width * channels
    pixels = raw.reshape(height, row_bytes)[:, : width * channels].reshape(height, width, channels)
    if encoding == 'bgr8':
        pixels = pixels[:, :, ::-1]
    elif encoding == 'rgba8':
        pixels = pixels[:, :, :3]
    elif encoding == 'bgra8':
        pixels = pixels[:, :, 2::-1]
    return np.ascontiguousarray(pixels, dtype=np.uint8)


class OneShot(Node):
    """Wait for one camera frame, POST it, print the action, then exit."""

    def __init__(self):
        super().__init__('openvla_one_shot')
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000/act')
        self.declare_parameter('instruction', 'move the red cup to the left')
        self.declare_parameter('unnorm_key', 'bridge_orig')

        self.topic = str(self.get_parameter('image_topic').value)
        self.server_url = str(self.get_parameter('server_url').value)
        self.instruction = str(self.get_parameter('instruction').value)
        self.unnorm_key = str(self.get_parameter('unnorm_key').value)
        self.sent = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.topic, self._on_image, qos)
        self.get_logger().info(
            f'Waiting for one {self.topic} frame, then POST {self.server_url}'
        )

    def _on_image(self, msg: Image):
        if self.sent:
            return
        self.sent = True
        image = image_to_rgb(msg)
        self.get_logger().info(
            f'Got {image.shape[1]}x{image.shape[0]} {msg.encoding}, instruction="{self.instruction}"'
        )
        body = dumps_numpy({
            'image': image,
            'instruction': self.instruction,
            'unnorm_key': self.unnorm_key,
        })
        try:
            response = requests.post(
                self.server_url,
                data=body,
                headers={'Content-Type': 'application/json'},
                timeout=120,
            )
        except requests.RequestException as exc:
            self.get_logger().error(f'OpenVLA request failed: {exc}')
            rclpy.shutdown()
            return

        if response.status_code != 200:
            self.get_logger().error(f'HTTP {response.status_code}: {response.text[:500]}')
            rclpy.shutdown()
            return

        action = np.asarray(loads_numpy(response.text), dtype=np.float64).reshape(-1)
        if action.size != 7:
            self.get_logger().error(f'Expected 7 values, got {action.tolist()}')
        else:
            for label, value in zip(LABELS, action):
                self.get_logger().info(f'{label}: {value:.6f}')
            self.get_logger().info(
                'Link ok. These are end-effector deltas. Gripper is the last value and is unused.'
            )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OneShot()
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
