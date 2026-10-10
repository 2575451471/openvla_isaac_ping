#!/usr/bin/env python3
"""Hold a fixed joint pose (no motion).

Isaac Sim must already be playing, with the ROS 2 Joint States graph
publishing /joint_states and subscribing /joint_command.

Home values are from the Isaac joint panel (degrees). ROS commands are radians.

Run:
    source /opt/ros/humble/setup.bash
    python3 /home/dt/openvla/shenle_projects/scripts/legacy/ur12e_random_motion.py
"""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


# Official UR arm joints. A name matches if it is exactly this, or ends with
# "/shoulder_pan_joint" / "_shoulder_pan_joint" after an Isaac prim prefix.
ARM_JOINTS = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)

# Fixed hold pose (degrees).
HOME_DEG = {
    'shoulder_pan_joint': 130.65,
    'shoulder_lift_joint': -117.4,
    'elbow_joint': -137.2,
    'wrist_1_joint': -69.0,
    'wrist_2_joint': 0.0,
    'wrist_3_joint': 0.0,
}

HOME_RAD = {name: math.radians(deg) for name, deg in HOME_DEG.items()}


def _joint_key(name: str):
    for arm in ARM_JOINTS:
        if name == arm or name.endswith('/' + arm) or name.endswith('_' + arm):
            return arm
    return None


class Ur12eMotion(Node):
    """Command a fixed home pose on /joint_command; no joint motion."""

    def __init__(self):
        super().__init__('ur12e_motion')

        self.declare_parameter('command_topic', '/joint_command')
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('publish_rate', 20.0)

        command_topic = str(self.get_parameter('command_topic').value)
        state_topic = str(self.get_parameter('state_topic').value)
        publish_rate = float(self.get_parameter('publish_rate').value)

        if publish_rate <= 0.0:
            raise ValueError('publish_rate must be > 0')

        # Isaac OmniGraph subscribers are often BEST_EFFORT; publish both
        # so the graph can match either.
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.command_pubs = [
            self.create_publisher(JointState, command_topic, reliable_qos),
            self.create_publisher(JointState, command_topic, sensor_qos),
        ]
        self.create_subscription(JointState, state_topic, self._on_joint_states, sensor_qos)
        self.create_subscription(JointState, state_topic, self._on_joint_states, reliable_qos)

        self.dt = 1.0 / publish_rate
        self.joint_names = None
        self.home_cmd = None
        self.received = 0
        self.waiting_logs = 0
        self.cmd_logs = 0

        self.timer = self.create_timer(self.dt, self._publish_command)
        self.get_logger().info(
            f'Waiting for {state_topic}, then hold pose on {command_topic} '
            f'at {publish_rate:.1f} Hz. Home (deg)={HOME_DEG}'
        )

    def _on_joint_states(self, msg: JointState):
        self.received += 1
        if self.joint_names is None and msg.name and msg.position:
            count = min(len(msg.name), len(msg.position))
            names = list(msg.name[:count])
            home = []
            matched = 0
            for i, name in enumerate(names):
                key = _joint_key(name)
                if key is None:
                    home.append(float(msg.position[i]))
                    continue
                home.append(HOME_RAD[key])
                matched += 1
            if matched < len(ARM_JOINTS):
                self.get_logger().error(
                    f'Missing arm joints in /joint_states names={names}'
                )
            else:
                self.joint_names = names
                self.home_cmd = home
                self.get_logger().info(
                    f'Locked names from Isaac: {names}. Holding fixed home pose'
                )

    def _publish_command(self):
        if self.joint_names is None:
            self.waiting_logs += 1
            if self.waiting_logs % 20 == 1:
                self.get_logger().info('Still waiting for /joint_states. Is Isaac Sim playing?')
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = list(self.home_cmd)
        for pub in self.command_pubs:
            pub.publish(msg)

        self.cmd_logs += 1
        if self.cmd_logs % 20 == 1:
            subs = sum(pub.get_subscription_count() for pub in self.command_pubs)
            self.get_logger().info(
                f'holding home pose (deg)={HOME_DEG}  '
                f'joint_command subscribers={subs}'
            )
            if subs == 0:
                self.get_logger().warning(
                    '/joint_command has 0 Isaac subscribers. Check the ROS 2 Joint States '
                    'graph is playing, topic is /joint_command, and Move Robot is enabled.'
                )


def main(args=None):
    rclpy.init(args=args)
    node = Ur12eMotion()
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
