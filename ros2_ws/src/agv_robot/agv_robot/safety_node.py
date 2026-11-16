"""The safety gate as a node: every velocity command goes through it to the drive.

Subscribes:
  scan               sensor_msgs/LaserScan   the filtered scan (the deck posts removed)
  odom               nav_msgs/Odometry       measured velocity
  cmd_vel_smoothed   geometry_msgs/Twist     Nav2, after its velocity smoother
  cmd_vel_dock       geometry_msgs/Twist     the mission's docking controller
  cmd_vel_manual     geometry_msgs/Twist     an operator; outranks the other two
  estop              std_msgs/Bool           the e-stop button, true while pressed
Publishes:
  cmd_vel                    geometry_msgs/Twist         to the drive, every cycle
  safety/status              agv_interfaces/SafetyStatus
  safety/protective_field    geometry_msgs/PolygonStamped
  safety/warning_field       geometry_msgs/PolygonStamped
Services:
  safety/reset       std_srvs/Trigger        clear a latched e-stop once released

The decisions are agv_core.safety.SafetyMonitor's; this node only feeds it and passes its
answer on. Parameters: `rate` (Hz), `base_frame`, and every field of SafetyConfig as
`safety.<field>`.
"""

import math

from agv_core.geometry import Pose2D
from agv_core.safety import Command, SafetyConfig, SafetyMonitor, SafetyState, Scan
from agv_interfaces.msg import SafetyStatus
from agv_robot.params import declare_config, stamp_seconds
from geometry_msgs.msg import Point32, PolygonStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import tf2_ros


class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')
        self.rate = self.declare_parameter('rate', 50.0).value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.monitor = SafetyMonitor(declare_config(self, 'safety', SafetyConfig))

        self.scan = None
        self.odom = None
        self.commands = {'nav': None, 'dock': None, 'manual': None}
        self.sensor_pose = {}
        self.last_state = None
        self.cycles = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, 'scan', self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self._on_odom, 10)
        for source, topic in (('nav', 'cmd_vel_smoothed'), ('dock', 'cmd_vel_dock'),
                              ('manual', 'cmd_vel_manual')):
            self.create_subscription(
                Twist, topic, lambda msg, source=source: self._on_command(source, msg), 10)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Bool, 'estop', self._on_estop, reliable)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(SafetyStatus, 'safety/status', 10)
        self.protective_pub = self.create_publisher(PolygonStamped, 'safety/protective_field', 1)
        self.warning_pub = self.create_publisher(PolygonStamped, 'safety/warning_field', 1)
        self.create_service(Trigger, 'safety/reset', self._on_reset)
        self.create_timer(1.0 / self.rate, self._cycle)
        self.get_logger().info(f'safety gate running at {self.rate:.0f} Hz')

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- inputs ----------------------------------------------------------------------

    def _on_scan(self, msg: LaserScan) -> None:
        sensor = self._sensor_pose(msg.header.frame_id)
        if sensor is None:
            return
        self.scan = Scan.from_ranges(
            stamp_seconds(msg.header.stamp), msg.ranges, msg.angle_min, msg.angle_increment,
            msg.range_min, msg.range_max, sensor)

    def _sensor_pose(self, frame: str):
        """Return where the scanner is on the vehicle (a fixed transform, looked up once)."""
        if frame not in self.sensor_pose:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time())
            except tf2_ros.TransformException:
                return None
            t, q = tf.transform.translation, tf.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.sensor_pose[frame] = Pose2D(t.x, t.y, yaw)
        return self.sensor_pose[frame]

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = Command(stamp_seconds(msg.header.stamp), msg.twist.twist.linear.x,
                            msg.twist.twist.angular.z)

    def _on_command(self, source: str, msg: Twist) -> None:
        self.commands[source] = Command(self.now(), msg.linear.x, msg.angular.z)

    def _on_estop(self, msg: Bool) -> None:
        if msg.data and not self.monitor.estop_pressed:
            self.monitor.press_estop()
            self.get_logger().warn('e-stop pressed')
        elif not msg.data and self.monitor.estop_pressed:
            self.monitor.release_estop()
            self.get_logger().info('e-stop released; call safety/reset to drive again')

    def _on_reset(self, request, response):
        response.success, response.message = self.monitor.reset()
        self.get_logger().info(f'reset: {response.message}')
        return response

    # -- the cycle ---------------------------------------------------------------------

    def _cycle(self) -> None:
        now = self.now()
        timeout = self.monitor.config.command_timeout
        dock = self.commands['dock']
        nav, source = self.commands['nav'], 'nav'
        if dock is not None and dock.fresh(now, timeout):
            nav, source = dock, 'dock'
        status = self.monitor.update(now, self.scan, self.odom, nav=nav,
                                     manual=self.commands['manual'], docking=source == 'dock')
        if status.source == 'nav':
            status_source = source
        else:
            status_source = status.source

        twist = Twist()
        twist.linear.x, twist.angular.z = status.v, status.w
        self.cmd_pub.publish(twist)

        self.cycles += 1
        changed = status.state is not self.last_state
        if changed:
            # rclpy wants one severity per logging call site.
            text = status.state.value + (f': {status.reason}' if status.reason else '')
            if status.state.stopped:
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)
            self.last_state = status.state
        if changed or self.cycles % max(1, int(self.rate / 10.0)) == 0:
            self._publish_status(status, status_source)

    def _publish_status(self, status, source: str) -> None:
        stamp = self.get_clock().now().to_msg()
        msg = SafetyStatus()
        msg.header.stamp, msg.header.frame_id = stamp, self.base_frame
        msg.state, msg.reason, msg.source = status.state.value, status.reason, source
        msg.linear_requested, msg.angular_requested = status.v_requested, status.w_requested
        msg.linear, msg.angular = status.v, status.w
        msg.gap = -1.0 if status.gap is None else float(status.gap)
        msg.estop_pressed = self.monitor.estop_pressed
        msg.protective_field.points = [Point32(x=float(x), y=float(y))
                                       for x, y in status.protective_field]
        msg.warning_field.points = [Point32(x=float(x), y=float(y))
                                    for x, y in status.warning_field]
        self.status_pub.publish(msg)
        for publisher, polygon in ((self.protective_pub, msg.protective_field),
                                   (self.warning_pub, msg.warning_field)):
            field = PolygonStamped()
            field.header = msg.header
            field.polygon = polygon
            publisher.publish(field)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Whatever happens, the last word to the drive is "stop".
        try:
            node.cmd_pub.publish(Twist())
        except Exception:  # noqa: BLE001 - the context may already be gone
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


__all__ = ['SafetyNode', 'SafetyState', 'main']
