"""A stand-in for the vehicle and its simulator: the whole stack runs without Gazebo.

Publishes what the simulated vehicle publishes, on the same topics:
  scan_raw              sensor_msgs/LaserScan    cast through the site's map (and obstacles),
                                                 the deck posts included, as the real scanner
  wheel/odometry        nav_msgs/Odometry        odom -> base_footprint, no TF (the EKF does that)
  imu                   sensor_msgs/Imu          the turn rate
  joint_states          sensor_msgs/JointState   wheel angles, for robot_state_publisher
  battery_state         sensor_msgs/BatteryState charges while docked at a charger
  load/state            agv_interfaces/LoadStatus
  ground_truth/odometry nav_msgs/Odometry        where the vehicle really is, map frame:
                                                 for tests and plots only
and takes:
  cmd_vel               geometry_msgs/Twist
  load/command          agv_interfaces/srv/LoadCommand
  sim/obstacles         std_msgs/String          obstacles not in the map, one per line:
                                                 "disc X Y R" or "box X Y LENGTH WIDTH [YAW_DEG]";
                                                 an empty string clears them

Everything it does is agv_core.simulation's. Parameters: `site`, `start` (a station name, or
"x,y,heading_deg" in the map frame), `rate`, `scan_rate`, `loaded`, `load_duration`,
`load_rejections`, `load_stuck`, `battery`, `charger_dead`, `charger_contact` (metres),
`odom_distance_scale`, `odom_turn_scale`, `obstacles` (a list in the format above), `seed`.

How well the vehicle docks here is how well AMCL localises on a scanner that casts its
rays through 5 cm map cells: within about 6 cm. The charger therefore takes the vehicle
within `charger_contact` = 8 cm by default, so that the tests check what the mission does
with a charger, not the localiser's precision. Docking precision is measured in Gazebo.
"""

import math

from agv_core.geometry import Pose2D
from agv_core.grid import GridMap
from agv_core.simulation import (
    BatterySim, Box, Disc, DriveConfig, KinematicVehicle, LoadHandlerSim, ScannerConfig,
    ScannerSim)
from agv_core.sites import Site
from agv_core.vehicle import AGV
from agv_interfaces.msg import LoadStatus
from agv_interfaces.srv import LoadCommand
from agv_robot.paths import MAP_FILE, site_file
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, Imu, JointState, LaserScan
from std_msgs.msg import String


def parse_obstacles(lines):
    """Turn "disc X Y R" / "box X Y LENGTH WIDTH [YAW_DEG]" lines into obstacles."""
    obstacles = []
    for line in lines:
        words = line.split()
        if not words:
            continue
        kind, values = words[0], [float(v) for v in words[1:]]
        if kind == 'disc' and len(values) == 3:
            obstacles.append(Disc(*values))
        elif kind == 'box' and len(values) in (4, 5):
            yaw = math.radians(values[4]) if len(values) == 5 else 0.0
            obstacles.append(Box(values[0], values[1], values[2], values[3], yaw))
        else:
            raise ValueError(f"cannot read obstacle '{line}'")
    return obstacles


class FakeBase(Node):

    def __init__(self):
        super().__init__('fake_base')
        site_name = self.declare_parameter('site', 'warehouse').value
        start = self.declare_parameter('start', 'charger').value
        self.rate = self.declare_parameter('rate', 50.0).value
        scan_rate = self.declare_parameter('scan_rate', 10.0).value
        loaded = self.declare_parameter('loaded', False).value
        load_duration = self.declare_parameter('load_duration', 3.0).value
        load_rejections = self.declare_parameter('load_rejections', 0).value
        load_stuck = self.declare_parameter('load_stuck', False).value
        battery = self.declare_parameter('battery', 0.8).value
        charger_dead = self.declare_parameter('charger_dead', False).value
        charger_contact = self.declare_parameter('charger_contact', 0.08).value
        distance_scale = self.declare_parameter('odom_distance_scale', 1.0).value
        turn_scale = self.declare_parameter('odom_turn_scale', 1.0).value
        obstacles = self.declare_parameter('obstacles', ['']).value
        seed = self.declare_parameter('seed', 0).value

        self.site = Site.load(site_file(site_name))
        grid = GridMap.load(site_file(site_name, MAP_FILE))
        if start in self.site.stations:
            pose = self.site.station(start).pose
        else:
            x, y, heading = (float(v) for v in start.split(','))
            pose = Pose2D(x, y, math.radians(heading))
        self.vehicle = KinematicVehicle(
            pose, DriveConfig(odom_distance_scale=distance_scale, odom_turn_scale=turn_scale))
        self.scanner = ScannerSim(grid, ScannerConfig(), seed=seed)
        self.scanner.obstacles.extend(parse_obstacles(obstacles))
        self.handler = LoadHandlerSim(loaded=loaded, duration=load_duration,
                                      rejections=load_rejections, stuck=load_stuck)
        self.battery = BatterySim(level=battery, chargers=[c.pose for c in self.site.chargers],
                                  contact_distance=charger_contact)
        self.battery.charger_dead = charger_dead
        self.wheel_angles = [0.0, 0.0]
        self.last_step = None

        self.scan_pub = self.create_publisher(LaserScan, 'scan_raw', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, 'wheel/odometry', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu', qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.battery_pub = self.create_publisher(BatteryState, 'battery_state', 10)
        self.load_pub = self.create_publisher(LoadStatus, 'load/state', 10)
        self.truth_pub = self.create_publisher(Odometry, 'ground_truth/odometry', 10)
        self.create_subscription(Twist, 'cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(String, 'sim/obstacles', self._on_obstacles, 10)
        self.create_service(LoadCommand, 'load/command', self._on_load_command)
        self.create_timer(1.0 / self.rate, self._step)
        self.create_timer(1.0 / scan_rate, self._publish_scan)
        self.create_timer(0.2, self._publish_slow)
        self.get_logger().info(
            f'fake base at ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.yaw):.0f} deg) in '
            f"'{self.site.name}', {len(self.scanner.obstacles)} obstacle(s)")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.vehicle.command(msg.linear.x, msg.angular.z, self.now())

    def _on_obstacles(self, msg: String) -> None:
        try:
            obstacles = parse_obstacles(msg.data.splitlines())
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        self.scanner.obstacles[:] = obstacles
        self.get_logger().info(f'{len(obstacles)} obstacle(s) in the hall now')

    def _on_load_command(self, request, response):
        response.accepted, response.message = self.handler.command(request.action, self.now())
        return response

    def _step(self) -> None:
        now = self.now()
        dt = 0.0 if self.last_step is None else min(0.2, max(0.0, now - self.last_step))
        self.last_step = now
        before = self.vehicle.distance
        self.vehicle.step(dt, now)
        self.handler.step(now)
        self.battery.step(dt, self.vehicle.pose, self.vehicle.distance - before, self.vehicle.v)
        # Wheel angles from the true motion: left and right speeds of a differential drive.
        for i, side in enumerate((1.0, -1.0)):
            speed = self.vehicle.v - side * self.vehicle.w * AGV.wheel_separation / 2.0
            self.wheel_angles[i] += speed / AGV.wheel_radius * dt

        stamp = self.get_clock().now().to_msg()
        odom = self.vehicle.odom
        msg = self._odometry(stamp, 'odom', odom)
        msg.twist.twist.linear.x = self.vehicle.v * self.vehicle.config.odom_distance_scale
        msg.twist.twist.angular.z = self.vehicle.w * self.vehicle.config.odom_turn_scale
        msg.twist.covariance[0] = msg.twist.covariance[7] = 1e-4
        msg.twist.covariance[35] = 1e-3
        self.odom_pub.publish(msg)

        truth = self._odometry(stamp, 'map', self.vehicle.pose)
        truth.twist.twist.linear.x, truth.twist.twist.angular.z = self.vehicle.v, self.vehicle.w
        self.truth_pub.publish(truth)

        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, 'imu_link'
        imu.orientation_covariance[0] = -1.0                 # no orientation
        imu.angular_velocity.z = self.vehicle.w
        imu.angular_velocity_covariance[8] = 4e-6
        imu.linear_acceleration.z = 9.81
        self.imu_pub.publish(imu)

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = ['left_wheel_joint', 'right_wheel_joint']
        joints.position = list(self.wheel_angles)
        self.joint_pub.publish(joints)

    def _odometry(self, stamp, frame: str, pose: Pose2D) -> Odometry:
        msg = Odometry()
        msg.header.stamp, msg.header.frame_id = stamp, frame
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x, msg.pose.pose.position.y = pose.x, pose.y
        msg.pose.pose.orientation.z = math.sin(pose.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(pose.yaw / 2.0)
        return msg

    def _publish_scan(self) -> None:
        cfg = self.scanner.config
        ranges = self.scanner.scan(self.vehicle.pose)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lidar_link'
        msg.angle_min = cfg.angle_min
        msg.angle_increment = cfg.angle_increment
        msg.angle_max = cfg.angle_min + cfg.angle_increment * (cfg.beams - 1)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min, msg.range_max = cfg.range_min, cfg.range_max
        msg.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(msg)

    def _publish_slow(self) -> None:
        stamp = self.get_clock().now().to_msg()
        battery = BatteryState()
        battery.header.stamp = stamp
        battery.percentage = float(self.battery.level)
        battery.present = True
        battery.power_supply_status = (
            BatteryState.POWER_SUPPLY_STATUS_CHARGING if self.battery.charging
            else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING)
        self.battery_pub.publish(battery)
        load = LoadStatus()
        load.header.stamp = stamp
        load.state = self.handler.state.value
        self.load_pub.publish(load)


def main(args=None):
    rclpy.init(args=args)
    node = FakeBase()
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
