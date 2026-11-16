"""Shared pieces of the dry-run launch tests: the launch description and a client.

The stack under test is the real one (bringup.launch.py: AMCL, Nav2, safety_node,
mission_node) on fake_base, which knows where the vehicle really is and publishes it on
ground_truth/odometry. The client sends orders, records everything that comes back, and
can put things in the vehicle's way.
"""

import math
import os
import time

from action_msgs.msg import GoalStatus
from agv_core.geometry import Pose2D
from agv_core.sites import Site
from agv_interfaces.action import Transport
from agv_interfaces.msg import MissionStatus, SafetyStatus
from agv_robot.paths import site_file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing.actions
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

SUCCEEDED = GoalStatus.STATUS_SUCCEEDED
ABORTED = GoalStatus.STATUS_ABORTED
CANCELED = GoalStatus.STATUS_CANCELED
SITE = Site.load(site_file('warehouse'))


def dry_run_launch(**arguments):
    share = get_package_share_directory('agv_robot')
    arguments.setdefault('log_level', 'warn')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'dry_run.launch.py')),
            launch_arguments={k: str(v) for k, v in arguments.items()}.items()),
        launch_testing.actions.ReadyToTest(),
    ])


class Client:
    """Transport action client that records what the stack says while it waits."""

    def __init__(self, node):
        self.node = node
        self.action = ActionClient(node, Transport, 'transport')
        self.feedback, self.statuses, self.safety = [], [], []
        self.truth = None
        self.track = []
        node.create_subscription(MissionStatus, 'mission/status', self.statuses.append, 10)
        node.create_subscription(SafetyStatus, 'safety/status', self.safety.append, 10)
        node.create_subscription(Odometry, 'ground_truth/odometry', self._on_truth, 10)
        self.obstacles = node.create_publisher(String, 'sim/obstacles', 10)
        self.estop = node.create_publisher(Bool, 'estop', 10)
        self.reset = node.create_client(Trigger, 'safety/reset')

    def _on_truth(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.truth = Pose2D(p.x, p.y, yaw)
        self.track.append((time.monotonic(), p.x, p.y, msg.twist.twist.linear.x))

    def spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return predicate()

    def spin_for(self, seconds):
        self.spin_until(lambda: False, seconds)

    def last_status(self):
        return self.statuses[-1] if self.statuses else None

    def last_safety(self):
        return self.safety[-1] if self.safety else None

    def wait_until_ready(self, timeout=120.0, load='empty'):
        """Wait for what an operator checks before dispatching: localised, gate up, load known."""
        assert self.action.wait_for_server(timeout_sec=timeout), 'transport action missing'
        assert self.spin_until(
            lambda: self.statuses and self.statuses[-1].localized
            and self.statuses[-1].load == load and self.statuses[-1].safety == 'ok'
            and self.statuses[-1].battery >= 0.0, timeout), \
            f'vehicle never became ready: {self.last_status()}'

    def send(self, pickup, dropoff, return_to_charger=False):
        goal = Transport.Goal(pickup=pickup, dropoff=dropoff, return_to_charger=return_to_charger)
        future = self.action.send_goal_async(
            goal, feedback_callback=lambda msg: self.feedback.append(msg.feedback))
        assert self.spin_until(future.done, 30.0), 'goal was never answered'
        handle = future.result()
        assert handle.accepted, 'goals are always accepted; refusals come back as results'
        return handle

    def wait_result(self, handle, timeout):
        future = handle.get_result_async()
        assert self.spin_until(future.done, timeout), f'no result within {timeout} s'
        return future.result()

    def order(self, pickup, dropoff, timeout, **options):
        return self.wait_result(self.send(pickup, dropoff, **options), timeout)

    def states(self):
        seen = []
        for feedback in self.feedback:
            if not seen or seen[-1] != feedback.state:
                seen.append(feedback.state)
        return seen

    def docking_error(self, station):
        """Metres along and across the station's centre line, and degrees off its heading."""
        local = self.truth.relative_to(SITE.station(station).pose)
        return local.x, local.y, math.degrees(local.yaw)

    def put(self, *lines):
        self.obstacles.publish(String(data='\n'.join(lines)))

    def press_estop(self, pressed=True):
        self.estop.publish(Bool(data=pressed))

    def reset_safety(self):
        assert self.reset.wait_for_service(timeout_sec=10.0)
        future = self.reset.call_async(Trigger.Request())
        assert self.spin_until(future.done, 10.0)
        return future.result()
