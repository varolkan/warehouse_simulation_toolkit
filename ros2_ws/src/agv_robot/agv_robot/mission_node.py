"""Transport orders as a ROS 2 action: the mission on top of Nav2.

Action server:
  transport                 agv_interfaces/action/Transport
Action clients:
  navigate_to_pose          nav2_msgs/action/NavigateToPose      travel
  compute_path_to_pose      nav2_msgs/action/ComputePathToPose   route checks before an order
Subscribes:
  load/state                agv_interfaces/LoadStatus
  battery_state             sensor_msgs/BatteryState
  safety/status             agv_interfaces/SafetyStatus
  tf                        map -> base_footprint
Publishes:
  mission/status            agv_interfaces/MissionStatus
  cmd_vel_dock              geometry_msgs/Twist      docking commands, through the safety gate
Calls:
  load/command              agv_interfaces/srv/LoadCommand

Every decision is agv_core.mission.Mission's. This node keeps it fed: it gathers what is
known, runs one cycle at `rate` Hz, and carries out what comes back. Parameters: `site`,
`rate`, the frames, and every field of MissionConfig, DockingConfig and BatteryConfig as
`mission.<field>`, `docking.<field>` and `battery.<field>`.
"""

import math
import threading

from action_msgs.msg import GoalStatus
from agv_core.battery import BatteryConfig
from agv_core.docking import DockingConfig
from agv_core.geometry import Pose2D
from agv_core.mission import (
    Mission, MissionConfig, MissionInputs, MissionResult, NavState, Order, RouteReply)
from agv_core.safety import SafetyState
from agv_core.simulation import LoadState
from agv_core.sites import Site
from agv_interfaces.action import Transport
from agv_interfaces.msg import LoadStatus, MissionStatus, SafetyStatus
from agv_interfaces.srv import LoadCommand
from agv_robot.params import declare_config, stamp_seconds
from agv_robot.paths import site_file
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import BatteryState
import tf2_ros

LOAD_STATES = {state.value: state for state in LoadState}
SAFETY_STATES = {state.value: state for state in SafetyState}
# Planner answers that say "not now" rather than "no": the costmap or the transforms are not
# up yet. The request is made again, within the mission's route_check_timeout.
TRANSIENT_PATH_ERRORS = (202, 207)
ROUTE_RETRY_PERIOD = 0.5
PATH_ERRORS = {
    200: 'unknown error', 201: 'invalid planner', 202: 'no transform',
    203: 'start outside the map', 204: 'goal outside the map', 205: 'start occupied',
    206: 'goal occupied', 207: 'timed out', 208: 'no valid path',
}


def to_pose_stamped(pose: Pose2D, frame: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.pose.position.x, msg.pose.position.y = pose.x, pose.y
    msg.pose.orientation.z, msg.pose.orientation.w = math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)
    return msg


class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')
        self.site_name = self.declare_parameter('site', 'warehouse').value
        self.rate = self.declare_parameter('rate', 20.0).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.planner_id = self.declare_parameter('planner_id', '').value
        self.load_timeout = self.declare_parameter('load_state_timeout', 2.0).value
        self.battery_timeout = self.declare_parameter('battery_state_timeout', 5.0).value
        self.safety_timeout = self.declare_parameter('safety_status_timeout', 0.5).value
        config = declare_config(self, 'mission', MissionConfig)
        docking = declare_config(self, 'docking', DockingConfig)
        battery = declare_config(self, 'battery', BatteryConfig)

        path = site_file(self.site_name)
        self.site = Site.load(path)
        self.mission = Mission(
            self.site, config, docking,
            battery_range=lambda level: battery.range_m(
                level, config.battery_critical, config.cruise_speed))
        self.lock = threading.RLock()

        # What is known, as last heard.
        self.load = (None, LoadState.UNKNOWN)                 # (received at, state)
        self.battery = (None, None, False)                    # (received at, level, charging)
        self.safety = None                                    # (received at, SafetyStatus)
        # Navigation.
        self.nav_state = NavState.IDLE
        self.nav_goal_id = 0
        self.nav_handle = None
        self.nav_distance = None
        self.nav_sending = 0
        # Route checks, one at a time: Nav2's planner drops a request when the next arrives.
        self.routes = {}
        self.route_queue = []          # (not before, request)
        self.route_requests = {}       # request id -> request, while it is being answered
        self.route_busy = False
        # The action goal being worked on.
        self.goal_handle = None
        self.goal_future = None
        self.cancel_sent = False
        self.last_status = None
        self.cycles = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(LoadStatus, 'load/state', self._on_load, 10)
        self.create_subscription(BatteryState, 'battery_state', self._on_battery, 10)
        self.create_subscription(SafetyStatus, 'safety/status', self._on_safety, 10)
        self.status_pub = self.create_publisher(MissionStatus, 'mission/status', 10)
        self.dock_pub = self.create_publisher(Twist, 'cmd_vel_dock', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.load_client = self.create_client(LoadCommand, 'load/command')
        self.server = ActionServer(
            self, Transport, 'transport', execute_callback=self._execute,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT)
        # One thread: the action's execute callback is a coroutine that waits for the
        # cycle to finish the order. Under simulated time the clock wakes every Python node
        # 250 times a second, and a multi-threaded executor made that cost a core.
        self.create_timer(1.0 / self.rate, self._cycle)
        stations = ', '.join(self.site.stations)
        self.get_logger().info(f"site '{self.site.name}' from {path}: {stations}")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- what is known ------------------------------------------------------------------------

    def _on_load(self, msg: LoadStatus) -> None:
        self.load = (self.now(), LOAD_STATES.get(msg.state, LoadState.UNKNOWN))

    def _on_battery(self, msg: BatteryState) -> None:
        level = None if math.isnan(msg.percentage) else max(0.0, min(1.0, msg.percentage))
        charging = msg.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING
        self.battery = (self.now(), level, charging)

    def _on_safety(self, msg: SafetyStatus) -> None:
        self.safety = (self.now(), msg)

    def _pose(self, now: float):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return None, math.inf
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        age = max(0.0, now - stamp_seconds(tf.header.stamp))
        return Pose2D(t.x, t.y, yaw), age

    def _inputs(self) -> MissionInputs:
        now = self.now()
        pose, age = self._pose(now)
        heard, load = self.load
        if heard is None or now - heard > self.load_timeout:
            load = LoadState.UNKNOWN
        heard, level, charging = self.battery
        if heard is None or now - heard > self.battery_timeout:
            level, charging = None, False
        if self.safety is None or now - self.safety[0] > self.safety_timeout:
            safety, reason, source = SafetyState.FAULT, 'safety_node is not reporting', 'none'
        else:
            msg = self.safety[1]
            safety = SAFETY_STATES.get(msg.state, SafetyState.FAULT)
            reason, source = msg.reason, msg.source
        return MissionInputs(
            now=now, pose=pose, pose_age=age, nav=self.nav_state, nav_goal_id=self.nav_goal_id,
            nav_distance_remaining=self.nav_distance, load=load, battery=level,
            charging=charging, safety=safety, safety_reason=reason, command_source=source,
            routes=dict(self.routes))

    # -- the cycle ----------------------------------------------------------------------------

    def _cycle(self) -> None:
        with self.lock:
            inputs = self._inputs()
            handle = self.goal_handle
            if handle is not None and handle.is_cancel_requested and not self.cancel_sent:
                self.mission.cancel()
                self.cancel_sent = True
                self.get_logger().info('cancel requested')
            out = self.mission.update(inputs)
            if out.result is not None and self.goal_future is not None \
                    and not self.goal_future.done():
                self.goal_future.set_result(out.result)
            self.last_status = out.status

        if out.cancel_navigation:
            self._cancel_navigation()
        if out.navigate is not None:
            self._navigate(out.navigate)
        if out.velocity is not None:
            twist = Twist()
            twist.linear.x, twist.angular.z = out.velocity
            self.dock_pub.publish(twist)
        if out.load_command is not None:
            self._command_load(out.load_command)
        if out.route_requests:
            with self.lock:
                self.route_queue.extend((0.0, request) for request in out.route_requests)
        self._next_route()

        self.cycles += 1
        if self.cycles % max(1, int(self.rate / 5.0)) == 0:
            self._publish_status(out.status, inputs)
        feedback_every = max(1, int(self.rate / 10.0))
        if handle is not None and out.status.active and self.cycles % feedback_every == 0:
            handle.publish_feedback(self._feedback(out.status))

    def _publish_status(self, status, inputs: MissionInputs) -> None:
        msg = MissionStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.state, msg.detail, msg.active = status.state, status.detail, status.active
        msg.pickup, msg.dropoff, msg.station = status.pickup, status.dropoff, status.station
        msg.carrying, msg.holding = status.carrying, status.holding
        msg.distance_remaining = float(status.distance_remaining)
        msg.distance_driven = float(status.distance_driven)
        msg.elapsed = float(status.elapsed)
        if inputs.pose is not None:
            msg.pose.x, msg.pose.y, msg.pose.theta = inputs.pose.x, inputs.pose.y, inputs.pose.yaw
        msg.localized = (inputs.pose is not None
                         and inputs.pose_age <= self.mission.config.pose_timeout)
        msg.battery = -1.0 if inputs.battery is None else float(inputs.battery)
        msg.charging = inputs.charging
        msg.load = inputs.load.value
        msg.safety = inputs.safety.value
        self.status_pub.publish(msg)

    # -- navigation ---------------------------------------------------------------------------

    def _navigate(self, goal) -> None:
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('navigate_to_pose is not available yet')
            return
        msg = NavigateToPose.Goal()
        msg.pose = to_pose_stamped(goal.pose, self.map_frame, self.get_clock().now().to_msg())
        self.get_logger().info(
            f"driving to '{goal.station}' ({goal.pose.x:.2f}, {goal.pose.y:.2f}, "
            f'{math.degrees(goal.pose.yaw):.0f} deg)')
        with self.lock:
            self.nav_sending = goal.goal_id
        future = self.nav_client.send_goal_async(msg, feedback_callback=self._on_nav_feedback)
        future.add_done_callback(lambda f, goal_id=goal.goal_id: self._on_nav_accepted(f, goal_id))

    def _on_nav_accepted(self, future, goal_id: int) -> None:
        handle = future.result()
        with self.lock:
            if goal_id != self.nav_sending:
                # A newer goal went out meanwhile: this one is not wanted any more.
                if handle is not None and handle.accepted:
                    handle.cancel_goal_async()
                return
            self.nav_goal_id = goal_id
            self.nav_distance = None
            if handle is None or not handle.accepted:
                self.nav_state, self.nav_handle = NavState.FAILED, None
                self.get_logger().warn('navigation refused the goal')
                return
            self.nav_state, self.nav_handle = NavState.ACTIVE, handle
        handle.get_result_async().add_done_callback(
            lambda f, goal_id=goal_id: self._on_nav_result(f, goal_id))

    def _on_nav_feedback(self, feedback) -> None:
        with self.lock:
            if self.nav_state is NavState.ACTIVE:
                self.nav_distance = float(feedback.feedback.distance_remaining)

    def _on_nav_result(self, future, goal_id: int) -> None:
        response = future.result()
        with self.lock:
            if goal_id != self.nav_goal_id:
                return
            status = response.status if response is not None else GoalStatus.STATUS_ABORTED
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.nav_state = NavState.SUCCEEDED
            elif status == GoalStatus.STATUS_CANCELED:
                self.nav_state = NavState.CANCELED
            else:
                self.nav_state = NavState.FAILED
                error = response.result.error_msg if response is not None else ''
                self.get_logger().warn(f"navigation gave up{': ' + error if error else ''}")
            self.nav_handle = None

    def _cancel_navigation(self) -> None:
        with self.lock:
            handle = self.nav_handle
            self.nav_sending = -1
        if handle is not None:
            handle.cancel_goal_async()

    # -- route checks -------------------------------------------------------------------------

    def _next_route(self) -> None:
        with self.lock:
            if self.route_busy or not self.route_queue:
                return
            not_before, request = self.route_queue[0]
            if self.now() < not_before:
                return
            self.route_queue.pop(0)
            if not self.path_client.server_is_ready():
                # Navigation is still starting: ask again shortly, until the mission's
                # route_check_timeout says otherwise.
                self.route_queue.append((self.now() + ROUTE_RETRY_PERIOD, request))
                return
            self.route_busy = True
            self.route_requests[request.request_id] = request
        stamp = self.get_clock().now().to_msg()
        goal = ComputePathToPose.Goal()
        goal.start = to_pose_stamped(request.start, self.map_frame, stamp)
        goal.goal = to_pose_stamped(request.goal, self.map_frame, stamp)
        goal.use_start = True
        goal.planner_id = self.planner_id
        future = self.path_client.send_goal_async(goal)
        future.add_done_callback(lambda f, rid=request.request_id: self._on_route_accepted(f, rid))

    def _on_route_accepted(self, future, request_id: int) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self._route_done(request_id, RouteReply(None, 'the planner refused the request'))
            return
        handle.get_result_async().add_done_callback(
            lambda f: self._on_route_result(f, request_id))

    def _on_route_result(self, future, request_id: int) -> None:
        response = future.result()
        poses = response.result.path.poses if response is not None else []
        if response is not None and response.status == GoalStatus.STATUS_SUCCEEDED and poses:
            length = sum(
                math.hypot(b.pose.position.x - a.pose.position.x,
                           b.pose.position.y - a.pose.position.y)
                for a, b in zip(poses, poses[1:]))
            self._route_done(request_id, RouteReply(length))
            return
        error = ''
        if response is not None:
            if response.result.error_code in TRANSIENT_PATH_ERRORS:
                self._route_again(request_id, response.result.error_msg)
                return
            error = response.result.error_msg or PATH_ERRORS.get(response.result.error_code, '')
        self._route_done(request_id, RouteReply(None, error or 'no path'))

    def _route_again(self, request_id: int, why: str) -> None:
        with self.lock:
            request = self.route_requests.pop(request_id, None)
            self.route_busy = False
            if request is not None:
                self.route_queue.append((self.now() + ROUTE_RETRY_PERIOD, request))
        self.get_logger().info(f'route check to be repeated: {why}')

    def _route_done(self, request_id: int, reply: RouteReply) -> None:
        with self.lock:
            self.routes[request_id] = reply
            self.route_requests.pop(request_id, None)
            self.route_busy = False

    # -- the load handler ---------------------------------------------------------------------

    def _command_load(self, command) -> None:
        if not self.load_client.service_is_ready():
            self.get_logger().warn('load/command is not available; the mission will try again')
            return
        request = LoadCommand.Request()
        request.action, request.station = command.action, command.station
        self.get_logger().info(f"{command.action} at '{command.station}'")
        future = self.load_client.call_async(request)
        future.add_done_callback(lambda f, action=command.action: self._on_load_reply(f, action))

    def _on_load_reply(self, future, action: str) -> None:
        reply = future.result()
        if reply is not None and not reply.accepted:
            self.get_logger().warn(f'the load handler refused {action}: {reply.message}')

    # -- the action ---------------------------------------------------------------------------

    async def _execute(self, goal_handle):
        request = goal_handle.request
        order = Order(request.pickup, request.dropoff, request.return_to_charger)
        with self.lock:
            refusal = self.mission.start(order, self._inputs())
            if refusal is None:
                self.goal_handle, self.goal_future = goal_handle, Future()
                self.cancel_sent = False
        description = (f"from '{order.pickup}' " if order.pickup else '') + f"to '{order.dropoff}'"
        if refusal is not None:
            self.get_logger().warn(f'order {description} refused: {refusal}')
            goal_handle.abort()
            return self._result(MissionResult('refused', refusal))
        self.get_logger().info(f'order {description} accepted')

        # The cycle publishes feedback, passes on a cancel, and completes this future.
        result = await self.goal_future
        with self.lock:
            self.goal_handle, self.goal_future = None, None
        if result is None:
            result = MissionResult('failed', 'shutting down')
        if result.success:
            self.get_logger().info(f'order {description}: {result.outcome}: {result.message}')
        else:
            self.get_logger().warn(f'order {description}: {result.outcome}: {result.message}')
        if result.success:
            goal_handle.succeed()
        elif result.outcome == 'canceled' and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return self._result(result)

    @staticmethod
    def _feedback(status) -> Transport.Feedback:
        feedback = Transport.Feedback()
        feedback.state, feedback.detail, feedback.station = (
            status.state, status.detail, status.station)
        feedback.carrying, feedback.holding = status.carrying, status.holding
        feedback.distance_remaining = float(status.distance_remaining)
        feedback.distance_driven = float(status.distance_driven)
        feedback.elapsed = float(status.elapsed)
        return feedback

    @staticmethod
    def _result(result: MissionResult) -> Transport.Result:
        msg = Transport.Result()
        msg.success, msg.delivered = result.success, result.delivered
        msg.outcome, msg.message = result.outcome, result.message
        msg.duration, msg.distance = float(result.duration), float(result.distance)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.dock_pub.publish(Twist())
        except Exception:  # noqa: BLE001 - the context may already be gone
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
