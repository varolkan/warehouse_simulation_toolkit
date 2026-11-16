"""The parts of the vehicle Gazebo does not simulate: the load handler and the battery.

Load handler (load/command, load/state). A transfer takes `transfer_time` seconds and is
only done docked at a transfer station: loading takes the tote off that station's stand
onto the deck, where Gazebo's DetachableJoint holds it; unloading lets go and puts it on
the stand. How the load moves between stand and deck (rollers, a lift) is not modelled:
the tote is set down where it goes, half way through the transfer. Gazebo answers a
request to move a model before it has moved it, so the handler watches the tote's own
ground truth: it closes the joint only once the tote is on the deck, sends it again if it
does not arrive, and reports a fault if it never does.

Battery (battery_state). agv_core's BatterySim, driven by the vehicle's true pose from
Gazebo: it charges only while the vehicle stands docked on a charger's contacts.

Through ros_gz_bridge (agv_gazebo/config/bridge.yaml): ground_truth/odometry, the
tote's /sim/load/odometry, the attach/detach topics and joint state of the load, and the
world's set_pose service.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from agv_core.geometry import Pose2D
from agv_core.simulation import BatterySim, LoadState
from agv_core.sites import Site, Station
from agv_gazebo.world import load_pose_on_stand, load_pose_on_vehicle, WORLD_NAME
from agv_interfaces.msg import LoadStatus
from agv_interfaces.srv import LoadCommand
from agv_robot.paths import site_file
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Empty, String

DOCKED_WITHIN = 0.10          # metres from a station's docked pose
DOCKED_HEADING = math.radians(8.0)
RELEASE_RETRY = 1.0           # seconds between requests to release the load, unanswered
ARRIVED_WITHIN = 0.03         # metres between where the tote was sent and where it is
MOVE_TIMEOUT = 1.0            # seconds for Gazebo to show the tote where it was sent
MOVE_TRIES = 3
ATTACH_RETRY = 0.5            # seconds between requests to close the joint, unanswered


@dataclass
class Transfer:
    """One load or unload, from the command to the joint in its final state."""

    action: str                                         # load | unload
    station: Station
    started: float
    target: Optional[Tuple[float, float, float]] = None  # where the tote was sent
    sent_at: float = -math.inf
    tries: int = 0
    arrived: bool = False
    attach_sent_at: float = -math.inf


class SimAgv(Node):

    def __init__(self):
        super().__init__('sim_agv')
        site_name = self.declare_parameter('site', 'warehouse').value
        self.load_model = self.declare_parameter('load_model', 'tote').value
        load_at = self.declare_parameter('load_at', '').value
        self.transfer_time = self.declare_parameter('transfer_time', 3.0).value
        battery = self.declare_parameter('battery', 0.8).value
        self.site = Site.load(site_file(site_name))

        self.truth = None
        self.speed = 0.0
        self.distance = 0.0
        self.attached = None            # as Gazebo last reported
        self.load_pose = None           # (x, y, z) of the tote, and when it was heard
        self.load_heard = -math.inf
        self.last_release = -math.inf   # when the load joint was last asked to open
        self.releases = 0
        self.load_station = load_at or None     # where the load stands, if on a stand
        # With a load in the world, nothing is known until Gazebo says the joint is open:
        # its DetachableJoint closes by itself the moment both models exist. Without one,
        # the vehicle has no joint and an empty deck.
        self.state = LoadState.UNKNOWN if self.load_station else LoadState.EMPTY
        self.transfer: Optional[Transfer] = None
        self.battery = BatterySim(level=battery, chargers=[c.pose for c in self.site.chargers])
        self.last_battery_step = None

        self.create_subscription(Odometry, 'ground_truth/odometry', self._on_truth, 10)
        self.create_subscription(String, '/agv/load/joint_state', self._on_joint, 10)
        self.create_subscription(Odometry, '/sim/load/odometry', self._on_load_truth, 10)
        self.attach_pub = self.create_publisher(Empty, '/agv/load/attach', 10)
        self.detach_pub = self.create_publisher(Empty, '/agv/load/detach', 10)
        self.state_pub = self.create_publisher(LoadStatus, 'load/state', 10)
        self.battery_pub = self.create_publisher(BatteryState, 'battery_state', 10)
        self.set_pose = self.create_client(SetEntityPose, f'/world/{WORLD_NAME}/set_pose')
        self.create_service(LoadCommand, 'load/command', self._on_command)
        self.create_timer(0.1, self._step)
        self.create_timer(0.2, self._publish)
        self.get_logger().info(
            f"load handler for '{self.load_model}'"
            + (f", load waiting at '{load_at}'" if load_at else ', no load in the world'))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- what Gazebo says ------------------------------------------------------------------

    def _on_truth(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = Pose2D(p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                           1 - 2 * (q.y * q.y + q.z * q.z)))
        if self.truth is not None:
            self.distance += self.truth.distance_to(pose)
        self.truth = pose
        self.speed = msg.twist.twist.linear.x

    def _on_joint(self, msg: String) -> None:
        attached = msg.data == 'attached'
        if attached != self.attached:
            self.get_logger().info(f'load joint {msg.data}')
        self.attached = attached

    def _on_load_truth(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.load_pose = (p.x, p.y, p.z)
        self.load_heard = self.now()

    # -- the load handler ------------------------------------------------------------------

    def _docked_station(self):
        if self.truth is None:
            return None
        for station in self.site.transfer_stations:
            if (self.truth.distance_to(station.pose) <= DOCKED_WITHIN
                    and abs(math.remainder(self.truth.yaw - station.pose.yaw, 2 * math.pi))
                    <= DOCKED_HEADING):
                return station
        return None

    def _on_command(self, request, response):
        action = request.action
        if action not in ('load', 'unload'):
            response.accepted, response.message = False, f"unknown load command '{action}'"
            return response
        if self.transfer is not None:
            response.accepted, response.message = False, 'a transfer is already under way'
            return response
        wanted = LoadState.LOADED if action == 'load' else LoadState.EMPTY
        if self.state is wanted:
            response.accepted, response.message = True, f'already {wanted.value}'
            return response
        station = self._docked_station()
        if station is None:
            response.accepted, response.message = False, 'not docked at a transfer station'
            return response
        if action == 'load' and self.load_station != station.name:
            where = f"at '{self.load_station}'" if self.load_station else 'on the vehicle'
            response.accepted = False
            response.message = f"no load on the stand at '{station.name}' (it is {where})"
            return response
        self.transfer = Transfer(action, station, self.now())
        self.state = LoadState.MOVING
        if action == 'unload':
            self.detach_pub.publish(Empty())
            self.last_release = self.now()
        response.accepted, response.message = True, ''
        self.get_logger().info(f"{action} at '{station.name}'")
        return response

    def _step(self) -> None:
        now = self.now()
        if self.state is LoadState.UNKNOWN and self.load_station:
            if self.attached is False:
                self.state = LoadState.EMPTY
                self.get_logger().info(f"ready: the load is on the stand at '{self.load_station}'")
            elif self.attached or now - self.last_release >= RELEASE_RETRY:
                # DetachableJoint attaches by itself as soon as both models exist: let go.
                # It reports its state only when the state changes, so the report of that
                # first attach can be gone before the bridge is up: then ask, until it
                # answers, at a calmer rate.
                self.detach_pub.publish(Empty())
                self.last_release = now
                self.releases += 1
                if self.attached is None and self.releases % 10 == 0:
                    self.get_logger().warn(
                        f'no word from Gazebo on the load joint after {self.releases} '
                        'requests to release it: is the bridge running?')
        if self.transfer is not None:
            self._advance(self.transfer, now)
        if self.truth is not None:
            dt = 0.0 if self.last_battery_step is None else now - self.last_battery_step
            self.battery.step(max(0.0, dt), self.truth, self.distance, self.speed)
            self.distance = 0.0
            self.last_battery_step = now

    def _advance(self, transfer: Transfer, now: float) -> None:
        """Take a transfer one step further: move the tote, then close or open the joint."""
        elapsed = now - transfer.started
        if transfer.target is None:
            if transfer.action == 'unload' and self.attached is not False \
                    and now - self.last_release >= RELEASE_RETRY:
                self.detach_pub.publish(Empty())
                self.last_release = now
            # Half way through, the tote changes place; to unload, once the joint is open.
            if elapsed >= self.transfer_time / 2.0 and (
                    transfer.action == 'load' or self.attached is False):
                self._send_load(transfer, now)
            elif elapsed >= self.transfer_time + 5.0:
                self._fail(f'{transfer.action}: the load joint did not open')
            return
        if not transfer.arrived:
            if self._load_is_at(transfer.target, since=transfer.sent_at):
                transfer.arrived = True
            elif now - transfer.sent_at >= MOVE_TIMEOUT:
                if transfer.tries < MOVE_TRIES:
                    self._send_load(transfer, now)
                else:
                    self._fail(f'{transfer.action}: Gazebo does not show the load where it '
                               f'was sent, {transfer.tries} times')
                return
            else:
                return
        if transfer.action == 'load' and not self.attached:
            if now - transfer.attach_sent_at >= ATTACH_RETRY:
                self.attach_pub.publish(Empty())
                transfer.attach_sent_at = now
        if elapsed < self.transfer_time:
            return
        if transfer.action == 'load' and self.attached:
            self.state, self.load_station = LoadState.LOADED, None
            self.transfer = None
        elif transfer.action == 'unload' and self.attached is False:
            self.state, self.load_station = LoadState.EMPTY, transfer.station.name
            self.transfer = None
        elif elapsed >= self.transfer_time + 5.0:
            self._fail(f'{transfer.action} did not complete: the joint is '
                       f"{'attached' if self.attached else 'detached'}")

    def _fail(self, why: str) -> None:
        self.get_logger().error(why)
        self.state = LoadState.FAULT
        self.transfer = None

    def _load_is_at(self, target, since: float) -> bool:
        if self.load_pose is None or self.load_heard <= since:
            return False
        x, y, z = self.load_pose
        return (math.hypot(x - target[0], y - target[1]) <= ARRIVED_WITHIN
                and abs(z - target[2]) <= ARRIVED_WITHIN)

    def _send_load(self, transfer: Transfer, now: float) -> None:
        if transfer.action == 'load':
            x, y, z, yaw = load_pose_on_vehicle(self.truth)
        else:
            x, y, z, yaw = load_pose_on_stand(transfer.station)
        transfer.target, transfer.sent_at = (x, y, z), now
        transfer.tries += 1
        if transfer.tries > 1:
            self.get_logger().warn(f'the load is not where it was sent: sending it again '
                                   f'({transfer.tries} of {MOVE_TRIES})')
        request = SetEntityPose.Request()
        request.entity.name = self.load_model
        request.entity.type = request.entity.MODEL
        request.pose.position.x, request.pose.position.y = x, y
        request.pose.position.z = z + 0.005
        request.pose.orientation.z = math.sin(yaw / 2)
        request.pose.orientation.w = math.cos(yaw / 2)
        self.set_pose.call_async(request)

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        msg = LoadStatus()
        msg.header.stamp = stamp
        msg.state = self.state.value
        self.state_pub.publish(msg)
        battery = BatteryState()
        battery.header.stamp = stamp
        battery.percentage = float(self.battery.level)
        battery.present = True
        battery.power_supply_status = (
            BatteryState.POWER_SUPPLY_STATUS_CHARGING if self.battery.charging
            else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING)
        self.battery_pub.publish(battery)


def main(args=None):
    rclpy.init(args=args)
    node = SimAgv()
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
