"""The parts of the vehicle Gazebo does not simulate: the load handler and the battery.

Load handler (load/command, load/state). A transfer takes `transfer_time` seconds and is
only done docked at a transfer station: loading takes the tote off that station's stand
onto the deck, where Gazebo's DetachableJoint holds it; unloading lets go and puts it on
the stand. How the load moves between stand and deck (rollers, a lift) is not modelled:
the tote is set down where it goes, half way through the transfer.

Battery (battery_state). agv_core's BatterySim, driven by the vehicle's true pose from
Gazebo: it charges only while the vehicle stands docked on a charger's contacts.

Through ros_gz_bridge (agv_gazebo/config/bridge.yaml): ground_truth/odometry, the
attach/detach topics and joint state of the load, and the world's set_pose service.
"""

import math

from agv_core.geometry import Pose2D
from agv_core.simulation import BatterySim, LoadState
from agv_core.sites import Site
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
        self.load_station = load_at or None     # where the load stands, if on a stand
        # With a load in the world, nothing is known until Gazebo says the joint is open:
        # its DetachableJoint closes by itself the moment both models exist. Without one,
        # the vehicle has no joint and an empty deck.
        self.state = LoadState.UNKNOWN if self.load_station else LoadState.EMPTY
        self.transfer = None            # (action, station, started)
        self.battery = BatterySim(level=battery, chargers=[c.pose for c in self.site.chargers])
        self.last_battery_step = None

        self.create_subscription(Odometry, 'ground_truth/odometry', self._on_truth, 10)
        self.create_subscription(String, '/agv/load/joint_state', self._on_joint, 10)
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
        self.transfer = (action, station, self.now(), False)
        self.state = LoadState.MOVING
        if action == 'unload':
            self.detach_pub.publish(Empty())
        response.accepted, response.message = True, ''
        self.get_logger().info(f"{action} at '{station.name}'")
        return response

    def _step(self) -> None:
        now = self.now()
        if self.state is LoadState.UNKNOWN and self.load_station:
            if self.attached is False:
                self.state = LoadState.EMPTY
                self.get_logger().info(f"ready: the load is on the stand at '{self.load_station}'")
            elif self.attached:
                # DetachableJoint attaches by itself as soon as both models exist: let go.
                self.detach_pub.publish(Empty())
        if self.transfer is not None:
            action, station, started, moved = self.transfer
            if not moved and now - started >= self.transfer_time / 2.0:
                self._move_load(action, station)
                self.transfer = (action, station, started, True)
            elif moved and now - started >= self.transfer_time:
                if action == 'load' and self.attached:
                    self.state, self.load_station = LoadState.LOADED, None
                    self.transfer = None
                elif action == 'unload' and self.attached is False:
                    self.state, self.load_station = LoadState.EMPTY, station.name
                    self.transfer = None
                elif now - started >= self.transfer_time + 5.0:
                    self.get_logger().error(f'{action} did not complete: the joint is '
                                            f"{'attached' if self.attached else 'detached'}")
                    self.state = LoadState.FAULT
                    self.transfer = None
        if self.truth is not None:
            dt = 0.0 if self.last_battery_step is None else now - self.last_battery_step
            self.battery.step(max(0.0, dt), self.truth, self.distance, self.speed)
            self.distance = 0.0
            self.last_battery_step = now

    def _move_load(self, action: str, station) -> None:
        if action == 'load':
            x, y, z, yaw = load_pose_on_vehicle(self.truth)
        else:
            x, y, z, yaw = load_pose_on_stand(station)
        request = SetEntityPose.Request()
        request.entity.name = self.load_model
        request.entity.type = request.entity.MODEL
        request.pose.position.x, request.pose.position.y = x, y
        request.pose.position.z = z + 0.005
        request.pose.orientation.z = math.sin(yaw / 2)
        request.pose.orientation.w = math.cos(yaw / 2)
        future = self.set_pose.call_async(request)
        if action == 'load':
            future.add_done_callback(lambda f: self.attach_pub.publish(Empty()))

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
