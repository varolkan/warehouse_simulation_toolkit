#!/usr/bin/env python3
"""Scripted Gazebo scenarios: transport orders, things put in the way, the e-stop, a cancel.

    ros2 launch agv_gazebo sim.launch.py              # in another terminal, and wait for it
    python3 tools/recorder.py results/track.csv       # optional, for tools/figures.py
    python3 tools/scenarios.py results tour crate person estop blocked cancel

Each scenario starts where the one before it left the vehicle and the tote, so they run in
this order, from a fresh simulation (vehicle on the charger, tote at 'inbound'):

  tour     six orders round every transfer station, then back to the charger
  crate    a crate appears 1.2 m ahead of the bumper at full speed, and stays
  person   a person steps in 0.9 m ahead at full speed, and leaves after 8 s
  estop    the e-stop is pressed at full speed, released and reset 8 s later
  blocked  a pallet closes the only way into rack aisle A; the order to rack_a fails with
           the load on board; the pallet is taken away and the load is delivered
  cancel   an order is cancelled 4 m into its delivery leg: the vehicle takes the load
           to its charger; the next order (no pickup) delivers it

Writes OUTDIR/<scenario>_events.csv (what happened when, in simulated time) and prints a
line per event. Obstacles are Gazebo models created and removed through ros_gz_sim; the
e-stop and the reset go through the stack's own topic and service.
"""

import csv
import math
import os
import subprocess
import sys
import time

from agv_core.geometry import Pose2D
from agv_core.sites import Site
from agv_core.vehicle import AGV
from agv_interfaces.action import Transport
from agv_interfaces.msg import MissionStatus, SafetyStatus
from agv_robot.paths import site_file
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

WORLD = 'warehouse'
CRATE_HALF, PERSON_RADIUS = 0.20, 0.22


class Runner:

    def __init__(self, outdir):
        self.outdir = outdir
        self.node = Node('scenarios', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.client = ActionClient(self.node, Transport, 'transport')
        self.estop_pub = self.node.create_publisher(Bool, 'estop', 10)
        self.reset = self.node.create_client(Trigger, 'safety/reset')
        self.site = Site.load(site_file('warehouse'))
        self.truth, self.speed, self.status, self.safety, self.handle = None, 0.0, None, None, None
        self.events = []
        self.node.create_subscription(Odometry, 'ground_truth/odometry', self._on_truth, 10)
        self.node.create_subscription(MissionStatus, 'mission/status',
                                      lambda m: setattr(self, 'status', m), 10)
        self.node.create_subscription(SafetyStatus, 'safety/status',
                                      lambda m: setattr(self, 'safety', m), 10)

    # -- plumbing ------------------------------------------------------------------------

    def _on_truth(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.truth = Pose2D(p.x, p.y, yaw)
        self.speed = msg.twist.twist.linear.x

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def spin_until(self, done, timeout):
        end = time.time() + timeout
        while time.time() < end and not done():
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return done()

    def event(self, what, **data):
        t = self.truth
        row = dict(sim_t=round(self.now(), 2), event=what, x=round(t.x, 3), y=round(t.y, 3),
                   yaw=round(t.yaw, 4), v=round(self.speed, 3),
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()})
        self.events.append(row)
        print(f'    [{row["sim_t"]:8.1f}] {what} {data if data else ""}', flush=True)

    def write(self, name):
        keys = []
        for row in self.events:
            keys += [k for k in row if k not in keys]
        with open(os.path.join(self.outdir, f'{name}_events.csv'), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.events)
        self.events = []

    def spawn(self, name, sdf, x, y, yaw=0.0):
        subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD, '-name', name,
                        '-string', sdf, '-x', str(x), '-y', str(y), '-z', '0.0', '-Y', str(yaw)],
                       capture_output=True, timeout=30, check=True)

    def remove(self, name):
        # This tool takes its target as parameters; it ignores -world and -name.
        subprocess.run(['ros2', 'run', 'ros_gz_sim', 'remove', '--ros-args',
                        '-p', f'world:={WORLD}', '-p', f'entity_name:={name}'],
                       capture_output=True, timeout=30, check=True)

    def dock_error(self, station):
        local = self.truth.relative_to(self.site.station(station).pose)
        return local.x * 100, local.y * 100, math.degrees(local.yaw)

    def send(self, pickup, dropoff, ret=False, during=None):
        goal = Transport.Goal(pickup=pickup, dropoff=dropoff, return_to_charger=ret)
        callback = (lambda m: during(m.feedback)) if during else None
        future = self.client.send_goal_async(goal, feedback_callback=callback)
        self.spin_until(future.done, 30)
        self.handle = future.result()
        return self.handle.get_result_async()

    def finish(self, label, result_future, timeout=1200.0, watch=None):
        """Wait for an order's result, calling `watch()` meanwhile; log it and return it."""
        end = time.time() + timeout
        while not result_future.done() and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if watch:
                watch()
        r = result_future.result().result
        self.spin(1.0)
        self.event('result', order=label, outcome=r.outcome, duration=r.duration,
                   distance=r.distance, message=r.message)
        return r

    # -- scenarios -----------------------------------------------------------------------

    def tour(self):
        legs = (('inbound', 'rack_a'), ('rack_a', 'outbound'), ('outbound', 'buffer'),
                ('buffer', 'rack_b'), ('rack_b', 'packing'), ('packing', 'inbound'))
        for i, (pickup, dropoff) in enumerate(legs):
            last = i == len(legs) - 1
            seen = set()

            def during(f, seen=seen):
                if f.state in ('loading', 'unloading') and f.state not in seen:
                    seen.add(f.state)
                    along, across, heading = self.dock_error(f.station)
                    self.event('docked', station=f.station, along_cm=along, across_cm=across,
                               heading_deg=heading)

            r = self.finish(f'{pickup}>{dropoff}', self.send(pickup, dropoff, last, during))
            if last and r.success:
                along, across, heading = self.dock_error('charger')
                self.event('docked', station='charger', along_cm=along, across_cm=across,
                           heading_deg=heading, charging=self.status.charging)

    def _obstacle_ahead(self, kind, gap, pickup, dropoff, hold):
        """Put a crate or a person `gap` ahead of the bumper once the vehicle is at speed."""
        name, half = f'{kind}_1', PERSON_RADIUS if kind == 'person' else CRATE_HALF
        state = {}

        def during(f):
            if state or f.state != 'to_dropoff' or self.speed < 0.7:
                return
            ahead = self.truth.advanced(AGV.half_length + gap + half)
            self.event(f'{kind} placed', obstacle_x=ahead.x, obstacle_y=ahead.y,
                       ahead_of_bumper=gap)
            if kind == 'person':
                sdf = cylinder_sdf(name, PERSON_RADIUS, 1.7, '0.9 0.35 0.1')
            else:
                sdf = box_sdf(name, 2 * CRATE_HALF, 2 * CRATE_HALF, 0.5, '0.55 0.35 0.15')
            self.spawn(name, sdf, ahead.x, ahead.y, self.truth.yaw)
            state['at'] = ((ahead.x, ahead.y), time.time())

        def watch():
            if not state:
                return
            if 'stopped' not in state and abs(self.speed) < 0.01:
                local = self.truth.to_local(state['at'][0])
                self.event('stopped', bumper_to_obstacle=local[0] - AGV.half_length - half,
                           safety=self.safety.state, reason=self.safety.reason)
                state['stopped'] = True
            if 'removed' not in state and time.time() - state['at'][1] > hold:
                self.remove(name)
                self.event(f'{kind} removed')
                state['removed'] = True

        self.finish(f'{pickup}>{dropoff}', self.send(pickup, dropoff, during=during),
                    watch=watch)
        if state and 'removed' not in state:
            self.remove(name)

    def crate(self):
        self._obstacle_ahead('crate', 1.2, 'inbound', 'packing', hold=math.inf)

    def person(self):
        self._obstacle_ahead('person', 0.9, 'packing', 'outbound', hold=8.0)

    def estop(self):
        state = {}

        def during(f):
            if state or f.state != 'to_dropoff' or self.speed < 0.7:
                return
            self.estop_pub.publish(Bool(data=True))
            state['pressed'] = (self.truth, time.time())
            self.event('e-stop pressed')

        def watch():
            if 'pressed' in state and 'stopped' not in state and abs(self.speed) < 0.005:
                self.event('stopped', stopping_distance=state['pressed'][0].distance_to(
                    self.truth))
                state['stopped'] = True
            if 'stopped' in state and 'reset' not in state \
                    and time.time() - state['pressed'][1] > 8.0:
                self.estop_pub.publish(Bool(data=False))
                self.spin(0.5)
                self.reset.wait_for_service(timeout_sec=5)
                future = self.reset.call_async(Trigger.Request())
                self.spin_until(future.done, 5)
                self.event('e-stop released and reset', ok=future.result().success)
                state['reset'] = True

        self.finish('outbound>buffer', self.send('outbound', 'buffer', during=during),
                    watch=watch)

    def blocked(self):
        # A pallet across the mouth of rack aisle A, between racks 5 and 4: the only way in.
        self.spawn('pallet_1', box_sdf('pallet_1', 1.3, 1.0, 0.6, '0.6 0.45 0.25'), 3.82, -5.0)
        self.event('pallet placed in aisle A', obstacle_x=3.82, obstacle_y=-5.0)
        self.finish('buffer>rack_a', self.send('buffer', 'rack_a'))
        self.remove('pallet_1')
        self.event('pallet removed')
        self.finish('on board>rack_a', self.send('', 'rack_a'))

    def cancel(self):
        state = {}

        def during(f):
            if state or f.state != 'to_dropoff' or f.distance_driven < 4.0:
                return
            self.handle.cancel_goal_async()
            state['sent'] = True
            self.event('cancel sent')

        self.finish('rack_a>inbound', self.send('rack_a', 'inbound', during=during))
        along, across, heading = self.dock_error('charger')
        self.event('docked', station='charger', along_cm=along, across_cm=across,
                   heading_deg=heading, charging=self.status.charging, load=self.status.load)
        self.finish('on board>inbound', self.send('', 'inbound'))


def box_sdf(name, sx, sy, sz, rgb):
    geometry = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    return _static_model(name, sz / 2, geometry, rgb)


def cylinder_sdf(name, radius, height, rgb):
    geometry = (f'<geometry><cylinder><radius>{radius}</radius><length>{height}</length>'
                '</cylinder></geometry>')
    return _static_model(name, height / 2, geometry, rgb)


def _static_model(name, z, geometry, rgb):
    pose = f'<pose>0 0 {z} 0 0 0</pose>'
    return (f'<sdf version="1.9"><model name="{name}"><static>true</static><link name="l">'
            f'<collision name="c">{pose}{geometry}</collision>'
            f'<visual name="v">{pose}{geometry}<material><ambient>{rgb} 1</ambient>'
            f'<diffuse>{rgb} 1</diffuse></material></visual></link></model></sdf>')


SCENARIOS = ('tour', 'crate', 'person', 'estop', 'blocked', 'cancel')


def main():
    if len(sys.argv) < 3 or any(name not in SCENARIOS for name in sys.argv[2:]):
        sys.exit(__doc__)
    os.makedirs(sys.argv[1], exist_ok=True)
    rclpy.init()
    runner = Runner(sys.argv[1])
    if not runner.client.wait_for_server(timeout_sec=120):
        sys.exit('no transport action server: is the simulation running?')
    runner.spin_until(lambda: runner.truth is not None and runner.status is not None, 60)
    for name in sys.argv[2:]:
        print(f'== {name}', flush=True)
        getattr(runner, name)()
        runner.write(name)
    runner.node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
