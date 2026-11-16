#!/usr/bin/env python3
"""Record a Gazebo run to CSV: mission and safety status, the true and the estimated pose.

    python3 tools/recorder.py OUT.csv

One row per safety status, the gate's cycle (20 Hz): wall-clock and simulated time, the
mission's state, the gate's state and the speed it was asked for and let through, the
vehicle's true pose (Gazebo's ground_truth/odometry) and its pose as the mission sees it,
and the scan point nearest the vehicle's outline. It reads ground truth, so it is for
simulation only. tools/figures.py draws the README's figures from it.
"""

import math
import sys
import time

from agv_core.vehicle import AGV
from agv_interfaces.msg import MissionStatus, SafetyStatus
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

COLUMNS = ('wall', 'sim', 'state', 'detail', 'safety', 'reason', 'source', 'v_req', 'v',
           'gap', 'gt_x', 'gt_y', 'gt_yaw', 'est_x', 'est_y', 'est_yaw', 'near_x', 'near_y')


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def clean(text: str) -> str:
    return text.replace(',', ';').replace('\n', ' ')


class Recorder(Node):

    def __init__(self, out):
        super().__init__('recorder', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.out = out
        self.mission = self.truth = self.scan = None
        self.create_subscription(MissionStatus, 'mission/status',
                                 lambda m: setattr(self, 'mission', m), 10)
        self.create_subscription(Odometry, 'ground_truth/odometry',
                                 lambda m: setattr(self, 'truth', m), 10)
        self.create_subscription(LaserScan, 'scan', lambda m: setattr(self, 'scan', m),
                                 qos_profile_sensor_data)
        self.create_subscription(SafetyStatus, 'safety/status', self._on_safety, 10)
        out.write(','.join(COLUMNS) + '\n')

    def _nearest(self):
        """Return the scan point nearest the outline, in the vehicle frame (scanner centred)."""
        if self.scan is None:
            return math.nan, math.nan
        r = np.array(self.scan.ranges)
        a = self.scan.angle_min + self.scan.angle_increment * np.arange(r.size)
        ok = np.isfinite(r)
        x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        if not x.size:
            return math.nan, math.nan
        d = np.hypot(np.maximum(np.abs(x) - AGV.half_length, 0),
                     np.maximum(np.abs(y) - AGV.half_width, 0))
        i = int(np.argmin(d))
        return float(x[i]), float(y[i])

    def _on_safety(self, s: SafetyStatus) -> None:
        m, g = self.mission, self.truth
        gt = (math.nan,) * 3
        if g is not None:
            p = g.pose.pose.position
            gt = (p.x, p.y, yaw_of(g.pose.pose.orientation))
        est = (m.pose.x, m.pose.y, m.pose.theta) if m is not None else (math.nan,) * 3
        near = self._nearest()
        row = (f'{time.time():.3f}', f'{self.get_clock().now().nanoseconds * 1e-9:.3f}',
               m.state if m else '', clean(m.detail) if m else '', s.state, clean(s.reason),
               s.source, f'{s.linear_requested:.3f}', f'{s.linear:.3f}', f'{s.gap:.3f}',
               *(f'{v:.4f}' for v in gt), *(f'{v:.4f}' for v in est),
               *(f'{v:.3f}' for v in near))
        self.out.write(','.join(row) + '\n')
        self.out.flush()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rclpy.init()
    with open(sys.argv[1], 'w') as out:
        node = Recorder(out)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
