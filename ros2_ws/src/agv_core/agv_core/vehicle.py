"""The vehicle's dimensions and limits, in one place.

The same numbers appear in the URDF (agv_robot/urdf/agv.urdf.xacro), in Nav2's footprint
(config/nav2.yaml) and in the scan filter (config/scan_filter.yaml), because each of those
formats wants them written out. agv_robot's tests compare all three with this file.
"""

from dataclasses import dataclass
import math
from typing import Tuple

from agv_core.geometry import Point, rectangle


@dataclass(frozen=True)
class Vehicle:
    """Outline, drive geometry and motion limits. Metres, seconds, radians."""

    length: float = 0.60
    width: float = 0.44
    # Top of the load deck above the floor: transfer stands are this high.
    deck_height: float = 0.26
    # Height of the scanner's beam plane above the floor.
    lidar_height: float = 0.20
    wheel_radius: float = 0.075
    wheel_separation: float = 0.29

    # What Nav2's velocity smoother lets through (config/nav2.yaml).
    max_speed: float = 1.0
    max_reverse_speed: float = 0.3
    max_turn_rate: float = 1.5
    max_accel: float = 0.8
    max_decel: float = 1.2
    # What the drive can do when it has to: a protective stop brakes this hard.
    brake_decel: float = 2.5

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def turning_radius(self) -> float:
        """Radius of the circle the corners sweep when the vehicle turns on the spot."""
        return math.hypot(self.half_length, self.half_width)

    @property
    def footprint(self) -> Tuple[Point, ...]:
        return rectangle(self.half_length, self.half_width)


AGV = Vehicle()
