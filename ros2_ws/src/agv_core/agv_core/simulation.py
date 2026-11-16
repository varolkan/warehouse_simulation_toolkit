"""A stand-in for the vehicle: kinematic drive, simulated scanner, load handler.

No physics engine: a differential drive that follows velocity commands within its
acceleration limits, a scanner that casts its beams through the site's occupancy map and
whatever obstacles have been put in it, a load handler that takes a few seconds to move a
load, and a battery. It is what the unit tests drive whole transport orders against in
milliseconds, and what `fake_base_node` wraps so the complete ROS stack (AMCL, Nav2, the
safety gate, the mission) can run without Gazebo.

It reproduces what the stack depends on, including what it can do wrong on request:
odometry that drifts, a scanner that sees the vehicle's own deck posts, a load handler
that refuses commands or jams, obstacles that appear in an aisle.
"""

from dataclasses import dataclass
import enum
import math
from typing import List, Optional, Sequence, Tuple, Union

from agv_core.battery import Battery, BatteryConfig
from agv_core.geometry import normalize_angle, Point, Pose2D
from agv_core.grid import GridMap
from agv_core.vehicle import AGV, Vehicle
import numpy as np


class LoadState(enum.Enum):
    """What the load handler reports."""

    UNKNOWN = 'unknown'
    EMPTY = 'empty'
    LOADED = 'loaded'
    MOVING = 'moving'
    FAULT = 'fault'


@dataclass(frozen=True)
class DriveConfig:
    """What the drive itself can do. Comfort limits are applied upstream of it."""

    max_speed: float = 1.2
    max_reverse_speed: float = 0.5
    max_turn_rate: float = 2.0
    max_accel: float = 2.5
    max_turn_accel: float = 6.0
    command_timeout: float = 0.5       # without a fresh command the drive stops
    # Odometry errors: reported distance and turn as a multiple of the real ones.
    odom_distance_scale: float = 1.0
    odom_turn_scale: float = 1.0


class KinematicVehicle:
    """A differential drive that does what it is told, within its limits."""

    def __init__(self, pose: Pose2D = Pose2D(), config: DriveConfig = DriveConfig()):
        self.config = config
        self.pose = pose
        # Odometry starts where the vehicle is switched on and knows nothing of the map.
        self.odom = Pose2D()
        self.v = 0.0
        self.w = 0.0
        self.distance = 0.0
        self._target = (0.0, 0.0)
        self._commanded_at: Optional[float] = None

    def command(self, v: float, w: float, now: float) -> None:
        cfg = self.config
        if not (math.isfinite(v) and math.isfinite(w)):
            v, w = 0.0, 0.0
        self._target = (max(-cfg.max_reverse_speed, min(cfg.max_speed, v)),
                        max(-cfg.max_turn_rate, min(cfg.max_turn_rate, w)))
        self._commanded_at = now

    def step(self, dt: float, now: float) -> Pose2D:
        """Advance by `dt` seconds. Returns the new true pose."""
        cfg = self.config
        target = self._target
        if self._commanded_at is None or now - self._commanded_at > cfg.command_timeout:
            target = (0.0, 0.0)
        self.v = _approach(self.v, target[0], cfg.max_accel * dt)
        self.w = _approach(self.w, target[1], cfg.max_turn_accel * dt)
        moved, turned = self.v * dt, self.w * dt
        self.pose = _advance(self.pose, moved, turned)
        self.odom = _advance(self.odom, moved * cfg.odom_distance_scale,
                             turned * cfg.odom_turn_scale)
        self.distance += abs(moved)
        return self.pose

    def place(self, pose: Pose2D) -> None:
        """Put the vehicle somewhere, standing still. Odometry is not told."""
        self.pose = pose
        self.v = self.w = 0.0
        self._target = (0.0, 0.0)


def _approach(value: float, target: float, step: float) -> float:
    if target > value:
        return min(target, value + step)
    return max(target, value - step)


def _advance(pose: Pose2D, moved: float, turned: float) -> Pose2D:
    """Move along an arc: exact for constant velocity over the step."""
    if abs(turned) < 1e-9:
        return Pose2D(pose.x + moved * math.cos(pose.yaw), pose.y + moved * math.sin(pose.yaw),
                      pose.yaw)
    radius = moved / turned
    yaw = pose.yaw + turned
    return Pose2D(pose.x + radius * (math.sin(yaw) - math.sin(pose.yaw)),
                  pose.y - radius * (math.cos(yaw) - math.cos(pose.yaw)),
                  normalize_angle(yaw))


# -- obstacles that are not in the map --------------------------------------------------------


@dataclass
class Disc:
    """A round obstacle: a person, a post, a drum."""

    x: float
    y: float
    radius: float

    def ray_ranges(self, origin: Point, angles: np.ndarray) -> np.ndarray:
        ox, oy = origin[0] - self.x, origin[1] - self.y
        dx, dy = np.cos(angles), np.sin(angles)
        b = ox * dx + oy * dy
        c = ox * ox + oy * oy - self.radius * self.radius
        disc = b * b - c
        with np.errstate(invalid='ignore'):
            near = -b - np.sqrt(disc)
        return np.where((disc >= 0.0) & (near > 0.0), near, np.inf)

    def contains(self, point: Point) -> bool:
        return math.hypot(point[0] - self.x, point[1] - self.y) <= self.radius

    def outline(self, spacing: float = 0.05) -> List[Point]:
        count = max(8, int(2.0 * math.pi * self.radius / spacing))
        return [(self.x + self.radius * math.cos(a), self.y + self.radius * math.sin(a))
                for a in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)]


@dataclass
class Box:
    """A rectangular obstacle: a crate, a pallet left in an aisle."""

    x: float
    y: float
    length: float
    width: float
    yaw: float = 0.0

    def ray_ranges(self, origin: Point, angles: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ox, oy = origin[0] - self.x, origin[1] - self.y
        lx, ly = c * ox + s * oy, -s * ox + c * oy          # ray origin in the box's frame
        dx, dy = np.cos(angles - self.yaw), np.sin(angles - self.yaw)
        hx, hy = self.length / 2.0, self.width / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            tx1, tx2 = (-hx - lx) / dx, (hx - lx) / dx
            ty1, ty2 = (-hy - ly) / dy, (hy - ly) / dy
        # A ray parallel to a pair of faces only passes between them if it starts there.
        inside_x, inside_y = abs(lx) <= hx, abs(ly) <= hy
        tx_lo = np.where(dx == 0.0, -np.inf if inside_x else np.inf, np.minimum(tx1, tx2))
        tx_hi = np.where(dx == 0.0, np.inf if inside_x else -np.inf, np.maximum(tx1, tx2))
        ty_lo = np.where(dy == 0.0, -np.inf if inside_y else np.inf, np.minimum(ty1, ty2))
        ty_hi = np.where(dy == 0.0, np.inf if inside_y else -np.inf, np.maximum(ty1, ty2))
        enter, leave = np.maximum(tx_lo, ty_lo), np.minimum(tx_hi, ty_hi)
        return np.where((leave >= enter) & (enter > 0.0), enter, np.inf)

    def contains(self, point: Point) -> bool:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = point[0] - self.x, point[1] - self.y
        return (abs(c * dx + s * dy) <= self.length / 2.0
                and abs(-s * dx + c * dy) <= self.width / 2.0)

    def outline(self, spacing: float = 0.05) -> List[Point]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = self.length / 2.0, self.width / 2.0
        corners = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
        points: List[Point] = []
        for (x1, y1), (x2, y2) in zip(corners, corners[1:] + corners[:1]):
            count = max(2, int(math.hypot(x2 - x1, y2 - y1) / spacing))
            for t in np.linspace(0.0, 1.0, count, endpoint=False):
                lx, ly = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                points.append((self.x + c * lx - s * ly, self.y + s * lx + c * ly))
        return points


Obstacle = Union[Disc, Box]


# -- the scanner ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScannerConfig:
    """A 2D scanner in the middle of the vehicle. The defaults are the simulated LiDAR's."""

    beams: int = 720
    angle_min: float = -math.pi
    angle_increment: float = 2.0 * math.pi / 720.0
    range_min: float = 0.08
    range_max: float = 16.0
    noise: float = 0.01                # standard deviation of a range, metres
    # Return the four posts that carry the deck, as the real scanner in its slot does.
    deck_posts: bool = True

    @property
    def angles(self) -> np.ndarray:
        return self.angle_min + self.angle_increment * np.arange(self.beams)


class ScannerSim:
    """Casts beams through an occupancy map and the obstacles added to it."""

    def __init__(self, grid: GridMap, config: ScannerConfig = ScannerConfig(),
                 vehicle: Vehicle = AGV, seed: int = 0):
        self.grid = grid
        self.config = config
        self.obstacles: List[Obstacle] = []
        self._rng = np.random.default_rng(seed)
        inset = 0.015
        corner_x, corner_y = vehicle.half_length - inset, vehicle.half_width - inset
        self._posts = [Disc(x, y, 0.0085) for x in (corner_x, -corner_x)
                       for y in (corner_y, -corner_y)]

    def scan(self, pose: Pose2D) -> np.ndarray:
        """Ranges for one scan from `pose`: +inf where nothing is within reach."""
        cfg = self.config
        local = cfg.angles
        world = local + pose.yaw
        ranges = self.grid.ray_cast(pose.x, pose.y, world, cfg.range_max)
        for obstacle in self.obstacles:
            ranges = np.minimum(ranges, obstacle.ray_ranges((pose.x, pose.y), world))
        if cfg.deck_posts:
            for post in self._posts:
                ranges = np.minimum(ranges, post.ray_ranges((0.0, 0.0), local))
        if cfg.noise > 0.0:
            ranges = ranges + self._rng.normal(0.0, cfg.noise, ranges.shape)
        ranges = np.where(ranges > cfg.range_max, np.inf, ranges)
        # Nearer than the scanner can measure: REP 117 says -inf.
        return np.where(ranges < cfg.range_min, -np.inf, ranges)


def footprint_collides(pose: Pose2D, grid: GridMap, obstacles: Sequence[Obstacle] = (),
                       vehicle: Vehicle = AGV, spacing: float = 0.04) -> bool:
    """Whether the vehicle's outline at `pose` touches the map or an obstacle."""
    hl, hw = vehicle.half_length, vehicle.half_width
    xs = np.arange(-hl, hl + spacing / 2.0, spacing)
    ys = np.arange(-hw, hw + spacing / 2.0, spacing)
    edge = ([(x, hw) for x in xs] + [(x, -hw) for x in xs]
            + [(hl, y) for y in ys] + [(-hl, y) for y in ys])
    for local in edge:
        point = pose.to_parent(local)
        if grid.value_at(*point) >= 100:
            return True
        if any(obstacle.contains(point) for obstacle in obstacles):
            return True
    # Something small enough to be wholly inside the outline.
    for obstacle in obstacles:
        lx, ly = pose.to_local((obstacle.x, obstacle.y))
        if abs(lx) <= hl and abs(ly) <= hw:
            return True
    return False


# -- the load handler -------------------------------------------------------------------------


class LoadHandlerSim:
    """Something that puts a load on the vehicle and takes it off again, in a few seconds.

    To exercise the failure paths it can refuse its first `rejections` commands, and it
    can jam: with `stuck`, a transfer starts and never finishes.
    """

    def __init__(self, loaded: bool = False, duration: float = 3.0, rejections: int = 0,
                 stuck: bool = False):
        self.state = LoadState.LOADED if loaded else LoadState.EMPTY
        self.duration = duration
        self.rejections = rejections
        self.stuck = stuck
        self.commands: List[Tuple[float, str]] = []
        self._finishes_at: Optional[float] = None
        self._becomes = self.state

    def command(self, action: str, now: float) -> Tuple[bool, str]:
        """Start a transfer: 'load' or 'unload'. Returns whether it was accepted, and why not."""
        self.commands.append((now, action))
        if action not in ('load', 'unload'):
            return False, f"unknown load command '{action}'"
        if self.rejections > 0:
            self.rejections -= 1
            return False, 'the load handler is not ready'
        if self.state is LoadState.MOVING:
            return False, 'a transfer is already under way'
        target = LoadState.LOADED if action == 'load' else LoadState.EMPTY
        if self.state is target:
            return True, f'already {target.value}'
        self.state = LoadState.MOVING
        self._becomes = target
        self._finishes_at = None if self.stuck else now + self.duration
        return True, ''

    def step(self, now: float) -> LoadState:
        if (self.state is LoadState.MOVING and self._finishes_at is not None
                and now >= self._finishes_at):
            self.state = self._becomes
            self._finishes_at = None
        return self.state

    def drop(self) -> None:
        """Lose the load without being asked to: it fell off, or someone took it."""
        self.state = LoadState.EMPTY
        self._finishes_at = None


class BatterySim:
    """A battery that charges when the vehicle stands docked at a charger."""

    def __init__(self, config: BatteryConfig = BatteryConfig(), level: float = 1.0,
                 chargers: Sequence[Pose2D] = (), contact_distance: float = 0.06,
                 contact_angle: float = math.radians(6.0)):
        self.battery = Battery(config, level)
        self.chargers = list(chargers)
        self.contact_distance = contact_distance
        self.contact_angle = contact_angle
        self.charging = False
        # Set to make the charger dead: the vehicle docks and no current flows.
        self.charger_dead = False

    @property
    def level(self) -> float:
        return self.battery.level

    def step(self, dt: float, pose: Pose2D, distance: float, speed: float) -> float:
        on_contacts = any(
            pose.distance_to(charger) <= self.contact_distance
            and abs(normalize_angle(pose.yaw - charger.yaw)) <= self.contact_angle
            for charger in self.chargers)
        self.charging = on_contacts and abs(speed) < 0.01 and not self.charger_dead
        return self.battery.step(dt, distance, self.charging)
