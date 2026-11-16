"""The safety gate: LiDAR fields, e-stop and watchdogs between every command and the drive.

Whatever wants to move the vehicle (Nav2, the docking controller, an operator with a
joystick) hands its velocity command to a SafetyMonitor, and the drive gets what comes
out. What comes out is the command, unless:

  e-stop       pressed: nothing moves until it is released and then reset by hand.
  fault        the scan or the odometry is stale: a vehicle that cannot see or cannot feel
               itself move stands still.
  protective   something is inside the protective field, the part of the vehicle's path it
               could only just stop within from the speed it is doing now. The command
               becomes zero at once, and stays zero until the field of the motion it is
               asked to resume has been clear for `restart_delay` (ISO 3691-4 asks for a
               delay before an automatic restart; 2 s here).
  slow         something is further along the path. The speed is cut to the one from which
               the vehicle would stop comfortably `standoff` short of it, so it arrives at
               an obstacle slowly instead of braking hard in front of it.

The fields are not fixed shapes. They follow the command: a rectangle ahead when driving
straight, a curved corridor when driving an arc (with the extra width the outer front
corner sweeps), behind the vehicle when reversing, and the turning circle when turning on
the spot. Their length is a stopping distance, reaction time x speed + speed^2 / 2a, so a
vehicle creeping into a station is not stopped by the stand it is driving up to.

This is a model of a safety function, not one. On a real vehicle the protective field
belongs in a safety-rated scanner wired to the drive's safe-torque-off, and this gate is
the layer above it.
"""

import dataclasses
from dataclasses import dataclass, field
import enum
import math
from typing import Optional, Tuple

from agv_core.geometry import Point, Pose2D
from agv_core.vehicle import AGV, Vehicle
import numpy as np


class SafetyState(enum.Enum):
    OK = 'ok'
    SLOW = 'slow'
    PROTECTIVE_STOP = 'protective_stop'
    ESTOP = 'estop'
    FAULT = 'fault'

    @property
    def stopped(self) -> bool:
        """Whether the gate is holding the vehicle still whatever it is asked to do."""
        return self in (SafetyState.PROTECTIVE_STOP, SafetyState.ESTOP, SafetyState.FAULT)


@dataclass(frozen=True)
class SafetyConfig:
    """Field geometry, speed caps and timeouts. Metres, seconds, radians."""

    # Protective field: stop from the speed the vehicle is doing now.
    reaction_time: float = 0.25    # one scan period, the gate's cycle, the drive's lag
    brake_decel: float = 1.5       # assumed when braking hard; the drive manages 2.5
    turn_decel: float = 3.0        # rad/s^2, the same for turning
    standoff: float = 0.08         # the field never ends closer than this to the vehicle
    side_margin: float = 0.05      # added to each side of the outline
    turn_margin: float = 0.03      # added round the outline when turning on the spot
    turn_standoff: float = 0.05    # rad left between a corner and what it would hit

    # Warning: slow down early enough to stop comfortably.
    warning_time: float = 1.0      # extra reaction time the slow-down plans with
    comfort_decel: float = 1.0     # and the gentler braking it plans with
    creep_speed: float = 0.03      # below this a limited speed becomes a stop

    # Returns needed inside a field before it counts: one stray point is not an obstacle.
    min_points: int = 2
    # How far the fields are evaluated: nothing beyond this can matter at top speed.
    horizon: float = 4.0

    max_speed: float = 1.0
    max_reverse_speed: float = 0.3
    max_turn_rate: float = 1.5
    manual_max_speed: float = 0.5
    manual_max_turn_rate: float = 1.0

    scan_timeout: float = 0.35     # three scans missed at 10 Hz
    odom_timeout: float = 0.5
    command_timeout: float = 0.5   # an old command is no command
    manual_timeout: float = 0.5    # how long the operator keeps control after a command
    restart_delay: float = 2.0     # clear time before moving again after a protective stop

    # The station field set, used while the docking controller drives: the vehicle is
    # meant to come up close to a stand there, slowly. Everything else stays as it is.
    dock_standoff: float = 0.02
    dock_warning_time: float = 0.3
    dock_creep_speed: float = 0.02
    dock_max_speed: float = 0.25

    def docking(self) -> 'SafetyConfig':
        """Return this configuration with the station field set in place."""
        return dataclasses.replace(
            self, standoff=self.dock_standoff, warning_time=self.dock_warning_time,
            creep_speed=self.dock_creep_speed, max_speed=min(self.max_speed, self.dock_max_speed),
            max_reverse_speed=min(self.max_reverse_speed, self.dock_max_speed))

    def stopping_distance(self, speed: float) -> float:
        speed = abs(speed)
        return speed * self.reaction_time + speed * speed / (2.0 * self.brake_decel)

    def stopping_angle(self, rate: float) -> float:
        rate = abs(rate)
        return rate * self.reaction_time + rate * rate / (2.0 * self.turn_decel)

    def speed_for_gap(self, gap: float) -> float:
        """Return the speed from which the vehicle stops comfortably short of a gap."""
        room = gap - self.standoff
        if room <= 0.0:
            return 0.0
        lag = self.comfort_decel * (self.reaction_time + self.warning_time)
        return -lag + math.sqrt(lag * lag + 2.0 * self.comfort_decel * room)

    def turn_rate_for_gap(self, gap: float) -> float:
        room = gap - self.turn_standoff
        if room <= 0.0:
            return 0.0
        lag = self.turn_decel * (self.reaction_time + self.warning_time)
        return -lag + math.sqrt(lag * lag + 2.0 * self.turn_decel * room)


@dataclass(frozen=True)
class Scan:
    """One LiDAR scan as the gate needs it: points in the vehicle frame, and its age."""

    stamp: float
    points: np.ndarray            # (N, 2), returns that are valid measurements
    too_close: int = 0            # returns nearer than the scanner can measure

    @classmethod
    def from_ranges(cls, stamp: float, ranges, angle_min: float, angle_increment: float,
                    range_min: float, range_max: float,
                    sensor: Pose2D = Pose2D()) -> 'Scan':
        """Build from a LaserScan. NaN is "no data", +inf "nothing there", -inf "too close"."""
        ranges = np.asarray(ranges, dtype=float)
        angles = angle_min + angle_increment * np.arange(ranges.size) + sensor.yaw
        too_close = np.isneginf(ranges) | (np.isfinite(ranges) & (ranges < range_min))
        valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
        points = np.column_stack((sensor.x + ranges[valid] * np.cos(angles[valid]),
                                  sensor.y + ranges[valid] * np.sin(angles[valid])))
        return cls(stamp, points, int(too_close.sum()))


@dataclass(frozen=True)
class Command:
    """A velocity command and when it was given."""

    stamp: float
    v: float = 0.0
    w: float = 0.0

    def fresh(self, now: float, timeout: float) -> bool:
        return 0.0 <= now - self.stamp <= timeout


@dataclass(frozen=True)
class SafetyStatus:
    """What the gate decided in one cycle."""

    state: SafetyState
    reason: str
    source: str                       # 'nav', 'manual' or 'none': whose command it is
    v: float                          # what goes to the drive
    w: float
    v_requested: float
    w_requested: float
    gap: Optional[float] = None       # clear path ahead of the outline, None if beyond the horizon
    protective_field: Tuple[Point, ...] = field(default_factory=tuple)
    warning_field: Tuple[Point, ...] = field(default_factory=tuple)

    @property
    def limited(self) -> bool:
        return abs(self.v - self.v_requested) > 1e-6 or abs(self.w - self.w_requested) > 1e-6


_V_EPS = 0.01      # m/s below which a command has no forward or backward part
_W_EPS = 0.01      # rad/s below which it has no turning part
_STRAIGHT_RADIUS = 100.0


class SafetyMonitor:
    """The gate. Feed it every cycle with update(); it remembers e-stop and stops."""

    def __init__(self, config: SafetyConfig = SafetyConfig(), vehicle: Vehicle = AGV):
        self.config = config
        self.vehicle = vehicle
        self._estop_pressed = False
        self._estop_latched = False
        self._estop_reason = ''
        self._protective_since: Optional[float] = None    # when the last violation was seen
        self._protective_reason = ''
        self._last_v = 0.0
        self._last_w = 0.0
        self._outline_radius = _outline_radius_table(vehicle, config.turn_margin)
        self._docking_config = config.docking()

    # -- e-stop ----------------------------------------------------------------------

    @property
    def estop_active(self) -> bool:
        return self._estop_latched

    @property
    def estop_pressed(self) -> bool:
        return self._estop_pressed

    def press_estop(self, reason: str = 'e-stop pressed') -> None:
        self._estop_pressed = True
        self._estop_latched = True
        self._estop_reason = reason

    def release_estop(self) -> None:
        """Release the button. The stop itself stays until reset()."""
        self._estop_pressed = False

    def reset(self) -> Tuple[bool, str]:
        """Clear a latched e-stop. Refused while the button is still pressed."""
        if not self._estop_latched:
            return True, 'nothing to reset'
        if self._estop_pressed:
            return False, 'the e-stop is still pressed: release it first'
        self._estop_latched = False
        self._estop_reason = ''
        return True, 'e-stop reset'

    # -- the cycle ---------------------------------------------------------------------

    def update(self, now: float, scan: Optional[Scan], measured: Optional[Command],
               nav: Optional[Command] = None, manual: Optional[Command] = None,
               docking: bool = False) -> SafetyStatus:
        """Decide what the drive gets.

        `measured` is the odometry's velocity; `nav` and `manual` are the latest commands
        from the navigation stack and from an operator. A fresh manual command wins.
        With `docking`, `nav` comes from the docking controller and the station field set
        applies (see SafetyConfig.docking).
        """
        if not docking:
            return self._update(now, scan, measured, nav, manual)
        normal = self.config
        self.config = self._docking_config
        try:
            return self._update(now, scan, measured, nav, manual)
        finally:
            self.config = normal

    def _update(self, now: float, scan: Optional[Scan], measured: Optional[Command],
                nav: Optional[Command], manual: Optional[Command]) -> SafetyStatus:
        cfg = self.config
        source, v_req, w_req = self._arbitrate(now, nav, manual)

        if self._estop_latched:
            return self._stop(SafetyState.ESTOP, self._estop_reason, source, v_req, w_req)
        if scan is None:
            return self._stop(SafetyState.FAULT, 'no scan received', source, v_req, w_req)
        if now - scan.stamp > cfg.scan_timeout:
            return self._stop(SafetyState.FAULT, f'scan is {now - scan.stamp:.2f} s old',
                              source, v_req, w_req)
        if measured is None:
            return self._stop(SafetyState.FAULT, 'no odometry received', source, v_req, w_req)
        if now - measured.stamp > cfg.odom_timeout:
            return self._stop(SafetyState.FAULT,
                              f'odometry is {now - measured.stamp:.2f} s old',
                              source, v_req, w_req)

        # The speed the fields are sized for: what the vehicle is doing, or what it was
        # last told to do if that is more.
        v_now = _larger(measured.v, self._last_v)
        w_now = _larger(measured.w, self._last_w)

        violation = None
        if scan.too_close >= cfg.min_points:
            violation = 'something is closer than the scanner can measure'

        moving = abs(v_now) > _V_EPS or abs(w_now) > _W_EPS
        protective_field: Tuple[Point, ...] = ()
        if violation is None and moving:
            reach, kind = self._reach(scan.points, v_now, w_now)
            protective_field = self._field_outline(
                v_now, w_now, self._protective_length(v_now, w_now))
            if reach is not None and reach < self._protective_length(v_now, w_now):
                violation = _describe(kind, reach, v_now)

        # What the command would run into, whether or not the vehicle is moving yet.
        gap, kind = (None, 'none')
        wants_motion = abs(v_req) > _V_EPS or abs(w_req) > _W_EPS
        if wants_motion:
            gap, kind = self._reach(scan.points, v_req, w_req)
        warning_field = self._field_outline(v_req, w_req, self._warning_length(v_req, w_req)) \
            if wants_motion else ()

        if violation is None and wants_motion and gap is not None:
            floor = cfg.turn_standoff if kind == 'turn' else cfg.standoff
            if self._protective_since is not None:
                # Held after a protective stop: the field that has to be clear is the one
                # of the motion being asked for, not the vehicle's while it stands still.
                # Otherwise the wait can run out while whoever stopped it is still there.
                floor = self._protective_length(v_req, w_req)
            if gap < floor:
                # Asked to drive into something it could not stop short of.
                violation = _describe(kind, gap, v_req)

        if violation is not None:
            self._protective_since = now
            self._protective_reason = violation
        if self._protective_since is not None:
            clear_for = now - self._protective_since
            if violation is not None or clear_for < cfg.restart_delay:
                reason = self._protective_reason
                if violation is None:
                    reason += f' (clear for {clear_for:.1f} s of {cfg.restart_delay:.1f} s)'
                return self._stop(SafetyState.PROTECTIVE_STOP, reason, source, v_req, w_req,
                                  gap, protective_field, warning_field)
            self._protective_since = None
            self._protective_reason = ''

        # Slow down for what is ahead.
        v_out, w_out = v_req, w_req
        state, reason = SafetyState.OK, ''
        if wants_motion and gap is not None:
            if kind == 'turn':
                allowed = cfg.turn_rate_for_gap(gap)
                if abs(w_req) > allowed:
                    scale = allowed / abs(w_req)
                    v_out, w_out = v_req * scale, w_req * scale
                    state = SafetyState.SLOW
                    reason = f'turning slowly: {math.degrees(gap):.0f} degrees of room'
            else:
                allowed = cfg.speed_for_gap(gap)
                if allowed < cfg.creep_speed:
                    allowed = 0.0
                if abs(v_req) > allowed:
                    scale = allowed / abs(v_req)
                    v_out, w_out = v_req * scale, w_req * scale
                    state = SafetyState.SLOW
                    reason = f'obstacle {gap:.2f} m {_direction(v_req)}'
        self._last_v, self._last_w = v_out, w_out
        return SafetyStatus(state, reason, source, v_out, w_out, v_req, w_req, gap,
                            protective_field, warning_field)

    # -- pieces ------------------------------------------------------------------------

    def _arbitrate(self, now: float, nav: Optional[Command],
                   manual: Optional[Command]) -> Tuple[str, float, float]:
        cfg = self.config
        if manual is not None and manual.fresh(now, cfg.manual_timeout):
            return ('manual',
                    _clamp(manual.v, -cfg.max_reverse_speed, cfg.manual_max_speed),
                    _clamp(manual.w, -cfg.manual_max_turn_rate, cfg.manual_max_turn_rate))
        if nav is not None and nav.fresh(now, cfg.command_timeout):
            return ('nav',
                    _clamp(nav.v, -cfg.max_reverse_speed, cfg.max_speed),
                    _clamp(nav.w, -cfg.max_turn_rate, cfg.max_turn_rate))
        return ('none', 0.0, 0.0)

    def _stop(self, state: SafetyState, reason: str, source: str, v_req: float, w_req: float,
              gap: Optional[float] = None, protective_field: Tuple[Point, ...] = (),
              warning_field: Tuple[Point, ...] = ()) -> SafetyStatus:
        self._last_v, self._last_w = 0.0, 0.0
        return SafetyStatus(state, reason, source, 0.0, 0.0, v_req, w_req, gap,
                            protective_field, warning_field)

    def _turning_on_the_spot(self, v: float, w: float) -> bool:
        if abs(w) <= _W_EPS:
            return False
        # The turning centre lies inside the vehicle: its corners swing, it goes nowhere.
        return abs(v) <= _V_EPS or abs(v / w) < self.vehicle.half_width

    def _protective_length(self, v: float, w: float) -> float:
        cfg = self.config
        if self._turning_on_the_spot(v, w):
            return cfg.turn_standoff + cfg.stopping_angle(w)
        return cfg.standoff + cfg.stopping_distance(v)

    def _warning_length(self, v: float, w: float) -> float:
        cfg = self.config
        if self._turning_on_the_spot(v, w):
            return cfg.turn_standoff + cfg.stopping_angle(w) + abs(w) * cfg.warning_time
        v = abs(v)
        return (cfg.standoff + v * (cfg.reaction_time + cfg.warning_time)
                + v * v / (2.0 * cfg.comfort_decel))

    def _reach(self, points: np.ndarray, v: float, w: float) -> Tuple[Optional[float], str]:
        """How far the vehicle can follow (v, w) before its outline meets a return.

        Metres along the path for a drive ('path'), radians for a turn on the spot
        ('turn'). None when nothing is in the way within the horizon.
        """
        if points.size == 0:
            return None, 'none'
        if self._turning_on_the_spot(v, w):
            return self._turn_reach(points, w), 'turn'
        return self._path_reach(points, v, w), 'path'

    def _path_reach(self, points: np.ndarray, v: float, w: float) -> Optional[float]:
        cfg, veh = self.config, self.vehicle
        along, across, outer_extra = _path_coordinates(points, v, w, veh)
        inside = ((across <= veh.half_width + cfg.side_margin + outer_extra)
                  & (across >= -(veh.half_width + cfg.side_margin))
                  & (along > -veh.half_length) & (along < veh.half_length + cfg.horizon))
        gaps = along[inside] - veh.half_length
        if gaps.size < cfg.min_points:
            return None
        # The min_points-th nearest: a single stray return does not set the distance.
        gap = float(np.partition(gaps, cfg.min_points - 1)[cfg.min_points - 1])
        return max(gap, 0.0)

    def _turn_reach(self, points: np.ndarray, w: float) -> Optional[float]:
        cfg, veh = self.config, self.vehicle
        ranges = np.hypot(points[:, 0], points[:, 1])
        near = ranges <= veh.turning_radius + cfg.turn_margin
        if int(near.sum()) < cfg.min_points:
            return None
        bearings = np.arctan2(points[near, 1], points[near, 0])
        direction = 1.0 if w > 0.0 else -1.0
        table = self._outline_radius
        steps = table.size
        # Turning by b moves a fixed point to bearing (its bearing - b) in the vehicle's
        # frame. Find the smallest b at which the outline there reaches the point.
        candidates = np.arange(0, steps // 2 + 1)
        index = (np.rint(np.degrees(bearings)[:, None] - direction * candidates[None, :])
                 .astype(int) % steps)
        hits = table[index] >= ranges[near][:, None]
        first = np.where(hits.any(axis=1), hits.argmax(axis=1), steps)
        first = np.sort(first)
        angle = int(first[cfg.min_points - 1])
        if angle >= steps:
            return None
        return math.radians(angle)

    def _field_outline(self, v: float, w: float, length: float) -> Tuple[Point, ...]:
        """Return the field as a polygon in the vehicle frame, for showing it."""
        cfg, veh = self.config, self.vehicle
        if self._turning_on_the_spot(v, w):
            radius = veh.turning_radius + cfg.turn_margin
            return tuple((radius * math.cos(a), radius * math.sin(a))
                         for a in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False))
        half = veh.half_width + cfg.side_margin
        sign = 1.0 if v >= 0.0 else -1.0
        start, end = -veh.half_length, veh.half_length + length
        if abs(w) <= _W_EPS or abs(v / w) > _STRAIGHT_RADIUS:
            return ((sign * start, half), (sign * end, half),
                    (sign * end, -half), (sign * start, -half))
        radius = v / w
        outer = math.hypot(abs(radius) + veh.half_width, veh.half_length) - abs(radius) \
            - veh.half_width
        left, right = [], []
        for s in np.linspace(start, end, 12):
            phi = s / abs(radius) * (1.0 if w > 0.0 else -1.0) * sign
            # Centre line point and heading after travelling s along the arc.
            cx, cy = radius * math.sin(phi), radius * (1.0 - math.cos(phi))
            nx, ny = -math.sin(phi), math.cos(phi)
            wide_left = half + (outer if radius < 0.0 else 0.0)
            wide_right = half + (outer if radius > 0.0 else 0.0)
            left.append((cx + nx * wide_left, cy + ny * wide_left))
            right.append((cx - nx * wide_right, cy - ny * wide_right))
        return tuple(left + right[::-1])


def _path_coordinates(points: np.ndarray, v: float, w: float,
                      vehicle: Vehicle) -> Tuple[np.ndarray, np.ndarray, float]:
    """Where each point lies relative to the path the command (v, w) follows.

    Returns, per point, the distance along the path at which the vehicle's centre would be
    level with it (positive in the direction of travel) and its offset across the path
    (positive on the outside of a turn; for a straight path, the absolute offset shifted
    so both sides compare alike), and how much wider than the vehicle the outside of the
    corridor is, because the outer front corner swings out in a turn.
    """
    x, y = points[:, 0], points[:, 1]
    if abs(w) <= _W_EPS or abs(v / w) > _STRAIGHT_RADIUS:
        along = x if v >= 0.0 else -x
        # Straight: both sides are alike. Fold them onto the positive side and report 0
        # extra width; the lower bound of the corridor then never bites.
        return along, np.abs(y), 0.0
    radius = v / w                       # signed: the turning centre is at (0, radius)
    dy = y - radius
    rho = np.hypot(x, dy)
    across = rho - abs(radius)
    start = math.atan2(-radius, 0.0)     # bearing of the vehicle seen from the centre
    turn = 1.0 if w > 0.0 else -1.0      # the vehicle circles the centre this way round
    phi = turn * (np.arctan2(dy, x) - start)
    phi = (phi + math.pi) % (2.0 * math.pi) - math.pi
    along = abs(radius) * phi
    outer_extra = (math.hypot(abs(radius) + vehicle.half_width, vehicle.half_length)
                   - abs(radius) - vehicle.half_width)
    return along, across, outer_extra


def _outline_radius_table(vehicle: Vehicle, margin: float) -> np.ndarray:
    """Distance from the centre to the (grown) outline, for each whole degree of bearing."""
    hl, hw = vehicle.half_length + margin, vehicle.half_width + margin
    angles = np.radians(np.arange(360))
    c, s = np.abs(np.cos(angles)), np.abs(np.sin(angles))
    with np.errstate(divide='ignore'):
        return np.minimum(np.where(c > 1e-12, hl / c, np.inf), np.where(s > 1e-12, hw / s, np.inf))


def _larger(a: float, b: float) -> float:
    return a if abs(a) >= abs(b) else b


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def _direction(v: float) -> str:
    return 'ahead' if v >= 0.0 else 'behind'


def _describe(kind: str, reach: float, v: float) -> str:
    if kind == 'turn':
        return f'obstacle in the turning circle, {math.degrees(reach):.0f} degrees away'
    return f'obstacle {reach:.2f} m {_direction(v)}'
