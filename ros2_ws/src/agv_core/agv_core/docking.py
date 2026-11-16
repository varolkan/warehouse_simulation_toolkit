"""The last metre into a station.

Nav2 brings the vehicle to a station's approach pose to within about 10 cm and 6 degrees:
good enough to travel, not to stand in front of a transfer stand or on charger contacts.
From there this controller drives the docking run. It works in the station's own frame
(x along the run, the docked pose at the origin):

  align     turn on the spot until the vehicle points at the run's centre line a little
            way ahead;
  approach  drive, steering at a point on the centre line `lookahead` ahead of the
            vehicle, so a sideways error is driven out along the run rather than chased
            at the end, slowing as the stop line comes up;
  settle    stopped on the line, turn to the station's heading.

It only ever drives forward. If the vehicle ends up further off the centre line than
`position_tolerance` the dock has failed, and the mission takes it back to the approach
pose for another try rather than shuffle in front of the station.

Every command goes through the safety gate like any other, which is why the run is slow:
the stand ahead is inside the warning field all the way in.
"""

from dataclasses import dataclass
import enum
import math
from typing import Optional

from agv_core.geometry import angle_diff, Pose2D


@dataclass(frozen=True)
class DockingConfig:
    """Speeds, gains and tolerances of the docking run. Metres, seconds, radians."""

    max_speed: float = 0.20
    min_speed: float = 0.04              # the crawl over the last centimetres
    speed_gain: float = 0.6              # m/s per metre still to go
    lookahead: float = 0.35              # how far ahead on the centre line it steers at
    heading_gain: float = 2.0            # rad/s per radian of heading error
    max_turn_rate: float = 0.6
    min_turn_rate: float = 0.08          # below this a turn on the spot does not move the drive

    align_tolerance: float = math.radians(12.0)    # worse than this: stop and turn first
    position_tolerance: float = 0.02               # off the centre line when docked
    heading_tolerance: float = math.radians(1.5)   # off the station's heading when docked

    # Where a run may start: the approach pose, give or take what Nav2 delivers.
    max_start_distance: float = 2.5
    min_start_distance: float = 0.25
    max_start_offset: float = 0.25
    max_overshoot: float = 0.05                    # past the stop line: something is wrong


class DockingPhase(enum.Enum):
    ALIGN = 'align'
    APPROACH = 'approach'
    SETTLE = 'settle'
    DONE = 'done'
    FAILED = 'failed'


@dataclass(frozen=True)
class DockingCommand:
    """One cycle of the controller."""

    v: float
    w: float
    phase: DockingPhase
    remaining: float          # metres to the stop line along the run (negative: past it)
    offset: float             # metres off the centre line (positive: to its left)
    heading_error: float      # radians off the station's heading
    reason: str = ''

    @property
    def done(self) -> bool:
        return self.phase is DockingPhase.DONE

    @property
    def failed(self) -> bool:
        return self.phase is DockingPhase.FAILED


class DockingController:
    """Drives one docking run. Make a new one for each attempt."""

    def __init__(self, target: Pose2D, config: DockingConfig = DockingConfig()):
        self.target = target
        self.config = config
        self.phase: Optional[DockingPhase] = None

    def update(self, pose: Pose2D) -> DockingCommand:
        """Return the command for the vehicle's current pose (map frame)."""
        cfg = self.config
        local = pose.relative_to(self.target)
        remaining, offset, heading = -local.x, local.y, local.yaw

        if self.phase is None:
            problem = self._start_problem(remaining, offset)
            if problem:
                return self._end(DockingPhase.FAILED, remaining, offset, heading, problem)
            self.phase = DockingPhase.ALIGN
        if self.phase in (DockingPhase.DONE, DockingPhase.FAILED):
            return DockingCommand(0.0, 0.0, self.phase, remaining, offset, heading)

        if self.phase is not DockingPhase.SETTLE:
            if remaining <= 0.0:
                if remaining < -cfg.max_overshoot:
                    return self._end(
                        DockingPhase.FAILED, remaining, offset, heading,
                        f'{-remaining:.2f} m past the stop line')
                self.phase = DockingPhase.SETTLE
            else:
                # Aim at the centre line `lookahead` ahead, or at the stop line if nearer.
                aim = math.atan2(-offset, min(cfg.lookahead, max(remaining, 0.05)))
                error = angle_diff(aim, heading)
                if self.phase is DockingPhase.ALIGN:
                    if abs(error) > cfg.align_tolerance / 2.0:
                        return DockingCommand(0.0, self._turn(error), self.phase, remaining,
                                              offset, heading)
                    self.phase = DockingPhase.APPROACH
                elif abs(error) > cfg.align_tolerance:
                    self.phase = DockingPhase.ALIGN
                    return DockingCommand(0.0, self._turn(error), self.phase, remaining,
                                          offset, heading)
                speed = max(cfg.min_speed, min(cfg.max_speed, cfg.speed_gain * remaining))
                turn = max(-cfg.max_turn_rate, min(cfg.max_turn_rate, cfg.heading_gain * error))
                return DockingCommand(speed, turn, self.phase, remaining, offset, heading)

        # Settle: on the line, turn to the station's heading.
        if abs(heading) > cfg.heading_tolerance:
            return DockingCommand(
                0.0, self._turn(-heading), self.phase, remaining, offset, heading)
        if abs(offset) > cfg.position_tolerance:
            return self._end(DockingPhase.FAILED, remaining, offset, heading,
                             f'{abs(offset):.3f} m off the centre line')
        return self._end(DockingPhase.DONE, remaining, offset, heading, '')

    def _start_problem(self, remaining: float, offset: float) -> str:
        cfg = self.config
        if remaining > cfg.max_start_distance:
            return f'{remaining:.2f} m from the station: too far to start the docking run'
        if remaining < cfg.min_start_distance:
            return f'{remaining:.2f} m from the stop line: too close to start the docking run'
        if abs(offset) > cfg.max_start_offset:
            return f'{abs(offset):.2f} m off the centre line: too far to start the docking run'
        return ''

    def _turn(self, error: float) -> float:
        cfg = self.config
        rate = max(cfg.min_turn_rate, min(cfg.max_turn_rate, cfg.heading_gain * abs(error)))
        return math.copysign(rate, error)

    def _end(self, phase: DockingPhase, remaining: float, offset: float, heading: float,
             reason: str) -> DockingCommand:
        self.phase = phase
        return DockingCommand(0.0, 0.0, phase, remaining, offset, heading, reason)


def is_docked(pose: Pose2D, station_pose: Pose2D, position: float = 0.05,
              heading: float = math.radians(4.0)) -> bool:
    """Whether a vehicle at `pose` is standing docked at a station."""
    return (pose.distance_to(station_pose) <= position
            and abs(angle_diff(pose.yaw, station_pose.yaw)) <= heading)
