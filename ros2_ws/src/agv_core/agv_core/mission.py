"""Transport orders: collect a load at one station, deliver it to another.

A Mission turns an order into travel, docking and transfers, and decides what to do when
any of them goes wrong. It talks to nothing. Each cycle it is given what is known (pose,
navigation state, load handler, battery, safety gate) and answers with what it wants
(a navigation goal, a docking velocity, a load command). The ROS node around it passes
those on to Nav2 and the drive; the tests pass them to a simulated vehicle, or to nothing
at all.

An order runs as a plan of steps:

    travel to the pickup's approach pose   Nav2, retried when it gives up
    dock                                   the docking controller, re-approached on a miss
    load                                   confirmed by the load handler, re-sent if ignored
    travel, dock, unload                   the same at the dropoff
    travel, dock at the charger            if asked for, or if the battery says so

What it holds itself to:

  - Nothing moves before the order has been checked: the stations exist and are the right
    kind, the vehicle knows where it is, the gate is not stopped, the load handler agrees
    with the order about whether a load is on board, the battery is above its starting
    level, a route exists for every leg and the battery covers them all and the way to a
    charger after.
  - A load counts as collected, and as delivered, only when the load handler says so.
  - Every wait has a deadline, and every deadline stops its clock while the vehicle is
    held or stopped for an obstacle, which have deadlines of their own.
  - The e-stop, a safety fault, an operator driving and a lost position hold the order
    where it is; it carries on when they clear and fails if they do not.
  - Cancelling an order sends the vehicle to its charger rather than leave it in an aisle;
    a transfer already under way is finished first.
"""

from dataclasses import dataclass, field
import enum
import math
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from agv_core.docking import DockingConfig, DockingController, is_docked
from agv_core.geometry import angle_diff, Pose2D
from agv_core.safety import SafetyState
from agv_core.simulation import LoadState
from agv_core.sites import Site, Station


class NavState(enum.Enum):
    """What the navigation stack says about the goal it was last given."""

    IDLE = 'idle'
    ACTIVE = 'active'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELED = 'canceled'


@dataclass(frozen=True)
class Order:
    """Carry a load from `pickup` to `dropoff`.

    With no pickup the load is taken to be on the vehicle already, which is how a load is
    delivered after an order failed with it on board.
    """

    pickup: str = ''
    dropoff: str = ''
    return_to_charger: bool = False


@dataclass(frozen=True)
class MissionConfig:
    """Thresholds, retries and deadlines. Fractions of charge, metres, seconds, radians."""

    # Battery.
    battery_min_start: float = 0.25        # below this no order is accepted
    battery_charge_below: float = 0.30     # idle or just delivered: go and charge
    battery_abandon_below: float = 0.15    # on the way to a pickup: charge instead
    battery_critical: float = 0.08         # stop where it is
    require_battery: bool = False          # refuse orders while the level is unknown
    auto_charge: bool = True               # go to the charger by itself when idle and low
    auto_charge_retry: float = 120.0       # wait this long after a failed trip to it
    cruise_speed: float = 0.6              # what range and travel time are estimated with

    # Before the order starts.
    check_routes: bool = True
    route_check_timeout: float = 10.0

    # Holds.
    pose_timeout: float = 1.0              # a pose older than this is no pose
    hold_timeout: float = 120.0            # e-stop, safety fault, operator, lost position
    manual_resume_delay: float = 3.0       # after the operator's last command
    blocked_timeout: float = 90.0          # stopped for an obstacle, in one stretch

    # Travel.
    nav_attempts: int = 3
    nav_retry_wait: float = 5.0
    nav_accept_timeout: float = 5.0        # for navigation to take a goal
    leg_speed_floor: float = 0.15          # a leg may take its length at this speed...
    leg_timeout_margin: float = 60.0       # ...plus this
    leg_timeout_default: float = 300.0     # when the length is not known
    # Navigation that comes no nearer its goal (Nav2's remaining distance down by
    # progress_distance) in progress_timeout of driving has failed that attempt. Nav2 keeps
    # finding ways past the side of an obstacle the scanner has not seen yet, and a vehicle
    # in front of a closed aisle would otherwise try one way and the other until the leg's
    # deadline. Time held or stopped for an obstacle does not count.
    progress_timeout: float = 60.0
    progress_distance: float = 0.5
    approach_tolerance: float = 0.25       # how near the approach pose "arrived" is
    approach_heading_tolerance: float = 0.6

    # Docking.
    dock_attempts: int = 3
    dock_timeout: float = 45.0
    charge_contact_timeout: float = 8.0    # for current to flow once docked at a charger
    undock_speed: float = 0.15             # backing out of a station
    undock_timeout: float = 20.0

    # Transfers.
    load_attempts: int = 3
    load_retry_period: float = 6.0         # command again if nothing has happened
    load_timeout: float = 40.0             # for the whole transfer
    load_debounce: float = 1.0             # a load state has to last this long to count


@dataclass(frozen=True)
class RouteReply:
    """Answer to a route request: its length, or why there is none."""

    length: Optional[float] = None
    error: str = ''


@dataclass(frozen=True)
class MissionInputs:
    """What is known this cycle."""

    now: float
    pose: Optional[Pose2D] = None          # map frame
    pose_age: float = 0.0
    nav: NavState = NavState.IDLE
    nav_goal_id: int = 0                   # the goal `nav` is about
    nav_distance_remaining: Optional[float] = None
    load: LoadState = LoadState.UNKNOWN
    battery: Optional[float] = None        # 0..1
    charging: bool = False
    safety: SafetyState = SafetyState.OK
    safety_reason: str = ''
    command_source: str = 'none'           # 'manual' while an operator drives
    routes: Mapping[int, RouteReply] = field(default_factory=dict)


@dataclass(frozen=True)
class NavGoal:
    goal_id: int
    pose: Pose2D
    station: str


@dataclass(frozen=True)
class RouteRequest:
    request_id: int
    start: Pose2D
    goal: Pose2D


@dataclass(frozen=True)
class LoadCommand:
    action: str        # 'load' or 'unload'
    station: str


@dataclass(frozen=True)
class MissionResult:
    """How an order ended."""

    outcome: str       # delivered | refused | failed | canceled
    message: str
    delivered: bool = False
    duration: float = 0.0
    distance: float = 0.0

    @property
    def success(self) -> bool:
        return self.outcome == 'delivered'


@dataclass(frozen=True)
class MissionStatus:
    """Where an order stands, for people and for the action's feedback."""

    state: str = 'idle'
    detail: str = ''
    pickup: str = ''
    dropoff: str = ''
    station: str = ''              # where the vehicle is heading or docked
    carrying: bool = False
    holding: str = ''              # why it is held, if it is
    distance_remaining: float = 0.0
    distance_driven: float = 0.0
    elapsed: float = 0.0
    battery: float = -1.0
    active: bool = False


@dataclass
class MissionOutputs:
    """What the mission wants this cycle."""

    navigate: Optional[NavGoal] = None
    cancel_navigation: bool = False
    velocity: Optional[Tuple[float, float]] = None      # docking command; None: say nothing
    load_command: Optional[LoadCommand] = None
    route_requests: List[RouteRequest] = field(default_factory=list)
    status: MissionStatus = field(default_factory=MissionStatus)
    result: Optional[MissionResult] = None


@dataclass
class _Travel:
    station: Station
    # Go even if the vehicle is already near: a re-approach after a failed dock.
    forced: bool = False
    # Why this is not the first try, for the status.
    note: str = ''
    attempts: int = 0
    goal_id: int = 0
    sent_at: float = 0.0
    wait_until: Optional[float] = None
    length: Optional[float] = None
    remaining: float = 0.0
    # The nearest navigation has brought the vehicle to its goal, and when (in step time).
    nearest: float = math.inf
    nearest_at: float = 0.0


@dataclass
class _Dock:
    station: Station
    attempts: int = 0
    controller: Optional[DockingController] = None
    docked_at: Optional[float] = None


@dataclass
class _Undock:
    """Back straight out of a station to its approach pose, where there is room to turn."""

    station: Station


@dataclass
class _Transfer:
    action: str
    station: Station
    attempts: int = 0
    sent_at: Optional[float] = None
    seen_since: Optional[float] = None


class Mission:
    """Runs one transport order at a time."""

    def __init__(self, site: Site, config: MissionConfig = MissionConfig(),
                 docking: DockingConfig = DockingConfig(),
                 battery_range: Optional[Callable[[float], float]] = None):
        self.site = site
        self.config = config
        self.docking = docking
        # battery level -> metres it is good for before the critical level.
        self.battery_range = battery_range
        self.last_result: Optional[MissionResult] = None
        self._next_id = 1
        self._auto_charge_after = -math.inf
        self._last_pose: Optional[Pose2D] = None
        self._last_now: Optional[float] = None
        self._manual_seen_at = -math.inf
        self._distance = 0.0
        self._call_off_navigation = False
        self._reset()

    def _reset(self) -> None:
        self._state = 'idle'
        self._detail = ''
        self._order: Optional[Order] = None
        self._internal = False                 # a trip the mission gave itself
        self._plan: List[object] = []
        self._index = 0
        self._started_at = 0.0
        self._start_distance = 0.0
        self._step_time = 0.0                  # active time spent in the current step
        self._carrying = False
        self._delivered = False
        self._returned = False
        self._cancel_requested = False
        self._canceling = False
        self._abandoned = ''
        self._hold_since: Optional[float] = None
        self._hold_reason = ''
        self._blocked_since: Optional[float] = None
        self._load_wrong_since: Optional[float] = None
        self._routes_pending: Dict[int, Tuple[int, str]] = {}
        self._routes_sent_at = 0.0
        self._route_lengths: Dict[int, float] = {}
        self._requests_to_send: List[RouteRequest] = []

    # -- public -------------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether an order (or a trip to the charger the mission started) is running."""
        return self._state not in ('idle', 'complete', 'failed', 'canceled')

    @property
    def busy(self) -> bool:
        """Whether an order someone gave is running. A self-started charge trip is not."""
        return self.active and not self._internal

    @property
    def state(self) -> str:
        return self._state

    def start(self, order: Order, inputs: MissionInputs) -> Optional[str]:
        """Take an order. Returns None when accepted, or the reason it is refused."""
        reason = self._refusal(order, inputs)
        if reason:
            return reason
        # Only a charge trip the mission gave itself can still be running here: the order
        # takes over, and whatever navigation goal the trip had is called off.
        preempting = self.active
        self._reset()
        self._call_off_navigation = preempting
        self._order = order
        self._started_at = inputs.now
        self._start_distance = self._distance
        self._carrying = not order.pickup
        self._plan = self._build_plan(order, inputs)
        self._state = 'checking'
        self._detail = 'checking the order'
        if self.config.check_routes:
            self._request_routes(inputs)
        return None

    def cancel(self) -> bool:
        """Ask for the running order to be abandoned. False if there is none."""
        if not self.busy:
            return False
        self._cancel_requested = True
        return True

    def update(self, inputs: MissionInputs) -> MissionOutputs:
        """Run one cycle."""
        out = MissionOutputs()
        dt = 0.0 if self._last_now is None else max(0.0, min(1.0, inputs.now - self._last_now))
        self._last_now = inputs.now
        self._track(inputs)
        if self._call_off_navigation:
            out.cancel_navigation = True
            self._call_off_navigation = False

        if not self.active:
            self._maybe_charge(inputs)
        if self.active:
            self._run(inputs, out, dt)
        out.status = self._status(inputs)
        return out

    # -- accepting an order -------------------------------------------------------------------

    def _refusal(self, order: Order, inputs: MissionInputs) -> str:
        cfg = self.config
        if self.busy:
            return f'busy with an order to {self._order.dropoff!r}'
        if not order.dropoff:
            return 'the order names no dropoff'
        for role, name in (('pickup', order.pickup), ('dropoff', order.dropoff)):
            if not name:
                continue
            if not self.site.has_station(name):
                known = ', '.join(s.name for s in self.site.transfer_stations) or 'none'
                return f"unknown {role} '{name}' (transfer stations here: {known})"
            if not self.site.station(name).is_transfer:
                return f"'{name}' is a {self.site.station(name).kind}, not a transfer station"
        if order.pickup == order.dropoff:
            return f"pickup and dropoff are both '{order.dropoff}'"
        if order.return_to_charger and not self.site.chargers:
            return 'the site has no charger to return to'
        if inputs.safety is SafetyState.ESTOP:
            return f'the e-stop is active: {inputs.safety_reason or "release and reset it"}'
        if inputs.safety is SafetyState.FAULT:
            return f'the safety gate reports a fault: {inputs.safety_reason}'
        if inputs.command_source == 'manual':
            return 'an operator is driving the vehicle'
        if inputs.pose is None or inputs.pose_age > cfg.pose_timeout:
            return 'the vehicle does not know where it is'
        if inputs.load in (LoadState.UNKNOWN, LoadState.FAULT):
            return 'the load handler is not reporting'
        if inputs.load is LoadState.MOVING:
            return 'the load handler is moving a load'
        if order.pickup and inputs.load is LoadState.LOADED:
            return ('the vehicle already carries a load: send the order without a pickup to '
                    'deliver it')
        if not order.pickup and inputs.load is not LoadState.LOADED:
            return 'the order names no pickup and the vehicle carries no load'
        if inputs.battery is None:
            if cfg.require_battery:
                return 'the battery level is not known'
        elif inputs.battery < cfg.battery_min_start:
            return (f'battery at {inputs.battery:.0%}, orders need '
                    f'{cfg.battery_min_start:.0%}')
        return ''

    def _build_plan(self, order: Order, inputs: MissionInputs) -> List[object]:
        plan: List[object] = []
        if order.pickup:
            pickup = self.site.station(order.pickup)
            plan += [_Travel(pickup), _Dock(pickup), _Transfer('load', pickup)]
        dropoff = self.site.station(order.dropoff)
        plan += [_Travel(dropoff), _Dock(dropoff), _Transfer('unload', dropoff)]
        if order.return_to_charger:
            plan += self._charger_trip(dropoff.pose)
        return plan

    def _charger_trip(self, start: Pose2D) -> List[object]:
        charger = self.site.nearest_charger(start)
        return [] if charger is None else [_Travel(charger), _Dock(charger)]

    def _request_routes(self, inputs: MissionInputs) -> None:
        """Ask for the length of every leg, and of the way to a charger after the last."""
        start = inputs.pose
        for index, step in enumerate(self._plan):
            if not isinstance(step, _Travel):
                continue
            self._add_route_request(start, step.station, index)
            start = step.station.pose
        if self._plan and not isinstance(self._plan[-1], _Dock):
            # Not returning to the charger: it still has to be reachable afterwards.
            charger = self.site.nearest_charger(start)
            if charger is not None:
                self._add_route_request(start, charger, -1)
        self._routes_sent_at = inputs.now

    def _add_route_request(self, start: Pose2D, station: Station, index: int) -> None:
        request_id = self._new_id()
        self._routes_pending[request_id] = (index, station.name)
        self._requests_to_send.append(RouteRequest(request_id, start, station.approach_pose))

    # -- the cycle ----------------------------------------------------------------------------

    def _run(self, inputs: MissionInputs, out: MissionOutputs, dt: float) -> None:
        cfg = self.config
        if self._requests_to_send:
            out.route_requests = self._requests_to_send
            self._requests_to_send = []

        if (inputs.battery is not None and inputs.battery < cfg.battery_critical
                and not inputs.charging):
            out.cancel_navigation = True
            self._fail(inputs, f'battery at {inputs.battery:.0%}: stopped where it is', out)
            return

        if self._cancel_requested and not self._canceling and not self._transfer_under_way(inputs):
            self._begin_cancel(inputs, out)
            if not self.active:
                return

        hold = self._hold_reason_now(inputs)
        if hold:
            if self._hold_since is None:
                self._hold_since = inputs.now
                out.cancel_navigation = True
                self._interrupt_step()
            self._hold_reason = hold
            if inputs.now - self._hold_since > cfg.hold_timeout:
                self._fail(inputs, f'{hold} for {cfg.hold_timeout:.0f} s', out)
            return
        if self._hold_since is not None:
            self._hold_since = None
            self._hold_reason = ''

        if inputs.safety is SafetyState.PROTECTIVE_STOP:
            if self._blocked_since is None:
                self._blocked_since = inputs.now
            elif inputs.now - self._blocked_since > cfg.blocked_timeout:
                out.cancel_navigation = True
                self._fail(inputs, f'blocked for {cfg.blocked_timeout:.0f} s: '
                           f'{inputs.safety_reason or "obstacle"}', out)
                return
        else:
            self._blocked_since = None
            self._step_time += dt

        if self._state == 'checking':
            self._check(inputs, out)
            if self._state == 'checking' or not self.active:
                return

        if not self._load_as_expected(inputs, out):
            return
        self._battery_policy(inputs, out)

        # A step that finishes hands over to the next in the same cycle, so a skipped
        # step (already there, already docked) costs no time.
        for _ in range(len(self._plan) + 2):
            if not self.active or self._index >= len(self._plan):
                break
            step = self._plan[self._index]
            if isinstance(step, _Travel):
                finished = self._travel(step, inputs, out)
            elif isinstance(step, _Dock):
                finished = self._dock(step, inputs, out)
            elif isinstance(step, _Undock):
                finished = self._undock(step, inputs, out)
            else:
                finished = self._transfer(step, inputs, out)
            if not finished:
                break
            self._index += 1
            self._step_time = 0.0
        if self.active and self._index >= len(self._plan):
            self._complete(inputs, out)

    def _hold_reason_now(self, inputs: MissionInputs) -> str:
        cfg = self.config
        if inputs.safety is SafetyState.ESTOP:
            return 'e-stop active'
        if inputs.safety is SafetyState.FAULT:
            if inputs.safety_reason:
                return f'safety fault ({inputs.safety_reason})'
            return 'safety fault'
        if inputs.command_source == 'manual':
            self._manual_seen_at = inputs.now
            return 'operator driving'
        if inputs.now - self._manual_seen_at < cfg.manual_resume_delay:
            return 'operator driving'
        if inputs.pose is None or inputs.pose_age > cfg.pose_timeout:
            return 'position lost'
        return ''

    def _interrupt_step(self) -> None:
        """Forget what the current step had in flight, so that it starts again cleanly."""
        if self._index >= len(self._plan):
            return
        step = self._plan[self._index]
        if isinstance(step, _Travel):
            step.goal_id = 0
            step.wait_until = None
        elif isinstance(step, _Dock):
            step.controller = None
            step.docked_at = None

    # -- checking -----------------------------------------------------------------------------

    def _check(self, inputs: MissionInputs, out: MissionOutputs) -> None:
        cfg = self.config
        for request_id in list(self._routes_pending):
            reply = inputs.routes.get(request_id)
            if reply is None:
                continue
            index, name = self._routes_pending.pop(request_id)
            if reply.length is None:
                self._refuse(inputs, f"no route to '{name}': {reply.error or 'none found'}", out)
                return
            self._route_lengths[index] = reply.length
            if index >= 0:
                self._plan[index].length = reply.length
        if self._routes_pending:
            if inputs.now - self._routes_sent_at > cfg.route_check_timeout:
                self._refuse(inputs, 'the route check did not answer', out)
            return
        total = sum(self._route_lengths.values())
        if (total > 0.0 and self.battery_range is not None and inputs.battery is not None):
            reach = self.battery_range(inputs.battery)
            if total > reach:
                self._refuse(
                    inputs, f'the order is {total:.0f} m with the way to the charger, the '
                    f'battery ({inputs.battery:.0%}) is good for {reach:.0f} m', out)
                return
        self._state = 'running'
        self._step_time = 0.0

    # -- watching the load and the battery ----------------------------------------------------

    def _load_as_expected(self, inputs: MissionInputs, out: MissionOutputs) -> bool:
        """Notice a load that vanished, or one that appeared, between transfers."""
        step = self._plan[self._index] if self._index < len(self._plan) else None
        if isinstance(step, _Transfer) and step.sent_at is not None:
            self._load_wrong_since = None
            return True
        wrong = ((self._carrying and inputs.load is LoadState.EMPTY)
                 or (not self._carrying and inputs.load is LoadState.LOADED))
        if not wrong:
            self._load_wrong_since = None
            return True
        if self._load_wrong_since is None:
            self._load_wrong_since = inputs.now
        if inputs.now - self._load_wrong_since < self.config.load_debounce:
            return True
        out.cancel_navigation = True
        if self._carrying:
            self._carrying = False
            self._fail(inputs, 'the load is no longer on the vehicle', out)
        else:
            self._carrying = True
            self._fail(inputs, 'a load is on the vehicle that this order did not put there', out)
        return False

    def _battery_policy(self, inputs: MissionInputs, out: MissionOutputs) -> None:
        cfg = self.config
        if inputs.battery is None or self._internal or self._canceling or self._abandoned:
            return
        collecting = (not self._carrying and not self._delivered
                      and not isinstance(self._plan[self._index], _Transfer))
        if collecting and inputs.battery < cfg.battery_abandon_below:
            trip = self._charger_trip(inputs.pose)
            if trip:
                out.cancel_navigation = True
                self._abandoned = (f'battery at {inputs.battery:.0%}: went to charge instead of '
                                   f"collecting from '{self._order.pickup}'")
                self._plan, self._index, self._step_time = trip, 0, 0.0

    # -- travel -------------------------------------------------------------------------------

    def _travel(self, step: _Travel, inputs: MissionInputs, out: MissionOutputs) -> bool:
        cfg = self.config
        target = step.station.approach_pose
        if step.goal_id == 0 and step.wait_until is None:
            there = self._at(inputs.pose, target) or is_docked(inputs.pose, step.station.pose)
            if there and not step.forced:
                return True
            inside = self._in_docking_run(inputs.pose)
            if inside is not None:
                # Docked, or part way in: back out before navigation turns the vehicle.
                self._plan.insert(self._index, _Undock(inside))
                self._step_time = 0.0
                return False
            self._send_goal(step, inputs, out)
            return False
        if step.wait_until is not None:
            if inputs.now < step.wait_until:
                return False
            step.wait_until = None
            self._send_goal(step, inputs, out)
            return False

        limit = cfg.leg_timeout_default if step.length is None else (
            step.length / cfg.leg_speed_floor + cfg.leg_timeout_margin)
        if inputs.nav_goal_id != step.goal_id:
            if inputs.now - step.sent_at > cfg.nav_accept_timeout:
                return self._travel_failed(step, inputs, out, 'navigation did not take the goal')
            return False
        if inputs.nav is NavState.SUCCEEDED:
            if self._at(inputs.pose, target):
                return True
            off = inputs.pose.distance_to(target)
            return self._travel_failed(
                step, inputs, out, f'navigation stopped {off:.2f} m from the approach pose')
        if inputs.nav in (NavState.FAILED, NavState.CANCELED):
            return self._travel_failed(step, inputs, out, 'navigation gave up')
        if inputs.nav_distance_remaining is not None:
            step.remaining = inputs.nav_distance_remaining
            if step.remaining < step.nearest - cfg.progress_distance:
                step.nearest, step.nearest_at = step.remaining, self._step_time
        if self._step_time > limit:
            out.cancel_navigation = True
            return self._travel_failed(step, inputs, out, f'not there after {limit:.0f} s')
        if step.nearest < math.inf and \
                self._step_time - step.nearest_at > cfg.progress_timeout:
            out.cancel_navigation = True
            return self._travel_failed(
                step, inputs, out, f'no nearer to it in {cfg.progress_timeout:.0f} s')
        return False

    def _send_goal(self, step: _Travel, inputs: MissionInputs, out: MissionOutputs) -> None:
        step.goal_id = self._new_id()
        step.sent_at = inputs.now
        step.nearest, step.nearest_at = math.inf, 0.0
        self._step_time = 0.0
        out.navigate = NavGoal(step.goal_id, step.station.approach_pose, step.station.name)

    def _in_docking_run(self, pose: Optional[Pose2D]) -> Optional[Station]:
        """Return the station whose docking run the vehicle stands in, if any."""
        if pose is None:
            return None
        for station in self.site.stations.values():
            local = pose.relative_to(station.pose)
            if (-(station.approach - 0.05) < local.x < 0.10 and abs(local.y) < 0.25
                    and abs(local.yaw) < math.radians(30.0)):
                return station
        return None

    def _undock(self, step: _Undock, inputs: MissionInputs, out: MissionOutputs) -> bool:
        cfg = self.config
        local = inputs.pose.relative_to(step.station.pose)
        to_go = step.station.approach + local.x          # metres still to back out
        if to_go <= 0.02:
            out.velocity = (0.0, 0.0)
            return True
        if self._step_time > cfg.undock_timeout:
            self._fail(inputs, f"could not back out of '{step.station.name}' in "
                       f'{cfg.undock_timeout:.0f} s', out)
            return False
        speed = max(0.05, min(cfg.undock_speed, 0.8 * to_go))
        # Hold the station's heading while backing out, on gentle curves only (see
        # DockingConfig.min_turn_radius).
        limit = min(0.3, speed / self.docking.min_turn_radius)
        turn = max(-limit, min(limit, -1.5 * local.yaw))
        out.velocity = (-speed, turn)
        return False

    def _travel_failed(self, step: _Travel, inputs: MissionInputs, out: MissionOutputs,
                       reason: str) -> bool:
        cfg = self.config
        step.attempts += 1
        step.goal_id = 0
        if step.attempts >= cfg.nav_attempts:
            self._fail(inputs, f"could not reach '{step.station.name}' in {step.attempts} "
                       f'attempts: {reason}', out)
            return False
        step.wait_until = inputs.now + cfg.nav_retry_wait
        step.note = f'{reason}; trying again ({step.attempts + 1} of {cfg.nav_attempts})'
        return False

    def _at(self, pose: Optional[Pose2D], target: Pose2D) -> bool:
        cfg = self.config
        return (pose is not None and pose.distance_to(target) <= cfg.approach_tolerance
                and abs(angle_diff(pose.yaw, target.yaw)) <= cfg.approach_heading_tolerance)

    # -- docking ------------------------------------------------------------------------------

    def _dock(self, step: _Dock, inputs: MissionInputs, out: MissionOutputs) -> bool:
        cfg = self.config
        if step.docked_at is None and step.controller is None:
            if is_docked(inputs.pose, step.station.pose):
                step.docked_at = inputs.now
            else:
                step.controller = DockingController(step.station.pose, self.docking)
                self._step_time = 0.0
        if step.docked_at is None:
            command = step.controller.update(inputs.pose)
            if command.failed:
                return self._dock_failed(step, inputs, out, command.reason)
            if self._step_time > cfg.dock_timeout:
                return self._dock_failed(
                    step, inputs, out, f'not docked after {cfg.dock_timeout:.0f} s')
            if not command.done:
                out.velocity = (command.v, command.w)
                return False
            step.docked_at = inputs.now
            out.velocity = (0.0, 0.0)
        if not step.station.is_charger or inputs.battery is None or inputs.charging:
            return True
        if inputs.now - step.docked_at > cfg.charge_contact_timeout:
            return self._dock_failed(step, inputs, out, 'docked but not charging')
        return False

    def _dock_failed(self, step: _Dock, inputs: MissionInputs, out: MissionOutputs,
                     reason: str) -> bool:
        cfg = self.config
        step.attempts += 1
        step.controller = None
        step.docked_at = None
        out.velocity = (0.0, 0.0)
        if step.attempts >= cfg.dock_attempts:
            self._fail(inputs, f"could not dock at '{step.station.name}' in {step.attempts} "
                       f'attempts: {reason}', out)
            return False
        # Back to the approach pose and in again. "Near enough" is where it just failed
        # from, so this time navigation is sent whatever the distance.
        note = f'{reason}; approaching again ({step.attempts + 1} of {cfg.dock_attempts})'
        self._plan.insert(self._index, _Travel(step.station, forced=True, note=note))
        self._step_time = 0.0
        return False

    # -- transfers ----------------------------------------------------------------------------

    def _transfer(self, step: _Transfer, inputs: MissionInputs, out: MissionOutputs) -> bool:
        cfg = self.config
        wanted = LoadState.LOADED if step.action == 'load' else LoadState.EMPTY
        if step.sent_at is None:
            self._send_load_command(step, inputs, out)
            return False
        if inputs.load is wanted:
            if step.seen_since is None:
                step.seen_since = inputs.now
            if inputs.now - step.seen_since >= cfg.load_debounce:
                self._carrying = step.action == 'load'
                self._delivered = self._delivered or step.action == 'unload'
                if step.action == 'unload':
                    self._after_delivery(inputs)
                return True
            return False
        step.seen_since = None
        if self._step_time > cfg.load_timeout:
            return self._transfer_failed(step, inputs, out, f'after {cfg.load_timeout:.0f} s')
        if inputs.load is LoadState.MOVING:
            return False
        if inputs.load is LoadState.FAULT or inputs.now - step.sent_at > cfg.load_retry_period:
            if step.attempts >= cfg.load_attempts:
                return self._transfer_failed(step, inputs, out, f'after {step.attempts} commands')
            self._send_load_command(step, inputs, out)
        return False

    def _send_load_command(self, step: _Transfer, inputs: MissionInputs,
                           out: MissionOutputs) -> None:
        if step.sent_at is None:
            self._step_time = 0.0
        step.attempts += 1
        step.sent_at = inputs.now
        out.load_command = LoadCommand(step.action, step.station.name)

    def _transfer_failed(self, step: _Transfer, inputs: MissionInputs, out: MissionOutputs,
                         when: str) -> bool:
        if step.action == 'load':
            self._fail(inputs, f"no load came aboard at '{step.station.name}' {when}", out)
        else:
            self._fail(inputs, f"the load did not leave the vehicle at '{step.station.name}' "
                       f'{when}', out)
        return False

    def _transfer_under_way(self, inputs: MissionInputs) -> bool:
        """Whether the load handler has started on the current transfer.

        True while the load is moving, and after it has arrived until the transfer step
        has counted it: cancelling in between would leave a load on board that the order
        does not know about.
        """
        if self._index >= len(self._plan):
            return False
        step = self._plan[self._index]
        if not isinstance(step, _Transfer) or step.sent_at is None:
            return False
        wanted = LoadState.LOADED if step.action == 'load' else LoadState.EMPTY
        return inputs.load in (LoadState.MOVING, wanted)

    def _after_delivery(self, inputs: MissionInputs) -> None:
        """Delivered and low on charge: go and charge even if the order did not ask."""
        cfg = self.config
        returning = any(isinstance(s, _Dock) and s.station.is_charger
                        for s in self._plan[self._index + 1:])
        if (not returning and inputs.battery is not None
                and inputs.battery < cfg.battery_charge_below):
            self._plan += self._charger_trip(inputs.pose)

    # -- endings ------------------------------------------------------------------------------

    def _begin_cancel(self, inputs: MissionInputs, out: MissionOutputs) -> None:
        self._canceling = True
        out.cancel_navigation = True
        out.velocity = (0.0, 0.0)
        trip = self._charger_trip(inputs.pose) if inputs.pose is not None else []
        if not trip or self._state == 'checking':
            self._finish(inputs, out, 'canceled', 'canceled', self._cancel_message(''))
            return
        self._state = 'running'
        self._plan, self._index, self._step_time = trip, 0, 0.0

    def _cancel_message(self, tail: str) -> str:
        message = 'cancelled'
        if self._carrying:
            message += ' with the load on board'
        return message + tail

    def _complete(self, inputs: MissionInputs, out: MissionOutputs) -> None:
        if self._internal:
            self._state, self._detail = 'idle', 'at the charger'
            self._order = None
            return
        at_charger = bool(self._plan) and isinstance(self._plan[-1], _Dock) \
            and self._plan[-1].station.is_charger
        if self._canceling:
            self._finish(inputs, out, 'canceled', 'canceled',
                         self._cancel_message(', back at the charger'))
        elif self._abandoned:
            self._finish(inputs, out, 'failed', 'failed', self._abandoned)
        else:
            message = self._delivered_message()
            if at_charger:
                message += ', back at the charger'
            self._finish(inputs, out, 'complete', 'delivered', message)

    def _delivered_message(self) -> str:
        if self._order.pickup:
            return f"delivered from '{self._order.pickup}' to '{self._order.dropoff}'"
        return f"delivered to '{self._order.dropoff}'"

    def _fail(self, inputs: MissionInputs, reason: str, out: MissionOutputs) -> None:
        out.velocity = (0.0, 0.0)
        if self._internal:
            self._auto_charge_after = inputs.now + self.config.auto_charge_retry
            self._state, self._detail = 'failed', f'could not get to the charger: {reason}'
            self._order = None
            return
        if self._canceling:
            self._finish(inputs, out, 'canceled', 'canceled',
                         self._cancel_message(f'; could not return to the charger: {reason}'))
        elif self._abandoned:
            self._finish(inputs, out, 'failed', 'failed', f'{self._abandoned}; then {reason}')
        elif self._delivered:
            # The order itself is done: what failed is the way back.
            self._finish(inputs, out, 'complete', 'delivered',
                         f'{self._delivered_message()}; could not return to the charger: {reason}')
        else:
            if self._carrying:
                reason += ' (the load is on board)'
            self._finish(inputs, out, 'failed', 'failed', reason)

    def _refuse(self, inputs: MissionInputs, reason: str, out: MissionOutputs) -> None:
        self._finish(inputs, out, 'failed', 'refused', reason)

    def _finish(self, inputs: MissionInputs, out: MissionOutputs, state: str, outcome: str,
                message: str) -> None:
        result = MissionResult(
            outcome, message, delivered=self._delivered,
            duration=inputs.now - self._started_at,
            distance=self._distance - self._start_distance)
        self.last_result = result
        out.result = result
        self._state, self._detail = state, message
        self._hold_since, self._hold_reason = None, ''

    # -- charging by itself -------------------------------------------------------------------

    def _maybe_charge(self, inputs: MissionInputs) -> None:
        cfg = self.config
        if (not cfg.auto_charge or inputs.battery is None or inputs.charging
                or inputs.battery >= cfg.battery_charge_below
                or inputs.now < self._auto_charge_after
                or inputs.pose is None or inputs.pose_age > cfg.pose_timeout
                or inputs.safety in (SafetyState.ESTOP, SafetyState.FAULT)
                or inputs.command_source == 'manual'):
            return
        trip = self._charger_trip(inputs.pose)
        if not trip:
            return
        charger = trip[-1].station
        if is_docked(inputs.pose, charger.pose):
            # On the contacts and no current: driving off and on again will not help.
            self._auto_charge_after = inputs.now + cfg.auto_charge_retry
            return
        carrying = inputs.load is LoadState.LOADED
        self._reset()
        self._internal = True
        self._carrying = carrying
        self._order = Order(dropoff=charger.name)
        self._started_at = inputs.now
        self._start_distance = self._distance
        self._plan = trip
        self._state = 'running'
        self._detail = f'battery at {inputs.battery:.0%}: going to charge'

    # -- reporting ----------------------------------------------------------------------------

    def _track(self, inputs: MissionInputs) -> None:
        if inputs.pose is not None:
            if self._last_pose is not None:
                moved = self._last_pose.distance_to(inputs.pose)
                if moved < 1.0:                  # a jump is the localiser, not the wheels
                    self._distance += moved
            self._last_pose = inputs.pose

    def _status(self, inputs: MissionInputs) -> MissionStatus:
        state, detail, station, remaining = self._state, self._detail, '', 0.0
        if self._state == 'running' and self._index < len(self._plan):
            step = self._plan[self._index]
            station = step.station.name
            if isinstance(step, _Travel):
                state = self._travel_name(step.station)
                remaining = step.remaining
                detail = f"driving to '{station}'"
                if step.note:
                    detail += f': {step.note}'
            elif isinstance(step, _Dock):
                state = 'docking'
                detail = (f"waiting for charging current at '{station}'" if step.docked_at
                          else f"docking at '{station}'")
            elif isinstance(step, _Undock):
                state = 'undocking'
                detail = f"backing out of '{station}'"
            else:
                state = 'loading' if step.action == 'load' else 'unloading'
                detail = f"{state} at '{station}' (command {step.attempts} of " \
                         f'{self.config.load_attempts})'
            if self._canceling:
                detail = f'cancelled: {detail}'
            elif self._abandoned or (self._internal and self._detail):
                detail = self._abandoned or self._detail
        if self._hold_reason:
            detail = f'holding: {self._hold_reason}'
        elif self._blocked_since is not None and self.active:
            detail = f'stopped: {inputs.safety_reason or "obstacle"}'
        order = self._order
        return MissionStatus(
            state=state, detail=detail,
            pickup='' if order is None or self._internal else order.pickup,
            dropoff='' if order is None or self._internal else order.dropoff,
            station=station, carrying=self._carrying, holding=self._hold_reason,
            distance_remaining=remaining,
            distance_driven=self._distance - self._start_distance if self.active else (
                self.last_result.distance if self.last_result else 0.0),
            elapsed=inputs.now - self._started_at if self.active else (
                self.last_result.duration if self.last_result else 0.0),
            battery=-1.0 if inputs.battery is None else inputs.battery,
            active=self.active)

    def _travel_name(self, station: Station) -> str:
        if station.is_charger:
            return 'to_charger'
        if self._order is not None and station.name == self._order.pickup and not self._carrying:
            return 'to_pickup'
        return 'to_dropoff'

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id
