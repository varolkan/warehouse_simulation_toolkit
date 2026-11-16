"""What the mission tests drive orders against.

Rig: the mission with a vehicle that has no surroundings. Travel is a straight line at a
fixed speed, docking is driven for real by the docking controller on the kinematic
vehicle, and the load handler takes its few seconds. Every way the outside world can let
the mission down is a knob on it. A whole order takes a few milliseconds.

The closed-loop world, with a map, the simulated scanner, the safety gate and a navigator
that plans round obstacles, is in world.py.
"""

import math

from agv_core.battery import BatteryConfig
from agv_core.docking import is_docked
from agv_core.geometry import angle_diff, Pose2D
from agv_core.mission import (
    Mission, MissionConfig, MissionInputs, MissionStatus, NavState, Order, RouteReply)
from agv_core.safety import SafetyState
from agv_core.simulation import KinematicVehicle, LoadHandlerSim
from agv_core.sites import Site

SITE = {
    'name': 'test hall',
    'stations': {
        'charger': {'kind': 'charger', 'pose': [0.0, 0.0, 180]},
        'a': {'kind': 'transfer', 'pose': [10.0, 0.0, 0]},
        'b': {'kind': 'transfer', 'pose': [10.0, 8.0, 90]},
        'c': {'kind': 'transfer', 'pose': [-6.0, 5.0, 180]},
    },
    'zones': [],
}


def site(**changes):
    data = {**SITE, 'stations': dict(SITE['stations'])}
    data.update(changes)
    return Site.from_dict(data)


BATTERY = BatteryConfig(idle_hours=10.0, range_km=10.0, charge_hours=1.0)


class Rig:
    """The mission, a vehicle without surroundings, and every knob that can break an order."""

    def __init__(self, the_site=None, config=None, at='charger', loaded=False, battery=0.8,
                 dt=0.05, battery_range=True):
        self.site = the_site or site()
        self.config = config or MissionConfig()

        def reach(level):
            return BATTERY.range_m(level, self.config.battery_critical,
                                   self.config.cruise_speed)
        self.mission = Mission(self.site, self.config,
                               battery_range=reach if battery_range else None)
        start = self.site.station(at).pose if isinstance(at, str) else at
        self.vehicle = KinematicVehicle(start)
        self.handler = LoadHandlerSim(loaded=loaded, duration=3.0)
        self.now = 100.0
        self.dt = dt
        self.battery = battery

        # Navigation stand-in.
        self.nav = NavState.IDLE
        self.nav_goal = None
        self.nav_speed = 0.8
        self.nav_failures = 0            # fail this many goals (a second after taking each)
        self.nav_fails_at = {}           # station -> goals to that station still to fail
        self.nav_deaf = False            # never take a goal
        self.nav_stalls = False          # take goals and never get anywhere
        self.arrival_errors = []         # (dx, dy, dyaw) for successive arrivals
        self.nav_lies = 0                # report success this many times without moving
        self.nav_wanders = False         # drive round the goal, coming no nearer to it

        # The rest of the world.
        self.route_errors = {}           # station -> why there is no route to it
        self.route_silent = False
        self.route_factor = 1.0          # routes are this much longer than a straight line
        self.safety = SafetyState.OK
        self.safety_reason = ''
        self.command_source = 'none'
        self.pose_lost = False
        self.pose_frozen = None          # what the localiser reports instead of the truth
        self.load_report = None          # overrides what the load handler reports
        self.charger_dead = False
        self.drive_dead = False          # docking commands move nothing
        self.battery_drain = 0.0         # per second

        # What happened.
        self.results = []
        self.statuses = []
        self.goals = []
        self.load_commands = []
        self.cancels = 0
        self.routes = {}
        self.route_requests = []
        self._taken_at = 0.0

    # -- driving it ------------------------------------------------------------------------

    def order(self, pickup='', dropoff='', **options):
        """Give an order. Returns the refusal, or None when it was accepted."""
        return self.mission.start(Order(pickup, dropoff, **options), self.inputs())

    def inputs(self):
        pose = None if self.pose_lost else (self.pose_frozen or self.vehicle.pose)
        charging = (self.battery is not None and not self.charger_dead
                    and abs(self.vehicle.v) < 0.01
                    and any(is_docked(self.vehicle.pose, c.pose, 0.06, math.radians(6.0))
                            for c in self.site.chargers))
        return MissionInputs(
            now=self.now, pose=pose, nav=self.nav,
            nav_goal_id=self.nav_goal.goal_id if self.nav_goal else 0,
            nav_distance_remaining=(
                self.vehicle.pose.distance_to(self.nav_goal.pose) if self.nav_goal else None),
            load=self.load_report or self.handler.state, battery=self.battery,
            charging=charging, safety=self.safety, safety_reason=self.safety_reason,
            command_source=self.command_source, routes=dict(self.routes))

    def step(self):
        out = self.mission.update(self.inputs())
        self.statuses.append(out.status)
        if out.result is not None:
            self.results.append(out.result)
        for request in out.route_requests:
            self.route_requests.append(request)
            self._answer_route(request)
        if out.cancel_navigation:
            self.cancels += 1
            if self.nav is NavState.ACTIVE:
                self.nav = NavState.CANCELED
        if out.navigate is not None:
            self.goals.append(out.navigate)
            if not self.nav_deaf:
                self.nav_goal, self.nav, self._taken_at = out.navigate, NavState.ACTIVE, self.now
        if out.load_command is not None:
            self.load_commands.append(out.load_command)
            self.handler.command(out.load_command.action, self.now)
        if out.velocity is not None and not self.drive_dead:
            self.vehicle.command(out.velocity[0], out.velocity[1], self.now)

        stopped = self.safety.stopped
        if stopped:
            self.vehicle.command(0.0, 0.0, self.now)
        if self.nav is NavState.ACTIVE and not stopped:
            self._navigate()
        self.vehicle.step(self.dt, self.now)
        self.handler.step(self.now)
        if self.battery is not None:
            self.battery = max(0.0, self.battery - self.battery_drain * self.dt)
        self.now += self.dt
        return out

    def run(self, seconds):
        for _ in range(int(round(seconds / self.dt))):
            self.step()

    def run_until(self, condition, timeout=600.0):
        """Step until `condition()` holds. Fails the test if it never does."""
        for _ in range(int(round(timeout / self.dt))):
            if condition():
                return
            self.step()
        raise AssertionError(
            f'never happened within {timeout} s; state {self.status.state!r}: '
            f'{self.status.detail!r}')

    def run_to_end(self, timeout=900.0):
        """Run until the order has a result and return it."""
        self.run_until(lambda: self.results, timeout)
        return self.results[-1]

    def run_until_state(self, state, timeout=600.0):
        self.run_until(lambda: self.statuses and self.statuses[-1].state == state, timeout)

    # -- looking at it -----------------------------------------------------------------------

    @property
    def status(self):
        return self.statuses[-1] if self.statuses else MissionStatus()

    @property
    def states(self):
        """Return the states the order went through, without repeats."""
        seen = []
        for status in self.statuses:
            if not seen or seen[-1] != status.state:
                seen.append(status.state)
        return seen

    def docked_at(self, name):
        return is_docked(self.vehicle.pose, self.site.station(name).pose)

    def docking_error(self, name):
        """Metres off the station's centre line and degrees off its heading."""
        local = self.vehicle.pose.relative_to(self.site.station(name).pose)
        return abs(local.y), abs(math.degrees(local.yaw)), abs(local.x)

    # -- the stand-ins -----------------------------------------------------------------------

    def _answer_route(self, request):
        if self.route_silent:
            return
        name = next(s.name for s in self.site.stations.values()
                    if s.approach_pose.distance_to(request.goal) < 1e-6)
        if name in self.route_errors:
            self.routes[request.request_id] = RouteReply(None, self.route_errors[name])
        else:
            length = request.start.distance_to(request.goal) * self.route_factor
            self.routes[request.request_id] = RouteReply(length)

    def _navigate(self):
        goal = self.nav_goal
        if self.nav_stalls:
            return
        to_fail = self.nav_fails_at.get(goal.station, 0)
        if self.nav_failures > 0 or to_fail > 0:
            if self.now - self._taken_at >= 1.0:
                if to_fail > 0:
                    self.nav_fails_at[goal.station] = to_fail - 1
                else:
                    self.nav_failures -= 1
                self.nav = NavState.FAILED
            return
        if self.nav_lies > 0:
            self.nav_lies -= 1
            self.nav = NavState.SUCCEEDED
            return
        pose = self.vehicle.pose
        distance = pose.distance_to(goal.pose)
        reach = self.nav_speed * self.dt
        if self.nav_wanders:
            # Round the goal at the distance the vehicle is from it: moving, never nearer.
            around = math.atan2(pose.y - goal.pose.y, pose.x - goal.pose.x) + reach / distance
            self.vehicle.place(Pose2D(goal.pose.x + distance * math.cos(around),
                                      goal.pose.y + distance * math.sin(around),
                                      around + math.pi / 2))
            self.vehicle.distance += reach
            return
        if distance <= reach:
            dx, dy, dyaw = self.arrival_errors.pop(0) if self.arrival_errors else (0.0, 0.0, 0.0)
            arrived = goal.pose.advanced(dx, dy)
            self.vehicle.place(Pose2D(arrived.x, arrived.y, arrived.yaw + dyaw))
            self.nav = NavState.SUCCEEDED
            return
        heading = math.atan2(goal.pose.y - pose.y, goal.pose.x - pose.x)
        self.vehicle.place(Pose2D(pose.x + reach * math.cos(heading),
                                  pose.y + reach * math.sin(heading), heading))
        # place() does not count as driving: keep the odometer honest for the tests.
        self.vehicle.distance += reach


def heading_error(pose, target):
    return abs(angle_diff(pose.yaw, target.yaw))
