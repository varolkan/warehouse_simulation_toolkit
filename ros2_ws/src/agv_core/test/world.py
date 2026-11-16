"""A hall to drive whole orders through, with everything in the loop.

World puts the mission on the kinematic vehicle inside an occupancy map, with the
simulated scanner feeding the safety gate and a navigator standing in for Nav2: it plans
on the grid with A*, follows the route, plans again when something blocks it, and gives up
when there is no way or no progress, as Nav2 does. Obstacles can be put down and taken
away while it runs. Every step checks whether the vehicle's outline touches anything.

A delivery takes a second or two of test time.
"""

import math

from agv_core.battery import BatteryConfig
from agv_core.docking import is_docked
from agv_core.geometry import angle_diff, path_length, Pose2D
from agv_core.grid import FREE, GridMap, OCCUPIED
from agv_core.masks import keepout_mask
from agv_core.mission import (
    Mission, MissionConfig, MissionInputs, MissionStatus, NavState, Order, RouteReply)
from agv_core.safety import Command, SafetyMonitor, Scan
from agv_core.simulation import (
    BatterySim, Box, Disc, footprint_collides, KinematicVehicle, LoadHandlerSim, ScannerConfig,
    ScannerSim)
from agv_core.sites import Site
from agv_core.vehicle import AGV
import numpy as np

CLEARANCE = AGV.half_width + 0.12


def hall():
    """Build a 16 x 12 m hall: a wall across it with a passage at each end, two racks.

    y=6  +------------------------------------------+
         |  c                              packing  |
         |                                          |
    y=0  |        ====== wall ======                |      passages at x < -5 and x > 5
         |                    [rack]  [rack]        |
         |  charger     a                 b         |
    y=-6 +------------------------------------------+
        x=-8                                       x=8
    """
    grid = GridMap(np.full((240, 320), FREE, dtype=np.int16), 0.05, (-8.0, -6.0))
    grid.data[0, :] = grid.data[-1, :] = OCCUPIED
    grid.data[:, 0] = grid.data[:, -1] = OCCUPIED
    grid.fill_polygon([(-5.0, -0.1), (5.0, -0.1), (5.0, 0.1), (-5.0, 0.1)], OCCUPIED)
    grid.fill_polygon([(1.0, -4.5), (1.8, -4.5), (1.8, -1.5), (1.0, -1.5)], OCCUPIED)
    grid.fill_polygon([(3.2, -4.5), (4.0, -4.5), (4.0, -1.5), (3.2, -1.5)], OCCUPIED)
    return grid


HALL_SITE = {
    'name': 'test hall',
    'stations': {
        'charger': {'kind': 'charger', 'pose': [-7.0, -5.0, 180]},
        'a': {'kind': 'transfer', 'pose': [-2.0, -5.2, -90]},
        'b': {'kind': 'transfer', 'pose': [6.0, -5.2, -90]},
        'c': {'kind': 'transfer', 'pose': [-7.2, 4.0, 180]},
        'packing': {'kind': 'transfer', 'pose': [6.0, 5.2, 90]},
    },
    'zones': [
        {'name': 'racks', 'kind': 'keepout',
         'polygon': [[0.9, -4.6], [4.1, -4.6], [4.1, -1.4], [0.9, -1.4]]},
        {'name': 'packing_area', 'kind': 'speed', 'limit': 0.3,
         'polygon': [[4.0, 3.0], [7.9, 3.0], [7.9, 5.9], [4.0, 5.9]]},
    ],
}


class GridNavigator:
    """Stands in for Nav2: plan on the grid, follow, replan when blocked, give up in time."""

    def __init__(self, grid, site, speed=0.8, patience=12.0, stuck_after=25.0, sight=4.0):
        self.grid = grid
        self.site = site
        self.speed = speed
        self.patience = patience          # how long it looks for a way before giving up
        self.stuck_after = stuck_after    # how long without progress before giving up
        self.sight = sight                # obstacles nearer than this are planned around
        self.static = grid.inflated(CLEARANCE) | keepout_mask(site, grid).occupied
        self.state = NavState.IDLE
        self.goal = None
        self.path = None
        self.plans = 0
        # Obstacles seen once are remembered, as Nav2's costmap marks them, until a look at
        # the place they stood shows them gone.
        self.known = []
        self._no_way_since = None
        self._progress = None

    def route(self, start, goal):
        """Answer a route request the way Nav2's planner would: a length, or why not."""
        points = self.grid.plan((start.x, start.y), (goal.x, goal.y), 0.0,
                                extra_blocked=self.static)
        if points is None:
            return RouteReply(None, 'no path')
        return RouteReply(path_length(points))

    def go(self, goal, pose, now):
        self.goal, self.state, self.path = goal, NavState.ACTIVE, None
        self._no_way_since = None
        self._progress = (now, pose)

    def cancel(self):
        if self.state is NavState.ACTIVE:
            self.state = NavState.CANCELED

    @property
    def distance_remaining(self):
        return None if self.path is None else path_length(self.path)

    def _blocked(self, pose, obstacles):
        blocked = self.static
        in_sight = [o for o in obstacles if math.hypot(o.x - pose.x, o.y - pose.y) <= self.sight]
        self.known = [o for o in self.known
                      if o in obstacles or math.hypot(o.x - pose.x, o.y - pose.y) > self.sight]
        self.known += [o for o in in_sight if o not in self.known]
        seen = self.known
        if not seen:
            return blocked
        blocked = blocked.copy()
        for obstacle in seen:
            reach = CLEARANCE + (obstacle.radius if isinstance(obstacle, Disc)
                                 else math.hypot(obstacle.length, obstacle.width) / 2.0)
            c0, r0 = self.grid.cell_of(obstacle.x - reach, obstacle.y - reach)
            c1, r1 = self.grid.cell_of(obstacle.x + reach, obstacle.y + reach)
            for row in range(max(0, r0), min(self.grid.height, r1 + 1)):
                for col in range(max(0, c0), min(self.grid.width, c1 + 1)):
                    x, y = self.grid.centre_of(col, row)
                    if isinstance(obstacle, Disc):
                        near = math.hypot(x - obstacle.x, y - obstacle.y) <= reach
                    else:
                        grown = Box(obstacle.x, obstacle.y, obstacle.length + 2 * CLEARANCE,
                                    obstacle.width + 2 * CLEARANCE, obstacle.yaw)
                        near = grown.contains((x, y))
                    if near:
                        blocked[row, col] = True
        return blocked

    def update(self, pose, now, obstacles=()):
        """Return (v, w), or None when navigation has nothing to command."""
        if self.state is not NavState.ACTIVE:
            return None
        target = self.goal.pose
        here = (pose.x, pose.y)
        if self._progress is None or pose.distance_to(self._progress[1]) > 0.3:
            self._progress = (now, pose)
        elif now - self._progress[0] > self.stuck_after:
            self.state = NavState.FAILED
            return None

        if pose.distance_to(target) <= 0.06:
            error = angle_diff(target.yaw, pose.yaw)
            if abs(error) <= 0.06:
                self.state = NavState.SUCCEEDED
                return (0.0, 0.0)
            self._progress = (now, pose)                      # turning is progress
            return (0.0, math.copysign(min(1.0, max(0.15, 2.0 * abs(error))), error))

        blocked = self._blocked(pose, obstacles)
        clear = self.path is not None and all(
            self.grid.line_is_clear(a, b, blocked)
            for a, b in zip([here] + self.path[1:], self.path[1:]))
        if not clear:
            route = self.grid.plan(here, (target.x, target.y), 0.0, extra_blocked=blocked)
            self.plans += 1
            if route is None:
                self.path = None
                if self._no_way_since is None:
                    self._no_way_since = now
                elif now - self._no_way_since > self.patience:
                    self.state = NavState.FAILED
                return (0.0, 0.0)
            self._no_way_since = None
            self.path = route
        # Drop the corners already reached, then steer at the next one.
        while len(self.path) > 2 and math.hypot(
                self.path[1][0] - pose.x, self.path[1][1] - pose.y) < 0.25:
            self.path = [self.path[0]] + self.path[2:]
        self.path[0] = here
        aim = self.path[1]
        distance = math.hypot(aim[0] - pose.x, aim[1] - pose.y)
        error = angle_diff(math.atan2(aim[1] - pose.y, aim[0] - pose.x), pose.yaw)
        if abs(error) > 0.5:
            return (0.0, math.copysign(1.0, error))
        limit = self.site.speed_limit_at(here) or self.speed
        final = len(self.path) == 2
        speed = min(self.speed, limit, 0.9 * distance + 0.05 if final else self.speed)
        return (speed * max(0.2, math.cos(error)), max(-1.2, min(1.2, 2.5 * error)))


class World:
    """The mission, the vehicle and everything between them, in a hall."""

    def __init__(self, at='charger', loaded=False, battery=0.8, config=None, dt=0.05,
                 site=None, grid=None, scan_every=2):
        self.site = Site.from_dict(site or HALL_SITE)
        self.grid = grid or hall()
        self.dt = dt
        self.now = 50.0
        start = self.site.station(at).pose if isinstance(at, str) else at
        self.vehicle = KinematicVehicle(start)
        self.obstacles = []
        self.scanner = ScannerSim(
            self.grid, ScannerConfig(beams=360, angle_increment=2.0 * math.pi / 360.0,
                                     range_max=8.0, noise=0.01, deck_posts=True))
        self.scanner.obstacles = self.obstacles
        self.safety = SafetyMonitor()
        self.navigator = GridNavigator(self.grid, self.site)
        self.handler = LoadHandlerSim(loaded=loaded)
        self.battery = BatterySim(
            BatteryConfig(), battery, chargers=[c.pose for c in self.site.chargers])
        self.config = config or MissionConfig()
        self.mission = Mission(
            self.site, self.config,
            battery_range=lambda level: BatteryConfig().range_m(
                level, self.config.battery_critical, self.config.cruise_speed))
        self.scan_every = scan_every
        self._scan = None
        self._steps = 0
        self.routes = {}
        self.results, self.statuses, self.safety_states = [], [], []
        self.track = []                      # (now, x, y, yaw, v, safety state)
        self.collided_at = None
        self.smallest_gap = math.inf         # nearest the outline ever came to a scan return

    def order(self, pickup='', dropoff='', **options):
        return self.mission.start(Order(pickup, dropoff, **options), self._inputs())

    def _inputs(self):
        nav = self.navigator
        return MissionInputs(
            now=self.now, pose=self.vehicle.pose, nav=nav.state,
            nav_goal_id=nav.goal.goal_id if nav.goal else 0,
            nav_distance_remaining=nav.distance_remaining,
            load=self.handler.state, battery=self.battery.level, charging=self.battery.charging,
            safety=self.safety_states[-1].state if self.safety_states else self._idle_state(),
            safety_reason=self.safety_states[-1].reason if self.safety_states else '',
            command_source=self.safety_states[-1].source if self.safety_states else 'none',
            routes=dict(self.routes))

    def _idle_state(self):
        from agv_core.safety import SafetyState
        return SafetyState.ESTOP if self.safety.estop_active else SafetyState.OK

    def _read_scanner(self):
        """Scan, and take the vehicle's own posts out again as the scan filter does."""
        cfg = self.scanner.config
        ranges = self.scanner.scan(self.vehicle.pose)
        angles = cfg.angles
        with np.errstate(invalid='ignore'):
            x, y = ranges * np.cos(angles), ranges * np.sin(angles)
        own = np.isfinite(ranges) & (np.abs(x) <= AGV.half_length + 0.02) \
            & (np.abs(y) <= AGV.half_width + 0.02)
        ranges = np.where(own, np.nan, ranges)
        return Scan.from_ranges(self.now, ranges, cfg.angle_min, cfg.angle_increment,
                                cfg.range_min, cfg.range_max)

    def step(self):
        if self._steps % self.scan_every == 0:
            self._scan = self._read_scanner()
        self._steps += 1

        out = self.mission.update(self._inputs())
        self.statuses.append(out.status)
        if out.result is not None:
            self.results.append(out.result)
        for request in out.route_requests:
            self.routes[request.request_id] = self.navigator.route(request.start, request.goal)
        if out.cancel_navigation:
            self.navigator.cancel()
        if out.navigate is not None:
            self.navigator.go(out.navigate, self.vehicle.pose, self.now)
        if out.load_command is not None:
            self.handler.command(out.load_command.action, self.now)

        command = self.navigator.update(self.vehicle.pose, self.now, self.obstacles)
        if out.velocity is not None:
            command = out.velocity
        nav = None if command is None else Command(self.now, command[0], command[1])
        status = self.safety.update(
            self.now, self._scan, Command(self.now, self.vehicle.v, self.vehicle.w), nav=nav,
            docking=out.velocity is not None)
        self.safety_states.append(status)
        self.vehicle.command(status.v, status.w, self.now)

        before = self.vehicle.distance
        self.vehicle.step(self.dt, self.now)
        self.handler.step(self.now)
        self.battery.step(self.dt, self.vehicle.pose, self.vehicle.distance - before,
                          self.vehicle.v)
        if self.collided_at is None and footprint_collides(
                self.vehicle.pose, self.grid, self.obstacles):
            self.collided_at = (self.now, self.vehicle.pose)
        if self._scan.points.size:
            px, py = np.abs(self._scan.points[:, 0]), np.abs(self._scan.points[:, 1])
            gaps = np.hypot(np.maximum(px - AGV.half_length, 0.0),
                            np.maximum(py - AGV.half_width, 0.0))
            self.smallest_gap = min(self.smallest_gap, float(gaps.min()))
        self.track.append((self.now, self.vehicle.pose.x, self.vehicle.pose.y,
                           self.vehicle.pose.yaw, self.vehicle.v, status.state))
        self.now += self.dt
        return out

    def run(self, seconds):
        for _ in range(int(round(seconds / self.dt))):
            self.step()

    def run_until(self, condition, timeout=900.0):
        for _ in range(int(round(timeout / self.dt))):
            if condition():
                return
            self.step()
        status = self.statuses[-1]
        raise AssertionError(
            f'never happened within {timeout} s; state {status.state!r}: {status.detail!r}; '
            f'safety {self.safety_states[-1].state.value}: {self.safety_states[-1].reason!r}')

    def run_to_end(self, timeout=900.0):
        self.run_until(lambda: self.results, timeout)
        return self.results[-1]

    def run_until_state(self, state, timeout=600.0):
        self.run_until(lambda: self.statuses and self.statuses[-1].state == state, timeout)

    @property
    def status(self):
        return self.statuses[-1] if self.statuses else MissionStatus()

    def docked_at(self, name):
        return is_docked(self.vehicle.pose, self.site.station(name).pose)

    def docking_error(self, name):
        local = self.vehicle.pose.relative_to(self.site.station(name).pose)
        return abs(local.y), abs(math.degrees(local.yaw)), abs(local.x)

    def ahead(self, distance, lateral=0.0):
        """Return the point `distance` ahead of the vehicle's centre, for putting things there."""
        return self.vehicle.pose.to_parent((distance, lateral))


def path_in_zone(world, zone_name):
    """Return the track samples (x, y) that lie inside a zone."""
    zone = next(z for z in world.site.zones if z.name == zone_name)
    return [(x, y) for _, x, y, *_ in world.track if zone.contains((x, y))]


__all__ = ['Box', 'Disc', 'GridNavigator', 'hall', 'HALL_SITE', 'path_in_zone', 'Pose2D', 'World']
