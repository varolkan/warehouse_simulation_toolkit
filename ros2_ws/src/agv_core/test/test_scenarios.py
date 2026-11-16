"""Whole orders in a hall, with the scanner, the safety gate and navigation in the loop.

Each test drives the vehicle through world.py's hall and puts something in its way. Two
things are checked in all of them: the order ends the way it should, and the vehicle's
outline never touches anything.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agv_core.geometry import point_segment_distance  # noqa: E402
from agv_core.mission import MissionConfig  # noqa: E402
from agv_core.safety import SafetyState  # noqa: E402
from agv_core.simulation import LoadState  # noqa: E402
from agv_core.vehicle import AGV  # noqa: E402
import pytest  # noqa: E402
from world import Box, Disc, path_in_zone, World  # noqa: E402


def crossings(world):
    """Return the x at which the track crossed the wall line (y = 0), in order."""
    xs = []
    for (_, x1, y1, *_), (_, x2, y2, *_) in zip(world.track, world.track[1:]):
        if (y1 < 0.0) != (y2 < 0.0):
            xs.append((x1 + x2) / 2.0)
    return xs


def states_of(world):
    seen = []
    for status in world.statuses:
        if not seen or seen[-1] != status.state:
            seen.append(status.state)
    return seen


def never_touched(world):
    assert world.collided_at is None, f'touched something at {world.collided_at}'


# -- deliveries -------------------------------------------------------------------------------


def test_a_delivery_through_the_hall():
    world = World()
    assert world.order('a', 'packing') is None
    result = world.run_to_end()

    assert result.success and result.message == "delivered from 'a' to 'packing'"
    assert states_of(world) == ['checking', 'to_pickup', 'docking', 'loading', 'to_dropoff',
                                'docking', 'unloading', 'complete']
    never_touched(world)
    assert world.smallest_gap > 0.05
    # Round the east end of the wall, the short way.
    assert len(crossings(world)) == 1 and crossings(world)[0] > 5.0
    offset, heading, along = world.docking_error('packing')
    assert offset <= 0.02 and heading <= 1.5 and along <= 0.02
    assert world.handler.state is LoadState.EMPTY
    assert 18.0 < result.distance < 26.0 and result.duration < 120.0


def test_the_route_stays_out_of_the_keepout_zone():
    world = World()
    world.order('a', 'b')                     # the racks stand between the two
    assert world.run_to_end().success
    never_touched(world)
    assert path_in_zone(world, 'racks') == []


def test_the_speed_limit_of_a_zone_is_kept():
    """Applied on entry, as Nav2's speed filter does: held once the vehicle has braked to it."""
    world = World()
    world.order('a', 'packing')
    assert world.run_to_end().success
    zone = next(z for z in world.site.zones if z.name == 'packing_area')
    edges = list(zip(zone.polygon, zone.polygon[1:] + zone.polygon[:1]))

    def depth(x, y):
        return min(point_segment_distance((x, y), a, b) for a, b in edges)

    # From 0.8 m/s to 0.3 m/s at the drive's 2.5 m/s^2 takes 0.2 s and 0.11 m.
    braked = [abs(v) for _, x, y, _, v, _ in world.track
              if zone.contains((x, y)) and depth(x, y) > 0.15]
    entering = [abs(v) for _, x, y, _, v, _ in world.track
                if zone.contains((x, y)) and depth(x, y) <= 0.15]
    outside = [abs(v) for _, x, y, _, v, _ in world.track if y < 2.5]
    assert braked and max(braked) <= zone.limit + 0.01
    assert entering and max(entering) < 0.8
    assert max(outside) > 0.7                  # it does drive faster where it may


def test_docking_precision_in_the_hall():
    worst_offset = worst_heading = worst_along = 0.0
    for pickup, dropoff in (('a', 'b'), ('b', 'c'), ('c', 'packing'), ('packing', 'a')):
        world = World(at=pickup)
        world.order(pickup, dropoff)
        assert world.run_to_end().success, (pickup, dropoff)
        never_touched(world)
        offset, heading, along = world.docking_error(dropoff)
        worst_offset = max(worst_offset, offset)
        worst_heading = max(worst_heading, heading)
        worst_along = max(worst_along, along)
    assert worst_offset <= 0.02 and worst_heading <= 1.5 and worst_along <= 0.02


def test_a_shift_of_orders():
    """Six deliveries one after another, the vehicle starting each where the last ended."""
    world = World()
    tour = [('a', 'packing'), ('packing', 'c'), ('c', 'b'), ('b', 'a'), ('a', 'c'),
            ('c', 'packing')]
    for pickup, dropoff in tour:
        world.results.clear()
        assert world.order(pickup, dropoff) is None, world.status
        result = world.run_to_end()
        assert result.success, (pickup, dropoff, result.message)
        assert world.docked_at(dropoff)
    never_touched(world)
    assert world.battery.level < 0.8            # it cost something


# -- things in the way ------------------------------------------------------------------------


def test_a_passage_blocked_by_a_pallet_is_driven_round():
    free = World()
    free.order('a', 'packing')
    direct = free.run_to_end()

    world = World()
    world.obstacles.append(Box(6.5, 0.0, 2.9, 1.2))            # across the east passage
    world.order('a', 'packing')
    result = world.run_to_end()
    assert result.success
    never_touched(world)
    # It went to the east passage, found it shut, and came back round the west end.
    assert crossings(world) and crossings(world)[-1] < -5.0
    assert result.distance > direct.distance + 8.0
    assert world.navigator.plans > free.navigator.plans


def test_someone_stepping_out_in_front_of_the_vehicle_is_not_hit():
    world = World()
    world.order('a', 'packing')
    world.run_until(lambda: world.status.state == 'to_dropoff' and world.vehicle.v > 0.75)
    # A person 0.35 m in front of the bumper, at 0.8 m/s.
    x, y = world.ahead(AGV.half_length + 0.35 + 0.25)
    person = Disc(x, y, 0.25)
    world.obstacles.append(person)
    world.run(4.0)
    stopped = [s for s in world.safety_states if s.state is SafetyState.PROTECTIVE_STOP]
    assert stopped and 'm ahead' in stopped[0].reason
    never_touched(world)
    # Stopped short of the person: bumper to the person's edge, measured along the heading.
    ahead_of_bumper = world.vehicle.pose.to_local((x, y))[0] - AGV.half_length - person.radius
    assert 0.0 < ahead_of_bumper < 0.35
    world.obstacles.remove(person)
    result = world.run_to_end()
    assert result.success
    never_touched(world)


def test_it_never_creeps_towards_someone_standing_in_front_of_it():
    world = World()
    world.order('a', 'packing')
    world.run_until(lambda: world.status.state == 'to_dropoff' and world.vehicle.v > 0.75)
    x, y = world.ahead(AGV.half_length + 0.35 + 0.25)
    person = Disc(x, y, 0.25)
    world.obstacles.append(person)
    world.run_until(lambda: abs(world.vehicle.v) < 1e-3, timeout=5.0)

    def gap():
        """Return the distance from the vehicle's outline to the person's edge."""
        lx, ly = world.vehicle.pose.to_local((x, y))
        dx = max(abs(lx) - AGV.half_length, 0.0)
        dy = max(abs(ly) - AGV.half_width, 0.0)
        return math.hypot(dx, dy) - person.radius

    def centre_distance():
        return math.hypot(world.vehicle.pose.x - x, world.vehicle.pose.y - y)

    stopped_at = centre_distance()
    closest_centre, closest_gap = stopped_at, gap()
    for _ in range(int(20.0 / world.dt)):          # they stand there for 20 s
        world.step()
        closest_centre = min(closest_centre, centre_distance())
        closest_gap = min(closest_gap, gap())
    # It may turn on the spot to go round them, which swings its front corners out to the
    # turning circle; its centre does not move towards them.
    assert closest_centre >= stopped_at - 0.01
    assert closest_gap > 0.1
    never_touched(world)
    world.obstacles.remove(person)
    assert world.run_to_end().success
    never_touched(world)


def test_with_no_way_through_the_order_fails_and_nothing_is_hit():
    world = World()
    world.obstacles.append(Box(6.5, 0.0, 2.9, 1.2))
    world.obstacles.append(Box(-6.5, 0.0, 2.9, 1.2))
    world.order('a', 'packing')
    result = world.run_to_end(timeout=1500.0)
    assert result.outcome == 'failed' and not result.delivered
    assert result.message == ("could not reach 'packing' in 3 attempts: navigation gave up "
                              '(the load is on board)')
    never_touched(world)
    assert world.handler.state is LoadState.LOADED
    assert not crossings(world)                 # it never got to the other side


def test_the_way_opening_again_lets_the_order_finish():
    world = World()
    east, west = Box(6.5, 0.0, 2.9, 1.2), Box(-6.5, 0.0, 2.9, 1.2)
    world.obstacles += [east, west]
    world.order('a', 'packing')
    world.run_until(lambda: 'trying again' in world.status.detail, timeout=900.0)
    world.obstacles.remove(west)
    result = world.run_to_end(timeout=900.0)
    assert result.success
    never_touched(world)


def test_something_standing_in_the_docking_run_fails_the_dock_not_the_bumper():
    world = World(config=MissionConfig(dock_timeout=15.0))
    world.order('a', 'b')
    world.run_until_state('to_dropoff')
    station = world.site.station('b')
    crate = station.pose.advanced(-0.05)                     # on the stop line
    world.obstacles.append(Box(crate.x, crate.y, 0.3, 0.3))
    result = world.run_to_end(timeout=1500.0)
    assert result.outcome == 'failed'
    assert "'b'" in result.message and '(the load is on board)' in result.message
    never_touched(world)
    assert world.handler.state is LoadState.LOADED            # never unloaded onto a crate


# -- the operator's buttons -------------------------------------------------------------------


def test_the_estop_stops_the_vehicle_and_the_order_carries_on_after_the_reset():
    world = World()
    world.order('a', 'packing')
    world.run_until(lambda: world.status.state == 'to_dropoff' and world.vehicle.v > 0.75)
    world.safety.press_estop()
    pressed_at = world.vehicle.pose
    world.run(3.0)
    assert world.vehicle.v == 0.0 and world.vehicle.w == 0.0
    # 0.8 m/s, braking at 2.5 m/s^2: stopped within a vehicle length.
    assert world.vehicle.pose.distance_to(pressed_at) < 0.3
    assert world.status.holding == 'e-stop active' and world.status.active
    world.run(20.0)
    assert world.vehicle.pose.distance_to(pressed_at) < 0.3
    world.safety.release_estop()
    assert world.safety.reset()[0]
    result = world.run_to_end()
    assert result.success
    never_touched(world)


def test_cancelling_brings_the_vehicle_and_its_load_home():
    world = World()
    world.order('a', 'packing')
    world.run_until_state('to_dropoff')
    world.run(3.0)
    assert world.mission.cancel()
    result = world.run_to_end()
    assert result.outcome == 'canceled'
    assert result.message == 'cancelled with the load on board, back at the charger'
    assert world.docked_at('charger') and world.handler.state is LoadState.LOADED
    never_touched(world)


# -- the battery ------------------------------------------------------------------------------


def test_a_low_battery_takes_the_idle_vehicle_to_its_charger():
    world = World(at='packing', battery=0.28)
    world.run_until(lambda: world.docked_at('charger') and world.battery.charging, timeout=300.0)
    level = world.battery.level
    world.run(60.0)
    assert world.battery.level > level
    assert world.results == []
    never_touched(world)


def test_a_delivery_on_a_low_battery_ends_at_the_charger():
    world = World(battery=0.27)
    world.order('a', 'packing')
    result = world.run_to_end()
    assert result.message == "delivered from 'a' to 'packing', back at the charger"
    assert world.docked_at('charger') and world.battery.charging
    never_touched(world)


@pytest.mark.parametrize('start', ['charger', 'a', 'b', 'c', 'packing'])
def test_every_station_is_reachable_from_every_other(start):
    world = World(at=start)
    for name, station in world.site.stations.items():
        if name == start:
            continue
        reply = world.navigator.route(world.site.station(start).pose, station.approach_pose)
        assert reply.length is not None, (start, name, reply.error)
        straight = world.site.station(start).pose.distance_to(station.approach_pose)
        assert reply.length >= straight - 0.1 and math.isfinite(reply.length)
