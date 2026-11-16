"""The stand-ins: kinematic drive, obstacles, scanner, load handler, battery at the charger."""

import math

from agv_core.battery import BatteryConfig
from agv_core.geometry import Pose2D
from agv_core.grid import FREE, GridMap, OCCUPIED
from agv_core.simulation import (
    BatterySim, Box, Disc, DriveConfig, footprint_collides, KinematicVehicle, LoadHandlerSim,
    LoadState, ScannerConfig, ScannerSim)
from agv_core.vehicle import AGV
import numpy as np
import pytest


def room(width=12.0, height=8.0):
    data = np.full((int(height / 0.05), int(width / 0.05)), FREE, dtype=np.int16)
    data[0, :] = data[-1, :] = OCCUPIED
    data[:, 0] = data[:, -1] = OCCUPIED
    return GridMap(data, 0.05, (-width / 2.0, -height / 2.0))


def run(vehicle, seconds, v, w, dt=0.01, start=0.0):
    now = start
    for _ in range(int(round(seconds / dt))):
        vehicle.command(v, w, now)
        vehicle.step(dt, now)
        now += dt
    return now


# -- the drive --------------------------------------------------------------------------------


def test_the_drive_accelerates_at_its_limit_and_holds_the_speed():
    vehicle = KinematicVehicle()
    run(vehicle, 0.2, 1.0, 0.0)
    assert vehicle.v == pytest.approx(0.5, abs=0.03)        # 2.5 m/s^2 for 0.2 s
    run(vehicle, 1.0, 1.0, 0.0, start=0.2)
    assert vehicle.v == pytest.approx(1.0)
    assert vehicle.pose.y == pytest.approx(0.0) and vehicle.pose.x > 0.9


def test_commands_are_clamped_to_what_the_drive_can_do():
    cfg = DriveConfig()
    vehicle = KinematicVehicle()
    run(vehicle, 2.0, 9.0, 9.0)
    assert vehicle.v == pytest.approx(cfg.max_speed)
    assert vehicle.w == pytest.approx(cfg.max_turn_rate)
    run(vehicle, 2.0, -9.0, -9.0, start=2.0)
    assert vehicle.v == pytest.approx(-cfg.max_reverse_speed)
    assert vehicle.w == pytest.approx(-cfg.max_turn_rate)


def test_without_fresh_commands_the_drive_stops():
    vehicle = KinematicVehicle()
    now = run(vehicle, 1.0, 0.8, 0.0)
    assert vehicle.v == pytest.approx(0.8)
    for _ in range(100):                      # a second with nobody commanding
        vehicle.step(0.01, now)
        now += 0.01
    assert vehicle.v == 0.0


def test_a_command_that_is_not_a_number_stops_the_drive():
    vehicle = KinematicVehicle()
    now = run(vehicle, 1.0, 0.8, 0.0)
    vehicle.command(float('nan'), 0.0, now)
    for _ in range(100):
        vehicle.step(0.01, now)
    assert vehicle.v == 0.0


def test_a_constant_turn_drives_a_circle():
    # 0.5 m/s at 0.5 rad/s is a 1 m circle: a quarter of it ends at (1, 1) facing north.
    vehicle = KinematicVehicle(config=DriveConfig(max_accel=1e6, max_turn_accel=1e6))
    run(vehicle, math.pi, 0.5, 0.5, dt=0.001)
    assert (vehicle.pose.x, vehicle.pose.y) == pytest.approx((1.0, 1.0), abs=0.01)
    assert vehicle.pose.yaw == pytest.approx(math.pi / 2.0, abs=0.01)
    assert vehicle.distance == pytest.approx(math.pi / 2.0, abs=0.01)


def test_reversing_counts_as_distance_driven():
    vehicle = KinematicVehicle(config=DriveConfig(max_accel=1e6))
    run(vehicle, 2.0, -0.3, 0.0)
    assert vehicle.pose.x == pytest.approx(-0.6, abs=0.01)
    assert vehicle.distance == pytest.approx(0.6, abs=0.01)


def test_odometry_starts_at_zero_wherever_the_vehicle_is():
    vehicle = KinematicVehicle(Pose2D(5.0, -3.0, math.pi / 2.0),
                               DriveConfig(max_accel=1e6))
    run(vehicle, 2.0, 0.5, 0.0)
    assert (vehicle.pose.x, vehicle.pose.y) == pytest.approx((5.0, -2.0), abs=0.01)
    odom = (vehicle.odom.x, vehicle.odom.y, vehicle.odom.yaw)
    assert odom == pytest.approx((1.0, 0.0, 0.0), abs=0.01)


def test_odometry_errors_make_it_drift_from_the_truth():
    cfg = DriveConfig(max_accel=1e6, max_turn_accel=1e6, odom_distance_scale=1.02,
                      odom_turn_scale=0.97)
    vehicle = KinematicVehicle(config=cfg)
    run(vehicle, 10.0, 0.5, 0.0)
    assert vehicle.pose.x == pytest.approx(5.0, abs=0.01)
    assert vehicle.odom.x == pytest.approx(5.1, abs=0.01)
    spin = KinematicVehicle(config=cfg)
    run(spin, 2.0, 0.0, 1.0)
    assert spin.pose.yaw == pytest.approx(2.0, abs=0.01)
    assert spin.odom.yaw == pytest.approx(1.94, abs=0.01)


def test_placing_the_vehicle_does_not_tell_odometry():
    vehicle = KinematicVehicle(config=DriveConfig(max_accel=1e6))
    run(vehicle, 1.0, 0.5, 0.0)
    before = vehicle.odom
    vehicle.place(Pose2D(9.0, 9.0, 1.0))
    assert vehicle.pose == Pose2D(9.0, 9.0, 1.0) and vehicle.v == 0.0
    assert vehicle.odom == before


# -- obstacles --------------------------------------------------------------------------------


def test_a_ray_at_a_disc():
    disc = Disc(3.0, 0.0, 0.5)
    angles = np.radians([0.0, 5.0, 12.0, 180.0])
    ranges = disc.ray_ranges((0.0, 0.0), angles)
    assert ranges[0] == pytest.approx(2.5)
    assert 2.5 < ranges[1] < 2.6
    assert np.isinf(ranges[2]) and np.isinf(ranges[3])      # past its edge; behind
    assert disc.contains((3.2, 0.3)) and not disc.contains((3.6, 0.0))
    assert all(math.hypot(x - 3.0, y) == pytest.approx(0.5) for x, y in disc.outline())


def test_a_ray_at_a_box():
    box = Box(3.0, 0.0, 1.0, 2.0)                             # x 2.5..3.5, y -1..1
    angles = np.radians([0.0, 20.0, 30.0, 90.0, 180.0])
    ranges = box.ray_ranges((0.0, 0.0), angles)
    assert ranges[0] == pytest.approx(2.5)
    assert ranges[1] == pytest.approx(2.5 / math.cos(math.radians(20.0)))
    assert np.isinf(ranges[2])                                # passes the corner at 21.8 degrees
    assert np.isinf(ranges[3]) and np.isinf(ranges[4])
    assert box.contains((3.4, -0.9)) and not box.contains((3.6, 0.0))


def test_a_turned_box():
    box = Box(0.0, 3.0, 2.0, 0.4, yaw=math.radians(45.0))
    straight_up = box.ray_ranges((0.0, 0.0), np.array([math.pi / 2.0]))[0]
    assert straight_up == pytest.approx(3.0 - 0.2 * math.sqrt(2.0))
    assert box.contains((0.6, 3.6)) and not box.contains((0.6, 2.4))
    corners = box.outline()
    assert max(math.hypot(x, y - 3.0) for x, y in corners) == pytest.approx(math.hypot(1.0, 0.2))


def test_rays_parallel_to_a_box_face():
    box = Box(3.0, 0.0, 1.0, 1.0)
    level = box.ray_ranges((0.0, 0.2), np.array([0.0]))[0]
    above = box.ray_ranges((0.0, 0.8), np.array([0.0]))[0]
    assert level == pytest.approx(2.5) and np.isinf(above)


# -- the scanner ------------------------------------------------------------------------------


def clean_scanner(grid, **options):
    return ScannerSim(grid, ScannerConfig(noise=0.0, deck_posts=False, **options))


def beam(config, degrees):
    """Return the index of the beam pointing `degrees` from straight ahead."""
    return int(round((math.radians(degrees) - config.angle_min) / config.angle_increment))


def test_the_scanner_sees_the_walls_where_they_are():
    scanner = clean_scanner(room())
    ranges = scanner.scan(Pose2D(1.0, 0.5, 0.0))
    cfg = scanner.config
    assert ranges.shape == (720,)
    # Inner wall faces at x = +-5.95, y = +-3.95; rays are sampled every 2.5 cm.
    for degrees, true in ((0.0, 4.95), (90.0, 3.45), (180.0, 6.95), (-90.0, 4.45)):
        assert ranges[beam(cfg, degrees) % 720] == pytest.approx(true, abs=0.06)


def test_the_scan_turns_with_the_vehicle():
    scanner = clean_scanner(room())
    ahead = beam(scanner.config, 0.0)
    assert scanner.scan(Pose2D(0.0, 0.0, 0.0))[ahead] == pytest.approx(5.95, abs=0.06)
    assert scanner.scan(Pose2D(0.0, 0.0, math.pi / 2.0))[ahead] == pytest.approx(3.95, abs=0.06)


def test_the_scanner_sees_obstacles_that_are_not_in_the_map():
    scanner = clean_scanner(room())
    scanner.obstacles.append(Disc(2.0, 0.0, 0.25))
    scanner.obstacles.append(Box(0.0, -2.0, 0.6, 0.4))
    ranges = scanner.scan(Pose2D())
    cfg = scanner.config
    assert ranges[beam(cfg, 0.0)] == pytest.approx(1.75)
    assert ranges[beam(cfg, -90.0)] == pytest.approx(1.8)
    assert ranges[beam(cfg, 90.0)] == pytest.approx(3.95, abs=0.06)     # nothing new that way
    scanner.obstacles.clear()
    assert scanner.scan(Pose2D())[beam(cfg, 0.0)] == pytest.approx(5.95, abs=0.06)


def test_beyond_its_range_the_scanner_reports_nothing_there():
    scanner = clean_scanner(room(), range_max=3.0)
    ranges = scanner.scan(Pose2D())
    cfg = scanner.config
    assert np.isposinf(ranges[beam(cfg, 0.0)])
    assert np.isposinf(ranges[beam(cfg, 90.0)])


def test_closer_than_it_can_measure_the_scanner_reports_too_close():
    scanner = clean_scanner(room())
    scanner.obstacles.append(Disc(0.3, 0.0, 0.25))            # surface 5 cm from the scanner
    assert np.isneginf(scanner.scan(Pose2D())[beam(scanner.config, 0.0)])


def test_the_scanner_sees_the_posts_that_carry_the_deck():
    scanner = ScannerSim(room(), ScannerConfig(noise=0.0, deck_posts=True))
    ranges = scanner.scan(Pose2D(0.0, 0.0, 0.7))             # wherever it points
    near = np.nonzero(np.isfinite(ranges) & (ranges < 0.5))[0]
    assert near.size >= 8
    bearings = np.degrees(scanner.config.angles[near])
    distance = math.hypot(AGV.half_length - 0.015, AGV.half_width - 0.015)
    assert ranges[near] == pytest.approx(distance, abs=0.012)
    # Four posts, one in each quadrant, at the corners' bearing.
    corner = math.degrees(math.atan2(AGV.half_width - 0.015, AGV.half_length - 0.015))
    for expected in (corner, 180.0 - corner, corner - 180.0, -corner):
        assert np.any(np.abs(bearings - expected) < 2.0)
    without = clean_scanner(room()).scan(Pose2D(0.0, 0.0, 0.7))
    assert not np.any(np.isfinite(without) & (without < 0.5))


def test_scanner_noise_is_small_and_repeatable():
    noisy = ScannerSim(room(), ScannerConfig(noise=0.01, deck_posts=False), seed=7)
    clean = clean_scanner(room())
    difference = noisy.scan(Pose2D()) - clean.scan(Pose2D())
    assert 0.005 < np.std(difference) < 0.015 and abs(np.mean(difference)) < 0.003
    again = ScannerSim(room(), ScannerConfig(noise=0.01, deck_posts=False), seed=7)
    assert np.array_equal(again.scan(Pose2D()), ScannerSim(
        room(), ScannerConfig(noise=0.01, deck_posts=False), seed=7).scan(Pose2D()))


def test_whether_the_vehicle_touches_something():
    grid = room()
    assert not footprint_collides(Pose2D(0.0, 0.0, 0.3), grid)
    # Front bumper 2 cm into the east wall (inner face at x = 5.95).
    assert footprint_collides(Pose2D(5.95 - AGV.half_length + 0.02, 0.0, 0.0), grid)
    assert not footprint_collides(Pose2D(5.95 - AGV.half_length - 0.06, 0.0, 0.0), grid)
    # Turned 90 degrees the same spot is clear: the vehicle is narrower than it is long.
    assert not footprint_collides(Pose2D(5.95 - AGV.half_length + 0.02, 0.0, math.pi / 2), grid)
    crate = Box(1.0, 0.0, 0.4, 0.4)
    assert footprint_collides(Pose2D(0.55, 0.0, 0.0), grid, [crate])
    assert not footprint_collides(Pose2D(0.4, 0.0, 0.0), grid, [crate])
    # Small enough to be under the vehicle without touching its outline.
    assert footprint_collides(Pose2D(), grid, [Disc(0.05, 0.02, 0.03)])


# -- the load handler -------------------------------------------------------------------------


def test_a_transfer_takes_its_time():
    handler = LoadHandlerSim(duration=3.0)
    assert handler.state is LoadState.EMPTY
    assert handler.command('load', 10.0) == (True, '')
    assert handler.step(10.1) is LoadState.MOVING
    assert handler.step(12.9) is LoadState.MOVING
    assert handler.step(13.0) is LoadState.LOADED
    assert handler.command('unload', 20.0) == (True, '')
    assert handler.step(23.0) is LoadState.EMPTY
    assert handler.commands == [(10.0, 'load'), (20.0, 'unload')]


def test_asking_for_what_is_already_so_is_accepted_and_changes_nothing():
    handler = LoadHandlerSim(loaded=True)
    assert handler.command('load', 0.0) == (True, 'already loaded')
    assert handler.state is LoadState.LOADED


def test_a_handler_that_is_not_ready_refuses_and_then_works():
    handler = LoadHandlerSim(rejections=2)
    assert handler.command('load', 0.0) == (False, 'the load handler is not ready')
    assert handler.command('load', 1.0)[0] is False
    assert handler.state is LoadState.EMPTY
    assert handler.command('load', 2.0)[0] is True
    assert handler.step(5.0) is LoadState.LOADED


def test_a_jammed_handler_starts_and_never_finishes():
    handler = LoadHandlerSim(stuck=True)
    assert handler.command('load', 0.0)[0] is True
    assert handler.step(1000.0) is LoadState.MOVING
    assert handler.command('load', 1001.0) == (False, 'a transfer is already under way')


def test_an_unknown_command_is_refused():
    assert LoadHandlerSim().command('juggle', 0.0) == (False, "unknown load command 'juggle'")


def test_a_load_can_be_lost():
    handler = LoadHandlerSim(loaded=True)
    handler.drop()
    assert handler.step(0.0) is LoadState.EMPTY


# -- the battery at the charger ---------------------------------------------------------------


def test_the_battery_charges_only_standing_on_the_contacts():
    config = BatteryConfig(idle_hours=10.0, range_km=10.0, charge_hours=1.0)
    charger = Pose2D(8.0, -8.0, -math.pi / 2.0)
    sim = BatterySim(config, level=0.5, chargers=[charger])
    sim.step(60.0, charger, 0.0, 0.0)
    assert sim.charging and sim.level > 0.5
    for pose, speed in ((charger.advanced(-0.3), 0.0),                       # short of it
                        (Pose2D(8.0, -8.0, 0.0), 0.0),                       # across it
                        (charger, 0.2)):                                     # still moving
        level = sim.level
        sim.step(60.0, pose, 0.0, speed)
        assert not sim.charging and sim.level < level


def test_a_dead_charger_charges_nothing():
    charger = Pose2D()
    sim = BatterySim(level=0.5, chargers=[charger])
    sim.charger_dead = True
    sim.step(600.0, charger, 0.0, 0.0)
    assert not sim.charging and sim.level < 0.5
