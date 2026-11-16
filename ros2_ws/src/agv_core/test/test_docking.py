"""The docking run, driven on the kinematic vehicle."""

import math

from agv_core.docking import DockingConfig, DockingController, DockingPhase, is_docked
from agv_core.geometry import Pose2D
from agv_core.simulation import KinematicVehicle
import pytest

CFG = DockingConfig()
STATION = Pose2D(4.0, -2.0, math.radians(135.0))     # nowhere special, pointing nowhere special
APPROACH = 0.8


def start_pose(back=APPROACH, left=0.0, turned=0.0, station=STATION):
    """Return a pose `back` metres short of the station, `left` of its centre line."""
    pose = station.advanced(-back, left)
    return Pose2D(pose.x, pose.y, pose.yaw + turned)


def dock(pose, station=STATION, config=CFG, lag=0.0, dt=0.02, limit=60.0):
    """Drive a docking run. Returns the last command, the vehicle, and every command given."""
    vehicle = KinematicVehicle(pose)
    controller = DockingController(station, config)
    commands, pending, now = [], [], 0.0
    while now < limit:
        command = controller.update(vehicle.pose)
        commands.append(command)
        if command.done or command.failed:
            break
        pending.append((now + lag, command.v, command.w))
        while pending and pending[0][0] <= now:
            _, v, w = pending.pop(0)
            vehicle.command(v, w, now)
        vehicle.step(dt, now)
        now += dt
    return commands[-1], vehicle, commands


def errors(vehicle, station=STATION):
    local = vehicle.pose.relative_to(station)
    return abs(local.y), abs(math.degrees(local.yaw)), local.x


def test_from_the_approach_pose_it_docks_on_the_spot():
    last, vehicle, commands = dock(start_pose())
    assert last.done and last.reason == ''
    offset, heading, along = errors(vehicle)
    assert offset < 0.002 and heading < 0.2
    assert -0.005 <= along <= 0.02              # on the stop line, a hair past at most
    assert is_docked(vehicle.pose, STATION)
    assert len(commands) * 0.02 < 15.0          # 0.8 m takes a few seconds, not a minute


@pytest.mark.parametrize('left', [-0.10, -0.05, 0.05, 0.10])
@pytest.mark.parametrize('turned', [-6.0, 0.0, 6.0])
@pytest.mark.parametrize('back', [0.7, 0.8, 0.9])
def test_it_docks_from_anywhere_navigation_delivers_the_vehicle(left, turned, back):
    """Nav2 stops within 10 cm and 6 degrees of the approach pose."""
    last, vehicle, _ = dock(start_pose(back, left, math.radians(turned)))
    assert last.done, last.reason
    offset, heading, along = errors(vehicle)
    assert offset <= CFG.position_tolerance
    assert heading <= math.degrees(CFG.heading_tolerance)
    assert -0.005 <= along <= 0.02


def test_the_sideways_error_is_driven_out_before_the_stop_line():
    _, _, commands = dock(start_pose(left=0.10))
    final_stretch = [abs(c.offset) for c in commands if 0.0 < c.remaining < 0.15]
    assert final_stretch and max(final_stretch) < 0.015


@pytest.mark.parametrize('turned', [90.0, -90.0, 180.0, 135.0])
def test_facing_the_wrong_way_it_turns_first(turned):
    last, vehicle, commands = dock(start_pose(turned=math.radians(turned)))
    assert last.done, last.reason
    assert commands[0].phase is DockingPhase.ALIGN
    assert commands[0].v == 0.0 and commands[0].w != 0.0
    assert errors(vehicle)[0] <= CFG.position_tolerance


def phases_of(commands):
    phases = []
    for command in commands:
        if not phases or phases[-1] is not command.phase:
            phases.append(command.phase)
    return phases


def test_the_phases_come_in_order():
    _, _, commands = dock(start_pose(left=0.08, turned=math.radians(20.0)))
    phases = phases_of(commands)
    assert phases[:2] == [DockingPhase.ALIGN, DockingPhase.APPROACH]
    assert phases[-1] is DockingPhase.DONE
    # Settling only shows when the heading on the stop line still needs trimming.
    assert phases[2:-1] in ([], [DockingPhase.SETTLE])


def test_arriving_crooked_it_turns_to_the_station_heading_on_the_stop_line():
    """Stopped on the line at an angle, as after a late sideways correction."""
    vehicle_pose = start_pose(back=0.0, turned=math.radians(10.0))
    controller = DockingController(STATION, CFG)
    controller.update(start_pose())                       # the run has started properly
    command = controller.update(vehicle_pose)
    assert command.phase is DockingPhase.SETTLE
    assert command.v == 0.0 and command.w < 0.0           # turning back, not driving
    last, vehicle, commands = dock(start_pose(left=0.10, turned=math.radians(-6.0)))
    assert last.done and errors(vehicle)[1] <= math.degrees(CFG.heading_tolerance)


def test_it_never_reverses_and_never_hurries():
    for left, turned in ((0.10, 6.0), (-0.10, -6.0), (0.0, 170.0)):
        _, _, commands = dock(start_pose(left=left, turned=math.radians(turned)))
        assert min(c.v for c in commands) >= 0.0
        assert max(c.v for c in commands) <= CFG.max_speed + 1e-9
        assert max(abs(c.w) for c in commands) <= CFG.max_turn_rate + 1e-9


def test_it_slows_to_a_crawl_at_the_stop_line():
    _, _, commands = dock(start_pose(back=1.5))
    moving = [c for c in commands if c.phase is DockingPhase.APPROACH]
    assert moving[0].v == pytest.approx(CFG.max_speed)
    assert moving[-1].v == pytest.approx(CFG.min_speed)
    speeds = [c.v for c in moving]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(speeds, speeds[1:]))


@pytest.mark.parametrize('lag', [0.05, 0.10, 0.20])
def test_a_drive_that_lags_still_docks(lag):
    last, vehicle, _ = dock(start_pose(left=0.08, turned=math.radians(-5.0)), lag=lag)
    assert last.done, last.reason
    offset, heading, along = errors(vehicle)
    assert offset <= CFG.position_tolerance and heading <= math.degrees(CFG.heading_tolerance)
    assert along <= 0.03


@pytest.mark.parametrize('station', [
    Pose2D(0.0, 0.0, 0.0), Pose2D(-7.5, 3.0, math.pi), Pose2D(2.0, 9.0, -math.pi / 2.0),
    Pose2D(100.0, -50.0, 0.3),
])
def test_it_works_wherever_the_station_is_and_whichever_way_it_faces(station):
    last, vehicle, _ = dock(start_pose(left=-0.07, turned=math.radians(4.0), station=station),
                            station=station)
    assert last.done and is_docked(vehicle.pose, station)


@pytest.mark.parametrize('back, left, expected', [
    (3.0, 0.0, '3.00 m from the station: too far'),
    (0.1, 0.0, '0.10 m from the stop line: too close'),
    (0.8, 0.4, '0.40 m off the centre line: too far'),
    (0.8, -0.3, '0.30 m off the centre line: too far'),
    (-0.2, 0.0, 'too close'),
])
def test_a_run_that_would_start_from_the_wrong_place_is_refused(back, left, expected):
    last, vehicle, commands = dock(start_pose(back, left))
    assert last.failed and expected in last.reason
    assert len(commands) == 1 and (last.v, last.w) == (0.0, 0.0)
    assert vehicle.distance == 0.0


def test_a_large_error_on_a_short_run_is_still_driven_out():
    # 0.2 m to the side with 0.3 m to go: it turns towards the centre line first.
    last, vehicle, _ = dock(start_pose(back=0.3, left=0.2))
    assert last.done, last.reason
    assert errors(vehicle)[0] <= CFG.position_tolerance


def test_pushed_off_the_centre_line_at_the_end_the_dock_fails():
    """Five centimetres sideways with two to go: no room left to correct, and it says so."""
    vehicle = KinematicVehicle(start_pose())
    controller = DockingController(STATION, CFG)
    now, pushed, command = 0.0, False, None
    while now < 60.0:
        command = controller.update(vehicle.pose)
        if command.done or command.failed:
            break
        if not pushed and command.remaining < 0.02:
            vehicle.place(vehicle.pose.advanced(0.0, 0.05))
            pushed = True
        vehicle.command(command.v, command.w, now)
        vehicle.step(0.02, now)
        now += 0.02
    assert pushed and command.failed
    assert 'm off the centre line' in command.reason
    assert errors(vehicle)[0] > CFG.position_tolerance


def test_ending_up_past_the_stop_line_is_a_failure():
    controller = DockingController(STATION, CFG)
    assert not controller.update(start_pose()).failed
    past = controller.update(start_pose(back=-0.08))
    assert past.failed and '0.08 m past the stop line' in past.reason


def test_a_finished_controller_stays_finished():
    last, vehicle, _ = dock(start_pose())
    controller = DockingController(STATION, CFG)
    for pose in (start_pose(), vehicle.pose):
        command = controller.update(pose)
    for _ in range(3):
        command = controller.update(vehicle.pose)
        assert (command.v, command.w) == (0.0, 0.0)


def test_is_docked():
    assert is_docked(STATION, STATION)
    assert is_docked(STATION.advanced(0.03, 0.03), STATION)
    assert not is_docked(STATION.advanced(0.08), STATION)
    turned = Pose2D(STATION.x, STATION.y, STATION.yaw + math.radians(6.0))
    assert not is_docked(turned, STATION)
    assert is_docked(turned, STATION, heading=math.radians(10.0))
