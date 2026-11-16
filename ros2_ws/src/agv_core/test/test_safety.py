"""The safety gate: what it lets through, what it slows, what it stops, and when it lets go."""

import math

from agv_core.geometry import Pose2D
from agv_core.safety import Command, SafetyConfig, SafetyMonitor, SafetyState, Scan
from agv_core.vehicle import AGV
import numpy as np
import pytest

CFG = SafetyConfig()
HL, HW = AGV.half_length, AGV.half_width


def scan(now, *points, too_close=0):
    """Build a scan from points given as (x, y) in the vehicle frame, each seen by two beams."""
    cloud = [p for point in points for p in (point, point)]
    return Scan(now, np.array(cloud, dtype=float).reshape(-1, 2), too_close)


def gate(monitor, now, points=(), v=0.0, w=0.0, measured=None, manual=None, too_close=0):
    """One cycle: a nav command (v, w), odometry `measured` (defaults to the command)."""
    measured = (v, w) if measured is None else measured
    return monitor.update(
        now, scan(now, *points, too_close=too_close), Command(now, *measured),
        nav=Command(now, v, w), manual=None if manual is None else Command(now, *manual))


def ahead(gap, lateral=0.0):
    """Return a point `gap` metres in front of the bumper."""
    return (HL + gap, lateral)


def behind(gap, lateral=0.0):
    return (-HL - gap, lateral)


# -- nothing in the way -----------------------------------------------------------------------


def test_a_command_with_nothing_in_the_way_goes_through_unchanged():
    status = gate(SafetyMonitor(), 0.0, v=0.8, w=0.3)
    assert status.state is SafetyState.OK
    assert (status.v, status.w) == (0.8, 0.3)
    assert status.source == 'nav' and status.gap is None and not status.limited


def test_standing_still_next_to_something_is_fine():
    status = gate(SafetyMonitor(), 0.0, [ahead(0.02), (0.0, HW + 0.02)], v=0.0, w=0.0)
    assert status.state is SafetyState.OK and (status.v, status.w) == (0.0, 0.0)


def test_commands_are_capped_at_the_vehicle_limits():
    status = gate(SafetyMonitor(), 0.0, v=5.0, w=-9.0)
    assert (status.v, status.w) == (CFG.max_speed, -CFG.max_turn_rate)
    assert gate(SafetyMonitor(), 0.0, v=-2.0).v == -CFG.max_reverse_speed
    assert gate(SafetyMonitor(), 0.0, v=float('nan'), w=float('inf')).v == 0.0


# -- slowing down -----------------------------------------------------------------------------


def test_an_obstacle_far_ahead_changes_nothing():
    status = gate(SafetyMonitor(), 0.0, [ahead(3.5)], v=1.0)
    assert status.state is SafetyState.OK and status.v == 1.0
    assert status.gap == pytest.approx(3.5, abs=0.01)


def test_the_vehicle_slows_as_an_obstacle_ahead_gets_closer():
    speeds = []
    for gap in (1.8, 1.5, 1.0, 0.6, 0.3, 0.15):
        status = gate(SafetyMonitor(), 0.0, [ahead(gap)], v=1.0, measured=(0.0, 0.0))
        assert status.state is SafetyState.SLOW, gap
        assert status.v == pytest.approx(CFG.speed_for_gap(gap), abs=1e-6)
        assert f'{gap:.2f} m ahead' in status.reason
        speeds.append(status.v)
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[0] > 0.9 and speeds[-1] < 0.1


def test_the_allowed_speed_is_one_the_vehicle_can_stop_from():
    for gap in (0.2, 0.5, 1.0, 2.0, 3.0):
        speed = CFG.speed_for_gap(gap)
        travelled = (speed * (CFG.reaction_time + CFG.warning_time)
                     + speed * speed / (2.0 * CFG.comfort_decel))
        assert travelled == pytest.approx(gap - CFG.standoff)
    assert CFG.speed_for_gap(CFG.standoff) == 0.0
    assert CFG.speed_for_gap(0.0) == 0.0


def test_slowing_keeps_the_curve_the_command_asked_for():
    status = gate(SafetyMonitor(), 0.0, [ahead(0.8)], v=1.0, w=0.2, measured=(0.0, 0.0))
    assert status.state is SafetyState.SLOW
    assert status.w / status.v == pytest.approx(0.2)


def test_a_limited_speed_below_creep_is_a_stop_not_a_crawl():
    gap = CFG.standoff + 0.01
    assert 0.0 < CFG.speed_for_gap(gap) < CFG.creep_speed
    status = gate(SafetyMonitor(), 0.0, [ahead(gap)], v=0.5, measured=(0.0, 0.0))
    assert status.state is SafetyState.SLOW and status.v == 0.0


def test_a_vehicle_creeping_into_a_station_is_not_stopped_by_its_stand():
    """Docked, the stand is 0.15 m ahead: the last centimetres are driven at 0.05 to 0.1 m/s."""
    monitor = SafetyMonitor()
    for gap, speed in ((0.60, 0.15), (0.40, 0.12), (0.25, 0.08), (0.16, 0.05)):
        status = gate(monitor, 0.0, [ahead(gap)], v=speed)
        assert not status.state.stopped, gap
        assert status.v == pytest.approx(min(speed, CFG.speed_for_gap(gap)))
        assert status.v >= 0.04


# -- what counts as in the way ----------------------------------------------------------------


def test_what_is_beside_the_path_is_not_in_the_way():
    clear = HW + CFG.side_margin + 0.03
    for lateral in (clear, -clear):
        status = gate(SafetyMonitor(), 0.0, [ahead(0.5, lateral)], v=1.0, measured=(0.0, 0.0))
        assert status.state is SafetyState.OK and status.gap is None
    grazing = HW + CFG.side_margin - 0.02
    for lateral in (grazing, -grazing):
        status = gate(SafetyMonitor(), 0.0, [ahead(0.5, lateral)], v=1.0, measured=(0.0, 0.0))
        assert status.state is SafetyState.SLOW


def test_what_is_behind_does_not_matter_when_driving_forward():
    status = gate(SafetyMonitor(), 0.0, [behind(0.1)], v=1.0)
    assert status.state is SafetyState.OK and status.v == 1.0


def test_reversing_looks_behind_and_not_ahead():
    status = gate(SafetyMonitor(), 0.0, [ahead(0.1)], v=-0.3)
    assert status.state is SafetyState.OK and status.v == -0.3
    status = gate(SafetyMonitor(), 0.0, [behind(0.4)], v=-0.3, measured=(0.0, 0.0))
    assert status.state is SafetyState.SLOW
    assert -0.3 < status.v < 0.0
    assert 'behind' in status.reason


def test_one_stray_return_is_not_an_obstacle():
    monitor = SafetyMonitor()
    lone = Scan(0.0, np.array([[HL + 0.3, 0.0]]))
    status = monitor.update(0.0, lone, Command(0.0, 0.5, 0.0), nav=Command(0.0, 0.5, 0.0))
    assert status.state is SafetyState.OK and status.gap is None
    pair = Scan(0.0, np.array([[HL + 0.3, 0.0], [HL + 0.9, 0.05]]))
    status = monitor.update(0.0, pair, Command(0.0, 0.0, 0.0), nav=Command(0.0, 0.5, 0.0))
    # Two returns: the distance is set by the second nearest, not by the stray one.
    assert status.gap == pytest.approx(0.9)


def test_an_arc_looks_along_the_arc():
    # v = 0.5, w = 0.5: a left turn of 1 m radius about (0, 1). A quarter turn on, the
    # vehicle is at (1, 1); straight ahead at (1.5, 0) it never is.
    on_the_arc, straight_ahead = (1.0, 1.0), (1.5, 0.0)
    status = gate(SafetyMonitor(), 0.0, [on_the_arc], v=0.5, w=0.5, measured=(0.0, 0.0))
    assert status.gap == pytest.approx(math.pi / 2.0 - HL, abs=0.02)
    status = gate(SafetyMonitor(), 0.0, [straight_ahead], v=0.5, w=0.5, measured=(0.0, 0.0))
    assert status.gap is None and status.state is SafetyState.OK
    # Driving straight, it is the other way round.
    straight = gate(SafetyMonitor(), 0.0, [straight_ahead], v=0.5)
    assert straight.gap == pytest.approx(1.5 - HL, abs=0.01)
    assert gate(SafetyMonitor(), 0.0, [on_the_arc], v=0.5).gap is None


def test_a_right_turn_mirrors_a_left_turn():
    left = gate(SafetyMonitor(), 0.0, [(1.0, 1.0)], v=0.5, w=0.5, measured=(0.0, 0.0))
    right = gate(SafetyMonitor(), 0.0, [(1.0, -1.0)], v=0.5, w=-0.5, measured=(0.0, 0.0))
    assert right.gap == pytest.approx(left.gap)
    assert gate(SafetyMonitor(), 0.0, [(1.0, 1.0)], v=0.5, w=-0.5, measured=(0.0, 0.0)).gap is None


def test_reversing_on_an_arc_looks_along_the_arc_behind():
    # v = -0.3, w = 0.3: backing up while turning left swings the tail to the right.
    status = gate(SafetyMonitor(), 0.0, [(-1.0, -1.0)], v=-0.3, w=0.3, measured=(0.0, 0.0))
    assert status.gap == pytest.approx(math.pi / 2.0 - HL, abs=0.02)
    other_side = gate(SafetyMonitor(), 0.0, [(-1.0, 1.0)], v=-0.3, w=0.3, measured=(0.0, 0.0))
    assert other_side.gap is None


def test_the_outer_front_corner_swings_wide_in_a_tight_turn():
    # Left turn of 0.5 m radius about (0, 0.5). A point level with the front axle line,
    # 0.30 m outside the centre line: clear of a vehicle driving straight (0.27 m), inside
    # what the outer front corner sweeps.
    radius = 0.5
    swing = math.hypot(radius + HW, HL) - radius - HW
    assert swing > 0.05
    outside = (0.45, radius - math.sqrt((radius + 0.30) ** 2 - 0.45 ** 2))
    inside = (0.20, radius - math.sqrt((radius - 0.30) ** 2 - 0.20 ** 2))
    turning = gate(SafetyMonitor(), 0.0, [outside], v=0.4, w=0.8, measured=(0.0, 0.0))
    assert turning.gap is not None
    # The same offset on the inside of the turn is not swept.
    assert gate(SafetyMonitor(), 0.0, [inside], v=0.4, w=0.8, measured=(0.0, 0.0)).gap is None


# -- the protective stop ----------------------------------------------------------------------


def test_something_inside_the_stopping_distance_stops_the_vehicle_at_once():
    monitor = SafetyMonitor()
    gap = CFG.standoff + CFG.stopping_distance(1.0) - 0.05
    status = gate(monitor, 0.0, [ahead(gap)], v=1.0, measured=(1.0, 0.0))
    assert status.state is SafetyState.PROTECTIVE_STOP
    assert (status.v, status.w) == (0.0, 0.0)
    assert 'm ahead' in status.reason
    assert status.state.stopped


def test_the_protective_field_grows_with_speed():
    lengths = [CFG.standoff + CFG.stopping_distance(v) for v in (0.0, 0.1, 0.5, 1.0)]
    assert lengths == sorted(lengths)
    assert lengths[0] == pytest.approx(CFG.standoff)
    assert lengths[-1] == pytest.approx(0.08 + 0.25 + 1.0 / 3.0)
    # The same obstacle that stops a fast vehicle only slows a slow one.
    gap = 0.4
    fast = gate(SafetyMonitor(), 0.0, [ahead(gap)], v=1.0, measured=(1.0, 0.0))
    slow = gate(SafetyMonitor(), 0.0, [ahead(gap)], v=1.0, measured=(0.2, 0.0))
    assert fast.state is SafetyState.PROTECTIVE_STOP
    assert slow.state is SafetyState.SLOW and slow.v > 0.0


def test_the_field_is_sized_for_what_the_vehicle_was_told_when_odometry_lags():
    monitor = SafetyMonitor()
    assert gate(monitor, 0.0, v=1.0, measured=(0.0, 0.0)).v == 1.0
    # Odometry still says standing; the gate remembers it let 1.0 m/s through.
    status = gate(monitor, 0.05, [ahead(0.5)], v=1.0, measured=(0.0, 0.0))
    assert status.state is SafetyState.PROTECTIVE_STOP


def test_a_protective_stop_holds_until_the_field_has_been_clear_long_enough():
    monitor = SafetyMonitor()
    assert gate(monitor, 0.0, [ahead(0.2)], v=1.0, measured=(1.0, 0.0)).state.stopped
    # The obstacle leaves. The command is still there; the vehicle is not released yet.
    for now in (0.1, 1.0, CFG.restart_delay - 0.01):
        status = gate(monitor, now, v=1.0, measured=(0.0, 0.0))
        assert status.state is SafetyState.PROTECTIVE_STOP and status.v == 0.0
        assert 'clear for' in status.reason
    status = gate(monitor, CFG.restart_delay + 0.01, v=1.0, measured=(0.0, 0.0))
    assert status.state is SafetyState.OK and status.v == 1.0


def test_coming_back_into_the_field_restarts_the_wait():
    monitor = SafetyMonitor()
    gate(monitor, 0.0, [ahead(0.2)], v=1.0, measured=(1.0, 0.0))
    gate(monitor, 1.5, v=1.0, measured=(0.0, 0.0))
    # Back again after 1.5 s, standing closer than the standoff with the vehicle told to go.
    assert gate(monitor, 1.6, [ahead(0.05)], v=1.0, measured=(0.0, 0.0)).state.stopped
    assert gate(monitor, 1.6 + CFG.restart_delay - 0.1, v=1.0, measured=(0.0, 0.0)).state.stopped
    assert gate(monitor, 1.6 + CFG.restart_delay + 0.1, v=1.0, measured=(0.0, 0.0)).v == 1.0


def test_after_a_stop_the_wait_starts_when_the_person_has_left():
    """Stopped 15 cm short of someone: held while they stand there, released 2 s after they go."""
    monitor = SafetyMonitor()
    assert gate(monitor, 0.0, [ahead(0.3)], v=0.8, measured=(0.8, 0.0)).state.stopped
    # Standing still now, 15 cm from them, still asked for 0.8 m/s: the stop holds.
    for now in (0.5, 2.5, 5.0, 9.9):
        status = gate(monitor, now, [ahead(0.15)], v=0.8, measured=(0.0, 0.0))
        assert status.state is SafetyState.PROTECTIVE_STOP and status.v == 0.0
    # Last seen at 9.9 s; they step aside before the next scan.
    assert gate(monitor, 10.0, v=0.8, measured=(0.0, 0.0)).state.stopped
    assert gate(monitor, 11.85, v=0.8, measured=(0.0, 0.0)).state.stopped
    assert gate(monitor, 11.95, v=0.8, measured=(0.0, 0.0)).v == 0.8


def test_after_a_stop_a_slow_request_needs_only_its_own_room():
    """The docking crawl: asked for 5 cm/s, a stand 15 cm ahead does not hold the vehicle."""
    monitor = SafetyMonitor()
    gate(monitor, 0.0, [ahead(0.05)], v=0.3, measured=(0.0, 0.0))
    for now in (0.5, 1.0, 1.5):
        assert gate(monitor, now, [ahead(0.15)], v=0.05, measured=(0.0, 0.0)).state.stopped
    status = gate(monitor, 2.1, [ahead(0.15)], v=0.05, measured=(0.0, 0.0))
    assert not status.state.stopped and status.v > 0.0


def test_a_standing_vehicle_will_not_start_into_something_touching_it():
    monitor = SafetyMonitor()
    status = gate(monitor, 0.0, [ahead(CFG.standoff - 0.03)], v=0.3, measured=(0.0, 0.0))
    assert status.state is SafetyState.PROTECTIVE_STOP and status.v == 0.0


def test_a_stopped_vehicle_may_back_away_after_the_wait():
    monitor = SafetyMonitor()
    wall = [ahead(0.04)]
    assert gate(monitor, 0.0, wall, v=0.3, measured=(0.0, 0.0)).state.stopped
    # Reverse is asked for; nothing is behind. The stop still lasts its restart delay.
    assert gate(monitor, 1.0, wall, v=-0.2, measured=(0.0, 0.0)).state.stopped
    status = gate(monitor, CFG.restart_delay + 0.1, wall, v=-0.2, measured=(0.0, 0.0))
    assert status.state is SafetyState.OK and status.v == -0.2


def test_returns_closer_than_the_scanner_can_measure_stop_the_vehicle():
    status = gate(SafetyMonitor(), 0.0, v=0.5, too_close=2)
    assert status.state is SafetyState.PROTECTIVE_STOP
    assert 'closer than the scanner can measure' in status.reason
    assert gate(SafetyMonitor(), 0.0, v=0.5, too_close=1).state is SafetyState.OK


# -- turning on the spot ----------------------------------------------------------------------


def test_turning_on_the_spot_is_free_when_nothing_is_inside_the_turning_circle():
    outside = AGV.turning_radius + CFG.turn_margin + 0.05
    status = gate(SafetyMonitor(), 0.0, [(0.0, outside), (-outside, 0.0)], w=1.0)
    assert status.state is SafetyState.OK and status.w == 1.0 and status.gap is None


def test_turning_on_the_spot_knows_how_far_it_can_turn():
    # 0.30 m to the left of the centre: 8 cm off the side, well inside the corners' circle.
    # The grown outline reaches it when |sin| of its bearing drops to 0.25 / 0.30.
    point = (0.0, 0.30)
    expected = math.pi / 2.0 - math.asin((HW + CFG.turn_margin) / 0.30)
    for rate in (1.0, -1.0):
        status = gate(SafetyMonitor(), 0.0, [point], w=rate, measured=(0.0, 0.0))
        assert status.gap == pytest.approx(expected, abs=math.radians(1.5))
        assert status.state is SafetyState.SLOW
        assert abs(status.w) == pytest.approx(CFG.turn_rate_for_gap(status.gap))
        assert 0.0 < abs(status.w) < 1.0
        assert 'degrees of room' in status.reason


def test_a_corner_about_to_hit_stops_the_turn():
    # A point 1 cm inside the circle the corners sweep, 5 degrees round from where the
    # front-left corner (grown by its margin) is now: a left turn brings the corner to it.
    radius = AGV.turning_radius - 0.01
    reached_at = math.asin((HW + CFG.turn_margin) / radius)
    bearing = reached_at + math.radians(5.0)
    point = (radius * math.cos(bearing), radius * math.sin(bearing))
    status = gate(SafetyMonitor(), 0.0, [point], w=1.0, measured=(0.0, 1.0))
    assert status.state is SafetyState.PROTECTIVE_STOP
    assert status.gap == pytest.approx(math.radians(5.0), abs=math.radians(1.5))
    assert 'turning circle' in status.reason
    # A right turn takes that corner away from it; the rear-left corner arrives after most
    # of a quarter turn.
    other = gate(SafetyMonitor(), 0.0, [point], w=-0.3, measured=(0.0, 0.0))
    assert not other.state.stopped
    assert math.radians(80.0) < other.gap < math.radians(95.0)


def test_something_touching_the_side_stops_a_turn_either_way():
    # 2 cm off the left side, level with the centre: inside the margin whichever way it turns.
    point = (0.0, HW + 0.02)
    for rate in (0.5, -0.5):
        status = gate(SafetyMonitor(), 0.0, [point], w=rate, measured=(0.0, 0.0))
        assert status.state is SafetyState.PROTECTIVE_STOP and status.w == 0.0


def test_a_slow_drive_with_a_fast_turn_is_treated_as_turning_on_the_spot():
    # v / w = 0.05 m: the turning centre is inside the vehicle.
    point = (0.0, 0.30)
    status = gate(SafetyMonitor(), 0.0, [point], v=0.05, w=1.0, measured=(0.0, 0.0))
    assert status.state is SafetyState.SLOW and 'degrees of room' in status.reason


# -- the station field set ------------------------------------------------------------------


def test_docking_may_come_close_to_a_stand_that_driving_may_not():
    # 6 cm from the stand, asked for 5 cm/s.
    driving = gate(SafetyMonitor(), 0.0, [ahead(0.06)], v=0.05, measured=(0.0, 0.0))
    docking = SafetyMonitor().update(
        0.0, scan(0.0, ahead(0.06)), Command(0.0, 0.0, 0.0), nav=Command(0.0, 0.05, 0.0),
        docking=True)
    assert driving.state is SafetyState.PROTECTIVE_STOP and driving.v == 0.0
    assert not docking.state.stopped and docking.v == pytest.approx(0.05)


def test_docking_still_stops_short_of_the_stand():
    monitor = SafetyMonitor()
    for gap, expected_stop in ((0.10, False), (0.05, False), (0.025, True), (0.01, True)):
        status = monitor.update(0.0, scan(0.0, ahead(gap)), Command(0.0, 0.0, 0.0),
                                nav=Command(0.0, 0.04, 0.0), docking=True)
        assert (status.v == 0.0) is expected_stop, (gap, status)


def test_docking_is_capped_at_crawling_speed_and_keeps_every_other_rule():
    cfg = SafetyConfig()
    fast = SafetyMonitor().update(0.0, scan(0.0), Command(0.0, 0.0, 0.0),
                                  nav=Command(0.0, 0.8, 0.0), docking=True)
    assert fast.v == pytest.approx(cfg.dock_max_speed)
    back = SafetyMonitor().update(0.0, scan(0.0), Command(0.0, 0.0, 0.0),
                                  nav=Command(0.0, -0.8, 0.0), docking=True)
    assert back.v == pytest.approx(-cfg.dock_max_speed)
    estopped = SafetyMonitor()
    estopped.press_estop()
    assert estopped.update(0.0, scan(0.0), Command(0.0, 0.0, 0.0),
                           nav=Command(0.0, 0.05, 0.0), docking=True).state is SafetyState.ESTOP
    assert SafetyMonitor().update(0.0, None, Command(0.0, 0.0, 0.0),
                                  nav=Command(0.0, 0.05, 0.0),
                                  docking=True).state is SafetyState.FAULT
    # And afterwards the normal fields apply again.
    monitor = SafetyMonitor()
    monitor.update(0.0, scan(0.0, ahead(0.06)), Command(0.0, 0.0, 0.0),
                   nav=Command(0.0, 0.05, 0.0), docking=True)
    assert monitor.config.standoff == cfg.standoff


# -- faults -----------------------------------------------------------------------------------


def test_no_scan_or_an_old_scan_is_a_fault():
    monitor = SafetyMonitor()
    odom, nav = Command(1.0, 0.0, 0.0), Command(1.0, 0.5, 0.0)
    status = monitor.update(1.0, None, odom, nav=nav)
    assert status.state is SafetyState.FAULT and status.v == 0.0
    assert status.reason == 'no scan received'
    stale = Scan(1.0 - CFG.scan_timeout - 0.05, np.empty((0, 2)))
    status = monitor.update(1.0, stale, odom, nav=nav)
    assert status.state is SafetyState.FAULT and 'scan is 0.40 s old' in status.reason
    fresh = Scan(1.0 - CFG.scan_timeout + 0.05, np.empty((0, 2)))
    assert monitor.update(1.0, fresh, odom, nav=nav).state is SafetyState.OK


def test_no_odometry_or_old_odometry_is_a_fault():
    monitor = SafetyMonitor()
    empty, nav = Scan(1.0, np.empty((0, 2))), Command(1.0, 0.5, 0.0)
    assert monitor.update(1.0, empty, None, nav=nav).reason == 'no odometry received'
    status = monitor.update(1.0, empty, Command(0.4, 0.5, 0.0), nav=nav)
    assert status.state is SafetyState.FAULT and 'odometry is 0.60 s old' in status.reason


def test_a_fault_clears_by_itself_when_the_data_comes_back():
    monitor = SafetyMonitor()
    nav = Command(1.0, 0.5, 0.0)
    assert monitor.update(1.0, None, Command(1.0), nav=nav).state is SafetyState.FAULT
    status = monitor.update(1.0, Scan(1.0, np.empty((0, 2))), Command(1.0), nav=nav)
    assert status.state is SafetyState.OK and status.v == 0.5


def test_an_old_command_is_no_command():
    monitor = SafetyMonitor()
    empty, odom = Scan(1.0, np.empty((0, 2))), Command(1.0)
    old = Command(1.0 - CFG.command_timeout - 0.05, 0.8, 0.0)
    status = monitor.update(1.0, empty, odom, nav=old)
    assert status.state is SafetyState.OK and status.source == 'none'
    assert (status.v, status.w) == (0.0, 0.0)
    assert monitor.update(1.0, empty, odom, nav=None).source == 'none'
    # A command stamped in the future is not trusted either.
    assert monitor.update(1.0, empty, odom, nav=Command(2.0, 0.8, 0.0)).source == 'none'


# -- e-stop -----------------------------------------------------------------------------------


def test_the_estop_stops_everything_until_released_and_reset():
    monitor = SafetyMonitor()
    monitor.press_estop()
    status = gate(monitor, 0.0, v=0.5)
    assert status.state is SafetyState.ESTOP and status.v == 0.0
    assert status.reason == 'e-stop pressed' and monitor.estop_active
    # Resetting with the button still in is refused.
    ok, message = monitor.reset()
    assert not ok and 'still pressed' in message
    assert gate(monitor, 0.1, v=0.5).state is SafetyState.ESTOP
    # Releasing alone does not release the vehicle.
    monitor.release_estop()
    assert not monitor.estop_pressed and monitor.estop_active
    assert gate(monitor, 0.2, v=0.5).state is SafetyState.ESTOP
    ok, message = monitor.reset()
    assert ok and message == 'e-stop reset'
    status = gate(monitor, 0.3, v=0.5)
    assert status.state is SafetyState.OK and status.v == 0.5


def test_resetting_with_nothing_to_reset_is_harmless():
    assert SafetyMonitor().reset() == (True, 'nothing to reset')


def test_the_estop_outranks_a_fault_and_a_protective_stop():
    monitor = SafetyMonitor()
    monitor.press_estop('light curtain')
    status = monitor.update(0.0, None, None, nav=Command(0.0, 0.5, 0.0))
    assert status.state is SafetyState.ESTOP and status.reason == 'light curtain'


# -- the operator -----------------------------------------------------------------------------


def test_a_manual_command_outranks_navigation():
    monitor = SafetyMonitor()
    status = gate(monitor, 0.0, v=0.8, w=0.1, manual=(0.2, -0.4))
    assert status.source == 'manual' and (status.v, status.w) == (0.2, -0.4)


def test_manual_driving_is_capped_lower_and_still_gated():
    monitor = SafetyMonitor()
    status = gate(monitor, 0.0, manual=(2.0, 3.0))
    assert (status.v, status.w) == (CFG.manual_max_speed, CFG.manual_max_turn_rate)
    blocked = gate(SafetyMonitor(), 0.0, [ahead(0.05)], manual=(0.3, 0.0), measured=(0.0, 0.0))
    assert blocked.state is SafetyState.PROTECTIVE_STOP and blocked.source == 'manual'


def test_navigation_gets_control_back_when_the_operator_lets_go():
    monitor = SafetyMonitor()
    empty = Scan(1.0, np.empty((0, 2)))
    manual = Command(1.0 - CFG.manual_timeout + 0.05, 0.2, 0.0)
    nav = Command(1.0, 0.8, 0.0)
    assert monitor.update(1.0, empty, Command(1.0), nav=nav, manual=manual).source == 'manual'
    later = Scan(1.2, np.empty((0, 2)))
    status = monitor.update(1.2, later, Command(1.2), nav=Command(1.2, 0.8, 0.0), manual=manual)
    assert status.source == 'nav' and status.v == 0.8


# -- reading a scan ---------------------------------------------------------------------------


def test_reading_a_laser_scan():
    ranges = [1.0, float('nan'), float('inf'), float('-inf'), 0.05, 2.0, 30.0]
    result = Scan.from_ranges(3.0, ranges, angle_min=0.0, angle_increment=math.pi / 2.0,
                              range_min=0.08, range_max=16.0)
    assert result.stamp == 3.0
    # Valid: 1.0 at 0 rad and 2.0 at 5 * 90 degrees = 90 degrees. NaN is no data, +inf is
    # nothing there, 30 m is beyond the scanner; -inf and 0.05 m are too close.
    assert result.points == pytest.approx(np.array([[1.0, 0.0], [0.0, 2.0]]), abs=1e-9)
    assert result.too_close == 2


def test_a_scanner_that_is_not_at_the_centre():
    result = Scan.from_ranges(
        0.0, [1.0], 0.0, 0.1, 0.05, 10.0, sensor=Pose2D(0.2, 0.1, math.pi / 2.0))
    assert result.points == pytest.approx(np.array([[0.2, 1.1]]))


def test_an_empty_scan_has_no_points():
    result = Scan.from_ranges(0.0, [], 0.0, 0.1, 0.05, 10.0)
    assert result.points.shape == (0, 2) and result.too_close == 0
    assert gate(SafetyMonitor(), 0.0, v=0.5).state is SafetyState.OK


# -- the fields as shapes ---------------------------------------------------------------------


def test_the_fields_are_rectangles_ahead_when_driving_straight():
    status = gate(SafetyMonitor(), 0.0, v=1.0)
    xs = [p[0] for p in status.protective_field]
    ys = [p[1] for p in status.protective_field]
    assert min(xs) == pytest.approx(-HL)
    assert max(xs) == pytest.approx(HL + CFG.standoff + CFG.stopping_distance(1.0))
    assert max(ys) == pytest.approx(HW + CFG.side_margin) and min(ys) == pytest.approx(-max(ys))
    assert max(p[0] for p in status.warning_field) > max(xs)


def test_the_fields_are_behind_when_reversing_and_round_when_turning():
    status = gate(SafetyMonitor(), 0.0, v=-0.3)
    xs = [p[0] for p in status.protective_field]
    assert max(xs) == pytest.approx(HL)
    assert min(xs) == pytest.approx(-HL - CFG.standoff - CFG.stopping_distance(0.3))
    turning = gate(SafetyMonitor(), 0.0, w=1.0)
    radii = [math.hypot(*p) for p in turning.protective_field]
    assert radii == pytest.approx([AGV.turning_radius + CFG.turn_margin] * len(radii))


def test_the_fields_bend_with_the_arc():
    left = gate(SafetyMonitor(), 0.0, v=0.6, w=0.6)
    right = gate(SafetyMonitor(), 0.0, v=0.6, w=-0.6)
    assert max(p[1] for p in left.warning_field) > HW + 0.3
    assert min(p[1] for p in right.warning_field) < -HW - 0.3
    assert left.protective_field != right.protective_field


def test_a_standing_vehicle_with_no_command_has_no_fields():
    status = gate(SafetyMonitor(), 0.0)
    assert status.protective_field == () and status.warning_field == ()


# -- does it actually stop in time ------------------------------------------------------------


def drive_at_a_wall(distance, appears_at=None, command=1.0, dt=0.02, lag=0.10):
    """Drive at a wall with a drive that lags `lag` s and brakes at the vehicle's limit.

    Returns the smallest gap between bumper and wall, and the states seen. With
    `appears_at`, the wall is invisible until the gap has shrunk to that.
    """
    monitor = SafetyMonitor()
    travelled, speed, now = 0.0, 0.0, 0.0
    pending = []
    states = set()
    smallest = distance
    while now < 30.0:
        gap = distance - travelled
        smallest = min(smallest, gap)
        visible = appears_at is None or gap <= appears_at
        # The scan is one cycle old, as it is at 10 Hz against a 50 Hz gate.
        points = [ahead(gap)] if visible else []
        status = gate(monitor, now, points, v=command, measured=(speed, 0.0))
        states.add(status.state)
        pending.append((now + lag, status.v))
        target = speed
        while pending and pending[0][0] <= now:
            target = pending.pop(0)[1]
        if target > speed:
            speed = min(target, speed + AGV.max_accel * dt)
        else:
            speed = max(target, speed - AGV.brake_decel * dt)
        travelled += speed * dt
        now += dt
        if speed == 0.0 and now > 5.0:
            break
    return smallest, states


def test_driving_at_a_wall_ends_in_a_gentle_stop_short_of_it():
    smallest, states = drive_at_a_wall(6.0)
    assert SafetyState.PROTECTIVE_STOP not in states       # slowed down, never had to brake hard
    assert SafetyState.SLOW in states
    assert CFG.standoff - 0.01 <= smallest <= CFG.standoff + 0.12


@pytest.mark.parametrize('appears_at', [0.9, 0.7, 0.5])
def test_something_stepping_out_in_front_at_full_speed_is_not_hit(appears_at):
    smallest, states = drive_at_a_wall(6.0, appears_at=appears_at)
    assert SafetyState.PROTECTIVE_STOP in states or SafetyState.SLOW in states
    assert smallest > 0.0, f'hit: {smallest:.3f} m'


def test_what_the_protective_field_promises_holds_with_the_brakes_it_assumes():
    """Appearing right at the edge of the field, it is not hit even braking at 1.5 m/s^2."""
    edge = CFG.standoff + CFG.stopping_distance(1.0)
    smallest, states = drive_at_a_wall(6.0, appears_at=edge)
    assert SafetyState.PROTECTIVE_STOP in states
    assert smallest > 0.05
