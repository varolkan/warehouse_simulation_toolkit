"""Transport orders from start to finish, and everything that can go wrong on the way."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agv_core.geometry import Pose2D  # noqa: E402
from agv_core.mission import MissionConfig, NavState  # noqa: E402
from agv_core.safety import SafetyState  # noqa: E402
from agv_core.simulation import LoadState  # noqa: E402
import pytest  # noqa: E402
from support import Rig, site  # noqa: E402


def started(pickup='a', dropoff='b', **options):
    """Return a rig with an accepted order."""
    rig = Rig(**{k: options.pop(k) for k in list(options) if k in (
        'config', 'at', 'loaded', 'battery', 'the_site', 'battery_range')})
    assert rig.order(pickup, dropoff, **options) is None
    return rig


# -- a delivery -------------------------------------------------------------------------------


def test_a_delivery_from_one_station_to_another():
    rig = started('a', 'b')
    result = rig.run_to_end()

    assert result.outcome == 'delivered' and result.success and result.delivered
    assert result.message == "delivered from 'a' to 'b'"
    assert rig.states == ['checking', 'to_pickup', 'docking', 'loading', 'to_dropoff',
                          'docking', 'unloading', 'complete']
    # Navigation was sent to the two approach poses, 0.8 m short of each station.
    assert [goal.station for goal in rig.goals] == ['a', 'b']
    assert (rig.goals[0].pose.x, rig.goals[0].pose.y) == pytest.approx((9.2, 0.0))
    assert (rig.goals[1].pose.x, rig.goals[1].pose.y) == pytest.approx((10.0, 7.2))
    # One command per transfer, each at its station.
    assert [(c.action, c.station) for c in rig.load_commands] == [('load', 'a'), ('unload', 'b')]
    assert rig.handler.state is LoadState.EMPTY
    # It ends docked at the dropoff, and says how far and how long.
    assert rig.docked_at('b')
    offset, heading, _ = rig.docking_error('b')
    assert offset < 0.02 and heading < 1.5
    assert result.distance == pytest.approx(rig.vehicle.distance, abs=0.2)
    assert 17.0 < result.distance < 19.5          # 9.2 m, 0.8 m, 7.3 m, 0.8 m
    assert 30.0 < result.duration < 80.0
    assert rig.cancels == 0


def test_the_status_follows_the_order():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    status = rig.status
    assert status.active and (status.pickup, status.dropoff) == ('a', 'b')
    assert status.station == 'a' and status.detail == "driving to 'a'"
    assert not status.carrying and status.battery == pytest.approx(0.8)
    rig.run(2.0)
    assert 0.0 < rig.status.distance_remaining < 9.2 and rig.status.distance_driven > 1.0
    rig.run_until_state('loading')
    assert rig.status.detail == "loading at 'a' (command 1 of 3)"
    rig.run_until_state('to_dropoff')
    assert rig.status.carrying and rig.status.station == 'b'
    rig.run_until_state('unloading')
    assert rig.status.carrying            # still on board until the handler says otherwise
    rig.run_to_end()
    status = rig.status
    assert status.state == 'complete' and not status.active and not status.carrying
    assert status.detail == "delivered from 'a' to 'b'"
    assert status.distance_driven == pytest.approx(rig.results[-1].distance)
    assert status.elapsed == pytest.approx(rig.results[-1].duration)


def test_a_delivery_that_returns_to_the_charger():
    rig = started('a', 'b', return_to_charger=True)
    result = rig.run_to_end()
    assert result.success and result.message == "delivered from 'a' to 'b', back at the charger"
    assert rig.states[-4:] == ['unloading', 'to_charger', 'docking', 'complete']
    assert [goal.station for goal in rig.goals] == ['a', 'b', 'charger']
    assert rig.docked_at('charger') and rig.inputs().charging


def test_a_pickup_where_the_vehicle_already_stands_needs_no_drive():
    rig = started('a', 'b', at='a')
    result = rig.run_to_end()
    assert result.success
    assert [goal.station for goal in rig.goals] == ['b']
    assert rig.states[:3] == ['checking', 'loading', 'to_dropoff']


def test_already_at_the_approach_pose_it_goes_straight_to_docking():
    rig = started('a', 'b', at=site().station('a').approach_pose)
    result = rig.run_to_end()
    assert result.success and [goal.station for goal in rig.goals] == ['b']
    assert rig.states[:3] == ['checking', 'docking', 'loading']


def test_an_order_without_a_pickup_delivers_the_load_already_on_board():
    rig = started('', 'b', loaded=True)
    result = rig.run_to_end()
    assert result.success and result.message == "delivered to 'b'"
    assert [(c.action, c.station) for c in rig.load_commands] == [('unload', 'b')]
    assert rig.states == ['checking', 'to_dropoff', 'docking', 'unloading', 'complete']


def test_one_order_after_another():
    rig = started('a', 'b')
    assert rig.run_to_end().success
    assert rig.order('b', 'c') is None
    rig.results.clear()
    second = rig.run_to_end()
    assert second.success and second.message == "delivered from 'b' to 'c'"
    assert rig.docked_at('c')
    # The second order's figures are its own.
    assert second.distance < 20.0 and second.duration < 80.0


# -- what is refused before anything moves ----------------------------------------------------


@pytest.mark.parametrize('pickup, dropoff, options, expected', [
    ('a', '', {}, 'the order names no dropoff'),
    ('x', 'b', {}, "unknown pickup 'x' (transfer stations here: a, b, c)"),
    ('a', 'nowhere', {}, "unknown dropoff 'nowhere' (transfer stations here: a, b, c)"),
    ('charger', 'b', {}, "'charger' is a charger, not a transfer station"),
    ('a', 'charger', {}, "'charger' is a charger, not a transfer station"),
    ('a', 'a', {}, "pickup and dropoff are both 'a'"),
    ('', 'b', {}, 'the order names no pickup and the vehicle carries no load'),
])
def test_orders_that_make_no_sense_are_refused(pickup, dropoff, options, expected):
    rig = Rig()
    assert rig.order(pickup, dropoff, **options) == expected
    assert not rig.mission.active and rig.goals == []


@pytest.mark.parametrize('arrange, expected', [
    (lambda r: setattr(r, 'safety', SafetyState.ESTOP),
     'the e-stop is active: release and reset it'),
    (lambda r: (setattr(r, 'safety', SafetyState.FAULT),
                setattr(r, 'safety_reason', 'scan is 0.50 s old')),
     'the safety gate reports a fault: scan is 0.50 s old'),
    (lambda r: setattr(r, 'command_source', 'manual'), 'an operator is driving the vehicle'),
    (lambda r: setattr(r, 'pose_lost', True), 'the vehicle does not know where it is'),
    (lambda r: setattr(r, 'load_report', LoadState.UNKNOWN), 'the load handler is not reporting'),
    (lambda r: setattr(r, 'load_report', LoadState.FAULT), 'the load handler is not reporting'),
    (lambda r: setattr(r, 'load_report', LoadState.MOVING), 'the load handler is moving a load'),
    (lambda r: setattr(r, 'load_report', LoadState.LOADED),
     'the vehicle already carries a load: send the order without a pickup to deliver it'),
    (lambda r: setattr(r, 'battery', 0.2), 'battery at 20%, orders need 25%'),
])
def test_a_vehicle_that_is_not_ready_refuses_the_order(arrange, expected):
    rig = Rig()
    arrange(rig)
    assert rig.order('a', 'b') == expected
    assert not rig.mission.active


def test_a_second_order_is_refused_while_one_is_running():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    assert rig.order('b', 'c') == "busy with an order to 'b'"
    assert rig.run_to_end().success


def test_returning_to_a_charger_the_site_does_not_have_is_refused():
    no_charger = site(stations={k: v for k, v in site_stations().items() if k != 'charger'})
    rig = Rig(the_site=no_charger, at='c')
    assert rig.order('a', 'b', return_to_charger=True) == 'the site has no charger to return to'
    assert rig.order('a', 'b') is None
    assert rig.run_to_end().success


def site_stations():
    from support import SITE
    return dict(SITE['stations'])


def test_an_unknown_battery_level_is_accepted_unless_it_is_required():
    assert Rig(battery=None).order('a', 'b') is None
    strict = Rig(battery=None, config=MissionConfig(require_battery=True))
    assert strict.order('a', 'b') == 'the battery level is not known'


def test_no_route_to_a_station_refuses_the_order():
    rig = started('a', 'b')
    rig.route_errors['b'] = 'the goal is inside a keep-out zone'
    result = rig.run_to_end()
    assert result.outcome == 'refused' and not result.success
    assert result.message == "no route to 'b': the goal is inside a keep-out zone"
    assert rig.goals == [] and rig.vehicle.distance == 0.0
    assert rig.states == ['checking', 'failed']


def test_the_way_to_the_charger_after_the_delivery_is_checked_too():
    rig = started('a', 'b')
    rig.route_errors['charger'] = 'no path'
    result = rig.run_to_end()
    assert result.outcome == 'refused' and result.message == "no route to 'charger': no path"
    # Three legs were asked about: to a, a to b, b to the charger.
    assert len(rig.route_requests) == 3


def test_a_route_check_that_never_answers_refuses_the_order():
    rig = started('a', 'b')
    rig.route_silent = True
    result = rig.run_to_end()
    assert result.outcome == 'refused' and result.message == 'the route check did not answer'
    assert result.duration == pytest.approx(rig.config.route_check_timeout, abs=0.2)
    assert rig.goals == []


def test_an_order_longer_than_the_battery_lasts_is_refused():
    # 26 % is above the starting level; make the hall's routes long enough to matter.
    rig = started('a', 'b', battery=0.26)
    rig.route_factor = 60.0
    result = rig.run_to_end()
    assert result.outcome == 'refused'
    assert 'the battery (26%) is good for' in result.message
    assert result.message.startswith('the order is ')
    assert rig.goals == []


def test_without_route_checks_the_order_starts_at_once():
    rig = started('a', 'b', config=MissionConfig(check_routes=False))
    rig.step()
    assert rig.route_requests == [] and [goal.station for goal in rig.goals] == ['a']
    assert rig.run_to_end().success


def test_the_route_lengths_set_the_deadline_of_each_leg():
    rig = started('a', 'b')
    rig.nav_stalls = True
    rig.run_until(lambda: len(rig.goals) == 2)
    # 9.2 m at 0.15 m/s plus 60 s, not the 300 s of an unknown leg.
    taken = rig.now - 100.0
    assert taken == pytest.approx(9.2 / 0.15 + 60.0 + rig.config.nav_retry_wait, abs=2.0)


# -- travel -----------------------------------------------------------------------------------


def test_navigation_giving_up_once_is_tried_again():
    rig = started('a', 'b')
    rig.nav_failures = 1
    rig.run_until(lambda: 'trying again (2 of 3)' in rig.status.detail)
    assert 'navigation gave up' in rig.status.detail
    result = rig.run_to_end()
    assert result.success
    assert [goal.station for goal in rig.goals] == ['a', 'a', 'b']
    # The retry waited before going again.
    assert rig.goals[1].goal_id != rig.goals[0].goal_id


def test_navigation_giving_up_every_time_fails_the_order():
    rig = started('a', 'b')
    rig.nav_failures = 99
    result = rig.run_to_end()
    assert result.outcome == 'failed' and not result.delivered
    assert result.message == "could not reach 'a' in 3 attempts: navigation gave up"
    assert len(rig.goals) == 3
    assert rig.status.state == 'failed'


def test_failing_with_the_load_on_board_says_so():
    rig = started('a', 'b')
    rig.nav_fails_at['b'] = 99
    result = rig.run_to_end()
    assert result.outcome == 'failed'
    assert result.message == ("could not reach 'b' in 3 attempts: navigation gave up "
                              '(the load is on board)')
    assert rig.status.carrying and rig.handler.state is LoadState.LOADED


def test_the_load_left_on_board_can_be_delivered_by_a_new_order():
    rig = started('a', 'b')
    rig.nav_fails_at['b'] = 99
    assert rig.run_to_end().outcome == 'failed'
    rig.nav_fails_at.clear()
    rig.results.clear()
    assert rig.order('a', 'c') == ('the vehicle already carries a load: send the order without '
                                   'a pickup to deliver it')
    assert rig.order('', 'c') is None
    assert rig.run_to_end().message == "delivered to 'c'"


def test_navigation_that_never_takes_the_goal_fails_the_order():
    rig = started('a', 'b')
    rig.nav_deaf = True
    result = rig.run_to_end()
    assert result.message == "could not reach 'a' in 3 attempts: navigation did not take the goal"
    assert result.duration == pytest.approx(3 * 5.0 + 2 * 5.0, abs=1.0)


def test_navigation_that_never_arrives_runs_into_the_leg_deadline():
    rig = started('a', 'b', config=MissionConfig(check_routes=False, leg_timeout_default=20.0))
    rig.nav_stalls = True
    result = rig.run_to_end()
    assert result.message == "could not reach 'a' in 3 attempts: not there after 20 s"
    assert rig.cancels == 3                        # each attempt was called off


def test_navigation_reporting_success_somewhere_else_is_not_believed():
    rig = started('a', 'b')
    rig.nav_lies = 1
    rig.run_until(lambda: 'trying again' in rig.status.detail)
    assert 'navigation stopped 9.20 m from the approach pose' in rig.status.detail
    assert rig.run_to_end().success


# -- docking ----------------------------------------------------------------------------------


def test_arriving_too_far_off_the_approach_pose_is_sent_there_again():
    rig = started('a', 'b')
    rig.arrival_errors = [(0.0, 0.28, 0.0)]        # 28 cm to the side of the approach pose
    rig.run_until(lambda: 'trying again (2 of 3)' in rig.status.detail)
    assert 'navigation stopped 0.28 m from the approach pose' in rig.status.detail
    assert rig.run_to_end().success
    assert [goal.station for goal in rig.goals] == ['a', 'a', 'b']


def push_off_line_when_docking(rig, station, pushes):
    """Shove the vehicle 5 cm sideways just short of the stop line, `pushes` runs in a row."""
    target = rig.site.station(station).pose
    done, goals_seen = 0, -1
    while not rig.results:
        rig.step()
        local = rig.vehicle.pose.relative_to(target)
        new_run = len(rig.goals) != goals_seen
        if (done < pushes and new_run and rig.status.state == 'docking'
                and -0.02 < local.x < 0.0 and abs(local.y) < 0.02):
            rig.vehicle.place(rig.vehicle.pose.advanced(0.0, 0.05))
            done, goals_seen = done + 1, len(rig.goals)
        assert rig.now < 1000.0
    return rig.results[-1]


def test_a_missed_dock_goes_back_to_the_approach_pose_and_in_again():
    rig = started('a', 'b')
    result = push_off_line_when_docking(rig, 'a', pushes=1)
    assert result.success
    # Back to a's approach pose (navigation was sent although it is only 0.8 m away), then on.
    assert [goal.station for goal in rig.goals] == ['a', 'a', 'b']
    details = ' | '.join(s.detail for s in rig.statuses)
    assert 'm off the centre line; approaching again (2 of 3)' in details
    assert rig.docked_at('b')
    # The load was only asked for once the vehicle was properly docked.
    assert [c.action for c in rig.load_commands] == ['load', 'unload']


def test_a_dock_that_never_works_fails_the_order():
    rig = started('a', 'b')
    result = push_off_line_when_docking(rig, 'a', pushes=3)
    assert result.outcome == 'failed'
    assert result.message.startswith("could not dock at 'a' in 3 attempts: 0.0")
    assert result.message.endswith('m off the centre line')
    assert [goal.station for goal in rig.goals] == ['a', 'a', 'a']
    assert rig.load_commands == []                 # never asked for a load it could not reach


def test_a_drive_that_does_not_move_runs_into_the_docking_deadline():
    rig = started('a', 'b', config=MissionConfig(dock_timeout=10.0))
    rig.run_until_state('docking')
    rig.drive_dead = True
    result = rig.run_to_end()
    assert result.message == "could not dock at 'a' in 3 attempts: not docked after 10 s"


def test_docking_precision_over_many_arrivals():
    """Whatever navigation's last 10 cm look like, the vehicle ends within 2 cm and 1.5 degrees."""
    worst_offset = worst_heading = 0.0
    for dx, dy, dyaw in [(0.1, 0.1, 0.1), (-0.1, -0.1, -0.1), (0.0, 0.1, -0.1), (0.1, -0.1, 0.0),
                         (-0.1, 0.05, 0.1), (0.05, -0.05, -0.05)]:
        rig = started('a', 'b')
        rig.arrival_errors = [(dx, dy, dyaw), (-dx, -dy, -dyaw)]
        assert rig.run_to_end().success
        offset, heading, _ = rig.docking_error('b')
        worst_offset, worst_heading = max(worst_offset, offset), max(worst_heading, heading)
        assert len(rig.goals) == 2                  # no second approach was needed
    assert worst_offset <= 0.02 and worst_heading <= 1.5


def test_a_charger_that_gives_no_current_does_not_undo_the_delivery():
    rig = started('a', 'b', return_to_charger=True)
    rig.charger_dead = True
    result = rig.run_to_end()
    assert result.success and result.delivered
    assert result.message == ("delivered from 'a' to 'b'; could not return to the charger: "
                              "could not dock at 'charger' in 3 attempts: docked but not charging")
    assert rig.status.state == 'complete'
    assert 'waiting for charging current' in ' '.join(s.detail for s in rig.statuses)


def test_a_charger_is_docked_without_waiting_when_the_battery_is_not_reported():
    rig = started('a', 'b', return_to_charger=True, battery=None)
    rig.charger_dead = True
    assert rig.run_to_end().message == "delivered from 'a' to 'b', back at the charger"


# -- the load ---------------------------------------------------------------------------------


def test_a_load_command_that_is_refused_is_sent_again():
    rig = started('a', 'b')
    rig.handler.rejections = 1
    result = rig.run_to_end()
    assert result.success
    assert [(c.action, c.station) for c in rig.load_commands] == [
        ('load', 'a'), ('load', 'a'), ('unload', 'b')]
    retry = rig.handler.commands[1][0] - rig.handler.commands[0][0]
    assert retry == pytest.approx(rig.config.load_retry_period, abs=0.2)


def test_a_load_that_never_comes_fails_the_order_with_the_vehicle_empty():
    rig = started('a', 'b')
    rig.handler.rejections = 99
    result = rig.run_to_end()
    assert result.outcome == 'failed' and not result.delivered
    assert result.message == "no load came aboard at 'a' after 3 commands"
    assert len(rig.load_commands) == 3 and not rig.status.carrying
    assert rig.docked_at('a')


def test_a_jammed_loading_runs_into_its_deadline():
    rig = started('a', 'b')
    rig.handler.stuck = True
    result = rig.run_to_end()
    assert result.message == "no load came aboard at 'a' after 40 s"
    assert len(rig.load_commands) == 1             # it was moving: no point commanding again


def test_a_load_that_will_not_leave_is_not_delivered():
    rig = started('a', 'b')
    rig.run_until_state('to_dropoff')
    rig.handler.rejections = 99
    result = rig.run_to_end()
    assert result.outcome == 'failed' and not result.delivered
    assert result.message == ("the load did not leave the vehicle at 'b' after 3 commands "
                              '(the load is on board)')
    assert rig.status.carrying


def test_a_handler_reporting_a_fault_is_commanded_again_at_once():
    rig = started('a', 'b')
    rig.run_until_state('loading')
    rig.load_report = LoadState.FAULT
    result = rig.run_to_end()
    assert result.message == "no load came aboard at 'a' after 3 commands"
    assert result.duration < 30.0                   # no waiting out the retry period


def test_a_load_counts_only_once_the_handler_has_held_it_for_a_moment():
    rig = started('a', 'b')
    rig.run_until_state('loading')
    rig.run_until(lambda: rig.handler.state is LoadState.LOADED)
    loaded_at = rig.now
    rig.run_until_state('to_dropoff')
    assert rig.now - loaded_at == pytest.approx(rig.config.load_debounce, abs=0.15)


def test_a_load_lost_on_the_way_stops_the_order():
    rig = started('a', 'b')
    rig.run_until_state('to_dropoff')
    rig.run(2.0)
    rig.handler.drop()
    result = rig.run_to_end()
    assert result.outcome == 'failed' and result.message == 'the load is no longer on the vehicle'
    assert not rig.status.carrying
    assert rig.nav is NavState.CANCELED            # it stopped where it noticed
    assert rig.load_commands[-1].action == 'load'  # and never pretended to unload


def test_a_load_that_appears_from_nowhere_stops_the_order():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.load_report = LoadState.LOADED
    result = rig.run_to_end()
    assert result.message == ('a load is on the vehicle that this order did not put there '
                              '(the load is on board)')


def test_a_flicker_of_the_load_sensor_is_ignored():
    rig = started('a', 'b')
    rig.run_until_state('to_dropoff')
    rig.load_report = LoadState.EMPTY
    rig.run(rig.config.load_debounce - 0.3)
    rig.load_report = None
    assert rig.run_to_end().success


# -- holds ------------------------------------------------------------------------------------


def hold_and_release(rig, hold, release, state='to_pickup', held_for=8.0):
    rig.run_until_state(state)
    rig.run(2.0)
    goals_before = len(rig.goals)
    hold(rig)
    rig.run(held_for)
    held = rig.status
    position = rig.vehicle.pose
    release(rig)
    return held, position, goals_before


def test_the_estop_holds_the_order_and_it_carries_on_after_the_reset():
    rig = started('a', 'b')
    held, position, goals_before = hold_and_release(
        rig, lambda r: setattr(r, 'safety', SafetyState.ESTOP),
        lambda r: setattr(r, 'safety', SafetyState.OK))
    assert held.active and held.state == 'to_pickup'
    assert held.holding == 'e-stop active' and held.detail == 'holding: e-stop active'
    assert rig.nav is NavState.CANCELED and rig.cancels == 1
    result = rig.run_to_end()
    assert result.success
    assert len(rig.goals) == goals_before + 2       # the same leg again, then the next
    assert rig.goals[goals_before].station == 'a'
    assert rig.status.holding == ''


def test_an_estop_that_is_never_reset_fails_the_order():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.safety = SafetyState.ESTOP
    result = rig.run_to_end()
    assert result.outcome == 'failed' and result.message == 'e-stop active for 120 s'


@pytest.mark.parametrize('hold, release, reason', [
    (lambda r: (setattr(r, 'safety', SafetyState.FAULT),
                setattr(r, 'safety_reason', 'no scan received')),
     lambda r: (setattr(r, 'safety', SafetyState.OK), setattr(r, 'safety_reason', '')),
     'safety fault (no scan received)'),
    (lambda r: setattr(r, 'pose_lost', True), lambda r: setattr(r, 'pose_lost', False),
     'position lost'),
    (lambda r: setattr(r, 'command_source', 'manual'),
     lambda r: setattr(r, 'command_source', 'none'),
     'operator driving'),
])
def test_other_things_that_hold_an_order(hold, release, reason):
    rig = started('a', 'b')
    held, _, _ = hold_and_release(rig, hold, release)
    assert held.holding == reason and held.active
    assert rig.run_to_end().success


def test_the_order_waits_a_moment_after_the_operator_lets_go():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.run(2.0)
    rig.command_source = 'manual'
    rig.run(5.0)
    goals = len(rig.goals)
    rig.command_source = 'none'
    rig.run(rig.config.manual_resume_delay - 0.5)
    assert rig.status.holding == 'operator driving' and len(rig.goals) == goals
    rig.run(1.0)
    assert rig.status.holding == '' and len(rig.goals) == goals + 1


def test_a_hold_during_docking_starts_the_docking_run_again():
    rig = started('a', 'b')
    rig.run_until_state('docking')
    rig.run(1.0)
    rig.safety = SafetyState.ESTOP
    rig.run(5.0)
    assert rig.status.holding == 'e-stop active'
    rig.safety = SafetyState.OK
    result = rig.run_to_end()
    assert result.success and rig.docked_at('b')


def test_holds_do_not_use_up_the_deadline_of_the_step():
    rig = started('a', 'b', config=MissionConfig(dock_timeout=12.0))
    rig.run_until_state('docking')
    rig.safety = SafetyState.ESTOP
    rig.run(60.0)                                    # five docking deadlines
    rig.safety = SafetyState.OK
    assert rig.run_to_end().success


def test_stopped_for_an_obstacle_the_order_waits_and_then_carries_on():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.run(2.0)
    rig.safety, rig.safety_reason = SafetyState.PROTECTIVE_STOP, 'obstacle 0.31 m ahead'
    position = rig.vehicle.pose
    rig.run(30.0)
    status = rig.status
    assert status.active and status.holding == ''   # navigation keeps the goal: not a hold
    assert status.detail == 'stopped: obstacle 0.31 m ahead'
    assert rig.vehicle.pose.distance_to(position) < 0.05 and rig.cancels == 0
    rig.safety, rig.safety_reason = SafetyState.OK, ''
    assert rig.run_to_end().success


def test_time_stopped_for_an_obstacle_does_not_count_against_the_leg():
    rig = started('a', 'b', config=MissionConfig(
        check_routes=False, leg_timeout_default=25.0, blocked_timeout=200.0))
    rig.run_until_state('to_pickup')
    rig.safety = SafetyState.PROTECTIVE_STOP
    rig.run(100.0)                                   # four leg deadlines
    rig.safety = SafetyState.OK
    assert rig.run_to_end().success and len(rig.goals) == 2


def test_blocked_for_too_long_fails_the_order():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.safety, rig.safety_reason = SafetyState.PROTECTIVE_STOP, 'obstacle 0.20 m ahead'
    result = rig.run_to_end()
    assert result.outcome == 'failed'
    assert result.message == 'blocked for 90 s: obstacle 0.20 m ahead'
    assert rig.nav is NavState.CANCELED


def test_an_obstacle_that_comes_and_goes_never_adds_up_to_blocked():
    rig = started('a', 'b')
    rig.nav_speed = 0.05                             # a long leg
    rig.run_until_state('to_pickup')
    for _ in range(6):                               # 6 x 60 s stopped, never 90 s in one go
        rig.safety = SafetyState.PROTECTIVE_STOP
        rig.run(60.0)
        rig.safety = SafetyState.OK
        rig.run(2.0)
    assert rig.status.active and rig.results == []


# -- the battery ------------------------------------------------------------------------------


def test_a_critical_battery_stops_the_vehicle_where_it_is():
    rig = started('a', 'b')
    rig.run_until_state('to_dropoff')
    rig.battery = 0.07
    result = rig.run_to_end()
    assert result.outcome == 'failed'
    assert result.message == 'battery at 7%: stopped where it is (the load is on board)'
    assert rig.nav is NavState.CANCELED


def test_too_little_charge_on_the_way_to_the_pickup_goes_to_charge_instead():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.run(2.0)
    rig.battery = 0.14
    result = rig.run_to_end()
    assert result.outcome == 'failed' and not result.delivered
    assert result.message == "battery at 14%: went to charge instead of collecting from 'a'"
    assert rig.docked_at('charger') and rig.load_commands == []
    assert 'to_charger' in rig.states


def test_with_the_load_on_board_a_low_battery_still_delivers_first():
    rig = started('a', 'b')
    rig.run_until_state('to_dropoff')
    rig.battery = 0.14
    result = rig.run_to_end()
    assert result.success
    # Not asked to return, but low: it went to the charger after delivering.
    assert result.message == "delivered from 'a' to 'b', back at the charger"
    assert rig.docked_at('charger')


def test_idle_and_low_the_vehicle_goes_to_charge_by_itself():
    rig = Rig(at='b', battery=0.28)
    rig.run_until(lambda: rig.docked_at('charger') and not rig.mission.active)
    assert 'to_charger' in rig.states and rig.status.state == 'idle'
    assert rig.results == []                        # nobody ordered it: nothing to report
    assert rig.inputs().charging
    # Charging, it stays where it is.
    goals = len(rig.goals)
    rig.run(30.0)
    assert len(rig.goals) == goals


def test_charging_by_itself_can_be_switched_off():
    rig = Rig(at='b', battery=0.28, config=MissionConfig(auto_charge=False))
    rig.run(30.0)
    assert rig.goals == [] and rig.status.state == 'idle'


def test_an_order_takes_precedence_over_a_trip_to_the_charger():
    rig = Rig(at='b', battery=0.28)
    rig.run_until_state('to_charger')
    rig.run(2.0)
    assert not rig.mission.busy
    assert rig.order('a', 'c') is None
    result = rig.run_to_end()
    # Delivered, and then to the charger after all: the battery is still low.
    assert result.message == "delivered from 'a' to 'c', back at the charger"
    assert rig.cancels >= 1


def test_a_failed_trip_to_the_charger_is_not_tried_again_at_once():
    rig = Rig(at='b', battery=0.28)
    rig.nav_failures = 3
    rig.run_until(lambda: rig.status.state == 'failed')
    assert rig.status.detail.startswith('could not get to the charger:')
    goals = len(rig.goals)
    rig.run(rig.config.auto_charge_retry - 10.0)
    assert len(rig.goals) == goals
    rig.run_until(lambda: rig.docked_at('charger'), timeout=200.0)


def test_on_the_charger_with_no_current_it_does_not_drive_off_and_on():
    rig = Rig(at='charger', battery=0.2)
    rig.charger_dead = True
    rig.run(60.0)
    assert rig.goals == [] and rig.vehicle.distance == 0.0


# -- cancelling -------------------------------------------------------------------------------


def test_cancelling_on_the_way_to_the_pickup_goes_home():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.run(3.0)
    assert rig.mission.cancel() is True
    result = rig.run_to_end()
    assert result.outcome == 'canceled' and not result.success and not result.delivered
    assert result.message == 'cancelled, back at the charger'
    assert rig.docked_at('charger') and rig.load_commands == []
    assert rig.status.state == 'canceled'
    assert any(s.detail.startswith('cancelled: ') for s in rig.statuses)


def test_cancelling_with_the_load_on_board_takes_it_home():
    rig = started('a', 'b')
    rig.run_until_state('to_dropoff')
    rig.run(2.0)
    rig.mission.cancel()
    result = rig.run_to_end()
    assert result.message == 'cancelled with the load on board, back at the charger'
    assert rig.handler.state is LoadState.LOADED and rig.docked_at('charger')


def test_a_transfer_under_way_is_finished_before_a_cancel_takes_effect():
    rig = started('a', 'b')
    rig.run_until(lambda: rig.handler.state is LoadState.MOVING)
    rig.mission.cancel()
    rig.run(1.0)
    assert rig.status.state == 'loading' and rig.goals[-1].station == 'a'
    result = rig.run_to_end()
    assert result.message == 'cancelled with the load on board, back at the charger'


def test_cancelling_while_the_order_is_still_being_checked_ends_it_at_once():
    rig = started('a', 'b')
    rig.route_silent = True
    rig.run(1.0)
    rig.mission.cancel()
    result = rig.run_to_end()
    assert result.outcome == 'canceled' and result.message == 'cancelled'
    assert rig.goals == []


def test_cancelling_where_there_is_no_charger_stops_on_the_spot():
    no_charger = site(stations={k: v for k, v in site_stations().items() if k != 'charger'})
    rig = Rig(the_site=no_charger, at='c')
    rig.order('a', 'b')
    rig.run_until_state('to_pickup')
    rig.run(2.0)
    rig.mission.cancel()
    result = rig.run_to_end()
    assert result.message == 'cancelled' and rig.nav is NavState.CANCELED


def test_a_cancelled_vehicle_that_cannot_get_home_says_so():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.mission.cancel()
    rig.nav_failures = 99
    result = rig.run_to_end()
    assert result.outcome == 'canceled'
    assert result.message == ('cancelled; could not return to the charger: could not reach '
                              "'charger' in 3 attempts: navigation gave up")


def test_there_is_nothing_to_cancel_when_idle():
    rig = Rig()
    assert rig.mission.cancel() is False
    assert Rig(at='b', battery=0.28).mission.cancel() is False


def test_cancelling_a_hold_takes_effect_when_the_hold_clears():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.safety = SafetyState.ESTOP
    rig.run(3.0)
    rig.mission.cancel()
    rig.run(3.0)
    rig.safety = SafetyState.OK
    assert rig.run_to_end().message == 'cancelled, back at the charger'


# -- keeping count ----------------------------------------------------------------------------


def test_a_jump_of_the_localiser_is_not_counted_as_distance():
    rig = started('a', 'b')
    rig.run_until_state('to_pickup')
    rig.run(2.0)
    before = rig.status.distance_driven
    rig.pose_frozen = Pose2D(rig.vehicle.pose.x + 30.0, rig.vehicle.pose.y, 0.0)
    rig.step()
    rig.pose_frozen = None
    rig.step()
    assert rig.status.distance_driven - before < 0.5


def test_every_wait_in_an_order_has_a_deadline():
    """Whatever stops answering, the order ends by itself with a reason."""
    breakages = {
        'navigation takes no goal': lambda r: setattr(r, 'nav_deaf', True),
        'navigation never arrives': lambda r: setattr(r, 'nav_stalls', True),
        'route check silent': lambda r: setattr(r, 'route_silent', True),
        'load handler jammed': lambda r: setattr(r.handler, 'stuck', True),
        'drive dead': lambda r: setattr(r, 'drive_dead', True),
        'e-stop never reset': lambda r: setattr(r, 'safety', SafetyState.ESTOP),
        'position never comes back': lambda r: setattr(r, 'pose_lost', True),
        'blocked for good': lambda r: setattr(r, 'safety', SafetyState.PROTECTIVE_STOP),
    }
    for name, break_it in breakages.items():
        rig = started('a', 'b')
        break_it(rig)
        result = rig.run_to_end(timeout=3600.0)
        assert not result.success and result.message, name
        assert not rig.mission.active, name


def test_headings_do_not_matter_to_arrival_checks_beyond_their_tolerance():
    rig = started('a', 'b')
    rig.arrival_errors = [(0.0, 0.0, math.radians(30.0))]
    assert rig.run_to_end().success
    assert len(rig.goals) == 2
