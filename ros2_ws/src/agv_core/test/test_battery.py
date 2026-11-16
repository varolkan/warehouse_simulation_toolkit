"""The battery model."""

from agv_core.battery import Battery, BatteryConfig
import pytest

CFG = BatteryConfig(idle_hours=10.0, range_km=10.0, charge_hours=1.0)


def test_standing_switched_on_drains_it_in_the_idle_time():
    battery = Battery(CFG)
    battery.step(5 * 3600.0, 0.0, charging=False)
    assert battery.level == pytest.approx(0.5)
    battery.step(5 * 3600.0, 0.0, charging=False)
    assert battery.level == pytest.approx(0.0) and battery.empty


def test_driving_costs_distance_on_top_of_time():
    battery = Battery(CFG)
    battery.step(0.0, 2500.0, charging=False)
    assert battery.level == pytest.approx(0.75)
    # An hour at 1 m/s: 3.6 km of driving and an hour of being on.
    battery = Battery(CFG)
    for _ in range(3600):
        battery.step(1.0, 1.0, charging=False)
    assert battery.level == pytest.approx(1.0 - 0.36 - 0.1)
    # Reversing costs the same as driving forward.
    assert Battery(CFG).step(0.0, -2500.0, charging=False) == pytest.approx(0.75)


def test_charging_fills_it_in_the_charge_time_and_stops_at_full():
    battery = Battery(CFG, level=0.2)
    battery.step(1800.0, 0.0, charging=True)
    assert battery.level == pytest.approx(0.2 + 0.5 - 0.05)
    battery.step(3600.0, 0.0, charging=True)
    assert battery.level == 1.0


def test_the_level_stays_between_empty_and_full():
    assert Battery(CFG, level=1.7).level == 1.0
    assert Battery(CFG, level=-0.3).level == 0.0
    battery = Battery(CFG, level=0.01)
    assert battery.step(0.0, 5000.0, charging=False) == 0.0


def test_range_is_what_is_left_above_the_reserve():
    # Driving only (so fast the standing cost vanishes): 10 km for a full battery.
    assert CFG.range_m(1.0, 0.0, speed=1e9) == pytest.approx(10000.0, rel=1e-6)
    assert CFG.range_m(0.5, 0.1, speed=1e9) == pytest.approx(4000.0, rel=1e-6)
    # Slower means longer on the road, and the standing drain eats into the range.
    fast, slow = CFG.range_m(0.5, 0.1, speed=1.0), CFG.range_m(0.5, 0.1, speed=0.2)
    assert slow < fast < 4000.0
    assert fast == pytest.approx(0.4 / (1e-4 + 1.0 / 36000.0))
    assert CFG.range_m(0.1, 0.1, speed=1.0) == 0.0
    assert CFG.range_m(0.05, 0.1, speed=1.0) == 0.0
    assert CFG.range_m(0.5, 0.1, speed=0.0) == 0.0


def test_driving_the_range_uses_what_range_promised():
    level, reserve, speed = 0.6, 0.2, 0.5
    distance = CFG.range_m(level, reserve, speed)
    battery = Battery(CFG, level=level)
    battery.step(distance / speed, distance, charging=False)
    assert battery.level == pytest.approx(reserve)


def test_the_default_battery_is_a_working_day():
    default = BatteryConfig()
    assert 5000.0 < default.range_m(1.0, 0.1, speed=0.6) < 14000.0
