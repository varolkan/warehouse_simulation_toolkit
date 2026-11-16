"""The shipped site: it loads, fits its map, its generated files are current, all is reachable."""

import os
import subprocess
import sys

from agv_core.grid import GridMap
from agv_core.masks import keepout_mask, site_map_problems
from agv_core.sites import Site
from agv_core.vehicle import AGV
import pytest

SOURCE = os.environ.get('AGV_ROBOT_SOURCE', os.path.join(os.path.dirname(__file__), '..'))
SITE_DIR = os.path.join(SOURCE, 'sites', 'warehouse')


@pytest.fixture(scope='module')
def site():
    return Site.load(os.path.join(SITE_DIR, 'site.yaml'))


@pytest.fixture(scope='module')
def grid():
    return GridMap.load(os.path.join(SITE_DIR, 'map.yaml'))


def test_the_site_has_stations_of_both_kinds(site):
    assert site.chargers and len(site.transfer_stations) >= 2


def test_every_station_fits_the_map(site, grid):
    assert site.vehicle_problems(AGV.half_length, AGV.half_width) == []
    assert site_map_problems(site, grid, AGV.half_length, AGV.half_width) == []


def test_the_zone_masks_are_up_to_date():
    script = os.path.join(SOURCE, 'scripts', 'generate_masks.py')
    result = subprocess.run([sys.executable, script, '--check', SITE_DIR],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_every_station_can_be_reached_from_every_other(site, grid):
    """On the map with the keep-out zones closed, with room for the vehicle's width."""
    blocked = keepout_mask(site, grid).occupied
    names = list(site.stations)
    clearance = AGV.half_width + 0.05
    for i, start in enumerate(names):
        for goal in names[i + 1:]:
            a, b = site.station(start).approach_pose, site.station(goal).approach_pose
            route = grid.plan((a.x, a.y), (b.x, b.y), clearance, extra_blocked=blocked)
            assert route is not None, f'no way from {start} to {goal}'


def test_no_station_is_in_a_speed_zone_it_cannot_dock_at(site):
    """Docking crawls at 0.25 m/s at most: any speed zone a station is in has to allow that."""
    for station in site.stations.values():
        limit = site.speed_limit_at((station.pose.x, station.pose.y))
        assert limit is None or limit >= 0.25, station.name
