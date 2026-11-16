"""The Gazebo world agrees with the site file, and holds what the simulation needs."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from agv_core.sites import Site
from agv_gazebo.world import BIN_POSITIONS, stand_pose, WORLD_NAME
from agv_robot.paths import site_file
import pytest

SOURCE = os.environ.get('AGV_GAZEBO_SOURCE', os.path.join(os.path.dirname(__file__), '..'))
WORLD = os.path.join(SOURCE, 'worlds', 'warehouse.sdf')
GENERATOR = os.path.join(SOURCE, 'scripts', 'generate_world.py')
SITE = site_file('warehouse')


@pytest.fixture(scope='module')
def world():
    return ET.parse(WORLD).getroot().find('world')


def models(world):
    return {m.get('name'): m for m in world.findall('model')}


def test_the_committed_world_is_what_the_site_file_gives():
    result = subprocess.run([sys.executable, GENERATOR, '--check', SITE, WORLD],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_world_name_is_the_one_the_launch_and_bridge_use(world):
    assert world.get('name') == WORLD_NAME


def test_the_systems_the_vehicle_needs_are_loaded(world):
    loaded = {p.get('filename') for p in world.findall('plugin')}
    for required in ('gz-sim-physics-system', 'gz-sim-sensors-system', 'gz-sim-imu-system',
                     'gz-sim-user-commands-system', 'gz-sim-scene-broadcaster-system'):
        assert required in loaded


def test_every_station_has_its_stand_where_the_site_puts_it(world):
    site = Site.load(SITE)
    present = models(world)
    for station in site.stations.values():
        name = f"{'charger' if station.is_charger else 'stand'}_{station.name}"
        assert name in present, name
        first = present[name].find('link').find('collision')
        x, y = (float(v) for v in first.findtext('pose').split()[:2])
        centre = stand_pose(station)
        assert (x, y) == pytest.approx((centre.x, centre.y), abs=1e-3)


def test_the_hall_and_its_bins_are_included(world):
    includes = [i.findtext('uri') for i in world.findall('include')]
    assert includes.count('model://workcell') == 1
    assert includes.count('model://workcell_bin') == len(BIN_POSITIONS)
    for uri in set(includes):
        assert os.path.isdir(os.path.join(SOURCE, 'models', uri.split('//')[1])), uri
