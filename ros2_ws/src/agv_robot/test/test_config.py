"""The vehicle is described in several formats; they all have to describe the same vehicle.

agv_core.vehicle holds the numbers. The URDF, Nav2's footprints and limits, the scan filter
and the safety gate each write them out again because each format wants them; these tests
fail the moment one of them is changed alone.
"""

import os
import subprocess
import xml.etree.ElementTree as ET

from agv_core.masks import SPEED_STEP
from agv_core.safety import SafetyConfig
from agv_core.vehicle import AGV
from ament_index_python.packages import get_package_share_directory
import pytest
import yaml

SHARE = get_package_share_directory('agv_robot')


def params(name):
    with open(os.path.join(SHARE, 'config', name)) as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope='module')
def urdf():
    xacro = os.path.join(SHARE, 'urdf', 'agv.urdf.xacro')
    text = subprocess.run(['xacro', xacro, 'sim:=true'], check=True, capture_output=True,
                          text=True).stdout
    return ET.fromstring(text)


def floats(text):
    return [float(v) for v in text.split()]


def joint_origin(urdf, name):
    joint = next(j for j in urdf.findall('joint') if j.get('name') == name)
    return floats(joint.find('origin').get('xyz'))


def test_the_urdf_body_is_the_vehicle_outline(urdf):
    base = next(link for link in urdf.findall('link') if link.get('name') == 'base_link')
    size = floats(base.find('collision/geometry/box').get('size'))
    assert size[:2] == pytest.approx([AGV.length, AGV.width])


def test_the_urdf_wheels_and_scanner_are_where_the_vehicle_says(urdf):
    wheel = next(link for link in urdf.findall('link') if link.get('name') == 'left_wheel_link')
    radius = float(wheel.find('visual/geometry/cylinder').get('radius'))
    assert radius == pytest.approx(AGV.wheel_radius)
    left, right = joint_origin(urdf, 'left_wheel_joint'), joint_origin(urdf, 'right_wheel_joint')
    assert left[1] - right[1] == pytest.approx(AGV.wheel_separation)
    base = joint_origin(urdf, 'base_joint')[2]
    assert base + joint_origin(urdf, 'lidar_joint')[2] == pytest.approx(AGV.lidar_height)
    # The diff drive plugin drives with the same wheels.
    plugin = next(p for p in urdf.iter('plugin') if p.get('name') == 'gz::sim::systems::DiffDrive')
    assert float(plugin.findtext('wheel_radius')) == pytest.approx(AGV.wheel_radius)
    assert float(plugin.findtext('wheel_separation')) == pytest.approx(AGV.wheel_separation)


def test_the_top_of_the_deck_is_the_deck_height(urdf):
    deck = next(link for link in urdf.findall('link') if link.get('name') == 'deck_link')
    thickness = floats(deck.find('collision/geometry/box').get('size'))[2]
    top = joint_origin(urdf, 'base_joint')[2] + joint_origin(urdf, 'deck_joint')[2] + thickness / 2
    assert top == pytest.approx(AGV.deck_height)


def test_nav2_footprints_are_the_vehicle_outline():
    nav2 = params('nav2.yaml')
    expected = sorted((round(x, 3), round(y, 3)) for x, y in AGV.footprint)
    for costmap in ('local_costmap', 'global_costmap'):
        footprint = yaml.safe_load(nav2[costmap][costmap]['ros__parameters']['footprint'])
        assert sorted((round(x, 3), round(y, 3)) for x, y in footprint) == expected, costmap


def test_nav2_speed_limits_are_the_vehicles():
    smoother = params('nav2.yaml')['velocity_smoother']['ros__parameters']
    assert smoother['max_velocity'][0] == pytest.approx(AGV.max_speed)
    assert smoother['min_velocity'][0] == pytest.approx(-AGV.max_reverse_speed)
    assert smoother['max_velocity'][2] == pytest.approx(AGV.max_turn_rate)
    assert smoother['max_accel'][0] == pytest.approx(AGV.max_accel)
    assert smoother['max_decel'][0] == pytest.approx(-AGV.max_decel)


def test_the_safety_gate_caps_at_the_vehicles_limits():
    gate = SafetyConfig()
    assert (gate.max_speed, gate.max_reverse_speed, gate.max_turn_rate) == pytest.approx(
        (AGV.max_speed, AGV.max_reverse_speed, AGV.max_turn_rate))
    # A protective stop must be possible with the brakes the drive has.
    assert gate.brake_decel <= AGV.brake_decel


def test_the_controller_and_the_costmap_agree_on_inflation():
    nav2 = params('nav2.yaml')
    follow = nav2['controller_server']['ros__parameters']['FollowPath']
    local = nav2['local_costmap']['local_costmap']['ros__parameters']['inflation_layer']
    assert follow['inflation_cost_scaling_factor'] == local['cost_scaling_factor']


def test_the_speed_mask_is_read_in_the_steps_it_is_written_in():
    info = params('nav2.yaml')['speed_filter_info_server']['ros__parameters']
    assert info['type'] == 2 and info['base'] == 0.0
    assert info['multiplier'] == pytest.approx(SPEED_STEP)


def test_the_scan_filter_removes_the_vehicle_and_nothing_much_more():
    box = params('scan_filter.yaml')['scan_filter']['ros__parameters']['filter1']['params']
    assert box['box_frame'] == 'base_link'
    for low, high, half in ((box['min_x'], box['max_x'], AGV.half_length),
                            (box['min_y'], box['max_y'], AGV.half_width)):
        assert low == -high
        # Covers the outline (the deck posts are inside it), by no more than 3 cm.
        assert half <= high <= half + 0.03
