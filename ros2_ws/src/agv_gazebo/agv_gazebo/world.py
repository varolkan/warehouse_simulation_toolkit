"""The Gazebo world, generated from the site file.

The hall itself (the ARIAC workcell, the row of bins) is fixed. What stands at each
station is not: a transfer stand in front of every transfer station and a charging post at
every charger, placed from the same site file the mission and Nav2 read. The scanner sees
them, so they are part of the map; move a station and the world, the map and the masks all
have to follow, and the tests say so.

Also here: where the load goes when it is on a stand or on the vehicle, which the
simulated load handler and the launch file share.
"""

import math
from typing import List, Tuple
from xml.sax.saxutils import escape

from agv_core.geometry import Pose2D
from agv_core.sites import Site, Station
from agv_core.vehicle import AGV, Vehicle

WORLD_NAME = 'warehouse'

# Clear floor between the docked vehicle's front and whatever the station is.
DOCKING_GAP = 0.15

# A transfer stand: a plinth as high as the vehicle's deck. depth is along the approach.
STAND_DEPTH = 0.40
STAND_WIDTH = 0.50
# A charging post.
CHARGER_DEPTH = 0.20
CHARGER_WIDTH = 0.40
CHARGER_HEIGHT = 0.45

# The row of six bins the fork's origin placed along the north side.
BIN_POSITIONS = [(float(x), 6.0) for x in range(3, 9)]

# The ARIAC workcell has walls to the west and south only. A guard barrier closes the
# other two sides just inside the edge of its floor (20.1 m square, centred on the origin).
BARRIER_HEIGHT = 1.1
BARRIER_THICKNESS = 0.06
BARRIER_EAST = (9.83, 0.05, 19.62)      # x of the centre line, centre y, length
BARRIER_NORTH = (0.43, 9.83, 18.86)     # centre x, y of the centre line, length


def stand_pose(station: Station, vehicle: Vehicle = AGV) -> Pose2D:
    """Return the centre of a station's stand or charging post on the floor."""
    depth = CHARGER_DEPTH if station.is_charger else STAND_DEPTH
    return station.pose.advanced(vehicle.half_length + DOCKING_GAP + depth / 2.0)


# x, y, z and yaw of the load's base.
LoadPose = Tuple[float, float, float, float]


def load_pose_on_stand(station: Station, vehicle: Vehicle = AGV) -> LoadPose:
    """Return where the load's base is when it stands on a station's transfer stand."""
    centre = stand_pose(station, vehicle)
    return (centre.x, centre.y, vehicle.deck_height, centre.yaw)


def load_pose_on_vehicle(vehicle_pose: Pose2D, vehicle: Vehicle = AGV) -> LoadPose:
    """Return where the load's base is when it is on the vehicle's deck."""
    return (vehicle_pose.x, vehicle_pose.y, vehicle.deck_height, vehicle_pose.yaw)


def _box(name: str, x: float, y: float, z: float, yaw: float, size: Tuple[float, float, float],
         colour: Tuple[float, float, float], collide: bool = True) -> str:
    sx, sy, sz = size
    ambient = ' '.join(f'{0.8 * c:.3f}' for c in colour)
    diffuse = ' '.join(f'{c:.3f}' for c in colour)
    geometry = f'<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>'
    pose = f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
    collision = ''
    if collide:
        collision = f'\n        <collision name="{name}">{pose}{geometry}</collision>'
    return (f'{collision}\n        <visual name="{name}">{pose}{geometry}\n'
            f'          <material><ambient>{ambient} 1</ambient><diffuse>{diffuse} 1</diffuse>'
            '<specular>0.1 0.1 0.1 1</specular></material></visual>')


def _station_model(station: Station, vehicle: Vehicle) -> str:
    centre = stand_pose(station, vehicle)
    c, s = math.cos(centre.yaw), math.sin(centre.yaw)
    parts: List[str] = []
    if station.is_charger:
        parts.append(_box('post', centre.x, centre.y, CHARGER_HEIGHT / 2.0, centre.yaw,
                          (CHARGER_DEPTH, CHARGER_WIDTH, CHARGER_HEIGHT), (0.93, 0.74, 0.08)))
        # Two contact plates on the face the vehicle drives up to.
        face = -CHARGER_DEPTH / 2.0 - 0.005
        for side in (0.09, -0.09):
            parts.append(_box(
                f'contact_{"left" if side > 0 else "right"}',
                centre.x + c * face - s * side, centre.y + s * face + c * side, 0.11, centre.yaw,
                (0.01, 0.06, 0.05), (0.72, 0.45, 0.20), collide=False))
        kind = 'charger'
    else:
        parts.append(_box('plinth', centre.x, centre.y, (vehicle.deck_height - 0.02) / 2.0,
                          centre.yaw, (STAND_DEPTH, STAND_WIDTH, vehicle.deck_height - 0.02),
                          (0.36, 0.39, 0.42)))
        parts.append(_box('top', centre.x, centre.y, vehicle.deck_height - 0.01, centre.yaw,
                          (STAND_DEPTH, STAND_WIDTH, 0.02), (0.05, 0.62, 0.30)))
        kind = 'stand'
    # The bay painted on the floor where the vehicle stands when docked: seen by cameras,
    # not by the scanner.
    bay = station.pose
    parts.append(_box('bay', bay.x, bay.y, 0.0015, bay.yaw,
                      (vehicle.length + 0.14, vehicle.width + 0.14, 0.003),
                      (0.93, 0.74, 0.08) if station.is_charger else (0.05, 0.62, 0.30),
                      collide=False))
    return (f'    <model name="{kind}_{escape(station.name)}">\n'
            '      <static>true</static>\n'
            f'      <link name="link">{"".join(parts)}\n'
            '      </link>\n'
            '    </model>\n')


def _barrier() -> str:
    colour = (0.93, 0.74, 0.08)
    east = _box('east', BARRIER_EAST[0], BARRIER_EAST[1], BARRIER_HEIGHT / 2.0, 0.0,
                (BARRIER_THICKNESS, BARRIER_EAST[2], BARRIER_HEIGHT), colour)
    north = _box('north', BARRIER_NORTH[0], BARRIER_NORTH[1], BARRIER_HEIGHT / 2.0, 0.0,
                 (BARRIER_NORTH[2], BARRIER_THICKNESS, BARRIER_HEIGHT), colour)
    return ('    <model name="barrier">\n'
            '      <static>true</static>\n'
            f'      <link name="link">{east}{north}\n'
            '      </link>\n'
            '    </model>\n')


def world_sdf(site: Site, vehicle: Vehicle = AGV) -> str:
    """Return the whole world file for a site."""
    bins = ''.join(
        f'    <include><uri>model://workcell_bin</uri><name>bin_{i + 1}</name>'
        f'<pose>{x:.1f} {y:.1f} 0 0 0 0</pose></include>\n'
        for i, (x, y) in enumerate(BIN_POSITIONS))
    stations = ''.join(_station_model(s, vehicle) for s in site.stations.values())
    return f"""<?xml version="1.0" ?>
<!--
  Generated by agv_gazebo/scripts/generate_world.py from the site file of '{escape(site.name)}'.
  Do not edit: change the site file (or agv_gazebo/world.py) and generate it again.
-->
<sdf version="1.9">
  <world name="{WORLD_NAME}">
    <!-- 4 ms steps: the vehicle is slow and its wheels are spheres, 250 Hz is plenty. -->
    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <!-- The LiDAR is a rendering sensor: it sees what a camera would, not the collisions. -->
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.75 0.78 0.80 1</background>
      <shadows>false</shadows>
    </scene>
    <light name="sun" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 12 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Under the hall, 2 cm below its floor, so the vehicle only ever touches the floor. -->
    <model name="ground">
      <static>true</static>
      <pose>0 0 -0.02 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <material>
            <ambient>0.45 0.45 0.45 1</ambient><diffuse>0.45 0.45 0.45 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <include><uri>model://workcell</uri><name>workcell</name><pose>0 0 0 0 0 0</pose></include>
{bins}{_barrier()}{stations}  </world>
</sdf>
"""
