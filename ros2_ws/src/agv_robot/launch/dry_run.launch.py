r"""The whole stack against a kinematic stand-in for the vehicle: no Gazebo, no GPU.

    ros2 launch agv_robot dry_run.launch.py
    ros2 action send_goal --feedback /transport agv_interfaces/action/Transport \
        "{pickup: inbound, dropoff: packing}"

fake_base stands in for the vehicle: its scanner casts rays through the site's map, so
AMCL, Nav2, the safety gate and the mission run exactly as on the vehicle, in real time.
It is what the launch tests and CI run.

What can be arranged:
  start:=rack_a                   where the vehicle starts: a station, or "x,y,heading_deg"
  obstacles:="box 6.5 0 2.9 1.2; disc 2 3 0.3"
                                  things in the hall that are not in the map; change them
                                  while it runs on /sim/obstacles (std_msgs/String, one per line)
  loaded:=true                    a load on board from the start
  battery:=0.3                    state of charge, 0..1
  load_rejections:=2              the load handler refuses its first 2 commands
  load_stuck:=true                and with this, jams on the first one it takes
  charger_dead:=true              docking at the charger gives no current
"""

import math
import os

from agv_core.sites import Site
from agv_robot.paths import site_file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _stack(context):
    share = get_package_share_directory('agv_robot')
    site_name = LaunchConfiguration('site').perform(context)
    start = LaunchConfiguration('start').perform(context)
    site = Site.load(site_file(site_name))
    if site.has_station(start):
        pose = site.station(start).pose
        x, y, yaw = pose.x, pose.y, pose.yaw
    else:
        x, y, heading = (float(v) for v in start.split(','))
        yaw = math.radians(heading)
    obstacles = [o.strip() for o in LaunchConfiguration('obstacles').perform(context).split(';')]
    obstacles = [o for o in obstacles if o] or ['']

    def flag(name):
        return LaunchConfiguration(name).perform(context) == 'true'

    fake_base = Node(
        package='agv_robot', executable='fake_base_node', name='fake_base', output='screen',
        parameters=[{
            'site': site_name, 'start': start, 'obstacles': obstacles,
            'loaded': flag('loaded'), 'load_stuck': flag('load_stuck'),
            'charger_dead': flag('charger_dead'),
            'battery': float(LaunchConfiguration('battery').perform(context)),
            'load_rejections': int(LaunchConfiguration('load_rejections').perform(context)),
        }],
    )
    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'bringup.launch.py')),
        launch_arguments={
            'site': site_name, 'use_sim_time': 'false',
            'initial_x': str(x), 'initial_y': str(y), 'initial_yaw': str(yaw),
            'log_level': LaunchConfiguration('log_level').perform(context),
        }.items(),
    )
    return [fake_base, stack]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('site', default_value='warehouse'),
        DeclareLaunchArgument('start', default_value='charger'),
        DeclareLaunchArgument('obstacles', default_value=''),
        DeclareLaunchArgument('loaded', default_value='false'),
        DeclareLaunchArgument('battery', default_value='0.8'),
        DeclareLaunchArgument('load_rejections', default_value='0'),
        DeclareLaunchArgument('load_stuck', default_value='false'),
        DeclareLaunchArgument('charger_dead', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        OpaqueFunction(function=_stack),
    ])
