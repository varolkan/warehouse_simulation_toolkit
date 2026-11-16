"""The vehicle's whole software stack: everything but the drive, the scanner and the load.

    ros2 launch agv_robot bringup.launch.py                  # work on a mapped site
    ros2 launch agv_robot bringup.launch.py mapping:=true    # map a new one

What starts:
  robot_state_publisher   the vehicle's frames                    description.launch.py
  scan_filter, ekf_node   the vehicle out of its own scan; odometry  localization.launch.py
  map_server, amcl        localisation on the site's map (slam_toolbox when mapping)
  Nav2                    planner, controller, recoveries, zones  navigation.launch.py
  safety_node             the gate between every command and the drive
  mission_node            the `transport` action (not while mapping)

The vehicle (or the simulator, or fake_base for a dry run) provides scan_raw,
wheel/odometry, imu, joint_states, battery_state, load/state and load/command, and drives
on cmd_vel.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('agv_robot')
    use_sim_time = LaunchConfiguration('use_sim_time')
    site = LaunchConfiguration('site')
    mapping = LaunchConfiguration('mapping')

    def include(name, arguments):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', name)),
            launch_arguments=arguments.items())

    return LaunchDescription([
        DeclareLaunchArgument(
            'site', default_value='warehouse',
            description='Site to work on: a name under agv_robot/sites, or a directory.'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'mapping', default_value='false',
            description='Map the site with slam_toolbox instead of working on its map.'),
        DeclareLaunchArgument(
            'sim', default_value='false',
            description='Publish the description with its Gazebo parts (for spawning).'),
        DeclareLaunchArgument('initial_x', default_value='',
                              description='Start pose for AMCL; default: the first charger.'),
        DeclareLaunchArgument('initial_y', default_value=''),
        DeclareLaunchArgument('initial_yaw', default_value=''),
        DeclareLaunchArgument('log_level', default_value='info'),

        include('description.launch.py', {
            'use_sim_time': use_sim_time, 'sim': LaunchConfiguration('sim')}),
        include('localization.launch.py', {
            'use_sim_time': use_sim_time, 'slam': mapping, 'site': site,
            'initial_x': LaunchConfiguration('initial_x'),
            'initial_y': LaunchConfiguration('initial_y'),
            'initial_yaw': LaunchConfiguration('initial_yaw')}),
        include('navigation.launch.py', {
            'use_sim_time': use_sim_time, 'mapping': mapping, 'site': site}),
        Node(
            package='agv_robot', executable='safety_node', name='safety_node', output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        ),
        Node(
            package='agv_robot', executable='mission_node', name='mission_node', output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'site': site}],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
            condition=UnlessCondition(mapping),
        ),
    ])
