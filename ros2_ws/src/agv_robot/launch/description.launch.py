"""Publish the vehicle description and its fixed transforms.

    ros2 launch agv_robot description.launch.py

`sim:=true` adds the Gazebo sensors and systems to the description, for the simulator to
spawn it from /robot_description.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    xacro_file = os.path.join(
        get_package_share_directory('agv_robot'), 'urdf', 'agv.urdf.xacro')
    description = ParameterValue(
        Command(['xacro ', xacro_file, ' sim:=', LaunchConfiguration('sim'),
                 ' load_joint:=', LaunchConfiguration('load_joint')]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'sim', default_value='false',
            description='Include the Gazebo sensors and systems in the description.'),
        DeclareLaunchArgument(
            'load_joint', default_value='true',
            description='Simulation: include the joint that holds the load on the deck.'),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robot_state_publisher', output='screen',
            parameters=[{
                'robot_description': description,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'publish_frequency': 30.0,
            }],
        ),
    ])
