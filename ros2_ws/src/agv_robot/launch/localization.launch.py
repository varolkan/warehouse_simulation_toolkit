"""Where the vehicle is: odometry, the scan filter, and SLAM or localisation on a map.

    ros2 launch agv_robot localization.launch.py slam:=true      # build a map
    ros2 launch agv_robot localization.launch.py                 # localise on the site's map

Always running:
  scan_filter   /scan_raw -> /scan with the vehicle's own posts removed
  ekf_node      wheel odometry + gyro -> /odom and odom -> base_footprint

With slam:=true, slam_toolbox maps and publishes map -> odom. Otherwise the site's map is
served and AMCL localises on it. AMCL needs to be told where it starts: by default that is
the site's first charger, where the vehicle is switched on; initial_x / initial_y /
initial_yaw (map frame, metres and radians) say otherwise.
"""

import os

from agv_core.sites import Site
from agv_robot.paths import MAP_FILE, site_file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _initial_pose(context, site_name):
    given = [LaunchConfiguration(f'initial_{axis}').perform(context) for axis in ('x', 'y', 'yaw')]
    if all(given):
        x, y, yaw = (float(value) for value in given)
    elif any(given):
        raise RuntimeError('give all of initial_x, initial_y and initial_yaw, or none of them')
    else:
        site = Site.load(site_file(site_name))
        if not site.chargers:
            raise RuntimeError(
                f"site '{site.name}' has no charger to start from: give initial_x, initial_y "
                'and initial_yaw')
        pose = site.chargers[0].pose
        x, y, yaw = pose.x, pose.y, pose.yaw
    return {'x': x, 'y': y, 'z': 0.0, 'yaw': yaw}


def _amcl(context):
    share = get_package_share_directory('agv_robot')
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context) == 'true'
    site_name = LaunchConfiguration('site').perform(context)
    map_file = site_file(site_name, MAP_FILE)
    if not os.path.isfile(map_file):
        raise RuntimeError(f'{map_file} not found: map the site first (slam:=true)')
    params = os.path.join(share, 'config', 'nav2.yaml')
    common = {'use_sim_time': use_sim_time}
    return [
        Node(
            package='nav2_map_server', executable='map_server', name='map_server',
            output='screen', parameters=[params, common, {'yaml_filename': map_file}],
        ),
        Node(
            package='nav2_amcl', executable='amcl', name='amcl', output='screen',
            parameters=[params, common, {
                'set_initial_pose': True,
                'initial_pose': _initial_pose(context, site_name),
            }],
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_localization', output='screen',
            parameters=[common, {
                'autostart': True, 'bond_timeout': 10.0,
                'node_names': ['map_server', 'amcl'],
            }],
        ),
    ]


def generate_launch_description():
    share = get_package_share_directory('agv_robot')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam = LaunchConfiguration('slam')

    scan_filter = Node(
        package='laser_filters', executable='scan_to_scan_filter_chain', name='scan_filter',
        output='screen',
        parameters=[os.path.join(share, 'config', 'scan_filter.yaml'),
                    {'use_sim_time': use_sim_time}],
        remappings=[('scan', 'scan_raw'), ('scan_filtered', 'scan')],
    )
    ekf = Node(
        package='robot_localization', executable='ekf_node', name='ekf_node', output='screen',
        parameters=[os.path.join(share, 'config', 'ekf.yaml'), {'use_sim_time': use_sim_time}],
        remappings=[('odometry/filtered', 'odom')],
    )
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')),
        launch_arguments={
            'slam_params_file': os.path.join(share, 'config', 'slam.yaml'),
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(slam),
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'slam', default_value='false',
            description='Build a map with slam_toolbox instead of localising on the site map.'),
        DeclareLaunchArgument(
            'site', default_value='warehouse',
            description='Site to localise on: a name under agv_robot/sites, or a directory.'),
        DeclareLaunchArgument('initial_x', default_value=''),
        DeclareLaunchArgument('initial_y', default_value=''),
        DeclareLaunchArgument('initial_yaw', default_value=''),
        scan_filter,
        ekf,
        slam_toolbox,
        OpaqueFunction(function=_amcl, condition=UnlessCondition(slam)),
    ])
