"""Nav2 for the AGV: planner, controller, recoveries, velocity smoother, zone filters.

    ros2 launch agv_robot navigation.launch.py

Needs a map -> odom -> base_footprint transform chain and /scan (localization.launch.py).
Commands leave on /cmd_vel_smoothed; the safety node turns them into /cmd_vel.

`mapping:=true` is for a site that has no map yet: the keep-out and speed-limit masks are
left out (the zones are drawn on the map, so they cannot exist before it does) and the
planner may cross space the map does not know yet.
"""

import os

from agv_robot.paths import KEEPOUT_MASK, PLANNER_KEEPOUT_MASK, site_file, SPEED_MASK
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    share = get_package_share_directory('agv_robot')
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context) == 'true'
    zones = LaunchConfiguration('mapping').perform(context) != 'true'
    params = os.path.join(share, 'config', 'nav2.yaml')
    common = {'use_sim_time': use_sim_time}
    # Costmap filters wait for their masks for ever; without zones they are switched off.
    filters = {} if zones else {
        'keepout_filter.enabled': False, 'speed_filter.enabled': False,
        'GridBased.allow_unknown': True,
    }
    nav_cmd = [('cmd_vel', 'cmd_vel_nav')]

    nodes = [
        Node(
            package='nav2_controller', executable='controller_server', name='controller_server',
            output='screen', parameters=[params, common, filters], remappings=nav_cmd,
        ),
        Node(
            package='nav2_planner', executable='planner_server', name='planner_server',
            output='screen', parameters=[params, common, filters],
        ),
        Node(
            package='nav2_behaviors', executable='behavior_server', name='behavior_server',
            output='screen', parameters=[params, common], remappings=nav_cmd,
        ),
        Node(
            package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
            output='screen',
            parameters=[params, common, {
                'default_nav_to_pose_bt_xml':
                    os.path.join(share, 'behavior_trees', 'navigate_to_pose.xml'),
            }],
        ),
        Node(
            package='nav2_velocity_smoother', executable='velocity_smoother',
            name='velocity_smoother', output='screen', parameters=[params, common],
            remappings=nav_cmd,
        ),
    ]
    managed = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
               'velocity_smoother']

    if zones:
        site = LaunchConfiguration('site').perform(context)
        masks = {'keepout': site_file(site, KEEPOUT_MASK),
                 'keepout_planner': site_file(site, PLANNER_KEEPOUT_MASK),
                 'speed': site_file(site, SPEED_MASK)}
        for kind, mask in masks.items():
            if not os.path.isfile(mask):
                raise RuntimeError(
                    f'{kind} mask {mask} not found. Generate the masks from the site file '
                    '(ros2 run agv_robot generate_masks) or launch with mapping:=true.')
            nodes += [
                Node(
                    package='nav2_map_server', executable='map_server',
                    name=f'{kind}_mask_server', output='screen',
                    parameters=[params, common, {'yaml_filename': mask}],
                ),
                Node(
                    package='nav2_map_server', executable='costmap_filter_info_server',
                    name=f'{kind}_filter_info_server', output='screen',
                    parameters=[params, common],
                ),
            ]
            # Masks first: the costmaps read them as they activate.
            managed = [f'{kind}_mask_server', f'{kind}_filter_info_server'] + managed

    nodes.append(Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[common, {'autostart': True, 'bond_timeout': 10.0, 'node_names': managed}],
    ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'mapping', default_value='false',
            description='The site has no map yet: no zone masks, unknown space may be crossed.'),
        DeclareLaunchArgument(
            'site', default_value='warehouse',
            description='Site whose zones apply: a name under agv_robot/sites, or a directory.'),
        OpaqueFunction(function=_nodes),
    ])
