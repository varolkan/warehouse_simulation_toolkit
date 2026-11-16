r"""The AGV in the Gazebo Harmonic warehouse, with the whole stack on top.

    ros2 launch agv_gazebo sim.launch.py
    ros2 action send_goal --feedback /transport agv_interfaces/action/Transport \
        "{pickup: inbound, dropoff: packing}"

What starts, in order:
  gz sim             the warehouse world (generated from the site file)
  create tote        the load, on the stand at `load_at`
  create agv         the vehicle, docked at `start`, once the tote is there
  parameter_bridge   clock, drive, sensors, ground truth, the load joint, set_pose
  sim_agv            the load handler and the battery Gazebo does not simulate
  bringup            robot_state_publisher, EKF, AMCL on the site's map, Nav2,
                     safety_node, mission_node (agv_robot/launch/bringup.launch.py)

`gui:=true` opens Gazebo's window. Headless (the default) still renders: the LiDAR is a
GPU sensor, so a machine without a display needs EGL or a virtual X server (Xvfb).
`mapping:=true` maps the site instead (slam_toolbox, no load, no mission).
"""

import os

from agv_core.sites import Site
from agv_gazebo.world import load_pose_on_stand
from agv_robot.paths import site_file
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable, DeclareLaunchArgument, EmitEvent, ExecuteProcess,
    IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _simulation(context):
    share = get_package_share_directory('agv_gazebo')
    robot_share = get_package_share_directory('agv_robot')
    site_name = LaunchConfiguration('site').perform(context)
    start = LaunchConfiguration('start').perform(context)
    mapping = LaunchConfiguration('mapping').perform(context) == 'true'
    load_at = '' if mapping else LaunchConfiguration('load_at').perform(context)
    site = Site.load(site_file(site_name))
    if not site.has_station(start):
        raise RuntimeError(f"start '{start}' is not a station of '{site.name}'")
    if load_at and not (site.has_station(load_at) and site.station(load_at).is_transfer):
        raise RuntimeError(f"load_at '{load_at}' is not a transfer station of '{site.name}'")
    pose = site.station(start).pose

    command = ['gz', 'sim', '-r', '-v', '2', os.path.join(share, 'worlds', 'warehouse.sdf')]
    if LaunchConfiguration('gui').perform(context) != 'true':
        command.insert(3, '-s')
    gazebo = ExecuteProcess(cmd=command, output='screen', name='gazebo')

    actions = [
        gazebo,
        RegisterEventHandler(OnProcessExit(target_action=gazebo,
                                           on_exit=[EmitEvent(event=Shutdown())])),
        Node(
            package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge',
            output='screen',
            parameters=[{'config_file': os.path.join(share, 'config', 'bridge.yaml')}],
        ),
        Node(
            package='agv_gazebo', executable='sim_agv_node', name='sim_agv', output='screen',
            parameters=[{
                'use_sim_time': True, 'site': site_name, 'load_at': load_at,
                'battery': float(LaunchConfiguration('battery').perform(context)),
            }],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(robot_share, 'launch',
                                                       'bringup.launch.py')),
            launch_arguments={
                'site': site_name, 'use_sim_time': 'true', 'sim': 'true',
                'load_joint': 'true' if load_at else 'false',
                'mapping': 'true' if mapping else 'false',
                'initial_x': str(pose.x), 'initial_y': str(pose.y),
                'initial_yaw': str(pose.yaw),
                'log_level': LaunchConfiguration('log_level').perform(context),
            }.items(),
        ),
    ]

    spawn_vehicle = Node(
        package='ros_gz_sim', executable='create', name='spawn_agv', output='screen',
        arguments=['-world', 'warehouse', '-name', 'agv', '-topic', 'robot_description',
                   '-x', str(pose.x), '-y', str(pose.y), '-z', '0.01', '-Y', str(pose.yaw)],
    )
    if load_at:
        x, y, z, yaw = load_pose_on_stand(site.station(load_at))
        spawn_tote = Node(
            package='ros_gz_sim', executable='create', name='spawn_tote', output='screen',
            arguments=['-world', 'warehouse', '-name', 'tote',
                       '-file', os.path.join(share, 'models', 'tote', 'model.sdf'),
                       '-x', str(x), '-y', str(y), '-z', str(z + 0.005), '-Y', str(yaw)],
        )
        # The tote first: the vehicle's load joint looks for it from the moment it exists.
        actions += [spawn_tote, RegisterEventHandler(
            OnProcessExit(target_action=spawn_tote, on_exit=[spawn_vehicle]))]
    else:
        actions.append(spawn_vehicle)
    return actions


def generate_launch_description():
    share = get_package_share_directory('agv_gazebo')
    return LaunchDescription([
        DeclareLaunchArgument('site', default_value='warehouse'),
        DeclareLaunchArgument('start', default_value='charger',
                              description='Station the vehicle starts docked at.'),
        DeclareLaunchArgument('load_at', default_value='inbound',
                              description="Transfer station the tote waits at; '' for none."),
        DeclareLaunchArgument('battery', default_value='0.8'),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('mapping', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(share, 'models')),
        OpaqueFunction(function=_simulation),
    ])
