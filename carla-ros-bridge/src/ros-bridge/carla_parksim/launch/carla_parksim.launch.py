import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

from parksim.base_node import read_yaml_file


parksim_dir = get_package_share_directory('parksim')
config_dir = os.path.join(parksim_dir, 'config')

global_params_file = os.path.join(config_dir, 'global_params.yaml')
global_params = read_yaml_file(global_params_file)

def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='host',
            default_value='localhost'
        ),
        launch.actions.DeclareLaunchArgument(
            name='port',
            default_value='2000'
        ),
        launch.actions.DeclareLaunchArgument(
            name='timeout',
            default_value='10'
        ),
        launch.actions.DeclareLaunchArgument(
            name='role_name',
            default_value='car_ego'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_name',
            default_value='camera_1'
        ),
        launch.actions.DeclareLaunchArgument(
            name='vehicle_filter',
            default_value='vehicle.*'
        ),
        launch.actions.DeclareLaunchArgument(
            name='spawn_point',
            default_value='None'
        ),
        launch.actions.DeclareLaunchArgument(
            name='town',
            default_value='gomentum_export'
        ),
        launch.actions.DeclareLaunchArgument(
            name='passive',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='synchronous_mode_wait_for_vehicle_control_command',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='fixed_delta_seconds',
            default_value='0.05'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_ros_bridge'), 'carla_ros_bridge.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'town': launch.substitutions.LaunchConfiguration('town'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'passive': launch.substitutions.LaunchConfiguration('passive'),
                'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command'),
                'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds')
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_spawn_objects'), 'carla_example_ego_vehicle.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'vehicle_filter': launch.substitutions.LaunchConfiguration('vehicle_filter'),
                'role_name': launch.substitutions.LaunchConfiguration('role_name'),
                'spawn_point': launch.substitutions.LaunchConfiguration('spawn_point')
            }.items()
        ),
        # launch.actions.IncludeLaunchDescription(
        #     launch.launch_description_sources.PythonLaunchDescriptionSource(
        #         os.path.join(get_package_share_directory(
        #             'carla_manual_control'), 'carla_manual_control.launch.py')
        #     ),
        #     launch_arguments={
        #         'role_name': launch.substitutions.LaunchConfiguration('role_name')
        #     }.items()
        # )
        launch_ros.actions.Node(
            package='carla_parksim',
            executable='carla_parksim.py',
            name='carla_parksim',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    'host': launch.substitutions.LaunchConfiguration('host'),
                    'port': launch.substitutions.LaunchConfiguration('port'),
                    'camera_name': launch.substitutions.LaunchConfiguration('camera_name')
                }
            ]
        ),
        launch_ros.actions.Node(
            package='parksim',
            executable='translator_node.py',
            name='translator_node',
            output='screen',
            emulate_tty=True
        ),
        launch_ros.actions.Node(
            package='parksim',
            executable='simulator_node.py',
            name='simulator',
            parameters=[os.path.join(config_dir, 'simulator.yaml')]+global_params,
            output='screen',
            emulate_tty=True
        )
            
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
