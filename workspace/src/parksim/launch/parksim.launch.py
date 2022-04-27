from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from parksim.base_node import read_yaml_file

import os

parksim_dir = get_package_share_directory('parksim')
config_dir = os.path.join(parksim_dir, 'config')

global_params_file = os.path.join(config_dir, 'global_params.yaml')
global_params = read_yaml_file(global_params_file)

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='parksim',
            executable='visualizer_node.py',
            name='visualizer',
            parameters=[os.path.join(config_dir, 'visualization.yaml')]+global_params,
            output='screen'
        ),

        # Delay simulator so that the visualization is ready
        TimerAction(period=3.0,
            actions=[
                Node(
                    package='parksim',
                    executable='simulator_node.py',
                    name='simulator',
                    parameters=[os.path.join(config_dir, 'simulator.yaml')]+global_params,
                    output='screen',
                    emulate_tty=True
                )
            ])
    ])