from launch import LaunchDescription
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

        Node(
            package='parksim',
            namespace='vehicle_1',
            executable='test_vehicle_node.py',
            name='vehicle',
            parameters=[os.path.join(config_dir, 'vehicle.yaml')]+global_params,
            output='screen'
        ),

        Node(
            package='parksim',
            namespace='vehicle_2',
            executable='test_vehicle_node.py',
            name='vehicle',
            parameters=[os.path.join(config_dir, 'vehicle.yaml')]+global_params,
            output='screen'
        )
    ])