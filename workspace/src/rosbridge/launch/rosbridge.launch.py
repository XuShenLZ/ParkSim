from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rosbridge',
            executable='hello_world_node',
            name='hello_world'
        )
    ])