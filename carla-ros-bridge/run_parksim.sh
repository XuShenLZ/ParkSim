#!/bin/bash

colcon build --packages-select carla_parksim carla_ros_bridge carla_spawn_objects
ros2 launch carla_parksim carla_parksim.launch.py