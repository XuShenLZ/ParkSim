#!/bin/bash

colcon build --packages-select carla_parksim
ros2 launch carla_parksim carla_parksim.launch.py