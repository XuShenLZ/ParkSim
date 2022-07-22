#!/bin/bash

colcon build --symlink-install --packages-select rosbridge
ros2 run rosbridge bridge_node