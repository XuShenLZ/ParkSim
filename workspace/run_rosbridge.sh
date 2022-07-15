#!/bin/bash

colcon build --packages-select rosbridge
ros2 run rosbridge bridge_node