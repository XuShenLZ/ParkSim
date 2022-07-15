#!/bin/bash

colcon build --packages-select ros
ros2 run ros publisher_node