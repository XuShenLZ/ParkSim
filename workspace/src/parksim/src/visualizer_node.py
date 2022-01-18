#!/usr/bin/env python3

from typing import Dict

import rclpy
import re

from pathlib import Path

import dearpygui.dearpygui as dpg

from dlp.dataset import Dataset

from parksim.msg import VehicleStateMsg
from parksim.pytypes import VehicleState, NodeParamTemplate
from parksim.vehicle_types import VehicleBody
from parksim.base_node import MPClabNode

from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

class VisualizerNodeParams(NodeParamTemplate):
    """
    template that stores all parameters needed for the node as well as default values
    """
    def __init__(self):
        self.dlp_path = '/dlp-dataset/data/DJI_0012'
        self.timer_period = 0.05

class VisualizerNode(MPClabNode):
    """
    Node class for visualizing everything
    """
    def __init__(self):
        super().__init__('visualizer')
        self.get_logger().info('Initializing Visualization Node')
        namespace = self.get_namespace()

        param_template = VisualizerNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.subs = {}
        self.states: Dict[int, VehicleState] = {}

        # Load dataset
        ds = Dataset()
        home_path = str(Path.home())
        ds.load(home_path + self.dlp_path)

        # Load Vehicle Body
        vehicle_body = VehicleBody()

        self.vis = RealtimeVisualizer(ds, vehicle_body)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def vehicle_state_cb(self, vehicle_id, msg):
        """
        subscriber callback to receive vehicle states
        """
        state = VehicleState()
        self.unpack_msg(msg, state)
        self.states[vehicle_id] = state

    def timer_callback(self):
        """
        update the list of subscribers and plot
        """
        topic_list_types = self.get_topic_names_and_types()

        states_list = []

        for topic_name, _ in topic_list_types:
            name_pattern = re.match("/vehicle_([1-9][0-9]*)/state", topic_name)

            if not name_pattern:
                continue
            else:
                vehicle_id = int(name_pattern.group(1))

                if vehicle_id not in self.subs:
                    self.subs[vehicle_id] = self.create_subscription(VehicleStateMsg, topic_name, lambda msg: self.vehicle_state_cb(vehicle_id, msg), 10)
                    self.states[vehicle_id] = VehicleState()
                else:
                    state = self.states[vehicle_id]
                    states_list.append([state.x.x, state.x.y, state.q.to_yaw()])

        if dpg.is_dearpygui_running():
            self.vis.clear_frame()
            self.vis.draw_vehicles(states_list)

            dpg.render_dearpygui_frame()
        
def main(args=None):
    rclpy.init(args=args)

    visualizer = VisualizerNode()

    rclpy.spin(visualizer)

    visualizer.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
