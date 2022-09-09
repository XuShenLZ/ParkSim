#!/usr/bin/env python3

from typing import Dict
from collections import defaultdict

import rclpy
import re

from pathlib import Path

from dlp.dataset import Dataset

from std_msgs.msg import Bool, Float32
from parksim.msg import VehicleStateMsg, VehicleInfoMsg
from parksim.pytypes import VehicleState, NodeParamTemplate
from parksim.vehicle_types import VehicleBody, VehicleInfo
from parksim.base_node import MPClabNode

from parksim.visualizer.realtime_visualizer import RealtimeVisualizer

class VisualizerNodeParams(NodeParamTemplate):
    """
    template that stores all parameters needed for the node as well as default values
    """
    def __init__(self):
        self.dlp_path = '/dlp-dataset/data/DJI_0012'
        self.timer_period = 0.05

        self.use_existing_agents = False
        self.dlp_time_offset = -1

        self.driving_color = [0, 255, 0, 255]
        self.parking_color = [255, 128, 0, 255]
        self.braking_color = [255, 0, 0, 255]
        self.alldone_color = [0, 0, 0, 255]

        self.dlp_color = [255, 255, 0, 128]

        self.disp_text_offset = [-2, 2]
        self.disp_text_size = 25

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

        self.state_subs = {}
        self.info_subs = {}
        self.states: Dict[int, VehicleState] = defaultdict(lambda: None)
        self.infos: Dict[int, VehicleInfo] = defaultdict(lambda: None)

        # Load dataset
        ds = Dataset()
        home_path = str(Path.home())
        ds.load(home_path + self.dlp_path)

        # Load Vehicle Body
        vehicle_body = VehicleBody()

        self.vis = RealtimeVisualizer(ds, vehicle_body)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.sim_status_pub = self.create_publisher(Bool, '/sim_status', 10)

        self.sim_time = 0.
        self.sim_time_sub = self.create_subscription(Float32, '/sim_time', self.sim_time_cb, 10)

    def sim_time_cb(self, msg: Float32):
        self.sim_time = msg.data

    def vehicle_state_cb(self, vehicle_id):
        def callback(msg):
            state = VehicleState()
            self.unpack_msg(msg, state)
            self.states[vehicle_id] = state

        return callback

    def vehicle_info_cb(self, vehicle_id):
        def callback(msg):
            info = VehicleInfo()
            self.unpack_msg(msg, info)
            self.infos[vehicle_id] = info

        return callback

    def update_subs(self):
        topic_list_types = self.get_topic_names_and_types()

        for topic_name, _ in topic_list_types:
            state_name_pattern = re.match("/vehicle_([1-9][0-9]*)/state", topic_name)
            info_name_pattern = re.match("/vehicle_([1-9][0-9]*)/info", topic_name)

            if state_name_pattern:
                vehicle_id = int(state_name_pattern.group(1))

                publisher = self.get_publishers_info_by_topic(topic_name=topic_name)

                if vehicle_id not in self.state_subs and publisher:
                    # If there is publisher, but we haven't subscribed to it
                    self.state_subs[vehicle_id] = self.create_subscription(VehicleStateMsg, topic_name, self.vehicle_state_cb(vehicle_id), 10)
                    self.get_logger().info("State subscriber to vehicle %d is built." % vehicle_id)
                elif vehicle_id in self.state_subs and not publisher:
                    # If we have subscribed to it, but there is no publisher anymore
                    self.destroy_subscription(self.state_subs[vehicle_id])
                    self.state_subs.pop(vehicle_id)
                    self.get_logger().info("Vehicle %d is not publishing anymore. State subscriber is destroyed." % vehicle_id)
                    # We don't delete state storage since we want the vehicle to remain in the visualizer

            elif info_name_pattern:
                vehicle_id = int(info_name_pattern.group(1))

                publisher = self.get_publishers_info_by_topic(topic_name=topic_name)

                if vehicle_id not in self.info_subs and publisher:
                    # If there is publisher, but we haven't subscribed to it
                    self.info_subs[vehicle_id] = self.create_subscription(VehicleInfoMsg, topic_name, self.vehicle_info_cb(vehicle_id), 10)
                    self.get_logger().info("Info subscriber to vehicle %d is built." % vehicle_id)
                elif vehicle_id in self.info_subs and not publisher:
                    # If we have subscribed to it, but there is no publisher anymore
                    self.destroy_subscription(self.info_subs[vehicle_id])
                    self.info_subs.pop(vehicle_id)
                    if vehicle_id in self.infos:
                        self.infos.pop(vehicle_id)
                    self.get_logger().info("Vehicle %d is not publishing anymore. Info ubscriber is destroyed." % vehicle_id)

            else:
                continue

    def timer_callback(self):
        """
        update the list of subscribers and plot
        """
        self.update_subs()

        self.vis.clear_frame()
        
        if self.use_existing_agents:
            scene_token = self.vis.dlpvis.dataset.list_scenes()[0]
            agent_token_list = self.vis.dlpvis.dataset.get('scene', scene_token)['agents']
            try:
                frame = self.vis.dlpvis.dataset.get_frame_at_time(
                    scene_token=scene_token, timestamp=max(self.sim_time + self.dlp_time_offset, 0))
                inst_tokens = frame['instances']
                for inst_token in inst_tokens:
                    instance = self.vis.dlpvis.dataset.get('instance', inst_token)
                    agent = self.vis.dlpvis.dataset.get('agent', instance['agent_token'])

                    vehicle_id = agent_token_list.index(instance['agent_token'])
                    if agent['type'] in {'Pedestrian', 'Undefined', 'Bicycle'}:
                        continue
                    state = VehicleState()
                    state.x.x = instance['coords'][0]
                    state.x.y = instance['coords'][1]
                    state.e.psi = instance['heading']

                    self.vis.draw_vehicle(state, fill=self.dlp_color)
                    self.vis.draw_text([state.x.x + self.disp_text_offset[0], state.x.y +
                                    self.disp_text_offset[1]], str(vehicle_id), size=self.disp_text_size)
            except:
                pass

        for vehicle_id in self.states:
            state = self.states[vehicle_id]
            info = self.infos[vehicle_id]

            if not info or info.is_all_done:
                color = self.alldone_color
            elif info.is_braking:
                color = self.braking_color
            elif info.task in ["PARK", "UNPARK"]:
                color = self.parking_color
            else:
                color = self.driving_color

            self.vis.draw_vehicle(state, fill=color)
            if info and info.disp_text:
                self.vis.draw_text([state.x.x + self.disp_text_offset[0], state.x.y + self.disp_text_offset[1]], info.disp_text, size=self.disp_text_size)

        self.vis.render()

        sim_status_msg = Bool()
        sim_status_msg.data = self.vis.is_running()
        self.sim_status_pub.publish(sim_status_msg)
        
def main(args=None):
    rclpy.init(args=args)

    visualizer = VisualizerNode()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print('Visualization is terminated')
    finally:
        visualizer.destroy_node()
        print('Visualization stopped cleanly')

        rclpy.shutdown()

if __name__ == "__main__":
    main()
