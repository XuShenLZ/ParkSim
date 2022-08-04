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

from geometry_msgs.msg import Pose, Twist

class VisualizerNodeParams(NodeParamTemplate):

    def __init__(self):
        self.timer_period = 0.05

class TranslatorNode(MPClabNode):
    """
    Node class for visualizing everything
    """
    def __init__(self):
        super().__init__('carla_translator')
        self.get_logger().info('Initializing Carla Trnaslator Node')
        namespace = self.get_namespace()

        param_template = VisualizerNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.state_subs = {}
        self.info_subs = {}

        self.state_pubs = {}

        self.states: Dict[int, VehicleState] = defaultdict(lambda: None)
        self.infos: Dict[int, VehicleInfo] = defaultdict(lambda: None)


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

            twist_msg = Twist()
            twist_msg.linear.x = state.v.v_long
            twist_msg.linear.y = state.v.v_tran
            twist_msg.linear.z = state.v.v_n
            twist_msg.angular.x = state.w.w_phi
            twist_msg.angular.y = state.w.w_theta
            twist_msg.angular.z = state.w.w_psi
            self.state_pubs[vehicle_id].publish(twist_msg)

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


                    # Create publisher to the carla ros-bridge topic for vehicle control
                    self.state_pubs[vehicle_id] = self.create_publisher(Twist, f'/carla/vehicle_{vehicle_id}/control/set_target_velocity', 10)


                    # If there is publisher, but we haven't subscribed to it
                    self.state_subs[vehicle_id] = self.create_subscription(VehicleStateMsg, topic_name, self.vehicle_state_cb(vehicle_id), 10)
                    self.get_logger().info("State subscriber to vehicle %d is built." % vehicle_id)

                elif vehicle_id in self.state_subs and not publisher:
                    # If we have subscribed to it, but there is no publisher anymore
                    self.destroy_subscription(self.state_subs[vehicle_id])
                    self.state_subs.pop(vehicle_id)
                    self.get_logger().info("Vehicle %d is not publishing anymore. State subscriber is destroyed." % vehicle_id)
                    # We don't delete state storage since we want the vehicle to remain in the visualizer

                    # Delete the corresponding carla ros-bridge publisher
                    self.destroy_publisher(self.state_pubs[vehicle_id])
                    self.state_pubs.pop(vehicle_id)

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

        sim_status_msg = Bool()
        sim_status_msg.data = True
        self.sim_status_pub.publish(sim_status_msg)

def main(args=None):
    rclpy.init(args=args)

    translator = TranslatorNode()

    try:
        rclpy.spin(translator)
    except KeyboardInterrupt:
        print('Translator is terminated')
    finally:
        translator.destroy_node()
        print('Translator stopped cleanly')

        rclpy.shutdown()

if __name__ == "__main__":
    main()