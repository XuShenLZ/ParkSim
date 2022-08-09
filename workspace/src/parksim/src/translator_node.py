#!/usr/bin/env python3

import json
import math
from typing import Dict
from collections import defaultdict

import rclpy
import re

from pathlib import Path

from dlp.dataset import Dataset

from transforms3d.euler import euler2quat


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
        self.get_logger().info('Initializing Carla Translator Node')
        namespace = self.get_namespace()

        param_template = VisualizerNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.state_subs = {}
        self.info_subs = {}

        self.twist_pubs = {}
        self.pose_pubs = {}
        with open('objects.json') as handle:
            self.objects_json = json.loads(handle.read())
        self.setup_pubs()

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
            self.twist_pubs[vehicle_id].publish(twist_msg)

        return callback

    def vehicle_info_cb(self, vehicle_id):
        def callback(msg):
            info = VehicleInfo()
            self.unpack_msg(msg, info)
            self.infos[vehicle_id] = info

        return callback

    # create publishers for all vehicles for set velocity and position
    def setup_pubs(self):
        vehicles = list(filter(lambda obj: obj['type'].split('.')[0] == 'vehicle', self.objects_json))
        for vehicle in vehicles:
            vehicle_id = vehicle['id'][8:]
            self.twist_pubs[vehicle_id] = self.create_publisher(Twist, f'/carla/vehicle_{vehicle_id}/control/set_target_velocity', 10)
            self.pose_pubs[vehicle_id] = self.create_publisher(Pose, f'/carla/vehicle_{vehicle_id}/contro/set_transform', 10)

    # move vehicle to entrance of parking lot
    def spawn_vehicle(self, vehicle_id):
        self.publish_pose(vehicle_id, 298.0, 20.0, 29.0, 0.0, 0.0, 154.0)

    # move vehicle back to its off-screen starting location as specified in objects2.json
    def remove_vehicle(self, vehicle_id):
        objects = self.objects_json
        vehicle = list(filter(lambda obj: obj.id == f'vehicle_{vehicle_id}', objects['objects']))[0]

        x = vehicle['spawn_objects']['x']
        y = vehicle['spawn_objects']['y']
        z = vehicle['spawn_objects']['z']
        roll = vehicle['spawn_objects']['roll']
        pitch = vehicle['spawn_objects']['pitch']
        yaw = vehicle['spawn_objects']['yaw']

        self.publish_pose(vehicle_id, x, y, z, roll, pitch, yaw)

    # helper method to create and publish a Pose message
    def publish_pose(self, vehicle_id, x, y, z, roll, pitch, yaw):
        pose_msg = Pose()
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = z
        roll = roll
        pitch = pitch
        yaw = yaw
        quat = euler2quat(math.radians(roll), math.radians(pitch), math.radians(yaw))
        pose_msg.orientation.w = quat[0]
        pose_msg.orientation.x = quat[1]
        pose_msg.orientation.y = quat[2]
        pose_msg.orientation.z = quat[3]

        self.pose_pubs[vehicle_id].publish(pose_msg)

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

                    # Spawn vehicle at entrance
                    self.spawn_vehicle(vehicle_id)

                elif vehicle_id in self.state_subs and not publisher:
                    # If we have subscribed to it, but there is no publisher anymore
                    self.destroy_subscription(self.state_subs[vehicle_id])
                    self.state_subs.pop(vehicle_id)
                    self.get_logger().info("Vehicle %d is not publishing anymore. State subscriber is destroyed." % vehicle_id)
                    # We don't delete state storage since we want the vehicle to remain in the visualizer

                    # Remove vehicle off-screen
                    self.remove_vehicle(vehicle_id)

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