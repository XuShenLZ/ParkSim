#!/usr/bin/env python3

from dataclasses import dataclass, field
import re

import rclpy

import numpy as np

from std_msgs.msg import Float64, String, Int16, Bool, Int16MultiArray
from parksim.msg import VehicleStateMsg, VehicleInfoMsg
from parksim.srv import OccupancySrv
from parksim.pytypes import PythonMsg, VehiclePrediction, VehicleState, NodeParamTemplate
from parksim.vehicle_types import VehicleBody, VehicleConfig
from parksim.base_node import MPClabNode
from parksim.agents.rule_based_stanley_vehicle import RuleBasedStanleyVehicle

@dataclass
class VehicleInfo(PythonMsg):
    ref_pose: VehiclePrediction = field(default=None)
    ref_v: float = field(default=0)
    target_idx: int = field(default=None)
    priority: int = field(default=None)
    parking_flag: str = field(default=None)
    parking_progress: str = field(default=None)
    is_braking: bool = field(default=None)
    parking_start_time: float = field(default=None)
    waiting_for: int = field(default=None)


class VehicleNodeParams(NodeParamTemplate):
    """
    template that stores all parameters needed for the node as well as default values
    """
    def __init__(self):
        self.timer_period = 0.1

        self.random_seed =0

        self.spots_data_path = '/ParkSim/data/spots_data.pickle'
        self.offline_maneuver_path = '/ParkSim/data/parking_maneuvers.pickle'
        self.waypoints_graph_path = '/ParkSim/data/waypoints_graph.pickle'
        self.intent_model_path = '/ParkSim/data/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'

class VehicleNode(MPClabNode):
    """
    Node for rule based stanley vehicle
    """
    def __init__(self):
        """
        init
        """
        super().__init__('vehicle')
        self.get_logger().info('Initializing Vehicle...')
        namespace = self.get_namespace()

        # ======== Parameters
        param_template = VehicleNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.declare_parameter('vehicle_id', 0)
        self.vehicle_id = self.get_parameter('vehicle_id').get_parameter_value().integer_value

        self.declare_parameter('spot_index', 0)
        self.spot_index = self.get_parameter('spot_index').get_parameter_value().integer_value

        self.get_logger().info("Spot Index: " + str(self.spot_index))

        # ======== Publishers, Subscribers, Services
        self.state_pub = self.create_publisher(VehicleStateMsg, 'state', 10)
        self.info_pub = self.create_publisher(VehicleInfoMsg, 'info', 10)

        self.state_subs = {}
        self.info_subs = {}
        self.occupancy_sub = self.create_subscription(Int16MultiArray, '/occupancy', self.occupancy_cb, 10)

        self.occupancy_cli = self.create_client(OccupancySrv, '/occupancy')
        while not self.occupancy_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')


        self.vehicle = RuleBasedStanleyVehicle(
            vehicle_id=self.vehicle_id, 
            vehicle_body=VehicleBody(), 
            vehicle_config=VehicleConfig(), 
            inst_centric_generator=None, 
            intent_predictor=None
            )
        
        self.vehicle.load_parking_spaces(spots_data_path=self.spots_data_path)
        self.vehicle.load_graph(waypoints_graph_path=self.waypoints_graph_path)
        self.vehicle.load_maneuver(offline_maneuver_path=self.offline_maneuver_path)
        self.vehicle.set_anchor(going_to_anchor=self.spot_index>0, spot_index=self.spot_index, should_overshoot=False)

        self.vehicle.set_method_to_change_central_occupancy(self.change_occupancy)

        self.vehicle.start_vehicle()

    def vehicle_state_cb(self, vehicle_id):
        def callback(msg):
            state = VehicleState()
            self.unpack_msg(msg, state)
            self.vehicle.other_state[vehicle_id] = state

        return callback

    def vehicle_info_cb(self, vehicle_id):
        def callback(msg):
            info = VehicleInfo()
            self.unpack_msg(msg, info)

            self.vehicle.other_ref_pose[vehicle_id] = info.ref_pose
            self.vehicle.other_ref_v[vehicle_id] = info.ref_v
            self.vehicle.other_target_idx[vehicle_id] = info.target_idx
            self.vehicle.other_priority[vehicle_id] = info.priority
            self.vehicle.other_parking_flag[vehicle_id] = info.parking_flag
            self.vehicle.other_parking_progress[vehicle_id] = info.parking_progress
            self.vehicle.other_is_braking[vehicle_id] = info.is_braking
            self.vehicle.other_parking_start_time[vehicle_id] = info.parking_start_time
            self.vehicle.other_waiting_for[vehicle_id] = info.waiting_for

        return callback

    def occupancy_cb(self, msg):
        self.vehicle.occupancy = msg.data

    def change_occupancy(self, idx, new_value):
        def response_cb(future):
            res = future.result()
            if res.status:
                self.get_logger().info("Service request from vehicle %d to change occupancy is successful" % self.vehicle_id)

        req = OccupancySrv.Request()
        req.vehicle_id = self.vehicle_id
        req.idx = int(idx)
        req.new_value = int(new_value)
        
        future = self.occupancy_cli.call_async(req)
        future.add_done_callback(response_cb)


    def update_subs(self):
        topic_list_types = self.get_topic_names_and_types()

        active_ids = set()

        for topic_name, _ in topic_list_types:
            state_name_pattern = re.match("/vehicle_([1-9][0-9]*)/state", topic_name)
            info_name_pattern = re.match("/vehicle_([1-9][0-9]*)/info", topic_name)

            if state_name_pattern:
                vehicle_id = int(state_name_pattern.group(1))
                if vehicle_id == self.vehicle_id:
                    continue

                if vehicle_id not in self.state_subs:
                    self.state_subs[vehicle_id] = self.create_subscription(VehicleStateMsg, topic_name, self.vehicle_state_cb(vehicle_id), 10)

                active_ids.add(vehicle_id)

            elif info_name_pattern:
                vehicle_id = int(info_name_pattern.group(1))
                if vehicle_id == self.vehicle_id:
                    continue

                if vehicle_id not in self.info_subs:
                    self.info_subs[vehicle_id] = self.create_subscription(VehicleInfoMsg, topic_name, self.vehicle_info_cb(vehicle_id), 10)

                active_ids.add(vehicle_id)

            else:
                continue

        self.vehicle.other_vehicles = active_ids

    def timer_callback(self):
        if self.vehicle.is_all_done():
            self.destroy_node()

        self.update_subs()
        self.vehicle.solve()

        state_msg = VehicleStateMsg()
        self.populate_msg(state_msg, self.vehicle.state)
        self.state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)

    vehicle = VehicleNode()

    rclpy.spin(vehicle)

    vehicle.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()