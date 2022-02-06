#!/usr/bin/env python3

import rclpy

import numpy as np

from parksim.msg import VehicleStateMsg
from parksim.pytypes import VehicleState, NodeParamTemplate
from parksim.vehicle_types import VehicleBody
from parksim.base_node import MPClabNode

class VehicleNodeParams(NodeParamTemplate):
    """
    template that stores all parameters needed for the node as well as default values
    """
    def __init__(self):
        self.timer_period = 0.05

class VehiclePublisher(MPClabNode):
    """
    A simple vehicle state publisher for testing
    """
    def __init__(self):
        """
        init
        """
        super().__init__('vehicle')
        self.get_logger().info('Initializing Vehicle')
        namespace = self.get_namespace()

        param_template = VehicleNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.pub = self.create_publisher(VehicleStateMsg, 'state', 10)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.i = np.random.rand()

        self.declare_parameter('spot_index', 0)
        self.spot_index = self.get_parameter('spot_index').get_parameter_value().integer_value
        self.get_logger().info(str(self.spot_index))

    def timer_callback(self):
        msg = VehicleStateMsg()

        msg.x.x = 10. + 10*np.sin(self.i)
        msg.x.y = 20. + 10*np.cos(self.i)

        self.i += 0.05

        self.pub.publish(msg)

def main():
    rclpy.init()

    vehicle = VehiclePublisher()

    rclpy.spin(vehicle)

    vehicle.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()