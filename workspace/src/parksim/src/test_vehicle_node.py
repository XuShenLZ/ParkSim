#!/usr/bin/env python3

import rclpy

from parksim.msg import VehicleStateMsg
from parksim.pytypes import VehicleState
from parksim.vehicle_types import VehicleBody
from parksim.base_node import MPClabNode

class VehiclePublisher(MPClabNode):
    """
    A simple vehicle state publisher for testing
    """
    def __init__(self):
        """
        init
        """
        super().__init__('vehicle')

        self.pub = self.create_publisher(VehicleStateMsg, 'state', 10)

        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = VehicleStateMsg()

        msg.x.x = 10. + self.i
        msg.x.y = 20.

        self.i += 1

        self.pub.publish(msg)

def main():
    rclpy.init()

    vehicle = VehiclePublisher()

    rclpy.spin(vehicle)

    vehicle.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()