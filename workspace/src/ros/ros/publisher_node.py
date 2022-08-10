import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from interfaces.msg import Num, VehicleStateMsg


class VehiclePublisher(Node):

    def __init__(self):
        super().__init__('vehicle_publisher')
        self.publisher_ = self.create_publisher(VehicleStateMsg, 'topic', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        state_msg = VehicleStateMsg()
        self.populate_state_msg(state_msg)
        self.publisher_.publish(state_msg)
        # self.get_logger() \
        #     .info(f'ROS sent vehicle state x: {state_msg.v.v_long} y: {state_msg.v.v_tran} z: {state_msg.v.v_n}')
        self.i += 1
    
    def populate_state_msg(self, msg):
        if self.i % 2 == 0:
            # PositionMsg: x
            msg.x.x = 294.0
            msg.x.y = 17.0
            msg.x.z = 29.0
            # OrientationQuaternionMsg: q
            msg.q.qr = 0.0
            msg.q.qi = 0.0
            msg.q.qj = 0.0
            msg.q.qk = 0.0
        else:
            # PositionMsg: x
            msg.x.x = -84.0
            msg.x.y = 54.0
            msg.x.z = 19.0
            # OrientationQuaternionMsg: q
            msg.q.qr = 0.0
            msg.q.qi = 0.0
            msg.q.qj = 0.0
            msg.q.qk = 0.0
        # BodyAngularVelocityMsg: w
        msg.w.w_phi = 0.0
        msg.w.w_theta = 0.0
        msg.w.w_psi = 0.0
        # BodyLinearVelocityMsg: v
        msg.v.v_long = 0.0              # drive forward slowly
        msg.v.v_tran = 0.0
        msg.v.v_n = 0.0


def main(args=None):
    rclpy.init(args=args)

    vehicle_publisher = VehiclePublisher()

    rclpy.spin(vehicle_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vehicle_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()