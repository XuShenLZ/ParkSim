import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from interfaces.msg import Num, VehicleStateMsg


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(VehicleStateMsg, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        # msg = Num()
        # msg.num = self.i
        state_msg = VehicleStateMsg()
        self.populate_state_msg(state_msg)
        self.publisher_.publish(state_msg)
        self.get_logger() \
            .info(f'ROS sent vehicle state x: {state_msg.x.x} y: {state_msg.x.y} z: {state_msg.x.z}')
        # self.get_logger().info('Publishing: "%s"' % msg.num)
        self.i += 1
    
    def populate_state_msg(self, msg):
        # PositionMsg: x
        msg.x.x = 0.0
        msg.x.y = 1.0
        msg.x.z = 2.0
        # OrientationQuaternionMsg: q
        msg.q.qr = 0.0
        msg.q.qi = 1.0
        msg.q.qj = 2.0
        msg.q.qk = 3.0
        # BodyAngularVelocityMsg: w
        msg.w.w_phi = 0.0
        msg.w.w_theta = 1.0
        msg.w.w_psi = 2.0
        # BodyLinearVelocityMsg: v
        msg.v.v_long = 0.0
        msg.v.v_tran = 1.0
        msg.v.v_n = 2.0


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()