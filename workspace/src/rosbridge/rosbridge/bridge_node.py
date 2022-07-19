import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from interfaces.msg import Num, VehicleStateMsg
from geometry_msgs.msg import Pose, Twist


class Translator(Node):

    def __init__(self):
        super().__init__('translator')
        self.subscription = self.create_subscription(
            VehicleStateMsg,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(Twist, '/carla/ego_vehicle/control/set_target_velocity', 10)

    def listener_callback(self, msg):
        pose_msg = Pose()
        pose_msg.position.x = msg.x.x
        pose_msg.position.y = msg.x.y
        pose_msg.position.z = msg.x.z
        pose_msg.orientation.x = msg.q.qr
        pose_msg.orientation.y = msg.q.qi
        pose_msg.orientation.z = msg.q.qj
        pose_msg.orientation.w = msg.q.qk
        twist_msg = Twist()
        twist_msg.linear.x = msg.v.v_long
        twist_msg.linear.y = msg.v.v_tran
        twist_msg.linear.z = msg.v.v_n
        twist_msg.angular.x = msg.w.w_phi
        twist_msg.angular.y = msg.w.w_theta
        twist_msg.angular.z = msg.w.w_psi
        self.publisher.publish(twist_msg)
        self.get_logger() \
            .info(f'Bridge received vehicle state x: {msg.v.v_long} y: {msg.v.v_tran} z: {msg.v.v_n}')
        

def main(args=None):
    rclpy.init(args=args)

    translator = Translator()

    rclpy.spin(translator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    translator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()