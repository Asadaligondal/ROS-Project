#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ContactsState
from std_srvs.srv import Empty

class BumperResetNode(Node):
    def __init__(self):
        super().__init__('bumper_reset_node')
        # ROS2 setup
        self.bumper_sub = self.create_subscription(
            ContactsState, '/bumper_states', self.bumper_callback, 10)  # Changed to /bumper_states
        self.reset_client = self.create_client(Empty, '/reset_world')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_world service...')
        self.get_logger().info('Bumper Reset Node started - Listening to /bumper_states')

    def reset_world(self):
        """Call /reset_world service to reset the Turtlebot."""
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        future.add_done_callback(self.reset_callback)

    def reset_callback(self, future):
        try:
            future.result()
            self.get_logger().info('World reset successful')
        except Exception as e:
            self.get_logger().error(f'World reset failed: {e}')

    def bumper_callback(self, msg):
        """Check for collisions and reset on detection."""
        self.get_logger().info(f'Bumper message received - States: {len(msg.states)}')
        if len(msg.states) > 0:
            self.get_logger().info('Collision detected!')
            self.reset_world()

def main():
    rclpy.init()
    node = BumperResetNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()