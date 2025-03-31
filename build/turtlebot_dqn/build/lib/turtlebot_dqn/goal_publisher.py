#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Pose2D

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        # Service client for /get_entity_state
        self.client = self.create_client(GetEntityState, '/get_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_entity_state service...')
        
        # Publisher for goal coordinates
        self.publisher = self.create_publisher(Pose2D, '/goal_pose', 10)
        
        # Timer to update goal position periodically
        self.timer = self.create_timer(1.0, self.timer_callback)  # Update every 1 second
        self.entity_name = 'goal_cube'  # Name of the object in Gazebo

    def timer_callback(self):
        # Prepare service request
        request = GetEntityState.Request()
        request.name = self.entity_name
        
        # Call service
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if response.success:
                # Extract position from response
                x = response.state.pose.position.x
                y = response.state.pose.position.y
                # Publish as Pose2D
                msg = Pose2D()
                msg.x = x
                msg.y = y
                msg.theta = 0.0  # Orientation not needed for now
                self.publisher.publish(msg)
                self.get_logger().info(f'Published goal pose: x={x}, y={y}')
            else:
                self.get_logger().warn(f'Failed to get state for {self.entity_name}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main():
    rclpy.init()
    node = GoalPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()