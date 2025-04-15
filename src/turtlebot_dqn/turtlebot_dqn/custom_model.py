#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState
import random
import os
import math
import time
import numpy as np

class RandomActionNode(Node):
    def __init__(self):
        super().__init__('random_action_node')
        # ROS2 setup
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.bumper_sub = self.create_subscription(
            ContactsState, '/bumper_states', self.bumper_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Setup entity state client for position reset
        self.set_entity_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # Actions setup
        self.actions = [
            (0.1, 0.0),  # Forward
            (0.0, 0.5),  # Left
            (0.0, -0.5), # Right
            (0.0, 0.0)   # Stop
        ]
        
        # Episode tracking
        self.max_steps = 300
        self.episode = 0
        self.step = 0
        
        # Safety flags
        self.is_resetting = False
        self.is_collision = False
        
        random.seed()
        self.stop_robot()
        
        # Create a timer for regular updates (10 Hz)
        self.update_timer = self.create_timer(0.1, self.update_callback)

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            self.get_logger().debug('Robot stopped')
        except Exception as e:
            self.get_logger().error(f'Error stopping robot: {e}')

    def choose_random_action(self):
        """Choose a random action from the available actions."""
        action_idx = random.randint(0, len(self.actions) - 1)
        return action_idx

    def reset_robot_position(self):
        """Reset the robot to a random starting position"""
        if self.is_resetting:
            self.get_logger().warn('Already resetting, ignoring additional reset request')
            return
            
        self.is_resetting = True
        
        try:
            # Generate random position within arena bounds
            x = random.uniform(-2.0, 2.0)
            y = random.uniform(-2.0, 2.0)
            z = 0.0
            yaw = random.uniform(-math.pi, math.pi)
            
            request = SetEntityState.Request()
            request.state = EntityState()
            request.state.name = 'burger'  # Your robot model name in Gazebo
            
            # Set position
            request.state.pose = Pose()
            request.state.pose.position.x = x
            request.state.pose.position.y = y
            request.state.pose.position.z = z
            
            # Set orientation (yaw only)
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            request.state.pose.orientation.w = cy
            request.state.pose.orientation.x = 0.0
            request.state.pose.orientation.y = 0.0
            request.state.pose.orientation.z = sy
            
            # Zero velocity
            request.state.twist = Twist()
            
            # Call service
            self.get_logger().info(f'Requesting position reset for episode {self.episode}')
            future = self.set_entity_client.call_async(request)
            future.add_done_callback(self.position_reset_callback)
        except Exception as e:
            self.get_logger().error(f'Error in reset_robot_position: {e}')
            self.is_resetting = False
    
    def position_reset_callback(self, future):
        """Callback after position reset"""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error('Failed to reset robot position')
            else:
                self.stop_robot()
                self.get_logger().info(f'Robot position reset for episode {self.episode}')
            
            # Wait a bit for physics to stabilize
            time.sleep(0.5)
            
        except Exception as e:
            self.get_logger().error(f'Robot position reset failed: {e}')
        finally:
            self.is_resetting = False
            self.is_collision = False

    def is_valid_scan(self, ranges):
        """Check if lidar scan data is valid."""
        if not ranges:
            return False
            
        # Check for NaN or Inf values
        for r in ranges:
            if not math.isfinite(r):
                return False
                
        return True

    def update_callback(self):
        """Regular timer callback for movement."""
        # Skip if we're in reset state or collision state
        if self.is_resetting or self.is_collision:
            return
            
        # Take random action occasionally
        if random.random() < 0.1:  # 10% chance to change action
            action_idx = self.choose_random_action()
            linear_x, angular_z = self.actions[action_idx]
            
            try:
                # Publish velocity
                twist = Twist()
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                self.vel_pub.publish(twist)
            except Exception as e:
                self.get_logger().error(f'Error publishing velocity: {e}')
                self.stop_robot()

    def lidar_callback(self, msg):
        """Process lidar data and take action."""
        # Skip if we're in reset or collision state
        if self.is_resetting or self.is_collision:
            return
            
        try:
            # Check for valid ranges
            if not self.is_valid_scan(msg.ranges):
                self.get_logger().warn('Received invalid lidar data, skipping this update')
                return
                
            # Choose a random action if we're not using the timer for movement
            # action_idx = self.choose_random_action()
            # linear_x, angular_z = self.actions[action_idx]
            
            # # Publish velocity
            # twist = Twist()
            # twist.linear.x = linear_x
            # twist.angular.z = angular_z
            # self.vel_pub.publish(twist)
            
            self.step += 1
            
            # Print episode progress every 50 steps
            if self.step % 50 == 0:
                self.get_logger().info(f'Episode {self.episode}: Step {self.step}')
            
            if self.step >= self.max_steps:
                self.get_logger().info(f'Episode {self.episode}: Max steps reached')
                self.reset_episode()
                
        except Exception as e:
            self.get_logger().error(f'Error in lidar_callback: {e}')
            self.stop_robot()

    def reset_episode(self):
        """Reset the episode counter and robot position."""
        if self.is_resetting:
            return
            
        try:
            self.get_logger().info(f'Episode {self.episode} completed - Steps: {self.step}')
            
            # Stop the robot first
            self.stop_robot()
            
            # Wait for robot to fully stop
            time.sleep(0.5)
            
            # Update episode counter
            self.episode += 1
            self.step = 0
            
            # Reset robot position
            self.reset_robot_position()
        except Exception as e:
            self.get_logger().error(f'Error in reset_episode: {e}')

    def bumper_callback(self, msg):
        """Handle collisions."""
        # Skip if already handling a collision or reset
        if self.is_collision or self.is_resetting:
            return
            
        try:
            if len(msg.states) > 0:
                self.is_collision = True
                self.get_logger().info(f'Episode {self.episode}: Collision detected at step {self.step}')
                
                # First stop the robot
                self.stop_robot()
                
                # Add a small delay to allow robot to fully stop
                time.sleep(0.5)
                
                # Increment episode counter
                self.episode += 1
                self.step = 0
                
                # Reset robot position to start new episode
                self.reset_robot_position()
        except Exception as e:
            self.get_logger().error(f'Error in bumper_callback: {e}')
            self.stop_robot()

def main():
    rclpy.init()
    node = RandomActionNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Stop the robot before shutting down
        node.stop_robot()
        rclpy.shutdown()

if __name__ == '__main__':
    main()