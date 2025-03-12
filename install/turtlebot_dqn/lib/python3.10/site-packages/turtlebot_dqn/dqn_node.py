#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from std_srvs.srv import Empty
import numpy as np
import torch
import torch.nn as nn
import random
import os

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNNode(Node):
    def __init__(self):
        super().__init__('dqn_node')
        # ROS2 setup
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.bumper_sub = self.create_subscription(
            ContactsState, '/bumper_states', self.bumper_callback, 10)  # Changed to /bumper_states
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_client = self.create_client(Empty, '/reset_world')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_world service...')
        
        # DQN setup
        self.state_size = 8
        self.action_size = 4
        self.dqn = DQNNetwork(self.state_size, self.action_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.actions = [
            (0.1, 0.0),  # Forward
            (0.0, 0.5),  # Left
            (0.0, -0.5), # Right
            (0.0, 0.0)   # Stop
        ]
        # Episode tracking
        self.max_episodes = 100
        self.max_steps = 100
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.current_state = None
        random.seed()  # Ensure random actions
        self.get_logger().info('DQN Node started - Listening to /bumper_states')
        self.stop_robot()  # Stop any initial movement

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        self.get_logger().info('Published stop command to /cmd_vel')

    def preprocess_lidar(self, ranges):
        sector_size = len(ranges) // 8
        sectors = []
        for i in range(8):
            start = i * sector_size
            end = (i + 1) * sector_size
            sector = ranges[start:end]
            min_dist = min(sector) if min(sector) < 5.0 else 5.0
            sectors.append(min_dist)
        return np.array(sectors, dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            self.get_logger().info(f'Random action chosen: {action}')
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float()
                q_values = self.dqn(state_tensor)
                action = q_values.argmax().item()
                self.get_logger().info(f'Q-value action chosen: {action}')
                return action

    def reset_episode(self):
        self.get_logger().info(f'Episode {self.episode} completed - Steps: {self.step}, Total Reward: {self.total_reward}')
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        
        # Reset world
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        future.add_done_callback(self.reset_callback)
        
        if self.episode >= self.max_episodes:
            self.save_model()
            self.get_logger().info('Training completed, shutting down...')
            rclpy.shutdown()

    def reset_callback(self, future):
        try:
            future.result()
            self.get_logger().info('World reset successful')
            self.stop_robot()  # Stop movement after reset
        except Exception as e:
            self.get_logger().error(f'World reset failed: {e}')

    def save_model(self):
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        torch.save(self.dqn.state_dict(), model_path)
        self.get_logger().info(f'Model saved to {model_path}')

    def lidar_callback(self, msg):
        if self.episode >= self.max_episodes:
            return
        
        self.current_state = self.preprocess_lidar(msg.ranges)
        action_idx = self.choose_action(self.current_state)
        linear_x, angular_z = self.actions[action_idx]
        
        # Publish velocity
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)
        
        # Reward: +1 per step
        self.total_reward += 1.0
        self.step += 1
        
        if self.step >= self.max_steps:
            self.get_logger().info(f'Max steps reached at episode {self.episode}')
            self.reset_episode()

    def bumper_callback(self, msg):
        if self.episode >= self.max_episodes:
            return
        
        self.get_logger().info(f'Bumper message received - States: {len(msg.states)}')
        if len(msg.states) > 0:
            self.get_logger().info(f'Collision detected at episode {self.episode}, step {self.step}')
            self.total_reward -= 100.0
            self.reset_episode()

def main():
    rclpy.init()
    node = DQNNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()