#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose, Pose2D
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState, GetEntityState
import numpy as np
import torch
import torch.nn as nn
import random
import os
import math
import matplotlib.pyplot as plt

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

class DQNTestNode(Node):
    def __init__(self):
        super().__init__('dqn_test_node')
        
        # QoS Profiles for better performance
        qos = QoSProfile(depth=10)
        
        # ROS2 setup
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile=qos_profile_sensor_data)
        self.bumper_sub = self.create_subscription(
            ContactsState, '/bumper_states', self.bumper_callback, qos)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        
        # Goal subscription (static goal from external source)
        self.goal_sub = self.create_subscription(
            Pose2D, '/goal_pose', self.goal_callback, qos)
        
        # Setup entity state clients
        self.set_entity_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.get_entity_client = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        
        # Wait for services
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        while not self.get_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/get_entity_state service...')
        
        # DQN setup - Match training model exactly
        self.state_size = 10  # 8 lidar sectors + 2 goal info (distance + angle)
        self.action_size = 4
        self.dqn = DQNNetwork(self.state_size, self.action_size)
        
        # Load pre-trained model
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        if os.path.exists(model_path):
            self.dqn.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.get_logger().info(f'Model loaded from {model_path}')
            self.dqn.eval()  # Set to evaluation mode
        else:
            self.get_logger().error(f'Model file not found at {model_path}')
            rclpy.shutdown()
            return
        
        # Actions - same as training
        self.actions = [
            (0.1, 0.0),   # Forward
            (0.0, 0.5),   # Left
            (0.0, -0.5),  # Right
            (0.0, 0.0)    # Stop
        ]
        
        # Goal tracking variables - match training exactly
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.last_distance_to_goal = None
        
        # Episode tracking
        self.max_episodes = 50
        self.max_steps = 300
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.current_state = None
        
        # Status flags
        self.position_updated = False
        self.goal_received = False
        
        # Data storage for plotting
        self.episode_steps = []
        self.episode_rewards = []
        
        # Initialize
        random.seed(42)
        self.stop_robot()

        # Matplotlib setup
        plt.ion()  # Enable interactive mode for non-blocking plotting
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()  # Dual y-axis
        self.update_plot()  # Initial empty plot

        # ROS2 timer to keep plot responsive
        self.plot_timer = self.create_timer(0.1, self.plot_callback)
        
        # Timer to periodically update robot position
        self.position_timer = self.create_timer(0.1, self.update_robot_position)
        
        self.get_logger().info('DQN test node initialized - waiting for goal...')
    
    def goal_callback(self, msg):
        """Update goal position from external source"""
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.goal_received = True
        self.get_logger().info(f'Goal received: x={self.goal_x:.2f}, y={self.goal_y:.2f}')
        
        # Start testing once goal is received
        if not self.goal_received:
            self.get_logger().info('Starting testing with received goal...')
    
    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def preprocess_lidar(self, ranges):
        """Process lidar data exactly the same as in training"""
        sector_size = len(ranges) // 8
        sectors = []
        for i in range(8):
            start = i * sector_size
            end = (i + 1) * sector_size
            sector = ranges[start:end]
            # Handle inf values and cap at 5.0
            valid_ranges = [r for r in sector if not math.isinf(r) and not math.isnan(r)]
            if valid_ranges:
                min_dist = min(valid_ranges)
                min_dist = min(min_dist, 5.0)
            else:
                min_dist = 5.0
            sectors.append(min_dist)
        return np.array(sectors, dtype=np.float32)

    def update_robot_position(self):
        """Update robot position from Gazebo (async)"""
        if not self.get_entity_client.service_is_ready():
            return
            
        request = GetEntityState.Request()
        request.name = 'burger'
        
        future = self.get_entity_client.call_async(request)
        future.add_done_callback(self.robot_position_callback)

    def robot_position_callback(self, future):
        """Handle robot position response"""
        try:
            response = future.result()
            if response.success:
                self.robot_x = response.state.pose.position.x
                self.robot_y = response.state.pose.position.y
                self.position_updated = True
            else:
                self.get_logger().warn('Failed to get robot position')
        except Exception as e:
            self.get_logger().error(f'Error in robot position callback: {e}')

    def create_state(self, lidar_data):
        """Create state vector that matches training exactly (8 lidar + 2 goal info)"""
        if not self.position_updated or not self.goal_received:
            # Return a default state if position/goal not available
            return np.zeros(self.state_size, dtype=np.float32)
        
        # Calculate distance and angle to goal
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx)
        
        # Normalize goal information exactly like in training
        normalized_distance = min(distance_to_goal / 5.0, 1.0)  # Normalize to [0, 1]
        normalized_angle = angle_to_goal / math.pi  # Normalize to [-1, 1]
        
        # Combine lidar data with goal information
        state = np.concatenate([
            lidar_data,  # 8 elements
            [normalized_distance, normalized_angle]  # 2 elements
        ])
        
        return state.astype(np.float32)

    def choose_action(self, state):
        """Choose action based on the trained model (no exploration)"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Add batch dimension
            q_values = self.dqn(state_tensor)
            action = q_values.argmax().item()
            self.get_logger().info(f'action choosen {action}')
            return action

    def calculate_reward(self, lidar_data):
        """Calculate reward similar to training"""
        # Base reward for staying alive
        reward = 1.0
        
        # Progress toward goal reward
        if self.position_updated and self.goal_received:
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            current_distance = math.sqrt(dx**2 + dy**2)
            
            if self.last_distance_to_goal is not None:
                distance_reward = (self.last_distance_to_goal - current_distance) * 10.0
                reward += distance_reward
            
            self.last_distance_to_goal = current_distance
            
            # Goal reached reward
            if current_distance < 0.5:
                reward += 100.0
                self.get_logger().info(f'Episode {self.episode}: Goal reached!')
                return reward, True  # Episode done
        
        # Obstacle avoidance penalty
        min_distance = min(lidar_data)
        if min_distance < 0.5:
            reward -= (0.5 - min_distance) * 20.0
        
        return reward, False

    def reset_episode(self):
        self.get_logger().info(f'Episode {self.episode} completed - Steps: {self.step}, Total Reward: {self.total_reward:.2f}')
        
        # Store data for plotting
        self.episode_steps.append(self.step)
        self.episode_rewards.append(self.total_reward)
        self.update_plot()
        
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        self.last_distance_to_goal = None
        
        # Reset robot position
        self.reset_robot_position()
        
        if self.episode >= self.max_episodes:
            self.save_results()
            self.get_logger().info('Testing completed, shutting down...')
            rclpy.shutdown()
    
    def reset_robot_position(self):
        """Reset the robot to a random starting position"""
        # Generate random position within arena bounds
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        z = 0.0
        yaw = random.uniform(-math.pi, math.pi)
        
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = 'burger'
        
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
        future = self.set_entity_client.call_async(request)
        future.add_done_callback(self.position_reset_callback)
    
    def position_reset_callback(self, future):
        """Callback after position reset"""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error('Failed to reset robot position')
            else:
                self.stop_robot()
                self.position_updated = False  # Force position update
        except Exception as e:
            self.get_logger().error(f'Robot position reset failed: {e}')

    def save_results(self):
        """Save testing results"""
        # Save final plot
        plot_path = os.path.expanduser('~/turtlebot0/dqn_testing_plot.png')
        self.update_plot()
        self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.get_logger().info(f'Testing plot saved to {plot_path}')
        
        # Save data as CSV
        import csv
        csv_path = os.path.expanduser('~/turtlebot0/dqn_testing_results.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Steps', 'Reward'])
            for i in range(len(self.episode_steps)):
                writer.writerow([i, self.episode_steps[i], self.episode_rewards[i]])
        self.get_logger().info(f'Testing data saved to {csv_path}')
        
        # Print summary statistics
        if self.episode_steps:
            avg_steps = np.mean(self.episode_steps)
            avg_reward = np.mean(self.episode_rewards)
            self.get_logger().info(f'Testing Summary: Avg Steps: {avg_steps:.2f}, Avg Reward: {avg_reward:.2f}')
        
        plt.close(self.fig)

    def update_plot(self):
        """Update the matplotlib plot with current data"""
        self.ax1.clear()
        self.ax2.clear()
        episodes = list(range(len(self.episode_steps)))
        if episodes:
            self.ax1.plot(episodes, self.episode_steps, 'b-', label='Steps', linewidth=2)
            self.ax2.plot(episodes, self.episode_rewards, 'r-', label='Reward', linewidth=2)
            self.ax1.set_xlabel('Episode')
            self.ax1.set_ylabel('Steps', color='b')
            self.ax2.set_ylabel('Reward', color='r')
            self.ax1.tick_params(axis='y', labelcolor='b')
            self.ax2.tick_params(axis='y', labelcolor='r')
            self.ax1.grid(True, alpha=0.3)
            self.fig.legend(loc='upper left')
            self.fig.tight_layout()
            self.ax1.set_title('DQN Testing Performance')
        plt.draw()
        plt.pause(0.001)

    def plot_callback(self):
        """Keep the matplotlib event loop alive"""
        plt.pause(0.001)

    def lidar_callback(self, msg):
        """Process lidar data and control the robot"""
        if self.episode >= self.max_episodes or not self.goal_received:
            return
        
        # Process lidar data
        lidar_data = self.preprocess_lidar(msg.ranges)
        
        # Create complete state (lidar + goal info)
        self.current_state = self.create_state(lidar_data)
        
        # Skip if state is not properly initialized
        if np.all(self.current_state == 0):
            return
        
        # Choose action using trained model
        action_idx = self.choose_action(self.current_state)
        linear_x, angular_z = self.actions[action_idx]
        
        # Publish velocity
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)
        
        # Calculate reward
        reward, done = self.calculate_reward(lidar_data)
        self.total_reward += reward
        self.step += 1
        
        # Check if episode should end
        if done or self.step >= self.max_steps:
            if done:
                self.get_logger().info(f'Episode {self.episode}: Goal reached at step {self.step}')
            else:
                self.get_logger().info(f'Episode {self.episode}: Max steps reached')
            self.reset_episode()
            return
        
        # Print progress every 50 steps
        if self.step % 50 == 0:
            self.get_logger().info(f'Episode {self.episode}: Step {self.step}, Reward: {self.total_reward:.2f}')

    def bumper_callback(self, msg):
        """Handle collisions"""
        if self.episode >= self.max_episodes:
            return
        
        if len(msg.states) > 0:
            self.get_logger().info(f'Episode {self.episode}: Collision detected at step {self.step}')
            self.total_reward -= 100.0  # Same penalty as in training
            
            # Store data for plotting
            self.episode_steps.append(self.step)
            self.episode_rewards.append(self.total_reward)
            self.update_plot()
            
            # Reset for next episode
            self.episode += 1
            self.step = 0
            self.total_reward = 0.0
            self.last_distance_to_goal = None
            
            # Check if testing is complete
            if self.episode >= self.max_episodes:
                self.save_results()
                self.get_logger().info('Testing completed, shutting down...')
                rclpy.shutdown()
            else:
                # Reset robot position to start new episode
                self.reset_robot_position()

def main():
    rclpy.init()
    node = DQNTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Testing interrupted by user')
    finally:
        if hasattr(node, 'fig'):
            plt.close(node.fig)
        rclpy.shutdown()

if __name__ == '__main__':
    main()