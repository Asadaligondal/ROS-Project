#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState
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
        
        # DQN setup
        self.state_size = 8
        self.action_size = 4
        self.dqn = DQNNetwork(self.state_size, self.action_size)
        
        # Load pre-trained model
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        if os.path.exists(model_path):
            self.dqn.load_state_dict(torch.load(model_path))
            self.get_logger().info(f'Model loaded from {model_path}')
            self.dqn.eval()  # Set to evaluation mode
        else:
            self.get_logger().error(f'Model file not found at {model_path}')
            rclpy.shutdown()
            return
        
        self.actions = [
            (0.1, 0.0),  # Forward
            (0.0, 0.5),  # Left
            (0.0, -0.5), # Right
            (0.0, 0.0)   # Stop
        ]
        
        # Episode tracking
        self.max_episodes = 50  # Test for fewer episodes than training
        self.max_steps = 300
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.current_state = None
        
        # Data storage for plotting
        self.episode_steps = []
        self.episode_rewards = []
        random.seed()
        self.stop_robot()

        # Matplotlib setup
        plt.ion()  # Enable interactive mode for non-blocking plotting
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()  # Dual y-axis
        self.update_plot()  # Initial empty plot

        # ROS2 timer to keep plot responsive
        self.plot_timer = self.create_timer(0.1, self.plot_callback)  # 10 Hz to process GUI events
        
        self.get_logger().info('DQN test node initialized - starting testing...')
        
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
            min_dist = min(sector) if min(sector) < 5.0 else 5.0
            sectors.append(min_dist)
        return np.array(sectors, dtype=np.float32)

    def choose_action(self, state):
        """Choose action based on the trained model (no exploration)"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            q_values = self.dqn(state_tensor)
            action = q_values.argmax().item()
            return action

    def reset_episode(self):
        self.get_logger().info(f'Episode {self.episode} completed - Steps: {self.step}, Total Reward: {self.total_reward}')
        # Store data for plotting
        self.episode_steps.append(self.step)
        self.episode_rewards.append(self.total_reward)
        self.update_plot()  # Refresh plot after each episode
        
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        
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
        except Exception as e:
            self.get_logger().error(f'Robot position reset failed: {e}')

    def save_results(self):
        """Save testing results"""
        # Save final plot
        plot_path = os.path.expanduser('~/turtlebot0/dqn_testing_plot.png')
        self.update_plot()  # Ensure final data is plotted
        self.fig.savefig(plot_path)
        self.get_logger().info(f'Testing plot saved to {plot_path}')
        
        # Save data as CSV for further analysis
        import csv
        csv_path = os.path.expanduser('~/turtlebot0/dqn_testing_results.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Steps', 'Reward'])
            for i in range(len(self.episode_steps)):
                writer.writerow([i, self.episode_steps[i], self.episode_rewards[i]])
        self.get_logger().info(f'Testing data saved to {csv_path}')
        
        plt.close(self.fig)

    def update_plot(self):
        """Update the matplotlib plot with current data"""
        self.ax1.clear()
        self.ax2.clear()
        episodes = list(range(len(self.episode_steps)))
        if episodes:
            self.ax1.plot(episodes, self.episode_steps, 'b-', label='Steps')
            self.ax2.plot(episodes, self.episode_rewards, 'r-', label='Reward')
            self.ax1.set_xlabel('Episode')
            self.ax1.set_ylabel('Steps', color='b')
            self.ax2.set_ylabel('Reward', color='r')
            self.ax1.tick_params(axis='y', labelcolor='b')
            self.ax2.tick_params(axis='y', labelcolor='r')
            self.fig.legend(loc='upper left')
            self.fig.tight_layout()
            self.ax1.set_title('DQN Testing Performance')
        plt.draw()  # Redraw the plot
        plt.pause(0.001)  # Brief pause to update GUI

    def plot_callback(self):
        """Keep the matplotlib event loop alive"""
        plt.pause(0.001)  # Process GUI events without blocking

    def lidar_callback(self, msg):
        """Process lidar data and control the robot"""
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
        
        # Reward: +1 per step (same as training)
        self.total_reward += 1.0
        self.step += 1
        
        # Print episode progress every 50 steps
        if self.step % 50 == 0:
            self.get_logger().info(f'Episode {self.episode}: Step {self.step}, Current reward: {self.total_reward}')
        
        if self.step >= self.max_steps:
            self.get_logger().info(f'Episode {self.episode}: Max steps reached')
            self.reset_episode()

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
            
            # Increment episode counter
            self.episode += 1
            self.step = 0
            self.total_reward = 0.0
            
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
    rclpy.spin(node)  # Run ROS2 loop
    rclpy.shutdown()

if __name__ == '__main__':
    main()