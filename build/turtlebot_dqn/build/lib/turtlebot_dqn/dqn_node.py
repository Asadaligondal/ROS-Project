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
import torch.optim as optim
import random
import os
import math
import matplotlib.pyplot as plt
from collections import deque

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions from the buffer"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )
    
    def __len__(self):
        return len(self.buffer)

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
            ContactsState, '/bumper_states', self.bumper_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Setup entity state client for position reset
        self.set_entity_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # DQN setup
        self.respawning = False
        self.respawn_timer = None
        self.state_size = 8
        self.action_size = 4
        self.dqn = DQNNetwork(self.state_size, self.action_size)
        self.target_dqn = DQNNetwork(self.state_size, self.action_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())  # Initialize target network with same weights
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10  # Update target network every 10 episodes
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=50000)
        self.min_replay_size = 1000  # Minimum transitions before training starts
        
        # Actions
        self.actions = [
            (0.1, 0.0),  # Forward
            (0.0, 0.5),  # Left
            (0.0, -0.5), # Right
            (0.0, 0.0)   # Stop
        ]
        
        # Episode tracking
        self.max_episodes = 1000
        self.max_steps = 500
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.current_state = None
        self.last_state = None
        self.last_action = None
        self.train_ready = False  # Flag to indicate if we're ready to start training
        
        # Data storage for plotting
        self.episode_steps = []
        self.episode_rewards = []
        self.losses = []
        random.seed(42)
        self.stop_robot()

        # Matplotlib setup
        plt.ion()  # Enable interactive mode for non-blocking plotting
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()  # Dual y-axis
        self.update_plot()  # Initial empty plot

        # ROS2 timer to keep plot responsive
        self.plot_timer = self.create_timer(0.1, self.plot_callback)  # 10 Hz to process GUI events
        
        # Training timer (decoupled from callbacks for stability)
        self.train_timer = self.create_timer(0.2, self.train_network)  # 5 Hz for training

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

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
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float()
                q_values = self.dqn(state_tensor)
                action = q_values.argmax().item()
                return action

    def train_network(self):
        """Train the DQN network using a batch from the replay buffer"""
        if len(self.memory) < self.min_replay_size or not self.train_ready:
            return
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert numpy arrays to PyTorch tensors
        states_tensor = torch.from_numpy(states).float()
        actions_tensor = torch.from_numpy(actions).long()
        rewards_tensor = torch.from_numpy(rewards).float()
        next_states_tensor = torch.from_numpy(next_states).float()
        dones_tensor = torch.from_numpy(dones).float()
        
        # Compute current Q values
        current_q_values = self.dqn(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using target network (double DQN)
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        self.losses.append(loss.item())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        
        self.optimizer.step()

    def reset_episode(self):
        # Store data for plotting
        self.episode_steps.append(self.step)
        self.episode_rewards.append(self.total_reward)
        
        # Update target network periodically
        if self.episode % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.get_logger().info(f'Target network updated at episode {self.episode}')
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        self.last_state = None
        self.last_action = None
        
        # Reset robot position
        self.reset_robot_position()
        
        if self.episode >= self.max_episodes:
            self.save_model()
            self.get_logger().info('Training completed, shutting down...')
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


    def enable_collision_detection(self):
        self.respawning = False
        if self.respawn_timer:
            self.respawn_timer.cancel()  # Self-cancel after execution
            self.respawn_timer = None

    def position_reset_callback(self, future):
        """Callback after position reset"""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error('Failed to reset robot position')
                self.respawning = False
            else:
                self.stop_robot()
                self.respawn_timer = self.create_timer(1.0, self.enable_collision_detection)
        except Exception as e:
            self.get_logger().error(f'Robot position reset failed: {e}')
            self.respawning = False
        

    def save_model(self):
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        torch.save(self.dqn.state_dict(), model_path)
        self.get_logger().info(f'Model saved to {model_path}')
        
        # Save final plot
        plot_path = os.path.expanduser('~/turtlebot0/dqn_training_plot.png')
        self.update_plot()  # Ensure final data is plotted
        self.fig.savefig(plot_path)
        self.get_logger().info(f'Final plot saved to {plot_path}')
        plt.close(self.fig)

    def update_plot(self):
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
            
            # Add loss plot if we have loss data
            if self.losses:
                ax3 = self.fig.add_subplot(212)
                ax3.plot(self.losses[-100:], 'g-', label='Loss')
                ax3.set_xlabel('Training Steps (last 100)')
                ax3.set_ylabel('Loss', color='g')
                ax3.tick_params(axis='y', labelcolor='g')
                
        plt.draw()  # Redraw the plot
        plt.pause(0.001)  # Brief pause to update GUI

    def plot_callback(self):
        # Keep the matplotlib event loop alive
        plt.pause(0.001)  # Process GUI events without blocking

    def lidar_callback(self, msg):
        if self.episode >= self.max_episodes:
            return
        
        # Get current state from lidar data
        current_state = self.preprocess_lidar(msg.ranges)
        self.current_state = current_state
        
        # Choose action based on current state
        action_idx = self.choose_action(current_state)
        linear_x, angular_z = self.actions[action_idx]
        
        # Publish velocity
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)
        
        # If we have a previous state-action pair, store the transition
        if self.last_state is not None:
            # Calculate reward: +1 per step (survival reward)
            reward = 1.0
            
            # Add transition to replay buffer (current step is not done yet)
            self.memory.add(
                self.last_state, 
                self.last_action, 
                reward, 
                current_state, 
                False  # not done yet
            )
            
            # Track reward for this episode
            self.total_reward += reward
        
        # Update state/action for next iteration
        self.last_state = current_state
        self.last_action = action_idx
        
        # Increment step counter
        self.step += 1
        
        # Set train ready flag once we have enough experience
        if len(self.memory) >= self.min_replay_size and not self.train_ready:
            self.train_ready = True
            self.get_logger().info(f'Starting training at episode {self.episode}, step {self.step}')
        
        # Print episode progress every 50 steps
        if self.step % 50 == 0:
            self.get_logger().info(
                f'Episode {self.episode}: Step {self.step}, '
                f'Reward: {self.total_reward}, Epsilon: {self.epsilon:.3f}, '
                f'Buffer: {len(self.memory)}'
            )
        
        # Check if episode is done due to max steps
        if self.step >= self.max_steps:
            self.get_logger().info(f'Episode {self.episode}: Max steps reached')
            
            # Add final transition with done=True
            if self.last_state is not None:
                self.memory.add(
                    self.last_state,
                    self.last_action,
                    1.0,  # Final reward
                    current_state,
                    True  # done
                )
            
            self.reset_episode()

    def bumper_callback(self, msg):

        if self.episode >= self.max_episodes:
            return
        
        if len(msg.states) > 0 and self.last_state is not None:
            self.get_logger().info(f'Episode {self.episode}: Collision detected at step {self.step}')
            
            # Add collision transition with negative reward and done=True
            self.memory.add(
                self.last_state,
                self.last_action,
                -100.0,  # Large negative reward for collision
                self.current_state if self.current_state is not None else self.last_state,
                True  # done
            )
            
            # Update total reward
            self.total_reward -= 100.0
            
            # Store data for plotting before reset
            self.episode_steps.append(self.step)
            self.episode_rewards.append(self.total_reward)
            self.update_plot()
            
            # Increment episode counter
            self.episode += 1
            self.step = 0
            self.total_reward = 0.0
            self.last_state = None
            self.last_action = None
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Check if training is complete
            if self.episode >= self.max_episodes:
                self.save_model()
                self.get_logger().info('Training completed, shutting down...')
                rclpy.shutdown()
            else:
                # Reset robot position to start new episode
                self.respawning = True 
                self.reset_robot_position()

def main():
    rclpy.init()
    node = DQNNode()
    rclpy.spin(node)  # Run ROS2 loop without blocking for plt.show()
    rclpy.shutdown()

if __name__ == '__main__':
    main()