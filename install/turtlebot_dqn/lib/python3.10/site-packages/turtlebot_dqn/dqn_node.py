#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState
from rosgraph_msgs.msg import Clock
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import math
from collections import deque
import time

# Constants
ACTION_LINEAR = 0
ACTION_ANGULAR = 1
MAX_STEPS = 500
LIDAR_DISTANCE_CAP = 5.0
COLLISION_THRESHOLD = 0.3
EPISODE_TIMEOUT_SECONDS = 60

class ReplayBuffer:
    """Efficient experience replay buffer with numpy arrays"""
    def __init__(self, capacity=50000, state_size=8, action_size=1):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        
        # Pre-allocate memory for all buffers
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

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
        
        # QoS Profiles for better performance
        qos = QoSProfile(depth=10)
        from rclpy.qos import QoSReliabilityPolicy, QoSDurabilityPolicy
        qos_clock = QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        
        # ROS2 publishers/subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile=qos_profile_sensor_data)
        self.bumper_sub = self.create_subscription(
            ContactsState, '/bumper_states', self.bumper_callback, qos)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        
        
        # Setup entity state client for position reset
        self.set_entity_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # DQN setup
        self.state_size = 8
        self.action_size = 4
        self.dqn = DQNNetwork(self.state_size, self.action_size)
        self.target_dqn = DQNNetwork(self.state_size, self.action_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # Training parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=50000, state_size=self.state_size)
        self.min_replay_size = 1000
        
        # Actions - same as your original
        self.actions = [
            (0.1, 0.0),   # Forward
            (0.0, 0.5),   # Left
            (0.0, -0.5),  # Right
            (0.0, 0.0)    # Stop
        ]
        
        # Episode tracking
        self.max_episodes = 1000
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.current_state = None
        self.last_state = None
        self.last_action = None
        self.train_ready = False
        
        # Metrics tracking (without real-time plotting)
        self.episode_steps = []
        self.episode_rewards = []
        self.losses = []
        
        # Status flags
        self.respawning = False
        self.collision_detected = False
        self.done = False
        
        # Time management (from reference code)
        self.time_sec = 0
        self.episode_deadline = float('inf')
        self.reset_deadline = False
        self.clock_msgs_skipped = 0
        
        # Initialize
        random.seed(42)
        self.stop_robot()
        
        # Create a separate timer for training (decoupled from sensor callbacks)
        self.train_timer = self.create_timer(0.1, self.train_step)
        
        # Periodic log timer (low frequency to minimize impact)
        self.log_timer = self.create_timer(5.0, self.log_progress)

    def clock_callback(self, msg):
        """Track simulation time for episode timeouts"""
        self.time_sec = msg.clock.sec
        
        if not self.reset_deadline:
            return
            
        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 10:  # Wait for simulation to reset clock
            return
            
        self.episode_deadline = self.time_sec + EPISODE_TIMEOUT_SECONDS
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

    def stop_robot(self):
        """Publish zero velocity to stop the robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def preprocess_lidar(self, ranges):
        """Efficiently process LiDAR data into state representation"""
        # Pre-allocate numpy array for better performance
        sectors = np.zeros(8, dtype=np.float32)
        sector_size = len(ranges) // 8
        
        for i in range(8):
            start = i * sector_size
            end = (i + 1) * sector_size
            # Use numpy min for efficiency
            min_dist = np.min(ranges[start:end])
            sectors[i] = min_dist if min_dist < LIDAR_DISTANCE_CAP else LIDAR_DISTANCE_CAP
            
        return sectors

    def choose_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            q_values = self.dqn(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Periodic training step (decoupled from sensor callbacks)"""
        # Check if episode is done (timeout)
        if self.time_sec >= self.episode_deadline and not self.done and not self.respawning:
            self.get_logger().info(f'Episode {self.episode}: Timeout at {self.time_sec}')
            if self.last_state is not None:
                self.memory.add(
                    self.last_state,
                    self.last_action,
                    -10.0,  # Penalty for timeout
                    self.current_state if self.current_state is not None else self.last_state,
                    True
                )
            self.done = True
            self.reset_episode()
            return
            
        # Skip training if not ready or during reset
        if len(self.memory) < self.min_replay_size or not self.train_ready or self.respawning:
            return
        
        # Sample batch and convert to tensors (single batch per training step)
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_tensor = torch.from_numpy(states).float()
        actions_tensor = torch.from_numpy(actions).long()
        rewards_tensor = torch.from_numpy(rewards).float()
        next_states_tensor = torch.from_numpy(next_states).float()
        dones_tensor = torch.from_numpy(dones).float()
        
        # Compute current Q values
        current_q_values = self.dqn(states_tensor).gather(1, actions_tensor).squeeze(1)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor.squeeze(1) + (1 - dones_tensor.squeeze(1)) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.loss_fn(current_q_values, target_q_values)
        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

    def reset_episode(self):
        """Reset environment for a new episode"""
        self.stop_robot()
        
        # Store metrics
        self.episode_steps.append(self.step)
        self.episode_rewards.append(self.total_reward)
        
        # Update target network periodically
        if self.episode % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.get_logger().info(f'Target network updated at episode {self.episode}')
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Increment episode counter and reset step counter
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        self.last_state = None
        self.last_action = None
        self.done = False
        
        # Save model if training is complete
        if self.episode >= self.max_episodes:
            self.save_model()
            self.get_logger().info('Training completed, shutting down...')
            rclpy.shutdown()
            return
        
        # Reset robot position
        self.respawning = True
        self.reset_robot_position()
        self.reset_deadline = True

    def reset_robot_position(self):
        """Reset the robot to a random starting position"""
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
                self.respawning = False
            else:
                self.stop_robot()
                # Create a one-shot timer to enable collision detection after a delay
                self.respawn_timer = self.create_timer(1.0, self.enable_collision_detection)
        except Exception as e:
            self.get_logger().error(f'Robot position reset failed: {e}')
            self.respawning = False

    def enable_collision_detection(self):
        """Re-enable collision detection after respawn"""
        self.respawning = False
        if self.respawn_timer:
            self.respawn_timer.cancel()  # Self-cancel after execution
            self.respawn_timer = None

    def save_model(self):
        """Save the trained model and metrics"""
        # Save model
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        torch.save(self.dqn.state_dict(), model_path)
        self.get_logger().info(f'Model saved to {model_path}')
        
        # Save metrics as numpy arrays for later analysis
        metrics_path = os.path.expanduser('~/turtlebot0/dqn_metrics.npz')
        np.savez(
            metrics_path,
            steps=np.array(self.episode_steps),
            rewards=np.array(self.episode_rewards),
            losses=np.array(self.losses)
        )
        self.get_logger().info(f'Metrics saved to {metrics_path}')

    def log_progress(self):
        """Periodically log training progress"""
        if self.episode == 0:
            return
            
        self.get_logger().info(
            f'Episode: {self.episode}/{self.max_episodes}, '
            f'Steps: {self.step}, '
            f'Reward: {self.total_reward:.2f}, '
            f'Epsilon: {self.epsilon:.3f}, '
            f'Buffer: {len(self.memory)}, '
            f'Loss: {self.losses[-1] if self.losses else 0:.4f}'
        )

    def lidar_callback(self, msg):
        """Process LiDAR data and take actions"""
        if self.episode >= self.max_episodes or self.respawning:
            return
        
        # Process LiDAR data
        current_state = self.preprocess_lidar(msg.ranges)
        self.current_state = current_state
        
        # Choose and execute action
        action_idx = self.choose_action(current_state)
        linear_x, angular_z = self.actions[action_idx]
        
        # Add motor noise if enabled (like in reference code)
        # Uncomment if you want this feature
        # if ENABLE_MOTOR_NOISE:
        #     linear_x += np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
        #     angular_z += np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
        
        # Publish velocity
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)
        
        # Add experience to replay buffer
        if self.last_state is not None and not self.done:
            # Basic reward function: +1 for survival, distance-based rewards could be added
            reward = 1.0
            
            # Store transition
            self.memory.add(
                self.last_state,
                self.last_action,
                reward,
                current_state,
                False  # not done yet
            )
            
            # Track reward
            self.total_reward += reward
        
        # Update for next iteration
        self.last_state = current_state
        self.last_action = action_idx
        self.step += 1
        
        # Set train ready flag once we have enough experience
        if len(self.memory) >= self.min_replay_size and not self.train_ready:
            self.train_ready = True
            self.get_logger().info(f'Starting training at episode {self.episode}')
        
        # Check if episode is done due to max steps
        if self.step >= MAX_STEPS and not self.done:
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
            
            self.done = True
            self.reset_episode()

    def bumper_callback(self, msg):
        """Handle collision events"""
        if self.episode >= self.max_episodes or self.respawning or self.done:
            return
        
        if len(msg.states) > 0 and self.last_state is not None:
            self.get_logger().info(f'Episode {self.episode}: Collision detected at step {self.step}')
            
            # Add collision transition with negative reward
            self.memory.add(
                self.last_state,
                self.last_action,
                -100.0,  # Large negative reward for collision
                self.current_state if self.current_state is not None else self.last_state,
                True  # done
            )
            
            # Update total reward
            self.total_reward -= 100.0
            self.done = True
            
            # Reset episode
            self.reset_episode()

def main():
    rclpy.init()
    node = DQNNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        # Save model on shutdown if not already done
        if node.episode > 0 and node.episode < node.max_episodes:
            node.save_model()
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()