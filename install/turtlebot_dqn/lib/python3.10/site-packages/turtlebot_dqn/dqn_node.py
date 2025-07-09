#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState, GetEntityState
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry  # Added for odometry
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import math
from collections import deque
import time
from geometry_msgs.msg import Twist, Pose, Pose2D 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

ACTION_SIZE = 5
HIDDEN_SIZE = 512
BATCH_SIZE = 128
BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.003
OBSERVE_STEPS = 25000
EPSILON_DECAY = 0.9995
EPSILON_MINIMUM = 0.05
TARGET_UPDATE_FREQUENCY = 1000

MAX_STEPS = 500
LIDAR_DISTANCE_CAP = 3.5
COLLISION_THRESHOLD = 0.13
GOAL_THRESHOLD = 0.20
EPISODE_TIMEOUT_SECONDS = 50

# Actions following GitHub's POSSIBLE_ACTIONS
POSSIBLE_ACTIONS = [[0.3, -1.0], [0.3, -0.5], [1.0, 0.0], [0.3, 0.5], [0.3, 1.0]]

def euler_from_quaternion(quaternion):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

class ReplayBuffer:
    """Efficient experience replay buffer with numpy arrays"""
    def __init__(self, capacity=BUFFER_SIZE, state_size=362, action_size=1):
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
    """DQN Network following GitHub architecture"""
    def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNNode(Node):
    def __init__(self):
        super().__init__('dqn_node')
        
        # QoS Profiles
        qos = QoSProfile(depth=10)
        from rclpy.qos import QoSReliabilityPolicy, QoSDurabilityPolicy
        qos_clock = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        
        # ROS2 publishers/subscribers
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile=qos_profile_sensor_data)
        self.bumper_sub = self.create_subscription(
            ContactsState, '/bumper_states', self.bumper_callback, qos)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.goal_sub = self.create_subscription(
            Pose2D, '/goal_pose', self.goal_callback, qos)
        
        # NEW: Odometry subscription for real-time position and heading
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)
        
        # Goal and robot position tracking
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0  # NEW: Robot's current heading
        self.goal_distance = 0.0   # NEW: Current distance to goal
        self.goal_angle = 0.0      # NEW: Relative angle to goal
        self.initial_distance_to_goal = 0.0  # NEW: For reward normalization
        self.last_distance_to_goal = None
        
        # Gazebo services (still needed for robot respawning)
        self.set_entity_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # DQN setup - 360 LiDAR + goal_distance + goal_angle = 362 inputs
        self.state_size = 362
        self.action_size = ACTION_SIZE
        
        # Create networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_dqn = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # Training parameters following GitHub config
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MINIMUM
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.target_update_freq = TARGET_UPDATE_FREQUENCY
        self.observe_steps = OBSERVE_STEPS
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=BUFFER_SIZE, state_size=self.state_size)
        
        # Episode tracking
        self.max_episodes = 10000
        self.episode = 0
        self.step = 0
        self.total_step = 0  # Global step counter for target updates
        self.total_reward = 0.0
        self.current_state = None
        self.last_state = None
        self.last_action = None
        
        # Metrics tracking
        self.episode_steps = []
        self.episode_rewards = []
        self.losses = []
        
        # Status flags
        self.respawning = False
        self.collision_detected = False
        self.done = False
        self.goal_reached = False
        
        # Time management
        self.time_sec = 0
        self.episode_deadline = float('inf')
        self.reset_deadline = False
        self.clock_msgs_skipped = 0
        
        # Initialize
        random.seed(42)
        torch.manual_seed(42)
        self.stop_robot()
        
        # Periodic log timer
        self.log_timer = self.create_timer(5.0, self.log_progress)

    def goal_callback(self, msg):
        """Update goal position"""
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.get_logger().info(f'New goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}')

    def odom_callback(self, msg):
        """NEW: Real-time odometry callback for position and heading"""
        # Update robot position
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Convert quaternion to euler angles to get heading
        _, _, self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)
        
        # Calculate goal distance and relative angle (like GitHub implementation)
        diff_x = self.goal_x - self.robot_x
        diff_y = self.goal_y - self.robot_y
        self.goal_distance = math.sqrt(diff_x**2 + diff_y**2)
        
        # Calculate heading to goal
        heading_to_goal = math.atan2(diff_y, diff_x)
        
        # Calculate relative angle (how much robot needs to turn to face goal)
        self.goal_angle = heading_to_goal - self.robot_heading
        
        # Normalize angle to [-π, π]
        while self.goal_angle > math.pi:
            self.goal_angle -= 2 * math.pi
        while self.goal_angle < -math.pi:
            self.goal_angle += 2 * math.pi

    def clock_callback(self, msg):
        """Track simulation time for episode timeouts"""
        self.time_sec = msg.clock.sec
        
        if not self.reset_deadline:
            return
            
        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 10:
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

    def preprocess_state(self, ranges):
        """Process full LiDAR data (360 readings) and goal information into state representation"""
        # Process full LiDAR scan (360 readings)
        lidar_data = np.array(ranges, dtype=np.float32)
        
        # Cap and normalize LiDAR values
        lidar_data = np.clip(lidar_data, 0, LIDAR_DISTANCE_CAP)
        lidar_data = lidar_data / LIDAR_DISTANCE_CAP  # Normalize to [0, 1]
        
        # NEW: Use goal distance and relative angle (from odometry)
        # Normalize goal distance (cap at 10m like GitHub)
        normalized_distance = min(self.goal_distance, 10.0) / 10.0
        
        # Normalize goal angle (already in [-π, π], so divide by π)
        normalized_angle = self.goal_angle / math.pi
        
        # Combine LiDAR (360) + goal info (2) = 362 dimensional state
        state = np.concatenate([lidar_data, [normalized_distance, normalized_angle]])
        
        return state

    def choose_action(self, state):
        """Select action using epsilon-greedy policy following GitHub approach"""
        # During initial exploration phase, take random actions
        if self.total_step < self.observe_steps:
            return random.randint(0, self.action_size - 1)
        
        # Epsilon-greedy after initial exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploit: use DQN to select action
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            q_values = self.dqn(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Train the DQN following GitHub approach"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device)
        
        # Current Q values
        current_q_values = self.dqn(states_tensor).gather(1, actions_tensor)
        
        # Next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states_tensor).amax(1, keepdim=True)
            target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0, norm_type=1)
        self.optimizer.step()
        
        # Update target network
        if self.total_step % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        return loss.item()

    def calculate_reward(self, action_linear, action_angular):
        """NEW: GitHub-style reward function with multiple components"""
        
        # Check for goal reached first
        if self.goal_distance < GOAL_THRESHOLD:
            self.goal_reached = True
            return 2500.0  # Large success reward like GitHub
        
        # Goal angle reward (penalize not facing goal) - Range: [-π, 0]
        r_yaw = -1.0 * abs(self.goal_angle)
        
        # Angular velocity penalty (penalize excessive turning) - Range: [-4, 0]
        r_vangular = -1.0 * (action_angular ** 2)
        
        # Distance progress reward using GitHub formula - Range: [-1, 1]
        if self.initial_distance_to_goal > 0:
            r_distance = (2 * self.initial_distance_to_goal) / (self.initial_distance_to_goal + self.goal_distance) - 1
        else:
            r_distance = 0.0
        
        # Linear velocity penalty (encourage moving forward) - Range: [-9.68, 0]
        r_vlinear = -1.0 * (((0.22 - action_linear) * 10) ** 2)
        
        # Combine all rewards with base penalty
        reward = r_yaw + r_distance + r_vangular + r_vlinear - 1.0
        
        return reward

    def initialize_episode(self):
        """NEW: Initialize episode with goal distance storage"""
        self.initial_distance_to_goal = self.goal_distance
        self.last_distance_to_goal = self.goal_distance
        self.get_logger().info(f'Episode {self.episode}: Initial distance to goal: {self.initial_distance_to_goal:.2f}m')

    def reset_episode(self):
        """Reset environment for a new episode"""
        self.stop_robot()
        
        # Store metrics
        self.episode_steps.append(self.step)
        self.episode_rewards.append(self.total_reward)
        
        # Decay epsilon after initial exploration phase
        if self.total_step >= self.observe_steps and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Increment episode counter and reset step counter
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        self.last_state = None
        self.last_action = None
        self.last_distance_to_goal = None
        self.initial_distance_to_goal = 0.0
        self.done = False
        self.goal_reached = False
        
        # Save model periodically
        if self.episode > 0 and self.episode % 100 == 0:
            self.save_model()
        
        # Check if training is complete
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
                self.respawning = False
            else:
                self.stop_robot()
                # Create a one-shot timer to enable collision detection and initialize episode
                self.respawn_timer = self.create_timer(1.0, self.enable_collision_detection)
        except Exception as e:
            self.get_logger().error(f'Robot position reset failed: {e}')
            self.respawning = False

    def enable_collision_detection(self):
        """Re-enable collision detection after respawn and initialize episode"""
        self.respawning = False
        # NEW: Initialize episode after robot is repositioned
        self.initialize_episode()
        if hasattr(self, 'respawn_timer') and self.respawn_timer:
            self.respawn_timer.cancel()
            self.respawn_timer = None

    def save_model(self):
        """Save the trained model and metrics"""
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        torch.save(self.dqn.state_dict(), model_path)
        self.get_logger().info(f'Model saved to {model_path}')
        
        # Save metrics
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
            f'Total Steps: {self.total_step}, '
            f'Reward: {self.total_reward:.2f}, '
            f'Epsilon: {self.epsilon:.3f}, '
            f'Distance to Goal: {self.goal_distance:.2f}m, '
            f'Buffer: {len(self.memory)}'
        )

    def lidar_callback(self, msg):
        """Process LiDAR data and take actions"""
        if self.episode >= self.max_episodes or self.respawning:
            return
        
        # Check episode timeout
        if self.time_sec >= self.episode_deadline and not self.done:
            self.get_logger().info(f'Episode {self.episode}: Timeout at {self.time_sec}')
            if self.last_state is not None:
                self.memory.add(
                    self.last_state,
                    self.last_action,
                    -50.0,  # Timeout penalty
                    self.current_state if self.current_state is not None else self.last_state,
                    True
                )
            self.done = True
            self.reset_episode()
            return
        
        # Process LiDAR data into state (now using odometry-based goal info)
        current_state = self.preprocess_state(msg.ranges)
        self.current_state = current_state
        
        # Choose action
        action_idx = self.choose_action(current_state)
        linear_x, angular_z = POSSIBLE_ACTIONS[action_idx]
        
        # Publish velocity command
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)
        
        # Add experience to replay buffer and train
        if self.last_state is not None and not self.done:
            # NEW: Calculate reward with action information and multiple components
            reward = self.calculate_reward(linear_x, angular_z)
            
            # Check for goal reached
            if self.goal_reached:
                self.get_logger().info(f'Episode {self.episode}: Goal reached at step {self.step}! Distance was {self.goal_distance:.2f}m')
                self.memory.add(self.last_state, self.last_action, reward, current_state, True)
                self.total_reward += reward
                self.done = True
                self.reset_episode()
                return
            
            # Store transition
            self.memory.add(self.last_state, self.last_action, reward, current_state, False)
            self.total_reward += reward
            
            # Train if we have enough experience
            if len(self.memory) >= self.batch_size:
                loss = self.train_step()
                if loss > 0:
                    self.losses.append(loss)
        
        # Update for next iteration
        self.last_state = current_state
        self.last_action = action_idx
        self.step += 1
        self.total_step += 1
        
        # Check if episode is done due to max steps
        if self.step >= MAX_STEPS and not self.done:
            self.get_logger().info(f'Episode {self.episode}: Max steps reached')
            if self.last_state is not None:
                self.memory.add(self.last_state, self.last_action, -10.0, current_state, True)
            self.done = True
            self.reset_episode()

    def bumper_callback(self, msg):
        """Handle collision events"""
        if self.episode >= self.max_episodes or self.respawning or self.done:
            return
        
        if len(msg.states) > 0 and self.last_state is not None:
            self.get_logger().info(f'Episode {self.episode}: Collision detected at step {self.step}')
            
            # NEW: Large collision penalty like GitHub
            self.memory.add(
                self.last_state,
                self.last_action,
                -2000.0,  # Much larger penalty like GitHub
                self.current_state if self.current_state is not None else self.last_state,
                True
            )
            
            self.total_reward -= 2000.0
            self.done = True
            self.reset_episode()

def main():
    rclpy.init()
    node = DQNNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        if node.episode > 0 and node.episode < node.max_episodes:
            node.save_model()
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()