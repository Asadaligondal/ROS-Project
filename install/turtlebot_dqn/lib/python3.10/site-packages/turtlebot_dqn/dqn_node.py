#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose2D
from gazebo_msgs.msg import ContactsState, EntityState
from gazebo_msgs.srv import SetEntityState
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
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

# Configuration
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
OBSTACLE_PENALTY_THRESHOLD = 0.22
EPISODE_TIMEOUT_SECONDS = 50

# Actions: [linear_velocity, angular_velocity]
POSSIBLE_ACTIONS = [[0.3, -1.0], [0.3, -0.5], [1.0, 0.0], [0.3, 0.5], [0.3, 1.0]]

def euler_from_quaternion(q):
    """Convert quaternion to euler angles"""
    x, y, z, w = q.x, q.y, q.z, q.w
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=BUFFER_SIZE, state_size=362):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices])
    
    def __len__(self):
        return self.size

class DQNNetwork(nn.Module):
    """DQN Network"""
    def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNNode(Node):
    def __init__(self):
        super().__init__('dqn_node')
        
        # ROS2 setup
        qos = QoSProfile(depth=10)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 
                                                  qos_profile=qos_profile_sensor_data)
        self.bumper_sub = self.create_subscription(ContactsState, '/bumper_states', 
                                                   self.bumper_callback, qos)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.goal_sub = self.create_subscription(Pose2D, '/goal_pose', self.goal_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        
        # Gazebo service
        self.set_entity_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        
        # Goal and position tracking
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0
        self.goal_distance = 0.0
        self.goal_angle = 0.0
        self.initial_distance_to_goal = 0.0
        self.min_obstacle_distance = LIDAR_DISTANCE_CAP
        
        # DQN setup
        self.state_size = 362  # 360 LiDAR + 2 goal info
        self.action_size = ACTION_SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dqn = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_dqn = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # Training parameters
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MINIMUM
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.target_update_freq = TARGET_UPDATE_FREQUENCY
        self.observe_steps = OBSERVE_STEPS
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(capacity=BUFFER_SIZE, state_size=self.state_size)
        
        # Episode tracking
        self.max_episodes = 10000
        self.episode = 0
        self.step = 0
        self.total_step = 0 
        self.total_reward = 0.0
        self.current_state = None
        self.last_state = None
        self.last_action = None
        
        # Performance tracking
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.performance_window = deque(maxlen=100)
        
        # Status flags
        self.respawning = False
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
        
        # Create logging directory
        self.log_dir = os.path.expanduser('~/turtlebot_dqn_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Periodic log timer
        self.log_timer = self.create_timer(10.0, self.log_progress)
        self.save_timer = self.create_timer(300.0, self.save_checkpoint)

    def goal_callback(self, msg):
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.get_logger().info(f'New goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}')

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        _, _, self.robot_heading = euler_from_quaternion(msg.pose.pose.orientation)
        
        # Calculate goal metrics
        diff_x = self.goal_x - self.robot_x
        diff_y = self.goal_y - self.robot_y
        self.goal_distance = math.sqrt(diff_x**2 + diff_y**2)
        
        heading_to_goal = math.atan2(diff_y, diff_x)
        self.goal_angle = heading_to_goal - self.robot_heading
        
        # Normalize angle to [-π, π]
        while self.goal_angle > math.pi:
            self.goal_angle -= 2 * math.pi
        while self.goal_angle < -math.pi:
            self.goal_angle += 2 * math.pi

    def clock_callback(self, msg):
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
        self.vel_pub.publish(Twist())

    def preprocess_state(self, ranges):
        """Process LiDAR and goal information into normalized state"""
        # Process LiDAR: normalize to [0, 1]
        lidar_data = np.array(ranges, dtype=np.float32)
        lidar_data = np.clip(lidar_data, 0, LIDAR_DISTANCE_CAP) / LIDAR_DISTANCE_CAP
        
        # Track minimum obstacle distance for reward
        self.min_obstacle_distance = np.min(ranges)
        
        # Normalize goal distance to [0, 1] (cap at 10m)
        normalized_distance = min(self.goal_distance, 10.0) / 10.0
        
        # Normalize goal angle to [-1, 1]
        normalized_angle = self.goal_angle / math.pi
        
        # Combine: LiDAR (360) + goal info (2) = 362
        state = np.concatenate([lidar_data, [normalized_distance, normalized_angle]])
        
        return state

    def choose_action(self, state):
        """Epsilon-greedy action selection with observation phase"""
        # Pure exploration during observation phase
        if self.total_step < self.observe_steps:
            return random.randint(0, self.action_size - 1)
        
        # Epsilon-greedy after observation
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            q_values = self.dqn(state_tensor)
            return q_values.argmax().item()

    def calculate_reward(self, action_linear, action_angular):
        """Calculate reward with obstacle proximity penalty"""
        # Goal reached
        if self.goal_distance < GOAL_THRESHOLD:
            self.goal_reached = True
            return 2500.0
        
        # Reward components
        r_yaw = -abs(self.goal_angle)
        r_vangular = -(action_angular ** 2)
        
        # Distance progress
        if self.initial_distance_to_goal > 0:
            r_distance = (2 * self.initial_distance_to_goal) / \
                        (self.initial_distance_to_goal + self.goal_distance) - 1
        else:
            r_distance = 0.0
        
        # Velocity encouragement
        r_vlinear = -(((0.22 - action_linear) * 10) ** 2)
        
        # Obstacle proximity penalty
        r_obstacle = -20 if self.min_obstacle_distance < OBSTACLE_PENALTY_THRESHOLD else 0
        
        return r_yaw + r_distance + r_vangular + r_vlinear + r_obstacle - 1.0

    def train_step(self):
        """Train DQN"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        # Current Q values
        current_q_values = self.dqn(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        if self.total_step % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        return loss.item()

    def reset_episode(self):
        """Reset for new episode"""
        self.stop_robot()
        
        # Update metrics
        self.episode_rewards.append(self.total_reward)
        self.performance_window.append(self.goal_reached)
        
        # Decay epsilon after observation phase
        if self.total_step >= self.observe_steps and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Reset episode variables
        self.episode += 1
        self.step = 0
        self.total_reward = 0.0
        self.last_state = None
        self.last_action = None
        self.done = False
        self.goal_reached = False
        
        # Check completion
        if self.episode >= self.max_episodes:
            self.save_model()
            self.get_logger().info('Training completed!')
            rclpy.shutdown()
            return
        
        # Reset robot position
        self.respawning = True
        self.reset_robot_position()
        self.reset_deadline = True

    def reset_robot_position(self):
        """Reset robot to random position"""
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        yaw = random.uniform(-math.pi, math.pi)
        
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = 'burger'
        request.state.pose.position.x = x
        request.state.pose.position.y = y
        request.state.pose.position.z = 0.0
        
        # Set orientation
        request.state.pose.orientation.w = math.cos(yaw * 0.5)
        request.state.pose.orientation.z = math.sin(yaw * 0.5)
        request.state.twist = Twist()
        
        future = self.set_entity_client.call_async(request)
        future.add_done_callback(self.position_reset_callback)

    def position_reset_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.stop_robot()
                self.create_timer(1.0, self.enable_movement)
        except Exception as e:
            self.get_logger().error(f'Reset failed: {e}')
            self.respawning = False

    def enable_movement(self):
        self.respawning = False
        self.initial_distance_to_goal = self.goal_distance

    
    def save_model(self):
        """Save the trained model and metrics"""
        model_path = os.path.expanduser('~/turtlebot0/dqn_model.pth')
        torch.save(self.dqn.state_dict(), model_path)
        self.get_logger().info(f'Model saved to {model_path}')
        
        # Save metrics
        metrics_path = os.path.expanduser('~/turtlebot0/dqn_metrics.npz')
        np.savez(
            metrics_path,
            rewards=self.episode_rewards,
            success_count=self.success_count,
            collision_count=self.collision_count,
            timeout_count=self.timeout_count)
        self.get_logger().info(f'Metrics saved to {metrics_path}')

    def save_checkpoint(self):
        """Periodic checkpoint save"""
        if self.episode > 0:
            self.save_model()

    def log_progress(self):
        """Log training progress"""
        if self.episode == 0 or self.total_step < self.observe_steps:
            if self.total_step < self.observe_steps:
                self.get_logger().info(f'Observation phase: {self.total_step}/{self.observe_steps}')
            return
        
        success_rate = np.mean(self.performance_window) * 100 if len(self.performance_window) > 0 else 0
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0
        
        self.get_logger().info(
            f'Episode: {self.episode}, Steps: {self.total_step}, '
            f'Epsilon: {self.epsilon:.3f}, Success Rate: {success_rate:.1f}%, '
            f'Avg Reward: {avg_reward:.1f}, Buffer: {len(self.memory)}'
        )

    def lidar_callback(self, msg):
        """Main processing loop"""
        if self.episode >= self.max_episodes or self.respawning:
            return
        
        # Check timeout
        if self.time_sec >= self.episode_deadline and not self.done:
            self.get_logger().info(f'Episode {self.episode}: Timeout')
            self.timeout_count += 1
            if self.last_state is not None:
                self.memory.add(self.last_state, self.last_action, -50.0, 
                               self.last_state, True)
                self.total_reward -= 50.0
            self.done = True
            self.reset_episode()
            return
        
        # Process state
        current_state = self.preprocess_state(msg.ranges)
        self.current_state = current_state
        
        # Choose and execute action
        action_idx = self.choose_action(current_state)
        linear_x, angular_z = POSSIBLE_ACTIONS[action_idx]
        
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.vel_pub.publish(twist)
        
        # Store experience and train
        if self.last_state is not None and not self.done:
            reward = self.calculate_reward(linear_x, angular_z)
            
            if self.goal_reached:
                self.get_logger().info(f'Episode {self.episode}: Goal reached! Distance: {self.goal_distance:.2f}m')
                self.success_count += 1
                self.memory.add(self.last_state, self.last_action, reward, current_state, True)
                self.total_reward += reward
                self.done = True
                self.reset_episode()
                return
            
            self.memory.add(self.last_state, self.last_action, reward, current_state, False)
            self.total_reward += reward
            
            # Train if ready
            if len(self.memory) >= self.batch_size and self.total_step >= self.observe_steps:
                self.train_step()
        
        # Update state
        self.last_state = current_state
        self.last_action = action_idx
        self.step += 1
        self.total_step += 1
        
        # Check max steps
        if self.step >= MAX_STEPS and not self.done:
            self.get_logger().info(f'Episode {self.episode}: Max steps reached')
            if self.last_state is not None:
                self.memory.add(self.last_state, self.last_action, -10.0, current_state, True)
                self.total_reward -= 10.0
            self.done = True
            self.reset_episode()

    def bumper_callback(self, msg):
        """Handle collisions"""
        if self.episode >= self.max_episodes or self.respawning or self.done:
            return
        
        if len(msg.states) > 0 and self.last_state is not None:
            self.get_logger().info(f'Episode {self.episode}: Collision')
            self.collision_count += 1
            self.memory.add(self.last_state, self.last_action, -2000.0,
                           self.current_state if self.current_state is not None else self.last_state,
                           True)
            self.total_reward -= 2000.0
            self.done = True
            self.reset_episode()

def main():
    rclpy.init()
    node = DQNNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by user')
    finally:
        if node.episode > 0:
            node.save_model()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()