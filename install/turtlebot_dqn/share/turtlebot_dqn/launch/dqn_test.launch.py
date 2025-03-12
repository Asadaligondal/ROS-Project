from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    world_file = os.path.join(pkg_turtlebot3_gazebo, 'worlds', 'turtlebot3_world.world')
    model_path = os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')

    world = LaunchConfiguration('world', default=world_file)

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=world_file,
            description='Path to the Gazebo world file'
        ),
        # Start Gazebo server with ROS2 plugins
        ExecuteProcess(
            cmd=['gzserver', world, '--verbose', 
                 '-s', 'libgazebo_ros_init.so', 
                 '-s', 'libgazebo_ros_factory.so', 
                 '-s', 'libgazebo_ros_state.so'],  # Added for /set_model_state
            output='screen',
            name='gzserver'
        ),
        # Start Gazebo client
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen',
            name='gzclient'
        ),
        # Spawn Turtlebot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'burger', '-file', model_path, '-x', '0.0', '-y', '0.0', '-z', '0.1'],
            output='screen',
            additional_env={'GAZEBO_MODEL_PATH': os.path.dirname(model_path)}
        )
    ])