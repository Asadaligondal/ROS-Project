from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    world_file = os.path.join(pkg_turtlebot3_gazebo, 'worlds', 'turtlebot3_world.world')
    model_path = os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')

    return LaunchDescription([
        # Start Gazebo server with ROS2 plugins
        ExecuteProcess(
            cmd=['gzserver', world_file, '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        # Start Gazebo client
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen'
        ),
        # Spawn Turtlebot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'burger', '-file', model_path, '-x', '0.0', '-y', '0.0', '-z', '0.1'],
            output='screen'
        )
    ])