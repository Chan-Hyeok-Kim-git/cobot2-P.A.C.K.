import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('grasp_execution'),
        'config', 'execution_coordinator.yaml')

    return LaunchDescription([
        Node(
            package='grasp_execution',
            executable='execution_coordinator_node',
            name='grasp_execution_coordinator',
            output='screen',
            parameters=[cfg],
        ),
    ])
