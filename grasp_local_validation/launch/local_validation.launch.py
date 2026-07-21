"""
grasp_local_validation 파이프라인 launch.

기본: candidate_generator_node + local_validator_node 를 함께 띄운다.
'use_mock:=true' 로 실행하면 mock_scene_publisher 도 같이 띄워서
YOLO/RealSense 없이 전체 흐름을 확인할 수 있다.

사용 예:
  ros2 launch grasp_local_validation local_validation.launch.py use_mock:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_mock = LaunchConfiguration('use_mock')

    local_validator_yaml = os.path.join(
        get_package_share_directory('grasp_local_validation'),
        'config', 'local_validator.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('use_mock', default_value='false',
                               description='mock_scene_publisher 함께 실행 여부'),

        Node(
            package='grasp_local_validation',
            executable='candidate_generator_node',
            name='grasp_candidate_generator',
            output='screen',
        ),
        Node(
            package='grasp_local_validation',
            executable='local_validator_node',
            name='grasp_local_validator',
            output='screen',
            parameters=[local_validator_yaml],
        ),
        Node(
            package='grasp_local_validation',
            executable='mock_scene_publisher',
            name='mock_scene_publisher',
            output='screen',
            condition=IfCondition(use_mock),
        ),
    ])
