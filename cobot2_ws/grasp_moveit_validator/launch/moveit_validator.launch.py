# moveit_validator_node 실행 launch.
# grasp_moveit_config 패키지의 SRDF/kinematics 등이 먼저 로드되어 있어야 한다.
# (일반적으로 이 launch는 robot_moveit demo launch 뒤에 include 하는 방식으로 사용)

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    validator_yaml = os.path.join(
        get_package_share_directory('grasp_moveit_config'),
        'config', 'moveit_validator.yaml')

    return LaunchDescription([
        Node(
            package='grasp_moveit_validator',
            executable='moveit_validator_node',
            name='grasp_moveit_validator',
            output='screen',
            parameters=[validator_yaml],
        ),
    ])
