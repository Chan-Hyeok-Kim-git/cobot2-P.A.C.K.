# ~/cobot_ws/src/cobot2_control/launch/cobot2_moveit_bringup.launch.py
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetRemap
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    original_launch = os.path.join(
        get_package_share_directory('m0609_rg2_moveit'),
        'launch',
        'demo.launch.py',  # ★ 위에서 find로 찾은 실제 파일명으로 교체
    )

    return LaunchDescription([
        # ★★★ 핵심: 여기서부터 실행되는 모든 노드(move_group 포함)에
        # /joint_states 구독을 /dsr01/joint_states로 강제 리매핑한다.
        # 원본 demo.launch.py 파일은 전혀 건드리지 않는다.
        SetRemap(src='/joint_states', dst='/dsr01/joint_states'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(original_launch)
        ),
    ])
