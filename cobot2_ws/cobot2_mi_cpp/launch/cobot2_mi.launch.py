"""
cobot2_mi.launch.py — MoveGroupInterface(C++) 노드 실행
​
★ 수정: cobot2_mi_node가 MoveGroupInterface를 초기화하려면
  robot_description(URDF) + robot_description_semantic(SRDF)이
  자기 자신의 파라미터로 있어야 한다. moveit.launch.py의 move_group
  노드에만 이 파라미터가 있어서, 별도 프로세스인 cobot2_mi_node는
  못 받는다 → 이 launch 파일에서 직접 로드해서 넘겨준다.
​
실행 순서 (반드시 이 순서로):
  1. ros2 launch m0609_rg2_bringup bringup.launch.py
  2. ros2 launch m0609_rg2_moveit  moveit.launch.py
  3. 
ros2 control load_controller --set-state active \
  dsr_moveit_controller -c /dsr01/controller_manager ...
  4. ros2 launch cobot2_mi_cpp cobot2_mi.launch.py   ← 이 파일
  5. ros2 run cobot2_control cobot2_grasp
  6. ros2 run cobot2_control cobot2_move
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ── URDF 로드 (xacro) ────────────────────────────────────────────
    # ★ 수정: grasp_description 패키지는 존재하지 않음
    # 실제 URDF는 m0609_rg2_bringup에 있음
    bringup_pkg = get_package_share_directory('m0609_rg2_bringup')
    xacro_file = os.path.join(
        bringup_pkg, 'urdf', 'm0609_with_rg2.urdf.xacro')
    robot_description = {
        'robot_description': Command(['xacro ', xacro_file])
    }

    # ── SRDF 로드 (파일을 직접 읽어서 파라미터로 전달) ──────────────────
    moveit_pkg = get_package_share_directory('m0609_rg2_moveit')
    srdf_file = os.path.join(moveit_pkg, 'config', 'm0609_rg2.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = {
            'robot_description_semantic': f.read()
        }

    # ── kinematics.yaml 로드 ─────────────────────────────────────────
    kinematics_file = os.path.join(moveit_pkg, 'config', 'kinematics.yaml')
    import yaml
    with open(kinematics_file, 'r') as f:
        kinematics_yaml = yaml.safe_load(f)
    robot_description_kinematics = {
        'robot_description_kinematics': kinematics_yaml
    }

    return LaunchDescription([
        Node(
            package='cobot2_mi_cpp',
            executable='cobot2_mi_node',
            name='cobot2_mi',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
            ],
        ),
    ])







