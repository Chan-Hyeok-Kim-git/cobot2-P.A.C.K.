import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():

    # -----------------------------------------------------------------------
    # 패키지 경로
    # -----------------------------------------------------------------------
    bringup_pkg = get_package_share_directory('m0609_rg2_bringup')
    moveit_pkg  = get_package_share_directory('m0609_rg2_moveit')

    # -----------------------------------------------------------------------
    # robot_description: 통합 xacro → URDF 변환
    # -----------------------------------------------------------------------
    xacro_file = os.path.join(bringup_pkg, 'urdf', 'm0609_with_rg2.urdf.xacro')
    robot_description = {
        'robot_description': Command(['xacro ', xacro_file])
    }

    # -----------------------------------------------------------------------
    # robot_description_semantic: SRDF 로드
    # -----------------------------------------------------------------------
    srdf_file = os.path.join(moveit_pkg, 'config', 'm0609_rg2.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = {
            'robot_description_semantic': f.read()
        }

    # -----------------------------------------------------------------------
    # kinematics: IK 솔버 설정 로드
    # -----------------------------------------------------------------------
    kinematics_yaml = load_yaml('m0609_rg2_moveit', 'config/kinematics.yaml')
    robot_description_kinematics = {
        'robot_description_kinematics': kinematics_yaml
    }

    # -----------------------------------------------------------------------
    # joint_limits: 관절 제한 로드
    # -----------------------------------------------------------------------
    joint_limits_yaml = load_yaml('m0609_rg2_moveit', 'config/joint_limits.yaml')
    joint_limits = {
        'robot_description_planning': joint_limits_yaml
    }

    # -----------------------------------------------------------------------
    # ompl_planning: 경로 계획 알고리즘 설정 로드
    # -----------------------------------------------------------------------
    ompl_planning_yaml = load_yaml('m0609_rg2_moveit', 'config/ompl_planning.yaml')

    # -----------------------------------------------------------------------
    # planning_pipelines: ompl을 기본 파이프라인으로 명시적 등록
    # -----------------------------------------------------------------------
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_planning_yaml,
    }

    # -----------------------------------------------------------------------
    # moveit_controllers: 컨트롤러 설정 로드
    # -----------------------------------------------------------------------
    moveit_controllers_yaml = load_yaml('m0609_rg2_moveit', 'config/moveit_controllers.yaml')

    # -----------------------------------------------------------------------
    # Static TF (world → base_link)
    # -----------------------------------------------------------------------
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'world', 'base_link']
    )

    # -----------------------------------------------------------------------
    # Robot State Publisher: URDF → TF 트리 퍼블리시
    # -----------------------------------------------------------------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[robot_description]
    )

    # -----------------------------------------------------------------------
    # Joint State Publisher: 시뮬레이션용 관절 상태 퍼블리시
    # -----------------------------------------------------------------------
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[robot_description]
    )

    # -----------------------------------------------------------------------
    # MoveGroup: 경로 계획 핵심 노드
    # -----------------------------------------------------------------------
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            joint_limits,
            planning_pipelines,
            moveit_controllers_yaml,
            {'use_sim_time': False},
        ]
    )

    # -----------------------------------------------------------------------
    # RViz: MoveIt 플러그인 포함한 시각화
    # -----------------------------------------------------------------------
    rviz_config_file = os.path.join(moveit_pkg, 'launch', 'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
            joint_limits,
        ]
    )

    return LaunchDescription([
        static_tf,
        robot_state_publisher,
        joint_state_publisher,
        move_group_node,
        rviz_node,
    ])
