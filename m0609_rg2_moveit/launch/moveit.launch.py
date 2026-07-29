#
#  bringup_moveit.launch.py
#  dsr_bringup2_rviz.launch.py(로봇+그리퍼 bringup)와
#  moveit.launch.py(MoveIt+RViz)를 하나로 합친 통합 launch 파일.
#
#  실행:
#    ros2 launch <패키지명> bringup_moveit.launch.py mode:=virtual
#    ros2 launch <패키지명> bringup_moveit.launch.py mode:=real host:=<로봇IP>
#

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except OSError as exc:
        raise RuntimeError(f'YAML 파일을 열 수 없습니다: {absolute_file_path}') from exc


def generate_launch_description():
    # ------------------------------------------------------------------
    # Launch arguments
    # dsr_bringup2_rviz.launch.py의 인자 구성을 그대로 따른다.
    # (name은 프로젝트 전체가 'dsr01'을 고정으로 쓰므로 인자로 두지 않고
    #  아래에서 상수로 고정한다 — grasp.py/mi_node/move.py와 반드시 일치해야 함)
    # ------------------------------------------------------------------
    launch_args = [
        DeclareLaunchArgument('host',    default_value='127.0.0.1',        description='ROBOT_IP'),
        DeclareLaunchArgument('port',    default_value='12345',            description='ROBOT_PORT'),
        DeclareLaunchArgument('mode',    default_value='virtual',          description='OPERATION MODE: real | virtual'),
        DeclareLaunchArgument('rt_host', default_value='192.168.137.50',   description='ROBOT_RT_IP'),
    ]

    ROBOT_NAME = 'dsr01'   # 프로젝트 전역에서 고정으로 사용하는 네임스페이스
    ROBOT_MODEL = 'm0609'

    host    = LaunchConfiguration('host')
    port    = LaunchConfiguration('port')
    mode    = LaunchConfiguration('mode')
    rt_host = LaunchConfiguration('rt_host')

    is_virtual = PythonExpression(["'", mode, "' == 'virtual'"])
    update_rate = '100'

    # ------------------------------------------------------------------
    # 패키지 경로
    # ------------------------------------------------------------------
    bringup_pkg = get_package_share_directory('m0609_rg2_bringup')
    moveit_pkg  = get_package_share_directory('m0609_rg2_moveit')

    # ------------------------------------------------------------------
    # ★ robot_description ①: 시각화/MoveIt용 (팔+RG2 그리퍼 통합, static)
    # m0609_with_rg2.urdf.xacro는 dsr_description2/urdf/m0609.urdf(정적 URDF,
    # ros2_control 매크로 없음)를 include하므로 host/rt_host/port/mode 인자가
    # 필요 없다. robot_state_publisher, MoveIt(move_group), RViz가 사용한다.
    # ------------------------------------------------------------------
    integrated_xacro = os.path.join(bringup_pkg, 'urdf', 'm0609_with_rg2_camera.urdf.xacro')
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', integrated_xacro]),
            value_type=str,
        )
    }

    # ------------------------------------------------------------------
    # ★ robot_description ②: 하드웨어 인터페이스용 (팔만, ros2_control 매크로 포함)
    # dsr_description2/xacro/{model}.urdf.xacro는 host/rt_host/port/mode 등을
    # 반드시 필요로 하는 ros2_control 하드웨어 플러그인 매크로가 정의되어 있다.
    # dsr_bringup2_rviz.launch.py 원본과 동일한 인자 순서로 구성한다.
    # (이전 병합 시도에서 rt_host를 빠뜨려 하드웨어 인터페이스 연결이
    #  실패했을 가능성이 있었음 — 원본과 동일하게 반드시 포함)
    # ------------------------------------------------------------------
    hw_xacro = PathJoinSubstitution([
        FindPackageShare('dsr_description2'), 'xacro', ROBOT_MODEL,
    ])
    hw_robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        hw_xacro, '.urdf.xacro',
        ' name:=', ROBOT_NAME,
        ' host:=', host,
        ' rt_host:=', rt_host,
        ' port:=', port,
        ' mode:=', mode,
        ' model:=', ROBOT_MODEL,
        ' update_rate:=', update_rate,
    ])
    hw_robot_description = {
        'robot_description': ParameterValue(hw_robot_description_content, value_type=str)
    }

    # ------------------------------------------------------------------
    # MoveIt 설정 로드
    # ------------------------------------------------------------------
    srdf_file = os.path.join(moveit_pkg, 'config', 'm0609_rg2.srdf')
    with open(srdf_file, 'r', encoding='utf-8') as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    robot_description_kinematics = {
        'robot_description_kinematics': load_yaml('m0609_rg2_moveit', 'config/kinematics.yaml')
    }
    joint_limits = {
        'robot_description_planning': load_yaml('m0609_rg2_moveit', 'config/joint_limits.yaml')
    }
    ompl_planning_yaml = load_yaml('m0609_rg2_moveit', 'config/ompl_planning.yaml')
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_planning_yaml,
    }
    moveit_controllers_yaml = load_yaml('m0609_rg2_moveit', 'config/moveit_controllers.yaml')

    # ------------------------------------------------------------------
    # [virtual] DRCF 에뮬레이터
    # dsr_bringup2_rviz.launch.py 원본과 동일 — 정리(cleanup) 없이 바로 실행
    # ------------------------------------------------------------------
    run_emulator_node = Node(
        package='dsr_bringup2',
        executable='run_emulator',
        namespace=ROBOT_NAME,
        parameters=[
            {'name': ROBOT_NAME},
            {'rate': 100},
            {'standby': 5000},
            {'command': True},
            {'host': host},
            {'port': port},
            {'mode': mode},
            {'model': ROBOT_MODEL},
            {'gripper': 'none'},
            {'mobile': 'none'},
            {'rt_host': rt_host},
        ],
        condition=IfCondition(is_virtual),
        output='screen',
    )

    # ------------------------------------------------------------------
    # ros2_control (하드웨어 인터페이스)
    # ------------------------------------------------------------------
    robot_controllers = [
        PathJoinSubstitution([FindPackageShare('dsr_controller2'), 'config', 'dsr_update_rate.yaml']),
        PathJoinSubstitution([FindPackageShare('dsr_controller2'), 'config', 'dsr_controller2.yaml']),
    ]

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace=ROBOT_NAME,
        parameters=[hw_robot_description] + robot_controllers,
        output='both',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=ROBOT_NAME,
        arguments=['joint_state_broadcaster', '-c', 'controller_manager'],
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=ROBOT_NAME,
        arguments=['dsr_controller2', '-c', 'controller_manager'],
    )

    # joint_state_broadcaster 로드 완료 → dsr_controller2 로드
    delay_robot_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )

    # ------------------------------------------------------------------
    # RG2 그리퍼 노드 (mode에 따라 virtual/real 분기)
    # ------------------------------------------------------------------
    is_real = PythonExpression(["'", mode, "' == 'real'"])

    gripper_virtual_node = Node(
        package='m0609_rg2_bringup',
        executable='gripper_virtual_node.py',
        name='gripper_virtual_node',
        condition=IfCondition(is_virtual),
        output='screen',
    )

    onrobot_driver = Node(
        package='onrobot_rg_control',
        executable='OnRobotRGControllerServer',
        name='OnRobotRGControllerServer',
        output='screen',
        parameters=[{
            '/onrobot/control': 'modbus',
            '/onrobot/ip': '192.168.1.1',
            '/onrobot/port': 502,
            '/onrobot/changer_addr': 65,
            '/onrobot/gripper': 'rg2',
            '/onrobot/offset': 5,
        }],
        remappings=[('/joint_states', '/onrobot_joint_states')],
        condition=IfCondition(is_real),
    )

    gripper_joint_state_publisher = Node(
        package='m0609_rg2_bringup',
        executable='gripper_joint_state_publisher.py',
        name='gripper_joint_state_publisher',
        condition=IfCondition(is_real),
        output='screen',
    )

    # 팔 관절(/dsr01/joint_states)과 그리퍼 관절(/gripper_joint_states)을
    # /joint_states 하나로 합친다. 가상 모드에서 그리퍼 토픽이 없더라도
    # URDF의 기본값으로 rg2_finger_joint를 발행한다.
    joint_state_merger = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_merger',
        parameters=[
            robot_description,
            {
                'source_list': [
                    f'/{ROBOT_NAME}/joint_states',
                    '/gripper_joint_states',
                ],
                'rate': 50,
                'publish_default_positions': True,
                'use_mimic_tags': True,
            },
        ],
        output='screen',
    )

    # ------------------------------------------------------------------
    # 공용 TF / robot_state_publisher
    # ------------------------------------------------------------------
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'world', 'base_link'],
        output='log',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[robot_description],
        output='both',
    )

    # ------------------------------------------------------------------
    # MoveIt move_group
    # ------------------------------------------------------------------
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
        ],
    )

    # ------------------------------------------------------------------
    # RViz — MoveIt 플러그인 포함 (dsr_bringup2 기본 RViz 대신 이것 하나만 사용)
    # ------------------------------------------------------------------
    moveit_rviz_config = os.path.join(moveit_pkg, 'launch', 'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', moveit_rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
            joint_limits,
        ],
    )

    # dsr_bringup2_rviz.launch.py 원본 방식을 따라 컨트롤러가 완전히
    # 활성화된 뒤에 MoveIt(move_group)과 RViz를 띄운다.
    # (원본에서 rviz를 robot_controller_spawner 종료 후 띄우던 패턴과 동일)
    delay_moveit_after_robot_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[move_group_node, rviz_node],
        )
    )

    return LaunchDescription(
        launch_args + [
            run_emulator_node,
            control_node,
            joint_state_broadcaster_spawner,
            delay_robot_controller_spawner,
            gripper_virtual_node,
            onrobot_driver,
            gripper_joint_state_publisher,
            joint_state_merger,
            static_tf,
            robot_state_publisher,
            delay_moveit_after_robot_controller,
        ]
    )