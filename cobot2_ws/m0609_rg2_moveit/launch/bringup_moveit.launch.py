import os
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
)
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
    # ------------------------------------------------------------------
    launch_args = [
        DeclareLaunchArgument(
            'mode',
            default_value='virtual',
            description='Operation mode: real | virtual',
        ),
        DeclareLaunchArgument(
            'host',
            default_value='127.0.0.1',
            description='Robot IP address',
        ),
        DeclareLaunchArgument(
            'port',
            default_value='12345',
            description='Robot port',
        ),
    ]

    mode = LaunchConfiguration('mode')
    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')

    is_real = PythonExpression(["'", mode, "' == 'real'"])
    is_virtual = PythonExpression(["'", mode, "' == 'virtual'"])

    # ------------------------------------------------------------------
    # Package paths and shared robot description
    # ------------------------------------------------------------------
    bringup_pkg = get_package_share_directory('m0609_rg2_bringup')
    moveit_pkg = get_package_share_directory('m0609_rg2_moveit')

    integrated_xacro = os.path.join(
        bringup_pkg,
        'urdf',
        'm0609_with_rg2.urdf.xacro',
    )

    integrated_robot_description = ParameterValue(
        Command(['xacro ', integrated_xacro]),
        value_type=str,
    )

    robot_description = {
        'robot_description': integrated_robot_description,
    }

    # ------------------------------------------------------------------
    # MoveIt configuration
    # ------------------------------------------------------------------
    srdf_file = os.path.join(moveit_pkg, 'config', 'm0609_rg2.srdf')
    with open(srdf_file, 'r', encoding='utf-8') as file:
        robot_description_semantic = {
            'robot_description_semantic': file.read(),
        }

    robot_description_kinematics = {
        'robot_description_kinematics': load_yaml(
            'm0609_rg2_moveit',
            'config/kinematics.yaml',
        )
    }

    joint_limits = {
        'robot_description_planning': load_yaml(
            'm0609_rg2_moveit',
            'config/joint_limits.yaml',
        )
    }

    ompl_planning_yaml = load_yaml(
        'm0609_rg2_moveit',
        'config/ompl_planning.yaml',
    )
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_planning_yaml,
    }

    moveit_controllers_yaml = load_yaml(
        'm0609_rg2_moveit',
        'config/moveit_controllers.yaml',
    )

    # ------------------------------------------------------------------
    # [virtual] DRCF emulator
    # ------------------------------------------------------------------
    emulator_cleanup = ExecuteProcess(
        cmd=[
            'bash',
            '-c',
            'docker rm -f dsr01_emulator 2>/dev/null || true',
        ],
        condition=IfCondition(is_virtual),
        output='log',
    )

    run_emulator_node = Node(
        package='dsr_bringup2',
        executable='run_emulator',
        namespace='dsr01',
        parameters=[
            {'name': 'dsr01'},
            {'host': host},
            {'port': port},
            {'mode': mode},
            {'model': 'm0609'},
            {'gripper': 'none'},
            {'mobile': 'none'},
        ],
        condition=IfCondition(is_virtual),
        output='screen',
    )

    start_emulator = RegisterEventHandler(
        OnProcessExit(
            target_action=emulator_cleanup,
            on_exit=[run_emulator_node],
        )
    )

    # ------------------------------------------------------------------
    # Doosan ros2_control description
    # ------------------------------------------------------------------
    doosan_xacro = PathJoinSubstitution([
        FindPackageShare('dsr_description2'),
        'xacro',
        'm0609.urdf.xacro',
    ])

    doosan_robot_description = Command([
        FindExecutable(name='xacro'),
        ' ',
        doosan_xacro,
        ' name:=dsr01',
        ' host:=',
        host,
        ' port:=',
        port,
        ' mode:=',
        mode,
        ' model:=m0609',
        ' update_rate:=100',
    ])

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace='dsr01',
        parameters=[
            {
                'robot_description': ParameterValue(
                    doosan_robot_description,
                    value_type=str,
                )
            },
            {'update_rate': 100},
            PathJoinSubstitution([
                FindPackageShare('dsr_controller2'),
                'config',
                'dsr_controller2.yaml',
            ]),
        ],
        output='both',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace='dsr01',
        arguments=[
            'joint_state_broadcaster',
            '-c',
            'controller_manager',
        ],
        output='screen',
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace='dsr01',
        arguments=[
            'dsr_controller2',
            '-c',
            'controller_manager',
        ],
        output='screen',
    )

    delay_robot_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )

    # ------------------------------------------------------------------
    # RG2 nodes
    # ------------------------------------------------------------------
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

    # 실제 로봇 관절과 그리퍼 관절을 하나의 /joint_states로 합친다.
    # MoveIt, RViz, robot_state_publisher는 이 토픽만 사용한다.
    joint_state_merger = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            robot_description,
            {
                'source_list': [
                    '/dsr01/joint_states',
                    '/gripper_joint_states',
                ]
            },
        ],
        output='screen',
    )

    # ------------------------------------------------------------------
    # Shared TF publishers
    # ------------------------------------------------------------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[robot_description],
        output='both',
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=[
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            '0.0',
            'world',
            'base_link',
        ],
        output='log',
    )

    # ------------------------------------------------------------------
    # MoveIt move_group
    # ------------------------------------------------------------------
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            joint_limits,
            planning_pipelines,
            moveit_controllers_yaml,
            {'use_sim_time': False},
        ],
        output='screen',
    )

    # ------------------------------------------------------------------
    # Single RViz: MoveIt RViz only
    # ------------------------------------------------------------------
    moveit_rviz_config = os.path.join(
        moveit_pkg,
        'launch',
        'moveit.rviz',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', moveit_rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
            joint_limits,
        ],
        output='log',
    )

    return LaunchDescription(
        launch_args
        + [
            emulator_cleanup,
            start_emulator,
            gripper_virtual_node,
            control_node,
            joint_state_broadcaster_spawner,
            delay_robot_controller,
            onrobot_driver,
            gripper_joint_state_publisher,
            joint_state_merger,
            robot_state_publisher,
            static_tf,
            move_group_node,
            rviz_node,
        ]
    )
