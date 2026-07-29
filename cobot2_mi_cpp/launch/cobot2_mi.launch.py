import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup_pkg = get_package_share_directory("m0609_rg2_bringup")
    moveit_pkg = get_package_share_directory("m0609_rg2_moveit")

    xacro = os.path.join(
        bringup_pkg,
        "urdf",
        "m0609_with_rg2_camera.urdf.xacro",
    )
    robot_description = {
        "robot_description": ParameterValue(
            Command([FindExecutable(name="xacro"), " ", xacro]),
            value_type=str,
        )
    }

    with open(
        os.path.join(moveit_pkg, "config", "m0609_rg2.srdf"),
        "r",
        encoding="utf-8",
    ) as file:
        robot_description_semantic = {
            "robot_description_semantic": file.read()
        }

    with open(
        os.path.join(moveit_pkg, "config", "kinematics.yaml"),
        "r",
        encoding="utf-8",
    ) as file:
        robot_description_kinematics = {
            "robot_description_kinematics": yaml.safe_load(file)
        }

    return LaunchDescription(
        [
            Node(
                package="cobot2_mi_cpp",
                executable="cobot2_mi_node",
                name="cobot2_mi",
                output="screen",
                parameters=[
                    robot_description,
                    robot_description_semantic,
                    robot_description_kinematics,
                    {
                        "planning_group": "manipulator",
                        "eef_link": "rg2_tcp",
                        "shelf_yaml": os.path.join(
                            moveit_pkg,
                            "config",
                            "shelf.yaml",
                        ),
                        "planning_time": 3.0,
                        "planning_attempts": 3,
                        "velocity_scaling": 0.5,
                        "acceleration_scaling": 0.5,
                        "pre_grasp_distance": 0.08,
                        "cartesian_step": 0.005,
                        "cartesian_min_fraction": 0.95,
                        "low_center_z_threshold": 0.23,
                        "front_direction_x": 0.0,
                        "front_direction_y": -1.0,

                        # rg2_tcp convention from the supplied URDF.
                        # the frame axes in RViz.
                        "gripper_closing_local_axis": "local_y",
                        "tcp_approach_z_sign": 1.0,

                        # Reduce each shelf box's total dimensions.
                        # 0.010 m means 5 mm less on each side.
                        "shelf_shrink_x": 0.010,
                        "shelf_shrink_y": 0.010,
                        "shelf_shrink_z": 0.002,
                        "shelf_offset_x": 0.0,
                        "shelf_offset_y": 0.0,
                        "shelf_offset_z": 0.0,
                    },
                ],
                # Use merged /joint_states, not /dsr01/joint_states directly.
            )
        ]
    )