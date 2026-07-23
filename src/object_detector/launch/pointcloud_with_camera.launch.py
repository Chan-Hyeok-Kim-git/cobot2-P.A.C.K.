from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    detector_share = Path(get_package_share_directory("object_detector"))
    model_path = LaunchConfiguration("model_path")
    sam_model_path = LaunchConfiguration("sam_model_path")
    device = LaunchConfiguration("device")
    start_camera = LaunchConfiguration("start_camera")
    start_rviz = LaunchConfiguration("start_rviz")

    camera = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="camera",
        name="camera",
        output="screen",
        condition=IfCondition(start_camera),
        parameters=[{
            "enable_color": True,
            "enable_depth": True,
            "enable_infra": False,
            "enable_infra1": False,
            "enable_infra2": False,
            "enable_motion": False,
            "enable_accel": False,
            "enable_gyro": False,
            "align_depth.enable": True,
            "pointcloud.enable": False,
            "rgb_camera.color_profile": "640x480x15",
            "depth_module.depth_profile": "640x480x15",
        }],
    )
    pointcloud = Node(
        package="object_detector",
        executable="yolo_pointcloud_mp",
        name="yolo_pointcloud",
        output="screen",
        parameters=[
            str(detector_share / "config" / "pointcloud.yaml"),
            {
                "model_path": model_path,
                "sam_model_path": sam_model_path,
                "device": ParameterValue(device, value_type=str),
            },
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="object_pointcloud_rviz",
        output="screen",
        arguments=["-d", str(detector_share / "config" / "object_pointcloud.rviz")],
        condition=IfCondition(start_rviz),
    )
    return LaunchDescription([
        DeclareLaunchArgument("model_path"),
        DeclareLaunchArgument("sam_model_path"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        camera,
        pointcloud,
        rviz,
    ])
