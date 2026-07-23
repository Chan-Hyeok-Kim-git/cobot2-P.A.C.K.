from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    detector_share = Path(get_package_share_directory("object_detector"))
    realsense_share = Path(get_package_share_directory("realsense2_camera"))

    model_path = LaunchConfiguration("model_path")
    start_camera = LaunchConfiguration("start_camera")
    start_viewer = LaunchConfiguration("start_viewer")
    confidence = LaunchConfiguration("confidence")
    device = LaunchConfiguration("device")

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(realsense_share / "launch" / "rs_launch.py")),
        condition=IfCondition(start_camera),
        launch_arguments={"enable_color": "true", "enable_depth": "false"}.items(),
    )
    detector = Node(
        package="object_detector",
        executable="yolo_detector",
        name="yolo_object_detector",
        output="screen",
        parameters=[
            str(detector_share / "config" / "detector.yaml"),
            {"model_path": model_path, "confidence": confidence, "device": device},
        ],
    )
    viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="yolo_detection_view",
        output="screen",
        arguments=["/ai/detections/image", "--on-top"],
        condition=IfCondition(start_viewer),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument("confidence", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="0"),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("start_viewer", default_value="true"),
            camera,
            detector,
            viewer,
        ]
    )
