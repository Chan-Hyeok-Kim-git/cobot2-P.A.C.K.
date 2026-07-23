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
    device = LaunchConfiguration("device")
    start_camera = LaunchConfiguration("start_camera")
    start_rviz = LaunchConfiguration("start_rviz")
    background_exclusion_mode = LaunchConfiguration("background_exclusion_mode")
    target_class = LaunchConfiguration("target_class")

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
        executable="yolo_seg_pointcloud_mp",
        name="yolo_pointcloud",
        output="screen",
        parameters=[
            str(detector_share / "config" / "seg_pointcloud.yaml"),
            {
                "model_path": model_path,
                "device": ParameterValue(device, value_type=str),
                "background_exclusion_mode": ParameterValue(
                    background_exclusion_mode, value_type=str
                ),
                "target_class": ParameterValue(target_class, value_type=str),
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
        DeclareLaunchArgument(
            "model_path",
            description="Absolute path to a YOLO segmentation model",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cpu",
            description="Ultralytics inference device, for example cpu or 0",
        ),
        DeclareLaunchArgument(
            "start_camera",
            default_value="true",
            description="Start the RealSense node in this launch",
        ),
        DeclareLaunchArgument(
            "start_rviz",
            default_value="true",
            description="Start RViz2 with the object point-cloud config",
        ),
        DeclareLaunchArgument(
            "background_exclusion_mode",
            default_value="all",
            description="Background removal mode: all, class, or none",
        ),
        DeclareLaunchArgument(
            "target_class",
            default_value="",
            description="Class removed when background_exclusion_mode is class",
        ),
        camera,
        pointcloud,
        rviz,
    ])
