from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    parent_frame = LaunchConfiguration("parent_frame")
    child_frame = LaunchConfiguration("child_frame")

    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")

    camera_roll = LaunchConfiguration("camera_roll")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_yaw = LaunchConfiguration("camera_yaw")

    camera_mount_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="link6_to_camera_tf",
        output="screen",
        arguments=[
            "--x",
            camera_x,
            "--y",
            camera_y,
            "--z",
            camera_z,
            "--roll",
            camera_roll,
            "--pitch",
            camera_pitch,
            "--yaw",
            camera_yaw,
            "--frame-id",
            parent_frame,
            "--child-frame-id",
            child_frame,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "parent_frame",
            default_value="link_6",
        ),
        DeclareLaunchArgument(
            "child_frame",
            default_value="camera_link",
        ),
        DeclareLaunchArgument(
            "camera_x",
            default_value="0.02",
        ),
        DeclareLaunchArgument(
            "camera_y",
            default_value="0.075",
        ),
        DeclareLaunchArgument(
            "camera_z",
            default_value="0.04",
        ),
        DeclareLaunchArgument(
            "camera_roll",
            default_value="-1.5708",
        ),
        DeclareLaunchArgument(
            "camera_pitch",
            default_value="-1.5708",
        ),
        DeclareLaunchArgument(
            "camera_yaw",
            default_value="0.0",
        ),
        camera_mount_tf,
    ])

