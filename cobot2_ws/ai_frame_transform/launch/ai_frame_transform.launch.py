from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    target_frame = LaunchConfiguration("target_frame")

    return LaunchDescription([
        DeclareLaunchArgument("target_frame", default_value="base_link"),
        DeclareLaunchArgument(
            "background_input_topic",
            default_value="/ai/background_points",
        ),
        DeclareLaunchArgument(
            "background_output_topic",
            default_value="/ai/background_points_base",
        ),
        DeclareLaunchArgument(
            "object_input_topic",
            default_value="/ai/object_points",
        ),
        DeclareLaunchArgument(
            "object_output_topic",
            default_value="/ai/object_points_base",
        ),
        DeclareLaunchArgument(
            "json_input_topic",
            default_value="/ai/objects_3d/json",
        ),
        DeclareLaunchArgument(
            "json_output_topic",
            default_value="/ai/objects_3d/base_json",
        ),
        DeclareLaunchArgument("tf_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("fallback_to_latest_tf", default_value="false"),
        DeclareLaunchArgument("json_input_scale", default_value="1.0"),

        Node(
            package="ai_frame_transform",
            executable="ai_frame_transform_node",
            name="ai_frame_transform_node",
            output="screen",
            parameters=[{
                "target_frame": target_frame,
                "background_input_topic": LaunchConfiguration(
                    "background_input_topic"
                ),
                "background_output_topic": LaunchConfiguration(
                    "background_output_topic"
                ),
                "object_input_topic": LaunchConfiguration("object_input_topic"),
                "object_output_topic": LaunchConfiguration("object_output_topic"),
                "json_input_topic": LaunchConfiguration("json_input_topic"),
                "json_output_topic": LaunchConfiguration("json_output_topic"),
                "tf_timeout_sec": ParameterValue(
                    LaunchConfiguration("tf_timeout_sec"), value_type=float
                ),
                "fallback_to_latest_tf": ParameterValue(
                    LaunchConfiguration("fallback_to_latest_tf"), value_type=bool
                ),
                "json_input_scale": ParameterValue(
                    LaunchConfiguration("json_input_scale"), value_type=float
                ),
            }],
        ),
    ])
