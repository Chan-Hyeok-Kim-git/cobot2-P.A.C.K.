#!/usr/bin/env python3

"""Transform AI PointCloud2 topics and object-position JSON into base_link.

Default inputs:
  /ai/background_points    sensor_msgs/msg/PointCloud2
  /ai/object_points        sensor_msgs/msg/PointCloud2
  /ai/objects_3d/json      std_msgs/msg/String

Default outputs:
  /ai/background_points_base
  /ai/object_points_base
  /ai/objects_3d/base_json

The JSON schema handled explicitly is:
  top-level frame_id, stamp, objects[]
  objects[].position_camera_xyz_m = [x, y, z]

For each valid object position the node preserves position_camera_xyz_m and adds:
  objects[].position_base_xyz_m = [x, y, z]
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class AiFrameTransformNode(Node):
    """Transforms two PointCloud2 streams and one JSON stream to a target frame."""

    def __init__(self) -> None:
        super().__init__("ai_frame_transform_node")

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("background_input_topic", "/ai/background_points")
        self.declare_parameter(
            "background_output_topic", "/ai/background_points_base"
        )
        self.declare_parameter("object_input_topic", "/ai/object_points")
        self.declare_parameter("object_output_topic", "/ai/object_points_base")
        self.declare_parameter("json_input_topic", "/ai/objects_3d/json")
        self.declare_parameter("json_output_topic", "/ai/objects_3d/base_json")
        self.declare_parameter("tf_timeout_sec", 0.5)
        self.declare_parameter("fallback_to_latest_tf", False)
        # 정지 상태 스캔에서는 메시지 시각의 TF를 기다리지 않고
        # TF 버퍼에 들어온 최신 변환을 즉시 사용한다.
        self.declare_parameter("use_latest_tf_only", True)
        self.declare_parameter("json_input_scale", 1.0)
        self.declare_parameter(
            "json_camera_position_key", "position_camera_xyz_m"
        )
        self.declare_parameter("json_base_position_key", "position_base_xyz_m")

        self.target_frame = self._normalize_frame(
            str(self.get_parameter("target_frame").value)
        )
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.fallback_to_latest_tf = bool(
            self.get_parameter("fallback_to_latest_tf").value
        )
        self.use_latest_tf_only = bool(
            self.get_parameter("use_latest_tf_only").value
        )
        self.json_input_scale = float(self.get_parameter("json_input_scale").value)
        self.json_camera_position_key = str(
            self.get_parameter("json_camera_position_key").value
        )
        self.json_base_position_key = str(
            self.get_parameter("json_base_position_key").value
        )

        if not self.target_frame:
            raise ValueError("target_frame must not be empty")
        if self.tf_timeout_sec <= 0.0 or not math.isfinite(self.tf_timeout_sec):
            raise ValueError("tf_timeout_sec must be a positive finite number")
        if self.json_input_scale <= 0.0 or not math.isfinite(self.json_input_scale):
            raise ValueError("json_input_scale must be a positive finite number")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # PointCloud/JSON 콜백을 TF 구독 콜백과 동시에 처리할 수 있도록
        # 별도의 reentrant callback group에 둔다.
        self.processing_callback_group = ReentrantCallbackGroup()

        # Input PointCloud2 topics are sensor streams, so subscribe with
        # BEST_EFFORT. This can receive from either BEST_EFFORT or RELIABLE
        # publishers without forcing the camera/perception pipeline to block.
        cloud_input_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Downstream processing nodes commonly subscribe with RELIABLE.
        # A BEST_EFFORT publisher cannot satisfy a RELIABLE subscriber, so the
        # transformed output clouds must use RELIABLE.
        cloud_output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # BEST_EFFORT subscription accepts JSON from either reliability mode.
        # Transformed JSON is an event/result stream, so publish it reliably.
        json_input_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        json_output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        background_input = str(
            self.get_parameter("background_input_topic").value
        )
        background_output = str(
            self.get_parameter("background_output_topic").value
        )
        object_input = str(self.get_parameter("object_input_topic").value)
        object_output = str(self.get_parameter("object_output_topic").value)
        json_input = str(self.get_parameter("json_input_topic").value)
        json_output = str(self.get_parameter("json_output_topic").value)

        self.background_pub = self.create_publisher(
            PointCloud2, background_output, cloud_output_qos
        )
        self.object_pub = self.create_publisher(
            PointCloud2, object_output, cloud_output_qos
        )
        self.json_pub = self.create_publisher(String, json_output, json_output_qos)

        self.background_sub = self.create_subscription(
            PointCloud2,
            background_input,
            lambda msg: self._cloud_callback(
                msg, self.background_pub, "background"
            ),
            cloud_input_qos,
            callback_group=self.processing_callback_group,
        )
        self.object_sub = self.create_subscription(
            PointCloud2,
            object_input,
            lambda msg: self._cloud_callback(msg, self.object_pub, "object"),
            cloud_input_qos,
            callback_group=self.processing_callback_group,
        )
        self.json_sub = self.create_subscription(
            String,
            json_input,
            self._json_callback,
            json_input_qos,
            callback_group=self.processing_callback_group,
        )

        self._last_warning_ns: Dict[str, int] = {}

        self.get_logger().info(
            "AI frame transformer started: "
            f"target_frame={self.target_frame}, "
            f"clouds=({background_input}, {object_input}), "
            f"json={json_input}, "
            f"json_position={self.json_camera_position_key}, "
            f"use_latest_tf_only={self.use_latest_tf_only}, "
            "executor_threads=2"
        )

    @staticmethod
    def _normalize_frame(frame_id: str) -> str:
        return frame_id.strip().lstrip("/")

    def _warn_throttled(
        self, key: str, message: str, period_sec: float = 2.0
    ) -> None:
        now_ns = self.get_clock().now().nanoseconds
        previous_ns = self._last_warning_ns.get(key, 0)
        if now_ns - previous_ns >= int(period_sec * 1e9):
            self.get_logger().warning(message)
            self._last_warning_ns[key] = now_ns

    def _lookup_transform(
        self,
        source_frame: str,
        stamp: Time,
    ) -> TransformStamped:
        source_frame = self._normalize_frame(source_frame)

        # 정지 상태 스캔용 모드. 정확한 메시지 시각의 TF를 기다리지 않고
        # 버퍼에 존재하는 최신 TF를 바로 사용한다. 긴 대기로 콜백이 밀리지
        # 않도록 latest lookup은 최대 0.05초만 기다린다.
        if self.use_latest_tf_only:
            latest_timeout_sec = min(self.tf_timeout_sec, 0.05)
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=latest_timeout_sec),
            )

        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException:
            if not self.fallback_to_latest_tf:
                raise

            self._warn_throttled(
                "latest_tf_fallback",
                "Exact-time TF lookup failed; latest TF is being used because "
                "fallback_to_latest_tf=true. Use this only while the robot is stopped.",
                period_sec=5.0,
            )
            latest_timeout_sec = min(self.tf_timeout_sec, 0.05)
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=latest_timeout_sec),
            )

    def _cloud_callback(self, msg: PointCloud2, publisher, label: str) -> None:
        source_frame = self._normalize_frame(msg.header.frame_id)
        if not source_frame:
            self._warn_throttled(
                f"{label}_empty_frame",
                f"The {label} PointCloud2 message has an empty header.frame_id.",
            )
            return

        if source_frame == self.target_frame:
            # Copy the normalized frame name without changing the cloud data.
            msg.header.frame_id = self.target_frame
            publisher.publish(msg)
            return

        try:
            transform = self._lookup_transform(
                source_frame,
                Time.from_msg(msg.header.stamp),
            )
            transformed = do_transform_cloud(msg, transform)
            transformed.header.frame_id = self.target_frame
            transformed.header.stamp = msg.header.stamp
            publisher.publish(transformed)
        except TransformException as exc:
            self._warn_throttled(
                f"{label}_tf",
                f"Cannot transform {label} cloud "
                f"{source_frame} -> {self.target_frame}: {exc}",
            )
        except Exception as exc:
            self._warn_throttled(
                f"{label}_cloud_error",
                f"Failed to transform {label} PointCloud2: "
                f"{type(exc).__name__}: {exc}",
            )

    def _json_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._warn_throttled(
                "json_decode",
                f"Invalid JSON on /ai/objects_3d/json: {exc}",
            )
            return

        if not isinstance(payload, dict):
            self._warn_throttled(
                "json_root",
                "The /ai/objects_3d/json root must be a JSON object.",
            )
            return

        source_frame = self._normalize_frame(str(payload.get("frame_id", "")))
        if not source_frame:
            self._warn_throttled(
                "json_frame",
                "The JSON message has no valid top-level frame_id.",
            )
            return

        stamp = self._json_stamp_to_time(payload.get("stamp"))
        if stamp is None:
            if not self.use_latest_tf_only and not self.fallback_to_latest_tf:
                self._warn_throttled(
                    "json_stamp",
                    "The JSON stamp is missing or invalid and "
                    "fallback_to_latest_tf=false.",
                )
                return
            stamp = Time()

        try:
            transform = self._lookup_transform(source_frame, stamp)
        except TransformException as exc:
            self._warn_throttled(
                "json_tf",
                f"Cannot transform JSON positions "
                f"{source_frame} -> {self.target_frame}: {exc}",
            )
            return

        output_payload = copy.deepcopy(payload)
        objects = output_payload.get("objects")
        if not isinstance(objects, list):
            self._warn_throttled(
                "json_objects",
                "The JSON field 'objects' must be an array.",
            )
            return

        transformed_count = 0
        skipped_count = 0

        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                skipped_count += 1
                continue

            camera_xyz = obj.get(self.json_camera_position_key)
            parsed_xyz = self._parse_xyz_list(camera_xyz)
            if parsed_xyz is None:
                skipped_count += 1
                obj["frame_transform_status"] = (
                    f"missing_or_invalid_{self.json_camera_position_key}"
                )
                continue

            try:
                x, y, z = self._transform_xyz(
                    parsed_xyz[0] * self.json_input_scale,
                    parsed_xyz[1] * self.json_input_scale,
                    parsed_xyz[2] * self.json_input_scale,
                    transform,
                )
            except ValueError as exc:
                skipped_count += 1
                obj["frame_transform_status"] = f"transform_error: {exc}"
                continue

            obj[self.json_base_position_key] = [x, y, z]
            obj["frame_transform_status"] = "success"
            transformed_count += 1

        output_payload["source_frame_id"] = source_frame
        output_payload["frame_id"] = self.target_frame
        output_payload["position_unit"] = "m"
        output_payload["transformed_object_count"] = transformed_count
        output_payload["skipped_object_count"] = skipped_count

        output = String()
        output.data = json.dumps(
            output_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.json_pub.publish(output)

        if transformed_count == 0:
            self._warn_throttled(
                "json_no_positions",
                f"No valid '{self.json_camera_position_key}' values were transformed.",
                period_sec=5.0,
            )

    @staticmethod
    def _json_stamp_to_time(stamp_value: Any) -> Optional[Time]:
        if not isinstance(stamp_value, dict):
            return None
        try:
            sec = int(stamp_value["sec"])
            nanosec = int(stamp_value["nanosec"])
        except (KeyError, TypeError, ValueError):
            return None

        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
            return None
        if sec == 0 and nanosec == 0:
            return None
        return Time(seconds=sec, nanoseconds=nanosec)

    @staticmethod
    def _parse_xyz_list(value: Any) -> Optional[Tuple[float, float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            xyz = (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(component) for component in xyz):
            return None
        return xyz

    @staticmethod
    def _transform_xyz(
        x: float,
        y: float,
        z: float,
        transform: TransformStamped,
    ) -> Tuple[float, float, float]:
        translation = transform.transform.translation
        quaternion = transform.transform.rotation

        rotation = AiFrameTransformNode._quaternion_to_matrix(
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
        source = np.array([x, y, z], dtype=np.float64)
        target = rotation @ source + np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float64,
        )
        return float(target[0]), float(target[1]), float(target[2])

    @staticmethod
    def _quaternion_to_matrix(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> np.ndarray:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            raise ValueError("TF contains a zero-length quaternion")

        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AiFrameTransformNode()

    # PointCloud 변환 콜백이 TF를 기다리거나 변환 연산을 수행하는 동안에도
    # /tf 및 /tf_static 구독 콜백을 처리할 수 있도록 2개 스레드를 사용한다.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()