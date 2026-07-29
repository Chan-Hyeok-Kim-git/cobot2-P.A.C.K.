#!/usr/bin/env python3
"""ROS 2 fake rope scene publisher for cobot2_grasp.py testing.

Publishes:
  /ai/object_points_base      sensor_msgs/PointCloud2
  /ai/background_points_base  sensor_msgs/PointCloud2
  /ai/objects_3d/base_json    std_msgs/String
  /voice/command              std_msgs/String (once, optional)

Default scene matches the supplied shelf.yaml:
  shelf panel top: z = 0.245 m
  rope size:       0.035 x 0.08 x 0.04 m
  rope center:     approximately [0.37, -0.48, 0.265] m

The rope is represented by a dense horizontal elliptical-cylinder cloud.
The surrounding shelf top plane is published as the background cloud.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String

try:
    import yaml
except ImportError:  # pragma: no cover - ROS desktop normally includes PyYAML
    yaml = None

try:
    from cobot2_interfaces.msg import GraspTarget
except ImportError:
    GraspTarget = None


OBJECT_TOPIC = "/ai/object_points_base"
BACKGROUND_TOPIC = "/ai/background_points_base"
BASE_JSON_TOPIC = "/ai/objects_3d/base_json"
VOICE_TOPIC = "/voice/command"
GRASP_TARGET_TOPIC = "/grasp/target"


def _pointcloud_xyz32(
    points: np.ndarray,
    frame_id: str,
    stamp: Time,
) -> PointCloud2:
    """Create an XYZ float32 PointCloud2 without sensor_msgs_py dependency."""
    xyz = np.asarray(points, dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {xyz.shape}")

    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = xyz.tobytes(order="C")
    msg.is_dense = bool(np.isfinite(xyz).all())
    return msg


def _rotation_z(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _make_rope_cloud(
    rng: np.random.Generator,
    center: np.ndarray,
    size: np.ndarray,
    yaw_rad: float,
    count: int,
) -> np.ndarray:
    """Generate a solid elongated elliptical-cylinder cloud."""
    length, width, height = map(float, size)

    local_x = rng.uniform(-0.5 * length, 0.5 * length, count)

    # Uniform samples inside an ellipse in the local Y-Z cross section.
    radius = np.sqrt(rng.uniform(0.0, 1.0, count))
    angle = rng.uniform(0.0, 2.0 * math.pi, count)
    local_y = 0.5 * width * radius * np.cos(angle)
    local_z = 0.5 * height * radius * np.sin(angle)

    local = np.column_stack((local_x, local_y, local_z))
    world = local @ _rotation_z(yaw_rad).T
    return world + center


def _make_shelf_background(
    rng: np.random.Generator,
    panel_center: np.ndarray,
    panel_size: np.ndarray,
    panel_top_z: float,
    rope_center: np.ndarray,
    rope_size: np.ndarray,
    rope_yaw_rad: float,
    count: int,
    exclusion_margin: float = 0.015,
) -> np.ndarray:
    """Generate shelf-top points around, but not beneath, the rope."""
    output: list[np.ndarray] = []
    required = int(count)
    rotation_inv = _rotation_z(-rope_yaw_rad)

    # Rejection sampling keeps the shelf cloud away from the rope footprint.
    while sum(len(batch) for batch in output) < required:
        batch_count = max(512, required)
        x = rng.uniform(
            panel_center[0] - panel_size[0] * 0.5,
            panel_center[0] + panel_size[0] * 0.5,
            batch_count,
        )
        y = rng.uniform(
            panel_center[1] - panel_size[1] * 0.5,
            panel_center[1] + panel_size[1] * 0.5,
            batch_count,
        )
        z = rng.normal(panel_top_z, 0.0005, batch_count)
        candidates = np.column_stack((x, y, z))

        relative = candidates - rope_center
        local = relative @ rotation_inv.T
        inside_padded_footprint = (
            np.abs(local[:, 0]) <= rope_size[0] * 0.5 + exclusion_margin
        ) & (
            np.abs(local[:, 1]) <= rope_size[1] * 0.5 + exclusion_margin
        )
        output.append(candidates[~inside_padded_footprint])

    return np.concatenate(output, axis=0)[:required]


def _load_panel_from_shelf_yaml(
    path_text: str,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not path_text:
        return None
    if yaml is None:
        raise RuntimeError("PyYAML이 없어 shelf_yaml을 읽을 수 없습니다.")

    expanded = os.path.expanduser(path_text)
    path = Path(expanded)
    if not path.is_file():
        raise FileNotFoundError(f"shelf_yaml 파일이 없습니다: {path}")

    with path.open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream)

    objects = root.get("collision_objects", []) if isinstance(root, dict) else []
    for item in objects:
        if item.get("id") == "shelf_back":
            center = np.asarray(item["center"], dtype=np.float64)
            size = np.asarray(item["size"], dtype=np.float64)
            if center.shape != (3,) or size.shape != (3,):
                raise ValueError("shelf_back center/size는 길이 3이어야 합니다.")
            return center, size

    raise ValueError("shelf_yaml에서 id='shelf_back' 항목을 찾지 못했습니다.")


class FakeRopeCloudPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fake_rope_cloud_publisher")

        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("class_name", "rope")
        self.declare_parameter("shelf_yaml", "")

        # 0.035 m x 0.08 m x 0.04 m.
        self.declare_parameter("size_x", 0.035)
        self.declare_parameter("size_y", 0.08)
        self.declare_parameter("size_z", 0.04)
        self.declare_parameter("yaw_deg", 25.0)

        # NaN means: derive from shelf_back center/top.
        self.declare_parameter("center_x", float("nan"))
        self.declare_parameter("center_y", float("nan"))
        self.declare_parameter("center_z", float("nan"))

        self.declare_parameter("object_point_count", 1200)
        self.declare_parameter("background_point_count", 3500)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("frame_noise_std_m", 0.0003)
        self.declare_parameter("random_seed", 7)

        self.declare_parameter("auto_command", True)
        self.declare_parameter("command_delay_sec", 1.0)

        self._frame_id = str(self.get_parameter("frame_id").value).strip().lstrip("/")
        self._class_name = str(self.get_parameter("class_name").value).strip()
        self._shelf_yaml = str(self.get_parameter("shelf_yaml").value).strip()

        self._size = np.array(
            [
                float(self.get_parameter("size_x").value),
                float(self.get_parameter("size_y").value),
                float(self.get_parameter("size_z").value),
            ],
            dtype=np.float64,
        )
        self._yaw_rad = math.radians(float(self.get_parameter("yaw_deg").value))
        self._object_count = int(self.get_parameter("object_point_count").value)
        self._background_count = int(self.get_parameter("background_point_count").value)
        self._rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._noise_std = float(self.get_parameter("frame_noise_std_m").value)
        self._auto_command = bool(self.get_parameter("auto_command").value)
        self._command_delay = float(self.get_parameter("command_delay_sec").value)

        if not self._class_name:
            raise ValueError("class_name은 비어 있을 수 없습니다.")
        if np.any(~np.isfinite(self._size)) or np.any(self._size <= 0.0):
            raise ValueError(f"size는 양수여야 합니다: {self._size.tolist()}")
        if self._object_count < 15:
            raise ValueError("object_point_count는 grasp 기본 cluster_min_points=15 이상이어야 합니다.")
        if self._background_count < 0:
            raise ValueError("background_point_count는 0 이상이어야 합니다.")
        if self._rate_hz <= 0.0:
            raise ValueError("publish_rate_hz는 0보다 커야 합니다.")

        panel_data = _load_panel_from_shelf_yaml(self._shelf_yaml)
        if panel_data is None:
            # Values from the supplied shelf.yaml.
            self._panel_center = np.array([0.37, -0.48, 0.24], dtype=np.float64)
            self._panel_size = np.array([1.01, 0.305, 0.01], dtype=np.float64)
            panel_source = "내장 shelf.yaml 기본값"
        else:
            self._panel_center, self._panel_size = panel_data
            panel_source = self._shelf_yaml

        self._panel_top_z = float(
            self._panel_center[2] + 0.5 * self._panel_size[2]
        )

        requested_center = np.array(
            [
                float(self.get_parameter("center_x").value),
                float(self.get_parameter("center_y").value),
                float(self.get_parameter("center_z").value),
            ],
            dtype=np.float64,
        )
        self._center = np.array(
            [
                requested_center[0] if math.isfinite(requested_center[0]) else self._panel_center[0],
                requested_center[1] if math.isfinite(requested_center[1]) else self._panel_center[1],
                requested_center[2] if math.isfinite(requested_center[2]) else self._panel_top_z + 0.5 * self._size[2],
            ],
            dtype=np.float64,
        )

        seed = int(self.get_parameter("random_seed").value)
        self._rng = np.random.default_rng(seed)
        self._object_base = _make_rope_cloud(
            self._rng,
            self._center,
            self._size,
            self._yaw_rad,
            self._object_count,
        )
        self._background_base = _make_shelf_background(
            self._rng,
            self._panel_center,
            self._panel_size,
            self._panel_top_z,
            self._center,
            self._size,
            self._yaw_rad,
            self._background_count,
        ) if self._background_count > 0 else np.zeros((0, 3), dtype=np.float64)

        self._object_pub = self.create_publisher(
            PointCloud2,
            OBJECT_TOPIC,
            qos_profile_sensor_data,
        )
        self._background_pub = self.create_publisher(
            PointCloud2,
            BACKGROUND_TOPIC,
            qos_profile_sensor_data,
        )
        self._json_pub = self.create_publisher(String, BASE_JSON_TOPIC, 10)
        self._command_pub = self.create_publisher(String, VOICE_TOPIC, 10)

        self._target_sub = None
        if GraspTarget is not None:
            self._target_sub = self.create_subscription(
                GraspTarget,
                GRASP_TARGET_TOPIC,
                self._on_grasp_target,
                10,
            )

        self._start_ns = self.get_clock().now().nanoseconds
        self._command_sent = False
        self._frame_count = 0
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_frame)

        self.get_logger().info(
            "Fake rope scene 시작 | "
            f"panel_source={panel_source} | "
            f"panel_top_z={self._panel_top_z:.4f}m | "
            f"rope_center={np.round(self._center, 4).tolist()}m | "
            f"rope_size={np.round(self._size, 4).tolist()}m | "
            f"yaw={math.degrees(self._yaw_rad):.1f}deg"
        )
        self.get_logger().info(
            f"publish: {OBJECT_TOPIC}, {BACKGROUND_TOPIC}, "
            f"{BASE_JSON_TOPIC}, {VOICE_TOPIC}"
        )

        if self._size[1] > 0.10:
            self.get_logger().warning(
                f"rope width={self._size[1]:.3f}m가 RG2 약 0.10m 개방폭보다 큽니다. "
                "PCA 테스트는 가능하지만 실제 파지 계획은 실패할 수 있습니다."
            )
        if self._size[1] > self._panel_size[1] or self._size[2] > 0.20:
            self.get_logger().warning(
                "지정한 물체 크기가 선반/눕힌 로프 시험치보다 큽니다. "
                "0.35 x 0.08 x 0.04m 사용을 권장합니다."
            )

    def _publish_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()

        if self._noise_std > 0.0:
            object_points = self._object_base + self._rng.normal(
                0.0,
                self._noise_std,
                self._object_base.shape,
            )
            background_points = self._background_base + self._rng.normal(
                0.0,
                self._noise_std,
                self._background_base.shape,
            )
        else:
            object_points = self._object_base
            background_points = self._background_base

        self._object_pub.publish(
            _pointcloud_xyz32(object_points, self._frame_id, stamp)
        )
        self._background_pub.publish(
            _pointcloud_xyz32(background_points, self._frame_id, stamp)
        )

        payload = {
            "frame_id": self._frame_id,
            "position_unit": "m",
            "objects": [
                {
                    "class_name": self._class_name,
                    "position_base_xyz_m": [float(value) for value in self._center],
                    "frame_transform_status": "success",
                }
            ],
        }
        self._json_pub.publish(String(data=json.dumps(payload)))

        self._frame_count += 1
        if self._frame_count % max(1, int(round(self._rate_hz * 2.0))) == 0:
            self.get_logger().info(
                f"fake frame={self._frame_count} | "
                f"object_points={len(object_points)} | "
                f"background_points={len(background_points)} | "
                f"center_z={self._center[2]:.4f}m"
            )

        self._maybe_send_command()

    def _maybe_send_command(self) -> None:
        if not self._auto_command or self._command_sent:
            return

        elapsed_sec = (
            self.get_clock().now().nanoseconds - self._start_ns
        ) / 1.0e9
        if elapsed_sec < self._command_delay:
            return

        if self._command_pub.get_subscription_count() < 1:
            return

        command = json.dumps([self._class_name])
        self._command_pub.publish(String(data=command))
        self._command_sent = True
        self.get_logger().warning(
            f"자동 voice command 발행: {command}"
        )

    def _on_grasp_target(self, msg) -> None:
        self.get_logger().warning(
            "/grasp/target 확인 | "
            f"target={msg.target_id} | class={msg.class_name} | "
            f"center=({msg.center.x:.4f}, {msg.center.y:.4f}, {msg.center.z:.4f})m | "
            f"top_z={msg.top_z:.4f}m | "
            f"required_width={msg.required_width * 1000.0:.1f}mm | "
            f"closing_axis=({msg.closing_axis.x:.4f}, "
            f"{msg.closing_axis.y:.4f}, {msg.closing_axis.z:.4f})"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeRopeCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()