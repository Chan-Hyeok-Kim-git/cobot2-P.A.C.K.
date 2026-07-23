#!/usr/bin/env python3
"""파지 파이프라인 테스트용 가짜 입력 노드.

기본 모드 ``exact_pose``는 아래 DSR 목표 자세를 ROS Pose로 변환해서
``/grasp/validated_grasp``에 한 번 발행한다.

    posx(436.04, -190.5, 314.18, 91.76, -124.53, -92.97)

이 모드는 ``cobot2_grasp.py``를 우회하고 ``cobot2_mi.py``부터 테스트한다.
PointCloud2만으로는 현재 cobot2_grasp.py의 고정 TOP/FRONT 접근축 때문에
임의의 A/B/C 자세를 정확히 만들 수 없기 때문이다.

기존 점군 시나리오도 유지한다.
- clear
- width_fail
- left_finger_collision
- body_collision

실행 예:
  ros2 run cobot2_control fake_grasp_cloud_publisher \
    --ros-args -p scenario:=exact_pose
"""

import math
import struct
from typing import Iterable

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

try:
    from cobot2_interfaces.msg import ValidatedGrasp
except ImportError:
    ValidatedGrasp = None


TARGET_TOPIC = "/grasp/target_cloud"
ENVIRONMENT_TOPIC = "/grasp/environment_cloud"
VALIDATED_GRASP_TOPIC = "/grasp/validated_grasp"
FRAME_ID = "base_link"

# 요청한 DSR posx: mm, deg
TARGET_POSX_MM_DEG = np.array(
    [568.92, -531.01, 275.0, 108.45, -120.44, -48.82],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# 자세 변환
# -----------------------------------------------------------------------------

def rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rot_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def dsr_zyz_to_rotation(a_deg: float, b_deg: float, c_deg: float) -> np.ndarray:
    """DSR posx의 Z-Y-Z Euler 각을 3x3 회전행렬로 변환한다."""
    a, b, c = np.deg2rad([a_deg, b_deg, c_deg])
    return rot_z(float(a)) @ rot_y(float(b)) @ rot_z(float(c))


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 회전행렬을 ROS quaternion(x, y, z, w)으로 변환한다."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-12:
        raise ValueError("유효한 quaternion을 계산할 수 없습니다.")
    q /= norm
    return tuple(float(v) for v in q)


def make_pose_stamped(
    node: Node,
    position_m: np.ndarray,
    rotation: np.ndarray,
) -> PoseStamped:
    pose = PoseStamped()
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.header.frame_id = FRAME_ID

    pose.pose.position.x = float(position_m[0])
    pose.pose.position.y = float(position_m[1])
    pose.pose.position.z = float(position_m[2])

    qx, qy, qz, qw = rotation_to_quaternion(rotation)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


# -----------------------------------------------------------------------------
# 점군 생성
# -----------------------------------------------------------------------------

def make_box_surface(
    center: Iterable[float],
    size: Iterable[float],
    points_per_axis: int = 12,
) -> np.ndarray:
    """직육면체 표면 점군을 생성한다. 단위는 meter."""
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    half = size / 2.0

    xs = np.linspace(-half[0], half[0], points_per_axis, dtype=np.float32)
    ys = np.linspace(-half[1], half[1], points_per_axis, dtype=np.float32)
    zs = np.linspace(-half[2], half[2], points_per_axis, dtype=np.float32)

    points = []
    for x in (-half[0], half[0]):
        for y in ys:
            for z in zs:
                points.append([x, y, z])
    for y in (-half[1], half[1]):
        for x in xs:
            for z in zs:
                points.append([x, y, z])
    for z in (-half[2], half[2]):
        for x in xs:
            for y in ys:
                points.append([x, y, z])

    return np.asarray(points, dtype=np.float32) + center


def make_dense_cluster(
    center: Iterable[float],
    size: Iterable[float],
    count: int = 80,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    return rng.uniform(center - size / 2.0, center + size / 2.0, (count, 3)).astype(
        np.float32
    )


def xyz_to_pointcloud2(node: Node, points: np.ndarray) -> PointCloud2:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    msg = PointCloud2()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = FRAME_ID
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = b"".join(
        struct.pack("<fff", float(x), float(y), float(z)) for x, y, z in points
    )
    return msg


class FakeGraspPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fake_grasp_cloud_publisher")

        self.declare_parameter("scenario", "exact_pose")
        self.declare_parameter("publish_period", 0.5)
        self.declare_parameter("pregrasp_distance", 0.100)
        self.declare_parameter("grasp_type", "FRONT")
        self.declare_parameter("grasp_joints_deg", [0.0] * 6)
        self.declare_parameter("pre_grasp_joints_deg", [0.0] * 6)

        self.scenario = str(self.get_parameter("scenario").value)
        period = float(self.get_parameter("publish_period").value)

        self.target_pub = self.create_publisher(PointCloud2, TARGET_TOPIC, 10)
        self.environment_pub = self.create_publisher(PointCloud2, ENVIRONMENT_TOPIC, 10)
        self.validated_pub = None
        self.target_points = np.empty((0, 3), dtype=np.float32)
        self.environment_points = np.empty((0, 3), dtype=np.float32)
        self._published_exact_pose = False

        if self.scenario == "exact_pose":
            if ValidatedGrasp is None:
                raise RuntimeError(
                    "cobot2_interfaces.msg.ValidatedGrasp를 import할 수 없습니다. "
                    "인터페이스 패키지를 먼저 빌드하고 source 하십시오."
                )
            self.validated_pub = self.create_publisher(
                ValidatedGrasp, VALIDATED_GRASP_TOPIC, 10
            )
            self.timer = self.create_timer(period, self.publish_exact_pose_once)
            self.get_logger().info(
                "exact_pose 대기 중: /grasp/validated_grasp 구독자가 연결되면 한 번 발행"
            )
        else:
            self.target_points, self.environment_points = self.build_cloud_scenario(
                self.scenario
            )
            self.timer = self.create_timer(period, self.publish_clouds)
            self.get_logger().info(
                f"가짜 점군 발행 시작 | scenario={self.scenario} | "
                f"target={len(self.target_points)}점 | "
                f"environment={len(self.environment_points)}점"
            )

    def publish_exact_pose_once(self) -> None:
        if self._published_exact_pose:
            return
        if self.validated_pub is None:
            return

        # cobot2_mi 구독자가 생기기 전에는 메시지를 버리지 않고 대기한다.
        if self.validated_pub.get_subscription_count() < 1:
            return

        x_mm, y_mm, z_mm, a_deg, b_deg, c_deg = TARGET_POSX_MM_DEG
        grasp_position = np.array(
            [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0],
            dtype=np.float64,
        )
        rotation = dsr_zyz_to_rotation(a_deg, b_deg, c_deg)

        # 접근 방향은 TCP local -Z, pre-grasp는 local +Z 방향으로 물러난 위치다.
        pregrasp_distance = float(self.get_parameter("pregrasp_distance").value)
        tool_positive_z_world = rotation[:, 2]
        pre_grasp_position = grasp_position + tool_positive_z_world * pregrasp_distance

        grasp_joints = [
            float(v) for v in self.get_parameter("grasp_joints_deg").value
        ]
        pre_grasp_joints = [
            float(v) for v in self.get_parameter("pre_grasp_joints_deg").value
        ]
        if len(grasp_joints) != 6 or len(pre_grasp_joints) != 6:
            raise ValueError("grasp_joints_deg와 pre_grasp_joints_deg는 각각 6개여야 합니다.")

        msg = ValidatedGrasp()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.grasp_type = str(self.get_parameter("grasp_type").value)
        msg.pre_grasp_pose = make_pose_stamped(self, pre_grasp_position, rotation)
        msg.grasp_pose = make_pose_stamped(self, grasp_position, rotation)
        msg.grasp_joints = grasp_joints
        msg.pre_grasp_joints = pre_grasp_joints

        self.validated_pub.publish(msg)
        self._published_exact_pose = True
        self.timer.cancel()

        q = msg.grasp_pose.pose.orientation
        self.get_logger().info(
            "ValidatedGrasp 한 번 발행 완료 | "
            f"grasp=({grasp_position[0]:.5f}, {grasp_position[1]:.5f}, "
            f"{grasp_position[2]:.5f}) m | "
            f"ABC=({a_deg:.2f}, {b_deg:.2f}, {c_deg:.2f}) deg | "
            f"quat=({q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}) | "
            f"pre=({pre_grasp_position[0]:.5f}, {pre_grasp_position[1]:.5f}, "
            f"{pre_grasp_position[2]:.5f}) m"
        )

        if all(abs(v) < 1.0e-9 for v in grasp_joints + pre_grasp_joints):
            self.get_logger().warn(
                "관절값 파라미터가 모두 0입니다. MoveIt trajectory가 정상 생성되는 "
                "경우에만 사용하십시오. MoveIt 없는 fallback 실행은 금지합니다."
            )

    def build_cloud_scenario(self, scenario: str) -> tuple[np.ndarray, np.ndarray]:
        # 요청한 목표 XYZ를 물체 중심으로 사용한다.
        # 단, 점군 모드에서는 cobot2_grasp.py가 자세를 다시 생성하므로 ABC는 정확히 유지되지 않는다.
        object_center = (TARGET_POSX_MM_DEG[:3] / 1000.0).astype(np.float32)

        if scenario == "clear":
            target = make_box_surface(object_center, [0.060, 0.040, 0.100])
            environment = np.empty((0, 3), dtype=np.float32)
        elif scenario == "width_fail":
            target = make_box_surface(object_center, [0.120, 0.110, 0.100])
            environment = np.empty((0, 3), dtype=np.float32)
        elif scenario == "left_finger_collision":
            target = make_box_surface(object_center, [0.060, 0.040, 0.100])
            obstacle_center = object_center + np.array([0.040, 0.000, 0.000])
            environment = make_dense_cluster(
                obstacle_center, [0.010, 0.040, 0.040]
            )
        elif scenario == "body_collision":
            target = make_box_surface(object_center, [0.060, 0.040, 0.100])
            obstacle_center = object_center + np.array([0.000, 0.060, 0.000])
            environment = make_dense_cluster(
                obstacle_center, [0.025, 0.070, 0.025]
            )
        else:
            raise ValueError(
                f"지원하지 않는 scenario={scenario!r}. "
                "exact_pose, clear, width_fail, left_finger_collision, "
                "body_collision 중 선택"
            )

        return target, environment

    def publish_clouds(self) -> None:
        self.target_pub.publish(xyz_to_pointcloud2(self, self.target_points))
        self.environment_pub.publish(
            xyz_to_pointcloud2(self, self.environment_points)
        )
        self.get_logger().info(
            f"점군 발행 | scenario={self.scenario} | "
            f"target={len(self.target_points)} | env={len(self.environment_points)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = FakeGraspPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        import traceback

        print(f"가짜 파지 노드 오류: {exc}")
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()