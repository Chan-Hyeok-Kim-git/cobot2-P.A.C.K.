#!/usr/bin/env python3
"""GraspValidatorNode 테스트용 가짜 PointCloud2 발행 노드.

발행 토픽
- /grasp/target_cloud
- /grasp/environment_cloud

시나리오
- clear: 정상 파지 가능한 물체 + 빈 환경
- width_fail: 그리퍼 최대 개폐폭보다 큰 물체
- left_finger_collision: 왼쪽 손가락 위치에 장애물 배치
- body_collision: 그리퍼 본체 위치에 장애물 배치

예:
  ros2 run <package_name> fake_grasp_cloud_publisher --ros-args -p scenario:=clear
"""

import struct
from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

TARGET_TOPIC = "/grasp/target_cloud"
ENVIRONMENT_TOPIC = "/grasp/environment_cloud"
FRAME_ID = "base_link"


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

    # x 고정 면 2개
    for x in (-half[0], half[0]):
        for y in ys:
            for z in zs:
                points.append([x, y, z])

    # y 고정 면 2개
    for y in (-half[1], half[1]):
        for x in xs:
            for z in zs:
                points.append([x, y, z])

    # z 고정 면 2개
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
    """충돌 검사용 조밀한 장애물 점군을 생성한다."""
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    return rng.uniform(center - size / 2.0, center + size / 2.0, (count, 3)).astype(
        np.float32
    )


def xyz_to_pointcloud2(node: Node, points: np.ndarray) -> PointCloud2:
    """Nx3 float32 배열을 xyz PointCloud2 메시지로 변환한다."""
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

    # 원본 validator가 '<f'로 읽으므로 little-endian float32로 고정한다.
    msg.data = b"".join(
        struct.pack("<fff", float(x), float(y), float(z)) for x, y, z in points
    )
    return msg


class FakeGraspCloudPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fake_grasp_cloud_publisher")

        self.declare_parameter("scenario", "clear")
        self.declare_parameter("publish_period", 1.0)
        # ★ 추가: 기본값을 "한 번만 발행"으로 변경.
        # 이전에는 publish_period(기본 1초)마다 계속 재발행해서,
        # cobot2_grasp.py가 같은 물체를 매번 "새 물체"로 착각하고
        # 처음부터 다시 계산 → cobot2_move 큐에 같은 동작이 계속
        # 쌓이는 문제가 있었다(로봇이 목표에 도달했다가 다시 같은
        # 동작을 무한 반복하는 것처럼 보였던 원인).
        # repeat:=true로 주면 기존처럼 반복 발행도 가능.
        self.declare_parameter("repeat", False)

        self.scenario = str(self.get_parameter("scenario").value)
        period = float(self.get_parameter("publish_period").value)
        self.repeat = bool(self.get_parameter("repeat").value)

        self.target_pub = self.create_publisher(PointCloud2, TARGET_TOPIC, 10)
        self.environment_pub = self.create_publisher(PointCloud2, ENVIRONMENT_TOPIC, 10)

        self.target_points, self.environment_points = self.build_scenario(self.scenario)

        if self.repeat:
            self.timer = self.create_timer(period, self.publish_clouds)
            self.get_logger().info(
                f"가짜 점군 발행 시작(반복 모드, {period}초 간격) | scenario={self.scenario} | "
                f"target={len(self.target_points)}점 | environment={len(self.environment_points)}점"
            )
        else:
            # 1회만 발행 — 로봇이 하나의 목표를 향해 끝까지 동작을
            # 완료할 수 있도록, 재발행으로 인한 중복 계산/명령을 없앤다.
            self.timer = self.create_timer(0.5, self._publish_once_then_stop)
            self.get_logger().info(
                f"가짜 점군 1회 발행 예정 | scenario={self.scenario} | "
                f"target={len(self.target_points)}점 | environment={len(self.environment_points)}점"
            )

    def _publish_once_then_stop(self) -> None:
        self.timer.cancel()
        self.publish_clouds()
        self.get_logger().info("1회 발행 완료 — 재발행 없음 (repeat:=true로 반복 가능)")

    def build_scenario(self, scenario: str) -> tuple[np.ndarray, np.ndarray]:
        # base_link 기준 예시 물체. 실제 로봇의 IK 가능 영역에 맞게 center를 수정할 수 있다.
        # ★ Global_ex 좌표 반영: posx(436.04, -190.5, 314.18, ...) [mm] -> [m]
        object_center = np.array([0.43604, -0.19050, 0.31418], dtype=np.float32)

        if scenario == "clear":
            # x=60 mm, y=40 mm, z=100 mm: FRONT 파지 시 약 60 mm 개폐폭 예상
            target = make_box_surface(object_center, [0.060, 0.040, 0.100])
            environment = np.empty((0, 3), dtype=np.float32)

        elif scenario == "width_fail":
            # FRONT closing 후보가 선택할 수평축 폭을 모두 110 mm 이상으로 구성
            target = make_box_surface(object_center, [0.120, 0.110, 0.100])
            environment = np.empty((0, 3), dtype=np.float32)

        elif scenario == "left_finger_collision":
            target = make_box_surface(object_center, [0.060, 0.040, 0.100])
            # clear 시 예상 FRONT 자세:
            # local y(closing)≈world x, local z≈world y.
            # 왼쪽 손가락은 물체 중심에서 world +x 쪽에 놓인다.
            obstacle_center = object_center + np.array([0.040, 0.000, 0.000])
            environment = make_dense_cluster(obstacle_center, [0.010, 0.040, 0.040])

        elif scenario == "body_collision":
            target = make_box_surface(object_center, [0.060, 0.040, 0.100])
            # FRONT에서 local +z(본체 방향)≈world +y.
            obstacle_center = object_center + np.array([0.000, 0.060, 0.000])
            environment = make_dense_cluster(obstacle_center, [0.025, 0.070, 0.025])

        else:
            raise ValueError(
                f"지원하지 않는 scenario={scenario!r}. "
                "clear, width_fail, left_finger_collision, body_collision 중 선택"
            )

        return target, environment

    def publish_clouds(self) -> None:
        self.target_pub.publish(xyz_to_pointcloud2(self, self.target_points))
        self.environment_pub.publish(xyz_to_pointcloud2(self, self.environment_points))
        self.get_logger().info(
            f"점군 발행 | scenario={self.scenario} | "
            f"target={len(self.target_points)} | env={len(self.environment_points)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = FakeGraspCloudPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"가짜 점군 노드 오류: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()