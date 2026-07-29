#!/usr/bin/env python3
"""Execute PlannedGrasp trajectories with DSR movesj and control the RG2."""

import json
import math
import time
import traceback
from typing import Any, Optional

import DR_init
import rclpy
from moveit_msgs.msg import RobotTrajectory
from rclpy.node import Node
from std_msgs.msg import String, Bool

from cobot2_interfaces.msg import PlannedGrasp

try:
    from .onrobot import RG
except (ImportError, ValueError):
    try:
        from onrobot import RG
    except ImportError as exc:
        RG = None
        RG_IMPORT_ERROR = exc
    else:
        RG_IMPORT_ERROR = None
else:
    RG_IMPORT_ERROR = None

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
PLANNED_TOPIC = "/moveit_grasp/planned"
RESULT_TOPIC = "/execution_result"
JOINTS = [f"joint_{index}" for index in range(1, 7)]
DSR: dict[str, Any] = {}


# ================================================================
# 사용자가 직접 입력할 하드코딩 자세
# ================================================================
# 관절각 단위: degree
# 예: SCAN_JOINT_DEG = [0.0, -20.0, 90.0, 0.0, 110.0, 0.0]
SCAN_JOINT_DEG = [14.49, -1.65, 93.18, -51.59, 99.09, 196.65]
SCAN_JOINT_DEG2 = [19.37, 5.52, 65.34, -51.23, 133.53, 196.67]
HOME_JOINT_DEG = [0, 0, 90, 0, 90, 180]

# 작업공간 단위: X/Y/Z=mm, A/B/C=degree
BAG_POSE_MM_DEG = [795.130, -24.170, 190.620, 150.25, -155.11, -14.94]


LIFT_AFTER_GRASP_MM = 80.0
Y_AFTER_GRASP_MM = 200.0

SCAN_JOINT_VEL = 30.0
SCAN_JOINT_ACC = 60.0
SCAN_SETTLE_SEC = 1.0

LIFT_LINEAR_VEL = 50.0
LIFT_LINEAR_ACC = 100.0
BAG_LINEAR_VEL = 100.0
BAG_LINEAR_ACC = 200.0

# 여러 물품을 순차 처리하며 가방 위치에서 선반이 보이지 않는 경우 True.
RETURN_TO_SCAN_AFTER_PLACE = True


class RobotExecutor(Node):
    def __init__(self) -> None:
        super().__init__("cobot2_move", namespace=ROBOT_ID)

        self.i = 0

        self.work_pub = self.create_publisher(Bool, "/robot/working", 10)
        self.finish_pub = self.create_publisher(Bool, "/robot/finished", 10)

        self._execute_enabled = bool(
            self.declare_parameter("execute_enabled", False).value
        )
        self._velocity = float(
            self.declare_parameter("movesj_velocity", 30.0).value
        )
        self._acceleration = float(
            self.declare_parameter("movesj_acceleration", 60.0).value
        )
        self._start_tolerance = float(
            self.declare_parameter("movesj_start_tolerance_deg", 3.0).value
        )
        self._min_waypoint_delta = float(
            self.declare_parameter(
                "movesj_min_waypoint_delta_deg",
                0.02,
            ).value
        )
        self._verify_alignment = bool(
            self.declare_parameter("verify_pca_alignment", True).value
        )
        self._alignment_tolerance = float(
            self.declare_parameter(
                "pca_alignment_tolerance_deg",
                5.0,
            ).value
        )

        self._enable_gripper = bool(
            self.declare_parameter("enable_gripper", True).value
        )
        self._gripper_ip = str(
            self.declare_parameter("onrobot_ip", "192.168.1.1").value
        )
        self._gripper_port = int(
            self.declare_parameter("onrobot_port", 502).value
        )
        self._gripper_type = str(
            self.declare_parameter("onrobot_gripper", "rg2").value
        )
        self._gripper_force = int(
            self.declare_parameter("gripper_force_raw", 400).value
        )
        self._open_margin = float(
            self.declare_parameter("gripper_open_margin_mm", 20.0).value
        )
        self._close_offset = float(
            self.declare_parameter("gripper_close_offset_mm", 10.0).value
        )
        self._min_clearance = float(
            self.declare_parameter(
                "gripper_min_open_clearance_mm",
                5.0,
            ).value
        )
        self._gripper_timeout = float(
            self.declare_parameter("gripper_timeout_sec", 5.0).value
        )
        self._poll_period = float(
            self.declare_parameter(
                "gripper_poll_period_sec",
                0.05,
            ).value
        )
        self._require_grip = bool(
            self.declare_parameter("require_grip_detected", True).value
        )

        if self._velocity <= 0.0 or self._acceleration <= 0.0:
            raise ValueError("movesj velocity/acceleration must be positive")
        if min(
            self._start_tolerance,
            self._min_waypoint_delta,
            self._alignment_tolerance,
        ) < 0.0:
            raise ValueError("tolerances must be non-negative")

        self._pending: Optional[PlannedGrasp] = None
        self._last_key = None
        self._busy = False
        self._rg2 = None

        self.create_subscription(
            PlannedGrasp,
            PLANNED_TOPIC,
            self._on_plan,
            10,
        )
        self._result_pub = self.create_publisher(String, RESULT_TOPIC, 10)

        if self._enable_gripper:
            self._connect_gripper()

        self.get_logger().info(
            "RobotExecutor 준비 완료 | "
            f"execute_enabled={self._execute_enabled} | "
            f"vel={self._velocity:.1f} | acc={self._acceleration:.1f} | "
            "RG2_Y=closing | RG2_Z=approach | "
            f"gripper={self._enable_gripper}"
        )
        if not self._execute_enabled:
            self.get_logger().warning(
                "실행 비활성화: --ros-args -p execute_enabled:=true"
            )

    def _on_plan(self, msg: PlannedGrasp) -> None:
        if not msg.success:
            self.get_logger().warning(
                "실패 계획 수신 — 실행 안 함 | "
                f"target={msg.target_id} | reason={msg.failure_reason}"
            )
            return
        if not self._execute_enabled:
            self.get_logger().warning(
                "성공 계획 수신, execute_enabled=False | "
                f"target={msg.target_id} | candidate={msg.candidate_id}"
            )
            return

        key = (
            int(msg.target_id),
            int(msg.candidate_id),
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
        )
        if key == self._last_key:
            self.get_logger().warning(f"중복 계획 무시 | key={key}")
            return
        if self._busy or self._pending is not None:
            self.get_logger().warning(
                "실행기 사용 중 — 계획 거부 | "
                f"target={msg.target_id} | candidate={msg.candidate_id}"
            )
            self._publish_result(msg, False, "EXECUTOR_BUSY")
            return

        self._last_key = key
        self._pending = msg
        self.get_logger().info(
            "계획 등록 | "
            f"target={msg.target_id} | class={msg.class_name} | "
            f"candidate={msg.candidate_id} | type={msg.grasp_type} | "
            f"closing=({msg.closing_axis.x:.4f}, "
            f"{msg.closing_axis.y:.4f}, {msg.closing_axis.z:.4f}) | "
            f"width={float(msg.required_width) * 1000.0:.1f}mm | "
            f"pre_points={len(msg.to_pre_grasp.joint_trajectory.points)} | "
            f"approach_points={len(msg.approach.joint_trajectory.points)}"
        )

    def process_pending(self) -> None:
        if self._busy or self._pending is None:
            return
        msg = self._pending
        self._pending = None
        self._busy = True
        try:
            self._execute(msg)
            self._publish_result(
                msg,
                True,
                "TRAJECTORY_EXECUTION_SUCCEEDED",
            )
        except Exception as exc:
            self.get_logger().error(
                "실행 실패 | "
                f"target={msg.target_id} | candidate={msg.candidate_id} | "
                f"error={exc}\n{traceback.format_exc()}"
            )
            self._publish_result(msg, False, str(exc))
        finally:
            self._busy = False

    def _execute(self, msg: PlannedGrasp) -> None:
        if self._verify_alignment:
            self._verify_pca_alignment(msg)

        width_mm = float(msg.required_width) * 1000.0
        if self._enable_gripper:
            open_mm, close_mm = self._gripper_widths(width_mm)
            self.get_logger().info(f"[1/7] RG2 열기 → {open_mm:.1f}mm")
            self._move_gripper(open_mm)
        else:
            close_mm = 0.0
            self.get_logger().info("[1/7] RG2 비활성화 — 열기 생략")

        self.get_logger().info("[2/7] 현재 위치 → pre-grasp")
        self._execute_trajectory(msg.to_pre_grasp, "to_pre_grasp")

        self.get_logger().info("[3/7] pre-grasp → grasp")
        self._execute_trajectory(msg.approach, "approach")

        if self._enable_gripper:
            self.get_logger().info(f"[4/7] RG2 닫기 → {close_mm:.1f}mm")
            self._move_gripper(close_mm)
            detected, actual_mm = self._read_grip_state()
            self.get_logger().info(
                "RG2 상태 | "
                f"grip_detected={detected} | actual_width={actual_mm:.1f}mm"
            )
            # if self._require_grip and not detected:
            #     raise RuntimeError(
            #         f"GRIP_NOT_DETECTED:actual_width={actual_mm:.1f}mm"
            #     )
        else:
            self.get_logger().info("[4/7] RG2 비활성화 — 닫기/파지검사 생략")

        self.get_logger().info(
            f"[5/7] 파지 후 base_link +Z로 {LIFT_AFTER_GRASP_MM:.1f}mm 상승"
        )
        self._lift_after_grasp()
        self.move_to_scan_pose()

        self.get_logger().info(
            f"[6/7] 가방 위치로 movel 이동 | pose={BAG_POSE_MM_DEG}"
        )
        self._move_to_bag()

        if self._enable_gripper:
            self.get_logger().info("[7/7] 가방 위치에서 RG2 완전 개방")
            self._open_gripper_fully()
        else:
            self.get_logger().info("[7/7] RG2 비활성화 — 놓기 생략")

        if RETURN_TO_SCAN_AFTER_PLACE:
            self.get_logger().info("다음 물품 탐색을 위해 스캔 자세로 복귀")
            self.move_to_scan_pose()
            self.i += 1

        self.get_logger().info("파지·수납 완료 — 가방 위치에서 물체 해제")
        if self.i == 4:
            self.publish_working(False)
            self.publish_finished(True)

        


    def move_to_scan_pose(self) -> None:
        """시작 또는 다음 작업 전에 하드코딩된 관절 스캔 자세로 이동한다."""
        if not self._execute_enabled:
            self.get_logger().warning(
                "execute_enabled=False — 스캔 자세 이동 생략"
            )
            return
        if not DSR:
            raise RuntimeError("SCAN_MOVE:DSR_NOT_INITIALIZED")
        if HOME_JOINT_DEG is None:
            self.get_logger().warning(
                "SCAN_JOINT_DEG가 None — 파일 상단에 6축 관절각을 입력해야 함"
            )
            return

        scan = self._validated_pose(HOME_JOINT_DEG, "SCAN_JOINT_DEG")
        self.get_logger().info(
            "스캔 자세 movej 시작 | "
            f"joint_deg={[round(value, 3) for value in scan]} | "
            f"vel={SCAN_JOINT_VEL:.1f} | acc={SCAN_JOINT_ACC:.1f}"
        )
        result = DSR["movej"](
            DSR["posj"](*scan),
            vel=SCAN_JOINT_VEL,
            acc=SCAN_JOINT_ACC,
        )
        self._check_motion_result(result, "SCAN_MOVEJ_FAILED")
        if SCAN_SETTLE_SEC > 0.0:
            time.sleep(SCAN_SETTLE_SEC)
        self.get_logger().info(
            f"스캔 자세 도착 — 카메라 안정화 {SCAN_SETTLE_SEC:.1f}s 완료"
        )

    def move_to_scan_pose2(self) -> None:
            """시작 또는 다음 작업 전에 하드코딩된 관절 스캔 자세로 이동한다."""
            if not self._execute_enabled:
                self.get_logger().warning(
                    "execute_enabled=False — 스캔 자세 이동 생략"
                )
                return
            if not DSR:
                raise RuntimeError("SCAN_MOVE:DSR_NOT_INITIALIZED")
            if SCAN_JOINT_DEG2 is None:
                self.get_logger().warning(
                    "SCAN_JOINT_DEG2가 None — 파일 상단에 6축 관절각을 입력해야 함"
                )
                return
    
            scan2 = self._validated_pose(SCAN_JOINT_DEG2, "SCAN_JOINT_DEG2")
            self.get_logger().info(
                "스캔 자세 movej 시작 | "
                f"joint_deg={[round(value, 3) for value in scan2]} | "
                f"vel={SCAN_JOINT_VEL:.1f} | acc={SCAN_JOINT_ACC:.1f}"
            )
            result = DSR["movej"](
                DSR["posj"](*scan2),
                vel=10,
                acc=30,
            )
            self._check_motion_result(result, "SCAN_MOVEJ_FAILED")
            if SCAN_SETTLE_SEC > 0.0:
                time.sleep(SCAN_SETTLE_SEC)
            self.get_logger().info(
                f"스캔 자세 도착 — 카메라 안정화 {SCAN_SETTLE_SEC:.1f}s 완료"
            )
        

    def _lift_after_grasp(self) -> None:
        """현재 자세에서 base_link +Z 방향으로 50mm 상대 직선 이동한다."""
        if not DSR:
            raise RuntimeError("LIFT_MOVE:DSR_NOT_INITIALIZED")

        result = DSR["movel"](
            DSR["posx"](0.0, 0.0, LIFT_AFTER_GRASP_MM, 0.0, 0.0, 0.0),
            vel=LIFT_LINEAR_VEL,
            acc=LIFT_LINEAR_ACC,
            ref=DSR["DR_BASE"],
            mod=DSR["DR_MV_MOD_REL"],
        )
        self._check_motion_result(result, "LIFT_MOVEL_FAILED")
        self.get_logger().info(
            f"상대 상승 완료 | base_link +Z={LIFT_AFTER_GRASP_MM:.1f}mm"
        )

    def _x_after_grasp(self) -> None:
            """현재 자세에서 base_link +x 방향으로 50mm 상대 직선 이동한다."""
            if not DSR:
                raise RuntimeError("Y_MOVE:DSR_NOT_INITIALIZED")
    
            result = DSR["movel"](
                DSR["posx"](0.0, 0.0, Y_AFTER_GRASP_MM, 0.0, 0.0, 0.0),
                vel=LIFT_LINEAR_VEL,
                acc=LIFT_LINEAR_ACC,
                ref=DSR["DR_BASE"],
                mod=DSR["DR_MV_MOD_REL"],
            )
            self._check_motion_result(result, "Y_MOVEL_FAILED")
            self.get_logger().info(
                f"상대 상승 완료 | base_link +Z={LIFT_AFTER_GRASP_MM:.1f}mm"
            )

    def _move_to_bag(self) -> None:
        """하드코딩된 base 좌표계 가방 posx로 절대 직선 이동한다."""
        if not DSR:
            raise RuntimeError("BAG_MOVE:DSR_NOT_INITIALIZED")
        if BAG_POSE_MM_DEG is None:
            raise RuntimeError(
                "BAG_POSE_MM_DEG_NOT_CONFIGURED:"
                "파일 상단에 [X,Y,Z,A,B,C]를 입력해야 함"
            )

        bag = self._validated_pose(BAG_POSE_MM_DEG, "BAG_POSE_MM_DEG")
        result = DSR["movel"](
            DSR["posx"](*bag),
            vel=BAG_LINEAR_VEL,
            acc=BAG_LINEAR_ACC,
            ref=DSR["DR_BASE"],
            mod=DSR["DR_MV_MOD_ABS"],
        )
        self._check_motion_result(result, "BAG_MOVEL_FAILED")
        self.get_logger().info(
            "가방 위치 도착 | "
            f"posx={[round(value, 3) for value in bag]}"
        )

    def _open_gripper_fully(self) -> None:
        self._require_gripper()
        max_width_mm = float(self._rg2.max_width) / 10.0
        self._move_gripper(max_width_mm)

    @staticmethod
    def _validated_pose(values, name: str) -> list[float]:
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise RuntimeError(f"{name}:EXPECTED_6_VALUES")
        output = [float(value) for value in values]
        if not all(math.isfinite(value) for value in output):
            raise RuntimeError(f"{name}:NON_FINITE_VALUE")
        return output

    @staticmethod
    def _check_motion_result(result, error: str) -> None:
        if isinstance(result, (int, float)) and result < 0:
            raise RuntimeError(f"{error}:return={result}")

    def _verify_pca_alignment(self, msg: PlannedGrasp) -> None:
        closing = self._normalize(
            [
                float(msg.closing_axis.x),
                float(msg.closing_axis.y),
                float(msg.closing_axis.z),
            ],
            "PCA_CLOSING_AXIS_ZERO",
        )
        q = msg.grasp_pose.pose.orientation
        x, y, z, w = self._normalize(
            [float(q.x), float(q.y), float(q.z), float(q.w)],
            "GRASP_QUATERNION_ZERO",
        )
        local_y = self._normalize(
            [
                2.0 * (x * y - z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + x * w),
            ],
            "GRASP_TCP_LOCAL_Y_ZERO",
        )
        dot = abs(sum(a * b for a, b in zip(closing, local_y)))
        error_deg = math.degrees(math.acos(max(0.0, min(1.0, dot))))
        self.get_logger().info(
            "PCA/그리퍼 방향 검사 | "
            f"closing={[round(v, 4) for v in closing]} | "
            f"RG2_local_Y={[round(v, 4) for v in local_y]} | "
            f"error={error_deg:.2f}deg"
        )
        if error_deg > self._alignment_tolerance:
            raise RuntimeError(
                "PCA_GRIPPER_AXIS_MISMATCH:"
                f"error={error_deg:.2f}:limit={self._alignment_tolerance:.2f}"
            )

    @staticmethod
    def _normalize(values, error: str):
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1.0e-9:
            raise RuntimeError(error)
        return [value / norm for value in values]

    def _execute_trajectory(
        self,
        trajectory: RobotTrajectory,
        label: str,
    ) -> None:
        if not DSR:
            raise RuntimeError(f"{label}:DSR_NOT_INITIALIZED")

        jt = trajectory.joint_trajectory
        if not jt.joint_names or not jt.points:
            raise RuntimeError(f"{label}:EMPTY_TRAJECTORY")
        indices = {name: index for index, name in enumerate(jt.joint_names)}
        missing = [name for name in JOINTS if name not in indices]
        if missing:
            raise RuntimeError(f"{label}:REQUIRED_JOINTS_MISSING:{missing}")

        raw = []
        for point_index, point in enumerate(jt.points):
            if len(point.positions) != len(jt.joint_names):
                raise RuntimeError(
                    f"{label}:POINT_SIZE_MISMATCH:index={point_index}"
                )
            waypoint = []
            for name in JOINTS:
                value = float(point.positions[indices[name]])
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"{label}:NON_FINITE_JOINT:point={point_index}:joint={name}"
                    )
                waypoint.append(math.degrees(value))
            raw.append(waypoint)

        waypoints = self._filter_waypoints(raw)
        self._check_start(waypoints[0], label)
        dsr_waypoints = [DSR["posj"](*waypoint) for waypoint in waypoints]
        duration = jt.points[-1].time_from_start
        duration_sec = float(duration.sec) + float(duration.nanosec) * 1e-9

        self.get_logger().info(
            f"{label} movesj 시작 | raw={len(raw)} | used={len(waypoints)} | "
            f"vel={self._velocity:.1f} | acc={self._acceleration:.1f} | "
            f"moveit_duration={duration_sec:.3f}s"
        )
        self.publish_finished(False)
        self.publish_working(True)
        result = DSR["movesj"](
            dsr_waypoints,
            vel=self._velocity,
            acc=self._acceleration,
        )
        if isinstance(result, (int, float)) and result < 0:
            raise RuntimeError(f"{label}:MOVESJ_FAILED:return={result}")
        self.get_logger().info(
            f"{label} movesj 완료 | waypoints={len(waypoints)}"
        )

    def _filter_waypoints(self, waypoints):
        if not waypoints:
            raise RuntimeError("NO_WAYPOINT")
        filtered = [waypoints[0]]
        for waypoint in waypoints[1:-1]:
            delta = max(
                abs(current - previous)
                for current, previous in zip(waypoint, filtered[-1])
            )
            if delta >= self._min_waypoint_delta:
                filtered.append(waypoint)
        if len(waypoints) > 1 and filtered[-1] != waypoints[-1]:
            filtered.append(waypoints[-1])
        return filtered

    def _check_start(self, first_waypoint, label: str) -> None:
        current = DSR["get_current_posj"]()
        if isinstance(current, tuple):
            current = current[0]
        current = [float(value) for value in list(current)[:6]]
        if len(current) != 6:
            raise RuntimeError(f"{label}:CURRENT_POSJ_SIZE_INVALID")

        errors = []
        for index, (actual, target) in enumerate(zip(current, first_waypoint)):
            error = target - actual
            if index in (0, 3, 5):
                error = (error + 180.0) % 360.0 - 180.0
            errors.append(abs(error))
        max_error = max(errors)
        self.get_logger().info(
            f"{label} 시작 상태 | max_error={max_error:.3f}deg | "
            f"per_joint={[round(value, 3) for value in errors]}"
        )
        if max_error > self._start_tolerance:
            raise RuntimeError(
                f"{label}:START_STATE_MISMATCH:max={max_error:.3f}:"
                f"limit={self._start_tolerance:.3f}"
            )

    def _connect_gripper(self) -> None:
        if RG is None:
            self.get_logger().error(
                f"onrobot.py import 실패: {RG_IMPORT_ERROR}"
            )
            return
        try:
            self._rg2 = RG(
                gripper=self._gripper_type,
                ip=self._gripper_ip,
                port=self._gripper_port,
            )
            self.get_logger().info(
                "OnRobot 연결 완료 | "
                f"address={self._gripper_ip}:{self._gripper_port} | "
                f"status={self._status_flags()}"
            )
        except Exception as exc:
            self._rg2 = None
            self.get_logger().error(f"OnRobot 연결 실패: {exc}")

    def _require_gripper(self) -> None:
        if self._rg2 is None:
            raise RuntimeError("RG2_NOT_CONNECTED")

    def _gripper_widths(self, object_mm: float):
        self._require_gripper()
        if object_mm <= 0.0:
            raise RuntimeError(f"INVALID_OBJECT_WIDTH:{object_mm}")
        max_mm = float(self._rg2.max_width) / 10.0
        if object_mm >= max_mm:
            raise RuntimeError(
                f"OBJECT_TOO_WIDE:object={object_mm:.1f}:max={max_mm:.1f}"
            )
        open_mm = min(max_mm, object_mm + self._open_margin)
        if open_mm - object_mm < self._min_clearance:
            raise RuntimeError("OPEN_CLEARANCE_TOO_SMALL")
        return open_mm, max(0.0, object_mm - self._close_offset)

    def _move_gripper(self, width_mm: float) -> None:
        self._require_gripper()
        width_raw = max(
            0,
            min(int(round(width_mm * 10.0)), int(self._rg2.max_width)),
        )
        force_raw = max(
            0,
            min(self._gripper_force, int(self._rg2.max_force)),
        )
        self._wait_gripper_idle()
        self._rg2.move_gripper(
            width_val=width_raw,
            force_val=force_raw,
        )
        time.sleep(0.05)
        self._wait_gripper_idle()
        self.get_logger().info(
            "RG2 이동 완료 | "
            f"width={width_raw / 10.0:.1f}mm | "
            f"force={force_raw / 10.0:.1f}N"
        )

    def _wait_gripper_idle(self) -> None:
        deadline = time.monotonic() + self._gripper_timeout
        while time.monotonic() < deadline:
            flags = self._status_flags()
            if any(flags[index] == 1 for index in (3, 5, 6)):
                raise RuntimeError(f"RG2_SAFETY_ERROR:{flags}")
            if flags[0] == 0:
                return
            time.sleep(self._poll_period)
        raise TimeoutError(f"RG2_TIMEOUT:{self._gripper_timeout:.1f}s")

    def _status_flags(self):
        self._require_gripper()
        result = self._rg2.client.read_holding_registers(
            address=268,
            count=1,
            unit=65,
        )
        self._check_modbus(result, "status")
        value = int(result.registers[0])
        return [(value >> bit) & 1 for bit in range(7)]

    def _read_grip_state(self):
        flags = self._status_flags()
        result = self._rg2.client.read_holding_registers(
            address=275,
            count=1,
            unit=65,
        )
        self._check_modbus(result, "width")
        return flags[1] == 1, float(result.registers[0]) / 10.0

    @staticmethod
    def _check_modbus(result, operation: str) -> None:
        if result is None:
            raise RuntimeError(f"MODBUS_NO_RESPONSE:{operation}")
        if hasattr(result, "isError") and result.isError():
            raise RuntimeError(f"MODBUS_ERROR:{operation}:{result}")
        if not getattr(result, "registers", None):
            raise RuntimeError(f"MODBUS_NO_REGISTERS:{operation}")

    def _publish_result(
        self,
        msg: PlannedGrasp,
        success: bool,
        reason: str,
    ) -> None:
        output = String()
        output.data = json.dumps(
            {
                "target_id": int(msg.target_id),
                "candidate_id": int(msg.candidate_id),
                "class_name": str(msg.class_name),
                "grasp_type": str(msg.grasp_type),
                "success": bool(success),
                "reason": str(reason),
            },
            ensure_ascii=False,
        )
        self._result_pub.publish(output)


    def publish_working(self, status: bool):
        msg = Bool()
        msg.data = status
        self.work_pub.publish(msg)
        self.get_logger().info("보냄 보이스에")

    def publish_finished(self, status: bool):
        msg = Bool()
        msg.data = status
        self.finish_pub.publish(msg)
        self.get_logger().info("보냄 끝나고 보이스에")
            
        
    


def main(args=None) -> None:
    rclpy.init(args=args)
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    node = RobotExecutor()
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import (
            DR_BASE,
            DR_MV_MOD_ABS,
            DR_MV_MOD_REL,
            get_current_posj,
            movej,
            movel,
            movesj,
        )
        from DR_common2 import posj, posx

        DSR.update(
            {
                "DR_BASE": DR_BASE,
                "DR_MV_MOD_ABS": DR_MV_MOD_ABS,
                "DR_MV_MOD_REL": DR_MV_MOD_REL,
                "get_current_posj": get_current_posj,
                "movej": movej,
                "movel": movel,
                "movesj": movesj,
                "posj": posj,
                "posx": posx,
            }
        )
        node.get_logger().info("두산 API import 완료")

        # 카메라로 물품 위치를 찾기 전에 한 번 스캔 자세로 이동한다.
        node.move_to_scan_pose2()
        
        #node.move_to_scan_pose()
    #     try:
    #         set_tcp("GripperDA_v1")
    #         set_tool("Tool Weight")
    #         node.get_logger().info(
    #             "TCP='GripperDA_v1', Tool='Tool Weight' 적용 완료"
    #         )
    #     except Exception as exc:
    #         node.get_logger().error(f"TCP/Tool 적용 실패: {exc}")
    except ImportError as exc:
        node.get_logger().error(f"DSR_ROBOT2 import 실패: {exc}")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.process_pending()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()