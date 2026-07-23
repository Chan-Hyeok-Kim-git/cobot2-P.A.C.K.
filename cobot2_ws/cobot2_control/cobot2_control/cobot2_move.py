#!/usr/bin/env python3
"""
cobot2_move.py — 두산 M0609 + OnRobot RG2 실제 실행 노드

역할:
  /motion_plan(JSON)을 순서대로 받아 실제 로봇과 RG2를 제어한다.

그리퍼 제어:
  기존 /onrobot/sendCommand ROS 서비스 대신 같은 패키지의 onrobot.py를
  직접 import하여 Compute Box의 Modbus TCP 레지스터를 제어한다.

/motion_plan에서 사용하는 주요 필드:
  class_name            string
  grasp_type            string
  pre_grasp_xyz         [x, y, z]            단위 m
  grasp_xyz             [x, y, z]            단위 m
  pre_grasp_joints      [j1 ... j6]          단위 deg, 선택
  grasp_joints          [j1 ... j6]          단위 deg, 선택
  pre_grasp_abc_deg     [a, b, c]            단위 deg, 선택
  grasp_abc_deg         [a, b, c]            단위 deg, 선택
  grip_width_mm         물체의 예상 파지 폭, 단위 mm
  trajectory            MoveIt trajectory, 선택

실행 순서:
  1. 물체 폭보다 여유 있게 RG2 열기
  2. pre-grasp 이동
  3. grasp 이동
  4. 물체 폭보다 조금 좁은 목표 폭으로 RG2 닫기
  5. RG2 grip detected 상태 확인
  6. 들어올리기
  7. 수납함 이동
  8. RG2 완전 열기
  9. 홈 복귀
"""

import json
import math
import queue
import threading
import time
from typing import Any, Dict, Optional, Sequence

import DR_init
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import String

RG_IMPORT_ERROR: Optional[Exception] = None
try:
    from .onrobot import RG
except (ImportError, ValueError):
    try:
        from onrobot import RG
    except ImportError as exc:
        RG = None  # type: ignore[assignment,misc]
        RG_IMPORT_ERROR = exc


_DSR: Dict[str, Any] = {}

ROBOT_ID = 'dsr01'
ROBOT_MODEL = 'm0609'

VELOCITY = 30
ACC = 30
JOINT_VEL = 30
JOINT_ACC = 30

LIFT_HEIGHT_MM = 100.0

BIN_JOINT_DEG = [781.1, -162.0, 214.0, 150.0, -155.0, -15.0]
HOME_JOINT_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 180.0]


class RobotExecutor(Node):
    def __init__(self) -> None:
        super().__init__('cobot2_move', namespace=ROBOT_ID)
        self._cb_group = ReentrantCallbackGroup()

        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._state_lock = threading.Lock()
        self._gripper_lock = threading.Lock()
        self._rg2 = None

        self.declare_parameter('onrobot_ip', '192.168.1.1')
        self.declare_parameter('onrobot_port', 502)
        self.declare_parameter('onrobot_gripper', 'rg2')
        self.declare_parameter('gripper_force_raw', 100)
        self.declare_parameter('gripper_timeout_sec', 15.0)
        self.declare_parameter('gripper_poll_period_sec', 0.1)
        self.declare_parameter('gripper_open_margin_mm', 15.0)
        self.declare_parameter('gripper_min_open_clearance_mm', 3.0)
        self.declare_parameter('gripper_close_offset_mm', 4.0)
        self.declare_parameter('require_grip_detected', True)

        self._onrobot_ip = str(self.get_parameter('onrobot_ip').get_parameter_value().string_value)
        self._onrobot_port = int(self.get_parameter('onrobot_port').get_parameter_value().integer_value)
        self._onrobot_gripper = str(self.get_parameter('onrobot_gripper').get_parameter_value().string_value).lower()
        self._gripper_force_raw = int(self.get_parameter('gripper_force_raw').get_parameter_value().integer_value)
        self._gripper_timeout_sec = float(self.get_parameter('gripper_timeout_sec').get_parameter_value().double_value)
        self._gripper_poll_period_sec = float(self.get_parameter('gripper_poll_period_sec').get_parameter_value().double_value)
        self._gripper_open_margin_mm = float(self.get_parameter('gripper_open_margin_mm').get_parameter_value().double_value)
        self._gripper_min_open_clearance_mm = float(self.get_parameter('gripper_min_open_clearance_mm').get_parameter_value().double_value)
        self._gripper_close_offset_mm = float(self.get_parameter('gripper_close_offset_mm').get_parameter_value().double_value)
        self._require_grip_detected = bool(self.get_parameter('require_grip_detected').get_parameter_value().bool_value)

        self.create_subscription(String, '/motion_plan', self._on_motion_plan, 10, callback_group=self._cb_group)
        self.create_subscription(String, '/grasp_result', self._on_grasp_result, 10, callback_group=self._cb_group)
        self.pub = self.create_publisher(String, '/execution_result', 10)

        self._connect_gripper()

        self._worker_thread = threading.Thread(target=self._worker, name='cobot2_move_worker', daemon=True)
        self._worker_thread.start()

        self.get_logger().info('RobotExecutor 준비 완료')

    def _on_motion_plan(self, msg: String) -> None:
        try:
            plan = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'/motion_plan JSON 오류: {exc}')
            return
        if not isinstance(plan, dict):
            self.get_logger().error('/motion_plan은 JSON object여야 함')
            return
        self._queue.put(plan)
        self.get_logger().info(f"[{plan.get('class_name', '?')}] 큐 추가 (현재 큐 크기={self._queue.qsize()})")

    def _on_grasp_result(self, msg: String) -> None:
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'/grasp_result JSON 오류: {exc}')
            return
        if isinstance(result, dict) and not result.get('success', True):
            name = result.get('class_name', '?')
            reason = result.get('reason', '파지 후보 없음')
            self.get_logger().warning(f'[{name}] 파지 불가: {reason}')
            self._publish_result(name, False, reason)

    def _worker(self) -> None:
        while rclpy.ok():
            try:
                plan = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._state_lock:
                self._busy = True
            try:
                self._execute(plan)
            except Exception as exc:
                name = plan.get('class_name', '?') if isinstance(plan, dict) else '?'
                self.get_logger().error(f'[{name}] 실행 오류: {exc}')
                self._publish_result(name, False, str(exc))
            finally:
                with self._state_lock:
                    self._busy = False
                self._queue.task_done()

    def _execute(self, plan: dict) -> None:
        if not _DSR:
            raise RuntimeError('DSR 함수가 바인딩되지 않음')

        name = str(plan.get('class_name', '?'))
        grasp_type = str(plan.get('grasp_type', 'FRONT'))

        pre_xyz = self._vector3(plan.get('pre_grasp_xyz'), [0.0, 0.0, 0.5])
        grasp_xyz = self._vector3(plan.get('grasp_xyz'), [0.0, 0.0, 0.3])

        grasp_abc = self._vector3(plan.get('grasp_abc_deg'), [0.0, 0.0, 0.0])
        pre_grasp_abc = self._vector3(plan.get('pre_grasp_abc_deg'), grasp_abc)

        grasp_joints = self._joint_vector(plan.get('grasp_joints'))
        pre_grasp_joints = self._joint_vector(plan.get('pre_grasp_joints'))

        object_width_mm = self._positive_float(plan.get('grip_width_mm', 50.0), field_name='grip_width_mm')

        pre_open_width_mm, grip_target_width_mm = self._calculate_gripper_widths(object_width_mm)

        movel = _DSR['movel']
        movej = _DSR['movej']
        movejx = _DSR['movejx']
        posx = _DSR['posx']
        posj = _DSR['posj']
        wait = _DSR['wait']

        self.get_logger().info(
            f'[{name}] {grasp_type} 파지 시작: 물체 폭={object_width_mm:.1f} mm, '
            f'접근 개방폭={pre_open_width_mm:.1f} mm, 파지 목표폭={grip_target_width_mm:.1f} mm'
        )

        self._move_gripper_mm(pre_open_width_mm)
        wait(0.2)

        if pre_grasp_joints is not None:
            movej(posj(*pre_grasp_joints), vel=JOINT_VEL, acc=JOINT_ACC)
        else:
            pre_posx = posx(pre_xyz[0]*1000.0, pre_xyz[1]*1000.0, pre_xyz[2]*1000.0,
                             pre_grasp_abc[0], pre_grasp_abc[1], pre_grasp_abc[2])
            movejx(pre_posx, vel=JOINT_VEL, acc=JOINT_ACC)

        # ★ 수정: trajectory(웨이포인트 목록)를 grasp_joints보다 우선 실행하면
        # _exec_trajectory가 웨이포인트마다 별도 movej()를 호출해서 "가다 서다"가
        # 수십~백여 번 반복되는 끊긴 움직임이 된다(Cartesian 경로는 보통 100개
        # 이상의 점으로 구성됨). grasp_joints는 이미 DSR ikin으로 검증됐고,
        # MoveIt plan() 성공 여부로 "충돌 없는 경로가 존재한다"는 것도 이미
        # 확인됐다. 중간 웨이포인트를 하나하나 재생할 필요 없이 검증된 최종
        # 목표로 movej 한 번에 이동하면 로봇 자체 컨트롤러가 가속-정속-감속을
        # 매끄럽게 처리한다. trajectory는 grasp_joints가 없을 때만 폴백으로 사용.
        trajectory = plan.get('trajectory')
        if grasp_joints is not None:
            movej(posj(*grasp_joints), vel=JOINT_VEL, acc=JOINT_ACC)
        elif isinstance(trajectory, dict) and trajectory.get('points'):
            self._exec_trajectory(trajectory)
        else:
            grasp_posx = posx(grasp_xyz[0]*1000.0, grasp_xyz[1]*1000.0, grasp_xyz[2]*1000.0,
                               grasp_abc[0], grasp_abc[1], grasp_abc[2])
            movejx(grasp_posx, vel=JOINT_VEL, acc=JOINT_ACC)

        wait(0.3)

        self._move_gripper_mm(grip_target_width_mm)
        wait(0.2)

        grip_detected, actual_width_mm = self._read_grip_state()
        self.get_logger().info(f'[{name}] RG2 상태: grip_detected={grip_detected}, 현재 폭={actual_width_mm:.1f} mm')

        if self._require_grip_detected and not grip_detected:
            raise RuntimeError(f'RG2 파지 감지 실패 (명령 폭={grip_target_width_mm:.1f} mm, 현재 폭={actual_width_mm:.1f} mm)')

        lift_posx = posx(grasp_xyz[0]*1000.0, grasp_xyz[1]*1000.0, grasp_xyz[2]*1000.0 + LIFT_HEIGHT_MM,
                          grasp_abc[0], grasp_abc[1], grasp_abc[2])
        movel(lift_posx, vel=VELOCITY, acc=ACC)

        movel(posx(*BIN_JOINT_DEG), vel=VELOCITY, acc=ACC)
        wait(0.5)

        self._open_gripper_fully()
        wait(0.3)

        movej(posj(*HOME_JOINT_DEG), vel=JOINT_VEL, acc=JOINT_ACC)

        self.get_logger().info(f'[{name}] 파지-수납 완료')
        self._publish_result(name, True, '파지-수납 완료')

    def _exec_trajectory(self, trajectory: dict) -> None:
        """
        MoveIt joint trajectory를 DSR로 실행 (grasp_joints가 없을 때만 쓰는 폴백 경로).

        ★ 수정: 이전에는 모든 웨이포인트마다 별도 movej()를 호출해서
        (Cartesian 경로는 보통 100개 이상 점) "가다 서다"가 반복되는
        끊긴 움직임이 됐다. movesj(스플라인 관절이동)가 있으면 전체
        웨이포인트를 한 번에 넘겨 하나의 매끄러운 연속 동작으로 실행하고,
        movesj를 못 쓰는 환경이면 중간 점은 건너뛰고 마지막 목표점으로
        movej 한 번만 실행한다.
        """
        if not _DSR:
            raise RuntimeError('DSR 함수가 바인딩되지 않음')

        posj = _DSR['posj']
        points = trajectory.get('points', [])
        if not isinstance(points, list) or not points:
            raise ValueError('trajectory.points가 비어 있음')

        posj_list = []
        for point in points:
            positions = point.get('positions', [])
            if not isinstance(positions, list) or len(positions) < 6:
                raise ValueError('trajectory point의 positions가 6축이 아님')
            positions_deg = [math.degrees(float(value)) for value in positions[:6]]
            posj_list.append(posj(*positions_deg))

        movesj = _DSR.get('movesj')
        if movesj is not None and len(posj_list) > 1:
            movesj(posj_list, vel=JOINT_VEL, acc=JOINT_ACC)
        else:
            movej = _DSR['movej']
            movej(posj_list[-1], vel=JOINT_VEL, acc=JOINT_ACC)

    def _connect_gripper(self) -> None:
        if RG is None:
            self.get_logger().error(
                f'onrobot.py import 실패: {RG_IMPORT_ERROR}. '
                'onrobot.py를 cobot2_move.py와 같은 Python 패키지에 넣고 pymodbus 호환 버전을 설치해야 함.'
            )
            return
        try:
            self._rg2 = RG(gripper=self._onrobot_gripper, ip=self._onrobot_ip, port=self._onrobot_port)
            flags = self._read_status_flags()
            self.get_logger().info(f'OnRobot {self._onrobot_gripper.upper()} Modbus 연결 완료: {self._onrobot_ip}:{self._onrobot_port}, status={flags}')
        except Exception as exc:
            self._rg2 = None
            self.get_logger().error(f'OnRobot Modbus 연결 실패 ({self._onrobot_ip}:{self._onrobot_port}): {exc}')

    def _ensure_gripper(self) -> None:
        if self._rg2 is None:
            raise RuntimeError('RG2가 연결되지 않음. onrobot_ip, onrobot_port와 Compute Box Modbus TCP 설정을 확인해야 함.')

    def _calculate_gripper_widths(self, object_width_mm: float):
        self._ensure_gripper()
        max_width_mm = float(self._rg2.max_width) / 10.0
        if object_width_mm >= max_width_mm:
            raise ValueError(f'물체 폭 {object_width_mm:.1f} mm가 {self._onrobot_gripper.upper()} 최대 폭 {max_width_mm:.1f} mm 이상임')
        pre_open_width_mm = min(max_width_mm, object_width_mm + self._gripper_open_margin_mm)
        actual_clearance_mm = pre_open_width_mm - object_width_mm
        if actual_clearance_mm < self._gripper_min_open_clearance_mm:
            raise ValueError(f'접근 개방 여유가 부족함: {actual_clearance_mm:.1f} mm (필요 최소 {self._gripper_min_open_clearance_mm:.1f} mm)')
        grip_target_width_mm = max(0.0, object_width_mm - self._gripper_close_offset_mm)
        return pre_open_width_mm, grip_target_width_mm

    def _move_gripper_mm(self, width_mm: float) -> None:
        self._ensure_gripper()
        max_width_raw = int(self._rg2.max_width)
        max_force_raw = int(self._rg2.max_force)
        width_raw = int(round(float(width_mm) * 10.0))
        width_raw = max(0, min(width_raw, max_width_raw))
        force_raw = max(0, min(self._gripper_force_raw, max_force_raw))
        with self._gripper_lock:
            self._wait_gripper_idle()
            self._rg2.move_gripper(width_val=width_raw, force_val=force_raw)
            time.sleep(0.05)
            self._wait_gripper_idle()
        self.get_logger().info(f'RG2 이동 완료: 목표 폭={width_raw/10.0:.1f} mm, 힘={force_raw/10.0:.1f} N')

    def _open_gripper_fully(self) -> None:
        self._ensure_gripper()
        max_force_raw = int(self._rg2.max_force)
        force_raw = max(0, min(self._gripper_force_raw, max_force_raw))
        with self._gripper_lock:
            self._wait_gripper_idle()
            self._rg2.open_gripper(force_val=force_raw)
            time.sleep(0.05)
            self._wait_gripper_idle()
        self.get_logger().info('RG2 완전 개방 완료')

    def _wait_gripper_idle(self) -> None:
        deadline = time.monotonic() + self._gripper_timeout_sec
        while time.monotonic() < deadline:
            flags = self._read_status_flags()
            busy = flags[0] == 1
            safety_error = any(flags[index] == 1 for index in (3, 5, 6))
            if safety_error:
                raise RuntimeError(f'RG2 safety 상태 발생: {flags}')
            if not busy:
                return
            time.sleep(self._gripper_poll_period_sec)
        raise TimeoutError(f'RG2 동작 타임아웃 ({self._gripper_timeout_sec:.1f}초)')

    def _read_status_flags(self):
        self._ensure_gripper()
        result = self._rg2.client.read_holding_registers(address=268, count=1, unit=65)
        self._check_modbus_result(result, 'status register read')
        value = int(result.registers[0])
        return [(value >> bit) & 0x1 for bit in range(7)]

    def _read_grip_state(self):
        self._ensure_gripper()
        with self._gripper_lock:
            flags = self._read_status_flags()
            result = self._rg2.client.read_holding_registers(address=275, count=1, unit=65)
            self._check_modbus_result(result, 'width-with-offset register read')
            width_mm = float(result.registers[0]) / 10.0
        return flags[1] == 1, width_mm

    @staticmethod
    def _check_modbus_result(result, operation: str) -> None:
        if result is None:
            raise RuntimeError(f'Modbus {operation}: 응답 없음')
        if hasattr(result, 'isError') and result.isError():
            raise RuntimeError(f'Modbus {operation} 실패: {result}')
        if not hasattr(result, 'registers') or not result.registers:
            raise RuntimeError(f'Modbus {operation}: registers 없음')

    @staticmethod
    def _vector3(value, default):
        source = default if value is None else value
        if not isinstance(source, (list, tuple)) or len(source) != 3:
            raise ValueError(f'3개 원소 벡터가 필요함: {source}')
        result = [float(component) for component in source]
        if not all(math.isfinite(component) for component in result):
            raise ValueError(f'벡터에 유효하지 않은 값이 있음: {source}')
        return result

    @staticmethod
    def _joint_vector(value):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError(f'관절값은 6개여야 함: {value}')
        result = [float(component) for component in value]
        if not all(math.isfinite(component) for component in result):
            raise ValueError(f'관절값에 유효하지 않은 값이 있음: {value}')
        return result

    @staticmethod
    def _positive_float(value, field_name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f'{field_name}은 0보다 큰 유한값이어야 함: {value}')
        return result

    def _publish_result(self, class_name: str, success: bool, reason: str) -> None:
        msg = String()
        msg.data = json.dumps({'class_name': class_name, 'success': success, 'reason': reason}, ensure_ascii=False)
        self.pub.publish(msg)

    def destroy_node(self) -> bool:
        if self._rg2 is not None:
            try:
                self._rg2.close_connection()
                self.get_logger().info('OnRobot Modbus 연결 종료')
            except Exception as exc:
                self.get_logger().warning(f'OnRobot 연결 종료 중 오류: {exc}')
            finally:
                self._rg2 = None
        return super().destroy_node()


def main(args=None) -> None:
    global _DSR
    rclpy.init(args=args)
    node = None
    try:
        node = RobotExecutor()
        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = node

        from DSR_ROBOT2 import (
            get_current_posj, get_current_posx, get_digital_input,
            movej, movejx, movel, set_digital_output, set_tcp, set_tool, wait,
        )
        from DR_common2 import posj, posx

        _DSR = {
            'movel': movel, 'movej': movej, 'movejx': movejx,
            'set_tool': set_tool, 'set_tcp': set_tcp,
            'set_digital_output': set_digital_output,
            'get_digital_input': get_digital_input,
            'get_current_posx': get_current_posx,
            'get_current_posj': get_current_posj,
            'posx': posx, 'posj': posj, 'wait': wait,
        }

        # ★ 추가: movesj(스플라인 관절이동) — 있으면 _exec_trajectory가
        # 여러 웨이포인트를 하나의 매끄러운 동작으로 실행하는 데 사용한다.
        # DSR_ROBOT2 버전에 따라 없을 수 있어 별도로 안전하게 시도한다.
        try:
            from DSR_ROBOT2 import movesj
            _DSR['movesj'] = movesj
            node.get_logger().info('movesj(스플라인 이동) 사용 가능')
        except ImportError:
            node.get_logger().warning('movesj 없음 — 궤적 실행 시 최종 목표점으로만 이동(폴백)')

        node.get_logger().info('두산 API import 완료 (DSR 활성화)')
    except Exception as exc:
        if node is None:
            node = RobotExecutor()
        node.get_logger().error(f'DSR_ROBOT2 초기화 실패: {exc}')

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()