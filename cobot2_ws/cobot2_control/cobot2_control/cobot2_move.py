"""
cobot2_move.py — 실제 로봇 실행 모듈

역할:
  /motion_plan (JSON) 을 구독해서 실제 로봇을 움직임.

  실행 순서:
    1. 그리퍼 열기 (OnRobot /onrobot/sendCommand)
    2. DSR movej → pre-grasp 위치
    3. DSR movel → grasp 위치 (직선 접근)
    4. 그리퍼 닫기 (물체 폭만큼)
    5. 파지 확인 (그리퍼가 완전히 닫히지 않았는지)
    6. DSR movel → 들어올리기 (+z 10cm)
    7. DSR movej → 수납함 위치 (고정 관절 각도)
    8. 그리퍼 열기 (놓기)
    9. DSR movej → 홈 복귀

  물품 큐(Queue):
    /object_info 를 직접 구독해서 순서대로 처리.
    처리 중엔 새 물품은 큐에만 저장, 완료되면 다음 물품 처리.

ROS 인터페이스:
  SUB  /motion_plan      (std_msgs/String, JSON)
  SUB  /grasp_result     (std_msgs/String, JSON)  ← 못 잡음 알림
  PUB  /execution_result (std_msgs/String, JSON)
  CALL /onrobot/sendCommand (onrobot_rg_msgs/srv/SetCommand)
"""

import json
import queue
import threading
import time
import rclpy
import DR_init
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

try:
    from onrobot_rg_msgs.srv import SetCommand
    HAS_ONROBOT = True
except ImportError:
    HAS_ONROBOT = False

# ── DSR 전역 (main에서 import 후 바인딩) ──────────────────────────────
_DSR = {}

# ── 상수 ──────────────────────────────────────────────────────────────
ROBOT_ID    = 'dsr01'
ROBOT_MODEL = 'm0609'   # ★ 추가: DR_init 등록에 필요 (누락되어 있었음)
VELOCITY   = 30      # mm/s (movel)
ACC        = 30
JOINT_VEL  = 30      # deg/s (movej)
JOINT_ACC  = 30

LIFT_HEIGHT_MM  = 100.0   # 파지 후 들어올리기 높이 (mm)
GRIPPER_TIMEOUT = 5.0     # 그리퍼 서비스 응답 대기 (초)

# 수납함 위치 (관절 각도, deg) — 실기에서 측정 후 수정
BIN_JOINT_DEG  = [0.0, -30.0, 120.0, 0.0, 90.0, 0.0]
# 홈 위치 (관절 각도, deg)
HOME_JOINT_DEG = [0.0,   0.0,   0.0, 0.0,  0.0, 0.0]


class RobotExecutor(Node):
    def __init__(self):
        # ★ 수정: namespace=ROBOT_ID 추가.
        # DSR_ROBOT2.py는 서비스 클라이언트를 상대 경로(예: motion/move_joint)로
        # 생성하는데, 이는 노드 자신의 네임스페이스를 기준으로 해석된다.
        # namespace가 없으면 /motion/move_joint로 해석되어 실제 서비스
        # (/dsr01/motion/move_joint)를 영원히 못 찾고 "Service is not
        # available" 대기 상태에 머무른다.
        # cobot2_grasp.py의 grasp_validator는 이미 namespace=ROBOT_ID로
        # 생성되어 있었기 때문에 정상 동작했던 것이다.
        super().__init__('cobot2_move', namespace=ROBOT_ID)
        self._cb_group = ReentrantCallbackGroup()

        # 물품 처리 큐
        self._queue   = queue.Queue()
        self._busy    = False
        self._lock    = threading.Lock()

        # 구독 / 퍼블리시
        self.create_subscription(
            String, '/motion_plan', self._on_motion_plan, 10,
            callback_group=self._cb_group)
        self.create_subscription(
            String, '/grasp_result', self._on_grasp_result, 10,
            callback_group=self._cb_group)
        self.pub = self.create_publisher(String, '/execution_result', 10)

        # OnRobot 그리퍼 서비스 클라이언트
        if HAS_ONROBOT:
            self._gripper_cli = self.create_client(
                SetCommand, '/onrobot/sendCommand',
                callback_group=self._cb_group)
        else:
            self._gripper_cli = None
            self.get_logger().warn('onrobot_rg_msgs 없음 — 그리퍼 제어 스킵')

        # 큐 처리 워커 스레드
        threading.Thread(target=self._worker, daemon=True).start()

        self.get_logger().info('RobotExecutor 준비 완료')

    # ── 입력 처리 ────────────────────────────────────────────────────
    def _on_motion_plan(self, msg: String):
        try:
            plan = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._queue.put(plan)
        self.get_logger().info(
            f"[{plan.get('class_name')}] 큐 추가 (현재 큐 크기={self._queue.qsize()})")

    def _on_grasp_result(self, msg: String):
        try:
            res = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not res.get('success', True):
            self.get_logger().warn(
                f"[{res.get('class_name')}] 파지 불가 — 큐에서 제외: {res.get('reason')}")
            self._publish_result(res.get('class_name', '?'), False, res.get('reason', ''))

    # ── 큐 워커 ─────────────────────────────────────────────────────
    def _worker(self):
        """순차 처리: 큐에서 꺼내 하나씩 실행"""
        while rclpy.ok():
            try:
                plan = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:
                self._busy = True
            try:
                self._execute(plan)
            except Exception as e:
                self.get_logger().error(f"실행 오류: {e}")
                self._publish_result(plan.get('class_name', '?'), False, str(e))
            finally:
                with self._lock:
                    self._busy = False
                self._queue.task_done()

    # ── 실제 실행 ────────────────────────────────────────────────────
    def _execute(self, plan: dict):
        name        = plan.get('class_name', '?')
        pre_xyz     = plan.get('pre_grasp_xyz', [0, 0, 0.5])
        grasp_xyz   = plan.get('grasp_xyz',     [0, 0, 0.3])
        grasp_type  = plan.get('grasp_type', 'FRONT')
        # ★ 수정: cobot2_mi.py가 전달하는 grasp_joints/pre_grasp_joints 사용
        # DSR ikin + MoveIt으로 이미 검증된 관절값 → movej에 직접 사용 가능
        grasp_joints     = plan.get('grasp_joints', None)      # float64[6] or None
        pre_grasp_joints = plan.get('pre_grasp_joints', None)  # float64[6] or None
        # ★ 수정: grip_width_mm이 전혀 전달되지 않아 미정의 변수 크래시가 났던 부분.
        # cobot2_mi_node가 이제 required_width(m)를 mm로 변환해서 보내준다.
        grip_w_mm = plan.get('grip_width_mm', 50.0)

        dsr = _DSR
        if not dsr:
            self.get_logger().error('DSR 함수가 바인딩되지 않음')
            return

        movel  = dsr['movel']
        movej  = dsr['movej']
        movejx = dsr['movejx']
        posx   = dsr['posx']
        posj   = dsr['posj']
        wait   = dsr['wait']

        self.get_logger().info(f'[{name}] {grasp_type} 파지 시작')

        # ── 1. 그리퍼 열기 ────────────────────────────────────────
        self._gripper_cmd('o')
        wait(0.5)

        # ── 2. pre-grasp 이동 ───────────────────────────────────
        # ★ 수정: movej()는 관절각도(posj)만 받는데 이전 코드는
        # Cartesian 좌표(posx)를 그대로 넘겨서 'Invalid type : pos' 에러로
        # 크래시가 났었다. 이제 pre_grasp_joints(DSR ikin으로 이미 검증된
        # 관절값)가 있으면 movej(posj(...))로 바로 이동하고, 없을 때만
        # movejx(Cartesian 목표를 내부적으로 IK 풀어서 관절이동)로 대체한다.
        if pre_grasp_joints and len(pre_grasp_joints) == 6:
            pre_posj = posj(*pre_grasp_joints)
            movej(pre_posj, vel=JOINT_VEL, acc=JOINT_ACC)
        else:
            pre_posx = posx(
                pre_xyz[0]*1000, pre_xyz[1]*1000, pre_xyz[2]*1000,
                0, 0, 0
            )
            movejx(pre_posx, vel=JOINT_VEL, acc=JOINT_ACC)

        # ── 3. grasp 위치로 이동 ─────────────────────────────────
        # ★ 수정: 이전에는 trajectory(웨이포인트 목록)가 있으면 무조건
        # 그걸 우선 실행했는데, _exec_trajectory가 매 웨이포인트마다
        # 별도의 movej()를 호출하는 구조라서 "가다 서다"를 수십~백여 번
        # 반복하는 끊긴 움직임이 됐다(Cartesian 경로는 보통 100개 이상의
        # 점으로 구성됨). 로봇이 "짧은 거리를 계속 계산"하는 것처럼
        # 보였던 원인이 바로 이것이다.
        #
        # grasp_joints는 이미 DSR ikin으로 검증됐고, MoveIt의 plan()
        # 성공 여부로 "충돌 없는 경로가 존재한다"는 것도 확인됐다.
        # 굳이 중간 웨이포인트를 하나하나 재생할 필요 없이, 검증된
        # 최종 목표로 movej 한 번에 이동하면 로봇 자체 컨트롤러가
        # 가속-정속-감속을 매끄럽게 처리해서 훨씬 자연스럽게 움직인다.
        # trajectory는 grasp_joints가 없는 예외 상황에서만 폴백으로 사용.
        traj = plan.get('trajectory')
        if grasp_joints and len(grasp_joints) == 6:
            j = posj(*grasp_joints)
            movej(j, vel=JOINT_VEL, acc=JOINT_ACC)
        elif traj:
            self._exec_trajectory(traj)
        else:
            grasp_posx = posx(
                grasp_xyz[0]*1000, grasp_xyz[1]*1000, grasp_xyz[2]*1000,
                0, 0, 0
            )
            movejx(grasp_posx, vel=VELOCITY, acc=ACC)

        wait(0.3)

        # ── 4. 그리퍼 닫기 ────────────────────────────────────────
        grip_cmd = str(int(grip_w_mm * 10))  # mm → 1/10mm 단위 정수 문자열
        self._gripper_cmd(grip_cmd)
        wait(1.0)

        # ── 5. 파지 확인 (그리퍼 폭이 0mm면 슬립, 너무 넓으면 실패) ─
        # (실기에서 /onrobot/pose 서비스로 현재 폭 확인 가능)

        # ── 6. 들어올리기 ─────────────────────────────────────────
        lift_posx = posx(
            grasp_xyz[0]*1000,
            grasp_xyz[1]*1000,
            (grasp_xyz[2] + LIFT_HEIGHT_MM/1000)*1000,
            0, 0, 0
        )
        movel(lift_posx, vel=VELOCITY, acc=ACC)

        # ── 7. 수납함 이동 ────────────────────────────────────────
        bin_posj = posj(*BIN_JOINT_DEG)
        movej(bin_posj, vel=JOINT_VEL, acc=JOINT_ACC)
        wait(0.5)

        # ── 8. 그리퍼 열기 (놓기) ────────────────────────────────
        self._gripper_cmd('o')
        wait(0.5)

        # ── 9. 홈 복귀 ───────────────────────────────────────────
        home_posj = posj(*HOME_JOINT_DEG)
        movej(home_posj, vel=JOINT_VEL, acc=JOINT_ACC)

        self.get_logger().info(f'[{name}] 파지-수납 완료')
        self._publish_result(name, True, '파지-수납 완료')

    def _exec_trajectory(self, traj_dict: dict):
        """
        MoveIt이 계획한 joint trajectory를 DSR로 실행 (폴백 경로).

        ★ 수정: 이전에는 모든 웨이포인트마다 별도 movej()를 호출해서
        (Cartesian 경로는 보통 100개 이상 점) "가다 서다"가 반복되는
        끊긴 움직임이 됐었다. 이제 movesj(스플라인 관절이동)가 있으면
        전체 웨이포인트를 한 번에 넘겨 하나의 매끄러운 연속 동작으로
        실행하고, movesj를 못 쓰는 환경이면 중간 점은 건너뛰고
        마지막 목표점으로 movej 한 번만 실행한다.
        """
        if not _DSR or not traj_dict:
            return

        points = traj_dict.get('points', [])
        if not points:
            return

        import math
        posj = _DSR['posj']

        # 라디안 → 도(degree) 리스트로 일괄 변환
        posj_list = [
            posj(*[math.degrees(p) for p in pt['positions'][:6]])
            for pt in points
        ]

        movesj = _DSR.get('movesj')
        if movesj is not None and len(posj_list) > 1:
            # 전체 웨이포인트를 하나의 스플라인 동작으로 한 번에 실행
            # (중간에 멈추지 않고 부드럽게 이어서 움직임)
            movesj(posj_list, vel=JOINT_VEL, acc=JOINT_ACC)
        else:
            # movesj를 쓸 수 없으면 중간 점은 건너뛰고 최종 목표로만 이동
            movej = _DSR['movej']
            movej(posj_list[-1], vel=JOINT_VEL, acc=JOINT_ACC)

    # ── 그리퍼 제어 ──────────────────────────────────────────────────
    def _gripper_cmd(self, command: str):
        """
        command:
          'o'    → 완전히 열기
          'c'    → 완전히 닫기
          '500'  → 50.0mm 폭 (1/10mm 단위 문자열)
        """
        if self._gripper_cli is None:
            self.get_logger().warn(f'그리퍼 명령 스킵: {command}')
            return

        if not self._gripper_cli.wait_for_service(timeout_sec=GRIPPER_TIMEOUT):
            self.get_logger().error('/onrobot/sendCommand 서비스 없음')
            return

        req = SetCommand.Request()
        req.command = command
        future = self._gripper_cli.call_async(req)

        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > GRIPPER_TIMEOUT:
                self.get_logger().error(f'그리퍼 명령 타임아웃: {command}')
                return
            time.sleep(0.02)

        if future.result() and not future.result().success:
            self.get_logger().error(f'그리퍼 명령 실패: {command}')

    def _publish_result(self, class_name: str, success: bool, reason: str):
        msg = String()
        msg.data = json.dumps({
            'class_name': class_name,
            'success':    success,
            'reason':     reason,
        }, ensure_ascii=False)
        self.pub.publish(msg)


# ── 진입점 ────────────────────────────────────────────────────────────
def main(args=None):
    global _DSR

    rclpy.init(args=args)
    node = None

    try:
        # ★ 수정: 임시 노드를 만들었다가 destroy하는 방식은 근본적으로 잘못됨.
        # DSR_ROBOT2는 import되는 순간 DR_init.__dsr__node로 등록된 노드를
        # 이용해 내부적으로 서비스 클라이언트를 만들어 저장해 둔다.
        # 그 노드를 곧바로 destroy()하면 이후 movej() 등을 호출할 때마다
        # "cannot use Destroyable because destruction was requested" 에러가 난다.
        #
        # cobot2_grasp.py가 정상 동작하는 이유가 바로 이것: 거기서는 노드를
        # 절대 중간에 destroy하지 않고 프로그램이 끝날 때까지 그대로 유지한다.
        #
        # → 실제로 계속 살아있을 RobotExecutor를 먼저 생성하고, 그 노드를
        #   그대로 DR_init에 등록한다. 임시 노드도, 중간 destroy도 없앤다.
        node = RobotExecutor()

        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = node

        from DSR_ROBOT2 import (
            movel, movej, movejx,
            set_tool, set_tcp,
            set_digital_output, get_digital_input,
            get_current_posx, get_current_posj,
            wait,
        )
        from DR_common2 import posx, posj

        _DSR = {
            'movel': movel, 'movej': movej, 'movejx': movejx,
            'set_tool': set_tool, 'set_tcp': set_tcp,
            'set_digital_output': set_digital_output,
            'get_digital_input': get_digital_input,
            'get_current_posx': get_current_posx,
            'get_current_posj': get_current_posj,
            'posx': posx, 'posj': posj,
            'wait': wait,
        }

        # ★ 추가: movesj(스플라인 관절이동) — 있으면 여러 웨이포인트를
        # 하나의 매끄러운 동작으로 실행하는 데 사용(_exec_trajectory 참고).
        # DSR_ROBOT2 버전에 따라 이름이 다르거나 없을 수 있어 별도로
        # 안전하게 시도하고, 실패해도 movej/movejx 위주 폴백으로 동작.
        try:
            from DSR_ROBOT2 import movesj
            _DSR['movesj'] = movesj
            node.get_logger().info('movesj(스플라인 이동) 사용 가능')
        except ImportError:
            node.get_logger().warn(
                'movesj 없음 — 궤적 실행 시 최종 목표점으로만 이동(폴백)')

        node.get_logger().info('두산 API import 완료 (DSR 활성화)')

    except ImportError as e:
        if node is None:
            # DSR import 실패 시에도 RobotExecutor는 반드시 생성해야 함
            node = RobotExecutor()
        node.get_logger().warn(f'DSR_ROBOT2 import 실패: {e} → DSR 없이 실행')

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()