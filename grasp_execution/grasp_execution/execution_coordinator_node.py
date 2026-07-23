"""
grasp_execution_coordinator

파이프라인의 마지막 조각. 하는 일:

  1. /grasp_validation/results (ValidatedGraspArray) 구독
     - grasp_local_validation 의 local_validator_node 가 발행
  2. valid=true 인 후보만 골라 local_score 기준 정렬
  3. grasp_moveit_validator 의 /validate_and_plan 서비스 호출
     - IK/충돌까지 통과한 궤적(RobotTrajectory)을 돌려받음
  4. 받은 궤적을 로봇 컨트롤러(dsr_moveit_controller)에 FollowJointTrajectory
     액션으로 직접 보내 실행 (MoveGroupInterface의 Python 바인딩 없이,
     이미 계획된 궤적을 그대로 전달하는 방식 - 의존성 최소화)
  5. onrobot_rg_control 의 실제 서비스(/onrobot/sendCommand) 로 그리퍼 개폐
  6. 들어올리기 -> 수납함 이동 -> 그리퍼 열기(놓기) -> 복귀

각 단계 실패 시 다음 단계로 진행하지 않고 즉시 실패 보고.
"""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from grasp_interfaces.msg import ValidatedGraspArray
from grasp_interfaces.srv import ValidateAndPlan
from onrobot_rg_msgs.srv import SetCommand


class GraspExecutionCoordinator(Node):

    def __init__(self):
        super().__init__('grasp_execution_coordinator')
        self._cb_group = ReentrantCallbackGroup()

        self._declare_params()
        gp = self.get_parameter

        ns = gp('namespace').value
        action_name = f"/{ns}/{gp('arm_controller_action').value}" if ns else \
            f"/{gp('arm_controller_action').value}"

        # ── 클라이언트/서브스크라이버 ──────────────────────────────
        self._validated_sub = self.create_subscription(
            ValidatedGraspArray, gp('validated_grasp_topic').value,
            self._on_validated_grasps, 10, callback_group=self._cb_group)

        self._plan_client = self.create_client(
            ValidateAndPlan, gp('validate_and_plan_service').value,
            callback_group=self._cb_group)

        self._gripper_client = self.create_client(
            SetCommand, gp('gripper_command_service').value,
            callback_group=self._cb_group)

        self._traj_action_client = ActionClient(
            self, FollowJointTrajectory, action_name,
            callback_group=self._cb_group)

        # 수동 테스트용 트리거 (업스트림 파이프라인 없이 지금 상태로 1회 실행)
        self._manual_trigger_srv = self.create_service(
            Trigger, 'grasp_execution_coordinator/trigger_last',
            self._on_manual_trigger, callback_group=self._cb_group)

        self._lock = threading.Lock()
        self._busy = False
        self._last_msg = None

        self.get_logger().info(
            f"grasp_execution_coordinator ready "
            f"(arm action='{action_name}', gripper srv='{gp('gripper_command_service').value}')"
        )

    def _declare_params(self):
        defaults = {
            'validated_grasp_topic': '/grasp_validation/results',
            'validate_and_plan_service': '/validate_and_plan',
            'namespace': 'dsr01',
            'arm_controller_action': 'dsr_moveit_controller/follow_joint_trajectory',
            'joint_names': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
            'gripper_command_service': '/onrobot/sendCommand',
            'gripper_open_width_m': 0.105,
            'gripper_close_margin_m': 0.0,
            'lift_height_m': 0.10,
            'post_grasp_settle_sec': 0.3,
            'bin_joint_positions_rad': [0.0, -0.3, 1.2, 0.0, 1.6, 0.0],
            'home_joint_positions_rad': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'trajectory_time_to_bin_sec': 4.0,
            'trajectory_time_to_home_sec': 4.0,
            'service_wait_timeout_sec': 5.0,
            'action_wait_timeout_sec': 5.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    # ------------------------------------------------------------------
    # 입력 처리
    # ------------------------------------------------------------------
    def _on_validated_grasps(self, msg: ValidatedGraspArray):
        with self._lock:
            self._last_msg = msg
            if self._busy:
                self.get_logger().warn('이미 실행 중, 새 결과는 저장만 하고 무시')
                return
            self._busy = True

        threading.Thread(target=self._run_cycle, args=(msg,), daemon=True).start()

    def _on_manual_trigger(self, request, response):
        with self._lock:
            msg = self._last_msg
            if msg is None:
                response.success = False
                response.message = '아직 수신된 ValidatedGraspArray 없음'
                return response
            if self._busy:
                response.success = False
                response.message = '이미 실행 중'
                return response
            self._busy = True
        self._run_cycle(msg)
        response.success = True
        response.message = 'triggered'
        return response

    # ------------------------------------------------------------------
    # 메인 실행 사이클
    # ------------------------------------------------------------------
    def _run_cycle(self, msg: ValidatedGraspArray):
        try:
            candidates = [g for g in msg.grasps if g.valid]
            if not candidates:
                self.get_logger().error('유효한(valid=true) 후보 없음 - 파지 스킵, 누락 보고')
                return
            candidates.sort(key=lambda g: g.local_score, reverse=True)

            self.get_logger().info(f'{len(candidates)}개 유효 후보 -> MoveIt 검증 요청')
            plan_resp = self._call_validate_and_plan(candidates)
            if plan_resp is None or not plan_resp.success:
                reason = plan_resp.failure_reason if plan_resp else 'service call failed'
                self.get_logger().error(f'MoveIt 검증/계획 실패: {reason}')
                return

            selected = next(
                (c for c in candidates if c.candidate.candidate_id == plan_resp.selected_candidate_id),
                candidates[0])
            self.get_logger().info(
                f'선택된 후보 id={plan_resp.selected_candidate_id} '
                f'(cartesian_fraction={plan_resp.cartesian_fraction:.2f})'
            )

            # 1. pre-grasp 이동
            if not self._execute_trajectory(plan_resp.to_pre_grasp, '(pre-grasp 이동)'):
                return

            # 2. 그리퍼 열기 (여유 확보)
            if not self._call_gripper('o'):
                self.get_logger().error('그리퍼 열기 실패')
                return

            # 3. grasp 위치까지 접근
            if not self._execute_trajectory(plan_resp.approach, '(접근)'):
                return

            # 4. 그리퍼 닫기 (요청 폭만큼)
            width_tenths_mm = int(selected.measured_width * 10000)  # m -> 1/10mm
            if not self._call_gripper(str(width_tenths_mm)):
                self.get_logger().error('그리퍼 닫기 실패')
                return

            # 5. 파지 확인 (너무 완전히 닫히면 = 놓침)
            # (실기 연결 시 onrobot_rg_control 의 /onrobot/pose 나 getStatus 로 폭 재확인 권장.
            #  여기서는 구조만 두고 실기에서 채워야 함)

            # 6. 들어올리기 - grasp pose에서 +z 오프셋으로 재계획 요청
            lift_ok = self._lift(selected)
            if not lift_ok:
                self.get_logger().error('들어올리기 계획 실패')
                return

            # 7. 수납함으로 이동 (고정 관절 목표)
            if not self._move_to_fixed_joints(
                    'bin_joint_positions_rad', 'trajectory_time_to_bin_sec', '(수납함 이동)'):
                return

            # 8. 그리퍼 열기 (놓기)
            self._call_gripper('o')

            # 9. 원위치 복귀
            self._move_to_fixed_joints(
                'home_joint_positions_rad', 'trajectory_time_to_home_sec', '(원위치 복귀)')

            self.get_logger().info('파지-수납 사이클 완료')

        finally:
            with self._lock:
                self._busy = False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _wait_for_future(self, future, timeout_sec: float, poll_interval: float = 0.02):
        """
        MultiThreadedExecutor 가 이미 별도 스레드에서 spin 중이므로
        여기서 rclpy.spin_until_future_complete() 를 쓰면 안 됨(재귀 spin 충돌).
        응답 콜백은 executor가 알아서 처리해주니, 여기서는 단순히
        future.done() 이 될 때까지 폴링만 한다.
        """
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > timeout_sec:
                return None
            time.sleep(poll_interval)
        return future.result()

    def _call_validate_and_plan(self, candidates):
        gp = self.get_parameter
        if not self._plan_client.wait_for_service(timeout_sec=gp('service_wait_timeout_sec').value):
            self.get_logger().error('validate_and_plan 서비스 응답 없음')
            return None

        req = ValidateAndPlan.Request()
        req.scene_id = candidates[0].candidate.scene_id
        req.candidates = candidates
        req.dynamic_obstacles = []  # 필요하면 environment_cloud 기반으로 채워서 확장

        future = self._plan_client.call_async(req)
        result = self._wait_for_future(future, timeout_sec=30.0)
        if result is None:
            self.get_logger().error('validate_and_plan 타임아웃')
            return None
        return result

    def _call_gripper(self, command: str) -> bool:
        gp = self.get_parameter
        if not self._gripper_client.wait_for_service(timeout_sec=gp('service_wait_timeout_sec').value):
            self.get_logger().error('/onrobot/sendCommand 서비스 응답 없음')
            return False
        req = SetCommand.Request()
        req.command = command
        future = self._gripper_client.call_async(req)
        result = self._wait_for_future(future, timeout_sec=10.0)
        if result is None:
            return False
        return result.success

    def _execute_trajectory(self, robot_trajectory, label: str) -> bool:
        """moveit_msgs/RobotTrajectory -> FollowJointTrajectory 액션 그대로 전달"""
        gp = self.get_parameter
        if not self._traj_action_client.wait_for_server(timeout_sec=gp('action_wait_timeout_sec').value):
            self.get_logger().error(f'{label} 컨트롤러 액션 서버 없음')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory
        if not goal.trajectory.points:
            self.get_logger().error(f'{label} 궤적이 비어있음')
            return False

        self.get_logger().info(f'{label} 실행 중 ({len(goal.trajectory.points)} 포인트)')
        send_future = self._traj_action_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, timeout_sec=10.0)
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'{label} 목표 거부됨')
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait_for_future(result_future, timeout_sec=60.0)
        if result is None:
            self.get_logger().error(f'{label} 결과 없음(타임아웃)')
            return False

        ok = result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if not ok:
            self.get_logger().error(f'{label} 실행 실패 (error_code={result.result.error_code})')
        return ok

    def _lift(self, selected_grasp) -> bool:
        """grasp pose 에서 +z 로 lift_height 만큼 이동하는 후보를 새로 만들어 재계획"""
        import copy
        lift_h = self.get_parameter('lift_height_m').value

        lifted = copy.deepcopy(selected_grasp)
        lifted.candidate.grasp_pose.position.z += lift_h
        lifted.candidate.pre_grasp_pose = selected_grasp.candidate.grasp_pose  # 현재 위치에서 시작
        lifted.candidate.candidate_id = -1  # 임시 후보 표시

        resp = self._call_validate_and_plan([lifted])
        if resp is None or not resp.success:
            return False
        return self._execute_trajectory(resp.approach, '(들어올리기)')

    def _move_to_fixed_joints(self, param_positions: str, param_time: str, label: str) -> bool:
        gp = self.get_parameter
        positions = gp(param_positions).value
        duration_sec = gp(param_time).value
        joint_names = gp('joint_names').value

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        goal.trajectory.points = [point]

        class _Wrapper:
            joint_trajectory = goal.trajectory

        return self._execute_trajectory(_Wrapper(), label)


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutionCoordinator()
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
