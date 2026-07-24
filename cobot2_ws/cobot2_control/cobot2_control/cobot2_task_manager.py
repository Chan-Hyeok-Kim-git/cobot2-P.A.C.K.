#!/usr/bin/env python3
"""전체 파이프라인 조율(coordinator) 노드. (최종 단순화 — 스캔 트리거 1회 방식)

이번 버전에서 바뀐 것
  웨이포인트마다 좌표를 주고받던 방식(scan_goto → scan_reached, 웨이포인트당
  1왕복씩 총 6번)이 cobot2_move 쪽에서 반복적으로(4번 연속, 항상 같은 지점)
  콜백 전달이 막히는 문제가 있었다. 여러 각도로 원인을 좁혀봤지만
  (폴링 제거, 워커스레드→콜백 직접실행 전환 등) 계속 재현되어, "메시지를
  여러 번 주고받는 구조" 자체를 없앴다.

  이제 이 노드는 스캔 좌표를 전혀 모른다 — 그건 cobot2_move.py 안에
  SCAN_WAYPOINTS로 하드코딩되어 있다. 이 노드는:
    1) active_item 발행 (cobot2_grasp에게 "이 class 찾는 중"이라고 알림)
    2) /task/start_scan 트리거 1번만 발행
    3) cobot2_move가 내부적으로 6곳을 전부 돌면서, 그 사이 cobot2_grasp이
       물체를 찾으면 자체적으로 파지까지 진행 → /execution_result 발행
    4) /execution_result(파지 성공/실패) 또는 /task/scan_complete(6곳 다
       돌았는데 못 찾음) 둘 중 먼저 오는 걸 보고 다음 물품으로 넘어감

큐가 하나만 있으면 되는 이유
  물품은 순서가 있어 큐(item_queue)가 맞지만, point cloud/스캔 좌표는
  이제 이 노드를 거치지 않는다. 이 노드는 "지금 이 class 차례다"라는
  gate 신호(active_item)와 "스캔 시작해"라는 트리거만 관리하면 된다.
"""

from collections import deque
from enum import Enum, auto
from typing import Optional
import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String


# ============================================================
# 사용자 설정
# ============================================================

# 음성이 이미 vision class 이름(emergency_food, first_aid_kit, ...) 그대로
# 보내주므로 별도 한글 매핑이 필요 없다. 오타/미등록 이름만 걸러내는 용도.
VALID_CLASS_NAMES = {
    "emergency_food", "first_aid_kit", "lantern", "protective_mask",
    "rain_boots_bag", "raincoat", "rope", "safety_goggles",
    "waterproof_tarp", "whistle", "work_gloves",
}

TICK_HZ = 5.0

# 물품 하나당 "스캔 시작 → (성공/실패/스캔완료) 결과" 전체를 기다리는 최대 시간.
# cobot2_move 쪽에서 6곳 순회(웨이포인트당 이동시간 + SCAN_DWELL_SEC 2초)를
# 다 돌아도 넉넉하게 잡아둔 안전장치 — 이것마저 안 오면 노드/통신 문제로 간주.
ITEM_TIMEOUT_SEC = 90.0

VOICE_ITEM_LIST_TOPIC = "/voice/command"  # ★ 수정: 음성팀 확정 토픽명 반영 (기존 /voice/item_list)
ACTIVE_ITEM_TOPIC = "/task/active_item"
START_SCAN_TOPIC = "/task/start_scan"
SCAN_COMPLETE_TOPIC = "/task/scan_complete"
EXECUTION_RESULT_TOPIC = "/execution_result"


# ============================================================
# 상태
# ============================================================

class TaskState(Enum):
    IDLE = auto()       # 다음 물품 pop 대기
    SEARCHING = auto()  # active_item + start_scan 발행 후 결과(성공/실패/스캔완료) 대기


# ============================================================
# 노드
# ============================================================

class TaskManagerNode(Node):

    def __init__(self) -> None:
        super().__init__("cobot2_task_manager")

        self.item_queue: deque[str] = deque()
        self.current_item: Optional[str] = None
        self.state = TaskState.IDLE
        self._item_deadline = None

        self.create_subscription(String, VOICE_ITEM_LIST_TOPIC, self.on_item_list, 10)
        self.create_subscription(String, EXECUTION_RESULT_TOPIC, self.on_execution_result, 10)
        self.create_subscription(String, SCAN_COMPLETE_TOPIC, self.on_scan_complete, 10)

        self.active_item_pub = self.create_publisher(String, ACTIVE_ITEM_TOPIC, 10)
        self.start_scan_pub = self.create_publisher(String, START_SCAN_TOPIC, 10)

        self.create_timer(1.0 / TICK_HZ, self.tick)

        self.get_logger().info("TaskManagerNode 시작 — 음성 리스트 대기 중")

    # ─────────────────────────────────────────────────────────
    # 콜백
    # ─────────────────────────────────────────────────────────

    def on_item_list(self, msg: String) -> None:
        """음성 리스트 도착. 콤마 구분 문자열 가정 (배열 msg면 파싱만 교체).
        음성이 이미 vision class 이름 그대로 보내주므로 별도 변환 없이 그대로 쓴다.
        VALID_CLASS_NAMES에 없는 이름이 오면(오타/미등록 클래스) 걸러내고 경고만 남긴다."""
        raw_names = [s.strip() for s in msg.data.split(",") if s.strip()]

        valid = [name for name in raw_names if name in VALID_CLASS_NAMES]
        unknown = [name for name in raw_names if name not in VALID_CLASS_NAMES]
        if unknown:
            self.get_logger().warning(
                f"등록되지 않은 class 이름 무시: {unknown} (VALID_CLASS_NAMES 확인)"
            )

        self.item_queue.extend(valid)
        self.get_logger().info(
            f"음성 리스트 수신: {raw_names} → 큐에 추가={valid} "
            f"| 큐 크기={len(self.item_queue)}"
        )

    def on_execution_result(self, msg: String) -> None:
        """cobot2_move가 파지 사이클 완료(성공/실패) 시 발행."""
        if self.state != TaskState.SEARCHING:
            return

        try:
            result = json.loads(msg.data)
            success = bool(result.get("success", False))
            reason = result.get("reason", "")
        except json.JSONDecodeError:
            success, reason = False, "execution_result JSON 파싱 실패"

        status = "성공" if success else "실패"
        self.get_logger().info(f"[{self.current_item}] 파지-수납 {status} ({reason}) → 다음 물품으로")
        self._finish_current_item()

    def on_scan_complete(self, msg: String) -> None:
        """cobot2_move가 6곳 웨이포인트를 다 돌았는데 아무것도 못 찾고 끝냈을 때 발행."""
        if self.state != TaskState.SEARCHING:
            return
        status = msg.data  # 'done' 또는 'error'
        self.get_logger().warning(
            f"[{self.current_item}] 스캔 완료({status}) — 이번 선반에서 못 찾음, 다음 물품으로"
        )
        self._finish_current_item()

    # ─────────────────────────────────────────────────────────
    # 물품 탐색 흐름
    # ─────────────────────────────────────────────────────────

    def _start_next_item(self) -> None:
        item = self.item_queue.popleft()
        self.current_item = item
        self.state = TaskState.SEARCHING
        self._item_deadline = self.get_clock().now() + Duration(seconds=ITEM_TIMEOUT_SEC)

        self.get_logger().info(
            f"[{item}] 탐색 시작 → active_item 발행 + 스캔 트리거 | 남은 큐={len(self.item_queue)}"
        )
        self.active_item_pub.publish(String(data=item))
        self.start_scan_pub.publish(String(data="start"))

    def _finish_current_item(self) -> None:
        self.active_item_pub.publish(String(data=""))  # gate CLOSE
        self.current_item = None
        self.state = TaskState.IDLE
        self._item_deadline = None

    # ─────────────────────────────────────────────────────────
    # 5Hz tick
    # ─────────────────────────────────────────────────────────

    def tick(self) -> None:
        if self.state == TaskState.IDLE:
            if self.item_queue:
                self._start_next_item()

        elif self.state == TaskState.SEARCHING:
            if self._item_deadline is not None and self.get_clock().now() >= self._item_deadline:
                self.get_logger().error(
                    f"[{self.current_item}] {ITEM_TIMEOUT_SEC:.0f}초 안에 결과 없음 "
                    f"— cobot2_move 쪽 문제로 보고 이 물품 탐색 포기, 다음 물품으로"
                )
                self._finish_current_item()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()