#!/usr/bin/env python3

"""전체 파이프라인 조율 노드.



동작 순서

---------

1. 음성 물품 목록에서 현재 물품을 꺼낸다.

2. /task/active_item과 /task/start_scan을 발행한다.

3. cobot2_move가 모든 스캔 웨이포인트를 순회한다.

4. cobot2_grasp는 스캔 중 파지 계획만 저장한다.

5. /task/scan_complete 수신 후 cobot2_grasp가 직접 movejx 파지를 실행한다.

6. 이 노드는 /grasp_result 성공/실패만 기다린 뒤 다음 물품으로 넘어간다.



/task/scan_complete는 cobot2_grasp가 처리한다. TaskManager가 이를 곧바로

"못 찾음"으로 처리하면 저장된 파지 계획이 실행되기 전에 active_item이

해제되므로, TaskManager는 scan_complete를 구독하지 않는다.

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



# 물품 하나당 "스캔 시작 → grasp_result" 전체를 기다리는 최대 시간.

# cobot2_move의 4곳 순회와 스캔 후 직접 파지 시간을 포함해 넉넉하게 잡는다.

ITEM_TIMEOUT_SEC = 90.0



VOICE_ITEM_LIST_TOPIC = "/voice/command"  # ★ 수정: 음성팀 확정 토픽명 반영 (기존 /voice/item_list)

ACTIVE_ITEM_TOPIC = "/task/active_item"

START_SCAN_TOPIC = "/task/start_scan"

GRASP_RESULT_TOPIC = "/grasp_result"





# ============================================================

# 상태

# ============================================================



class TaskState(Enum):

    IDLE = auto()       # 다음 물품 pop 대기

    SEARCHING = auto()  # active_item + start_scan 발행 후 grasp_result 대기





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

        self.create_subscription(String, GRASP_RESULT_TOPIC, self.on_grasp_result, 10)



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



    def on_grasp_result(self, msg: String) -> None:

        """cobot2_grasp가 스캔 후 직접 파지 결과를 발행한다."""

        if self.state != TaskState.SEARCHING:

            return



        result_class = ""

        try:

            result = json.loads(msg.data)

            if not isinstance(result, dict):

                raise ValueError("grasp_result 최상위 값이 JSON object가 아님")

            success = bool(result.get("success", False))

            reason = str(result.get("reason", ""))

            result_class = str(result.get("class_name", ""))

        except (json.JSONDecodeError, ValueError) as exc:

            success = False

            reason = f"grasp_result 파싱 실패: {exc}"



        if result_class and result_class != self.current_item:

            self.get_logger().warning(

                f"오래된 grasp_result 무시: result={result_class!r}, "

                f"current={self.current_item!r}"

            )

            return



        status = "성공" if success else "실패"

        self.get_logger().info(

            f"[{self.current_item}] 스캔 후 직접 파지 {status} ({reason}) "

            "→ 다음 물품으로"

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

                    f"— scan_complete 또는 grasp_result 통신 문제로 보고 "

                    f"이 물품 탐색 포기, 다음 물품으로"

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