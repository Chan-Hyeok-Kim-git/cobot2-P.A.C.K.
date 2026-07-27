import json
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class RobotStatusDummyNode(Node):
    def __init__(self):
        super().__init__("robot_status_test_node")
        self.items = []
        self.finished_timer = None

        # 퍼블리셔 생성
        self.work_pub = self.create_publisher(Bool, "/robot/working", 10)
        self.fin_pub = self.create_publisher(Bool, "/robot/finished", 10)

        # 구독자 생성
        self.command_sub = self.create_subscription(String, "/voice/command", self.command_callback, 10)
        self.additional_sub = self.create_subscription(String, "/voice/additional_item", self.additional_callback, 10)

        self.get_logger().info("실행 완료, 물품 목록 수령 대기중... (키보드 명령어: 'f'=완료, 'w'=작업중)")

    def publish_working(self, status: bool):
        msg = Bool()
        msg.data = status
        self.work_pub.publish(msg)
        self.get_logger().info(f"Published /robot/working: {status}")

    def publish_finished(self, status: bool):
        msg = Bool()
        msg.data = status
        self.fin_pub.publish(msg)
        self.get_logger().info(f"Published /robot/finished: {status}")

    def command_callback(self, msg: String):
        try:
            self.items = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("JSON 파싱 실패! 올바른 JSON 문자열인지 확인하세요.")
            return

        self.get_logger().info(f"기본 물품 수신: {self.items}")

        # 작업 시작 시 상태 리셋
        self.publish_finished(False)
        self.publish_working(True)

        # 기존 타이머 취소
        self.reset_timer()

        # 300초 뒤 auto_finish 실행
        self.finished_timer = self.create_timer(300.0, self.auto_finish)

    def additional_callback(self, msg: String):
        self.items.append(msg.data)
        self.get_logger().info(f"추가 물품 수신: {msg.data}")
        self.get_logger().info(f"현재 물품 목록: {self.items}")

    def reset_timer(self):
        if self.finished_timer is not None:
            self.finished_timer.cancel()
            self.destroy_timer(self.finished_timer)
            self.finished_timer = None

    def complete_task(self, reason: str):
        """타이머나 키보드 입력 시 작업을 완료 처리하는 공통 메서드"""
        self.reset_timer()
        self.publish_working(False)
        self.publish_finished(True)
        self.get_logger().info(f"[{reason}] /robot/working=False, /robot/finished=True 발행 완료.")

        self.destroy_node()
        rclpy.shutdown()
        
    def auto_finish(self):
        self.complete_task("300초 자동 완료")


def keyboard_loop(node: RobotStatusDummyNode):
    """키보드 입력을 전담하는 스레드 함수"""
    while rclpy.ok():
        try:
            user_input = input().strip().lower()
            if user_input == 'f':
                node.complete_task("사용자 'f' 입력 수동 완료")
            elif user_input == 'w':
                node.publish_working(True)
        except (EOFError, KeyboardInterrupt):
            break


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatusDummyNode()

    # 입력 전담 데몬 스레드 시작
    input_thread = threading.Thread(target=keyboard_loop, args=(node,), daemon=True)
    input_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()