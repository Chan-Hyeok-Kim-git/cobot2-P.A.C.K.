import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

class RobotStatusDummyNode(Node):
    def __init__(self):
        # 1. super() 괄호 수정
        super().__init__("robot_status_test_node")
        self.items = []
        self.finished_timer = None
        self.shutdown_timer = None

        # 퍼블리셔 생성
        self.work_pub = self.create_publisher(Bool, "/robot/working", 10)
        self.fin_pub = self.create_publisher(Bool, "/robot/finished", 10)

        # 구독자 생성
        # /robot/working 토픽 구독자 (로봇 작동 여부 수신)
        self.command_sub = self.create_subscription(String, "/voice/command", self.command_callback, 10)
        self.additional_sub = self.create_subscription(String, "/voice/additional_item", self.additional_callback, 10)

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

        if status and self.shutdown_timer is None:
            self.shutdown_timer = self.create_timer(0.2, self.shutdown_node)

    def command_callback(self, msg: String):
        self.items = json.loads(msg.data)
        self.get_logger().info(f"기본 물품 수신: {self.items}")

        self.publish_working(True)
        # 기존 자동 완료 timer가 있으면 제거
        if self.finished_timer is not None:
            self.finished_timer.cancel()
            self.destroy_timer(self.finished_timer)

        # 300초 뒤 auto_finish() 한 번 실행
        self.finished_timer = self.create_timer(300.0,self.auto_finish)

    def additional_callback(self, msg: String):
        self.items.append(msg.data)
        self.get_logger().info(f"추가 물품 수신: {msg.data}")
        self.get_logger().info(f"현재 물품 목록: {self.items}")

    def auto_finish(self):
        timer = self.finished_timer

        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
            self.finished_timer = None

        self.publish_working(False)
        self.publish_finished(True)
        self.get_logger().info("300초 경과. /robot/finished=True 발행")

    def shutdown_node(self):
        timer = self.shutdown_timer

        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
            self.shutdown_timer = None

        self.get_logger().info("Dummy node 종료")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatusDummyNode()

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