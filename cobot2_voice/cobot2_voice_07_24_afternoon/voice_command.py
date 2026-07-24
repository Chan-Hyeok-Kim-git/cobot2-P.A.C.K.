import json
import os
import threading
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from cobot2_voice.get_keyword import KeywordParser
from cobot2_voice.MicController import MicConfig, MicController
from cobot2_voice.stt import STT
from cobot2_voice.wakeup_word import WakeupWord
from cobot2_voice.wakeup_word_interrupt import WakeupWordInterrupt


class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__("voice_command_node")

        self.busy = False
        self.is_working = False
        self.is_shutting_down = False  # 종료 진행 여부 플래그

        current_file_path = Path(__file__).resolve()
        model_dir = current_file_path.parents[1] / "resource" / "whisper_models"
        os.makedirs(model_dir, exist_ok=True)

        config = MicConfig(rate=48000, channels=1, buffer_size=24000)
        self.mic = MicController(config=config)
        self.mic.open_stream()

        self.wakeup1 = WakeupWord(config.buffer_size)
        self.wakeup1.set_stream(self.mic.stream)

        self.stt = STT(model_dir, model_size="small", device="cpu", compute_type="int8")
        self.parser = KeywordParser()

        self.wakeup2 = WakeupWordInterrupt(threshold=0.3)
        self.wakeup2.set_stream(self.mic.stream)

        self.command_pub = self.create_publisher(String, "/voice/command", 10)
        self.add_item_pub = self.create_publisher(String, "/voice/additional_item", 10)

        self.working_sub = self.create_subscription(
            Bool, "/robot/working", self.working_callback, 10
        )

        self.finished_sub = self.create_subscription(
            Bool, "/robot/finished", self.finished_callback, 10
        )

        self.wakeup_timer = self.create_timer(0.08, self.check_wakeup)
        self.get_logger().info("음성 인식 노드 초기화 완료 - Wakeup word 대기 시작")

    def safe_close_mic(self):
        """마이크 스트림을 안전하게 닫는 헬퍼 함수"""
        try:
            if hasattr(self, 'mic') and self.mic:
                self.mic.close_stream()
        except Exception as e:
            self.get_logger().warn(f"마이크 닫기 중 경고: {e}")

    def working_callback(self, msg: Bool):
        self.is_working = msg.data
        self.get_logger().info(f"로봇 작업 상태 업데이트: working = {self.is_working}")

    def finished_callback(self, msg: Bool):
        if msg.data and not self.is_shutting_down:
            self.is_shutting_down = True
            self.get_logger().info("로봇 작업 완료 수신. Voice node 종료 절차 시작")
            
            # 타이머 멈춤
            if self.wakeup_timer:
                self.wakeup_timer.cancel()
                self.destroy_timer(self.wakeup_timer)
                self.wakeup_timer = None
            
            # spin()을 안전하게 탈출하도록 예외 발생
            raise ExternalShutdownException()

    def check_wakeup(self):
        if self.busy or self.is_shutting_down:
            return

        if not self.is_working:
            if self.wakeup1.is_wakeup():
                self.busy = True
                if self.wakeup_timer:
                    self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_disaster_command, daemon=True
                ).start()
        else:
            if self.wakeup2.is_wakeup():
                self.busy = True
                if self.wakeup_timer:
                    self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_additional_item_command, daemon=True
                ).start()

    def process_disaster_command(self):
        DISASTER_SUPPLIES = {
            "지진": ["work_gloves", "first_aid_kit", "rope", "lantern"],
            "홍수": ["raincoat", "rain_boots_bag", "rope", "waterproof_tarp"],
            "화재": ["protective_mask", "whistle", "work_gloves", "safety_goggles"],
            "공습": ["protective_mask", "first_aid_kit", "lantern", "emergency_food"],
        }
        try:
            self.safe_close_mic()
            text = self.stt.speech2text()
            text = text.replace("화제", "화재")
            disaster = self.parser.extract_disaster(text)

            if disaster is None:
                self.get_logger().warn("인식 가능한 재난 유형을 찾지 못했습니다.")
                return

            supplies = DISASTER_SUPPLIES.get(disaster, [])
            message = String()
            message.data = json.dumps(supplies, ensure_ascii=False)

            self.command_pub.publish(message)
            self.get_logger().info(f"재난 명령 발행: {message.data}")

        except Exception as error:
            self.get_logger().error(f"음성 명령 처리 실패: {error}")
        finally:
            if not self.is_shutting_down:
                self.mic.open_stream()
                self.wakeup1.set_stream(self.mic.stream)
                self.wakeup2.set_stream(self.mic.stream)
                self.busy = False
                if self.wakeup_timer:
                    self.wakeup_timer.reset()

    def process_additional_item_command(self):
        ADDITIONAL_ITEMS = {
            "장갑": "work_gloves",
            "구급상자": "first_aid_kit",
            "밧줄": "rope",
            "랜턴": "lantern",
            "우비": "raincoat",
            "장화주머니": "rain_boots_bag",
            "방수포": "waterproof_tarp",
            "방독면": "protective_mask",
            "호루라기": "whistle",
            "보호안경": "safety_goggles",
            "비상식량": "emergency_food",
            "비상 식량": "emergency_food",
            "보호 안경": "safety_goggles",
            "장화 주머니": "rain_boots_bag",
            "구급 상자": "first_aid_kit",
        }
        should_shutdown = False
        try:
            self.safe_close_mic()
            text = self.stt.speech2text()

            item = self.parser.extract_item(text)
            if not item:
                self.get_logger().warn("추가 물품을 찾지 못했습니다.")
                return

            supply = ADDITIONAL_ITEMS.get(item)
            if supply:
                message = String()
                message.data = supply
                self.add_item_pub.publish(message)
                self.get_logger().info(f"명령 발행: {message.data}")
                should_shutdown = True

        except Exception as error:
            self.get_logger().error(f"추가 물품 처리 실패: {error}")

        finally:
            if should_shutdown:
                self.get_logger().info("추가 물품 발행 완료. 음성 인식 노드 종료")
                self.is_shutting_down = True
                # 안전한 메인 스레드 종료 유도
                raise ExternalShutdownException()
            elif not self.is_shutting_down:
                self.mic.open_stream()
                self.wakeup1.set_stream(self.mic.stream)
                self.wakeup2.set_stream(self.mic.stream)
                self.busy = False
                if self.wakeup_timer:
                    self.wakeup_timer.reset()

    def destroy_node(self):
        # 노드 파괴 시 마이크 자원 확실하게 정리
        self.safe_close_mic()
        super().destroy_node()


def main():
    rclpy.init()
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 노드 자원 정리 및 rclpy 종료는 main의 finally에서 일괄 처리
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()