import json
import os
import threading


from pathlib import Path
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from cobot2_voice.get_keyword import KeywordParser
from cobot2_voice.MicController import MicConfig, MicController
from cobot2_voice.stt import STT
from cobot2_voice.tts import TTS
from cobot2_voice.wakeup_word import WakeupWord
from cobot2_voice.wakeup_word_interrupt import WakeupWordInterrupt

from ament_index_python.packages import get_package_share_directory

STATE_WAIT_DISASTER = 0
STATE_WAIT_ADDITIONAL_ITEM = 1
STATE_WAIT_FINISHED = 2

PACKAGE_NAME = "cobot2_voice"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
RESOURCE_PATH = os.path.join(PACKAGE_PATH, "resource")
ENV_PATH = os.path.join(RESOURCE_PATH, ".env")
load_dotenv(dotenv_path=ENV_PATH)
openai_api_key = os.getenv("OPENAI_API_KEY")

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__("voice_command_node")

        self.state = STATE_WAIT_DISASTER
        self.current_disaster = None
        self.busy = False
        self.is_working = False
        self.is_shutting_down = False
        self.is_announcing = False
        
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

        self.tts = TTS(openai_api_key)

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
        if not msg.data or self.is_announcing:
            return

        self.state = STATE_WAIT_FINISHED
        self.is_shutting_down = True
        self.is_announcing = True
        self.get_logger().info("로봇 작업 완료 수신. 완료 안내 방송 시작.")

        threading.Thread(
            target=self.announce_completion,
            daemon=True,
        ).start()

        # 타이머 멈춤
        if self.wakeup_timer:
            self.wakeup_timer.cancel()
            self.destroy_timer(self.wakeup_timer)
            self.wakeup_timer = None

    def check_wakeup(self):
        if self.state == STATE_WAIT_FINISHED:
            return  # wakeup1, wakeup2 모두 중단

        if self.busy or self.is_shutting_down:
            return

        if self.state == STATE_WAIT_DISASTER:
            if self.wakeup1.is_wakeup():
                self.busy = True
                if self.wakeup_timer:
                    self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_disaster_command, daemon=True
                ).start()
        elif self.state == STATE_WAIT_ADDITIONAL_ITEM:
            if not self.is_working:
                return

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
            "홍수": ["whistle", "work_gloves",  "rope",  "protective_mask"], #  "work_gloves",  "rope",  "protective_mask"
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

            self.current_disaster = disaster
            supplies = DISASTER_SUPPLIES.get(disaster, [])
            message = String()
            message.data = json.dumps(supplies, ensure_ascii=False)

            self.command_pub.publish(message)
            self.get_logger().info(f"재난 명령 발행: {message.data}")
            self.state = STATE_WAIT_ADDITIONAL_ITEM

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
            "로프" : "rope",
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
        item_sent = False
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
                item_sent = True

        except Exception as error:
            self.get_logger().error(f"추가 물품 처리 실패: {error}")

        finally:
            if item_sent:
                self.get_logger().info("추가 물품 발행 완료. 로봇 완료 신호 대기.")
                self.state = STATE_WAIT_FINISHED
                self.busy = False

                if self.wakeup_timer:
                    self.wakeup_timer.cancel()

            elif not self.is_shutting_down:
                self.mic.open_stream()
                self.wakeup1.set_stream(self.mic.stream)
                self.wakeup2.set_stream(self.mic.stream)
                self.busy = False
                if self.wakeup_timer:
                    self.wakeup_timer.reset()


    def announce_completion(self):
        try:
            self.safe_close_mic()

            base_script = "비상 물품 준비가 완료되었습니다. 안전한 장소로 대피하세요."

            scripts = {
                "지진": "지진 대비 비상 물품 준비가 완료되었습니다. 우선 헬멧을 착용하시고 가방을 챙겨 대피하세요.",
                "홍수": "홍수 대비 비상 물품 준비가 완료되었습니다. 우비를 입으시고 장화를 신으신 후 가방을 챙겨 안전한 장소로 대피하세요.",
                "화재": "화재 대비 비상 물품 준비가 완료되었습니다. 마스크를 반드시 착용하시고 가방을 챙겨 안전한 장소로 대피하세요.",
                "공습": "공습 대비 비상 물품 준비가 완료되었습니다. 안내에 따라 가방을 챙겨 방공호로 대피하세요.",
            }

            if self.current_disaster is None:
                text = base_script
            else:
                text = scripts.get(self.current_disaster, base_script)
            self.tts.speak(text)  # 재생 종료까지 기다리는 함수

        except Exception as error:
            self.get_logger().error(f"완료 안내 방송 실패: {error}")

        finally:
            self.get_logger().info("완료 안내 종료. Voice node 종료")
            rclpy.shutdown()
    def destroy_node(self):
        # 노드 파괴 시 마이크 자원 확실하게 정리
        self.safe_close_mic()
        super().destroy_node()


def main():
    rclpy.init()
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 자원 정리 및 rclpy 종료는 main의 finally에서 일괄 처리
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()