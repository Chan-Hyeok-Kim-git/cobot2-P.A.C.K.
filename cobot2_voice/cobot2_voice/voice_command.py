"""Wake word, STT, keyword extraction, ROS topic publication node."""
import os
import threading
import json

import rclpy
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from rclpy.node import Node
from std_msgs.msg import String

from cobot2_voice.MicController import MicConfig, MicController
from cobot2_voice.get_keyword import KeywordParser
from cobot2_voice.stt import STT
from cobot2_voice.wakeup_word import WakeupWord


class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__("voice_command_node")
        self.busy = False

        package_path = get_package_share_directory("cobot2_voice")
        load_dotenv(os.path.join(package_path, "resource", ".env"))
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY를 resource/.env에 설정하세요.")

        config = MicConfig(rate=48000, channels=1, buffer_size=24000)
        self.mic = MicController(config=config)
        self.mic.open_stream()

        self.wakeup = WakeupWord(config.buffer_size)
        self.wakeup.set_stream(self.mic.stream)
        self.stt = STT(openai_api_key)
        self.parser = KeywordParser(openai_api_key)

        self.command_pub = self.create_publisher(String, "/voice/command", 10)
        # WakeupWord가 약 1초 오디오를 읽으므로 1.1초 주기로 검사한다.
        self.wakeup_timer = self.create_timer(1.1, self.check_wakeup)
        self.get_logger().info("Wake word 대기 시작")

    def check_wakeup(self):
        if self.busy:
            return
        if self.wakeup.is_wakeup():
            self.busy = True
            self.wakeup_timer.cancel()
            threading.Thread(target=self.process_command, daemon=True).start()

    def process_command(self):
        try:
            DISASTER_SUPPLIES = {
                "지진": ["장갑", "구급상자", "마스크", "랜턴"],
                "홍수": ["우비", "장화 주머니", "밧줄", "방수포"],
                "화재": ["방독면", "호루라기", "장갑", "보호 안경"],
                "공습": ["방독면", "구급상자", "랜턴", "비상식량"],
            }
            # sounddevice 기반 STT가 마이크를 열 수 있도록 wakeup stream을 먼저 닫는다.
            self.mic.close_stream()
            text = self.stt.speech2text()
            disaster = self.parser.extract(text)

            if disaster is None:
                self.get_logger().warn("인식 가능한 재난 유형을 찾지 못했습니다.")
                return
            
            supplies = DISASTER_SUPPLIES.get(disaster, [])
            message = String()
            message.data = json.dumps(supplies, ensure_ascii=False)
            self.command_pub.publish(message)
            self.get_logger().info(f"명령 발행: {message.data}")
        except Exception as error:
            self.get_logger().error(f"음성 명령 처리 실패: {error}")
        finally:
            self.mic.open_stream()
            self.wakeup.set_stream(self.mic.stream)
            self.busy = False
            self.wakeup_timer.reset()

    def destroy_node(self):
        self.mic.close_stream()
        super().destroy_node()


def main():
    rclpy.init()
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
