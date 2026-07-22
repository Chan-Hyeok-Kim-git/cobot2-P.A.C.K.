# 이 파일은 음성 명령 전체 과정을 관리하는 ROS 2 노드이다.
# 순서: wake word 감지 → STT → 재난 찾기 → 물품 목록 topic 발행
# Wake word, STT, keyword extraction, ROS topic publication node.
import os
import threading
import json  # Python list를 ROS String topic으로 보내기 위한 JSON 변환에 사용한다.
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from cobot2_voice.MicController import MicConfig, MicController
from cobot2_voice.get_keyword import KeywordParser
from cobot2_voice.stt import STT
from cobot2_voice.wakeup_word import WakeupWord

# 음성 입력을 받아 재난별 비상 물품 목록을 발행하는 ROS 2 노드.
class VoiceCommandNode(Node):
    def __init__(self):
        # 필요한 마이크, AI 처리 객체, ROS publisher와 timer를 준비.
        super().__init__("voice_command_node")
        # True면 STT/API 처리 중이니 이때 새 wake word가 들어와도 무시.
        self.busy = False

        current_file_path = Path(__file__).resolve()
        # 설치된 패키지의 resource 폴더 위치를 찾기.
        model_dir = current_file_path.parents[1] / "resource" / "whisper_models"
        os.makedirs(model_dir, exist_ok=True)  # 폴더가 없으면 자동 생성

        # wake word 모델 학습 조건에 맞는 마이크 설정
        config = MicConfig(rate=48000, channels=1, buffer_size=24000)
        self.mic = MicController(config=config)
        self.mic.open_stream()

        # wake word 판별기와 STT, 재난 판별기 준비.
        self.wakeup = WakeupWord(config.buffer_size)
        self.wakeup.set_stream(self.mic.stream)
        self.stt = STT(model_dir, model_size="small", device="cpu", compute_type="int8")
        self.parser = KeywordParser()

        # 물품 목록 JSON 문자열을 다른 ROS 노드로 보낼 publisher.
        self.command_pub = self.create_publisher(String, "/voice/command", 10)
        # WakeupWord가 최소 48000 샘플을 읽기 때문에 한번에 1초정도 걸림. 그러므로 오디오를 1.1초 주기로 검사.
        self.wakeup_timer = self.create_timer(1.1, self.check_wakeup)
        self.get_logger().info("Wake word 대기 시작")

    def check_wakeup(self):
        # Timer가 주기적으로 호출. wake word가 감지되면 음성 처리를 시작.
        if self.busy:
            # 이미 처리 중이면 같은 방송을 여러 번 처리 안함.
            return
        if self.wakeup.is_wakeup():
            # 이후 timer 호출은 무시하도록 처리 상태를 킨다.
            self.busy = True
            # STT 중에는 wake word 판별을 멈춤.
            self.wakeup_timer.cancel()
            # 녹음과 API 호출은 오래 걸리므로 ROS timer를 막지 않도록 별도 thread에서 실행.
            threading.Thread(target=self.process_command, daemon=True).start()

    # STT 결과에서 재난 찾고 해당 물품 목록을 ROS topic으로 발행.
    def process_command(self):
        try:
            # 재난별로 미리 정한 비상 물품 목록.
            DISASTER_SUPPLIES = {
                "지진": ["장갑", "구급상자", "마스크", "랜턴"],
                "홍수": ["우비", "장화 주머니", "밧줄", "방수포"],
                "화재": ["방독면", "호루라기", "장갑", "보호 안경"],
                "공습": ["방독면", "구급상자", "랜턴", "비상식량"],
            }
            # sounddevice 기반 STT가 마이크를 열 수 있도록 wakeup stream을 먼저 닫음.
            # wake word용 PyAudio 마이크를 닫아 STT가 같은 마이크를 사용할 수 있게 함.
            self.mic.close_stream()
            # 5초 음성을 녹음하고 Whisper가 만든 텍스트를 받음.
            text = self.stt.speech2text()
            # STT 문장에서 지진·홍수 같은 허용 재난을 찾음.
            disaster = self.parser.extract(text)

            if disaster is None:
                self.get_logger().warn("인식 가능한 재난 유형을 찾지 못했습니다.")
                return
            
            # 재난 이름을 key로 사용해 대응 물품 list를 가져옴.
            supplies = DISASTER_SUPPLIES.get(disaster, [])
            message = String()
            # ROS String에는 Python list를 직접 넣을 수 없어서 JSON 문자열로 바꿈.
            message.data = json.dumps(supplies, ensure_ascii=False)
            # 구독 중인 다른 노드에 물품 목록을 전송한다.
            self.command_pub.publish(message)
            self.get_logger().info(f"명령 발행: {message.data}")
        except Exception as error:
            self.get_logger().error(f"음성 명령 처리 실패: {error}")
        finally:
            # 다음 재난 방송을 기다릴 수 있도록 wake word용 마이크를 다시 연다.
            self.mic.open_stream()
            self.wakeup.set_stream(self.mic.stream)
            self.busy = False
            self.wakeup_timer.reset()

    def destroy_node(self):
        # 프로그램 종료 전에 열려 있는 마이크를 안전하게 닫음.
        self.mic.close_stream()
        super().destroy_node()


# ROS 2 노드를 만들고 종료될 때까지 실행하는 프로그램 시작점.
def main():
    # ROS 2 통신 기능을 시작.
    rclpy.init()
    node = VoiceCommandNode()
    try:
        # timer와 topic 같은 ROS 이벤트를 계속 처리.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 프로그램을 닫기 전 노드와 ROS 2 자원을 차례로 정리.
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
