# 이 파일은 음성 명령 전체 과정을 관리하는 ROS 2 노드이다.
# 순서: wake word 감지 → STT → 재난 찾기 → 물품 목록 topic 발행
# Wake word, STT, keyword extraction, ROS topic publication node.
import os
import threading
import json  # Python list를 ROS String topic으로 보내기 위한 JSON 변환에 사용한다.
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from cobot2_voice.MicController import MicConfig, MicController
from cobot2_voice.get_keyword import KeywordParser
from cobot2_voice.stt import STT
from cobot2_voice.wakeup_word import WakeupWord

from cobot2_voice.wakeup_word_interrupt import WakeupWordInterrupt

STATE_WAIT_DISASTER = 0
STATE_WAIT_ADDITIONAL_ITEM = 1

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

        self.state = STATE_WAIT_DISASTER

        # wake word 모델 학습 조건에 맞는 마이크 설정
        config = MicConfig(rate=48000, channels=1, buffer_size=24000)
        self.mic = MicController(config=config)
        self.mic.open_stream()

        # wake word 판별기와 STT, 재난 판별기 준비.
        self.wakeup1 = WakeupWord(config.buffer_size)
        self.wakeup1.set_stream(self.mic.stream)

        self.stt = STT(model_dir, model_size="small", device="cpu", compute_type="int8")
        self.parser = KeywordParser()

        self.wakeup2 = WakeupWordInterrupt(threshold=0.3)
        self.wakeup2.set_stream(self.mic.stream)

        # 물품 목록 JSON 문자열을 다른 ROS 노드로 보낼 publisher.
        self.command_pub = self.create_publisher(String, "/voice/command", 10)
        self.add_item_pub = self.create_publisher(String, "/voice/additional_item", 10)
        # WakeupWord가 최소 48000 샘플을 읽기 때문에 한번에 1초정도 걸림. 그러므로 오디오를 1.1초 주기로 검사.
        self.wakeup_timer = self.create_timer(0.08, self.check_wakeup)
        self.get_logger().info("Wake word 대기 시작")

    def check_wakeup(self):
        if self.busy:
            return

        if self.state == STATE_WAIT_DISASTER:
            if self.wakeup1.is_wakeup():
                self.busy = True
                self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_disaster_command,
                    daemon=True,
                ).start()

        elif self.state == STATE_WAIT_ADDITIONAL_ITEM:
            if self.wakeup2.is_wakeup():
                self.busy = True
                self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_additional_item_command,
                    daemon=True,
                ).start()
    
    # STT 결과에서 재난 찾고 해당 물품 목록을 ROS topic으로 발행.
    def process_disaster_command(self):
        try:
            # 재난별로 미리 정한 비상 물품 목록.
            DISASTER_SUPPLIES = {
                "지진": ["work_gloves", "first_aid_kit", "rope", "lantern"],
                "홍수": ["raincoat", "rain_boots_bag", "rope", "waterproof_tarp"],
                "화재": ["protective_mask", "whistle", "work_gloves", "safety_goggles"],
                "공습": ["protective_mask", "first_aid_kit", "lantern", "emergency_food"],
            }
            # sounddevice 기반 STT가 마이크를 열 수 있도록 wakeup stream을 먼저 닫음.
            # wake word용 PyAudio 마이크를 닫아 STT가 같은 마이크를 사용할 수 있게 함.
            self.mic.close_stream()
            # 5초 음성을 녹음하고 Whisper가 만든 텍스트를 받음.
            text = self.stt.speech2text()
            # STT 문장에서 지진·홍수 같은 허용 재난을 찾음.
            disaster = self.parser.extract_disaster(text)

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
            self.state = STATE_WAIT_ADDITIONAL_ITEM

        except Exception as error:
            self.get_logger().error(f"음성 명령 처리 실패: {error}")
        finally:
            # 다음 재난 방송을 기다릴 수 있도록 wake word용 마이크를 다시 연다.
            self.mic.open_stream()
            self.wakeup1.set_stream(self.mic.stream)
            self.wakeup2.set_stream(self.mic.stream)  # 필수
            self.busy = False
            self.wakeup_timer.reset()

    def process_additional_item_command(self):
        ADDITIONAL_ITEMS = {"장갑" : "work_gloves", 
                            "구급상자" : "first_aid_kit", 
                            "밧줄" : "rope", 
                            "랜턴" : "lantern", 
                            "우비" : "raincoat", 
                            "장화주머니" :"rain_boots_bag", 
                            "방수포" : "waterproof_tarp", 
                            "방독면" : "protective_mask", 
                            "호루라기" : "whistle", 
                            "보호안경" : "safety_goggles", 
                            "비상식량" : "emergency_food",
                            "비상 식량" : "emergency_food", 
                            "보호 안경" : "safety_goggles", 
                            "장화 주머니" : "rain_boots_bag", 
                            "구급 상자" : "first_aid_kit"
                            }
        try:
            self.mic.close_stream()
            text = self.stt.speech2text()

            item = self.parser.extract_item(text)

            if not item:
                self.get_logger().warn("추가 물품을 찾지 못했습니다.")
                return

            supply = ADDITIONAL_ITEMS.get(item)
            message = String()
            message.data = supply
            self.add_item_pub.publish(message)
            self.get_logger().info(f"명령 발행: {message.data}")

        except Exception as error:
            self.get_logger().error(f"추가 물품 처리 실패: {error}")

        finally:
            self.mic.open_stream()
            self.wakeup1.set_stream(self.mic.stream)
            self.wakeup2.set_stream(self.mic.stream)
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
