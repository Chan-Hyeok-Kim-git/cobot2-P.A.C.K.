# 이 파일은 음성 명령 전체 과정을 관리하는 ROS 2 노드이다.
# 순서: wake word 감지 → STT → 재난 찾기/추가 물품 찾기 → ROS topic 발행
import os
import threading
import json  # Python list를 ROS String topic으로 보내기 위한 JSON 변환에 사용
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from cobot2_voice.MicController import MicConfig, MicController
from cobot2_voice.get_keyword import KeywordParser
from cobot2_voice.stt import STT
from cobot2_voice.wakeup_word import WakeupWord
from cobot2_voice.wakeup_word_interrupt import WakeupWordInterrupt

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__("voice_command_node")

        # True면 STT/API 처리 중이니 이때 새 wake word가 들어와도 무시.
        self.busy = False

        # 로봇의 작업 상태 (False: 미작업/대기중, True: 작업중)
        self.is_working = False

        current_file_path = Path(__file__).resolve()
        # 설치된 패키지의 resource 폴더 위치를 찾기.
        model_dir = current_file_path.parents[1] / "resource" / "whisper_models"
        os.makedirs(model_dir, exist_ok=True)  # 폴더가 없으면 자동 생성

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

        # 물품 목록 JSON 문자열 및 추가 물품 전송 Publisher
        self.command_pub = self.create_publisher(String, "/voice/command", 10)
        self.add_item_pub = self.create_publisher(String, "/voice/additional_item", 10)

        # /robot/working 토픽 구독자 (로봇 작동 여부 수신)
        self.working_sub = self.create_subscription(
            Bool,
            "/robot/working",
            self.working_callback,
            10
        )

        self.finished_sub = self.create_subscription(
            Bool,
            "/robot/finished",
            self.finished_callback,
            10,
        )

        # WakeupWord 감지를 위한 타이머 (0.08초 주기로 감사)
        self.wakeup_timer = self.create_timer(0.08, self.check_wakeup)
        self.get_logger().info("Voice Command Node 초기화 완료 - Wake word 대기 시작")

    def working_callback(self, msg: Bool):
        """ /robot/working 토픽을 받아서 로봇의 작업 유무 상태 업데이트 """
        self.is_working = msg.data
        self.get_logger().info(f"로봇 작업 상태 업데이트: working = {self.is_working}")

    def finished_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("로봇 작업 완료. Voice node 종료")
            self.wakeup_timer.cancel()
            rclpy.shutdown()

    def check_wakeup(self):
        if self.busy:
            return

        # 1. 로봇이 작업 중이 아닐 때 (/robot/working == False) -> 재난 명령 처리
        if not self.is_working:
            if self.wakeup1.is_wakeup():
                self.busy = True
                self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_disaster_command,
                    daemon=True,
                ).start()

        # 2. 로봇이 작업 중일 때 (/robot/working == True) -> 추가 물품 명령 처리
        else:
            if self.wakeup2.is_wakeup():
                self.busy = True
                self.wakeup_timer.cancel()
                threading.Thread(
                    target=self.process_additional_item_command,
                    daemon=True,
                ).start()

    def process_disaster_command(self):
        """ STT 결과에서 재난을 찾고 해당 물품 목록을 ROS topic으로 발행 """
        try:
            DISASTER_SUPPLIES = {
                "지진": ["work_gloves", "first_aid_kit", "rope", "lantern"],
                "홍수": ["raincoat", "rain_boots_bag", "rope", "waterproof_tarp"],
                "화재": ["protective_mask", "whistle", "work_gloves", "safety_goggles"],
                "공습": ["protective_mask", "first_aid_kit", "lantern", "emergency_food"],
            }

            self.mic.close_stream()
            text = self.stt.speech2text()
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
            self.mic.open_stream()
            self.wakeup1.set_stream(self.mic.stream)
            self.wakeup2.set_stream(self.mic.stream)
            self.busy = False
            self.wakeup_timer.reset()

    def process_additional_item_command(self):
        should_shutdown = False
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

            should_shutdown = True
            
        except Exception as error:
            self.get_logger().error(f"추가 물품 처리 실패: {error}")

        finally:
            if should_shutdown:
                self.get_logger().info("추가 물품 발행 완료. Voice node 종료")
                rclpy.shutdown()
            else:
                self.mic.open_stream()
                self.wakeup1.set_stream(self.mic.stream)
                self.wakeup2.set_stream(self.mic.stream)
                self.busy = False
                self.wakeup_timer.reset()

    def destroy_node(self):
        # 프로그램 종료 전에 열려 있는 마이크를 안전하게 닫음.
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

        # shutdown이 스레드 내부에서 호출되었더라도 안전하게 node 정리
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()