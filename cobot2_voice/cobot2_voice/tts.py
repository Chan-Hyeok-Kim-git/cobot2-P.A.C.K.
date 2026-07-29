import io
import os
import time
import pygame
from openai import OpenAI

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv

PACKAGE_NAME = "cobot2_voice"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
RESOURCE_PATH = os.path.join(PACKAGE_PATH, "resource")
ENV_PATH = os.path.join(RESOURCE_PATH, ".env")
load_dotenv(dotenv_path=ENV_PATH)
openai_api_key = os.getenv("OPENAI_API_KEY")

class TTS:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise RuntimeError("resource/.env에 OPENAI_API_KEY가 없습니다.")

        self.client = OpenAI(api_key=openai_api_key)

        pygame.mixer.init()

    def speak(self, text: str, voice: str = "nova", model = "tts-1"):
        """
        OpenAI TTS로 텍스트를 음성으로 변환하고 재생이 끝날 때까지 대기합니다.
        
        :param text: 읽을 문자열
        :param voice: OpenAI 음성 모델 종류 (alloy, echo, fable, onyx, nova, shimmer)
        :param model: tts-1 (빠름, 실시간용) 또는 tts-1-hd (고품질)
        """
        if not text or not text.strip():
            return

        try:
            # 1. OpenAI TTS API 호출 (스트림 바이너리 수신)
            response = self.client.audio.speech.create(
                model=model, voice=voice, input=text
            )
            
            # 2. 파일 저장 없이 메모리 버퍼(BytesIO)에 담아 즉시 재생
            audio_data = io.BytesIO(response.content)
            pygame.mixer.music.load(audio_data)
            pygame.mixer.music.play()

            # 3. 재생이 완전히 끝날 때까지 블로킹 (VoiceCommandNode의 요구사항 만족)
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"[TTS Error] 음성 합성 또는 재생 중 오류 발생: {e}")
        finally:
            # 다음 재생을 위한 메모리 해제 및 믹서 정리
            pygame.mixer.music.unload()


if __name__ == "__main__":
    # 간단 테스트용
    try:
        tts = TTS(openai_api_key)
        tts.speak("홍수 대비 비상 물품 준비가 완료되었습니다. 우비를 입으시고 장화를 신으신 후 가방을 챙겨 안전한 장소로 대피하세요.")
    except Exception as err:
        print(f"테스트 실패: {err}")