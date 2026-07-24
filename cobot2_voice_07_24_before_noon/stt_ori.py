# 마이크 음성을 녹음하고 OpenAI STT로 글자로 바꾸는 파일이다.
from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os

class STT:
    """마이크 소리를 녹음한 뒤 Whisper API로 텍스트로 바꾸는 클래스."""
    def __init__(self, openai_api_key):
        # OpenAI API를 호출할 클라이언트를 만든다.
        self.client = OpenAI(api_key=openai_api_key)
        # self.openai_api_key = openai_api_key
        self.duration = 5  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호


    def speech2text(self):
        """5초간 녹음하고, 인식된 문장을 문자열로 반환한다."""
        # 녹음 설정
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        # 녹음할 전체 표본 수 = 녹음 시간(초) × 초당 표본 수이다.
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16')
        # 녹음이 끝날 때까지 다음 코드 실행을 기다린다.
        sd.wait()
        print("녹음 완료. Whisper에 전송 중...")

        # 임시 WAV 파일 저장
        # Whisper API는 파일을 받으므로, 잠시 쓸 WAV 파일 경로를 만든다.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name

        try:
            # 메모리에 있는 녹음 데이터를 실제 WAV 파일로 저장한다.
            wav.write(temp_path, self.samplerate, audio)

            # Whisper API 호출
            with open(temp_path, "rb") as f:
                # 음성 파일을 Whisper에 보내고 인식 결과를 받는다.
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=f)
        finally:
            # 성공·실패와 관계없이 임시 파일을 지워 디스크에 남기지 않는다.
            if os.path.exists(temp_path):
                os.remove(temp_path)

        #print("STT 결과: ", transcript['text'])
        print("STT 결과: ", transcript.text)
        #return transcript['text']
        # 다음 단계가 사용할 수 있도록 STT 결과 문장을 돌려준다.
        return transcript.text
