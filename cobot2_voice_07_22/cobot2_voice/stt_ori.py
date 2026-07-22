from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os

class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        # self.openai_api_key = openai_api_key
        self.duration = 5  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호


    def speech2text(self):
        # 녹음 설정
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16')
        sd.wait()
        print("녹음 완료. Whisper에 전송 중...")

        # 임시 WAV 파일 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name

        try:
            wav.write(temp_path, self.samplerate, audio)

            # Whisper API 호출
            with open(temp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=f)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        #print("STT 결과: ", transcript['text'])
        print("STT 결과: ", transcript.text)
        #return transcript['text']
        return transcript.text
