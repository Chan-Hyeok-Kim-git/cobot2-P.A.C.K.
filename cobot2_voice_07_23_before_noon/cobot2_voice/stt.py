import os
import sounddevice as sd
from faster_whisper import WhisperModel

class STT:
    def __init__(self, model_dir, model_size="small", device="cpu", compute_type="int8"):
        """
        :param model_dir: 모델 파일들이 저장될 디렉토리 경로 (예: resource/whisper_models)
        """
        # download_root를 지정하면 해당 경로로 모델이 다운로드되거나 불러와집니다.
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=model_dir
        )
        self.duration = 5  # seconds
        self.samplerate = 16000  # 16kHz

    def speech2text(self):
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        
        audio = sd.rec(
            int(self.duration * self.samplerate), 
            samplerate=self.samplerate, 
            channels=1, 
            dtype='float32'
        )
        sd.wait()
        
        print("녹음 완료. 분석 중...")

        audio_data = audio.flatten()

        segments, info = self.model.transcribe(
            audio_data, 
            language="ko", 
            beam_size=1,
            vad_filter=True
        )

        transcript_text = "".join([segment.text for segment in segments]).strip()
        print("STT 결과: ", transcript_text)
        return transcript_text