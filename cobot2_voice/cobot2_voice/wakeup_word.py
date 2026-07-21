import os
import numpy as np
from scipy.signal import resample
import tflite_runtime.interpreter as tflite # 또는 import tensorflow.lite as tflite
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "cobot2_voice"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

MODEL_NAME = "soundclassifier_with_metadata.tflite"
MODEL_PATH = os.path.join(PACKAGE_PATH, f"resource/{MODEL_NAME}")

class WakeupWord:
    def __init__(self, buffer_size):
        # MicController 기본값은 48 kHz. 모델에 약 1초를 제공한다.
        self.buffer_size = max(buffer_size, 48000)
        self.stream = None
        
        # TFLite 모델 로드 및 인터프리터 초기화
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 모델 입력 텐서 정보
        self.input_shape = self.input_details[0]['shape'] # 예: [1, 15600] 또는 [1, 44100] 등

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        # 모델은 약 1초 길이의 44,032-sample float32 오디오를 받는다.
        # 48 kHz 마이크 입력을 16 kHz로 낮추면 음성 길이와 피치가 달라져
        # Teachable Machine AudioClassifier의 학습 입력과 맞지 않는다.
        raw_data = self.stream.read(self.buffer_size, exception_on_overflow=False)
        audio_chunk = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 입력 길이에 맞춰 직접 리샘플링한다.
        target_len = self.input_shape[-1]
        audio_chunk = resample(audio_chunk, target_len).astype(np.float32)
        input_data = np.expand_dims(audio_chunk, axis=0)

        # 4. 추론 실행
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 5. 결과 확인 (★ Index 0: Disaster_situation 확률 ★)
        disaster_prob = output_data[0][0]
        bg_noise_prob = output_data[0][1]
        
        print(f"[감지 중...] 재난상황 확률: {disaster_prob:.2f} | 배경소음 확률: {bg_noise_prob:.2f}")

        # Disaster_situation(인덱스 0) 확률이 70% 이상일 때만 Wakeup!
        if disaster_prob > 0.7:
            print("🚀 [Wakeup!] 재난 방송 키워드 감지!")
            
            return True
            
        return False
