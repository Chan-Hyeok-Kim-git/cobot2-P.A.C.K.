# 마이크 소리에서 미리 학습한 wake word(재난 방송 신호)를 찾는 파일이다.
import os
import numpy as np
from scipy.signal import resample
import tflite_runtime.interpreter as tflite
from ament_index_python.packages import get_package_share_directory

# ROS 패키지 이름과 설치된 resource 폴더 위치를 구한다.
PACKAGE_NAME = "cobot2_voice"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

# TFLite는 작은 기기에서도 실행할 수 있는 AI 모델 파일 형식이다.
MODEL_NAME = "soundclassifier_with_metadata.tflite"
MODEL_PATH = os.path.join(PACKAGE_PATH, f"resource/{MODEL_NAME}")

class WakeupWord:
    """TFLite 모델로 현재 마이크 소리에 wake word가 있는지 판단한다."""
    def __init__(self, buffer_size):
        """모델을 한 번만 불러오고, 나중에 마이크 stream을 받을 준비를 한다."""
        # MicController 기본값은 48 kHz. 모델에 약 1초를 제공한다.
        self.buffer_size = max(buffer_size, 48000)
        self.stream = None
        
        # TFLite 모델 로드 및 인터프리터 초기화
        # AI 모델 파일을 메모리에 불러온다.
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        # 모델이 사용할 입력·출력 공간을 준비한다.
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 모델 입력 텐서 정보
        self.input_shape = self.input_details[0]['shape'] # 예: [1, 15600] 또는 [1, 44100] 등

    def set_stream(self, stream):
        """MicController가 연 마이크 stream을 이 객체에 연결한다."""
        self.stream = stream

    def is_wakeup(self):
        """마이크 소리를 한 번 분석해 wake word 감지 여부를 True/False로 반환한다."""
        # 모델은 약 1초 길이의 44,032-sample float32 오디오를 받는다.
        # 48 kHz 마이크 입력을 16 kHz로 낮추면 음성 길이와 피치가 달라져
        # Teachable Machine AudioClassifier의 학습 입력과 맞지 않는다.
        # 마이크에서 정해진 길이만큼의 원본 소리 바이트를 읽는다.
        raw_data = self.stream.read(self.buffer_size, exception_on_overflow=False)
        # int16 소리를 -1.0~1.0 실수로 바꿔 AI 모델 입력 형식에 맞춘다.
        audio_chunk = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 입력 길이에 맞춰 직접 리샘플링한다.
        # 모델이 요구하는 정확한 소리 길이를 가져온다.
        target_len = self.input_shape[-1]
        # 입력 소리 길이를 모델 입력 길이로 맞춘다.
        audio_chunk = resample(audio_chunk, target_len).astype(np.float32)
        # 모델은 여러 입력을 한 번에 받을 수 있어 첫 번째 차원을 추가한다.
        input_data = np.expand_dims(audio_chunk, axis=0)

        # 4. 추론 실행
        # 준비한 소리를 모델의 입력 칸에 넣는다.
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        # 실제 AI 추론을 실행한다.
        self.interpreter.invoke()
        # 모델이 계산한 각 분류의 확률을 꺼낸다.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 5. 결과 확인 (★ Index 0: Disaster_situation 확률 ★)
        # 모델의 0번 결과는 재난 방송, 1번 결과는 배경 소음 확률이다.
        disaster_prob = output_data[0][0]
        bg_noise_prob = output_data[0][1]
        
        print(f"[감지 중...] 재난상황 확률: {disaster_prob:.2f} | 배경소음 확률: {bg_noise_prob:.2f}")

        # 재난 방송일 확률이 80%보다 클 때만 다음 STT 단계로 넘어간다.
        if disaster_prob > 0.8:
            print("🚀 [Wakeup!] 재난 방송 키워드 감지!")
            
            return True
            
        return False
