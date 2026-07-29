import os
import numpy as np
from scipy.signal import resample_poly
from openwakeword.model import Model
from ament_index_python.packages import get_package_share_directory

PACKAGE_PATH = get_package_share_directory("cobot2_voice")
MODEL_NAME = "hello_rokey_8332_32.tflite"
MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", MODEL_NAME)


class WakeupWordInterrupt:
    def __init__(self, threshold=0.3):
        self.model = Model(wakeword_models=[MODEL_PATH])

        # custom tflite 파일명에서 확장자를 뺀 값이 prediction key다.
        self.model_name = os.path.splitext(MODEL_NAME)[0]

        self.stream = None
        self.threshold = threshold

        # 48 kHz에서 80ms 길이.
        self.buffer_size = 3840

    def set_stream(self, stream):
        # STT 후 새로 연 마이크 stream만 연결한다.
        self.stream = stream

    def is_wakeup(self):
        if self.stream is None:
            return False

        raw_data = self.stream.read(
            self.buffer_size,
            exception_on_overflow=False,
        )
        audio_48k = np.frombuffer(raw_data, dtype=np.int16)

        # 48 kHz → 16 kHz. up=1, down=3.
        audio_16k = resample_poly(audio_48k, up=1, down=3)

        # openWakeWord 입력 형식: int16 PCM.
        audio_16k = np.clip(
            audio_16k,
            -32768,
            32767,
        ).astype(np.int16)

        prediction = self.model.predict(audio_16k)

        # Openwakeword 버전에 따라 (predictions, 부가정보) 튜플이 올 수 있음
        if isinstance(prediction, tuple):
            outputs = prediction[0]
        else:
            outputs = prediction

        confidence = float(outputs.get(self.model_name, 0.0))

        print(f"2차 wake word confidence: {confidence:.2f}")

        return confidence >= self.threshold