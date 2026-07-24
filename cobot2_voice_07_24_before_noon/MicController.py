# 마이크를 열고, 녹음한 데이터를 WAV 파일로 저장하는 보조 파일이다.
from dataclasses import dataclass
import wave
import io
import pyaudio

# 마이크 설정값을 한데 모아 두는 영역이다.

@dataclass
class MicConfig:
    """마이크가 어떤 방식으로 소리를 읽을지 정하는 설정 묶음."""
    # 한 번에 읽을 소리 데이터의 크기이다.
    chunk: int = 12000
    # 1초에 읽는 소리 표본 수이다. 48000은 일반적인 마이크 품질이다.
    rate: int = 48000
    # 1은 한 개 마이크(모노), 2는 좌우 두 개 마이크(스테레오)이다.
    channels: int = 1
    # record_audio()가 녹음할 시간(초)이다.
    record_seconds: int = 5
    # 소리 숫자를 저장하는 형식이다. paInt16은 흔히 쓰는 16비트 형식이다.
    fmt: int = pyaudio.paInt16
    # 사용할 마이크 번호이다. 현재 open_stream에서는 기본 마이크를 사용한다.
    device_index: int = 10
    # wake word 모델에 전달할 소리 데이터 크기이다.
    buffer_size: int = 24000
    # check your device index
    # import pyaudio
    # p = pyaudio.PyAudio()
    # [(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]


class MicController:
    """PyAudio를 사용해 실제 마이크 stream을 열고 닫는 클래스."""
    def __init__(self, config: MicConfig = MicConfig()):
        # 전달받은 마이크 설정을 저장한다.
        self.config = config
        # record_audio()가 읽은 소리 조각을 순서대로 담는다.
        self.frames = []
        self.audio = None     # open_stream()에서 생성
        self.stream = None
        self.sample_width = None  # 스트림 열 때 샘플 폭을 저장

    def open_stream(self):
        """새로운 PyAudio 인스턴스를 생성하고 스트림을 엽니다."""
        # 운영체제의 마이크 장치와 연결할 PyAudio 객체를 만든다.
        self.audio = pyaudio.PyAudio()
        # WAV 파일을 만들 때 필요한 소리 한 표본의 바이트 수를 구한다.
        self.sample_width = self.audio.get_sample_size(self.config.fmt)
        # input=True이므로 스피커 출력이 아니라 마이크 입력을 연다.
        self.stream = self.audio.open(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
            # input_device_index=self.config.device_index
        )

    def record_audio(self):
        """설정된 시간만큼 마이크 소리를 읽어 self.frames에 저장한다."""
        print("start recording for 5 seconds")
        self.frames = []  # 이전 프레임 초기화
        # 전체 녹음 시간을 채우기 위해 몇 번 읽어야 하는지 계산한다.
        num_chunks = int(self.config.rate / self.config.chunk * self.config.record_seconds)

        for _ in range(num_chunks):
            # overflow가 나도 프로그램이 멈추지 않도록 False를 사용한다.
            data = self.stream.read(self.config.chunk, exception_on_overflow=False)
            self.frames.append(data)

    def close_stream(self):
        """스트림과 PyAudio 인스턴스를 종료합니다."""
        print("stop recording")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def save_wav(self, filename):
        """녹음된 데이터를 WAV 파일로 저장합니다."""
        # wave 모듈로 표준 WAV 파일을 열고, 앞에서 모은 소리를 기록한다.
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.config.rate)
            wf.writeframes(b''.join(self.frames))
        print("파일 저장 완료!")

    def get_wav_data(self):
        """녹음 데이터를 파일 대신 메모리 속 WAV 바이트로 만든다."""
        # BytesIO는 디스크 파일처럼 보이지만 실제로는 메모리에만 저장된다.
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.config.fmt))
            wf.setframerate(self.config.rate)
            wf.writeframes(b''.join(self.frames))
        return wav_buffer.getvalue()
