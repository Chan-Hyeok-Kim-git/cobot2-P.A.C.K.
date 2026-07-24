# Cobot2 Voice

재난 방송을 감지하고 재난별 기본 비상 물품 목록, 사용자가 요청한 추가 물품을 ROS 2 topic으로 발행하는 패키지다.

## 동작 순서

1. 1차 재난 wake word 감지
2. Faster-Whisper STT
3. 재난 유형 추출
4. 기본 물품 목록 발행
5. dummy 노드가 `/robot/working=True` 발행
6. 2차 추가 물품 wake word 대기
7. Faster-Whisper STT
8. 물품 하나 발행
9. dummy 노드가 300초 뒤 또는 `f` 입력 뒤 `/robot/finished=True` 발행

## 요구 환경

- ROS 2 Humble
- Python 3.10 이상
- 입력 마이크
- faster-whisper 실행용 CPU 또는 NVIDIA GPU

### Python 의존성 설치

```bash
python3 -m pip install --user faster-whisper openwakeword sounddevice scipy numpy pyaudio
```

현재 CPU 설정은 `int8` 추론이다. NVIDIA GPU 사용 시 `cobot2_voice/voice_command.py`의 STT 생성부를 아래처럼 변경할 수 있다.

```python
self.stt = STT(model_dir, model_size="small", device="cuda", compute_type="float16")
```

## 포함 모델

`resource/` 폴더에는 다음 wake word 모델이 있어야 한다.

```text
resource/
├── soundclassifier_with_metadata.tflite  # 1차 재난 방송 감지 모델
└── hello_rokey_8332_32.tflite            # 2차 추가 물품 wake word 모델
```

Faster-Whisper `small` 모델은 첫 실행 때 자동 다운로드된다. 현재 코드는 `resource/whisper_models/` 아래에 모델 cache를 만든다.

## 빌드

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths src/cobot2_ws/cobot2_voice_07_24_full_package
source install/setup.bash
```

동일 workspace에 이름이 같은 `cobot2_voice` 백업 패키지가 있으면 일반 `colcon build`는 중복 패키지 오류가 난다. 위처럼 현재 패키지 경로만 지정하거나 백업 패키지를 workspace 밖으로 옮긴다.

## 실행

터미널 1:

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run cobot2_voice voice_command
```

터미널 2:

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run cobot2_voice dummy_test
```

`dummy_test` 터미널 입력:

```text
w  /robot/working=True 발행
f  작업 완료. /robot/working=False, /robot/finished=True 발행
```

`/voice/command` 수신 뒤 dummy 노드는 자동으로 `/robot/working=True`를 발행하고 300초 timer를 시작한다.

## 상태 흐름

### 1차: 재난 방송 대기

`/robot/working=False`일 때 `soundclassifier_with_metadata.tflite` 모델로 재난 방송 wake word를 감지한다. 감지 뒤 5초간 녹음하고 아래 재난명을 STT 텍스트에서 찾는다.

```text
지진, 홍수, 화재, 공습
```

찾으면 기본 물품 목록을 `/voice/command`로 발행한다.

### 2차: 추가 물품 대기

dummy 노드가 `/robot/working=True`를 발행하면 `hello_rokey_8332_32.tflite` 모델로 2차 wake word를 대기한다. 감지 뒤 5초간 녹음하고 허용 물품 하나를 찾으면 `/voice/additional_item`으로 발행한다.

### 종료

- dummy 노드: 300초 경과 또는 `f` 입력 시 `/robot/working=False`, `/robot/finished=True`를 발행하고 종료한다.
- voice 노드: `/robot/finished=True` 수신 시 wake word timer와 마이크를 정리하고 종료한다.

## ROS 2 Topic 계약

### `/voice/command`

- 타입: `std_msgs/msg/String`
- 의미: 재난별 기본 물품 목록
- 형식: JSON 배열

```json
["work_gloves", "first_aid_kit", "rope", "lantern"]
```

### `/voice/additional_item`

- 타입: `std_msgs/msg/String`
- 의미: 추가할 물품 하나
- 형식: 물품 식별자 문자열

```text
emergency_food
```

### `/robot/working`, `/robot/finished`

- 타입: `std_msgs/msg/Bool`
- 의미: 로봇 작업 중 상태, 작업 종료 신호

수신 노드는 `/voice/command`를 받으면 초기 작업 큐를 만들고, `/voice/additional_item`을 받으면 해당 물품을 기존 큐 끝에 추가한다.

## 재난별 기본 물품

| 재난 | 발행 물품 |
|---|---|
| 지진 | `work_gloves`, `first_aid_kit`, `rope`, `lantern` |
| 홍수 | `raincoat`, `rain_boots_bag`, `rope`, `waterproof_tarp` |
| 화재 | `protective_mask`, `whistle`, `work_gloves`, `safety_goggles` |
| 공습 | `protective_mask`, `first_aid_kit`, `lantern`, `emergency_food` |

## 추가 물품 음성 명령

| 음성 키워드 | 발행 식별자 |
|---|---|
| 장갑 | `work_gloves` |
| 구급상자, 구급 상자 | `first_aid_kit` |
| 밧줄 | `rope` |
| 랜턴 | `lantern` |
| 우비 | `raincoat` |
| 장화주머니, 장화 주머니 | `rain_boots_bag` |
| 방수포 | `waterproof_tarp` |
| 방독면 | `protective_mask` |
| 호루라기 | `whistle` |
| 보호안경, 보호 안경 | `safety_goggles` |
| 비상식량, 비상 식량 | `emergency_food` |

## Topic 확인

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic echo /voice/command
ros2 topic echo /voice/additional_item
ros2 topic echo /robot/working
ros2 topic echo /robot/finished
```

## 문제 확인

### 마이크 입력 장치 확인

`cobot2_voice/MicController.py`는 현재 기본 입력 마이크를 사용한다. 다른 마이크를 강제하려면 `input_device_index=self.config.device_index` 줄 주석을 해제하고 `device_index` 값을 맞춘다.

### Faster-Whisper 모델 다운로드 실패

첫 실행에는 모델 다운로드가 필요하다. 인터넷 연결과 `resource/whisper_models/` 쓰기 권한을 확인한다.

### 2차 wake word가 감지되지 않음

`resource/hello_rokey_8332_32.tflite` 존재를 확인한다. `WakeupWordInterrupt(threshold=0.3)` threshold를 주변 소음 환경에 맞게 조절한다.
