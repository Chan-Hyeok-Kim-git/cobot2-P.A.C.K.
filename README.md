# Cobot2 Voice

재난 방송을 감지하고, 재난별 기본 비상 물품 목록과 사용자가 추가로 요청한 물품을 ROS 2 topic으로 발행하는 패키지다.

동작 순서:

```text
1차 재난 wake word 감지
  → Faster-Whisper STT
  → 재난 유형 추출
  → 기본 물품 목록 발행
  → 2차 추가 물품 wake word 대기
  → Faster-Whisper STT
  → 물품 하나 발행
```

## 요구 환경

- ROS 2 Humble
- Python 3.10 이상
- 입력 마이크
- `faster-whisper` 모델을 실행할 CPU 또는 NVIDIA GPU

### Python 의존성 설치

```bash
python3 -m pip install --user faster-whisper openwakeword sounddevice scipy numpy pyaudio
```

CPU 실행은 현재 `int8` 추론을 사용한다. NVIDIA GPU 사용 시 `voice_command.py`의 STT 설정을 `device="cuda"`, `compute_type="float16"`으로 변경할 수 있다.

## 포함 모델

`resource/` 폴더에는 다음 wake word 모델이 있어야 한다.

```text
resource/
├── soundclassifier_with_metadata.tflite  # 1차 재난 방송 감지 모델
└── hello_rokey_8332_32.tflite            # 2차 추가 물품 wake word 모델
```

Faster-Whisper `small` 모델은 첫 실행 때 자동 다운로드된다. 실행 코드가 계산한 `resource/whisper_models/` 폴더에 cache된다.

## 빌드

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select cobot2_voice
source install/setup.bash
```

## 실행

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run cobot2_voice voice_command
```

종료는 실행 터미널에서 `Ctrl+C`를 누른다.

## 상태 흐름

### 1차: 재난 방송 대기

`STATE_WAIT_DISASTER` 상태에서 `soundclassifier_with_metadata.tflite` 모델이 재난 방송 wake word를 감지한다.

감지 뒤 5초 동안 음성을 녹음하고 STT 결과에서 아래 재난명을 찾는다.

```text
지진, 홍수, 화재, 공습
```

재난별 기본 물품 목록을 `/voice/command`로 발행한 뒤 `STATE_WAIT_ADDITIONAL_ITEM` 상태로 바뀐다.

### 2차: 추가 물품 대기

`STATE_WAIT_ADDITIONAL_ITEM` 상태에서 `hello_rokey_8332_32.tflite` 모델이 추가 물품 wake word를 감지한다.

감지 뒤 5초 동안 음성을 녹음한다. STT 결과에서 허용 물품 하나를 찾으면 `/voice/additional_item`으로 발행한다. 이 상태는 유지되므로 추가 물품을 여러 번 요청할 수 있다.

## ROS 2 Topic 계약

### `/voice/command`

- 타입: `std_msgs/msg/String`
- 의미: 재난별 기본 물품 목록
- 형식: JSON 배열

예시:

```json
["work_gloves", "first_aid_kit", "rope", "lantern"]
```

### `/voice/additional_item`

- 타입: `std_msgs/msg/String`
- 의미: 추가할 물품 하나
- 형식: 물품 식별자 문자열

예시:

```text
emergency_food
```

수신 노드는 `/voice/command`를 받으면 초기 작업 큐를 만들고, `/voice/additional_item`을 받으면 해당 물품 하나를 기존 작업 큐 끝에 추가한다.

## 재난별 기본 물품

| 재난 | 발행 물품 |
| --- | --- |
| 지진 | `work_gloves`, `first_aid_kit`, `rope`, `lantern` |
| 홍수 | `raincoat`, `rain_boots_bag`, `rope`, `waterproof_tarp` |
| 화재 | `protective_mask`, `whistle`, `work_gloves`, `safety_goggles` |
| 공습 | `protective_mask`, `first_aid_kit`, `lantern`, `emergency_food` |

## 추가 물품 음성 명령

다음 한국어 물품명을 인식한다. 공백이 포함된 표현도 일부 지원한다.

| 음성 키워드 | 발행 식별자 |
| --- | --- |
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

새 터미널에서 실행한다.

```bash
cd /home/chk/cobot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic echo /voice/command
ros2 topic echo /voice/additional_item
```

## 문제 확인

### 마이크 입력 장치 확인

`MicController.py`의 `device_index`는 현재 기본 입력 장치를 사용하도록 설정되어 있다. 다른 마이크를 강제하려면 `input_device_index` 설정을 활성화해야 한다.

### Faster-Whisper 모델 다운로드 실패

첫 실행에는 모델 다운로드가 필요하다. 인터넷 연결과 모델 cache 경로의 쓰기 권한을 확인한다.

### 2차 wake word가 감지되지 않음

`hello_rokey_8332_32.tflite` 파일이 `resource/`에 있는지 확인하고, `wakeup_word_interrupt.py`의 threshold 값을 주변 소음 환경에 맞게 조절한다.
