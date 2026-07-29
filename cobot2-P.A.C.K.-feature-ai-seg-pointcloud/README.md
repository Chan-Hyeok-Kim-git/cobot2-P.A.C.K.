# Rokey Cobot2

## 영상 촬영

### 최초 1회 빌드

```bash
cd /home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/rokey-cobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select dataset_recorder
```

### 실행

새 터미널:

```bash
cd /home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/rokey-cobot2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dataset_recorder record_with_camera.launch.py
```

명령 하나가 RealSense RGB와 녹화 노드를 함께 실행한다. Depth는 끈다.

동작:

- `rqt_image_view` 화면 자동 실행
- 첫 RGB 프레임부터 자동 녹화
- 터미널 `Ctrl+C`: 녹화 저장 후 전체 종료

저장 위치:

```text
dataset/sessions/<촬영시각>/
├── video.mkv
└── metadata.json
```

## 1초마다 1프레임 추출

녹화 종료 후 새 터미널:

```bash
cd /home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/rokey-cobot2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run dataset_recorder extract_frames \
  dataset/sessions/<촬영시각>/video.mkv --interval 1.0
```

결과:

```text
dataset/sessions/<촬영시각>/frames/
├── frame_000000_t000000.000.jpg
├── frame_000001_t000001.000.jpg
└── frames.json
```

## 문제 확인

화면 안 나오면:

```bash
ros2 topic hz /camera/camera/color/image_raw
```

RealSense 인식 확인:

```bash
rs-enumerate-devices
```

## YOLO26s 실시간 객체 탐지

### 설치·빌드

```bash
cd /home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/rokey-cobot2
source /opt/ros/humble/setup.bash
python3 -m pip install --user ultralytics==8.4.102
colcon build --symlink-install --packages-select object_detector
```

`best.pt`는 Git 밖에 보관한다.

### 카메라와 탐지 노드 실행

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch object_detector detect_with_camera.launch.py \
  model_path:=/absolute/path/to/best.pt \
  confidence:=0.4 \
  device:=0
```

출력:

- `/ai/detections/image`: 박스가 표시된 영상
- `/ai/detections/json`: 클래스, confidence, 박스, 중심 픽셀, 추론 시간

카메라가 이미 실행 중이면:

```bash
ros2 launch object_detector detect_with_camera.launch.py \
  model_path:=/absolute/path/to/best.pt \
  start_camera:=false
```

### 오류 프레임 저장

```bash
ros2 service call /yolo_object_detector/save_frame std_srvs/srv/Trigger '{}'
```

원본 이미지, 예측 이미지, JSON이 `dataset/detection_captures/YYYYMMDD/`에 저장된다.

## YOLO + MobileSAM 물체 Point Cloud + RViz2

RGB와 aligned depth를 사용한다. YOLO bbox를 MobileSAM prompt로 전달하고,
mask 내부 depth만 Point Cloud와 카메라 기준 XYZ로 발행한다. mask는 기본 3 px
침식하며 median depth band와 voxel downsampling을 적용한다. D435i, AI 노드,
RViz2를 명령 하나로 실행한다. `use_sam:=false`로 설정하면 기존 bbox 중앙 ROI,
RANSAC, DBSCAN 경로를 사용할 수 있다.

### 빌드

```bash
cd /home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/rokey-cobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select object_detector
source install/setup.bash
```

### 실행

```bash
ros2 launch object_detector pointcloud_with_camera.launch.py \
  model_path:=/home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/data/processed/models/yolo26s_newdata_baseline_v3/yolo26s_v3.pt \
  sam_model_path:=/home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/ai/mobile_sam.pt \
  device:=cpu
```

통합 launch는 RGB/Depth를 640x480x30으로 열고 depth를 color에 align한다.
infrared와 D435i IMU는 사용하지 않는다. RViz2의 Fixed Frame은
`camera_color_optical_frame`으로 설정되어 있다.

카메라 노드를 이미 별도로 실행했다면 aligned depth가 활성화됐는지 확인한 뒤:

```bash
ros2 launch object_detector pointcloud_with_camera.launch.py \
  model_path:=/home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/data/processed/models/yolo26s_newdata_baseline_v3/yolo26s_v3.pt \
  sam_model_path:=/home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/ai/mobile_sam.pt \
  start_camera:=false
```

### 출력 토픽 계약

- `/ai/object_points` (`sensor_msgs/msg/PointCloud2`): 배경이 제거된 모든 검출 물체점. 필드는 `x`, `y`, `z`, `rgb`, `class_id`다.
- `/ai/objects_3d/json` (`std_msgs/msg/String`): 클래스, bbox, camera XYZ(m), 대표점 픽셀 `representative_pixel_uv`, 대표점 depth `representative_depth_m`, depth 품질, 점 개수, 처리 상태.
- `/ai/objects_3d/markers` (`visualization_msgs/msg/MarkerArray`): RViz2 중심 구와 클래스·거리 텍스트.
- `/ai/detections_3d/image` (`sensor_msgs/msg/Image`): bbox와 XYZ가 표시된 RGB 영상.

좌표계는 `camera_color_optical_frame`이며 `+X` 오른쪽, `+Y` 아래,
`+Z` 전방, 단위는 meter다. 로봇팔 팀은 이후 TF로 base frame 좌표로 변환한다.

토픽 확인:

```bash
ros2 topic hz /ai/object_points
ros2 topic echo /ai/objects_3d/json --once
```

## YOLO Segmentation 단일 모델 Point Cloud

현재 기본 실행 경로다. MobileSAM 없이 segmentation `best.pt`의 instance mask를
aligned depth에 바로 적용한다. 세그멘테이션 전용 파라미터는
`src/object_detector/config/seg_pointcloud.yaml`에서 관리한다.

```bash
cd /home/spacewhale0107/cobot_ws/projects/doosan_rokey_bootcamp/collaborative_project_2/code/rokey-cobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch object_detector seg_pointcloud_with_camera.launch.py \
  model_path:=/absolute/path/to/segmentation_best.pt \
  device:=cpu
```

GPU 사용 시 `device:=0`. 출력 topic과 RViz 설정은 기존 Point Cloud launch와 같다.

주요 출력은 다음과 같다.

- `/ai/object_points`: 검출 물체 포인트. 필드 `x`, `y`, `z`, `rgb`, `class_id`.
- `/ai/background_points`: 검출된 모든 물체를 제거한 배경 포인트.
- `/ai/objects_3d/json`: 물체별 대표 픽셀, depth, camera-frame XYZ와 품질 정보.
- `/ai/objects_3d/markers`: RViz2용 중심점과 클래스·거리 표기.
- `/ai/detections_3d/image`: 세그멘테이션 마스크와 대표점이 표시된 영상.

기본 `background_exclusion_mode:=all`은 검출된 모든 물체를 배경에서 제거한다.
필요하면 launch 인자로 `class` 또는 `none`을 지정할 수 있다.

상태 로그의 `queued`와 `published`는 현재 큐 크기가 아닌 실행 후 누적 건수다.
입력과 출력 큐의 실제 최대 크기는 각각 1이며, 처리 지연 시 오래된 결과보다
최신 결과를 우선한다.
