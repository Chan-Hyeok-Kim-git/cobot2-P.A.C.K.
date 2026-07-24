import struct
import math
import json
import threading
import numpy as np
import rclpy
import DR_init

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from scipy.spatial import cKDTree
from std_msgs.msg import String

from cobot2_interfaces.msg import ValidatedGrasp


# ============================================================
# 1. 로봇 설정
# ============================================================

ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


# ============================================================
# 2. 그리퍼 치수
# ============================================================

GRIPPER_MAX_OPENING = 0.100
GRIPPER_MARGIN      = 0.005
FINGER_THICKNESS    = 0.012
FINGER_LENGTH       = 0.055
FINGER_WIDTH        = 0.020
BODY_SIZE_X         = 0.036
BODY_SIZE_Y         = 0.075
BODY_SIZE_Z         = 0.132
TOTAL_LENGTH        = 0.213

COLLISION_THRESHOLD = 30
PREGRASP_DISTANCE   = 0.100
APPROACH_STEP       = 0.010


# ============================================================
# 3. 선반 설정
# ============================================================

# base_link 기준 물체 중심 Z가 이 값보다 낮으면 FRONT만 허용한다.
# 단위: meter
GRASP_Z_THRESHOLD = 0.240

# ★ 추가: cobot2_mi_node.cpp의 선반 칸막이("shelf_back"이라는 이름이지만
# 실제로는 위층 선반의 바닥 겸 아래층 천장) 실측 위치.
# kShelfBoxes[] = {"shelf_back", cx=0.37, cy=-0.48, cz=0.215, sx=1.01, sy=0.305, sz=0.01}
# → 중심 z=0.215m, 두께 10mm → 상단면(위층 바닥면)=0.220m
SHELF_PANEL_CENTER_Z = 0.215
SHELF_PANEL_HALF_THICKNESS = 0.005

# 물체가 자기 선반 바닥에서 이 거리(m) 이내로 가까우면(위층 물체가 바닥에
# 거의 붙어있는 경우) FRONT 완전 수평 접근 대신 위쪽에서 비스듬히 내려오는
# FRONT_TILT_UP 후보를 우선 시도한다 — 실측으로 "shelf_back과 link_5 충돌"이
# 이런 상황에서 반복 확인됨 (팔뚝이 바닥판 위 낮은 공간을 스칠 수밖에 없는
# 기하학적 문제라 좌우 오프셋만으로는 해결 안 됨).
SHELF_FLOOR_TILT_TRIGGER_M = 0.05
# approach_direction의 z 성분에 더할 위쪽 기울기 비율(정규화 전).
# 예: [0,-1,0] + [0,0,0.35] → 정규화 후 대략 20도 정도 위에서 내려오는 접근.
SHELF_FLOOR_TILT_UP_RATIO = 0.2

# TF 변환 노드가 만드는 /ai/objects_3d/base_json의 실제 필드명.
# position_base_xyz_m은 항상 base_link 기준 meter 단위다.
JSON_CLASS_NAME_KEY = "class_name"
JSON_BASE_POSITION_KEY = "position_base_xyz_m"
JSON_TRANSFORM_STATUS_KEY = "frame_transform_status"

# 저장 좌표와 점군 클러스터 중심 사이의 최대 허용 거리.
# 같은 class 물체가 여러 개일 때 가장 가까운 물체를 고르는 데 사용한다.
POSITION_MATCH_MAX_DISTANCE_M = 0.150


# ============================================================
# 4. ROS 토픽명
# ============================================================

TARGET_CLOUD_TOPIC      = "/ai/object_points_base"
ENVIRONMENT_CLOUD_TOPIC = "/ai/background_points_base"
EXPECTED_CLOUD_FRAME     = "base_link"
GRASP_RESULT_TOPIC      = "/grasp/validated_grasp"
RETRY_REQUEST_TOPIC     = "/grasp/retry_request"
FINAL_RESULT_TOPIC      = "/grasp_result"

# ★★★ 재수정: cobot2_mi_node.cpp를 다시 확인해보니 최근에 업데이트되어
# 있었다. 예전엔 publishFail()이 /grasp_result로 실패를 알렸는데, 지금은
# requestNextCandidate()가 /grasp/retry_request라는 별도 토픽으로 알린다
# (아마 이 노드의 최종 실패 발행(/grasp_result)이랑 겹치는 걸 막으려고
# 분리한 것으로 보임). 그래서 처음에 /grasp_result를 구독하게 넣었던
# 코드는 실제로는 여전히 아무 메시지도 못 받는 상태였다 — 토픽 이름이
# 안 맞았기 때문.
MI_NODE_RETRY_REQUEST_TOPIC = "/grasp/retry_request"
OBJECTS_BASE_JSON_TOPIC  = "/ai/objects_3d/base_json"

# ★ 삭제됨: hand-eye 캘리브레이션(T_gripper2camera.npy) 로드 관련 코드 전부 제거.
# vision 쪽에서 이제 카메라→base_link 변환을 이미 끝낸 point cloud를 직접
# 보내주기로 했으므로, 이 노드가 다시 변환할 필요가 없어졌다.

# target_cloud에 여러 물체가 섞여서 오므로 가까운 점끼리 클러스터링한 뒤,
# base_json에 저장된 active_item 위치와 가장 가까운 클러스터를 선택한다.
ACTIVE_ITEM_TOPIC = "/task/active_item"

CLUSTER_EPS_MM = 25.0        # 이 거리(mm) 안의 점들을 같은 물체로 묶음
CLUSTER_MIN_POINTS = 15      # 이보다 점이 적은 뭉치는 노이즈로 버림

# 같은 물체에 대해 MoveIt 충돌 시 순차 시도할 파지 후보 오프셋
CANDIDATE_OFFSET_M = 0.010

# ★ 재수정: target 물체의 bounding box를 통째로 확장해서 빼는 방식은
# 물체 바로 옆에 진짜로 있는 장애물(선반 벽, 다른 물체)까지 같이
# 가려버리는 문제가 있다. 대신 environment_points 하나하나에 대해
# "target_points 중 가장 가까운 점까지의 거리"를 계산해서, 그게 이
# 값(5mm) 이내면 "물체 자신이 새어들어온 점"으로만 콕 집어 제외한다.
# object_points와 background_points는 별도 메시지라 완전히 똑같은
# 부동소수점 좌표는 기대하기 어려워서, 정확히 일치가 아니라 "충분히
# 가까움"으로 판단한다.
OBJECT_POINT_MATCH_EPSILON_M = 0.005
MAX_IK_SOLUTIONS_PER_VARIANT = 3
MAX_MOVEIT_ATTEMPTS = 12

JOINT_LIMITS_DEG = (
    (-360.0, 360.0),  # J1
    (-95.0,   95.0),  # J2
    (-135.0, 135.0),  # J3
    (-360.0, 360.0),  # J4
    (-135.0, 135.0),  # J5
    (-360.0, 360.0),  # J6
)

JOINT_LIMIT_EPS_DEG = 0.05

def cluster_points(
    points: np.ndarray,
    eps_mm: float,
    min_points: int,
) -> list[np.ndarray]:
    """cKDTree 기반으로 가까운 점들을 하나의 클러스터로 묶는다.

    Args:
        points:
            shape=(N, 3), base_link 기준 점군.
            단위는 meter.
        eps_mm:
            같은 클러스터로 연결할 최대 거리.
            단위는 millimeter.
        min_points:
            클러스터로 인정할 최소 점 개수.

    Returns:
        각 원소가 shape=(M, 3)인 NumPy 배열인 클러스터 목록.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        return []

    if len(points) == 0:
        return []

    # NaN, inf가 있는 점 제거
    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]

    if len(points) < min_points:
        return []

    # 입력 점군은 meter이고 설정값은 mm이므로 meter로 변환
    eps_m = float(eps_mm) / 1000.0

    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    clusters = []

    for start_index in range(len(points)):
        if visited[start_index]:
            continue

        # 연결된 점들을 탐색하기 위한 큐
        queue = [start_index]
        visited[start_index] = True
        cluster_indices = []

        while queue:
            current_index = queue.pop()
            cluster_indices.append(current_index)

            neighbor_indices = tree.query_ball_point(
                points[current_index],
                r=eps_m,
            )

            for neighbor_index in neighbor_indices:
                if visited[neighbor_index]:
                    continue

                visited[neighbor_index] = True
                queue.append(neighbor_index)

        if len(cluster_indices) >= min_points:
            clusters.append(points[cluster_indices])

    return clusters


# ============================================================
# 5. 노드 본체
# ============================================================

class GraspValidatorNode(Node):

    def __init__(self):
        super().__init__("grasp_validator", namespace=ROBOT_ID)

        self.target_points      = None
        self.environment_points = None
        # ★ 삭제됨: self.validation_pending — 더 이상 플래그+폴링 방식을 안 씀
        # (try_run_validation()이 조건 충족 시 즉시 실행하는 방식으로 변경됨)

        # 최초 스캔에서 받은 물체 위치 목록.
        # 같은 class가 여러 개일 수 있으므로 class_name -> [xyz, xyz, ...] 형태로 저장한다.
        self.object_positions = {}
        self.active_item_position = None

        # 스캔 단계에서는 들어오는 base_json으로 목록을 계속 갱신하고,
        # 첫 active_item이 들어온 뒤에는 그 시점의 목록을 고정해서 사용한다.
        # active_item이 먼저 들어오고 JSON이 늦게 오면, 첫 유효 JSON을 저장한 뒤 고정한다.
        self.object_snapshot_locked = False

        # ★ 삭제됨: T_gripper2cam 로드 블록 (더 이상 이 노드에서 좌표 변환을 안 함)

        # ★ 추가: 지금 어떤 물품을 찾는 중인지 (task_manager가 알려줌).
        # target_cloud에 여러 물체가 섞여 오므로, 이 값과 매칭되는 클러스터만
        # target_points로 사용한다. 빈 문자열이면 아무것도 처리 안 함.
        self.active_item = ""
        # ★ 추가: 이번 active_item에 대해 이미 ValidatedGrasp를 발행했는지.
        # vision이 계속 스트리밍하는 동안 매 프레임마다 재검증·재발행하면
        # 같은 물체를 여러 번 집으려 드는 문제가 생기므로, 한 번 발행하면
        # active_item이 바뀔 때까지 더 이상 처리하지 않는다.
        self._published_for_current_item = False

        # MoveIt이 후보 하나를 거절하면 다음 후보/다음 solution space를
        # 발행하기 위한 재시도 상태. 기하학 후보는 먼저 저장하고, 실제 IK는
        # /grasp/retry_request가 올 때 필요한 후보만 지연 계산한다.
        self._candidate_attempts = []
        self._candidate_attempt_index = 0
        self._candidate_failure_reasons = []
        self._candidate_variants = []
        self._candidate_variant_index = 0
        self._retry_target_points = None
        self._retry_environment_points = None
        self._moveit_attempt_count = 0
        self._last_published_stamp = None
        self._last_published_label = ""
        self.current_solution_space = 0

        # ★★★ 추가: 진단용 카운터/타임스탬프. 콜백이 "한 번이라도 실행됐는지"를
        # 100% 확실하게 알기 위해, 조용히 return하던 모든 경로에 로그를 남긴다.
        # 지금까지 두 번의 실습 영상에서 "클러스터 매칭 성공"만 반복되고
        # environment_points 관련 로그가 단 한 번도 안 찍혔는데, 이게
        # "environment_cloud_callback이 아예 호출된 적이 없어서"인지
        # "호출은 되는데 뭔가에 막혀서"인지조차 구분이 안 됐었다.
        # 아래 카운터들로 이걸 명확히 가른다.
        self._env_callback_count = 0
        self._target_callback_count = 0
        self._env_frame_id_seen = None

        # ★★★ 추가: DSR API 함수(get_current_posx, ikin 등)는 내부적으로
        # ROS2 서비스 호출이라 응답을 받으려면 executor가 계속 spin해야
        # 하는데, 지금 검증 로직을 콜백 안에서 직접(동기적으로) 실행하다보니
        # "이 콜백을 처리 중인 그 스레드"가 곧 "그 서비스 응답을 처리해야
        # 할 스레드"이기도 해서 데드락이 걸렸다(자기 자신이 자기 응답을
        # 기다리는 상태). ReentrantCallbackGroup + MultiThreadedExecutor로
        # 스레드를 여러 개 두면, 하나가 DSR 응답을 기다리며 막혀도 다른
        # 스레드가 그 응답 처리를 대신 진행할 수 있다.
        # (cobot2_move.py에서 movel/get_current_posx 관련해 똑같은 종류의
        # 문제를 이 방식으로 해결했던 것과 동일한 조치)
        self.callback_group = ReentrantCallbackGroup()

        # ★★★ 추가: ReentrantCallbackGroup 때문에 target_cloud_callback과
        # environment_cloud_callback이 이제 진짜로 서로 다른 스레드에서
        # 동시에 실행될 수 있다. 둘 다 try_run_validation()을 부르는데,
        # 이게 동시에 두 번 들어가면 검증 로직이 중복 실행되거나
        # ValidatedGrasp가 두 번 발행될 위험이 있다. 락으로 한 번에
        # 하나만 검증하게 막는다.
        self._validation_lock = threading.Lock()

        self.create_subscription(
            String, ACTIVE_ITEM_TOPIC, self.on_active_item, 10,
            callback_group=self.callback_group)
        self.create_subscription(
            String, RETRY_REQUEST_TOPIC, self.on_retry_request, 10,
            callback_group=self.callback_group)
        self.create_subscription(
            String,
            OBJECTS_BASE_JSON_TOPIC,
            self.objects_base_json_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            PointCloud2,
            TARGET_CLOUD_TOPIC,
            self.target_cloud_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointCloud2,
            ENVIRONMENT_CLOUD_TOPIC,
            self.environment_cloud_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )

        self.result_pub = self.create_publisher(
            ValidatedGrasp, GRASP_RESULT_TOPIC, 10)
        self.final_result_pub = self.create_publisher(
            String, FINAL_RESULT_TOPIC, 10)

        # ★★★ 재수정: /grasp_result가 아니라 mi_node가 실제로 쓰는
        # /grasp/retry_request를 구독해야 한다 (토픽 이름 정정).
        self.create_subscription(
            String, MI_NODE_RETRY_REQUEST_TOPIC, self.on_grasp_result_feedback, 10,
            callback_group=self.callback_group)

        # ★★★ 추가: 5초마다 내부 상태를 무조건 찍는 진단 타이머.
        # 어떤 콜백도 안 불려도(vision이 완전히 침묵해도) 이 로그만은 반드시
        # 찍히므로, "노드가 살아있는데 아무 데이터도 못 받는 상태"인지
        # 바로 알 수 있다.
        self.create_timer(5.0, self._diagnostic_heartbeat)

        self.get_logger().info(
            f"GraspValidatorNode 시작 | JSON={JSON_BASE_POSITION_KEY} | "
            f"Z 기준={GRASP_Z_THRESHOLD:.3f}m | "
            f"Z < 기준: FRONT만 | Z >= 기준: TOP/FRONT 모두 검증"
        )
    def check_joint_limits_deg(
        self,
        joints,
    ) -> tuple[bool, str]:
        if joints is None or len(joints) != 6:
            return False, "JOINT_VALUE_INVALID"

        violations = []

        for index, (value, limits) in enumerate(
            zip(joints, JOINT_LIMITS_DEG),
            start=1,
        ):
            value = float(value)
            min_value, max_value = limits

            if (
                value < min_value - JOINT_LIMIT_EPS_DEG
                or value > max_value + JOINT_LIMIT_EPS_DEG
            ):
                violations.append(
                    f"J{index}={value:.3f}deg"
                    f"(limit={min_value:.1f}~{max_value:.1f})"
                )

        if violations:
            return False, "JOINT_LIMIT_VIOLATION:" + ",".join(violations)

        return True, "OK"

    

    # ★★★ 추가: 진단 heartbeat. 5초마다 현재 상태를 그대로 찍는다.
    def _diagnostic_heartbeat(self) -> None:
        self.get_logger().warn(
            f"[진단] active_item='{self.active_item}' | "
            f"target_cloud 콜백 수신={self._target_callback_count}회 | "
            f"environment_cloud 콜백 수신={self._env_callback_count}회 "
            f"(frame_id={self._env_frame_id_seen}) | "
            f"target_points={'있음' if self.target_points is not None else '없음'} | "
            f"environment_points={'있음' if self.environment_points is not None else '없음'} | "
            f"object_positions class 목록={sorted(self.object_positions.keys())} | "
            f"active_item_position={'있음' if self.active_item_position is not None else '없음'}"
        )

    def _load_doosan_api(self):
        self.dsr_ikin             = None
        self.dsr_get_current_posj = None
        self.dsr_get_current_posx = None
        self.dsr_posx             = None
        self.DR_BASE              = None
        self._dsr_ready           = False

        try:
            from DSR_ROBOT2 import (
                ikin,
                get_current_posj,
                get_current_posx,
                DR_BASE,
            )
            from DR_common2 import posx

            self.dsr_ikin             = ikin
            self.dsr_get_current_posj = get_current_posj
            self.dsr_get_current_posx = get_current_posx
            self.dsr_posx             = posx
            self.DR_BASE              = DR_BASE
            self._dsr_ready           = True
            self.get_logger().info("두산 API import 완료")

        except ImportError as e:
            self.get_logger().warn(
                f"두산 API 없음: {e} → IK 스킵 모드로 동작"
            )

    # ★ 삭제됨: get_robot_pose_matrix(), transform_cloud_to_base() —
    # 카메라→base_link 변환 로직 전체. vision이 이미 base_link 좌표로
    # 변환해서 주므로 이 노드에서 다시 변환할 필요가 없어졌다.

    def objects_base_json_callback(self, msg: String) -> None:
        """TF 변환 노드의 base_json을 class별 base_link 위치 목록으로 저장한다.

        기대 형식:
          {
            "frame_id": "base_link",
            "position_unit": "m",
            "objects": [
              {
                "class_name": "rope",
                "position_base_xyz_m": [x, y, z],
                "frame_transform_status": "success"
              }
            ]
          }
        """
        if self.object_snapshot_locked:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"base_json 파싱 실패: {exc}")
            return

        if not isinstance(payload, dict):
            self.get_logger().warn("base_json 최상위 값은 JSON object여야 합니다.")
            return

        frame_id = str(payload.get("frame_id", "")).strip().lstrip("/")
        if frame_id != EXPECTED_CLOUD_FRAME:
            self.get_logger().warn(
                f"base_json frame_id 오류: expected={EXPECTED_CLOUD_FRAME}, "
                f"received={frame_id or '<empty>'}"
            )
            return

        position_unit = payload.get("position_unit")
        if position_unit not in (None, "m"):
            self.get_logger().warn(
                f"base_json position_unit이 m가 아닙니다: {position_unit}"
            )
            return

        objects = payload.get("objects")
        if not isinstance(objects, list):
            self.get_logger().warn("base_json의 objects 필드는 배열이어야 합니다.")
            return

        new_positions = {}
        skipped = 0

        for obj in objects:
            if not isinstance(obj, dict):
                skipped += 1
                continue

            class_name = obj.get(JSON_CLASS_NAME_KEY)
            base_xyz = obj.get(JSON_BASE_POSITION_KEY)
            transform_status = obj.get(JSON_TRANSFORM_STATUS_KEY)

            if not isinstance(class_name, str) or not class_name.strip():
                skipped += 1
                continue

            # TF 노드 출력에서는 정상 변환된 물체만 사용한다.
            if transform_status != "success":
                skipped += 1
                continue

            if not isinstance(base_xyz, (list, tuple)) or len(base_xyz) < 3:
                skipped += 1
                continue

            try:
                xyz = np.array(
                    [float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                skipped += 1
                continue

            if not np.isfinite(xyz).all():
                skipped += 1
                continue

            new_positions.setdefault(class_name.strip(), []).append(xyz)

        if not new_positions:
            self.get_logger().warn(
                "base_json에서 변환 성공한 class_name/position_base_xyz_m을 "
                "찾지 못했습니다."
            )
            return

        self.object_positions = new_positions
        self.get_logger().info(f"{self.object_positions}")
        self.active_item_position = self._resolve_active_item_position()

        summary = ", ".join(
            f"{name}:{len(positions)}개"
            for name, positions in sorted(self.object_positions.items())
        )
        self.get_logger().info(
            f"스캔 위치 목록 저장: {summary} | skipped={skipped}"
        )

        # 작업이 이미 시작된 상태라면 이 첫 유효 목록을 고정한다.
        if self.active_item:
            self.object_snapshot_locked = True
            self.get_logger().info("스캔 위치 목록 고정")

        self.try_run_validation()

    def _resolve_active_item_position(self):
        """현재 target cluster와 가장 가까운 저장 위치를 선택한다."""
        if not self.active_item:
            return None

        positions = self.object_positions.get(self.active_item, [])
        if not positions:
            return None

        if self.target_points is None or len(self.target_points) == 0:
            return np.asarray(positions[0], dtype=np.float64).copy()

        target_center = np.asarray(
            self.target_points, dtype=np.float64
        ).mean(axis=0)
        return min(
            (np.asarray(pos, dtype=np.float64) for pos in positions),
            key=lambda pos: float(np.linalg.norm(pos - target_center)),
        ).copy()

    def on_active_item(self, msg: String) -> None:
        """task_manager가 현재 처리할 class 이름을 보내면 상태를 초기화한다."""
        new_item = msg.data.strip()
        if new_item == self.active_item:
            return

        self.active_item = new_item
        self.target_points = None
        self.environment_points = None
        self.active_item_position = self._resolve_active_item_position()
        self._published_for_current_item = False
        self._candidate_attempts = []
        self._candidate_attempt_index = 0
        self._candidate_failure_reasons = []
        self._candidate_variants = []
        self._candidate_variant_index = 0
        self._retry_target_points = None
        self._retry_environment_points = None
        self._moveit_attempt_count = 0
        self._last_published_stamp = None
        self._last_published_label = ""

        if new_item and self.active_item_position is not None:
            self.object_snapshot_locked = True

        if new_item and self.active_item_position is None:
            self.get_logger().warn(
                f"[{new_item}] 스캔 위치 목록에 position_base_xyz_m이 아직 없습니다."
            )
        elif new_item:
            x, y, z = self.active_item_position
            self.get_logger().info(
                f"[{new_item}] 저장 위치 선택: "
                f"x={x:.3f}, y={y:.3f}, z={z:.3f}m"
            )

    def on_retry_request(self, msg: String) -> None:
        """MoveIt이 현재 후보를 거절했을 때 다음 후보를 발행한다.

        요청 JSON에는 실패한 ValidatedGrasp의 header stamp가 포함된다.
        오래된 계획 결과가 새 active_item의 후보를 넘기는 것을 막기 위해
        마지막 발행 stamp와 일치하는 요청만 처리한다.
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"retry_request JSON 파싱 실패: {exc}")
            return

        if not isinstance(payload, dict):
            return

        if not self.active_item or self._last_published_stamp is None:
            self.get_logger().debug("재시도 요청 무시: 활성 후보 없음")
            return

        try:
            request_stamp = (
                int(payload.get("stamp_sec", -1)),
                int(payload.get("stamp_nanosec", -1)),
            )
        except (TypeError, ValueError):
            self.get_logger().warn("재시도 요청 stamp 형식 오류")
            return

        if request_stamp != self._last_published_stamp:
            self.get_logger().warn(
                f"오래된 재시도 요청 무시: request={request_stamp}, "
                f"current={self._last_published_stamp}"
            )
            return

        reason = str(payload.get("reason", "MOVEIT_REJECTED"))
        failed_label = str(payload.get("grasp_type", self._last_published_label))
        self._candidate_failure_reasons.append(
            f"{failed_label}:{reason}"
        )

        self.get_logger().warn(
            f"MoveIt 후보 거절 | {failed_label} | {reason} → 다음 후보 시도"
        )

        # 재시도 콜백이 동시에 중복 실행되는 것을 방지한다.
        if not self._validation_lock.acquire(blocking=False):
            self.get_logger().warn("재시도 처리 중복 요청 무시")
            return
        try:
            self.publish_next_candidate()
        finally:
            self._validation_lock.release()

    def publish_final_failure(self, reason: str) -> None:
        """모든 후보가 실패했을 때만 최종 실패를 알린다."""
        payload = {
            "class_name": self.active_item,
            "success": False,
            "reason": reason,
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.final_result_pub.publish(out)
        self._published_for_current_item = True
        self._last_published_stamp = None
        self._candidate_attempts = []
        self._candidate_variants = []
        self._retry_target_points = None
        self._retry_environment_points = None
        self.get_logger().error(
            f"[{self.active_item}] 모든 파지 후보 실패: {reason}"
        )

    def target_cloud_callback(self, msg: PointCloud2):
        # ★★★ 추가: 이 콜백이 실제로 호출됐다는 걸 무조건 기록 (게이트로 막히기 전에)
        self._target_callback_count += 1

        if not self.active_item or self._published_for_current_item:
            return

        frame_id = msg.header.frame_id.strip().lstrip("/")
        if frame_id != EXPECTED_CLOUD_FRAME:
            self.get_logger().warn(
                f"목표 점군 frame_id 오류: expected={EXPECTED_CLOUD_FRAME}, "
                f"received={msg.header.frame_id}"
            )
            return

        base_pts = self.pointcloud2_to_numpy(msg)
        if len(base_pts) == 0:
            return

        saved_positions = self.object_positions.get(self.active_item, [])
        if not saved_positions:
            self.get_logger().debug(
                f"[{self.active_item}] 저장 위치 목록을 기다리는 중"
            )
            return

        clusters = cluster_points(
            base_pts, CLUSTER_EPS_MM, CLUSTER_MIN_POINTS
        )
        if not clusters:
            self.get_logger().debug(
                f"[{self.active_item}] 목표 점군에서 클러스터를 만들지 못함"
            )
            return

        matched_cluster = None
        matched_saved_position = None
        best_distance = float("inf")

        # 같은 class 물체가 여러 개 있어도 저장 좌표와 클러스터 중심의
        # 모든 조합 중 가장 가까운 쌍을 선택한다.
        for cluster in clusters:
            cluster_center = np.asarray(cluster, dtype=np.float64).mean(axis=0)
            for saved_position in saved_positions:
                saved_position = np.asarray(
                    saved_position, dtype=np.float64
                )
                distance = float(
                    np.linalg.norm(cluster_center - saved_position)
                )
                if distance < best_distance:
                    best_distance = distance
                    matched_cluster = cluster
                    matched_saved_position = saved_position.copy()

        if (
            matched_cluster is None
            or matched_saved_position is None
            or best_distance > POSITION_MATCH_MAX_DISTANCE_M
        ):
            self.get_logger().debug(
                f"[{self.active_item}] 저장 위치와 맞는 클러스터 없음 | "
                f"best={best_distance:.3f}m, "
                f"limit={POSITION_MATCH_MAX_DISTANCE_M:.3f}m"
            )
            self.target_points = None
            self.active_item_position = None
            return

        self.target_points = matched_cluster
        self.active_item_position = matched_saved_position

        self.get_logger().info(
            f"[{self.active_item}] 클러스터 매칭 성공 | "
            f"points={matched_cluster.shape[0]}, "
            f"distance={best_distance:.3f}m, "
            f"saved_z={self.active_item_position[2]:.3f}m"
        )
        self.try_run_validation()

    def environment_cloud_callback(self, msg: PointCloud2):
        # ★★★ 추가: 이 콜백이 실제로 호출됐다는 걸 게이트보다 먼저 무조건 기록.
        # 지금까지 이게 안 찍혀서 "콜백이 아예 안 불리는지" 확인이 불가능했다.
        self._env_callback_count += 1
        self._env_frame_id_seen = msg.header.frame_id

        if not self.active_item or self._published_for_current_item:
            return  # target과 동일한 gate 적용

        frame_id = msg.header.frame_id.strip().lstrip("/")
        if frame_id != EXPECTED_CLOUD_FRAME:
            self.get_logger().warn(
                f"환경 점군 frame_id 오류: expected={EXPECTED_CLOUD_FRAME}, "
                f"received={msg.header.frame_id}"
            )
            return

        # ★ 수정: 여기도 마찬가지로 변환 없이 받은 좌표를 그대로 사용
        self.environment_points = self.pointcloud2_to_numpy(msg)

        # ★★★ 추가: 성공적으로 environment_points가 채워졌다는 걸 명시적으로 로그.
        # 이게 없어서 "채워지는지 안 채워지는지"조차 기존엔 알 수 없었다.
        self.get_logger().info(
            f"[{self.active_item}] environment_points 갱신 성공: "
            f"{len(self.environment_points)}점"
        )
        self.try_run_validation()

    def try_run_validation(self):
        # ★★★ 재수정: 근본 원인 발견 — validation_pending 플래그를 세워두고
        # 메인 루프의 다음 spin_once() 틱까지 기다렸다가 실행하는 구조였는데,
        # 그 짧은 시차 사이에 다른 콜백(target 또는 environment)이 먼저 끼어들어
        # target_points/environment_points를 다시 None으로 되돌려버리는
        # 레이스 컨디션이 있었다. target/environment가 초당 여러 번씩 들어오는
        # 상황이라 이 레이스가 거의 매번 발동해서, "둘 다 있다"고 확인한 바로
        # 그 순간에 검증을 못 하고 계속 스킵되고 있었다 (그래서 "TOP 실패",
        # "FRONT 실패", "ValidatedGrasp 발행" 로그가 단 한 번도 안 찍혔음).
        #
        # 해결: 플래그를 세우고 나중에 처리하는 대신, 조건이 맞는 바로 그
        # 자리에서 즉시(synchronously) run_validation()을 호출한다. 이러면
        # "체크"와 "실행" 사이에 다른 콜백이 끼어들 틈 자체가 없어진다.
        if self.target_points is None or self.environment_points is None:
            self.get_logger().warn(
                f"[{self.active_item}] 검증 보류 — "
                f"target_points={'있음' if self.target_points is not None else '없음'}, "
                f"environment_points={'있음' if self.environment_points is not None else '없음'} "
                f"(environment_cloud 콜백 수신 횟수={self._env_callback_count}회, "
                f"target_cloud 콜백 수신 횟수={self._target_callback_count}회)"
            )
            return
        if self.active_item_position is None:
            self.get_logger().warn(
                f"[{self.active_item}] 검증 보류 — active_item_position이 없음 "
                f"(base_json에서 이 class 위치를 아직 못 받음)"
            )
            return

        # ★★★ 추가: ReentrantCallbackGroup으로 두 콜백이 동시에 여기 들어올
        # 수 있으니, non-blocking으로 락을 시도해서 이미 다른 스레드가
        # 검증 중이면 이번 건 스킵한다(다음 프레임에서 다시 시도되므로
        # 데이터 유실 걱정 없음).
        if not self._validation_lock.acquire(blocking=False):
            self.get_logger().debug(
                f"[{self.active_item}] 다른 스레드가 이미 검증 중 — 이번 프레임 스킵"
            )
            return
        try:
            # 즉시 실행 — 더 이상 validation_pending 플래그로 다음 루프까지
            # 미루지 않는다.
            self.run_validation()
            self.target_points = None
            self.environment_points = None
        finally:
            self._validation_lock.release()

    def pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        offsets = {}
        for field in msg.fields:
            if field.name in ('x', 'y', 'z'):
                offsets[field.name] = field.offset

        if len(offsets) < 3:
            self.get_logger().warn("PointCloud2에 x/y/z 필드 없음")
            return np.zeros((0, 3), dtype=np.float32)

        x_off = offsets['x']
        y_off = offsets['y']
        z_off = offsets['z']

        n_pts = msg.width * msg.height
        step  = msg.point_step
        data  = bytes(msg.data)

        pts = np.zeros((n_pts, 3), dtype=np.float32)
        for i in range(n_pts):
            base = i * step
            pts[i, 0] = struct.unpack_from('<f', data, base + x_off)[0]
            pts[i, 1] = struct.unpack_from('<f', data, base + y_off)[0]
            pts[i, 2] = struct.unpack_from('<f', data, base + z_off)[0]

        return pts[np.isfinite(pts).all(axis=1)]

    def calculate_required_width(
        self,
        target_points: np.ndarray,
        closing_direction: np.ndarray,
    ) -> float:
        d = np.array(closing_direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-9)
        proj = target_points @ d
        return float(proj.max() - proj.min())

    def transform_to_gripper_frame(
        self,
        points: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        return (points - position) @ rotation

    def count_points_in_box(
        self,
        points: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> int:
        if len(points) == 0:
            return 0
        mask = np.all((points >= box_min) & (points <= box_max), axis=1)
        return int(mask.sum())

    def check_gripper_clearance(
        self,
        environment_points: np.ndarray,
        candidate: dict,
        required_width: float,
    ) -> tuple:
        position = np.array(candidate["position"])
        rotation = np.array(candidate["rotation"])
        half_w   = required_width / 2.0

        local_env = self.transform_to_gripper_frame(
            environment_points, position, rotation)

        l_n = self.count_points_in_box(
            local_env,
            np.array([-FINGER_THICKNESS/2,  half_w,                -FINGER_LENGTH]),
            np.array([ FINGER_THICKNESS/2,  half_w + FINGER_WIDTH,  0.0]),
        )
        if l_n > COLLISION_THRESHOLD:
            return False, f"LEFT_FINGER_BLOCKED({l_n}pts)"

        r_n = self.count_points_in_box(
            local_env,
            np.array([-FINGER_THICKNESS/2, -(half_w + FINGER_WIDTH), -FINGER_LENGTH]),
            np.array([ FINGER_THICKNESS/2, -half_w,                    0.0]),
        )
        if r_n > COLLISION_THRESHOLD:
            return False, f"RIGHT_FINGER_BLOCKED({r_n}pts)"

        b_n = self.count_points_in_box(
            local_env,
            np.array([-BODY_SIZE_X/2, -BODY_SIZE_Y/2, 0.0]),
            np.array([ BODY_SIZE_X/2,  BODY_SIZE_Y/2,  BODY_SIZE_Z]),
        )
        if b_n > COLLISION_THRESHOLD:
            return False, f"BODY_BLOCKED({b_n}pts)"

        return True, "CLEAR"

    def check_approach_path(
        self,
        environment_points: np.ndarray,
        candidate: dict,
        required_width: float,
    ) -> tuple:
        position     = np.array(candidate["position"])
        approach_dir = np.array(candidate["approach_direction"])
        approach_dir /= (np.linalg.norm(approach_dir) + 1e-9)

        pre_pos = position - approach_dir * PREGRASP_DISTANCE

        n_steps  = max(2, int(PREGRASP_DISTANCE / APPROACH_STEP))
        rotation = np.array(candidate["rotation"])
        half_w   = required_width / 2.0

        for i in range(n_steps + 1):
            t = i / n_steps
            current_pos = pre_pos + (position - pre_pos) * t
            local_env = self.transform_to_gripper_frame(
                environment_points, current_pos, rotation)

            l_n = self.count_points_in_box(
                local_env,
                np.array([-FINGER_THICKNESS/2,  half_w,                -FINGER_LENGTH]),
                np.array([ FINGER_THICKNESS/2,   half_w + FINGER_WIDTH,  0.0]),
            )
            if l_n > COLLISION_THRESHOLD:
                return False, f"APPROACH_L_BLOCKED(t={t:.2f},{l_n}pts)"

            r_n = self.count_points_in_box(
                local_env,
                np.array([-FINGER_THICKNESS/2, -(half_w + FINGER_WIDTH), -FINGER_LENGTH]),
                np.array([ FINGER_THICKNESS/2,  -half_w,                   0.0]),
            )
            if r_n > COLLISION_THRESHOLD:
                return False, f"APPROACH_R_BLOCKED(t={t:.2f},{r_n}pts)"

        return True, "PATH_CLEAR"

    def compute_obb(self, points: np.ndarray) -> dict:
        center  = points.mean(axis=0)
        shifted = points - center

        cov = np.cov(shifted.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order   = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        long_axis  = eigvecs[:, 0].copy()
        mid_axis   = eigvecs[:, 1].copy()
        short_axis = eigvecs[:, 2].copy()

        for ax in (long_axis, mid_axis, short_axis):
            if ax[2] < 0:
                ax *= -1

        def _size(ax):
            p = shifted @ ax
            return float(p.max() - p.min())

        return {
            "center"     : center,
            "long_axis"  : long_axis,
            "mid_axis"   : mid_axis,
            "short_axis" : short_axis,
            "size_long"  : _size(long_axis),
            "size_mid"   : _size(mid_axis),
            "size_short" : _size(short_axis),
            "top_z"      : float(points[:, 2].max()),
            "bottom_z"   : float(points[:, 2].min()),
        }

    def make_rotation_matrix(
        self,
        approach_direction: np.ndarray,
        closing_direction: np.ndarray,
    ) -> np.ndarray:
        z = -np.array(approach_direction, dtype=float)
        z /= (np.linalg.norm(z) + 1e-9)

        y = np.array(closing_direction, dtype=float)
        y -= y.dot(z) * z
        y /= (np.linalg.norm(y) + 1e-9)

        x = np.cross(y, z)
        x /= (np.linalg.norm(x) + 1e-9)

        return np.column_stack([x, y, z])

    def _rotation_to_dsr_euler(self, R: np.ndarray) -> tuple:
        b = math.atan2(math.sqrt(R[2,0]**2 + R[2,1]**2), R[2,2])

        if abs(math.sin(b)) > 1e-6:
            a = math.atan2(R[1,2],  R[0,2])
            c = math.atan2(R[2,1], -R[2,0])
        else:
            a = math.atan2(-R[1,0], R[0,0])
            c = 0.0

        return math.degrees(a), math.degrees(b), math.degrees(c)

    def create_top_candidate(self) -> dict:
        if self.target_points is None or len(self.target_points) < 10:
            return None

        obb = self.compute_obb(self.target_points)

        approach = np.array([0.0, 0.0, -1.0])

        sx   = obb["short_axis"].copy()
        sx[2] = 0.0
        norm = np.linalg.norm(sx)

        if norm < 1e-6:
            sx   = obb["long_axis"].copy()
            sx[2] = 0.0
            norm = np.linalg.norm(sx)
            sx   = np.array([1.0, 0.0, 0.0]) if norm < 1e-6 else sx / norm
        else:
            sx /= norm

        closing  = sx
        rotation = self.make_rotation_matrix(approach, closing)

        position = np.array([obb["center"][0], obb["center"][1], obb["top_z"]])

        a, b, c = self._rotation_to_dsr_euler(rotation)
        return {
            "grasp_type"         : "TOP",
            "position"           : position,
            "rotation"           : rotation,
            "approach_direction" : approach,
            "closing_direction"  : closing,
            "pose_mm_deg"        : [
                position[0]*1000, position[1]*1000, position[2]*1000,
                a, b, c
            ],
            "obb": obb,
        }

    def create_front_candidate(self) -> dict:
        if self.target_points is None or len(self.target_points) < 10:
            return None

        obb = self.compute_obb(self.target_points)

        approach = np.array([0.0, -1.0, 0.0])

        closing = np.array([1.0, 0.0, 0.0])
        for key in ("short_axis", "mid_axis", "long_axis"):
            ax   = obb[key].copy()
            ax[2] = 0.0
            norm = np.linalg.norm(ax)
            if norm < 1e-6:
                continue
            ax /= norm

            if abs(ax.dot(approach)) < 0.8:
                closing = ax
                break

        rotation = self.make_rotation_matrix(approach, closing)

        position = obb["center"].copy()

        a, b, c = self._rotation_to_dsr_euler(rotation)
        return {
            "grasp_type"         : "FRONT",
            "position"           : position,
            "rotation"           : rotation,
            "approach_direction" : approach,
            "closing_direction"  : closing,
            "pose_mm_deg"        : [
                position[0]*1000, position[1]*1000, position[2]*1000,
                a, b, c
            ],
            "obb": obb,
        }

    def get_solution_space_order(self) -> list[int]:
        """현재 로봇 configuration과 비트 차이가 작은 solution부터 반환한다."""
        current = int(getattr(self, "current_solution_space", 0))
        if current < 0 or current > 7:
            current = 0
        return sorted(
            range(8),
            key=lambda sol: (
                0 if sol == current else 1,
                int(sol ^ current).bit_count(),
                sol,
            ),
        )

    def solve_doosan_ik_space(
        self,
        candidate: dict,
        sol_space: int,
    ):
        """지정한 solution space 하나만 검사해 관절값 또는 None을 반환한다."""
        pose = candidate.get("pose_mm_deg")
        if pose is None or len(pose) != 6 or not self._dsr_ready:
            return None

        try:
            target = self.dsr_posx(*pose)
            self.get_logger().info(
                f"IK 검사 {candidate.get('label', '?')} | sol={sol_space} | "
                + ", ".join(f"{float(v):.2f}" for v in pose)
            )
            result = self.dsr_ikin(
                target,
                int(sol_space),
                self.DR_BASE,
            )
        except Exception as exc:
            self.get_logger().debug(
                f"IK sol={sol_space} 예외: {exc}"
            )
            return None

        if result is None:
            return None
        if np.isscalar(result) and float(result) == -1.0:
            return None

        try:
            joints = [float(v) for v in result]
        except (TypeError, ValueError):
            return None

        if len(joints) != 6 or not np.isfinite(joints).all():
            return None

        self.get_logger().info(
            f"IK 성공 {candidate.get('label', '?')} (sol={sol_space}): "
            + ", ".join(f"{v:.2f}" for v in joints)
        )
        return joints

    def get_doosan_ik_solutions(self, candidate: dict) -> list[dict]:
        """후보 pose에 대한 유효한 8개 solution space를 모두 반환한다."""
        pose = candidate.get("pose_mm_deg")
        if pose is None or len(pose) != 6:
            return []

        if not self._dsr_ready:
            return []

        try:
            target = self.dsr_posx(*pose)
            self.get_logger().info(
                "IK 전체 해 검사 pose: "
                + ", ".join(f"{float(v):.2f}" for v in pose)
            )

            current_sol = int(getattr(self, "current_solution_space", 0))
            sol_candidates = [current_sol] + [
                sol for sol in range(8) if sol != current_sol
            ]

            solutions = []
            for sol_space in sol_candidates:
                self.get_logger().info(
                    f"IK solution space 검사: sol={sol_space}"
                )
                try:
                    result = self.dsr_ikin(
                        target,
                        sol_space,
                        self.DR_BASE,
                    )
                except Exception as exc:
                    self.get_logger().debug(
                        f"IK sol={sol_space} 예외: {exc}"
                    )
                    continue

                if result is None:
                    continue
                if np.isscalar(result) and float(result) == -1.0:
                    continue

                try:
                    joints = [float(v) for v in result]
                except (TypeError, ValueError):
                    continue

                if len(joints) != 6 or not np.isfinite(joints).all():
                    continue

                solutions.append({
                    "sol_space": int(sol_space),
                    "joints": joints,
                })
                self.get_logger().info(
                    f"IK 성공(sol={sol_space}): "
                    + ", ".join(f"{v:.2f}" for v in joints)
                )

            return solutions

        except Exception as exc:
            self.get_logger().error(f"IK 전체 해 검사 예외: {exc}")
            return []

    def check_doosan_ik(self, candidate: dict) -> tuple:
        """기존 호출부 호환용: 전체 IK 해 중 우선순위 첫 해를 반환한다."""
        solutions = self.get_doosan_ik_solutions(candidate)
        if not solutions:
            return False, None, "IK_NO_SOLUTION"
        return True, solutions[0]["joints"], "OK"

    def check_current_robot_state(self):
        try:
            posj = self.dsr_get_current_posj()
            posx, sol = self.dsr_get_current_posx(ref=self.DR_BASE)
            self.current_solution_space = int(sol)
            self.get_logger().info(
                "현재 조인트: " + ", ".join(f"{float(v):.2f}" for v in posj))
            self.get_logger().info(
                "현재 TCP: " + ", ".join(f"{float(v):.2f}" for v in posx)
                + f" sol={sol}")
            return True
        except Exception as e:
            self.get_logger().error(f"로봇 상태 조회 실패: {e}")
            return False

    def validate_candidate(self, candidate, target_pts, env_pts) -> dict:
        """후보의 점군/그리퍼 국소 검증만 수행한다.

        IK는 build_attempts_for_candidate()에서 8개 solution space 전체를
        검사하며, pre-grasp와 grasp에 공통으로 존재하는 해만 시도 목록에 넣는다.
        """
        if candidate is None:
            return {
                "success": False,
                "reason": "CANDIDATE_NOT_CREATED",
                "required_width": 0.0,
            }

        w = self.calculate_required_width(
            target_pts, candidate["closing_direction"])
        if w + GRIPPER_MARGIN > GRIPPER_MAX_OPENING:
            return {
                "success": False,
                "reason": "WIDTH_TOO_LARGE",
                "required_width": w,
            }

        ok, reason = self.check_gripper_clearance(env_pts, candidate, w)
        if not ok:
            return {"success": False, "reason": reason, "required_width": w}

        ok, reason = self.check_approach_path(env_pts, candidate, w)
        if not ok:
            return {"success": False, "reason": reason, "required_width": w}

        return {"success": True, "reason": "LOCAL_OK", "required_width": w}

    def _clone_candidate(
        self,
        source: dict,
        label: str,
        position: np.ndarray = None,
        closing_direction: np.ndarray = None,
        approach_direction: np.ndarray = None,
    ) -> dict:
        candidate = dict(source)
        candidate["label"] = label
        candidate["position"] = np.asarray(
            source["position"] if position is None else position,
            dtype=np.float64,
        ).copy()
        candidate["closing_direction"] = np.asarray(
            source["closing_direction"]
            if closing_direction is None else closing_direction,
            dtype=np.float64,
        ).copy()
        # ★ 추가: approach_direction도 오버라이드 가능하게. 물체가 자기
        # 선반 바닥에 거의 붙어있으면(clearance 작음) 완전 수평 접근은
        # 팔뚝(link_5 등)이 바로 아래 바닥판을 스치는 충돌이 실측으로
        # 확인됐다 — 위쪽에서 비스듬히 접근하는 변형(FRONT_TILT_UP)에서 씀.
        raw_approach = np.asarray(
            source["approach_direction"]
            if approach_direction is None else approach_direction,
            dtype=np.float64,
        )
        norm = np.linalg.norm(raw_approach)
        candidate["approach_direction"] = (
            raw_approach / norm if norm > 1e-9 else raw_approach
        ).copy()
        candidate["rotation"] = self.make_rotation_matrix(
            candidate["approach_direction"],
            candidate["closing_direction"],
        )
        a, b, c = self._rotation_to_dsr_euler(candidate["rotation"])
        pos = candidate["position"]
        candidate["pose_mm_deg"] = [
            pos[0] * 1000.0,
            pos[1] * 1000.0,
            pos[2] * 1000.0,
            a, b, c,
        ]
        return candidate

    def generate_candidate_variants(
        self,
        top_candidate: dict,
        front_candidate: dict,
        allow_top: bool,
    ) -> list[dict]:
        """MoveIt 충돌 시 순차 시도할 기하학 후보를 우선순위대로 만든다.

        상부 물체에서도 TOP 계열만 모두 소진한 뒤 FRONT를 보는 것이 아니라,
        TOP_CENTER → FRONT_CENTER → TOP_FLIP → FRONT_FLIP 순으로 교차 배치해
        서로 다른 접근 방식이 일찍 시도되도록 한다.
        """
        top_variants = []
        front_variants = []

        if allow_top and top_candidate is not None:
            top = top_candidate
            top["label"] = "TOP_CENTER"
            top_variants.append(top)
            top_variants.append(self._clone_candidate(
                top, "TOP_FLIP", closing_direction=-top["closing_direction"]))

            long_axis = np.asarray(top["obb"]["long_axis"], dtype=float).copy()
            long_axis[2] = 0.0
            norm = np.linalg.norm(long_axis)
            if norm > 1e-6:
                long_axis /= norm
                top_variants.append(self._clone_candidate(
                    top, "TOP_YAW_90", closing_direction=long_axis))
                top_variants.append(self._clone_candidate(
                    top,
                    "TOP_OFFSET_POS",
                    position=top["position"] + long_axis * CANDIDATE_OFFSET_M,
                ))
                top_variants.append(self._clone_candidate(
                    top,
                    "TOP_OFFSET_NEG",
                    position=top["position"] - long_axis * CANDIDATE_OFFSET_M,
                ))

        if front_candidate is not None:
            front = front_candidate
            front["label"] = "FRONT_CENTER"
            front_variants.append(front)
            front_variants.append(self._clone_candidate(
                front,
                "FRONT_FLIP",
                closing_direction=-front["closing_direction"],
            ))

            # closing/approach에 모두 수직인 gripper local X축 방향으로
            # 10mm 오프셋해 손목/링크 충돌 회피 후보를 만든다.
            offset_axis = np.asarray(front["rotation"][:, 0], dtype=float)
            offset_axis /= (np.linalg.norm(offset_axis) + 1e-9)
            front_variants.append(self._clone_candidate(
                front,
                "FRONT_OFFSET_POS",
                position=front["position"] + offset_axis * CANDIDATE_OFFSET_M,
            ))
            front_variants.append(self._clone_candidate(
                front,
                "FRONT_OFFSET_NEG",
                position=front["position"] - offset_axis * CANDIDATE_OFFSET_M,
            ))

            # ★★★ 추가: 물체가 자기 선반 바닥(shelf_back 충돌체, 위층
            # 바닥 겸 아래층 천장)에서 SHELF_FLOOR_TILT_TRIGGER_M 이내로
            # 가까이 있으면, 완전 수평(approach=[0,-1,0]) 접근 대신 위쪽
            # 에서 비스듬히 내려오는 접근을 우선 시도한다. 실측으로
            # "shelf_back과 link_5 충돌"이 물체가 바닥에 거의 붙어있을
            # 때(약 26mm 여유) 반복 확인됐다 — 완전 수평 경로로는 팔뚝이
            # 바닥판 위 낮은 공간을 스칠 수밖에 없는 기하학적 문제라
            # 오프셋(좌우/폭 조정)만으로는 해결이 안 된다.
            obj_z = float(front["position"][2])
            panel_top = SHELF_PANEL_CENTER_Z + SHELF_PANEL_HALF_THICKNESS
            panel_bottom = SHELF_PANEL_CENTER_Z - SHELF_PANEL_HALF_THICKNESS
            is_above_panel = obj_z >= SHELF_PANEL_CENTER_Z
            # 물체가 있는 쪽 판 표면까지의 실제 여유 거리 (중심이 아니라
            # 표면 기준 — 물체는 표면 위/아래에 놓이지, 판 중심에 놓이지 않음)
            clearance_to_panel = (
                obj_z - panel_top if is_above_panel else panel_bottom - obj_z
            )
            near_own_floor = (
                0.0 <= clearance_to_panel < SHELF_FLOOR_TILT_TRIGGER_M
            )
            if near_own_floor:
                approach = np.asarray(front["approach_direction"], dtype=float).copy()
                tilted_approach = approach.copy()
                # 물체가 판 위(위층)에 있으면 위쪽으로, 판 아래(아래층)에
                # 있으면 아래쪽으로 기울여 팔이 항상 판에서 먼 쪽으로
                # 우회하게 한다.
                tilt_sign = 1.0 if is_above_panel else -1.0
                tilted_approach[2] += SHELF_FLOOR_TILT_UP_RATIO * tilt_sign
                tilt_variant = self._clone_candidate(
                    front,
                    "FRONT_TILT_UP",
                    approach_direction=tilted_approach,
                )
                # 바닥에 가까운 상황에서는 이 변형을 최우선으로 시도
                front_variants.insert(0, tilt_variant)
                front_variants.append(self._clone_candidate(
                    front,
                    "FRONT_TILT_UP_FLIP",
                    approach_direction=tilted_approach,
                    closing_direction=-front["closing_direction"],
                ))
                self.get_logger().info(
                    f"[{self.active_item}] grasp z={obj_z*1000:.1f}mm가 선반 바닥/천장"
                    f"(clearance={clearance_to_panel*1000:.1f}mm)에 가까움 "
                    f"→ FRONT_TILT_UP(비스듬한 접근) 후보 최우선 추가 "
                    f"({'위로' if is_above_panel else '아래로'} 회피)"
                )

        if not allow_top:
            return front_variants

        # ★ 재수정: 원래는 TOP/FRONT를 번갈아 시도했는데, 이러면 TOP의
        # 다른 변형(FLIP/YAW_90/OFFSET)을 다 시도해보기도 전에 FRONT로
        # 넘어가서 (실측으로 확인된) shelf_back 충돌 위험을 굳이 먼저
        # 감수하게 된다. TOP은 위쪽에서 접근하므로 선반 바닥/천장 충돌
        # 위험이 구조적으로 없는 반면, FRONT는 물체가 바닥에 가까우면
        # 팔이 바닥판을 스칠 위험이 있다(FRONT_TILT_UP으로 완화는 했지만
        # 완전히 없앤 건 아님). 그러니 위층 물체는 TOP 계열을 모두 소진한
        # 뒤에야 FRONT로 넘어가게 순서를 바꾼다 — "위층이면 최대한 수직
        #으로 잡는다"는 원래 설계 의도를 후보 순서 레벨에서도 지킨다.
        return top_variants + front_variants

    def build_attempts_for_candidate(
        self,
        candidate: dict,
        target_pts: np.ndarray,
        env_pts: np.ndarray,
    ) -> tuple[list[dict], str]:
        """한 기하학 후보에서 공통 solution space별 실행 후보를 만든다.

        8개 grasp IK와 8개 pre-grasp IK를 모두 먼저 계산하지 않고,
        우선순위가 높은 solution space부터 grasp/pre 쌍으로 검사한다.
        필요한 개수의 공통 해를 찾으면 즉시 중단한다.
        """
        local = self.validate_candidate(candidate, target_pts, env_pts)
        if not local["success"]:
            return [], local["reason"]

        approach = np.asarray(candidate["approach_direction"], dtype=np.float64)
        approach /= (np.linalg.norm(approach) + 1e-9)
        pre_position = (
            np.asarray(candidate["position"], dtype=np.float64)
            - approach * PREGRASP_DISTANCE
        )
        pre_candidate = self._clone_candidate(
            candidate,
            candidate["label"] + "_PRE",
            position=pre_position,
        )

        attempts = []
        grasp_solution_found = False
        pre_solution_found = False

        for sol_space in self.get_solution_space_order():
                grasp_joints = self.solve_doosan_ik_space(
                    candidate,
                    sol_space,
                )
                if grasp_joints is None:
                    continue

                grasp_solution_found = True

                grasp_limit_ok, grasp_limit_reason = (
                    self.check_joint_limits_deg(grasp_joints)
                )
                if not grasp_limit_ok:
                    self.get_logger().warn(
                        f"IK 해 제외 | {candidate['label']} | "
                        f"sol={sol_space} | grasp | "
                        f"{grasp_limit_reason}"
                    )
                    continue

                pre_joints = self.solve_doosan_ik_space(
                    pre_candidate,
                    sol_space,
                )
                if pre_joints is None:
                    continue

                pre_solution_found = True

                pre_limit_ok, pre_limit_reason = (
                    self.check_joint_limits_deg(pre_joints)
                )
                if not pre_limit_ok:
                    self.get_logger().warn(
                        f"IK 해 제외 | {candidate['label']} | "
                        f"sol={sol_space} | pre-grasp | "
                        f"{pre_limit_reason}"
                    )
                    continue

                attempts.append({
                    "label": f"{candidate['label']}_SOL{sol_space}",
                    "sol_space": int(sol_space),
                    "candidate": candidate,
                    "pre_grasp_position": pre_position.copy(),
                    "grasp_joints": list(grasp_joints),
                    "pre_grasp_joints": list(pre_joints),
                    "required_width": float(local["required_width"]),
                })
                if len(attempts) >= MAX_IK_SOLUTIONS_PER_VARIANT:
                    break

        if attempts:
            return attempts, "OK"
        if not grasp_solution_found:
            return [], "GRASP_IK_NO_SOLUTION"
        if not pre_solution_found:
            return [], "NO_COMMON_PRE_GRASP_SOLUTION_SPACE"
        return [], "IK_PAIR_NOT_FOUND"

    def select_grasp(self, top_cand, front_cand, target_pts, env_pts) -> dict:
        """이전 인터페이스 호환용. 현재는 전체 후보 목록을 생성한다."""
        allow_top = (
            self.active_item_position is not None
            and float(self.active_item_position[2]) >= GRASP_Z_THRESHOLD
        )
        variants = self.generate_candidate_variants(
            top_cand, front_cand, allow_top)
        attempts = []
        rejected = []
        for candidate in variants:
            candidate_attempts, reason = self.build_attempts_for_candidate(
                candidate, target_pts, env_pts)
            if candidate_attempts:
                attempts.extend(candidate_attempts)
            else:
                rejected.append(f"{candidate.get('label', '?')}:{reason}")
        return {
            "success": bool(attempts),
            "attempts": attempts,
            "rejected": rejected,
        }

    def rotation_matrix_to_quaternion(self, rotation: np.ndarray) -> tuple:
        r = np.asarray(rotation, dtype=np.float64)
        if r.shape != (3, 3):
            raise ValueError("rotation은 3x3 행렬이어야 합니다.")

        trace = float(np.trace(r))

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (r[2, 1] - r[1, 2]) / s
            qy = (r[0, 2] - r[2, 0]) / s
            qz = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s

        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            raise ValueError("유효한 quaternion을 만들 수 없습니다.")
        q /= norm
        return tuple(float(v) for v in q)

    def make_pose_stamped(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        stamp,
    ) -> PoseStamped:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("position은 길이 3인 벡터여야 합니다.")

        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rotation)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "base_link"
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def publish_validated_grasp(self, attempt: dict) -> bool:
        """미리 계산된 후보 하나를 MoveIt 검증 노드로 발행한다."""
        candidate = attempt["candidate"]
        joints = attempt["grasp_joints"]
        pre_joints = attempt["pre_grasp_joints"]

        if len(joints) != 6 or len(pre_joints) != 6:
            self.get_logger().error("IK 관절값이 6개가 아니므로 후보를 건너뜁니다.")
            return False

        grasp_position = np.asarray(candidate["position"], dtype=np.float64)
        pre_grasp_position = np.asarray(
            attempt["pre_grasp_position"], dtype=np.float64)
        rotation = np.asarray(candidate["rotation"], dtype=np.float64)

        stamp = self.get_clock().now().to_msg()
        msg = ValidatedGrasp()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.grasp_type = str(attempt["label"])
        msg.pre_grasp_pose = self.make_pose_stamped(
            pre_grasp_position, rotation, stamp)
        msg.grasp_pose = self.make_pose_stamped(
            grasp_position, rotation, stamp)
        msg.grasp_joints = [float(v) for v in joints]
        msg.pre_grasp_joints = [float(v) for v in pre_joints]
        msg.required_width = float(attempt["required_width"])

        self.result_pub.publish(msg)
        self._published_for_current_item = True
        self._last_published_stamp = (
            int(stamp.sec), int(stamp.nanosec))
        self._last_published_label = msg.grasp_type

        self.get_logger().info(
            f"ValidatedGrasp 발행 "
            f"[전체 시도 {self._moveit_attempt_count}/{MAX_MOVEIT_ATTEMPTS}] | "
            f"type={msg.grasp_type} | sol={attempt['sol_space']} | "
            f"pre=({pre_grasp_position[0]:.3f}, "
            f"{pre_grasp_position[1]:.3f}, "
            f"{pre_grasp_position[2]:.3f}) | "
            f"grasp=({grasp_position[0]:.3f}, "
            f"{grasp_position[1]:.3f}, "
            f"{grasp_position[2]:.3f})"
        )
        return True

    def on_grasp_result_feedback(self, msg: String) -> None:
        """/grasp/retry_request 구독 콜백 — mi_node의 개별 시도 실패
        피드백을 받아 다음 후보로 진행시킨다.

        mi_node의 requestNextCandidate()가 stamp_sec/stamp_nanosec을
        같이 보내주므로, self._last_published_stamp와 비교해서 "지금
        기다리고 있는 바로 그 시도"에 대한 응답인지 정확히 확인한다.
        (오래된/중복된 응답이 뒤늦게 와서 이미 다음 시도로 넘어간 걸
        다시 헷갈리게 만드는 걸 방지)
        """
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(result, dict):
            return

        if not self.active_item or not self._published_for_current_item:
            return  # 지금 발행해둔 시도가 없으면 무관한 메시지

        if self._last_published_stamp is not None:
            incoming_stamp = (
                result.get("stamp_sec"), result.get("stamp_nanosec"))
            if incoming_stamp != self._last_published_stamp:
                self.get_logger().debug(
                    f"[{self.active_item}] 이전 시도에 대한 지연 응답 무시 "
                    f"(현재={self._last_published_stamp}, 수신={incoming_stamp})"
                )
                return

        reason = str(result.get("reason", ""))
        self.get_logger().warn(
            f"[{self.active_item}] mi_node로부터 MoveIt 계획 실패 수신: "
            f"{reason} → 다음 후보로 진행"
        )
        self._published_for_current_item = False  # 다음 시도 발행 가능하게 gate 재오픈
        self.publish_next_candidate()

    def publish_next_candidate(self) -> None:
        """현재 variant의 다음 IK 해, 이후 다음 기하학 variant를 순차 발행한다.

        기하학 후보 전체의 IK를 처음부터 모두 계산하지 않는다. 현재 후보가
        MoveIt에서 실제로 실패했을 때만 다음 variant의 IK를 계산하므로,
        첫 후보가 성공하는 일반 경우의 초기 지연을 줄인다.
        """
        while True:
            if self._moveit_attempt_count >= MAX_MOVEIT_ATTEMPTS:
                reason = (
                    f"MAX_MOVEIT_ATTEMPTS_REACHED({MAX_MOVEIT_ATTEMPTS})"
                )
                if self._candidate_failure_reasons:
                    reason += "|" + "|".join(
                        self._candidate_failure_reasons[-12:])
                self.publish_final_failure(reason)
                return

            # 현재 기하학 variant에서 아직 시도하지 않은 solution space가 있으면 발행.
            while self._candidate_attempt_index < len(self._candidate_attempts):
                attempt = self._candidate_attempts[self._candidate_attempt_index]
                self._candidate_attempt_index += 1
                self._moveit_attempt_count += 1
                if self.publish_validated_grasp(attempt):
                    return

            # 현재 variant의 IK 해를 다 썼으면 다음 기하학 variant를 준비한다.
            if self._candidate_variant_index >= len(self._candidate_variants):
                reason = "ALL_MOVEIT_CANDIDATES_FAILED"
                if self._candidate_failure_reasons:
                    reason += "|" + "|".join(
                        self._candidate_failure_reasons[-12:])
                self.publish_final_failure(reason)
                return

            if (
                self._retry_target_points is None
                or self._retry_environment_points is None
            ):
                self.publish_final_failure("RETRY_POINT_SNAPSHOT_MISSING")
                return

            candidate = self._candidate_variants[
                self._candidate_variant_index]
            self._candidate_variant_index += 1
            label = candidate.get("label", candidate.get("grasp_type", "?"))

            attempts, reason = self.build_attempts_for_candidate(
                candidate,
                self._retry_target_points,
                self._retry_environment_points,
            )
            if not attempts:
                self._candidate_failure_reasons.append(
                    f"{label}:{reason}")
                self.get_logger().warn(
                    f"후보 사전 탈락: {label} → {reason}"
                )
                continue

            self._candidate_attempts = attempts
            self._candidate_attempt_index = 0
            self.get_logger().info(
                f"후보 준비 완료: {label} → "
                f"공통 solution {len(attempts)}개"
            )

    def _exclude_target_from_environment(
        self,
        environment_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """environment_points에서 target_points(물체 자신)와 사실상 같은
        좌표인 점만 콕 집어서 제외한다 (bounding box 통째로가 아님).

        vision의 background_points가 감지된 물체를 완벽히 제외 못 하고
        살짝 겹쳐서 들어오면, check_gripper_clearance/check_approach_path가
        "잡으려는 물체 자체"를 장애물로 오인해서 어느 방향/오프셋으로
        시도해도 항상 BODY_BLOCKED나 LEFT/RIGHT_FINGER_BLOCKED로 실패하는
        현상이 생긴다.

        ★ 재수정: 처음엔 물체 bounding box를 3cm 확장해서 그 안의 배경
        점을 통째로 뺐는데, 이러면 물체 바로 옆에 진짜로 있는 장애물
        (선반 벽, 다른 물체)까지 같이 가려져서 정작 필요한 충돌 체크가
        무력화될 위험이 있었다. 대신 cKDTree로 각 environment 점에서
        가장 가까운 target 점까지의 거리를 계산해서, OBJECT_POINT_MATCH_
        EPSILON_M(5mm) 이내인 점만 "물체 자신"으로 판단해 제외한다.
        물체에서 5mm보다 멀리 떨어진 진짜 장애물은 그대로 남아서 충돌
        체크가 계속 작동한다.
        """
        if environment_points is None or len(environment_points) == 0:
            return environment_points
        if target_points is None or len(target_points) == 0:
            return environment_points

        target_tree = cKDTree(target_points)
        nearest_distances, _ = target_tree.query(environment_points, k=1)
        is_self_point = nearest_distances <= OBJECT_POINT_MATCH_EPSILON_M

        removed = int(is_self_point.sum())
        if removed > 0:
            self.get_logger().warn(
                f"[{self.active_item}] environment_points에서 물체 자신과 "
                f"거의 같은 좌표(±{OBJECT_POINT_MATCH_EPSILON_M*1000:.0f}mm)인 "
                f"점 {removed}개 제외 ({len(environment_points)} → "
                f"{len(environment_points) - removed}점) — vision "
                f"background_points가 물체를 완전히 못 뺐을 가능성 있음"
            )
        return environment_points[~is_self_point]

    def run_validation(self):
        if self.active_item_position is None:
            self.get_logger().warn(
                f"[{self.active_item}] 스캔 목록의 저장 좌표가 없어 검증하지 않습니다."
            )
            return

        if self.target_points is None or self.environment_points is None:
            self.get_logger().warn(
                f"[{self.active_item}] run_validation 내부에서 target/environment "
                f"points가 비어있음"
            )
            return

        # ★★★ 추가: environment_points(배경 점군)에서 지금 잡으려는 물체
        # 자기 자신 근처의 점을 제외한다.
        # 지금까지 모든 파지 후보(TOP_CENTER/FLIP/YAW_90/OFFSET_*, FRONT_CENTER/
        # FLIP/OFFSET_*)가 WIDTH_TOO_LARGE가 아닌 BODY_BLOCKED/LEFT_FINGER_BLOCKED
        # /RIGHT_FINGER_BLOCKED로 실패했는데, 특히 물체 바로 옆으로 살짝만
        # 옮긴 후보(FRONT_OFFSET_POS)가 504점이나 걸리고 좀 더 멀리 옮긴
        # 후보(TOP_OFFSET_*)는 9~20점만 걸리는 패턴이 "후보 위치가 실제
        # 로봇 팔과 충돌해서"가 아니라 "잡으려는 물체 자신이 계속 장애물로
        # 잡혀서"라는 걸 시사한다. vision의 background_points가 감지된
        # 물체를 완벽히 제외 못 하고 살짝 겹쳐 들어오면 이런 현상이 생긴다.
        # → environment_points에서 target_points 주변을 명시적으로 걸러낸다.
        self.environment_points = self._exclude_target_from_environment(
            self.environment_points, self.target_points
        )

        object_z = float(self.active_item_position[2])
        allow_top = object_z >= GRASP_Z_THRESHOLD

        front_cand = self.create_front_candidate()
        top_cand = self.create_top_candidate() if allow_top else None

        if front_cand is None and top_cand is None:
            self.publish_final_failure("CANDIDATE_GENERATION_FAILED")
            return

        self._candidate_variants = self.generate_candidate_variants(
            top_cand,
            front_cand,
            allow_top,
        )
        self._candidate_variant_index = 0
        self._candidate_attempts = []
        self._candidate_attempt_index = 0
        self._candidate_failure_reasons = []
        self._moveit_attempt_count = 0
        self._last_published_stamp = None
        self._last_published_label = ""

        # try_run_validation() 종료 후 원본 target/environment는 None으로
        # 초기화되므로, 다음 후보를 지연 계산할 수 있도록 장면 스냅샷을 보관한다.
        self._retry_target_points = np.asarray(
            self.target_points, dtype=np.float64).copy()
        self._retry_environment_points = np.asarray(
            self.environment_points, dtype=np.float64).copy()

        if not self._candidate_variants:
            self.publish_final_failure("NO_GEOMETRIC_CANDIDATE")
            return

        self.get_logger().info(
            f"[{self.active_item}] 기하학 후보 "
            f"{len(self._candidate_variants)}개 준비 | "
            f"MoveIt 최대 시도={MAX_MOVEIT_ATTEMPTS}"
        )
        self.publish_next_candidate()

# ============================================================
# 진입점
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GraspValidatorNode()

        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = node

        node._load_doosan_api()

        node.check_current_robot_state()
        # ★★★ 재수정: rclpy.spin(node) 기본값(SingleThreadedExecutor)은
        # 스레드가 1개뿐이라, 콜백 안에서 DSR 서비스 호출(get_current_posx,
        # ikin 등)을 하면 그 응답을 처리해줄 다른 스레드가 없어서
        # 데드락에 걸렸다(실측으로 "IK 검사 pose" 로그 찍고 그 다음
        # get_current_posx() 호출에서 영원히 멈추는 걸로 확인됨).
        # MultiThreadedExecutor로 스레드를 여러 개 주면, 한 스레드가
        # DSR 응답을 기다리며 막혀 있어도 다른 스레드가 그 응답 처리를
        # 대신 진행할 수 있어서 데드락이 안 걸린다.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"노드 오류: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()