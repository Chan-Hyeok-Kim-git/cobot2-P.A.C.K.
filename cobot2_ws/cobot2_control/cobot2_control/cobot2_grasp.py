import struct
import math
import numpy as np
import rclpy
import DR_init

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

from cobot2_interfaces.msg import ValidatedGrasp


# ============================================================
# 1. 로봇 설정
# ============================================================

ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"

# 두산 드라이버(DR_init)에 어떤 로봇을 쓸지 전역으로 등록
# 이 두 줄이 없으면 DSR_ROBOT2의 함수들이 어떤 로봇에 명령할지 모름
DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


# ============================================================
# 2. 그리퍼 치수 (rg2_collision.yaml 실측값, 단위: meter)
#
#  [TCP 기준 로컬 좌표계]  ← 이 좌표계로 충돌 박스를 정의
#
#    TCP(손끝) = 원점 (0, 0, 0)
#    +Z = 본체(마운트) 방향  ← TCP에서 로봇팔 쪽
#    -Z = 물체 방향          ← TCP에서 물체 쪽 (손끝 앞)
#    +Y = 손가락이 닫히는 방향
#
#  z축 영역:
#    본체   z: 0      ~ +0.132  (TCP 뒤쪽)
#    손가락 z: -0.055 ~  0      (TCP 앞쪽, 물체 잡는 부분)
# ============================================================

GRIPPER_MAX_OPENING = 0.100   # RG2 최대 개폐폭 — 이보다 넓은 물체는 못 잡음
GRIPPER_MARGIN      = 0.005   # 안전 여유 — 물체폭 + MARGIN <= MAX_OPENING 이어야 파지 가능
FINGER_THICKNESS    = 0.012   # 손가락 두께 (x방향, 로컬 좌표 기준)
FINGER_LENGTH       = 0.055   # 손가락 길이 (z방향, TCP 앞쪽으로 뻗은 길이)
FINGER_WIDTH        = 0.020   # 손가락 폭 (y방향, 물체 잡는 면의 폭)
BODY_SIZE_X         = 0.036   # 본체 x 크기
BODY_SIZE_Y         = 0.075   # 본체 y 크기
BODY_SIZE_Z         = 0.132   # 본체 z 크기 (TCP 뒤쪽으로 뻗은 길이)
TOTAL_LENGTH        = 0.213   # 마운트~TCP 전체 길이 (rg2_collision.yaml 실측)

COLLISION_THRESHOLD = 5       # 박스 안에 이 점 개수 이상이면 충돌로 판정
PREGRASP_DISTANCE   = 0.100   # pre-grasp 시작 거리 — grasp에서 이 거리만큼 물러난 위치
APPROACH_STEP       = 0.010   # 접근 경로 충돌 검사 간격 (10mm마다 한 번씩 검사)

# MoveIt joint_limits.yaml과 동일한 제한 [degree]
MOVEIT_JOINT_LIMITS_DEG = [
    (-180.0, 180.0),   # joint_1
    (-95.0,   95.0),   # joint_2
    (-125.0, 125.0),   # joint_3
    (-180.0, 180.0),   # joint_4
    (-135.0, 135.0),   # joint_5
    (-180.0, 180.0),   # joint_6
]

# 제한 경계에 너무 가까운 자세를 피하기 위한 여유
JOINT_LIMIT_MARGIN_DEG = 1.0


# ============================================================
# 3. 선반 설정 (shelf_collision.yaml 실측값)
#
#  shelf_floor center z=0.00 → 바닥 윗면 z ≈ 0.005m
#  shelf_back  center z=0.24 → 천장 아랫면 z ≈ 0.235m
#  → 실제 사용 가능 수직 공간 ≈ 0.235m
#
#  TOP 파지 필요 공간 계산:
#    TOTAL_LENGTH(0.213) + 안전여유(0.030) = 0.243m
#    0.235m < 0.243m → 이 선반에서는 항상 FRONT 파지
# ============================================================

SHELF_USABLE_HEIGHT = 0.235                 # 선반 칸 유효 높이 (실측)
REQUIRED_TOP_SPACE  = TOTAL_LENGTH + 0.030  # TOP 파지에 필요한 최소 수직 공간 (0.243m)


# ============================================================
# 4. ROS 토픽명 (비전팀 토픽명에 맞게 수정 필요)
# ============================================================

TARGET_CLOUD_TOPIC      = "/grasp/target_cloud"       # 물체 표면 점군
ENVIRONMENT_CLOUD_TOPIC = "/grasp/environment_cloud"  # 환경(선반/주변) 점군
GRASP_RESULT_TOPIC      = "/grasp/validated_grasp"  # IK까지 통과한 파지 후보 발행


# ============================================================
# 5. 노드 본체
# ============================================================

class GraspValidatorNode(Node):

    def __init__(self):
        super().__init__("grasp_validator", namespace=ROBOT_ID)

        # 두산 API 등록/import는 main()에서 노드 생성이 완전히 끝난 뒤 수행한다.
        # super().__init__()과 클래스의 나머지 구조는 그대로 유지한다.

        # 두 점군을 별도로 저장 — 둘 다 도착해야 검증 시작 (게이트 패턴)
        self.target_points      = None   # (N,3) numpy 배열 — 물체 표면 점군
        self.environment_points = None   # (M,3) numpy 배열 — 선반/주변 환경 점군
        self.validation_pending = False

        # ★ [추가] 물체가 상부/하부 선반에 있는지 구분하는 변수
        #   비전팀이 별도 토픽으로 알려주면 여기에 저장
        #   현재는 기본값 'bottom' 사용 (실제 연동 시 구독 추가 필요)
        self.object_layer = 'bottom'  # 'top' 또는 'bottom'

        # 비전팀 점군 구독
        self.create_subscription(
            PointCloud2, TARGET_CLOUD_TOPIC,
            self.target_cloud_callback, 10)
        self.create_subscription(
            PointCloud2, ENVIRONMENT_CLOUD_TOPIC,
            self.environment_cloud_callback, 10)

        # 폭/충돌/접근경로/두산 IK까지 통과한 후보만 MoveIt 검증 노드로 발행
        self.result_pub = self.create_publisher(
            ValidatedGrasp, GRASP_RESULT_TOPIC, 10)

        self.get_logger().info(
            f"GraspValidatorNode 시작 | "
            f"TOP 필요={REQUIRED_TOP_SPACE*1000:.0f}mm | "
            f"선반 높이={SHELF_USABLE_HEIGHT*1000:.0f}mm | "
            f"→ 이 선반은 항상 FRONT 파지"
        )

    # ──────────────────────────────────────────────────────────────
    # 두산 API 로드
    # ──────────────────────────────────────────────────────────────

    def _load_doosan_api(self):
        # ★ 수정: DSR 변수 먼저 None 초기화 → import 실패해도 노드 살아있음
        self.dsr_ikin             = None
        self.dsr_get_current_posj = None
        self.dsr_get_current_posx = None
        self.dsr_posx             = None
        self.DR_BASE              = None
        self._dsr_ready           = False

        try:
            from DSR_ROBOT2 import (
                ikin,              # Inverse Kinematics — pose → 관절각도
                get_current_posj,  # 현재 관절 각도 조회
                get_current_posx,  # 현재 TCP 위치 조회
                DR_BASE,           # 좌표계 상수 (base_link 기준)
            
            )
            from DR_common2 import posx  # 위치 자료형 생성 함수
        except ImportError as e:
            self.get_logger().error(f"두산 API import 실패: {e}")
            raise

        # 인스턴스 변수로 바인딩 — 메서드에서 self.dsr_ikin() 으로 호출
        self.dsr_ikin             = ikin
        self.dsr_get_current_posj = get_current_posj
        self.dsr_get_current_posx = get_current_posx
        self.dsr_posx             = posx
        self.DR_BASE              = DR_BASE
        self._dsr_ready           = True
        self.get_logger().info("두산 API import 완료")

    # ──────────────────────────────────────────────────────────────
    # 콜백 — 점군 수신
    # ──────────────────────────────────────────────────────────────

    def target_cloud_callback(self, msg: PointCloud2):
        """비전팀의 '물체 점군' 수신 → numpy 변환 → 검증 시도"""
        self.target_points = self.pointcloud2_to_numpy(msg)
        self.get_logger().debug(f"target cloud: {len(self.target_points)}점")
        self.try_run_validation()

    def environment_cloud_callback(self, msg: PointCloud2):
        """비전팀의 '환경 점군' 수신 → numpy 변환 → 검증 시도"""
        self.environment_points = self.pointcloud2_to_numpy(msg)
        self.get_logger().debug(f"environment cloud: {len(self.environment_points)}점")
        self.try_run_validation()

    def try_run_validation(self):
        """
        게이트(Gate) 패턴:
        target / environment 두 점군이 모두 도착했을 때만 검증 실행.
        하나만 왔을 때 실행하면 불완전한 정보로 잘못된 판단을 할 수 있음.
        """
        if self.target_points is None or self.environment_points is None:
            return  # 아직 둘 다 안 왔으면 대기
        self.validation_pending = True

    # ──────────────────────────────────────────────────────────────
    # [함수 1] pointcloud2_to_numpy  ← 신규 구현
    #
    # PointCloud2 메시지는 바이너리 형태로 저장되어 있다.
    #
    # 메시지 구조:
    #   msg.fields    → 각 채널(x, y, z, intensity 등)의 이름 + 바이트 오프셋
    #   msg.point_step → 점 하나당 바이트 수 (x,y,z float32 = 12바이트)
    #   msg.width      → 점의 총 개수
    #   msg.data       → 실제 바이트 데이터 (바이너리)
    #
    # 읽는 방법:
    #   i번째 점의 시작 바이트 = i * point_step
    #   x값 = struct.unpack_from('<f', data, 시작바이트 + x_offset)[0]
    #   '<f' = 리틀엔디안 float32 (대부분의 카메라가 이 형식)
    # ──────────────────────────────────────────────────────────────

    def pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        # fields에서 x, y, z 각각의 바이트 오프셋을 찾아 딕셔너리에 저장
        offsets = {}
        for field in msg.fields:
            if field.name in ('x', 'y', 'z'):
                offsets[field.name] = field.offset

        if len(offsets) < 3:
            self.get_logger().warn("PointCloud2에 x/y/z 필드 없음")
            return np.zeros((0, 3), dtype=np.float32)

        x_off = offsets['x']  # x채널이 점 시작바이트에서 몇 바이트 뒤에 있는지
        y_off = offsets['y']
        z_off = offsets['z']

        n_pts = msg.width * msg.height  # 총 점 개수 (height=1이면 width=점 개수)
        step  = msg.point_step          # 점 하나당 바이트 수
        data  = bytes(msg.data)         # 바이트 배열로 변환

        pts = np.zeros((n_pts, 3), dtype=np.float32)
        for i in range(n_pts):
            base = i * step  # i번째 점의 시작 바이트 위치
            pts[i, 0] = struct.unpack_from('<f', data, base + x_off)[0]
            pts[i, 1] = struct.unpack_from('<f', data, base + y_off)[0]
            pts[i, 2] = struct.unpack_from('<f', data, base + z_off)[0]

        # 깊이 카메라는 가끔 NaN(측정 불가) 또는 Inf(무한) 값을 내보냄 → 제거
        return pts[np.isfinite(pts).all(axis=1)]

    # ──────────────────────────────────────────────────────────────
    # [함수 2] calculate_required_width  ← 신규 구현
    #
    # "이 방향으로 손가락이 닫힐 때 물체를 잡으려면 얼마나 벌어야 하나?"
    #
    # 핵심 아이디어: 투영(내적)
    #   물체의 모든 점을 closing_direction 벡터에 내적하면
    #   그 방향에서 봤을 때의 1D 좌표(scalar)가 나온다.
    #   max - min = 그 방향에서 본 물체의 너비
    #
    # 예)
    #   closing_direction = [1, 0, 0]  (X방향으로 손가락이 닫힘)
    #   물체 점들의 X좌표 범위: 0.50 ~ 0.65m
    #   required_width = 0.65 - 0.50 = 0.15m (150mm)
    # ──────────────────────────────────────────────────────────────

    def calculate_required_width(
        self,
        target_points: np.ndarray,
        closing_direction: np.ndarray,
    ) -> float:
        d = np.array(closing_direction, dtype=float)
        d /= (np.linalg.norm(d) + 1e-9)  # 단위벡터로 정규화 (크기=1로 만들기)

        # @ 연산자 = 행렬 곱. shape (N,3) @ (3,) = shape (N,) → 각 점의 투영값
        proj = target_points @ d

        return float(proj.max() - proj.min())  # 최대 - 최소 = 필요 개폐폭

    # ──────────────────────────────────────────────────────────────
    # [함수 3] transform_to_gripper_frame  ← 신규 구현
    #
    # world 좌표계 점군 → 그리퍼 로컬 좌표계로 변환
    #
    # 왜 변환이 필요한가?
    #   충돌 박스(손가락, 본체)는 그리퍼 로컬 좌표계에서 정의되어 있음.
    #   환경 점군은 world(base_link) 좌표계에 있음.
    #   같은 좌표계로 맞춰야 "박스 안에 점이 있는지" 계산 가능.
    #
    # 수식: local = (point - tcp_pos) @ rotation
    #   (point - tcp_pos): TCP를 원점으로 이동
    #   @ rotation: world → local 회전 변환
    #   (rotation의 열벡터가 로컬 축의 world 방향이므로,
    #    @ 연산이 자동으로 R^T(전치) 효과 = world→local 방향)
    # ──────────────────────────────────────────────────────────────

    def transform_to_gripper_frame(
        self,
        points: np.ndarray,    # (N,3) world 좌표계 점군
        position: np.ndarray,  # (3,)  TCP 위치 (world 좌표계)
        rotation: np.ndarray,  # (3,3) 그리퍼 회전행렬 (열=로컬 축 방향)
    ) -> np.ndarray:
        return (points - position) @ rotation  # 평행이동 후 회전 변환

    # ──────────────────────────────────────────────────────────────
    # [함수 4] count_points_in_box  ← 신규 구현
    #
    # axis-aligned box(축 정렬 박스) 안에 있는 점 개수 반환.
    # 충돌 검사에서 "이 공간에 환경 점이 몇 개 있나?" 확인에 사용.
    #
    # 구현 설명:
    #   (points >= box_min) → shape(N,3), 각 좌표가 최솟값 이상인지 True/False
    #   (points <= box_max) → shape(N,3), 각 좌표가 최댓값 이하인지 True/False
    #   & 로 AND 처리 → 두 조건 모두 만족
    #   .all(axis=1) → 행 방향(x,y,z 모두)이 True인 점만 선택 → shape(N,)
    #   .sum() → True 개수 = 박스 안의 점 개수
    # ──────────────────────────────────────────────────────────────

    def count_points_in_box(
        self,
        points: np.ndarray,   # (N,3) 검사할 점군 (그리퍼 로컬 좌표계)
        box_min: np.ndarray,  # (3,)  박스 최솟값 [x_min, y_min, z_min]
        box_max: np.ndarray,  # (3,)  박스 최댓값 [x_max, y_max, z_max]
    ) -> int:
        if len(points) == 0:
            return 0
        # x, y, z 모두 박스 범위 안인 점만 True
        mask = np.all((points >= box_min) & (points <= box_max), axis=1)
        return int(mask.sum())

    # ──────────────────────────────────────────────────────────────
    # [함수 5] check_gripper_clearance  ← 신규 구현
    #
    # 그리퍼를 grasp pose에 배치했을 때 환경 점군과 충돌하는지 검사.
    # 그리퍼 형상을 3개의 박스로 근사:
    #
    #  [TCP 로컬 좌표 기준 그리퍼 형상]
    #
    #  z=+0.132 ┌──────────────┐  ← 본체(body)
    #           │              │     x: -0.018 ~ +0.018
    #  z= 0.000 └──────────────┘  ← TCP (원점)
    #
    #  z=-0.000 ┌──┐       ┌──┐  ← 손가락 L(왼쪽) / R(오른쪽)
    #           │L │       │R │     x: -0.006 ~ +0.006
    #  z=-0.055 └──┘       └──┘     y(L): +half_w ~ +half_w+0.020
    #                               y(R): -half_w-0.020 ~ -half_w
    #            ← half_w →← half_w →
    #              (물체 폭 절반)
    #
    # 각 박스에 환경 점이 COLLISION_THRESHOLD(5)개 이상 → 충돌 판정
    # ──────────────────────────────────────────────────────────────

    def check_gripper_clearance(
        self,
        environment_points: np.ndarray,  # (M,3) 환경 점군 (world 좌표계)
        candidate: dict,                  # 파지 후보 (position, rotation 포함)
        required_width: float,            # 그리퍼 필요 개폐폭 (m)
    ) -> tuple:
        position = np.array(candidate["position"])  # TCP 위치 (world)
        rotation = np.array(candidate["rotation"])  # 그리퍼 회전행렬
        half_w   = required_width / 2.0             # 물체 폭의 절반

        # 환경 점군을 그리퍼 로컬 좌표계로 변환 → 박스와 같은 좌표계
        local_env = self.transform_to_gripper_frame(
            environment_points, position, rotation)

        # ── 왼쪽 손가락 박스 (y 양수 방향) ─────────────────────────
        # y 범위: 물체 폭 절반(half_w)에서 손가락 폭(FINGER_WIDTH)만큼 더 바깥
        l_n = self.count_points_in_box(
            local_env,
            np.array([-FINGER_THICKNESS/2,  half_w,                -FINGER_LENGTH]),
            np.array([ FINGER_THICKNESS/2,  half_w + FINGER_WIDTH,  0.0]),
        )
        if l_n > COLLISION_THRESHOLD:
            return False, f"LEFT_FINGER_BLOCKED({l_n}pts)"

        # ── 오른쪽 손가락 박스 (y 음수 방향) ────────────────────────
        r_n = self.count_points_in_box(
            local_env,
            np.array([-FINGER_THICKNESS/2, -(half_w + FINGER_WIDTH), -FINGER_LENGTH]),
            np.array([ FINGER_THICKNESS/2, -half_w,                    0.0]),
        )
        if r_n > COLLISION_THRESHOLD:
            return False, f"RIGHT_FINGER_BLOCKED({r_n}pts)"

        # ── 본체 박스 (TCP 뒤쪽, z 양수 방향) ───────────────────────
        b_n = self.count_points_in_box(
            local_env,
            np.array([-BODY_SIZE_X/2, -BODY_SIZE_Y/2, 0.0]),
            np.array([ BODY_SIZE_X/2,  BODY_SIZE_Y/2,  BODY_SIZE_Z]),
        )
        if b_n > COLLISION_THRESHOLD:
            return False, f"BODY_BLOCKED({b_n}pts)"

        return True, "CLEAR"

    # ──────────────────────────────────────────────────────────────
    # [함수 6] check_approach_path  ← 신규 구현
    #
    # pre-grasp 위치 → grasp 위치까지 이동 경로 상의 충돌 검사.
    #
    # 방법: 경로를 APPROACH_STEP(10mm) 간격으로 샘플링해서
    #       각 위치에서 손가락 박스와 환경 점군을 비교.
    #
    # t=0.0: pre-grasp (grasp에서 approach 반대로 100mm 물러난 위치)
    # t=1.0: grasp 위치
    #
    # 예) FRONT 파지, approach=[0,-1,0]:
    #   pre_pos = grasp_pos - [0,-1,0]*0.1 = grasp_pos + [0,0.1,0]
    #   → 선반 앞 10cm에서 시작해서 물체 쪽으로 접근
    #   t=0.0: 10cm 앞, t=0.5: 5cm 앞, t=1.0: 물체 위치
    # ──────────────────────────────────────────────────────────────

    def check_approach_path(
        self,
        environment_points: np.ndarray,
        candidate: dict,
        required_width: float,
    ) -> tuple:
        position     = np.array(candidate["position"])
        approach_dir = np.array(candidate["approach_direction"])
        approach_dir /= (np.linalg.norm(approach_dir) + 1e-9)  # 단위벡터

        # pre-grasp 위치: grasp에서 approach 반대 방향으로 PREGRASP_DISTANCE
        pre_pos = position - approach_dir * PREGRASP_DISTANCE

        n_steps  = max(2, int(PREGRASP_DISTANCE / APPROACH_STEP))  # 검사 횟수
        rotation = np.array(candidate["rotation"])
        half_w   = required_width / 2.0

        for i in range(n_steps + 1):
            t = i / n_steps  # 0.0 ~ 1.0

            # 선형 보간: t=0이면 pre_pos, t=1이면 position
            current_pos = pre_pos + (position - pre_pos) * t

            # 현재 그리퍼 위치에서 로컬 좌표 변환
            local_env = self.transform_to_gripper_frame(
                environment_points, current_pos, rotation)

            # 왼쪽 손가락 충돌 검사
            l_n = self.count_points_in_box(
                local_env,
                np.array([-FINGER_THICKNESS/2,  half_w,                -FINGER_LENGTH]),
                np.array([ FINGER_THICKNESS/2,   half_w + FINGER_WIDTH,  0.0]),
            )
            if l_n > COLLISION_THRESHOLD:
                return False, f"APPROACH_L_BLOCKED(t={t:.2f},{l_n}pts)"

            # 오른쪽 손가락 충돌 검사
            r_n = self.count_points_in_box(
                local_env,
                np.array([-FINGER_THICKNESS/2, -(half_w + FINGER_WIDTH), -FINGER_LENGTH]),
                np.array([ FINGER_THICKNESS/2,  -half_w,                   0.0]),
            )
            if r_n > COLLISION_THRESHOLD:
                return False, f"APPROACH_R_BLOCKED(t={t:.2f},{r_n}pts)"

        return True, "PATH_CLEAR"

    # ──────────────────────────────────────────────────────────────
    # [함수 7] compute_obb  ← 신규 구현 (PCA 기반 OBB 계산)
    #
    # OBB = Oriented Bounding Box (물체 방향에 맞게 회전된 최소 경계 박스)
    # AABB(축 정렬)와 달리 물체가 기울어져 있어도 꼭 맞게 계산됨.
    #
    # PCA(주성분 분석) 원리:
    #   점들이 가장 많이 퍼진 방향 → 주성분 1 = long_axis (분산 최대)
    #   그 다음으로 퍼진 방향     → 주성분 2 = mid_axis
    #   제일 안 퍼진 방향         → 주성분 3 = short_axis (분산 최소)
    #
    # 계산 순서:
    #   1. 중심(center) 계산 → 원점으로 이동(shifted)
    #   2. 공분산 행렬(3x3) 계산 → 각 방향의 분산 크기와 방향 담김
    #   3. 고유값 분해 → 고유벡터 = 주성분 방향, 고유값 = 분산 크기
    #   4. 고유값 내림차순 정렬 → long > mid > short
    #   5. 각 축 방향으로 투영해서 크기(size) 계산
    # ──────────────────────────────────────────────────────────────

    def compute_obb(self, points: np.ndarray) -> dict:
        center  = points.mean(axis=0)   # 점군의 무게중심
        shifted = points - center        # 중심을 원점으로 이동

        # 공분산 행렬: 각 방향으로 점들이 얼마나 퍼져있는지 나타냄 (3x3)
        cov = np.cov(shifted.T)

        # 고유값(eigvals) = 각 방향의 분산 크기
        # 고유벡터(eigvecs) = 각 방향의 단위벡터 (열벡터)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 고유값 내림차순 정렬 → 분산 큰 순서 = long, mid, short
        order   = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        long_axis  = eigvecs[:, 0].copy()  # 제일 긴 방향 (분산 최대)
        mid_axis   = eigvecs[:, 1].copy()  # 중간 방향
        short_axis = eigvecs[:, 2].copy()  # 제일 짧은 방향 (분산 최소)

        # 부호 통일: z성분이 음수면 반전
        # → 같은 물체라도 호출할 때마다 부호가 바뀌면 closing 방향이 바뀌어 불안정
        for ax in (long_axis, mid_axis, short_axis):
            if ax[2] < 0:
                ax *= -1

        # 각 축 방향 크기 계산 (투영 후 max-min)
        def _size(ax):
            p = shifted @ ax          # 각 점을 해당 축에 투영
            return float(p.max() - p.min())  # 그 방향의 물체 크기

        return {
            "center"     : center,
            "long_axis"  : long_axis,
            "mid_axis"   : mid_axis,
            "short_axis" : short_axis,
            "size_long"  : _size(long_axis),
            "size_mid"   : _size(mid_axis),
            "size_short" : _size(short_axis),
            "top_z"      : float(points[:, 2].max()),    # 물체 최상단 z
            "bottom_z"   : float(points[:, 2].min()),    # 물체 최하단 z
        }

    # ──────────────────────────────────────────────────────────────
    # [함수 8] make_rotation_matrix  ← 신규 구현
    #
    # approach와 closing 방향으로 그리퍼의 3x3 회전행렬 생성.
    #
    # [★ 수정 포인트: Z축 = -approach]
    # 이전 코드: z = +approach_direction  → 잘못됨
    # 수정 후:  z = -approach_direction  → 올바름
    #
    # 이유:
    #   approach = [0,-1,0]: 그리퍼가 Y 음방향으로 이동 (선반 안으로)
    #   그러면 그리퍼 손끝은 Y 양방향을 바라봄 (물체 쪽)
    #   로컬 +Z(본체 방향)는 손끝 반대 = Y 양방향 = -approach
    #
    # 로컬 좌표계:
    #   +Z = 본체 방향 = -approach   (마운트 쪽)
    #   +Y = closing_direction        (손가락 닫히는 방향)
    #   +X = Y × Z                   (오른손 법칙으로 자동 결정)
    # ──────────────────────────────────────────────────────────────

    def make_rotation_matrix(
        self,
        approach_direction: np.ndarray,  # 그리퍼 이동 방향 (world 좌표계)
        closing_direction: np.ndarray,   # 손가락 닫히는 방향 (world 좌표계)
    ) -> np.ndarray:
        # ★ 수정: z = -approach (본체 방향 = 그리퍼 진입 반대)
        z = -np.array(approach_direction, dtype=float)
        z /= (np.linalg.norm(z) + 1e-9)  # 단위벡터 정규화

        y = np.array(closing_direction, dtype=float)
        # y를 z에 수직이 되도록 정사영 (그람-슈미트)
        # 완전히 수직이 아닐 경우를 대비해 z성분 제거
        y -= y.dot(z) * z
        y /= (np.linalg.norm(y) + 1e-9)

        x = np.cross(y, z)  # 오른손 법칙: y와 z에 모두 수직인 방향
        x /= (np.linalg.norm(x) + 1e-9)

        # 열벡터로 쌓기: 각 열이 로컬 x,y,z 축의 world 방향
        return np.column_stack([x, y, z])  # shape: (3,3)

    # ──────────────────────────────────────────────────────────────
    # [함수 9] _rotation_to_dsr_euler
    #
    # 3x3 회전행렬 → DSR posx에 넣을 오일러각 (a, b, c, deg)
    #
    # 두산 DSR은 ZYZ 오일러각 표현 사용.
    # ※ 실제 동작 확인 후 ZYX이면 이 함수만 수정하면 됨
    #
    # ZYZ 변환 공식 (b=0 또는 π 특이점 처리 포함):
    #   b = atan2(√(R[2,0]²+R[2,1]²), R[2,2])
    #   a = atan2(R[1,2], R[0,2])      (b≠0 일 때)
    #   c = atan2(R[2,1], -R[2,0])     (b≠0 일 때)
    # ──────────────────────────────────────────────────────────────

    def _rotation_to_dsr_euler(self, R: np.ndarray) -> tuple:
        b = math.atan2(math.sqrt(R[2,0]**2 + R[2,1]**2), R[2,2])

        if abs(math.sin(b)) > 1e-6:
            # 일반적인 경우 (b가 0 또는 π가 아닐 때)
            a = math.atan2(R[1,2],  R[0,2])
            c = math.atan2(R[2,1], -R[2,0])
        else:
            # 특이점: b ≈ 0 또는 π일 때 a와 c가 얽혀 개별 결정 불가
            # a-c 또는 a+c 합만 결정 가능 → a에 몰아넣고 c=0
            a = math.atan2(-R[1,0], R[0,0])
            c = 0.0

        return math.degrees(a), math.degrees(b), math.degrees(c)

    # ──────────────────────────────────────────────────────────────
    # [함수 10] create_top_candidate  ← 신규 구현 + ★수정
    #
    # 위에서 수직으로 내려오는 파지 후보 생성.
    #
    # [★ 수정1: closing = short_axis_xy (이전: long_axis_xy)]
    #
    #   위에서 본 물체:
    #     long_axis →  ┌──────────────────┐
    #                  │    물체           │  ↕ short_axis (예: 80mm)
    #                  └──────────────────┘
    #                  ←    200mm         →
    #
    #   closing = long_axis → 그리퍼 200mm 필요 → 한계(100mm) 초과 → 실패
    #   closing = short_axis → 그리퍼 80mm 필요 → 파지 가능 ✓
    #   → 파지 성공 가능성을 높이려면 항상 짧은 방향으로 손가락을 닫아야 함
    #
    # [★ 수정2: grasp z = top_z (이전: center_z)]
    #   center_z 사용 시: 손가락이 물체 중간 높이로 접근
    #   → 위에서 내려오다가 선반 천장에 그리퍼 본체가 충돌할 위험
    #   top_z 사용 시: 물체 맨 위에서 살짝 누르듯 잡음 → 안전
    # ──────────────────────────────────────────────────────────────

    def create_top_candidate(self) -> dict | None:
        if self.target_points is None or len(self.target_points) < 10:
            return None

        obb = self.compute_obb(self.target_points)

        approach = np.array([0.0, 0.0, -1.0])  # Z 음방향 (위→아래)

        # ★ short_axis를 XY 평면에 투영 → 가장 짧은 수평 방향
        #   z성분 제거: 위에서 잡을 때 손가락은 수평으로 닫히므로 수직 성분 불필요
        sx   = obb["short_axis"].copy()
        sx[2] = 0.0  # z성분 제거
        norm = np.linalg.norm(sx)

        if norm < 1e-6:
            # short_axis가 거의 수직인 경우 (예: 세워진 원통)
            # → 수평 방향 정보가 없으므로 long_axis로 대체
            sx   = obb["long_axis"].copy()
            sx[2] = 0.0
            norm = np.linalg.norm(sx)
            sx   = np.array([1.0, 0.0, 0.0]) if norm < 1e-6 else sx / norm
        else:
            sx /= norm  # 단위벡터로 정규화

        closing  = sx
        rotation = self.make_rotation_matrix(approach, closing)

        # ★ grasp 위치: xy=물체 중심, z=물체 최상단(top_z)
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
            "obb": obb,  # 디버깅/로그용으로 포함
        }

    # ──────────────────────────────────────────────────────────────
    # [함수 11] create_front_candidate  ← 신규 구현 + ★수정
    #
    # 선반 앞(Y 음방향)에서 수평으로 접근하는 파지 후보 생성.
    #
    # [★ 수정: OBB 사용해서 closing 방향 자동 선택 (이전: 고정 [1,0,0])]
    #
    #   이전 코드: closing = [1,0,0] 항상 고정
    #   → 물체가 45도 기울어지면 실제 물체 폭보다 더 넓게 벌어야 함 → 실패
    #
    #   수정 후: approach와 수직인 OBB 축을 자동 선택
    #   → 물체가 어느 방향으로 기울어져 있어도 실제 짧은 방향으로 잡음
    #
    #   선택 원칙:
    #     short → mid → long 순으로 시도
    #     approach 방향([0,-1,0])과 내적이 0.8 미만이면 채택 (충분히 수직)
    #
    #     왜 approach와 수직이어야 하나?
    #       closing이 approach와 평행하면 손가락이 그리퍼 진입 방향으로 닫힘
    #       → 물체를 옆에서 잡는 게 아니라 앞뒤로 눌러버리는 꼴 → 물리적으로 이상
    #
    # [grasp z: center_z 사용]
    #   수평 접근이므로 물체 중심 높이에서 접근하는 게 자연스러움
    # ──────────────────────────────────────────────────────────────

    def create_front_candidate(self) -> dict | None:
        if self.target_points is None or len(self.target_points) < 10:
            return None

        obb = self.compute_obb(self.target_points)

        approach = np.array([0.0, -1.0, 0.0])  # Y 음방향 (선반 안으로)

        # ★ approach와 수직인 OBB 축 자동 선택
        # short → mid → long 순서: 가능한 한 짧은 폭으로 잡기 위해
        closing = np.array([1.0, 0.0, 0.0])  # fallback (모든 축이 평행한 극단적 경우)
        for key in ("short_axis", "mid_axis", "long_axis"):
            ax   = obb[key].copy()
            ax[2] = 0.0  # z성분 제거 (수평 방향만 고려)
            norm = np.linalg.norm(ax)
            if norm < 1e-6:
                continue  # 이 축이 거의 수직이면 스킵
            ax /= norm

            # approach 방향과의 내적 절대값 < 0.8이면 "충분히 수직" → 채택
            if abs(ax.dot(approach)) < 0.8:
                closing = ax
                break  # 첫 번째로 조건을 만족하는 축 사용

        rotation = self.make_rotation_matrix(approach, closing)

        # grasp 위치: 물체 중심 그대로 사용 (수평 접근)
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
            "obb": obb,  # 디버깅/로그용으로 포함
        }

    # ──────────────────────────────────────────────────────────────
    # [함수 12] check_doosan_ik  (기존 코드 유지)
    #
    # DSR ikin() 함수로 IK(역기구학) 검사.
    # pose_mm_deg = [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]
    #
    # ikin() 반환 상태:
    #   status=0 → IK 성공, joint_solution(관절 6개 각도, deg) 반환
    #   status=1 → 작업영역 밖 (팔이 닿지 않는 위치)
    #   status=2 → 손목 특이점 (wrist singularity, 수학적으로 해가 무한)
    # ──────────────────────────────────────────────────────────────

    def check_doosan_ik(self, candidate: dict) -> tuple:
        pose = candidate.get("pose_mm_deg")
        if pose is None or len(pose) != 6:
            return False, None, "INVALID_POSE"

        try:
            target = self.dsr_posx(*pose)

            self.get_logger().info(
                "IK 검사 pose: "
                + ", ".join(f"{float(v):.2f}" for v in pose)
            )

            current_sol = 0
            try:
                _, current_sol = self.dsr_get_current_posx(ref=self.DR_BASE)
                current_sol = int(current_sol)
            except Exception:
                current_sol = 0

            sol_candidates = [current_sol] + [
                sol for sol in range(8) if sol != current_sol
            ]

            for sol_space in sol_candidates:
                self.get_logger().info(
                    f"IK solution space 검사: sol={sol_space}"
                )

                result = self.dsr_ikin(
                    target,
                    sol_space,
                    self.DR_BASE,
                )

                if result is None:
                    continue

                if np.isscalar(result) and float(result) == -1.0:
                    continue

                try:
                    joints = [float(v) for v in result]
                except (TypeError, ValueError):
                    continue

                if len(joints) != 6:
                    continue

                self.get_logger().info(
                    f"IK 계산 성공(sol={sol_space}): "
                    + ", ".join(f"{v:.2f}" for v in joints)
                )

                # MoveIt과 동일한 관절 제한 검사
                limits_ok, limit_reason = self.check_joint_limits(joints)

                if not limits_ok:
                    self.get_logger().warn(
                        f"IK 해 제외(sol={sol_space}) | "
                        f"관절 제한 위반: {limit_reason}"
                    )

                    # 현재 solution space를 버리고 다음 sol 검사
                    continue

                self.get_logger().info(
                    f"IK 최종 채택(sol={sol_space}): "
                    + ", ".join(f"{v:.2f}" for v in joints)
                )

                return True, joints, "OK"

            return False, None, "IK_NO_SOLUTION_WITHIN_JOINT_LIMITS"

        except Exception as e:
            return False, None, f"IK_EXCEPTION:{e}"

    def check_current_robot_state(self):
        """노드 시작 시 두산 API 연결 확인용 — 성공하면 현재 자세 로그 출력"""
        try:
            posj = self.dsr_get_current_posj()
            posx, sol = self.dsr_get_current_posx(ref=self.DR_BASE)
            self.get_logger().info(
                "현재 조인트: " + ", ".join(f"{float(v):.2f}" for v in posj))
            self.get_logger().info(
                "현재 TCP: " + ", ".join(f"{float(v):.2f}" for v in posx)
                + f" sol={sol}")
            return True
        except Exception as e:
            self.get_logger().error(f"로봇 상태 조회 실패: {e}")
            return False
    
    def check_joint_limits(self, joints: list[float]) -> tuple[bool, str]:
        if len(joints) != 6:
            return False, f"JOINT_COUNT_INVALID:{len(joints)}"

        violations = []

        for index, (value, limits) in enumerate(
            zip(joints, MOVEIT_JOINT_LIMITS_DEG),
            start=1,
        ):
            minimum, maximum = limits

            safe_minimum = minimum + JOINT_LIMIT_MARGIN_DEG
            safe_maximum = maximum - JOINT_LIMIT_MARGIN_DEG

            if value < safe_minimum or value > safe_maximum:
                violations.append(
                    f"J{index}={value:.2f}deg "
                    f"allowed=[{safe_minimum:.2f},{safe_maximum:.2f}]"
                )

        if violations:
            return False, "; ".join(violations)

        return True, "JOINT_LIMITS_OK"

    # ──────────────────────────────────────────────────────────────
    # 검증 파이프라인 (기존 로직 유지, layer 분기 추가)
    # ──────────────────────────────────────────────────────────────

    def validate_candidate(self, candidate, target_pts, env_pts) -> dict:
        """
        단일 파지 후보 검증 — 4단계 순차 검사.
        앞 단계 실패 시 즉시 반환 (이후 단계 스킵 → 효율적).
        """
        # 1단계: 물체 폭이 그리퍼 최대폭 안에 들어오는지
        w = self.calculate_required_width(target_pts, candidate["closing_direction"])
        if w + GRIPPER_MARGIN > GRIPPER_MAX_OPENING:
            return {"success": False, "reason": "WIDTH_TOO_LARGE", "required_width": w}

        # 2단계: grasp 위치에서 손가락/본체 공간에 장애물 없는지
        ok, reason = self.check_gripper_clearance(env_pts, candidate, w)
        if not ok:
            return {"success": False, "reason": reason, "required_width": w}

        # 3단계: pre-grasp → grasp 접근 경로에 장애물 없는지
        ok, reason = self.check_approach_path(env_pts, candidate, w)
        if not ok:
            return {"success": False, "reason": reason, "required_width": w}

        # 4단계: 두산 IK — 실제로 이 pose에 팔이 도달 가능한지
        ik_ok, joints, ik_reason = self.check_doosan_ik(candidate)
        if not ik_ok:
            return {"success": False, "reason": ik_reason, "required_width": w}

        return {"success": True, "reason": "OK", "required_width": w, "joints": joints}

    def select_grasp(self, top_cand, front_cand, target_pts, env_pts) -> dict:
        """
        층(layer) 판단 후 파지 전략 결정.

        bottom: FRONT만 시도
            → 하부 선반은 TOP 접근 공간 부족 (0.235m < 0.243m)

        top: TOP 먼저 시도 → 실패 시 FRONT → 둘 다 실패 시 NONE
            → 공간이 충분할 경우에만 TOP 시도
        """
        layer = self.object_layer or 'bottom'

        # TOP 시도 조건: 상부 선반(top) AND 공간 충분
        if layer == 'top' and SHELF_USABLE_HEIGHT > REQUIRED_TOP_SPACE:
            r = self.validate_candidate(top_cand, target_pts, env_pts)
            if r["success"]:
                return {"success": True, "grasp_type": "TOP",
                        "candidate": top_cand, "validation": r}
            self.get_logger().warn(f"TOP 실패: {r['reason']} → FRONT 시도")

        # FRONT 시도 (bottom이거나 TOP 실패 시)
        r = self.validate_candidate(front_cand, target_pts, env_pts)
        if r["success"]:
            return {"success": True, "grasp_type": "FRONT",
                    "candidate": front_cand, "validation": r}

        self.get_logger().warn(f"FRONT 실패: {r['reason']} → NONE")
        return {"success": False, "grasp_type": "NONE", "reason": "UNGRASPABLE"}

    def rotation_matrix_to_quaternion(self, rotation: np.ndarray) -> tuple:
        """3x3 회전행렬을 ROS quaternion (x, y, z, w)으로 변환."""
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
        """base_link 기준 position/rotation을 PoseStamped로 변환."""
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

    def publish_validated_grasp(self, result: dict):
        """성공한 후보 하나를 ValidatedGrasp 메시지로 발행."""
        candidate = result["candidate"]
        validation = result["validation"]
        joints = validation.get("joints")

        if joints is None or len(joints) != 6:
            self.get_logger().error(
                "IK 관절값이 6개가 아니므로 후보를 발행하지 않습니다.")
            return

        grasp_position = np.asarray(candidate["position"], dtype=np.float64)
        rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        approach = np.asarray(
            candidate["approach_direction"], dtype=np.float64)

        approach_norm = np.linalg.norm(approach)
        if approach_norm < 1e-9:
            self.get_logger().error(
                "approach_direction이 0 벡터이므로 후보를 발행하지 않습니다.")
            return
        approach /= approach_norm

        # grasp 쪽으로 이동하는 벡터가 approach이므로, 반대 방향으로 물러난 위치가 pre-grasp
        pre_grasp_position = (
            grasp_position - approach * PREGRASP_DISTANCE
        )

        a, b, c = self._rotation_to_dsr_euler(rotation)
        pre_grasp_candidate = {
            "pose_mm_deg": [
                pre_grasp_position[0] * 1000,
                pre_grasp_position[1] * 1000,
                pre_grasp_position[2] * 1000,
                a, b, c,
            ]
        }
        pre_ik_ok, pre_joints, pre_reason = self.check_doosan_ik(pre_grasp_candidate)

        if not pre_ik_ok:
            self.get_logger().error(
                f"pre-grasp IK 실패({pre_reason}) — 후보를 발행하지 않습니다.")
            return

        stamp = self.get_clock().now().to_msg()
        msg = ValidatedGrasp()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.grasp_type = str(result["grasp_type"])
        msg.pre_grasp_pose = self.make_pose_stamped(
            pre_grasp_position, rotation, stamp)
        msg.grasp_pose = self.make_pose_stamped(
            grasp_position, rotation, stamp)
        msg.grasp_joints = [float(v) for v in joints]
        msg.required_width = float(validation["required_width"])

        self.result_pub.publish(msg)
        self.get_logger().info(
            f"ValidatedGrasp 발행 | type={msg.grasp_type} | "
            f"pre=({pre_grasp_position[0]:.3f}, "
            f"{pre_grasp_position[1]:.3f}, "
            f"{pre_grasp_position[2]:.3f}) | "
            f"grasp=({grasp_position[0]:.3f}, "
            f"{grasp_position[1]:.3f}, "
            f"{grasp_position[2]:.3f})"
        )

    def run_validation(self):
        """TOP/FRONT 후보 생성 → 검증 → 성공 후보만 메시지 발행."""
        top_cand = self.create_top_candidate()
        front_cand = self.create_front_candidate()

        if top_cand is None or front_cand is None:
            self.get_logger().warn("후보 생성 실패: 메시지를 발행하지 않습니다.")
            return

        result = self.select_grasp(
            top_cand, front_cand,
            self.target_points, self.environment_points)

        if not result.get("success", False):
            self.get_logger().warn(
                f"검증 통과 후보 없음: {result.get('reason', 'UNKNOWN')}")
            return

        self.publish_validated_grasp(result)


# ============================================================
# 진입점
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # 1) ROS 노드를 먼저 완전히 생성한다.
        node = GraspValidatorNode()

        # 2) 생성된 노드를 DR_init에 등록한다.
        #    DSR_ROBOT2는 import될 때 이 노드로 서비스 클라이언트를 생성한다.
        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = node

        # 3) 반드시 DR_init 설정이 끝난 다음 두산 API를 import한다.
        node._load_doosan_api()

        node.check_current_robot_state()  # 시작 시 두산 API 연결 확인
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.validation_pending:
                node.validation_pending = False
                node.run_validation()
                node.target_points = None
                node.environment_points = None                  # 콜백 루프 시작
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