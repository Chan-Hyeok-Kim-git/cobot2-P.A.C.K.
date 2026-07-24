import struct
import math
import numpy as np
import rclpy
import DR_init

from collections import deque
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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

COLLISION_THRESHOLD = 5
PREGRASP_DISTANCE   = 0.100
APPROACH_STEP       = 0.010


# ============================================================
# 3. 선반 설정
# ============================================================

SHELF_USABLE_HEIGHT = 0.235
REQUIRED_TOP_SPACE  = TOTAL_LENGTH + 0.030


# ============================================================
# 4. ROS 토픽명
# ============================================================

TARGET_CLOUD_TOPIC      = "/ai/object_points_base"
ENVIRONMENT_CLOUD_TOPIC = "/ai/background_points_base"
EXPECTED_CLOUD_FRAME     = "base_link"
GRASP_RESULT_TOPIC      = "/grasp/validated_grasp"

# ★ 삭제됨: hand-eye 캘리브레이션(T_gripper2camera.npy) 로드 관련 코드 전부 제거.
# vision 쪽에서 이제 카메라→base_link 변환을 이미 끝낸 point cloud를 직접
# 보내주기로 했으므로, 이 노드가 다시 변환할 필요가 없어졌다.

# ★ 추가: target_cloud에 여러 물체가 섞여서 오므로, 그중 지금 찾는 물품(active_item)의
# 뭉치만 골라내기 위한 클러스터링 + 크기매칭 설정.
ACTIVE_ITEM_TOPIC = "/task/active_item"

CLUSTER_EPS_MM = 25.0        # 이 거리(mm) 안의 점들을 같은 물체로 묶음
CLUSTER_MIN_POINTS = 15      # 이보다 점이 적은 뭉치는 노이즈로 버림
MATCH_MAX_DIST_MM = 60.0     # 이보다 멀면 매칭 실패로 간주

# 각 class를 PCA로 정렬했을 때 3개 주축 길이를 오름차순 정렬한 값 (mm)
# ★ 실측치로 교체 필요
CLASS_DIMENSIONS_MM = {
    "gas_mask":   (120.0, 180.0, 220.0),
    "flashlight": (30.0,  35.0,  200.0),
    "rope":       (70.0,  70.0,  90.0),
}


def cluster_points(points_m: np.ndarray, eps_mm: float, min_points: int) -> list:
    """가까운 점끼리 묶어서 물체 단위 뭉치로 분리 (DBSCAN 스타일, cKDTree 사용)."""
    if points_m.shape[0] == 0:
        return []
    eps_m = eps_mm / 1000.0
    tree = cKDTree(points_m)
    n = points_m.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        members = [i]
        while stack:
            cur = stack.pop()
            for nb in tree.query_ball_point(points_m[cur], eps_m):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
                    members.append(nb)
        if len(members) >= min_points:
            clusters.append(points_m[members])
    return clusters


def _sorted_extents_mm(points_m: np.ndarray) -> np.ndarray:
    """PCA로 물체를 정렬한 뒤 3개 주축 길이를 오름차순 반환 (mm). 회전에 무관."""
    center = points_m.mean(axis=0)
    centered = points_m - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    local = centered @ eigvecs
    extents = local.max(axis=0) - local.min(axis=0)
    return np.sort(extents * 1000.0)


def classify_cluster(points_m: np.ndarray) -> tuple:
    """클러스터 크기를 CLASS_DIMENSIONS_MM과 비교해 가장 가까운 class 반환.
    반환: (class_name 또는 None, 거리mm)"""
    if points_m.shape[0] < 4:
        return None, float('inf')
    extents = _sorted_extents_mm(points_m)
    best_name, best_dist = None, float('inf')
    for name, ref in CLASS_DIMENSIONS_MM.items():
        ref_sorted = np.sort(np.array(ref, dtype=float))
        dist = float(np.linalg.norm(extents - ref_sorted))
        if dist < best_dist:
            best_name, best_dist = name, dist
    if best_dist > MATCH_MAX_DIST_MM:
        return None, best_dist
    return best_name, best_dist


# ============================================================
# 5. 노드 본체
# ============================================================

class GraspValidatorNode(Node):

    def __init__(self):
        super().__init__("grasp_validator", namespace=ROBOT_ID)

        self.target_points      = None
        self.environment_points = None
        self.validation_pending = False

        self.object_layer = 'bottom'

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
        self.create_subscription(String, ACTIVE_ITEM_TOPIC, self.on_active_item, 10)

        self.create_subscription(
            PointCloud2,
            TARGET_CLOUD_TOPIC,
            self.target_cloud_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            ENVIRONMENT_CLOUD_TOPIC,
            self.environment_cloud_callback,
            qos_profile_sensor_data,
        )

        self.result_pub = self.create_publisher(
            ValidatedGrasp, GRASP_RESULT_TOPIC, 10)

        self.get_logger().info(
            f"GraspValidatorNode 시작 | "
            f"TOP 필요={REQUIRED_TOP_SPACE*1000:.0f}mm | "
            f"선반 높이={SHELF_USABLE_HEIGHT*1000:.0f}mm | "
            f"→ 이 선반은 항상 FRONT 파지"
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

    def on_active_item(self, msg: String) -> None:
        """task_manager가 '지금 이 물품을 찾는 중'이라고 알려주면 저장.
        바뀌면 이전 물품의 잔여 상태를 초기화한다."""
        new_item = msg.data
        if new_item == self.active_item:
            return
        self.active_item = new_item
        self.target_points = None
        self.environment_points = None
        self._published_for_current_item = False

    def target_cloud_callback(self, msg: PointCloud2):
        if not self.active_item or self._published_for_current_item:
            return  # gate 닫힘 또는 이번 물품은 이미 발행 완료 — 재처리 안 함

        if msg.header.frame_id != EXPECTED_CLOUD_FRAME:
            self.get_logger().warn(
                f"목표 점군 frame_id 오류: expected={EXPECTED_CLOUD_FRAME}, "
                f"received={msg.header.frame_id}"
            )
            return

        # ★ 수정: vision이 이미 base_link 좌표로 변환해서 주므로,
        # 예전처럼 get_current_posx()로 로봇 자세를 읽어서 변환하는 과정이
        # 통째로 빠졌다. 받은 점을 그대로 base_link 좌표로 사용한다.
        base_pts = self.pointcloud2_to_numpy(msg)
        if len(base_pts) == 0:
            return

        # ★ 핵심: 섞인 점들을 물체 단위로 분리(클러스터링) →
        # 각 뭉치 크기를 CLASS_DIMENSIONS_MM과 비교(크기매칭) →
        # active_item과 일치하는 뭉치만 골라서 target_points로 사용
        clusters = cluster_points(base_pts, CLUSTER_EPS_MM, CLUSTER_MIN_POINTS)
        matched_cluster, best_dist = None, float('inf')
        for cluster in clusters:
            name, dist = classify_cluster(cluster)
            if name == self.active_item and dist < best_dist:
                matched_cluster, best_dist = cluster, dist

        if matched_cluster is None:
            self.get_logger().debug(
                f"[{self.active_item}] 이번 프레임에서 매칭되는 뭉치 없음 "
                f"(클러스터 {len(clusters)}개 중 매칭 실패) — 대기"
            )
            self.target_points = None
            return

        self.get_logger().debug(
            f"[{self.active_item}] 매칭 성공: {matched_cluster.shape[0]}점, "
            f"매칭거리={best_dist:.1f}mm"
        )
        self.target_points = matched_cluster
        self.try_run_validation()

    def environment_cloud_callback(self, msg: PointCloud2):
        if not self.active_item or self._published_for_current_item:
            return  # target과 동일한 gate 적용

        if msg.header.frame_id != EXPECTED_CLOUD_FRAME:
            self.get_logger().warn(
                f"환경 점군 frame_id 오류: expected={EXPECTED_CLOUD_FRAME}, "
                f"received={msg.header.frame_id}"
            )
            return

        # ★ 수정: 여기도 마찬가지로 변환 없이 받은 좌표를 그대로 사용
        self.environment_points = self.pointcloud2_to_numpy(msg)
        self.try_run_validation()

    def try_run_validation(self):
        if self.target_points is None or self.environment_points is None:
            return
        self.validation_pending = True

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
                    f"IK 성공(sol={sol_space}): "
                    + ", ".join(f"{v:.2f}" for v in joints)
                )

                return True, joints, "OK"

            return False, None, "IK_NO_SOLUTION"

        except Exception as e:
            return False, None, f"IK_EXCEPTION:{e}"

    def check_current_robot_state(self):
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

    def validate_candidate(self, candidate, target_pts, env_pts) -> dict:
        w = self.calculate_required_width(target_pts, candidate["closing_direction"])
        if w + GRIPPER_MARGIN > GRIPPER_MAX_OPENING:
            return {"success": False, "reason": "WIDTH_TOO_LARGE", "required_width": w}

        ok, reason = self.check_gripper_clearance(env_pts, candidate, w)
        if not ok:
            return {"success": False, "reason": reason, "required_width": w}

        ok, reason = self.check_approach_path(env_pts, candidate, w)
        if not ok:
            return {"success": False, "reason": reason, "required_width": w}

        ik_ok, joints, ik_reason = self.check_doosan_ik(candidate)
        if not ik_ok:
            return {"success": False, "reason": ik_reason, "required_width": w}

        return {"success": True, "reason": "OK", "required_width": w, "joints": joints}

    def select_grasp(self, top_cand, front_cand, target_pts, env_pts) -> dict:
        layer = self.object_layer or 'bottom'

        if layer == 'top' and SHELF_USABLE_HEIGHT > REQUIRED_TOP_SPACE:
            r = self.validate_candidate(top_cand, target_pts, env_pts)
            if r["success"]:
                return {"success": True, "grasp_type": "TOP",
                        "candidate": top_cand, "validation": r}
            self.get_logger().warn(f"TOP 실패: {r['reason']} → FRONT 시도")

        r = self.validate_candidate(front_cand, target_pts, env_pts)
        if r["success"]:
            return {"success": True, "grasp_type": "FRONT",
                    "candidate": front_cand, "validation": r}

        self.get_logger().warn(f"FRONT 실패: {r['reason']} → NONE")
        return {"success": False, "grasp_type": "NONE", "reason": "UNGRASPABLE"}

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

    def publish_validated_grasp(self, result: dict):
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
        msg.pre_grasp_joints = [float(v) for v in pre_joints]

        self.result_pub.publish(msg)
        self._published_for_current_item = True  # ★ 추가: 이번 물품 끝 — 재발행 방지
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
        node = GraspValidatorNode()

        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = node

        node._load_doosan_api()

        node.check_current_robot_state()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.validation_pending:
                node.validation_pending = False
                node.run_validation()

                node.target_points = None
                node.environment_points = None
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