"""
GraspCandidate 하나를 받아 RG2 형상을 배치하고,
target_cloud(접촉 확인용) / environment_cloud(충돌 확인용) / shelf_collision.yaml(고정 장애물)
과 비교해 검증한다.

여기서 사용하는 실패 코드는 grasp_interfaces/msg/ValidatedGrasp.msg 상수와 동일하다.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from . import shelf_geometry as shelfgeo

VALID = 0
WIDTH_TOO_LARGE = 1
WIDTH_TOO_SMALL = 2
CONTACT_MISSING = 3
FINGER_COLLISION = 4
BODY_COLLISION = 5
SHELF_COLLISION = 6
APPROACH_COLLISION = 7


def _box_corners_world(local_min, local_max, pos, quat_xyzw):
    rot = R.from_quat(quat_xyzw).as_matrix()
    corners = np.array([
        [local_min[0], local_min[1], local_min[2]],
        [local_max[0], local_min[1], local_min[2]],
        [local_min[0], local_max[1], local_min[2]],
        [local_max[0], local_max[1], local_min[2]],
        [local_min[0], local_min[1], local_max[2]],
        [local_max[0], local_min[1], local_max[2]],
        [local_min[0], local_max[1], local_max[2]],
        [local_max[0], local_max[1], local_max[2]],
    ])
    world = corners @ rot.T + pos
    return world.min(axis=0), world.max(axis=0)


def compute_gripper_boxes(pos, quat_xyzw, opening_width, rg2_cfg):
    """
    grasp pose(pos, quat) 는 "손가락 끝 접촉점(TCP)"을 의미한다.
    rg2_collision.yaml 의 좌표들은 rg2_base_link(마운트 플랜지) 기준이므로,
    손가락 끝(z = total_length) 이 로컬 원점(0,0,0)이 되도록 z를 이동시켜서 사용한다.
    이렇게 하면 로컬 +z 는 "손끝 -> 몸통" 방향이 되고, 그리퍼가 향하는 진행 방향은 로컬 -z 이다.

    camera_mount 가 rg2_collision.yaml 에 있고 enabled:true 이면 그리퍼에 붙은
    카메라 마운트도 충돌 박스로 함께 계산한다 (없으면 무시).
    """
    cg = rg2_cfg['collision_geometry']
    spec = rg2_cfg['spec']
    z_shift = spec['total_length']  # 손끝을 원점으로
    half_open = opening_width / 2.0

    lf = cg['left_finger']
    rf = cg['right_finger']

    def finger_box(finger_cfg, half_open):
        sx, sy, sz = finger_cfg['size']
        cx = finger_cfg['center_x']
        cz = finger_cfg['center_z'] - z_shift
        y_sign = finger_cfg['y_sign']
        cy = y_sign * (half_open + sy / 2.0)
        lo = np.array([cx - sx/2, cy - sy/2, cz - sz/2])
        hi = np.array([cx + sx/2, cy + sy/2, cz + sz/2])
        return lo, hi

    left_lo, left_hi = finger_box(lf, half_open)
    right_lo, right_hi = finger_box(rf, half_open)

    body = cg['body']
    body_center = np.array(body['center']) - np.array([0, 0, z_shift])
    body_lo = body_center - np.array(body['size']) / 2.0
    body_hi = body_center + np.array(body['size']) / 2.0

    linkage = cg['linkage_envelope']
    link_center = np.array(linkage['center']) - np.array([0, 0, z_shift])
    link_lo = link_center - np.array(linkage['size']) / 2.0
    link_hi = link_center + np.array(linkage['size']) / 2.0

    boxes = {
        "left": _box_corners_world(left_lo, left_hi, pos, quat_xyzw),
        "right": _box_corners_world(right_lo, right_hi, pos, quat_xyzw),
        "body": _box_corners_world(body_lo, body_hi, pos, quat_xyzw),
        "linkage": _box_corners_world(link_lo, link_hi, pos, quat_xyzw),
    }

    cam = cg.get('camera_mount')
    if cam and cam.get('enabled', False):
        cam_center = np.array(cam['center_local']) - np.array([0, 0, z_shift])
        cam_lo = cam_center - np.array(cam['size']) / 2.0
        cam_hi = cam_center + np.array(cam['size']) / 2.0
        boxes["camera_mount"] = _box_corners_world(cam_lo, cam_hi, pos, quat_xyzw)

    return boxes


FINGER_PARTS = ["left", "right"]


def rigid_parts(boxes):
    """손가락을 제외한 나머지(본체/링키지/카메라마운트 등) - 접촉이 허용되지 않는 부분"""
    return [name for name in boxes.keys() if name not in FINGER_PARTS]


def points_in_box(points, box_min, box_max):
    if points is None or len(points) == 0:
        return 0
    mask = np.all((points >= box_min) & (points <= box_max), axis=1)
    return int(mask.sum())


def boxes_overlap(a_min, a_max, b_min, b_max):
    return bool(np.all(a_max >= b_min) and np.all(b_max >= a_min))


def validate_candidate(candidate, target_cloud, environment_cloud, rg2_cfg, shelf_cfg,
                        min_contact_points=10, approach_sample_step=0.01):
    """
    candidate: dict, candidate_generation.py 가 만든 형식
               (grasp_pos, pre_grasp_pos, orientation_xyzw, required_width 등)
    반환: (valid: bool, failure_code: int, failure_reason: str, extra: dict)
    """
    spec = rg2_cfg['spec']
    safety = rg2_cfg['safety']

    width = candidate['required_width']

    # 1. 폭 검사
    if width > spec['usable_max_opening']:
        return False, WIDTH_TOO_LARGE, f"required_width {width:.3f} > usable_max {spec['usable_max_opening']:.3f}", {}
    if width < spec['physical_min_opening']:
        return False, WIDTH_TOO_SMALL, f"required_width {width:.3f} < physical_min", {}

    pos = candidate['grasp_pos']
    quat = candidate['orientation_xyzw']
    boxes = compute_gripper_boxes(pos, quat, width, rg2_cfg)

    # 2. 접촉 확인 (손가락 안쪽 영역에 target_cloud 점이 실제로 있는가)
    left_contact = points_in_box(target_cloud, *boxes['left'])
    right_contact = points_in_box(target_cloud, *boxes['right'])
    if left_contact < min_contact_points or right_contact < min_contact_points:
        return False, CONTACT_MISSING, (
            f"contact points L={left_contact} R={right_contact} (min {min_contact_points})"
        ), {"left_contact": left_contact, "right_contact": right_contact}

    # 3. 손가락 주변 environment_cloud 충돌 (진입 공간 막혔는지)
    thr = safety['collision_point_threshold']
    for name in FINGER_PARTS:
        n = points_in_box(environment_cloud, *boxes[name])
        if n > thr:
            return False, FINGER_COLLISION, f"{name} finger blocked by {n} env points", {}

    # 4. 본체/링키지/카메라마운트 충돌 (environment_cloud + 고정 선반 박스)
    rigid = rigid_parts(boxes)
    for name in rigid:
        n = points_in_box(environment_cloud, *boxes[name])
        if n > 0:
            return False, BODY_COLLISION, f"{name} hit by {n} env points", {}

    all_parts = FINGER_PARTS + rigid
    for box in shelf_cfg.get('collision_objects', []):
        b_min, b_max = shelfgeo.box_min_max(box)
        for name in all_parts:
            if boxes_overlap(boxes[name][0], boxes[name][1], b_min, b_max):
                return False, SHELF_COLLISION, f"{name} overlaps shelf box '{box.get('id')}'", {}

    # 5. 접근 경로 충돌 (pre_grasp -> grasp 사이 샘플링)
    pre = candidate['pre_grasp_pos']
    n_samples = max(2, int(np.linalg.norm(pos - pre) / approach_sample_step))
    min_clearance = float('inf')
    for t in np.linspace(0.0, 1.0, n_samples):
        interp_pos = pre + (pos - pre) * t
        step_boxes = compute_gripper_boxes(interp_pos, quat, width, rg2_cfg)
        for name in all_parts:
            n = points_in_box(environment_cloud, *step_boxes[name])
            if n > 0:
                return False, APPROACH_COLLISION, f"approach step t={t:.2f} '{name}' hit {n} env points", {}
            for box in shelf_cfg.get('collision_objects', []):
                b_min, b_max = shelfgeo.box_min_max(box)
                if boxes_overlap(step_boxes[name][0], step_boxes[name][1], b_min, b_max):
                    return False, APPROACH_COLLISION, (
                        f"approach step t={t:.2f} '{name}' overlaps shelf '{box.get('id')}'"
                    ), {}

    extra = {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "min_clearance": min_clearance if min_clearance != float('inf') else 0.0,
    }
    return True, VALID, "ok", extra


def score_candidate(candidate, extra):
    """local_score 계산 (초기 순위용, 최종 판단 아님)"""
    score = candidate.get('initial_score', 0.0)
    if candidate['type'].startswith('top'):
        score += 2.0
    if 'center' in candidate['type']:
        score += 1.0
    if 'rot90' in candidate['type']:
        score -= 0.5
    contact = extra.get('left_contact', 0) + extra.get('right_contact', 0)
    score += min(contact, 200) * 0.01
    return score
