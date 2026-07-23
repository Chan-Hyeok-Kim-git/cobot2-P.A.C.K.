"""
ObjectGeometry(물체 OBB) -> GraspCandidate 후보 목록 생성.

담당자1(디텍션) 쪽에서 실시간으로 물체의 3D 위치/치수만 넘겨주면,
top이 물리적으로 가능한지(clearance_above vs RG2 필요 공간)를
그 자리에서 계산해서 top/front를 순수 기하학적으로 결정한다.
클래스별 사전 선호도(grasp_profiles.yaml) 는 더 이상 사용하지 않는다.

여기서 반환하는 dict 의 key 들은 grasp_interfaces/msg/GraspCandidate.msg 필드와
1:1로 대응된다. ROS 노드(candidate_generator_node.py)에서 이 dict 를 실제
GraspCandidate 메시지로 변환한다.

grasp_type : 0=TOP, 1=FRONT, 2=SIDE  (GraspCandidate.msg 의 상수와 동일)
"""
import numpy as np
from . import geometry as geo
from . import shelf_geometry as shelfgeo

GRASP_TOP = 0
GRASP_FRONT = 1
GRASP_SIDE = 2

APPROACH_OFFSET_M = 0.10
POSITION_JITTER_M = 0.015
FLOOR_SAFETY_MARGIN_M = 0.008


def _top_candidate(cand_type, center, grip_width, finger_axis, approach_offset):
    quat = geo.top_down_orientation(finger_axis)
    pre = center + np.array([0.0, 0.0, approach_offset])
    return {
        "type": cand_type,
        "grasp_type": GRASP_TOP,
        "grasp_pos": center,
        "pre_grasp_pos": pre,
        "orientation_xyzw": quat,
        "approach_direction": np.array([0.0, 0.0, -1.0]),
        "closing_direction": geo.project_to_horizontal(finger_axis),
        "finger_direction": geo.project_to_horizontal(finger_axis),
        "required_width": float(grip_width),
        "approach_distance": float(approach_offset),
    }


def _front_candidate(cand_type, center, grip_width, approach_dir, approach_offset):
    quat = geo.front_orientation(approach_dir)
    pre = center - np.array(approach_dir) * approach_offset
    closing = np.cross(approach_dir, [0, 0, 1])
    closing = closing / (np.linalg.norm(closing) + 1e-9)
    return {
        "type": cand_type,
        "grasp_type": GRASP_FRONT,
        "grasp_pos": center,
        "pre_grasp_pos": pre,
        "orientation_xyzw": quat,
        "approach_direction": np.array(approach_dir, dtype=float),
        "closing_direction": closing,
        "finger_direction": np.array([0.0, 0.0, 1.0]),
        "required_width": float(grip_width),
        "approach_distance": float(approach_offset),
    }


def required_top_clearance_m(rg2_cfg, extra_margin=0.03):
    """
    Top approach 가 물리적으로 가능하려면 물체 위쪽에 최소 이만큼의
    수직 공간이 있어야 한다 (그리퍼 전체 길이 + 안전 여유).
    RG2 total_length 는 마운트~손끝까지의 전체 길이이므로 이 값을 그대로 쓴다.
    """
    return rg2_cfg['spec']['total_length'] + extra_margin


def decide_approach_order(clearance_above_m, rg2_cfg):
    """
    순수 기하학적 판단: top 이 물리적으로 들어갈 공간이 있으면 top을 먼저 시도하고
    (여러 위치/회전 변형 포함) front 를 대비책으로 남겨둔다.
    공간이 없으면 아예 top 후보를 만들지 않고 front 부터 시도한다.
    """
    top_feasible = clearance_above_m > required_top_clearance_m(rg2_cfg)
    if top_feasible:
        return ["top_center", "top_ofs_p", "top_ofs_n", "top_rot90",
                "front_center", "front_ofs_p", "front_ofs_n"]
    return ["front_center", "front_ofs_p", "front_ofs_n"]


def generate_candidates(obb, rg2_cfg, shelf_cfg, approach_dir_front=(0.0, -1.0, 0.0)):
    """
    obb: geometry.compute_obb() 결과 (또는 실시간 디텍션에서 만든 동일 형식의 dict)
    반환: dict 리스트 (각 원소가 GraspCandidate 필드)

    담당자1 이 실시간으로 넘겨주는 정보는 obb 하나로 충분하다:
      center, long_axis/mid_axis/short_axis, extents, top_z, bottom_z
    """
    center = obb["center"]
    long_ext, mid_ext, short_ext = obb["extents"]
    long_h = geo.project_to_horizontal(obb["long_axis"])
    short_h = geo.project_to_horizontal(obb["short_axis"])

    spec = rg2_cfg['spec']
    safety = rg2_cfg['safety']
    max_open = spec['usable_max_opening']
    margin = safety['width_margin']

    clearance = shelfgeo.clearance_above(center[:2], obb["top_z"], shelf_cfg)
    order = decide_approach_order(clearance, rg2_cfg)

    candidates = []
    seen = set()

    for p in order:
        if p.startswith("top") and "top_center" not in seen:
            grip_w = min(mid_ext, short_ext) + margin
            if grip_w < max_open:
                candidates.append(_top_candidate("top_center", center, grip_w, long_h, APPROACH_OFFSET_M))
                seen.add("top_center")
                for name, off in [("top_ofs_p", POSITION_JITTER_M), ("top_ofs_n", -POSITION_JITTER_M)]:
                    shifted = center + long_h * off
                    candidates.append(_top_candidate(name, shifted, grip_w, long_h, APPROACH_OFFSET_M))
                    seen.add(name)
            grip_w90 = long_ext + margin
            if grip_w90 < max_open:
                candidates.append(_top_candidate("top_rot90", center, grip_w90, short_h, APPROACH_OFFSET_M))
                seen.add("top_rot90")

        elif p.startswith("front") and "front_center" not in seen:
            grip_w = short_ext + margin
            if grip_w < max_open:
                # 손가락이 바닥에 닿는지 대략 체크 (정밀 검사는 collision.py 에서 다시 수행)
                floor_z = shelfgeo.nearest_floor_z_below(center[:2], obb["bottom_z"], shelf_cfg)
                finger_half = rg2_cfg['collision_geometry']['left_finger']['size'][2] / 2.0
                if center[2] - finger_half > floor_z + FLOOR_SAFETY_MARGIN_M:
                    candidates.append(_front_candidate("front_center", center, grip_w, approach_dir_front, APPROACH_OFFSET_M))
                    seen.add("front_center")
                    for name, off in [("front_ofs_p", POSITION_JITTER_M), ("front_ofs_n", -POSITION_JITTER_M)]:
                        shifted = center + np.array([off, 0.0, 0.0])
                        candidates.append(_front_candidate(name, shifted, grip_w, approach_dir_front, APPROACH_OFFSET_M))
                        seen.add(name)

    return candidates
