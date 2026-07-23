"""
shelf_collision.yaml 의 collision_objects (임의 개수의 axis-aligned box) 를
공통적으로 다루는 유틸리티.

박스 1개 형식:
  id: string
  center: [x, y, z]
  size: [sx, sy, sz]

2층 선반이라면 collision_objects 에 "shelf_mid_underside" 같은 박스를
추가로 등록하면 됨 (이 모듈은 이름에 의존하지 않고 기하학적으로만 판단).
"""
import numpy as np


def box_min_max(box):
    center = np.array(box['center'], dtype=float)
    size = np.array(box['size'], dtype=float)
    return center - size / 2.0, center + size / 2.0


def boxes_overlap(a_min, a_max, b_min, b_max):
    return bool(np.all(a_max >= b_min) and np.all(b_max >= a_min))


def horizontal_overlap(point_xy, box_min, box_max, pad=0.0):
    """점(x,y)이 박스의 xy 범위 안(pad만큼 확장)에 있는지"""
    return (box_min[0] - pad <= point_xy[0] <= box_max[0] + pad and
            box_min[1] - pad <= point_xy[1] <= box_max[1] + pad)


def clearance_above(object_center_xy, object_top_z, shelf_cfg, pad=0.05):
    """
    물체 바로 위에서 가장 가까운 장애물(선반 상판 등)까지의 수직 여유공간(m).
    장애물이 없으면 workspace z_max 기준으로 계산.
    """
    best = None
    for box in shelf_cfg.get('collision_objects', []):
        b_min, b_max = box_min_max(box)
        if b_min[2] <= object_top_z:
            continue  # 물체보다 아래/같은 높이는 후보 아님
        if not horizontal_overlap(object_center_xy, b_min, b_max, pad=pad):
            continue
        gap = b_min[2] - object_top_z
        if best is None or gap < best:
            best = gap

    if best is not None:
        return float(best)

    ws = shelf_cfg.get('workspace', {})
    z_max = ws.get('z_max', 1.5)
    return float(z_max - object_top_z)


def nearest_floor_z_below(object_center_xy, object_bottom_z, shelf_cfg, pad=0.05):
    """물체 바로 아래 바닥(선반)의 z. 없으면 workspace z_min."""
    best = None
    for box in shelf_cfg.get('collision_objects', []):
        b_min, b_max = box_min_max(box)
        if b_max[2] > object_bottom_z + 0.01:
            continue
        if not horizontal_overlap(object_center_xy, b_min, b_max, pad=pad):
            continue
        if best is None or b_max[2] > best:
            best = b_max[2]

    if best is not None:
        return float(best)
    ws = shelf_cfg.get('workspace', {})
    return float(ws.get('z_min', 0.0))


def collect_boxes_as_arrays(shelf_cfg):
    """모든 선반 박스를 (min, max) 리스트로 변환 (충돌검사용)"""
    result = []
    for box in shelf_cfg.get('collision_objects', []):
        b_min, b_max = box_min_max(box)
        result.append((box.get('id', 'unnamed'), b_min, b_max))
    return result
