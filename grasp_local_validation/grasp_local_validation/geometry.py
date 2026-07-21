"""
좌표 변환, PCA 기반 OBB 계산, 방향(orientation) 유틸리티.

이 모듈은 ROS 메시지에 의존하지 않는 순수 numpy 로직만 담는다.
(단위테스트를 ROS 없이 돌리기 위함)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# 좌표 변환
# ---------------------------------------------------------------------------

def pose_to_matrix(position, quat_xyzw):
    """geometry_msgs/Pose 성분 -> 4x4 동차행렬"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def matrix_to_pose_components(T):
    """4x4 동차행렬 -> (position(3,), quat_xyzw(4,))"""
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
    return pos, quat


def transform_points(points, T):
    """(N,3) 점군을 4x4 행렬로 변환"""
    if len(points) == 0:
        return points
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    return (T @ pts_h.T).T[:, :3]


# ---------------------------------------------------------------------------
# OBB / PCA
# ---------------------------------------------------------------------------

def compute_obb(target_cloud, min_points=10):
    """
    PCA 기반 OBB 계산.
    반환 dict 는 grasp_interfaces/msg/ObjectGeometry 필드와 1:1 대응되도록 구성.
    """
    if target_cloud is None or len(target_cloud) < min_points:
        return None

    center = target_cloud.mean(axis=0)
    centered = target_cloud - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # [long, mid, short]

    # 부호 고정: z성분이 양수가 되도록 (재현성 있는 축 방향)
    for i in range(3):
        if axes[i][2] < 0:
            axes[i] = -axes[i]

    projected = centered @ axes.T
    extents = projected.max(axis=0) - projected.min(axis=0)

    return {
        "center": center,
        "long_axis": axes[0],
        "mid_axis": axes[1],
        "short_axis": axes[2],
        "extents": extents,           # [long, mid, short] length (m)
        "top_z": float(target_cloud[:, 2].max()),
        "bottom_z": float(target_cloud[:, 2].min()),
        "point_count": int(len(target_cloud)),
    }


def sanity_check_obb(obb, profile_size_mm, tol_low=0.5, tol_high=2.0):
    """OBB 크기가 프로파일과 크게 다르면 가림/노이즈로 판단"""
    if obb is None:
        return False, "obb_none"

    profile_sorted = sorted([s / 1000.0 for s in profile_size_mm], reverse=True)
    obb_sorted = sorted(obb["extents"], reverse=True)

    for p, o in zip(profile_sorted, obb_sorted):
        if p <= 1e-6:
            continue
        ratio = o / p
        if ratio < tol_low or ratio > tol_high:
            return False, f"size_mismatch profile={p:.3f} obb={o:.3f}"
    return True, "ok"


def project_to_horizontal(axis_3d):
    """3D 축을 xy 평면에 투영, 정규화. 완전 수직축이면 기본값 반환."""
    h = np.array(axis_3d, dtype=float).copy()
    h[2] = 0.0
    n = np.linalg.norm(h)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return h / n


# ---------------------------------------------------------------------------
# 그리퍼 방향(orientation) 생성
# ---------------------------------------------------------------------------

def top_down_orientation(finger_axis_world):
    """
    Top approach: 그리퍼 진입 방향(z') = -Z(world, 아래 방향)
    finger_axis_world: 손가락이 벌어지는 방향(수평면에 투영된 벡터)
    반환: quaternion (x,y,z,w)
    """
    finger_dir = project_to_horizontal(finger_axis_world)
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = finger_dir
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    return R.from_matrix(rot_mat).as_quat()


def front_orientation(approach_axis_world=(0.0, -1.0, 0.0)):
    """
    Front approach: 그리퍼가 -Y(선반 앞 -> 안쪽) 방향으로 수평 진입.
    approach_axis_world: 실제 로봇 베이스 좌표계에서 선반을 향하는 방향으로 조정 필요.
    """
    z_axis = np.array(approach_axis_world, dtype=float)
    z_axis /= (np.linalg.norm(z_axis) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])
    return R.from_matrix(rot_mat).as_quat()
