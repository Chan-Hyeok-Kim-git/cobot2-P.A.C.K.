"""
sensor_msgs/PointCloud2 <-> numpy 변환.
sensor_msgs_py 의존성 없이 struct 로 직접 처리 (환경에 따라 미설치인 경우 대비).
"""
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def pointcloud2_to_xyz_array(cloud_msg: PointCloud2) -> np.ndarray:
    """PointCloud2(xyz 포함, float32 가정) -> (N,3) ndarray"""
    if cloud_msg is None or cloud_msg.width * cloud_msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fmt = _make_struct_format(cloud_msg)
    unpacker = struct.Struct(fmt)
    step = cloud_msg.point_step
    n_points = cloud_msg.width * cloud_msg.height
    data = cloud_msg.data

    offsets = {f.name: f.offset for f in cloud_msg.fields}
    x_off, y_off, z_off = offsets.get('x'), offsets.get('y'), offsets.get('z')
    if x_off is None or y_off is None or z_off is None:
        return np.zeros((0, 3), dtype=np.float32)

    pts = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        base = i * step
        pts[i, 0] = struct.unpack_from('<f', data, base + x_off)[0]
        pts[i, 1] = struct.unpack_from('<f', data, base + y_off)[0]
        pts[i, 2] = struct.unpack_from('<f', data, base + z_off)[0]

    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def _make_struct_format(cloud_msg):
    # 현재는 사용하지 않지만 확장 대비 유지 (컬러/인텐시티 등)
    return '<' + 'f' * (cloud_msg.point_step // 4)


def xyz_array_to_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """(N,3) ndarray -> PointCloud2 (디버그/시각화용)"""
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = True
    pts32 = points.astype(np.float32)
    msg.data = pts32.tobytes()
    return msg
