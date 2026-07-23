"""ROS message builders for image and point-cloud visualization topics."""

import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField


XYZRGB_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
]
CLASS_ID_FIELD = PointField(
    name="class_id", offset=16, datatype=PointField.UINT16, count=1
)


def _packed_rgb(colors):
    colors = colors.astype(np.uint32, copy=False)
    if not len(colors):
        return np.empty((0,), dtype=np.uint32)
    return (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]


def create_pointcloud2(header, points, colors, class_ids=None):
    """Build an XYZRGB cloud, optionally including a uint16 class ID field."""
    points = points.astype(np.float32, copy=False)
    packed_rgb = _packed_rgb(colors)
    include_class_ids = class_ids is not None

    if include_class_ids:
        dtype = np.dtype({
            "names": ["x", "y", "z", "rgb", "class_id"],
            "formats": ["<f4", "<f4", "<f4", "<u4", "<u2"],
            "offsets": [0, 4, 8, 12, 16],
            "itemsize": 18,
        })
    else:
        dtype = np.dtype({
            "names": ["x", "y", "z", "rgb"],
            "formats": ["<f4", "<f4", "<f4", "<u4"],
            "offsets": [0, 4, 8, 12],
            "itemsize": 16,
        })

    cloud_array = np.empty(len(points), dtype=dtype)
    if len(points):
        cloud_array["x"] = points[:, 0]
        cloud_array["y"] = points[:, 1]
        cloud_array["z"] = points[:, 2]
        cloud_array["rgb"] = packed_rgb
        if include_class_ids:
            cloud_array["class_id"] = class_ids.astype(np.uint16, copy=False)

    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = len(points)
    message.fields = XYZRGB_FIELDS + ([CLASS_ID_FIELD] if include_class_ids else [])
    message.is_bigendian = False
    message.point_step = dtype.itemsize
    message.row_step = dtype.itemsize * len(points)
    message.data = cloud_array.tobytes()
    message.is_dense = bool(np.isfinite(points).all())
    return message


def create_bgr8_image(header, image):
    """Build a contiguous ``bgr8`` ROS image message."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    message = Image()
    message.header = header
    message.height = image.shape[0]
    message.width = image.shape[1]
    message.encoding = "bgr8"
    message.is_bigendian = False
    message.step = image.shape[1] * 3
    message.data = image.tobytes()
    return message
