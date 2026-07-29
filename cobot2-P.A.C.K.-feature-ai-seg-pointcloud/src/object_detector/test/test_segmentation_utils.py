"""Unit tests for the active YOLO segmentation point-cloud helpers."""

import numpy as np
from std_msgs.msg import Header

from object_detector.ros_message_utils import create_bgr8_image, create_pointcloud2
from object_detector.segmentation_utils import (
    dilate_mask,
    evenly_limit_samples,
    erode_mask,
    project_depth_samples,
    select_representative_sample,
)


def test_mask_morphology_returns_boolean_masks():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    eroded = erode_mask(mask, 1)
    dilated = dilate_mask(mask, 1)

    assert eroded.dtype == np.bool_
    assert dilated.dtype == np.bool_
    assert np.count_nonzero(eroded) < np.count_nonzero(mask)
    assert np.count_nonzero(dilated) > np.count_nonzero(mask)


def test_project_depth_samples_uses_camera_intrinsics():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    valid = np.array([[False, True], [False, False]])

    points, pixels = project_depth_samples(depth, valid, (2.0, 2.0, 0.0, 0.0))

    np.testing.assert_allclose(points, [[1.0, 0.0, 2.0]])
    np.testing.assert_array_equal(pixels, [[1, 0]])


def test_sample_limit_preserves_alignment_and_endpoints():
    first = np.arange(10)
    second = first + 100

    limited_first, limited_second = evenly_limit_samples(4, first, second)

    np.testing.assert_array_equal(limited_first, [0, 3, 6, 9])
    np.testing.assert_array_equal(limited_second, [100, 103, 106, 109])


def test_representative_sample_is_an_observed_depth_point():
    points = np.array(
        [[0.0, 0.0, 1.0], [0.2, 0.2, 1.2], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    pixels = np.array([[10, 20], [11, 21], [12, 22]], dtype=np.int32)

    xyz, pixel, depth = select_representative_sample(points, pixels)

    np.testing.assert_array_equal(xyz, points[1])
    assert pixel == [11, 21]
    assert np.isclose(depth, 1.2)


def test_pointcloud_message_layout_with_class_ids():
    header = Header(frame_id="camera_color_optical_frame")
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    colors = np.array([[255, 128, 0]], dtype=np.uint8)
    class_ids = np.array([7], dtype=np.uint16)

    message = create_pointcloud2(header, points, colors, class_ids)

    assert message.width == 1
    assert message.point_step == 18
    assert [field.name for field in message.fields] == [
        "x", "y", "z", "rgb", "class_id"
    ]
    assert len(message.data) == 18
    unpacked = np.frombuffer(
        message.data,
        dtype=np.dtype({
            "names": ["x", "y", "z", "rgb", "class_id"],
            "formats": ["<f4", "<f4", "<f4", "<u4", "<u2"],
            "offsets": [0, 4, 8, 12, 16],
            "itemsize": 18,
        }),
    )
    assert unpacked["rgb"][0] == 0xFF8000
    assert unpacked["class_id"][0] == 7


def test_background_cloud_and_image_layouts():
    header = Header(frame_id="camera_color_optical_frame")
    empty_cloud = create_pointcloud2(
        header,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
    )
    image = create_bgr8_image(header, np.zeros((2, 3, 3), dtype=np.uint8))

    assert empty_cloud.width == 0
    assert empty_cloud.point_step == 16
    assert [field.name for field in empty_cloud.fields] == ["x", "y", "z", "rgb"]
    assert image.encoding == "bgr8"
    assert image.height == 2
    assert image.width == 3
    assert image.step == 9
