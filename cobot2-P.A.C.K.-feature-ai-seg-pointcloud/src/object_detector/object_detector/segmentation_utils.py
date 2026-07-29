"""Array helpers shared by the segmentation point-cloud pipeline."""

import cv2
import numpy as np


def erode_mask(mask, pixels):
    """Return a boolean mask eroded by ``pixels`` around its boundary."""
    binary = mask > 0
    if pixels <= 0:
        return binary
    size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(binary.astype(np.uint8), kernel).astype(bool)


def dilate_mask(mask, pixels):
    """Expand a mask so object boundaries are excluded from the scene cloud."""
    binary = mask > 0
    if pixels <= 0:
        return binary
    size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)


def select_representative_sample(points, pixels):
    """Return the measured sample nearest the coordinate-wise point median."""
    if not len(points):
        return None, None, None
    median_xyz = np.median(points, axis=0)
    nearest = int(np.argmin(np.sum((points - median_xyz) ** 2, axis=1)))
    xyz = points[nearest]
    pixel_uv = [int(pixels[nearest, 0]), int(pixels[nearest, 1])]
    return xyz, pixel_uv, float(xyz[2])


def project_depth_samples(depth_m, valid_mask, intrinsics):
    """Project valid depth pixels into camera-frame XYZ coordinates."""
    fx, fy, cx, cy = intrinsics
    ys, xs = np.nonzero(valid_mask)
    zs = depth_m[ys, xs].astype(np.float32)
    points = np.column_stack(
        ((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs)
    ).astype(np.float32)
    pixels = np.column_stack((xs, ys)).astype(np.int32)
    return points, pixels


def evenly_limit_samples(limit, *arrays):
    """Apply the same evenly spaced sample indices to aligned arrays."""
    if not arrays or limit <= 0 or len(arrays[0]) <= limit:
        return arrays
    selected = np.linspace(0, len(arrays[0]) - 1, limit, dtype=np.int64)
    return tuple(array[selected] for array in arrays)
