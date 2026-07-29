#!/usr/bin/env python3
"""Initial scene cache -> PCA grasp targets -> FIFO execution coordination."""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from cobot2_interfaces.msg import GraspTarget, PlannedGrasp

OBJECT_CLOUD_TOPIC = "/ai/object_points_base"
OBJECTS_JSON_TOPIC = "/ai/objects_3d/base_json"
VOICE_TOPIC = "/voice/command"
TARGET_TOPIC = "/grasp/target"
PLANNED_TOPIC = "/moveit_grasp/planned"
EXECUTION_TOPIC = "/execution_result"


@dataclass
class Track:
    track_id: int
    class_name: str
    position: np.ndarray
    last_seen: float
    seen_count: int = 1


@dataclass
class CachedTarget:
    track_id: int
    class_name: str
    json_reference: np.ndarray
    fused: dict


def cluster_points(points: np.ndarray, eps_m: float, min_points: int) -> list[np.ndarray]:
    """Radius-connected components for object-only point clouds."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return []
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < min_points:
        return []

    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []

    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        indices: list[int] = []
        while stack:
            current = stack.pop()
            indices.append(current)
            for neighbor in tree.query_ball_point(points[current], r=eps_m):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(indices) >= min_points:
            clusters.append(points[indices])
    return clusters


class Cobot2GraspNode(Node):
    def __init__(self) -> None:
        super().__init__("cobot2_grasp")
        self._group = ReentrantCallbackGroup()
        self._lock = threading.RLock()

        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self._consumed: set[int] = set()
        self._queue: deque[str] = deque()

        # Initial scan cache. Once every queued item has a cached target,
        # the scene is frozen and no re-scan is required between objects.
        self._track_samples: dict[int, list[dict]] = {}
        self._target_cache: dict[int, CachedTarget] = {}
        self._scene_frozen = False
        self._last_wait_log = 0.0

        self._active_class = ""
        self._active_track_id: Optional[int] = None
        self._active_target_id: Optional[int] = None
        self._next_target_id = 1

        defaults = {
            "expected_frame": "base_link",
            "cluster_eps_mm": 25.0,
            "cluster_min_points": 15,
            "position_match_max_distance_m": 0.150,
            "detection_merge_distance_m": 0.030,
            "detection_max_age_sec": 3.0,
            "max_tracks_per_class": 20,
            "percentile_low": 2.5,
            "percentile_high": 97.5,
            "min_pca_points": 10,
            "min_aspect_ratio": 1.15,
            "fusion_required_samples": 6,
            "fusion_max_samples": 12,
            "center_std_max_m": 0.006,
            "axis_spread_max_deg": 10.0,
            # Final TOP target = observed top_z + this offset.
            "grasp_z_offset_m": 0.010,
            "target_offset_x_m": 0.0,
            "target_offset_y_m": 0.0,
            "target_offset_z_m": 0.0,
            "requeue_on_failure": False,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

        value = lambda name: self.get_parameter(name).value
        self.expected_frame = str(value("expected_frame")).strip().lstrip("/")
        self.cluster_eps_m = float(value("cluster_eps_mm")) / 1000.0
        self.cluster_min_points = int(value("cluster_min_points"))
        self.match_max_m = float(value("position_match_max_distance_m"))
        self.merge_m = float(value("detection_merge_distance_m"))
        self.track_max_age = float(value("detection_max_age_sec"))
        self.max_tracks = int(value("max_tracks_per_class"))
        self.p_low = float(value("percentile_low"))
        self.p_high = float(value("percentile_high"))
        self.min_pca_points = int(value("min_pca_points"))
        self.min_aspect = float(value("min_aspect_ratio"))
        self.required_samples = int(value("fusion_required_samples"))
        self.max_samples = int(value("fusion_max_samples"))
        self.center_std_max = float(value("center_std_max_m"))
        self.axis_spread_max = float(value("axis_spread_max_deg"))
        self.grasp_z_offset = float(value("grasp_z_offset_m"))
        self.target_offset = np.asarray(
            [
                float(value("target_offset_x_m")),
                float(value("target_offset_y_m")),
                float(value("target_offset_z_m")),
            ],
            dtype=np.float64,
        )
        self.requeue_on_failure = bool(value("requeue_on_failure"))
        self._validate_parameters()

        self.create_subscription(
            PointCloud2,
            OBJECT_CLOUD_TOPIC,
            self._on_cloud,
            qos_profile_sensor_data,
            callback_group=self._group,
        )
        self.create_subscription(
            String, OBJECTS_JSON_TOPIC, self._on_json, 10,
            callback_group=self._group,
        )
        self.create_subscription(
            String, VOICE_TOPIC, self._on_voice, 10,
            callback_group=self._group,
        )
        self.create_subscription(
            PlannedGrasp, PLANNED_TOPIC, self._on_planned, 10,
            callback_group=self._group,
        )
        self.create_subscription(
            String, EXECUTION_TOPIC, self._on_execution, 10,
            callback_group=self._group,
        )
        self._target_pub = self.create_publisher(GraspTarget, TARGET_TOPIC, 10)
        self.create_timer(0.5, self._housekeeping, callback_group=self._group)

        self.get_logger().info(
            "cobot2_grasp 시작 | initial-scene cache | "
            f"cloud={OBJECT_CLOUD_TOPIC} | json={OBJECTS_JSON_TOPIC} | "
            f"voice={VOICE_TOPIC} | target={TARGET_TOPIC}"
        )

    def _validate_parameters(self) -> None:
        if not 0.0 <= self.p_low < self.p_high <= 100.0:
            raise ValueError("percentile 범위가 잘못되었습니다")
        if self.cluster_eps_m <= 0.0 or self.cluster_min_points < 1:
            raise ValueError("cluster 파라미터가 잘못되었습니다")
        if self.required_samples < 1 or self.max_samples < self.required_samples:
            raise ValueError("fusion sample 파라미터가 잘못되었습니다")
        if not math.isfinite(self.grasp_z_offset) or abs(self.grasp_z_offset) > 0.050:
            raise ValueError("grasp_z_offset_m은 -50~50mm 범위여야 합니다")
        if not np.isfinite(self.target_offset).all():
            raise ValueError("target offset이 유효하지 않습니다")

    # ------------------------------------------------------------------
    # Voice queue and JSON tracks
    # ------------------------------------------------------------------
    def _on_voice(self, msg: String) -> None:
        items = self._parse_items(msg.data)
        if not items:
            self.get_logger().warning(f"voice 물품 파싱 실패: {msg.data!r}")
            return

        with self._lock:
            self._queue.extend(items)
            pending = list(self._queue)
            frozen = self._scene_frozen

        self.get_logger().warning(
            f"voice 큐 추가={items} | 대기={pending} | scene_frozen={frozen}"
        )
        self._try_activate_next()

    @staticmethod
    def _parse_items(raw: str) -> list[str]:
        text = raw.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = text

        values: Iterable[object]
        if isinstance(decoded, list):
            values = decoded
        elif isinstance(decoded, dict):
            candidate = (
                decoded.get("items") or decoded.get("classes")
                or decoded.get("objects") or decoded.get("command")
            )
            values = (
                candidate
                if isinstance(candidate, list)
                else str(candidate or "").replace(",", " ").split()
            )
        else:
            values = str(decoded).replace(",", " ").split()
        return [str(item).strip() for item in values if str(item).strip()]

    def _on_json(self, msg: String) -> None:
        with self._lock:
            if self._scene_frozen:
                return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"base_json 파싱 실패: {exc}")
            return

        detections = self._extract_detections(payload)
        if not detections:
            return

        now = time.monotonic()
        with self._lock:
            for class_name, position in detections:
                self._update_track_locked(class_name, position, now)
            self._prune_tracks_locked(now)
            summary = Counter(track.class_name for track in self._tracks.values())
        self.get_logger().info(f"base_json 객체 갱신: {dict(sorted(summary.items()))}")

    def _extract_detections(self, payload) -> list[tuple[str, np.ndarray]]:
        if isinstance(payload, list):
            objects, frame, unit = payload, self.expected_frame, "m"
        elif isinstance(payload, dict):
            objects = payload.get("objects") or payload.get("detections") or payload.get("items")
            frame = str(payload.get("frame_id", self.expected_frame)).strip().lstrip("/")
            unit = str(payload.get("position_unit", "m") or "m").strip().lower()
        else:
            return []

        if frame != self.expected_frame or unit != "m" or not isinstance(objects, list):
            self.get_logger().warning(
                f"base_json 형식 불일치 | frame={frame!r} | unit={unit!r}"
            )
            return []

        output: list[tuple[str, np.ndarray]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            class_name = str(obj.get("class_name", "")).strip()
            if not class_name or obj.get("frame_transform_status") not in (None, "", "success"):
                continue
            xyz = (
                obj.get("position_base_xyz_m") or obj.get("center_base_xyz_m")
                or obj.get("position") or obj.get("center")
            )
            if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
                continue
            try:
                position = np.asarray(xyz[:3], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if np.isfinite(position).all():
                output.append((class_name, position))
        return output

    def _update_track_locked(self, class_name: str, position: np.ndarray, now: float) -> None:
        candidates = [
            track for track in self._tracks.values()
            if track.class_name == class_name and track.track_id not in self._consumed
        ]
        nearest = min(
            candidates,
            key=lambda track: float(np.linalg.norm(track.position - position)),
            default=None,
        )
        distance = math.inf if nearest is None else float(np.linalg.norm(nearest.position - position))
        if nearest is not None and distance <= self.merge_m:
            alpha = 1.0 / min(nearest.seen_count + 1, 8)
            nearest.position = (1.0 - alpha) * nearest.position + alpha * position
            nearest.last_seen = now
            nearest.seen_count += 1
            return

        track = Track(self._next_track_id, class_name, position.copy(), now)
        self._tracks[track.track_id] = track
        self._next_track_id += 1

        same_class = sorted(
            (item for item in self._tracks.values() if item.class_name == class_name),
            key=lambda item: item.last_seen,
            reverse=True,
        )
        for stale in same_class[self.max_tracks:]:
            if stale.track_id not in self._target_cache:
                self._tracks.pop(stale.track_id, None)
                self._track_samples.pop(stale.track_id, None)

    def _prune_tracks_locked(self, now: float) -> None:
        for track_id, track in list(self._tracks.items()):
            keep = (
                now - track.last_seen <= self.track_max_age
                or track_id in self._target_cache
                or track_id == self._active_track_id
            )
            if not keep:
                self._tracks.pop(track_id, None)
                self._track_samples.pop(track_id, None)

    # ------------------------------------------------------------------
    # Initial cloud: match every JSON track to a cluster and cache PCA
    # ------------------------------------------------------------------
    def _on_cloud(self, msg: PointCloud2) -> None:
        frame = msg.header.frame_id.strip().lstrip("/")
        if frame != self.expected_frame:
            self.get_logger().warning(
                f"PointCloud frame 불일치: expected={self.expected_frame}, received={frame}"
            )
            return

        with self._lock:
            if self._scene_frozen:
                return
            tracks = [
                (track.track_id, track.position.copy())
                for track in self._tracks.values()
                if track.track_id not in self._consumed
            ]
        if not tracks:
            return

        points = self._pointcloud_to_numpy(msg)
        clusters = cluster_points(points, self.cluster_eps_m, self.cluster_min_points)
        if not clusters:
            return

        centers = np.asarray([np.mean(cluster, axis=0) for cluster in clusters])
        mapping = self._map_tracks_to_clusters(tracks, centers)
        newly_ready: list[CachedTarget] = []

        for track_id, cluster_index in mapping.items():
            with self._lock:
                track = self._tracks.get(track_id)
                samples = list(self._track_samples.get(track_id, []))
            if track is None:
                continue

            previous_axis = None if not samples else np.asarray(samples[-1]["long_axis"])
            try:
                sample = self._analyze_cluster(clusters[cluster_index], previous_axis)
                samples.append(sample)
                samples = samples[-self.max_samples:]
                fused = self._fuse_samples(samples)
            except Exception as exc:
                self.get_logger().warning(
                    f"[{track.class_name}] track={track_id} PCA 실패: {exc}"
                )
                continue

            ready = (
                fused["sample_count"] >= self.required_samples
                and (fused["stable"] or fused["sample_count"] >= self.max_samples)
            )
            with self._lock:
                self._track_samples[track_id] = samples
                if ready:
                    cached = CachedTarget(
                        track_id=track_id,
                        class_name=track.class_name,
                        json_reference=track.position.copy(),
                        fused=fused,
                    )
                    self._target_cache[track_id] = cached
                    newly_ready.append(cached)

            self.get_logger().info(
                f"[{track.class_name}] 초기 장면 캐시 | track={track_id} | cluster={cluster_index} | "
                f"samples={fused['sample_count']} | center={np.round(fused['center'], 4).tolist()}m | "
                f"width={fused['width_m'] * 1000.0:.1f}mm | "
                f"center_std={fused['center_std_m'] * 1000.0:.1f}mm | "
                f"axis_spread={fused['axis_spread_deg']:.1f}deg | ready={ready}"
            )

        if newly_ready:
            with self._lock:
                summary = Counter(
                    cached.class_name
                    for track_id, cached in self._target_cache.items()
                    if track_id not in self._consumed
                )
            self.get_logger().warning(f"초기 파지 목록 갱신: {dict(sorted(summary.items()))}")

        self._try_activate_next()

    def _map_tracks_to_clusters(
        self,
        tracks: list[tuple[int, np.ndarray]],
        centers: np.ndarray,
    ) -> dict[int, int]:
        pairs = [
            (float(np.linalg.norm(position - center)), track_id, cluster_index)
            for track_id, position in tracks
            for cluster_index, center in enumerate(centers)
            if float(np.linalg.norm(position - center)) <= self.match_max_m
        ]
        pairs.sort(key=lambda item: item[0])
        used_tracks: set[int] = set()
        used_clusters: set[int] = set()
        result: dict[int, int] = {}
        for _, track_id, cluster_index in pairs:
            if track_id in used_tracks or cluster_index in used_clusters:
                continue
            result[track_id] = cluster_index
            used_tracks.add(track_id)
            used_clusters.add(cluster_index)
        return result

    @staticmethod
    def _pointcloud_to_numpy(msg: PointCloud2) -> np.ndarray:
        offsets = {
            field.name: int(field.offset)
            for field in msg.fields
            if field.name in ("x", "y", "z")
        }
        if len(offsets) != 3 or msg.point_step <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        count = min(int(msg.width) * int(msg.height), len(msg.data) // int(msg.point_step))
        endian = ">" if msg.is_bigendian else "<"
        unpack = struct.Struct(endian + "f").unpack_from
        data = bytes(msg.data)
        points = np.empty((count, 3), dtype=np.float64)
        for index in range(count):
            base = index * int(msg.point_step)
            points[index] = [
                unpack(data, base + offsets[axis])[0]
                for axis in ("x", "y", "z")
            ]
        return points[np.all(np.isfinite(points), axis=1)]

    def _analyze_cluster(
        self,
        points: np.ndarray,
        previous_long_axis: Optional[np.ndarray],
    ) -> dict:
        points = np.asarray(points, dtype=np.float64)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < self.min_pca_points:
            raise ValueError(f"PCA point 부족: {len(points)}")

        xy = points[:, :2]
        seed = np.median(xy, axis=0)
        long_axis = self._principal_axis(xy - seed)
        short_axis = np.array([-long_axis[1], long_axis[0]])
        rel = xy - seed
        lp = rel @ long_axis
        sp = rel @ short_axis
        ll, lh = np.percentile(lp, [self.p_low, self.p_high])
        sl, sh = np.percentile(sp, [self.p_low, self.p_high])
        mask = (lp >= ll) & (lp <= lh) & (sp >= sl) & (sp <= sh)
        robust = points[mask] if int(np.count_nonzero(mask)) >= self.min_pca_points else points

        robust_xy = robust[:, :2]
        center_xy = np.mean(robust_xy, axis=0)
        long_axis = self._principal_axis(robust_xy - center_xy)
        if long_axis[0] < 0.0 or (abs(long_axis[0]) < 1.0e-9 and long_axis[1] < 0.0):
            long_axis *= -1.0
        short_axis = np.array([-long_axis[1], long_axis[0]])

        relative = robust_xy - center_xy
        lp = relative @ long_axis
        sp = relative @ short_axis
        lp_low, lp_high = np.percentile(lp, [self.p_low, self.p_high])
        sp_low, sp_high = np.percentile(sp, [self.p_low, self.p_high])
        length = float(lp_high - lp_low)
        width = float(sp_high - sp_low)

        if length / max(width, 1.0e-6) < self.min_aspect:
            long_axis = np.asarray(
                previous_long_axis if previous_long_axis is not None else [1.0, 0.0],
                dtype=np.float64,
            )
            long_axis /= np.linalg.norm(long_axis)
            short_axis = np.array([-long_axis[1], long_axis[0]])
            lp = relative @ long_axis
            sp = relative @ short_axis
            lp_low, lp_high = np.percentile(lp, [self.p_low, self.p_high])
            sp_low, sp_high = np.percentile(sp, [self.p_low, self.p_high])
            length = float(lp_high - lp_low)
            width = float(sp_high - sp_low)

        low = np.percentile(robust, self.p_low, axis=0)
        high = np.percentile(robust, self.p_high, axis=0)
        geometric_xy = (
            center_xy
            + long_axis * (0.5 * (lp_low + lp_high))
            + short_axis * (0.5 * (sp_low + sp_high))
        )
        bottom_z = float(low[2])
        top_z = float(high[2])
        center = np.array(
            [geometric_xy[0], geometric_xy[1], 0.5 * (bottom_z + top_z)],
            dtype=np.float64,
        )
        return {
            "center": center,
            "dimensions": np.maximum(high - low, 1.0e-4),
            "long_axis": long_axis,
            "width_m": width,
            "top_z": top_z,
            "bottom_z": bottom_z,
        }

    @staticmethod
    def _principal_axis(centered_xy: np.ndarray) -> np.ndarray:
        covariance = np.cov(centered_xy.T)
        if not np.isfinite(covariance).all():
            raise ValueError("covariance 비정상")
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            raise ValueError("PCA axis 비정상")
        return axis / norm

    def _fuse_samples(self, samples: list[dict]) -> dict:
        reference = np.asarray(samples[0]["long_axis"])
        axes = []
        for item in samples:
            axis = np.asarray(item["long_axis"])
            axes.append(axis if float(np.dot(axis, reference)) >= 0.0 else -axis)

        long_axis = np.mean(np.asarray(axes), axis=0)
        long_axis /= np.linalg.norm(long_axis)
        if long_axis[0] < 0.0 or (abs(long_axis[0]) < 1.0e-9 and long_axis[1] < 0.0):
            long_axis *= -1.0
        short_axis = np.array([-long_axis[1], long_axis[0]])

        centers = np.asarray([item["center"] for item in samples])
        center_std = float(np.max(np.std(centers, axis=0)))
        spread = max(
            (
                math.degrees(
                    math.acos(abs(float(np.clip(np.dot(axis, long_axis), -1.0, 1.0))))
                )
                for axis in axes
            ),
            default=0.0,
        )
        return {
            "center": np.median(centers, axis=0),
            "dimensions": np.median(
                np.asarray([item["dimensions"] for item in samples]), axis=0
            ),
            "long_axis": long_axis,
            "short_axis": short_axis,
            "width_m": float(np.median([item["width_m"] for item in samples])),
            "top_z": float(np.median([item["top_z"] for item in samples])),
            "bottom_z": float(np.median([item["bottom_z"] for item in samples])),
            "center_std_m": center_std,
            "axis_spread_deg": float(spread),
            "sample_count": len(samples),
            "stable": center_std <= self.center_std_max and spread <= self.axis_spread_max,
        }

    # ------------------------------------------------------------------
    # Frozen scene FIFO execution
    # ------------------------------------------------------------------
    def _missing_cached_items_locked(self) -> dict[str, int]:
        required = Counter(self._queue)
        available = Counter(
            cached.class_name
            for track_id, cached in self._target_cache.items()
            if track_id not in self._consumed
        )
        return {
            class_name: count - available[class_name]
            for class_name, count in required.items()
            if available[class_name] < count
        }

    def _try_activate_next(self) -> None:
        cached: Optional[CachedTarget] = None
        with self._lock:
            if self._active_class or not self._queue:
                return

            if not self._scene_frozen:
                missing = self._missing_cached_items_locked()
                if missing:
                    return
                self._scene_frozen = True
                snapshot = [
                    {
                        "class": item.class_name,
                        "track": item.track_id,
                        "center": np.round(item.fused["center"], 4).tolist(),
                    }
                    for item in sorted(self._target_cache.values(), key=lambda value: value.track_id)
                    if item.track_id not in self._consumed
                ]
                self.get_logger().warning(
                    f"초기 장면 고정 완료 | cached_targets={snapshot}"
                )

            class_name = self._queue[0]
            candidates = [
                item for track_id, item in self._target_cache.items()
                if item.class_name == class_name and track_id not in self._consumed
            ]
            if not candidates:
                return

            cached = min(candidates, key=lambda item: item.track_id)
            self._queue.popleft()
            self._active_class = class_name
            self._active_track_id = cached.track_id
            self._active_target_id = None
            remaining = list(self._queue)

        self.get_logger().warning(
            f"[{cached.class_name}] 캐시 목표 활성화 | track={cached.track_id} | "
            f"center={np.round(cached.fused['center'], 4).tolist()}m | 남은 큐={remaining}"
        )
        self._publish_cached_target(cached)

    def _publish_cached_target(self, cached: CachedTarget) -> None:
        with self._lock:
            if (
                self._active_track_id != cached.track_id
                or self._active_target_id is not None
            ):
                return
            target_id = self._next_target_id
            self._next_target_id += 1
            self._active_target_id = target_id

        fused = cached.fused
        raw_center = np.asarray(fused["center"], dtype=np.float64)
        center = raw_center + self.target_offset
        top_z = float(fused["top_z"] + self.grasp_z_offset)
        size = np.asarray(fused["dimensions"], dtype=np.float64)
        short_axis = np.asarray(fused["short_axis"], dtype=np.float64)

        msg = GraspTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.expected_frame
        msg.target_id = int(target_id)
        msg.class_name = cached.class_name
        msg.center.x, msg.center.y, msg.center.z = map(float, center)
        msg.dimensions.x, msg.dimensions.y, msg.dimensions.z = map(float, size)
        msg.top_z = top_z
        msg.required_width = float(fused["width_m"])
        msg.closing_axis.x = float(short_axis[0])
        msg.closing_axis.y = float(short_axis[1])
        msg.closing_axis.z = 0.0
        msg.grasp_depth = 0.0
        msg.shelf_level = 1
        self._target_pub.publish(msg)

        self.get_logger().warning(
            f"[{cached.class_name}] 캐시 /grasp/target 발행 | id={target_id} | "
            f"json_reference={cached.json_reference.round(4).tolist()}m | "
            f"raw_cluster_center={raw_center.round(4).tolist()}m | "
            f"target_center={center.round(4).tolist()}m | "
            f"top_z={top_z:.4f} | width={msg.required_width * 1000.0:.1f}mm | "
            f"closing=({msg.closing_axis.x:.4f}, {msg.closing_axis.y:.4f}, 0)"
        )

    def _on_planned(self, msg: PlannedGrasp) -> None:
        with self._lock:
            if self._active_target_id is None or int(msg.target_id) != self._active_target_id:
                return
            class_name = self._active_class
        if msg.success:
            self.get_logger().info(f"[{class_name}] MoveIt 계획 성공 — 실행 결과 대기")
            return
        reason = str(msg.failure_reason)
        self.get_logger().error(f"[{class_name}] MoveIt 계획 실패: {reason}")
        self._finish_active(False, f"MOVEIT_PLAN_FAILED:{reason}")

    def _on_execution(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            target_id = int(payload.get("target_id", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        with self._lock:
            if self._active_target_id is None or target_id != self._active_target_id:
                return
        self._finish_active(
            bool(payload.get("success", False)),
            str(payload.get("reason", "")),
        )

    def _finish_active(self, success: bool, reason: str) -> None:
        with self._lock:
            class_name = self._active_class
            track_id = self._active_track_id
            if success and track_id is not None:
                self._consumed.add(track_id)
            elif not success and self.requeue_on_failure and class_name:
                self._queue.appendleft(class_name)

            self._active_class = ""
            self._active_track_id = None
            self._active_target_id = None

        self.get_logger().warning(
            f"[{class_name}] 작업 종료 | success={success} | reason={reason}"
        )
        # move.py publishes success only after bag release. The next cached target
        # is therefore sent immediately from the bag pose without re-scanning.
        self._try_activate_next()

    def _housekeeping(self) -> None:
        with self._lock:
            if not self._scene_frozen:
                self._prune_tracks_locked(time.monotonic())
                missing = self._missing_cached_items_locked() if self._queue else {}
            else:
                missing = {}
        self._try_activate_next()

        now = time.monotonic()
        if missing and now - self._last_wait_log >= 2.0:
            self._last_wait_log = now
            self.get_logger().warning(
                f"초기 장면 캐시 대기 | 아직 필요한 물품={missing}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Cobot2GraspNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()