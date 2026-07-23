"""ROS 2 YOLO + aligned depth object point-cloud publisher."""

import json
import time
import warnings
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

warnings.filterwarnings("ignore", message=r"Unable to import Axes3D.*")


def clamp_bbox(box, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )


def central_roi(bbox, scale):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    return tuple(int(round(value)) for value in (cx - hw, cy - hh, cx + hw, cy + hh))


def pack_rgb(rgb):
    r, g, b = (int(value) for value in rgb)
    return (r << 16) | (g << 8) | b


class TopicReceiver(Node):
    """One-topic node so large RGB and depth streams cannot starve each other."""

    def __init__(self, name, message_type, topic, callback, qos):
        super().__init__(name)
        self.subscription = self.create_subscription(
            message_type, topic, callback, qos
        )


class YoloPointCloudNode(Node):
    def __init__(self):
        super().__init__("yolo_pointcloud")
        defaults = {
            "model_path": "",
            "sam_model_path": "",
            "use_sam": True,
            "color_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "points_topic": "/ai/object_points",
            "objects_topic": "/ai/objects_3d/json",
            "markers_topic": "/ai/objects_3d/markers",
            "annotated_topic": "/ai/detections_3d/image",
            "confidence": 0.4,
            "iou": 0.7,
            "image_size": 640,
            "device": "cpu",
            "process_every_n": 1,
            "depth_scale": 0.001,
            "roi_scale": 0.4,
            "min_depth": 0.15,
            "max_depth": 3.0,
            "point_stride": 2,
            "depth_band": 0.05,
            "mask_erode_px": 3,
            "voxel_size": 0.005,
            "ransac_enabled": True,
            "ransac_distance_threshold": 0.008,
            "ransac_iterations": 100,
            "ransac_max_normal_angle_deg": 35.0,
            "ransac_min_plane_ratio": 0.15,
            "ransac_max_plane_ratio": 0.70,
            "dbscan_eps": 0.02,
            "dbscan_min_points": 20,
            "max_cluster_points": 5000,
            "min_valid_ratio": 0.3,
            "min_object_points": 20,
            "sync_slop_sec": 0.08,
            "torch_num_threads": 2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        model_path = Path(str(self.p("model_path"))).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")
        import torch
        from ultralytics import SAM, YOLO
        import open3d as o3d

        torch.set_num_threads(max(1, int(self.p("torch_num_threads"))))
        self.model = YOLO(str(model_path))
        self.sam = None
        if bool(self.p("use_sam")):
            sam_model_path = Path(str(self.p("sam_model_path"))).expanduser()
            if not sam_model_path.is_file():
                raise FileNotFoundError(
                    f"sam_model_path does not exist: {sam_model_path}"
                )
            self.sam = SAM(str(sam_model_path))
        self.o3d = o3d
        self.bridge = CvBridge()
        self.frame_count = 0
        self.depth_count = 0
        self.info_count = 0
        self.processed_count = 0
        self.latest_info = None
        self.latest_depth = None
        self.latest_color = None
        self.latest_color_count = 0
        self.last_processed_stamp = None
        self.latest_stamp_delta = None
        self.work_event = Event()
        self.stop_event = Event()

        self.points_pub = self.create_publisher(PointCloud2, str(self.p("points_topic")), 10)
        self.objects_pub = self.create_publisher(String, str(self.p("objects_topic")), 10)
        self.markers_pub = self.create_publisher(MarkerArray, str(self.p("markers_topic")), 10)
        self.annotated_pub = self.create_publisher(Image, str(self.p("annotated_topic")), 10)

        self.get_logger().info(
            f"Ready: color={self.p('color_topic')} depth={self.p('depth_topic')} "
            f"points={self.p('points_topic')} use_sam={self.sam is not None}"
        )
        self.status_timer = self.create_timer(5.0, self.log_status)
        self.worker = Thread(
            target=self.worker_loop, name="yolo_sam_worker", daemon=True
        )
        self.worker.start()

    def p(self, name):
        return self.get_parameter(name).value

    def info_callback(self, message):
        if self.latest_info is not None:
            return
        self.info_count += 1
        self.latest_info = message
        self.get_logger().info("CameraInfo captured")

    def depth_callback(self, message):
        self.depth_count += 1
        self.latest_depth = message

    def log_status(self):
        delta = "none" if self.latest_stamp_delta is None else f"{self.latest_stamp_delta:.4f}s"
        self.get_logger().info(
            f"received color/depth/info={self.frame_count}/{self.depth_count}/{self.info_count} "
            f"processed={self.processed_count} latest_stamp_delta={delta}"
        )

    def extract_mask_object(self, box, mask, color, depth_m, info):
        height, width = depth_m.shape
        bbox = clamp_bbox(box, width, height)
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        if mask.shape != depth_m.shape:
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        binary = mask > 0
        erode_px = int(self.p("mask_erode_px"))
        if erode_px > 0:
            size = erode_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            binary = cv2.erode(binary.astype(np.uint8), kernel).astype(bool)
        mask_count = int(np.count_nonzero(binary))
        valid_depth = (
            np.isfinite(depth_m)
            & (depth_m >= float(self.p("min_depth")))
            & (depth_m <= float(self.p("max_depth")))
        )
        valid = binary & valid_depth
        valid_count = int(np.count_nonzero(valid))
        valid_ratio = valid_count / mask_count if mask_count else 0.0
        if not valid_count or valid_ratio < float(self.p("min_valid_ratio")):
            return bbox, center, None, valid_ratio, "insufficient_depth", mask_count, valid_count, False, 0, np.empty((0, 3)), np.empty((0, 3))

        median_depth = float(np.median(depth_m[valid]))
        valid &= np.abs(depth_m - median_depth) <= float(self.p("depth_band"))
        stride = int(self.p("point_stride"))
        if stride > 1:
            sample_grid = np.zeros_like(valid)
            sample_grid[::stride, ::stride] = True
            valid &= sample_grid
        ys, xs = np.nonzero(valid)
        candidate_count = len(xs)
        if candidate_count < int(self.p("min_object_points")):
            return bbox, center, median_depth, valid_ratio, "too_few_candidates", mask_count, candidate_count, False, 0, np.empty((0, 3)), np.empty((0, 3))

        zs = depth_m[ys, xs].astype(np.float64)
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        points = np.column_stack(
            ((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs)
        )
        colors = color[ys, xs][:, ::-1].astype(np.float64) / 255.0
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(points)
        cloud.colors = self.o3d.utility.Vector3dVector(colors)
        voxel = float(self.p("voxel_size"))
        if voxel > 0:
            cloud = cloud.voxel_down_sample(voxel)
        points = np.asarray(cloud.points).copy()
        colors = np.asarray(cloud.colors).copy()
        status = (
            "success"
            if len(points) >= int(self.p("min_object_points"))
            else "too_few_object_points"
        )
        if status != "success":
            points, colors = np.empty((0, 3)), np.empty((0, 3))
        return bbox, center, median_depth, valid_ratio, status, mask_count, candidate_count, False, 0, points, colors

    def extract_object(self, box, color, depth_m, info):
        height, width = depth_m.shape
        bbox = clamp_bbox(box, width, height)
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        rx1, ry1, rx2, ry2 = central_roi(bbox, float(self.p("roi_scale")))
        roi_depth = depth_m[ry1:ry2, rx1:rx2]
        valid_roi = (
            np.isfinite(roi_depth)
            & (roi_depth >= float(self.p("min_depth")))
            & (roi_depth <= float(self.p("max_depth")))
        )
        valid_ratio = float(valid_roi.mean()) if valid_roi.size else 0.0
        if not np.any(valid_roi) or valid_ratio < float(self.p("min_valid_ratio")):
            return bbox, center, None, valid_ratio, "insufficient_depth", 0, 0, False, 0, np.empty((0, 3)), np.empty((0, 3))

        median_depth = float(np.median(roi_depth[valid_roi]))
        stride = int(self.p("point_stride"))
        ys, xs = np.mgrid[y1:y2:stride, x1:x2:stride]
        sampled = depth_m[y1:y2:stride, x1:x2:stride]
        valid = (
            np.isfinite(sampled)
            & (sampled >= float(self.p("min_depth")))
            & (sampled <= float(self.p("max_depth")))
        )
        bbox_count = int(np.count_nonzero(valid))
        valid &= np.abs(sampled - median_depth) <= float(self.p("depth_band"))
        candidate_count = int(np.count_nonzero(valid))
        if candidate_count < int(self.p("min_object_points")):
            return bbox, center, median_depth, valid_ratio, "too_few_candidates", bbox_count, candidate_count, False, 0, np.empty((0, 3)), np.empty((0, 3))

        us = xs[valid].astype(np.float64)
        vs = ys[valid].astype(np.float64)
        zs = sampled[valid].astype(np.float64)
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        points = np.column_stack(((us - cx) * zs / fx, (vs - cy) * zs / fy, zs))
        colors = color[vs.astype(int), us.astype(int)][:, ::-1].astype(np.float64) / 255.0
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(points)
        cloud.colors = self.o3d.utility.Vector3dVector(colors)
        voxel = float(self.p("voxel_size"))
        if voxel > 0:
            cloud = cloud.voxel_down_sample(voxel)
        # Bound both RANSAC and DBSCAN cost. This must happen before RANSAC;
        # otherwise a large planar bbox can block the image callback.
        max_cluster_points = int(self.p("max_cluster_points"))
        if max_cluster_points > 0 and len(cloud.points) > max_cluster_points:
            indices = np.linspace(
                0, len(cloud.points) - 1, max_cluster_points, dtype=np.int64
            )
            cloud = cloud.select_by_index(indices.tolist())
        ransac_applied = False
        plane_point_count = 0
        if bool(self.p("ransac_enabled")) and len(cloud.points) >= 3:
            plane_model, inliers = cloud.segment_plane(
                distance_threshold=float(self.p("ransac_distance_threshold")),
                ransac_n=3,
                num_iterations=int(self.p("ransac_iterations")),
            )
            normal = np.asarray(plane_model[:3], dtype=np.float64)
            normal_norm = float(np.linalg.norm(normal))
            y_alignment = abs(float(normal[1])) / normal_norm if normal_norm else 0.0
            min_y_alignment = float(np.cos(np.deg2rad(
                float(self.p("ransac_max_normal_angle_deg"))
            )))
            plane_ratio = len(inliers) / max(len(cloud.points), 1)
            remaining_count = len(cloud.points) - len(inliers)
            if (
                y_alignment >= min_y_alignment
                and float(self.p("ransac_min_plane_ratio")) <= plane_ratio
                <= float(self.p("ransac_max_plane_ratio"))
                and remaining_count >= int(self.p("min_object_points"))
            ):
                cloud = cloud.select_by_index(inliers, invert=True)
                ransac_applied = True
                plane_point_count = len(inliers)
        labels = np.asarray(cloud.cluster_dbscan(
            eps=float(self.p("dbscan_eps")),
            min_points=int(self.p("dbscan_min_points")),
            print_progress=False,
        ))
        cluster_ids = np.unique(labels[labels >= 0])
        if not len(cluster_ids):
            return bbox, center, median_depth, valid_ratio, "clustering_failed", bbox_count, candidate_count, ransac_applied, plane_point_count, np.empty((0, 3)), np.empty((0, 3))

        cloud_points = np.asarray(cloud.points)
        seed = np.array(((center[0] - cx) * median_depth / fx,
                         (center[1] - cy) * median_depth / fy, median_depth))
        selected_id = min(
            cluster_ids,
            key=lambda label: float(np.linalg.norm(
                np.median(cloud_points[labels == label], axis=0) - seed
            )),
        )
        selected = labels == selected_id
        points = cloud_points[selected].copy()
        colors = np.asarray(cloud.colors)[selected].copy()
        status = "success" if len(points) >= int(self.p("min_object_points")) else "too_few_object_points"
        if status != "success":
            points, colors = np.empty((0, 3)), np.empty((0, 3))
        return bbox, center, median_depth, valid_ratio, status, bbox_count, candidate_count, ransac_applied, plane_point_count, points, colors

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def color_callback(self, message):
        self.frame_count += 1
        self.latest_color_count = self.frame_count
        self.latest_color = message
        self.work_event.set()

    def worker_loop(self):
        while not self.stop_event.is_set():
            if not self.work_event.wait(timeout=0.2):
                continue
            self.work_event.clear()
            try:
                self.process_latest()
            except Exception as error:
                self.get_logger().error(f"Processing worker failed: {error}")

    def destroy_node(self):
        self.stop_event.set()
        self.work_event.set()
        if hasattr(self, "worker"):
            self.worker.join(timeout=3.0)
        return super().destroy_node()

    def process_latest(self):
        color_msg = self.latest_color
        info_msg = self.latest_info
        depth_msg = self.latest_depth
        if color_msg is None or info_msg is None or depth_msg is None:
            return
        stamp_key = (color_msg.header.stamp.sec, color_msg.header.stamp.nanosec)
        if stamp_key == self.last_processed_stamp:
            return
        stamp_delta = abs(
            self.stamp_seconds(color_msg.header.stamp)
            - self.stamp_seconds(depth_msg.header.stamp)
        )
        self.latest_stamp_delta = stamp_delta
        if stamp_delta > float(self.p("sync_slop_sec")):
            return
        self.last_processed_stamp = stamp_key
        if self.latest_color_count % int(self.p("process_every_n")):
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as error:
            self.get_logger().error(f"Image conversion failed: {error}")
            return
        depth_m = np.asarray(depth_raw, dtype=np.float32)
        if depth_msg.encoding in ("16UC1", "mono16"):
            depth_m *= float(self.p("depth_scale"))

        started = time.perf_counter()
        try:
            result = self.model.predict(
                color, conf=float(self.p("confidence")), iou=float(self.p("iou")),
                imgsz=int(self.p("image_size")), device=str(self.p("device")), verbose=False,
            )[0]
        except Exception as error:
            self.get_logger().error(f"YOLO inference failed: {error}")
            return

        masks = []
        if self.sam is not None and result.boxes is not None and len(result.boxes):
            try:
                boxes = result.boxes.xyxy.cpu().numpy().tolist()
                sam_result = self.sam.predict(
                    source=color, bboxes=boxes, device=str(self.p("device")),
                    verbose=False,
                )[0]
                if sam_result.masks is not None:
                    masks = sam_result.masks.data.cpu().numpy()
            except Exception as error:
                self.get_logger().error(f"SAM inference failed: {error}")
                return

        objects, all_points, all_colors, all_class_ids = [], [], [], []
        if result.boxes is not None:
            for object_index, (box, confidence, class_id) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                class_index = int(class_id.item())
                class_name = str(result.names[class_index])
                if self.sam is not None:
                    if object_index >= len(masks):
                        self.get_logger().warning(
                            f"Missing SAM mask: object_index={object_index}"
                        )
                        continue
                    extracted = self.extract_mask_object(
                        box.tolist(), masks[object_index], color, depth_m, info_msg
                    )
                else:
                    extracted = self.extract_object(
                        box.tolist(), color, depth_m, info_msg
                    )
                bbox, center, median_depth, valid_ratio, status, bbox_count, candidate_count, ransac_applied, plane_point_count, points, colors = extracted
                xyz = np.median(points, axis=0) if len(points) else None
                objects.append({
                    "class_id": class_index,
                    "class_name": class_name,
                    "confidence": round(float(confidence.item()), 6),
                    "bbox_xyxy": list(bbox),
                    "center_pixel": list(center),
                    "median_depth_m": round(median_depth, 6) if median_depth is not None else None,
                    "valid_depth_ratio": round(valid_ratio, 6),
                    "position_camera_xyz_m": [round(float(v), 6) for v in xyz] if xyz is not None else None,
                    "bbox_point_count": bbox_count,
                    "candidate_point_count": candidate_count,
                    "ransac_applied": ransac_applied,
                    "ransac_plane_point_count": plane_point_count,
                    "object_point_count": len(points),
                    "status": status,
                    "segmentation_source": "mobile_sam" if self.sam is not None else "bbox_dbscan",
                })
                if len(points):
                    all_points.append(points)
                    all_colors.append(colors)
                    all_class_ids.append(np.full(len(points), class_index, dtype=np.uint16))

        header = color_msg.header
        self.publish_cloud(header, all_points, all_colors, all_class_ids)
        self.publish_markers(header, objects)
        payload = {
            "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec},
            "frame_id": header.frame_id,
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "objects": objects,
        }
        self.objects_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        annotated = result.plot()
        if self.sam is not None:
            for mask in masks:
                if mask.shape != annotated.shape[:2]:
                    mask = cv2.resize(
                        mask, (annotated.shape[1], annotated.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                overlay = np.zeros_like(annotated)
                overlay[mask > 0] = (0, 255, 0)
                annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.25, 0.0)
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            if self.sam is None:
                rx1, ry1, rx2, ry2 = central_roi(
                    (x1, y1, x2, y2), float(self.p("roi_scale"))
                )
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
            status_color = (0, 255, 0) if obj["status"] == "success" else (0, 0, 255)
            status_text = (
                f"{obj['status']} cand={obj['candidate_point_count']} "
                f"plane={obj['ransac_plane_point_count']} obj={obj['object_point_count']}"
            )
            cv2.putText(
                annotated, status_text, (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA,
            )
            if obj["position_camera_xyz_m"] is not None:
                xyz = obj["position_camera_xyz_m"]
                text = f"XYZ {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}m pts={obj['object_point_count']}"
                cv2.putText(annotated, text, (x1, min(y2 + 18, annotated.shape[0] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = header
        self.annotated_pub.publish(annotated_msg)
        self.processed_count += 1

    def publish_cloud(self, header, point_parts, color_parts, class_parts):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="class_id", offset=16, datatype=PointField.UINT16, count=1),
        ]
        rows = []
        if point_parts:
            points = np.concatenate(point_parts)
            colors = np.concatenate(color_parts)
            classes = np.concatenate(class_parts)
            rows = [(float(p[0]), float(p[1]), float(p[2]), pack_rgb(c * 255.0), int(k))
                    for p, c, k in zip(points, colors, classes)]
        self.points_pub.publish(point_cloud2.create_cloud(header, fields, rows))

    def publish_markers(self, header, objects):
        markers = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        marker_id = 0
        for obj in objects:
            xyz = obj["position_camera_xyz_m"]
            if xyz is None:
                continue
            sphere = Marker()
            sphere.header = header
            sphere.ns = "object_centers"
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = xyz
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.035
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.1, 1.0, 0.1, 1.0
            markers.markers.append(sphere)
            marker_id += 1
            text = Marker()
            text.header = header
            text.ns = "object_labels"
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = xyz
            text.pose.position.y -= 0.04
            text.pose.orientation.w = 1.0
            text.scale.z = 0.035
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"{obj['class_name']} {xyz[2]:.3f}m"
            markers.markers.append(text)
            marker_id += 1
        self.markers_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = None
    executors = []
    executor_threads = []
    receivers = []
    try:
        node = YoloPointCloudNode()
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        receiver_specs = [
            (
                "yolo_color_receiver", Image, str(node.p("color_topic")),
                node.color_callback, image_qos,
            ),
            (
                "yolo_depth_receiver", Image, str(node.p("depth_topic")),
                node.depth_callback, image_qos,
            ),
            (
                "yolo_info_receiver", CameraInfo,
                str(node.p("camera_info_topic")), node.info_callback, image_qos,
            ),
        ]
        for spec in receiver_specs:
            receiver = TopicReceiver(*spec)
            executor = SingleThreadedExecutor()
            executor.add_node(receiver)
            thread = Thread(target=executor.spin, daemon=True)
            thread.start()
            receivers.append(receiver)
            executors.append(executor)
            executor_threads.append(thread)
        main_executor = SingleThreadedExecutor()
        main_executor.add_node(node)
        executors.append(main_executor)
        main_executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for executor in executors:
            executor.shutdown(timeout_sec=1.0)
        for thread in executor_threads:
            thread.join(timeout=1.0)
        for receiver in receivers:
            receiver.destroy_node()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
