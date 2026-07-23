"""ROS 2 RGB-D receiver with YOLO instance segmentation in an isolated process."""

import json
import multiprocessing as mp
import queue
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

from object_detector.ros_message_utils import create_bgr8_image, create_pointcloud2
from object_detector.segmentation_utils import (
    dilate_mask,
    evenly_limit_samples,
    erode_mask,
    project_depth_samples,
    select_representative_sample,
)


NODE_PARAMETERS = {
    "model_path": "",
    "color_topic": "/camera/camera/color/image_raw",
    "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/camera/color/camera_info",
    "points_topic": "/ai/object_points",
    "background_points_topic": "/ai/background_points",
    "objects_topic": "/ai/objects_3d/json",
    "markers_topic": "/ai/objects_3d/markers",
    "annotated_topic": "/ai/detections_3d/image",
    "target_class_topic": "/ai/target_class",
    "confidence": 0.4,
    "iou": 0.7,
    "image_size": 640,
    "device": "cpu",
    "depth_scale": 0.001,
    "min_depth": 0.15,
    "max_depth": 3.0,
    "depth_band": 0.05,
    "mask_erode_px": 3,
    "point_stride": 2,
    "publish_background_points": True,
    "background_exclusion_mode": "all",
    "target_class": "",
    "background_mask_dilate_px": 5,
    "background_point_stride": 4,
    "max_background_points": 20000,
    "min_valid_ratio": 0.3,
    "min_object_points": 20,
    "max_points_per_object": 5000,
    "sync_slop_sec": 0.08,
    "torch_num_threads": 2,
    "width": 640,
    "height": 480,
}
VALID_BACKGROUND_EXCLUSION_MODES = {"all", "class", "none"}
BINARY_PAYLOAD_KEYS = {
    "annotated",
    "points",
    "colors",
    "class_ids",
    "background_points",
    "background_colors",
}


def inference_worker(input_queue, output_queue, frame_busy, config):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(max(1, int(config["torch_num_threads"])))
    yolo = YOLO(config["model_path"])
    color_shm = shared_memory.SharedMemory(name=config["color_shm_name"])
    depth_shm = shared_memory.SharedMemory(name=config["depth_shm_name"])
    color_view = np.ndarray(config["color_shape"], dtype=np.uint8, buffer=color_shm.buf)
    depth_view = np.ndarray(config["depth_shape"], dtype=np.float32, buffer=depth_shm.buf)
    while True:
        item = input_queue.get()
        if item is None:
            color_shm.close()
            depth_shm.close()
            return
        color = color_view.copy()
        depth_m = depth_view.copy()
        with frame_busy.get_lock():
            frame_busy.value = 0
        fx, fy, cx, cy = item["intrinsics"]
        started = time.perf_counter()
        try:
            result = yolo.predict(
                color,
                conf=config["confidence"],
                iou=config["iou"],
                imgsz=config["image_size"],
                device=config["device"],
                verbose=False,
            )[0]
            boxes = (
                result.boxes.xyxy.cpu().numpy()
                if result.boxes is not None and len(result.boxes)
                else np.empty((0, 4))
            )
            masks = (
                result.masks.data.cpu().numpy()
                if result.masks is not None else []
            )

            annotated = result.plot()
            objects = []
            point_parts, color_parts, class_parts = [], [], []
            removal_mask = np.zeros(depth_m.shape, dtype=bool)
            removed_classes = []
            exclusion_mode = str(config["background_exclusion_mode"]).strip().lower()
            target_class = str(item.get("target_class", "")).strip()
            for index, (box, confidence, class_tensor) in enumerate(
                zip(boxes, result.boxes.conf, result.boxes.cls)
            ):
                class_id = int(class_tensor.item())
                class_name = str(result.names[class_id])
                x1, y1, x2, y2 = [int(round(value)) for value in box]
                x1 = max(0, min(color.shape[1] - 1, x1))
                y1 = max(0, min(color.shape[0] - 1, y1))
                x2 = max(1, min(color.shape[1], x2))
                y2 = max(1, min(color.shape[0], y2))
                status = "missing_mask"
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.uint8)
                pixels = np.empty((0, 2), dtype=np.int32)
                median_depth = None
                valid_ratio = 0.0
                mask_pixels = 0
                if index < len(masks):
                    mask = masks[index]
                    if mask.shape != depth_m.shape:
                        mask = cv2.resize(
                            mask, (depth_m.shape[1], depth_m.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    raw_mask = mask > 0
                    remove_this_object = (
                        exclusion_mode == "all"
                        or (
                            exclusion_mode == "class"
                            and target_class
                            and class_name == target_class
                        )
                    )
                    if remove_this_object:
                        removal_mask |= dilate_mask(
                            raw_mask, config["background_mask_dilate_px"]
                        )
                        if class_name not in removed_classes:
                            removed_classes.append(class_name)

                    mask = erode_mask(raw_mask, config["mask_erode_px"])
                    mask_pixels = int(np.count_nonzero(mask))
                    valid = (
                        mask
                        & np.isfinite(depth_m)
                        & (depth_m >= config["min_depth"])
                        & (depth_m <= config["max_depth"])
                    )
                    valid_ratio = (
                        float(np.count_nonzero(valid) / mask_pixels)
                        if mask_pixels else 0.0
                    )
                    if np.any(valid) and valid_ratio >= config["min_valid_ratio"]:
                        median_depth = float(np.median(depth_m[valid]))
                        valid &= np.abs(depth_m - median_depth) <= config["depth_band"]
                        stride = config["point_stride"]
                        if stride > 1:
                            grid = np.zeros_like(valid)
                            grid[::stride, ::stride] = True
                            valid &= grid
                        points, pixels = project_depth_samples(
                            depth_m, valid, (fx, fy, cx, cy)
                        )
                        xs = pixels[:, 0]
                        ys = pixels[:, 1]
                        colors = color[ys, xs][:, ::-1].copy()
                        max_points = config["max_points_per_object"]
                        points, colors, pixels = evenly_limit_samples(
                            max_points, points, colors, pixels
                        )
                        status = (
                            "success"
                            if len(points) >= config["min_object_points"]
                            else "too_few_object_points"
                        )
                        if status != "success":
                            points = np.empty((0, 3), dtype=np.float32)
                            colors = np.empty((0, 3), dtype=np.uint8)
                            pixels = np.empty((0, 2), dtype=np.int32)
                    else:
                        status = "insufficient_depth"
                    overlay = np.zeros_like(annotated)
                    overlay[mask] = (0, 255, 0)
                    annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.25, 0.0)

                xyz, representative_pixel, representative_depth = (
                    select_representative_sample(points, pixels)
                )
                if representative_pixel is not None:
                    cv2.circle(
                        annotated,
                        tuple(representative_pixel),
                        5,
                        (0, 255, 0),
                        -1,
                    )
                objects.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(confidence.item()), 6),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_pixel": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "representative_pixel_uv": representative_pixel,
                    "representative_depth_m": (
                        None
                        if representative_depth is None
                        else round(representative_depth, 6)
                    ),
                    "median_depth_m": None if median_depth is None else round(median_depth, 6),
                    "valid_depth_ratio": round(valid_ratio, 6),
                    "position_camera_xyz_m": (
                        None
                        if xyz is None
                        else [round(float(value), 6) for value in xyz]
                    ),
                    "mask_pixel_count": mask_pixels,
                    "object_point_count": len(points),
                    "status": status,
                    "segmentation_source": "yolo_seg",
                })
                if len(points):
                    point_parts.append(points)
                    color_parts.append(colors)
                    class_parts.append(np.full(len(points), class_id, dtype=np.uint16))

            background_points = np.empty((0, 3), dtype=np.float32)
            background_colors = np.empty((0, 3), dtype=np.uint8)
            if config["publish_background_points"]:
                background_valid = (
                    ~removal_mask
                    & np.isfinite(depth_m)
                    & (depth_m >= config["min_depth"])
                    & (depth_m <= config["max_depth"])
                )
                background_stride = int(config["background_point_stride"])
                if background_stride > 1:
                    sampling_grid = np.zeros_like(background_valid)
                    sampling_grid[::background_stride, ::background_stride] = True
                    background_valid &= sampling_grid
                background_points, background_pixels = project_depth_samples(
                    depth_m, background_valid, (fx, fy, cx, cy)
                )
                xs = background_pixels[:, 0]
                ys = background_pixels[:, 1]
                background_colors = color[ys, xs][:, ::-1].copy()
                max_background_points = int(config["max_background_points"])
                background_points, background_colors = evenly_limit_samples(
                    max_background_points, background_points, background_colors
                )

            payload = {
                "stamp": item["stamp"],
                "frame_id": item["frame_id"],
                "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "objects": objects,
                "annotated": annotated,
                "points": (
                    np.concatenate(point_parts)
                    if point_parts
                    else np.empty((0, 3), dtype=np.float32)
                ),
                "colors": (
                    np.concatenate(color_parts)
                    if color_parts
                    else np.empty((0, 3), dtype=np.uint8)
                ),
                "class_ids": (
                    np.concatenate(class_parts)
                    if class_parts
                    else np.empty((0,), dtype=np.uint16)
                ),
                "background_points": background_points,
                "background_colors": background_colors,
                "background_exclusion_mode": exclusion_mode,
                "target_class": target_class,
                "removed_classes": removed_classes,
                "background_point_count": len(background_points),
            }
        except Exception as error:
            payload = {"error": f"{type(error).__name__}: {error}"}
        try:
            while True:
                output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            output_queue.put_nowait(payload)
        except queue.Full:
            pass


class YoloSegPointCloudMpNode(Node):
    def __init__(self):
        super().__init__("yolo_pointcloud")
        for name, value in NODE_PARAMETERS.items():
            self.declare_parameter(name, value)
        exclusion_mode = str(self.p("background_exclusion_mode")).strip().lower()
        if exclusion_mode not in VALID_BACKGROUND_EXCLUSION_MODES:
            raise ValueError(
                "background_exclusion_mode must be one of: all, class, none"
            )
        for name in ("model_path",):
            path = Path(str(self.p(name))).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{name} does not exist: {path}")

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_info = None
        self.target_class = str(self.p("target_class")).strip()
        self.color_count = 0
        self.depth_count = 0
        self.queued_count = 0
        self.published_count = 0
        self.mp_context = mp.get_context("spawn")
        self.input_queue = self.mp_context.Queue(maxsize=1)
        self.output_queue = self.mp_context.Queue(maxsize=1)
        self.frame_busy = self.mp_context.Value("b", 0)
        color_shape = (int(self.p("height")), int(self.p("width")), 3)
        depth_shape = (int(self.p("height")), int(self.p("width")))
        self.color_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(color_shape)) * np.dtype(np.uint8).itemsize
        )
        self.depth_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(depth_shape)) * np.dtype(np.float32).itemsize
        )
        self.color_buffer = np.ndarray(color_shape, dtype=np.uint8, buffer=self.color_shm.buf)
        self.depth_buffer = np.ndarray(depth_shape, dtype=np.float32, buffer=self.depth_shm.buf)
        config = {name: self.p(name) for name in NODE_PARAMETERS}
        config["model_path"] = str(Path(config["model_path"]).expanduser())
        config["color_shape"] = color_shape
        config["depth_shape"] = depth_shape
        config["color_shm_name"] = self.color_shm.name
        config["depth_shm_name"] = self.depth_shm.name
        self.worker = self.mp_context.Process(
            target=inference_worker,
            args=(self.input_queue, self.output_queue, self.frame_busy, config),
            daemon=True,
        )
        self.worker.start()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.color_group = MutuallyExclusiveCallbackGroup()
        self.depth_group = MutuallyExclusiveCallbackGroup()
        self.info_group = MutuallyExclusiveCallbackGroup()
        self.color_sub = self.create_subscription(
            Image, str(self.p("color_topic")), self.color_callback, qos,
            callback_group=self.color_group,
        )
        self.depth_sub = self.create_subscription(
            Image, str(self.p("depth_topic")), self.depth_callback, qos,
            callback_group=self.depth_group,
        )
        self.info_sub = self.create_subscription(
            CameraInfo, str(self.p("camera_info_topic")), self.info_callback, qos,
            callback_group=self.info_group,
        )
        self.target_class_sub = None
        if exclusion_mode == "class":
            self.target_class_sub = self.create_subscription(
                String,
                str(self.p("target_class_topic")),
                self.target_class_callback,
                10,
            )
        visualization_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.points_pub = self.create_publisher(
            PointCloud2, str(self.p("points_topic")), visualization_qos
        )
        self.background_points_pub = self.create_publisher(
            PointCloud2, str(self.p("background_points_topic")), qos
        )
        self.objects_pub = self.create_publisher(String, str(self.p("objects_topic")), 10)
        self.markers_pub = self.create_publisher(MarkerArray, str(self.p("markers_topic")), 10)
        self.annotated_pub = self.create_publisher(
            Image, str(self.p("annotated_topic")), visualization_qos
        )
        self.result_timer = self.create_timer(0.05, self.publish_latest_result)
        self.status_timer = self.create_timer(5.0, self.log_status)
        self.get_logger().info(
            f"Ready multiprocessing worker pid={self.worker.pid} "
            f"background_mode={exclusion_mode} target_class={self.target_class or '<none>'}"
        )

    def p(self, name):
        return self.get_parameter(name).value

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def info_callback(self, message):
        if self.latest_info is None:
            self.latest_info = message

    def target_class_callback(self, message):
        self.target_class = message.data.strip()
        self.get_logger().info(
            f"Background exclusion target changed to: {self.target_class or '<none>'}"
        )

    def depth_callback(self, message):
        self.depth_count += 1
        self.latest_depth = message

    def color_callback(self, message):
        self.color_count += 1
        if self.latest_depth is None or self.latest_info is None:
            return
        with self.frame_busy.get_lock():
            if self.frame_busy.value:
                return
        delta = abs(
            self.stamp_seconds(message.header.stamp)
            - self.stamp_seconds(self.latest_depth.header.stamp)
        )
        if delta > float(self.p("sync_slop_sec")):
            return
        try:
            color = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth, desired_encoding="passthrough"
            ).astype(np.float32)
        except CvBridgeError as error:
            self.get_logger().error(f"Image conversion failed: {error}")
            return
        if self.latest_depth.encoding in ("16UC1", "mono16"):
            depth *= float(self.p("depth_scale"))
        if color.shape != self.color_buffer.shape or depth.shape != self.depth_buffer.shape:
            self.get_logger().error(
                f"Unexpected RGB-D shape: {color.shape}/{depth.shape}"
            )
            return
        info = self.latest_info
        self.color_buffer[:] = color
        self.depth_buffer[:] = depth
        item = {
            "intrinsics": (info.k[0], info.k[4], info.k[2], info.k[5]),
            "stamp": {
                "sec": message.header.stamp.sec,
                "nanosec": message.header.stamp.nanosec,
            },
            "frame_id": message.header.frame_id,
            "target_class": self.target_class,
        }
        try:
            with self.frame_busy.get_lock():
                self.frame_busy.value = 1
            self.input_queue.put_nowait(item)
            self.queued_count += 1
        except queue.Full:
            with self.frame_busy.get_lock():
                self.frame_busy.value = 0

    def publish_latest_result(self):
        try:
            payload = self.output_queue.get_nowait()
        except queue.Empty:
            return
        if "error" in payload:
            self.get_logger().error(f"Inference process failed: {payload['error']}")
            return
        header = Header()
        header.stamp.sec = payload["stamp"]["sec"]
        header.stamp.nanosec = payload["stamp"]["nanosec"]
        header.frame_id = payload["frame_id"]
        self.points_pub.publish(
            create_pointcloud2(
                header,
                payload["points"],
                payload["colors"],
                payload["class_ids"],
            )
        )
        self.background_points_pub.publish(
            create_pointcloud2(
                header,
                payload["background_points"],
                payload["background_colors"],
            )
        )

        objects_payload = {
            key: value
            for key, value in payload.items()
            if key not in BINARY_PAYLOAD_KEYS
        }
        self.objects_pub.publish(
            String(data=json.dumps(objects_payload, ensure_ascii=False))
        )
        self.annotated_pub.publish(create_bgr8_image(header, payload["annotated"]))
        self.publish_markers(header, payload["objects"])
        self.published_count += 1

    def publish_markers(self, header, objects):
        array = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        marker_id = 0
        for obj in objects:
            xyz = obj["position_camera_xyz_m"]
            if xyz is None:
                continue
            array.markers.append(self.create_center_marker(header, marker_id, xyz))
            marker_id += 1
            array.markers.append(
                self.create_label_marker(header, marker_id, xyz, obj)
            )
            marker_id += 1
        self.markers_pub.publish(array)

    @staticmethod
    def create_center_marker(header, marker_id, xyz):
        marker = Marker()
        marker.header = header
        marker.ns = "object_centers"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = xyz
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.035
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 1.0
        return marker

    @staticmethod
    def create_label_marker(header, marker_id, xyz, obj):
        marker = Marker()
        marker.header = header
        marker.ns = "object_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = xyz
        marker.pose.position.y -= 0.025
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.022
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0

        pixel = obj.get("representative_pixel_uv")
        depth = obj.get("representative_depth_m")
        if pixel is not None and depth is not None:
            marker.text = (
                f"{obj['class_name']} ({pixel[0]},{pixel[1]}) {depth:.3f}m"
            )
        else:
            marker.text = f"{obj['class_name']} {xyz[2]:.3f}m"
        return marker

    def log_status(self):
        self.get_logger().info(
            f"received color/depth={self.color_count}/{self.depth_count} "
            f"queued={self.queued_count} published={self.published_count} "
            f"worker_alive={self.worker.is_alive()}"
        )

    def destroy_node(self):
        if hasattr(self, "worker"):
            try:
                self.input_queue.put_nowait(None)
            except queue.Full:
                pass
            self.worker.join(timeout=3.0)
            if self.worker.is_alive():
                self.worker.terminate()
        for memory in (getattr(self, "color_shm", None), getattr(self, "depth_shm", None)):
            if memory is not None:
                memory.close()
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = YoloSegPointCloudMpNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if executor is not None:
                executor.remove_node(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
