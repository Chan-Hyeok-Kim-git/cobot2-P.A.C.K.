import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger


class YoloDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_object_detector")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("annotated_topic", "/ai/detections/image")
        self.declare_parameter("detections_topic", "/ai/detections/json")
        self.declare_parameter("confidence", 0.4)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("device", "0")
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("capture_root", "dataset/detection_captures")
        self.declare_parameter("log_period_sec", 5.0)

        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")

        confidence = float(self.get_parameter("confidence").value)
        iou = float(self.get_parameter("iou").value)
        process_every_n = int(self.get_parameter("process_every_n").value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if process_every_n < 1:
            raise ValueError("process_every_n must be at least 1")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics missing. Run: python3 -m pip install --user "
                "ultralytics==8.4.102"
            ) from error

        self.model = YOLO(str(model_path))
        self.bridge = CvBridge()
        self.frame_count = 0
        self.processed_count = 0
        self.started_at = time.perf_counter()
        self.latest_lock = Lock()
        self.latest_raw = None
        self.latest_annotated = None
        self.latest_payload = None

        image_topic = str(self.get_parameter("image_topic").value)
        annotated_topic = str(self.get_parameter("annotated_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        self.image_subscription = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.annotated_publisher = self.create_publisher(Image, annotated_topic, 10)
        self.detections_publisher = self.create_publisher(String, detections_topic, 10)
        self.capture_service = self.create_service(
            Trigger, "~/save_frame", self.save_frame_callback
        )
        self.log_timer = self.create_timer(
            float(self.get_parameter("log_period_sec").value), self.log_status
        )
        self.get_logger().info(
            f"Ready. model={model_path} image_topic={image_topic} "
            f"confidence={confidence}"
        )

    def image_callback(self, message: Image) -> None:
        self.frame_count += 1
        process_every_n = int(self.get_parameter("process_every_n").value)
        if self.frame_count % process_every_n:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().error(f"Image conversion failed: {error}")
            return

        started = time.perf_counter()
        try:
            result = self.model.predict(
                source=frame,
                conf=float(self.get_parameter("confidence").value),
                iou=float(self.get_parameter("iou").value),
                imgsz=int(self.get_parameter("image_size").value),
                device=str(self.get_parameter("device").value),
                verbose=False,
            )[0]
        except Exception as error:
            self.get_logger().error(f"YOLO inference failed: {error}")
            return
        inference_ms = (time.perf_counter() - started) * 1000.0

        detections = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(
                result.boxes.xyxy, result.boxes.conf, result.boxes.cls
            ):
                class_index = int(class_id.item())
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                detections.append(
                    {
                        "class_id": class_index,
                        "class_name": str(result.names[class_index]),
                        "confidence": round(float(confidence.item()), 6),
                        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "center_pixel": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                    }
                )

        payload = {
            "stamp": {
                "sec": message.header.stamp.sec,
                "nanosec": message.header.stamp.nanosec,
            },
            "frame_id": message.header.frame_id,
            "image_width": message.width,
            "image_height": message.height,
            "inference_ms": round(inference_ms, 3),
            "detections": detections,
        }
        annotated = result.plot()
        annotated_message = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_message.header = message.header
        self.annotated_publisher.publish(annotated_message)
        self.detections_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )

        with self.latest_lock:
            self.latest_raw = frame.copy()
            self.latest_annotated = annotated.copy()
            self.latest_payload = payload
        self.processed_count += 1

    def save_frame_callback(self, _request, response):
        with self.latest_lock:
            if self.latest_raw is None:
                response.success = False
                response.message = "no processed frame available"
                return response
            raw = self.latest_raw.copy()
            annotated = self.latest_annotated.copy()
            payload = dict(self.latest_payload)

        capture_root = Path(
            str(self.get_parameter("capture_root").value)
        ).expanduser()
        session = capture_root / datetime.now().strftime("%Y%m%d")
        session.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("frame_%H%M%S_%f")
        raw_path = session / f"{stem}.jpg"
        annotated_path = session / f"{stem}_pred.jpg"
        json_path = session / f"{stem}.json"
        if not cv2.imwrite(str(raw_path), raw) or not cv2.imwrite(
            str(annotated_path), annotated
        ):
            response.success = False
            response.message = "failed to write image"
            return response
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        response.success = True
        response.message = str(raw_path)
        return response

    def log_status(self) -> None:
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        self.get_logger().info(
            f"frames_received={self.frame_count} processed={self.processed_count} "
            f"average_processed_fps={self.processed_count / elapsed:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
