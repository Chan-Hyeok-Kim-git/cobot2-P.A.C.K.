"""
YOLO/RealSense 없이 파이프라인을 통째로 테스트하기 위한 mock 노드.

가상의 물체(직육면체) 하나를 만들어:
  - target_cloud (물체 표면 점군)
  - environment_cloud (주변 노이즈 점 몇 개)
  - ObjectGeometry (물체 OBB, 완벽한 값으로 미리 계산해서 발행)
를 주기적으로 퍼블리시한다.

사용법:
  ros2 run grasp_local_validation mock_scene_publisher \
      --ros-args -p class_name:=first_aid_kit -p center_xyz:="[0.55,0.0,0.45]" \
                 -p size_xyz:="[0.20,0.13,0.08]"
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from sensor_msgs.msg import PointCloud2

from grasp_interfaces.msg import ObjectGeometry
from . import pointcloud_ros as pc_ros


def make_box_surface_points(center, size, n_per_face=800):
    """직육면체 표면에 점을 뿌려 target_cloud 근사"""
    cx, cy, cz = center
    sx, sy, sz = size
    pts = []
    rng = np.random.default_rng(0)
    for axis in range(3):
        for sign in [-1, 1]:
            u = rng.uniform(-0.5, 0.5, n_per_face)
            v = rng.uniform(-0.5, 0.5, n_per_face)
            face = np.zeros((n_per_face, 3))
            dims = [0, 1, 2]
            dims.remove(axis)
            face[:, dims[0]] = u * [sx, sy, sz][dims[0]]
            face[:, dims[1]] = v * [sx, sy, sz][dims[1]]
            face[:, axis] = sign * 0.5 * [sx, sy, sz][axis]
            pts.append(face)
    pts = np.vstack(pts)
    pts += np.array([cx, cy, cz])
    return pts.astype(np.float32)


def make_noise_points(n=100, bounds=((0.2, 1.0), (-0.6, 0.6), (0.0, 1.4))):
    rng = np.random.default_rng(1)
    xs = rng.uniform(*bounds[0], n)
    ys = rng.uniform(*bounds[1], n)
    zs = rng.uniform(*bounds[2], n)
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


class MockScenePublisher(Node):
    def __init__(self):
        super().__init__('mock_scene_publisher')

        self.declare_parameter('class_name', 'first_aid_kit')
        self.declare_parameter('class_id', 11)
        self.declare_parameter('center_xyz', [0.55, 0.0, 0.45])
        self.declare_parameter('size_xyz', [0.20, 0.13, 0.08])
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('target_cloud_topic', '/grasp_scene/target_cloud')
        self.declare_parameter('environment_cloud_topic', '/grasp_scene/environment_cloud')
        self.declare_parameter('object_geometry_topic', '/grasp_scene/object_geometry')
        self.declare_parameter('publish_rate_hz', 1.0)

        gp = self.get_parameter
        self.class_name = gp('class_name').value
        self.class_id = gp('class_id').value
        self.center = np.array(gp('center_xyz').value)
        self.size = np.array(gp('size_xyz').value)
        self.frame_id = gp('frame_id').value

        self.target_pub = self.create_publisher(PointCloud2, gp('target_cloud_topic').value, 10)
        self.env_pub = self.create_publisher(PointCloud2, gp('environment_cloud_topic').value, 10)
        self.geom_pub = self.create_publisher(ObjectGeometry, gp('object_geometry_topic').value, 10)

        self._scene_id = 0
        period = 1.0 / max(gp('publish_rate_hz').value, 0.1)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"mock_scene_publisher: class={self.class_name} center={self.center.tolist()} size={self.size.tolist()}"
        )

    def _tick(self):
        now = self.get_clock().now().to_msg()
        self._scene_id += 1

        target = make_box_surface_points(self.center, self.size)
        noise = make_noise_points()

        self.target_pub.publish(pc_ros.xyz_array_to_pointcloud2(target, self.frame_id, now))
        self.env_pub.publish(pc_ros.xyz_array_to_pointcloud2(noise, self.frame_id, now))

        geom = ObjectGeometry()
        geom.header = Header(frame_id=self.frame_id, stamp=now)
        geom.scene_id = self._scene_id
        geom.class_id = self.class_id
        geom.class_name = self.class_name
        geom.detection_confidence = 0.95
        geom.pose = Pose(
            position=Point(x=float(self.center[0]), y=float(self.center[1]), z=float(self.center[2])),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        # dimensions = [long, mid, short] 순으로 정렬해서 채움
        sorted_size = sorted(self.size.tolist(), reverse=True)
        geom.dimensions = Vector3(x=sorted_size[0], y=sorted_size[1], z=sorted_size[2])
        geom.long_axis = Vector3(x=1.0, y=0.0, z=0.0)
        geom.short_axis = Vector3(x=0.0, y=1.0, z=0.0)
        geom.vertical_axis = Vector3(x=0.0, y=0.0, z=1.0)
        geom.point_count = len(target)
        self.geom_pub.publish(geom)


def main(args=None):
    rclpy.init(args=args)
    node = MockScenePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
