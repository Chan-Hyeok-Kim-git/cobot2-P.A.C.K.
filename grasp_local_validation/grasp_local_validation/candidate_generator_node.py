"""
ObjectGeometry(담당자1이 실시간 디텍션+3D변환으로 넘겨주는 물체 OBB) 를 구독해서
rg2_collision.yaml + shelf_collision.yaml 기준으로 순수 기하학적으로
GraspCandidate 후보들을 생성하고 candidate_topic 에 퍼블리시한다.

클래스별 사전 프로파일(grasp_profiles.yaml) 은 사용하지 않는다 - 어떤 클래스든
그 순간 들어온 실측 위치/치수만으로 top/front 가능 여부를 계산한다.

local_validator_node 와 별도 프로세스로 띄워도 되고, 같이 launch 해도 된다.
"""
import itertools
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

from grasp_interfaces.msg import ObjectGeometry, GraspCandidate, GraspCandidateArray

from . import candidate_generation as cg
from . import config_loader


def _vec3_to_np(v: Vector3):
    return np.array([v.x, v.y, v.z])


def object_geometry_to_obb(msg: ObjectGeometry):
    center = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    ext = _vec3_to_np(msg.dimensions)  # [long, mid, short] 로 가정 (담당자1과 합의된 규약)
    return {
        "center": center,
        "long_axis": _vec3_to_np(msg.long_axis),
        "mid_axis": np.cross(_vec3_to_np(msg.vertical_axis), _vec3_to_np(msg.long_axis)),
        "short_axis": _vec3_to_np(msg.short_axis),
        "extents": ext,
        "top_z": center[2] + ext[2] / 2.0 if ext[2] else center[2],
        "bottom_z": center[2] - ext[2] / 2.0 if ext[2] else center[2],
        "point_count": msg.point_count,
    }


class CandidateGeneratorNode(Node):
    def __init__(self):
        super().__init__('grasp_candidate_generator')

        self.declare_parameter('object_geometry_topic', '/grasp_scene/object_geometry')
        self.declare_parameter('candidate_topic', '/grasp_scene/grasp_candidates')
        self.declare_parameter('rg2_config_path', '')
        self.declare_parameter('shelf_config_path', '')

        gp = self.get_parameter
        self.rg2_cfg = config_loader.load_rg2_config(gp('rg2_config_path').value or None)
        self.shelf_cfg = config_loader.load_shelf_config(gp('shelf_config_path').value or None)

        self._candidate_id_counter = itertools.count()

        self.create_subscription(
            ObjectGeometry, gp('object_geometry_topic').value, self._on_object, 10)
        self.pub = self.create_publisher(
            GraspCandidateArray, gp('candidate_topic').value, 10)

        self.get_logger().info("grasp_candidate_generator ready (profile-free, 순수 기하학 판단)")

    def _on_object(self, msg: ObjectGeometry):
        obb = object_geometry_to_obb(msg)
        cand_dicts = cg.generate_candidates(obb, self.rg2_cfg, self.shelf_cfg)

        arr = GraspCandidateArray()
        arr.header = Header(frame_id=msg.header.frame_id, stamp=self.get_clock().now().to_msg())
        arr.scene_id = msg.scene_id

        for d in cand_dicts:
            gm = GraspCandidate()
            gm.header = arr.header
            gm.scene_id = msg.scene_id
            gm.candidate_id = next(self._candidate_id_counter)
            gm.grasp_type = d['grasp_type']

            gm.grasp_pose = Pose(
                position=Point(x=float(d['grasp_pos'][0]), y=float(d['grasp_pos'][1]), z=float(d['grasp_pos'][2])),
                orientation=Quaternion(
                    x=float(d['orientation_xyzw'][0]), y=float(d['orientation_xyzw'][1]),
                    z=float(d['orientation_xyzw'][2]), w=float(d['orientation_xyzw'][3])),
            )
            gm.pre_grasp_pose = Pose(
                position=Point(x=float(d['pre_grasp_pos'][0]), y=float(d['pre_grasp_pos'][1]), z=float(d['pre_grasp_pos'][2])),
                orientation=gm.grasp_pose.orientation,
            )
            gm.approach_direction = Vector3(x=float(d['approach_direction'][0]),
                                             y=float(d['approach_direction'][1]),
                                             z=float(d['approach_direction'][2]))
            gm.closing_direction = Vector3(x=float(d['closing_direction'][0]),
                                            y=float(d['closing_direction'][1]),
                                            z=float(d['closing_direction'][2]))
            gm.finger_direction = Vector3(x=float(d['finger_direction'][0]),
                                           y=float(d['finger_direction'][1]),
                                           z=float(d['finger_direction'][2]))
            gm.required_width = float(d['required_width'])
            gm.approach_distance = float(d['approach_distance'])
            gm.initial_score = float(d.get('initial_score', 0.0))
            arr.candidates.append(gm)

        self.pub.publish(arr)
        self.get_logger().info(f"'{msg.class_name}' -> 후보 {len(arr.candidates)}개 생성")


def main(args=None):
    rclpy.init(args=args)
    node = CandidateGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
