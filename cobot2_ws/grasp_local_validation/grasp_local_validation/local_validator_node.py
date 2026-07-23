"""
grasp_local_validation 패키지의 메인 노드.

역할
1. target_cloud_topic / environment_cloud_topic (PointCloud2) 구독, 최신 점군 보관
2. candidate_topic (GraspCandidateArray) 구독 -> 자동 검증 -> result_topic 퍼블리시
3. ValidateLocalGrasps 서비스 제공 (동기 요청/응답)
4. publish_debug_markers=true 이면 RViz 마커 퍼블리시

local_validator.yaml 의 파라미터 이름과 정확히 일치시킨다.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

from grasp_interfaces.msg import (
    GraspCandidate, GraspCandidateArray,
    ValidatedGrasp, ValidatedGraspArray,
)
from grasp_interfaces.srv import ValidateLocalGrasps

from . import collision
from . import pointcloud_ros as pc_ros
from . import config_loader


def candidate_msg_to_dict(msg: GraspCandidate):
    return {
        "type": f"type_{msg.candidate_id}",
        "grasp_type": msg.grasp_type,
        "grasp_pos": np.array([msg.grasp_pose.position.x,
                                msg.grasp_pose.position.y,
                                msg.grasp_pose.position.z]),
        "orientation_xyzw": np.array([msg.grasp_pose.orientation.x,
                                       msg.grasp_pose.orientation.y,
                                       msg.grasp_pose.orientation.z,
                                       msg.grasp_pose.orientation.w]),
        "pre_grasp_pos": np.array([msg.pre_grasp_pose.position.x,
                                    msg.pre_grasp_pose.position.y,
                                    msg.pre_grasp_pose.position.z]),
        "required_width": msg.required_width,
        "approach_distance": msg.approach_distance,
        "initial_score": msg.initial_score,
    }


class LocalValidatorNode(Node):
    def __init__(self):
        super().__init__('grasp_local_validator')

        self.declare_parameter('target_cloud_topic', '/grasp_scene/target_cloud')
        self.declare_parameter('environment_cloud_topic', '/grasp_scene/environment_cloud')
        self.declare_parameter('candidate_topic', '/grasp_scene/grasp_candidates')
        self.declare_parameter('result_topic', '/grasp_validation/results')
        self.declare_parameter('approach_sample_step', 0.010)
        self.declare_parameter('enable_width_check', True)
        self.declare_parameter('enable_contact_check', True)
        self.declare_parameter('enable_environment_collision', True)
        self.declare_parameter('enable_target_collision', True)
        self.declare_parameter('enable_shelf_collision', True)
        self.declare_parameter('enable_ik_check', True)  # moveit_validator 단계에서 실제 처리
        self.declare_parameter('min_contact_points', 10)
        self.declare_parameter('max_joint_jump_rad', 0.6)
        self.declare_parameter('publish_debug_markers', True)
        self.declare_parameter('rg2_config_path', '')   # 비워두면 ament_index로 자동탐색
        self.declare_parameter('shelf_config_path', '')

        gp = self.get_parameter
        self._target_topic = gp('target_cloud_topic').value
        self._env_topic = gp('environment_cloud_topic').value
        self._cand_topic = gp('candidate_topic').value
        self._result_topic = gp('result_topic').value
        self._approach_step = gp('approach_sample_step').value
        self._min_contact = gp('min_contact_points').value
        self._publish_markers = gp('publish_debug_markers').value

        rg2_override = gp('rg2_config_path').value or None
        shelf_override = gp('shelf_config_path').value or None
        self.rg2_cfg = config_loader.load_rg2_config(rg2_override)
        self.shelf_cfg = config_loader.load_shelf_config(shelf_override)

        self._latest_target = np.zeros((0, 3), dtype=np.float32)
        self._latest_env = np.zeros((0, 3), dtype=np.float32)
        self._frame_id = 'base_link'

        self.create_subscription(PointCloud2, self._target_topic, self._on_target_cloud, 10)
        self.create_subscription(PointCloud2, self._env_topic, self._on_env_cloud, 10)
        self.create_subscription(GraspCandidateArray, self._cand_topic, self._on_candidates, 10)

        self.result_pub = self.create_publisher(ValidatedGraspArray, self._result_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/grasp_validation/debug_markers', 10)

        self.srv = self.create_service(
            ValidateLocalGrasps, 'validate_local_grasps', self._on_validate_service)

        self.get_logger().info(
            f"grasp_local_validator ready. target='{self._target_topic}' "
            f"env='{self._env_topic}' candidates='{self._cand_topic}'"
        )

    # ------------------------------------------------------------------
    def _on_target_cloud(self, msg: PointCloud2):
        self._latest_target = pc_ros.pointcloud2_to_xyz_array(msg)
        self._frame_id = msg.header.frame_id or self._frame_id

    def _on_env_cloud(self, msg: PointCloud2):
        self._latest_env = pc_ros.pointcloud2_to_xyz_array(msg)

    def _on_candidates(self, msg: GraspCandidateArray):
        results = [self._validate_one(c) for c in msg.candidates]
        out = ValidatedGraspArray()
        out.header = Header(frame_id=self._frame_id, stamp=self.get_clock().now().to_msg())
        out.scene_id = msg.scene_id
        out.grasps = results
        self.result_pub.publish(out)
        if self._publish_markers:
            self._publish_debug_markers(results)

    def _on_validate_service(self, request: ValidateLocalGrasps.Request,
                              response: ValidateLocalGrasps.Response):
        try:
            results = [self._validate_one(c) for c in request.candidates]
            response.success = True
            response.failure_reason = ""
            response.results = results
            if self._publish_markers:
                self._publish_debug_markers(results)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"validate_local_grasps failed: {e}")
            response.success = False
            response.failure_reason = str(e)
            response.results = []
        return response

    # ------------------------------------------------------------------
    def _validate_one(self, cand_msg: GraspCandidate) -> ValidatedGrasp:
        cand = candidate_msg_to_dict(cand_msg)
        try:
            valid, code, reason, extra = collision.validate_candidate(
                cand, self._latest_target, self._latest_env,
                self.rg2_cfg, self.shelf_cfg,
                min_contact_points=self._min_contact,
                approach_sample_step=self._approach_step,
            )
        except Exception as e:  # noqa: BLE001
            valid, code, reason, extra = False, ValidatedGrasp.BODY_COLLISION, f"exception: {e}", {}

        out = ValidatedGrasp()
        out.candidate = cand_msg
        out.valid = valid
        out.failure_code = code
        out.failure_reason = reason
        out.measured_width = float(cand['required_width'])
        out.left_contact_score = float(extra.get('left_contact', 0))
        out.right_contact_score = float(extra.get('right_contact', 0))
        out.min_clearance = float(extra.get('min_clearance', 0.0))
        out.collision_count = 0
        out.local_score = float(collision.score_candidate(cand, extra)) if valid else -1.0
        out.ik_checked = False  # grasp_moveit_validator 단계에서 채움
        return out

    # ------------------------------------------------------------------
    def _publish_debug_markers(self, results):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, r in enumerate(results):
            m = Marker()
            m.header = Header(frame_id=self._frame_id, stamp=now)
            m.ns = 'grasp_candidates'
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = r.candidate.grasp_pose
            m.scale.x = 0.08
            m.scale.y = 0.015
            m.scale.z = 0.015
            if r.valid:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.1, 0.9
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.1, 0.1, 0.6
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = LocalValidatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
