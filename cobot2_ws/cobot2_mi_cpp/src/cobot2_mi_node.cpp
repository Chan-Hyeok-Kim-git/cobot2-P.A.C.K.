// ============================================================
// cobot2_mi_node.cpp — MoveIt 연동 노드 (C++)
//
// ROS2 Humble에서 Python moveit_commander / moveit_py는
// 바이너리(apt) 배포가 안 되어 사용할 수 없다.
// MoveGroupInterface(C++)는 Humble에서 완전히 정식 지원되므로
// 이 노드는 C++로 작성한다.
//
// 역할:
//   cobot2_grasp.py가 DSR ikin까지 통과시켜 발행한 ValidatedGrasp를
//   구독 → MoveIt Cartesian 경로계획(pre_grasp → grasp 직선 접근)
//   → 성공 시 /motion_plan(JSON) 발행, 실패 시 /grasp_result(JSON) 발행
//
// ROS 인터페이스:
//   SUB  /grasp/validated_grasp  (cobot2_interfaces/msg/ValidatedGrasp)
//   PUB  /motion_plan            (std_msgs/String, JSON)
//   PUB  /grasp_result           (std_msgs/String, JSON)  ← 실패 시
// ============================================================

#include <memory>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iterator>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "cobot2_interfaces/msg/validated_grasp.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

using ValidatedGrasp = cobot2_interfaces::msg::ValidatedGrasp;
using moveit::planning_interface::MoveGroupInterface;

namespace {
constexpr double kCartesianStepM = 0.005;   // Cartesian 경로 해상도 (5mm)
constexpr double kCartesianMinFrac = 0.95;  // 최소 경로 완성률 (95%)
constexpr double kPlanningTimeSec = 5.0;
constexpr double kVelocityScaling = 0.3;
constexpr double kAccelScaling = 0.3;
const char* kPlanningGroup = "manipulator";
const char* kShelfFrame = "base_link";

struct ShelfBox
{
  const char * id;
  double cx;
  double cy;
  double cz;
  double sx;
  double sy;
  double sz;
};

constexpr ShelfBox kShelfBoxes[] = {
  {"shelf_floor", 0.57, -0.48, 0.00, 1.01, 0.305, 0.01},
  {"shelf_back",  0.57, -0.48, 0.24, 1.01, 0.305, 0.01},
  {"shelf_left", -0.04, -0.47, 0.12, 0.50, 0.020, 0.24},
  {"shelf_right", 0.92, -0.47, 0.12, 0.50, 0.020, 0.24},
};
}  // namespace

class Cobot2MiNode : public rclcpp::Node
{
public:
  Cobot2MiNode() : Node("cobot2_mi")
  {
    pub_plan_ = create_publisher<std_msgs::msg::String>("/motion_plan", 10);
    pub_result_ = create_publisher<std_msgs::msg::String>("/grasp_result", 10);

    sub_ = create_subscription<ValidatedGrasp>(
      "/grasp/validated_grasp", 10,
      std::bind(&Cobot2MiNode::onValidatedGrasp, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "cobot2_mi(C++) 노드 시작 — MoveGroupInterface 초기화 대기 중");
  }

  // MoveGroupInterface는 shared_from_this()가 필요해서
  // 생성자 밖에서(노드 완전히 생성된 후) 초기화한다.
  void initMoveGroup()
  {
    try {
      move_group_ = std::make_shared<MoveGroupInterface>(
        shared_from_this(), kPlanningGroup);
      move_group_->setPlanningTime(kPlanningTimeSec);
      move_group_->setMaxVelocityScalingFactor(kVelocityScaling);
      move_group_->setMaxAccelerationScalingFactor(kAccelScaling);
      move_group_->setStartStateToCurrentState();

      if (!applyShelfCollisionObjects()) {
        RCLCPP_WARN(
          get_logger(),
          "선반 장애물 등록 확인에 실패했습니다. MoveIt 계획은 계속 사용하지만 "
          "RViz/PlanningScene에서 선반 등록 상태를 확인하십시오.");
      }

      move_group_ready_ = true;
      RCLCPP_INFO(
        get_logger(), "MoveGroupInterface 준비 완료 (end_effector: %s)",
        move_group_->getEndEffectorLink().c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        get_logger(),
        "MoveGroupInterface 초기화 실패: %s → moveit.launch.py 먼저 실행 필요", e.what());
      move_group_ready_ = false;
    }
  }

private:
  bool applyShelfCollisionObjects()
  {
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    objects.reserve(std::size(kShelfBoxes));

    for (const auto & box : kShelfBoxes) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = kShelfFrame;
      object.id = box.id;
      object.operation = moveit_msgs::msg::CollisionObject::ADD;

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {box.sx, box.sy, box.sz};

      geometry_msgs::msg::Pose pose;
      pose.position.x = box.cx;
      pose.position.y = box.cy;
      pose.position.z = box.cz;
      pose.orientation.x = 0.0;
      pose.orientation.y = 0.0;
      pose.orientation.z = 0.0;
      pose.orientation.w = 1.0;

      object.primitives.push_back(primitive);
      object.primitive_poses.push_back(pose);
      objects.push_back(object);
    }

    RCLCPP_INFO(
      get_logger(),
      "PlanningScene에 선반 장애물 %zu개 등록 요청",
      objects.size());

    const bool applied = planning_scene_interface_.applyCollisionObjects(objects);
    if (!applied) {
      RCLCPP_ERROR(get_logger(), "선반 CollisionObject 등록 요청 실패");
      return false;
    }

    // PlanningScene 반영은 비동기일 수 있으므로 짧게 확인한다.
    const auto deadline = now() + rclcpp::Duration::from_seconds(3.0);
    while (rclcpp::ok() && now() < deadline) {
      const auto known = planning_scene_interface_.getKnownObjectNames();
      bool all_found = true;

      for (const auto & box : kShelfBoxes) {
        if (std::find(known.begin(), known.end(), std::string(box.id)) == known.end()) {
          all_found = false;
          break;
        }
      }

      if (all_found) {
        RCLCPP_INFO(
          get_logger(),
          "선반 장애물 등록 완료: shelf_floor, shelf_back, shelf_left, shelf_right");
        return true;
      }

      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }

    RCLCPP_ERROR(get_logger(), "선반 장애물이 PlanningScene에 모두 나타나지 않았습니다.");
    return false;
  }

  // ── 콜백: ValidatedGrasp 수신 ──────────────────────────────────────
  void onValidatedGrasp(const ValidatedGrasp::SharedPtr msg)
  {
    RCLCPP_INFO(
      get_logger(), "ValidatedGrasp 수신 | type=%s | grasp=(%.3f, %.3f, %.3f)",
      msg->grasp_type.c_str(),
      msg->grasp_pose.pose.position.x,
      msg->grasp_pose.pose.position.y,
      msg->grasp_pose.pose.position.z);

    // MoveIt 계획은 몇 초 걸릴 수 있으므로 별도 스레드로 처리
    // (콜백을 오래 붙잡으면 다른 콜백 처리가 막힘)
    std::thread(&Cobot2MiNode::planAndPublish, this, *msg).detach();
  }


  // ── MoveIt Cartesian 경로 계획 ──────────────────────────────────────
  void planAndPublish(const ValidatedGrasp & msg)
  {
    const auto & grasp_type = msg.grasp_type;
    const auto & pre_pose = msg.pre_grasp_pose.pose;
    const auto & grasp_pose = msg.grasp_pose.pose;

    // grasp_joints(float64[6])를 vector로 변환
    // — DSR ikin으로 이미 검증된 관절값, cobot2_move.py가 movej에 바로 사용
    std::vector<double> joints(msg.grasp_joints.begin(), msg.grasp_joints.end());

    if (!move_group_ready_) {
      RCLCPP_WARN(get_logger(), "MoveIt 없음 — 관절값만 전달");
      publishPlan(msg, joints, /*trajectory=*/nullptr, 0.0);
      return;
    }

    std::lock_guard<std::mutex> lock(plan_mutex_);

    // 1단계: pre-grasp 위치로 경로 계획
    // ★ 수정: setPoseTarget() 대신 setJointValueTarget() 사용.
    // pre_grasp_joints는 grasp.py에서 DSR ikin(해석적 solver)으로
    // 이미 검증된 관절값이다. MoveIt 기본 IK 솔버(KDL, 수치해석)는
    // 같은 pose에 대해 IK를 다시 계산하다 실패하는 경우가 많으므로
    // (Unable to sample any valid states) pose 기반 목표 대신
    // 이미 풀린 관절값으로 직접 목표를 설정해 이 문제를 회피한다.
    std::vector<double> pre_joints(
      msg.pre_grasp_joints.begin(), msg.pre_grasp_joints.end());

    bool success_pre = false;
    MoveGroupInterface::Plan plan_to_pre;

    if (pre_joints.size() == 6) {
      // DSR 관절값은 degree 단위 → MoveIt은 radian 단위 사용
      std::vector<double> pre_joints_rad;
      for (double d : pre_joints) {
        pre_joints_rad.push_back(d * M_PI / 180.0);
      }
      move_group_->setJointValueTarget(pre_joints_rad);
      success_pre = (move_group_->plan(plan_to_pre) ==
        moveit::core::MoveItErrorCode::SUCCESS);
    } else {
      // pre_grasp_joints가 없는 예외 상황 → pose 기반으로 폴백
      RCLCPP_WARN(get_logger(), "pre_grasp_joints 없음 — pose 기반 IK로 폴백");
      move_group_->setPoseTarget(pre_pose);
      success_pre = (move_group_->plan(plan_to_pre) ==
        moveit::core::MoveItErrorCode::SUCCESS);
    }

    if (!success_pre) {
      std::string reason = grasp_type + " pre-grasp 계획 실패 (충돌 또는 워크스페이스 밖)";
      RCLCPP_ERROR(get_logger(), "%s", reason.c_str());
      publishFail(msg, reason);
      return;
    }

    // 2단계: pre-grasp → grasp Cartesian 직선 접근
    // (그리퍼가 물체 쪽으로 흔들림 없이 직선으로 들어가는 경로)
    std::vector<geometry_msgs::msg::Pose> waypoints = {pre_pose, grasp_pose};
    moveit_msgs::msg::RobotTrajectory cartesian_traj;
    // Humble API: (waypoints, eef_step, jump_threshold, trajectory, avoid_collisions, error_code)
    // jump_threshold=0.0 → 점프 검사 비활성화 (deprecated지만 Humble에서 여전히 필요한 인자)
    double fraction = move_group_->computeCartesianPath(
      waypoints, kCartesianStepM, /*jump_threshold=*/0.0, cartesian_traj);

    if (fraction < kCartesianMinFrac) {
      // ★ 수정: Cartesian 직선 경로가 부족해도 바로 실패 처리하지 않고
      // 관절공간 경로로 폴백한다. grasp_joints는 DSR ikin으로 이미
      // 도달 가능함이 검증됐으므로, 직선은 아니어도 충돌 없이
      // 도달하는 경로는 OMPL이 찾아줄 수 있다.
      // (joint_2가 관절한계 근처라 직선 보간 중 IK가 자주 끊기는 것이
      //  Cartesian 완성률이 낮은 원인 — 물리적으로 도달 불가능한 건 아님)
      RCLCPP_WARN(
        get_logger(),
        "[%s] Cartesian 경로 불완전(%.0f%% < %.0f%%) — 관절공간 경로로 재시도",
        grasp_type.c_str(), fraction * 100, kCartesianMinFrac * 100);

      std::vector<double> joints_rad;
      for (double d : joints) {joints_rad.push_back(d * M_PI / 180.0);}

      move_group_->setJointValueTarget(joints_rad);
      MoveGroupInterface::Plan plan_to_grasp;
      bool success_grasp = (move_group_->plan(plan_to_grasp) ==
        moveit::core::MoveItErrorCode::SUCCESS);

      if (!success_grasp) {
        std::string reason = grasp_type + " grasp 관절공간 계획도 실패 (충돌 가능성)";
        RCLCPP_ERROR(get_logger(), "%s", reason.c_str());
        publishFail(msg, reason);
        return;
      }

      RCLCPP_INFO(
        get_logger(), "[%s] 관절공간 폴백 경로 계획 성공 (직선은 아님)",
        grasp_type.c_str());

      // pre-grasp 궤적 + grasp 관절공간 궤적을 함께 전달
      publishPlan(msg, joints, &plan_to_grasp.trajectory_, /*fraction=*/-1.0);
      return;
    }

    RCLCPP_INFO(
      get_logger(), "[%s] 경로 계획 성공 (Cartesian %.0f%%)",
      grasp_type.c_str(), fraction * 100);

    publishPlan(msg, joints, &cartesian_traj, fraction);
  }

  // ── /motion_plan 발행 (성공) ─────────────────────────────────────────
  // cobot2_move.py가 이 JSON을 받아서 실제 로봇을 움직인다.
  void get_current_info(std::shared_ptr<rclcpp::Node> node)
  {
      moveit::planning_interface::MoveGroupInterface move_group(node, "arm_group");

      // 1. 플래닝 시작점을 현재 위치로 설정 (기본값이지만 명시할 때 사용)
      move_group.setStartStateToCurrentState();

      // 2. 현재 관절 각도(double 벡터) 가져오기
      std::vector<double> current_joints = move_group.getCurrentJointValues();
      RCLCPP_INFO(node->get_logger(), "첫 번째 관절의 현재 각도: %f", current_joints[0]);

      // 3. 현재 엔드이펙터의 포즈(XYZ 위치, 쿼터니언 자세) 가져오기
      geometry_msgs::msg::PoseStamped current_pose = move_group.getCurrentPose();
      RCLCPP_INFO(node->get_logger(), "현재 X 좌표: %f, Y 좌표: %f", 
                  current_pose.pose.position.x, current_pose.pose.position.y);
  }
  void publishPlan(
    const ValidatedGrasp & msg,
    const std::vector<double> & joints,
    const moveit_msgs::msg::RobotTrajectory * traj,
    double fraction)
  {
    // ★ 추가: pre_grasp_joints도 그대로 전달.
    // cobot2_move.py가 pre-grasp 이동 시 movej(posj(*pre_grasp_joints))로
    // 바로 쓸 수 있게 한다. (이전에는 이게 없어서 posx를 movej에 잘못
    // 넘기다가 'Invalid type : pos' 에러로 크래시가 났었음)
    std::vector<double> pre_joints(
      msg.pre_grasp_joints.begin(), msg.pre_grasp_joints.end());

    std::ostringstream oss;
    oss << "{"
        << "\"grasp_type\":\"" << msg.grasp_type << "\","
        << "\"grasp_joints\":" << jointsToJsonArray(joints) << ","
        << "\"pre_grasp_joints\":" << jointsToJsonArray(pre_joints) << ","
        << "\"grip_width_mm\":" << (msg.required_width * 1000.0) << ","
        << "\"pre_grasp_xyz\":[" << msg.pre_grasp_pose.pose.position.x << ","
        << msg.pre_grasp_pose.pose.position.y << ","
        << msg.pre_grasp_pose.pose.position.z << "],"
        << "\"grasp_xyz\":[" << msg.grasp_pose.pose.position.x << ","
        << msg.grasp_pose.pose.position.y << ","
        << msg.grasp_pose.pose.position.z << "],"
        << "\"trajectory\":" << (traj ? trajectoryToJson(*traj) : "null") << ","
        << "\"cartesian_fraction\":" << fraction << ","
        << "\"success\":true"
        << "}";

    std_msgs::msg::String out;
    out.data = oss.str();
    pub_plan_->publish(out);
    RCLCPP_INFO(get_logger(), "[%s] /motion_plan 발행 완료", msg.grasp_type.c_str());
  }

  // ── /grasp_result 발행 (실패) ────────────────────────────────────────
  void publishFail(const ValidatedGrasp & msg, const std::string & reason)
  {
    std::ostringstream oss;
    oss << "{"
        << "\"grasp_type\":\"" << msg.grasp_type << "\","
        << "\"success\":false,"
        << "\"reason\":\"" << reason << "\""
        << "}";
    std_msgs::msg::String out;
    out.data = oss.str();
    pub_result_->publish(out);
  }

  // ── 헬퍼: 관절값 배열 → JSON 배열 문자열 ────────────────────────────
  static std::string jointsToJsonArray(const std::vector<double> & joints)
  {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < joints.size(); ++i) {
      oss << joints[i];
      if (i + 1 < joints.size()) {oss << ",";}
    }
    oss << "]";
    return oss.str();
  }

  // ── 헬퍼: RobotTrajectory → JSON (cobot2_move.py에서 실행용) ────────
  static std::string trajectoryToJson(const moveit_msgs::msg::RobotTrajectory & traj)
  {
    const auto & jt = traj.joint_trajectory;
    std::ostringstream oss;
    oss << "{\"joint_names\":[";
    for (size_t i = 0; i < jt.joint_names.size(); ++i) {
      oss << "\"" << jt.joint_names[i] << "\"";
      if (i + 1 < jt.joint_names.size()) {oss << ",";}
    }
    oss << "],\"points\":[";
    for (size_t i = 0; i < jt.points.size(); ++i) {
      const auto & pt = jt.points[i];
      oss << "{\"positions\":" << jointsToJsonArray(
        std::vector<double>(pt.positions.begin(), pt.positions.end()))
          << ",\"time_from_start\":"
          << (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9)
          << "}";
      if (i + 1 < jt.points.size()) {oss << ",";}
    }
    oss << "]}";
    return oss.str();
  }

  rclcpp::Subscription<ValidatedGrasp>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_plan_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_result_;
  std::shared_ptr<MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  bool move_group_ready_ {false};
  std::mutex plan_mutex_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<Cobot2MiNode>();

  // MoveGroupInterface는 노드가 shared_ptr로 완전히 생성된 뒤에
  // 초기화해야 한다 (shared_from_this() 사용 때문).
  node->initMoveGroup();

  // MoveGroupInterface 내부적으로 별도 스레드에서 spin이 필요할 수 있어
  // MultiThreadedExecutor 사용
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
} 