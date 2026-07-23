// grasp_moveit_validator_node.cpp
//
// ValidateAndPlan 서비스 구현체.
// grasp_local_validation 을 통과한 ValidatedGrasp 후보들을 받아
// 실제 MoveIt Planning Scene 기준으로:
//   1) pre_grasp / grasp 자세에 대한 IK 존재 여부
//   2) pre_grasp -> grasp 사이 Cartesian 경로 계획 (충돌 포함)
//   3) dynamic_obstacles(CollisionBox[])를 planning scene에 반영한 뒤 재검사
// 를 수행하고, 가장 점수가 높은 후보의 궤적을 반환한다.
//
// 빌드/실행 전 필요조건 (이 워크스페이스만으로는 불가능, 반드시 준비 필요):
//   - grasp_moveit_config 패키지에 실제 로봇(M0609 + RG2)의 SRDF/kinematics.yaml 이 있어야 함
//   - moveit_validator.yaml 의 planning_group / end_effector_link 이름이
//     실제 SRDF의 move_group, link 이름과 정확히 일치해야 함

#include <memory>
#include <vector>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include "grasp_interfaces/srv/validate_and_plan.hpp"
#include "grasp_interfaces/msg/validated_grasp.hpp"
#include "grasp_interfaces/msg/collision_box.hpp"

using ValidateAndPlan = grasp_interfaces::srv::ValidateAndPlan;
using grasp_interfaces::msg::CollisionBox;
using grasp_interfaces::msg::ValidatedGrasp;

class MoveitValidatorNode : public rclcpp::Node
{
public:
  MoveitValidatorNode()
  : Node("grasp_moveit_validator")
  {
    this->declare_parameter<std::string>("planning_group", "manipulator");
    this->declare_parameter<std::string>("base_frame", "base_link");
    this->declare_parameter<std::string>("end_effector_link", "rg2_tcp");
    this->declare_parameter<std::string>("planner_id", "RRTConnectkConfigDefault");
    this->declare_parameter<double>("planning_time", 2.0);
    this->declare_parameter<int>("planning_attempts", 3);
    this->declare_parameter<double>("velocity_scaling", 0.20);
    this->declare_parameter<double>("acceleration_scaling", 0.20);
    this->declare_parameter<double>("cartesian_eef_step", 0.005);
    this->declare_parameter<double>("cartesian_jump_threshold", 0.0);
    this->declare_parameter<double>("cartesian_fraction_threshold", 0.99);
    this->declare_parameter<double>("dynamic_obstacle_margin", 0.005);

    planning_group_ = this->get_parameter("planning_group").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    ee_link_ = this->get_parameter("end_effector_link").as_string();

    // MoveGroupInterface는 노드가 spin을 시작한 뒤 생성해야 하므로
    // 타이머로 한 번만 지연 초기화한다.
    init_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&MoveitValidatorNode::lazyInit, this));
  }

private:
  void lazyInit()
  {
    init_timer_->cancel();

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);
    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setEndEffectorLink(ee_link_);
    move_group_->setPlannerId(this->get_parameter("planner_id").as_string());
    move_group_->setPlanningTime(this->get_parameter("planning_time").as_double());
    move_group_->setNumPlanningAttempts(this->get_parameter("planning_attempts").as_int());
    move_group_->setMaxVelocityScalingFactor(this->get_parameter("velocity_scaling").as_double());
    move_group_->setMaxAccelerationScalingFactor(
      this->get_parameter("acceleration_scaling").as_double());

    psi_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

    srv_ = this->create_service<ValidateAndPlan>(
      "validate_and_plan",
      std::bind(&MoveitValidatorNode::onRequest, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "grasp_moveit_validator ready (group=%s, ee=%s)",
                planning_group_.c_str(), ee_link_.c_str());
  }

  // dynamic_obstacles(CollisionBox[]) 를 planning scene collision object 로 반영
  void applyDynamicObstacles(const std::vector<CollisionBox> & boxes)
  {
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    double margin = this->get_parameter("dynamic_obstacle_margin").as_double();

    for (const auto & box : boxes) {
      moveit_msgs::msg::CollisionObject co;
      co.header.frame_id = base_frame_;
      co.id = box.object_id.empty() ? ("dyn_obstacle_" + std::to_string(objects.size()))
                                     : box.object_id;

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = primitive.BOX;
      primitive.dimensions = {
        box.dimensions.x + 2 * margin,
        box.dimensions.y + 2 * margin,
        box.dimensions.z + 2 * margin
      };
      co.primitives.push_back(primitive);
      co.primitive_poses.push_back(box.pose);
      co.operation = co.ADD;
      objects.push_back(co);
    }
    if (!objects.empty()) {
      psi_->applyCollisionObjects(objects);
    }
  }

  void removeDynamicObstacles(const std::vector<CollisionBox> & boxes)
  {
    std::vector<std::string> ids;
    for (const auto & box : boxes) {
      ids.push_back(box.object_id.empty() ? "dyn_obstacle" : box.object_id);
    }
    if (!ids.empty()) {
      psi_->removeCollisionObjects(ids);
    }
  }

  // 후보 하나에 대해 IK + cartesian 경로를 계획. 성공하면 true.
  bool planOne(
    const ValidatedGrasp & candidate,
    moveit_msgs::msg::RobotTrajectory & to_pre_grasp_traj,
    moveit_msgs::msg::RobotTrajectory & approach_traj,
    double & cartesian_fraction,
    std::string & failure_reason)
  {
    const auto & pre_pose = candidate.candidate.pre_grasp_pose;
    const auto & grasp_pose = candidate.candidate.grasp_pose;

    // 1) pre-grasp 로 이동하는 계획 (joint-space, 충돌 검사 포함 -> 사실상 IK+충돌 동시 확인)
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(pre_pose, ee_link_);

    moveit::planning_interface::MoveGroupInterface::Plan pre_plan;
    auto pre_result = move_group_->plan(pre_plan);
    if (pre_result != moveit::core::MoveItErrorCode::SUCCESS) {
      failure_reason = "pre_grasp planning failed (IK/충돌)";
      return false;
    }
    to_pre_grasp_traj = pre_plan.trajectory_;

    // 2) pre-grasp -> grasp 구간은 직선 접근이 중요하므로 Cartesian 경로로 계획
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(pre_pose);
    waypoints.push_back(grasp_pose);

    moveit_msgs::msg::RobotTrajectory cartesian_traj;
    double eef_step = this->get_parameter("cartesian_eef_step").as_double();
    double jump_thr = this->get_parameter("cartesian_jump_threshold").as_double();

    cartesian_fraction = move_group_->computeCartesianPath(
      waypoints, eef_step, jump_thr, cartesian_traj);

    double fraction_threshold =
      this->get_parameter("cartesian_fraction_threshold").as_double();
    if (cartesian_fraction < fraction_threshold) {
      failure_reason = "approach cartesian path incomplete (fraction=" +
        std::to_string(cartesian_fraction) + ")";
      return false;
    }
    approach_traj = cartesian_traj;
    return true;
  }

  void onRequest(
    const std::shared_ptr<ValidateAndPlan::Request> request,
    std::shared_ptr<ValidateAndPlan::Response> response)
  {
    applyDynamicObstacles(request->dynamic_obstacles);

    int best_id = -1;
    double best_score = -1e9;
    moveit_msgs::msg::RobotTrajectory best_pre, best_approach;
    double best_fraction = 0.0;
    std::string last_failure;

    for (const auto & cand : request->candidates) {
      if (!cand.valid) {
        continue;  // local validator 단계에서 이미 탈락한 후보는 스킵
      }

      moveit_msgs::msg::RobotTrajectory pre_traj, approach_traj;
      double fraction = 0.0;
      std::string reason;
      bool ok = planOne(cand, pre_traj, approach_traj, fraction, reason);

      if (!ok) {
        last_failure = reason;
        continue;
      }

      double score = cand.local_score + fraction;  // 단순 결합 점수 (필요시 조정)
      if (score > best_score) {
        best_score = score;
        best_id = cand.candidate.candidate_id;
        best_pre = pre_traj;
        best_approach = approach_traj;
        best_fraction = fraction;
      }
    }

    removeDynamicObstacles(request->dynamic_obstacles);

    if (best_id < 0) {
      response->success = false;
      response->failure_reason = last_failure.empty()
        ? "no valid candidate reached planning stage" : last_failure;
      response->selected_candidate_id = -1;
      return;
    }

    response->success = true;
    response->failure_reason = "";
    response->selected_candidate_id = best_id;
    response->to_pre_grasp = best_pre;
    response->approach = best_approach;
    response->cartesian_fraction = static_cast<float>(best_fraction);
    response->planning_time =
      static_cast<float>(this->get_parameter("planning_time").as_double());
  }

  std::string planning_group_, base_frame_, ee_link_;
  rclcpp::TimerBase::SharedPtr init_timer_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> psi_;
  rclcpp::Service<ValidateAndPlan>::SharedPtr srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveitValidatorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
