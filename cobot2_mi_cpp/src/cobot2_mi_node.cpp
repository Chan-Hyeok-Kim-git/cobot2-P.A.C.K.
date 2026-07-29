// cobot2_mi_node.cpp
// /grasp/target -> MoveIt planning -> /moveit_grasp/planned
// End-effector target frame: the actual rg2_tcp frame from URDF.
// Configurable TCP convention: selected local axis = closing axis, signed local Z = approach.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "moveit_msgs/msg/display_trajectory.hpp"
#include "moveit_msgs/msg/robot_state.hpp"
#include "moveit_msgs/msg/robot_trajectory.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"
#include "cobot2_interfaces/msg/grasp_target.hpp"
#include "cobot2_interfaces/msg/planned_grasp.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>

using GraspTarget = cobot2_interfaces::msg::GraspTarget;
using PlannedGrasp = cobot2_interfaces::msg::PlannedGrasp;
using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using PlanningSceneInterface = moveit::planning_interface::PlanningSceneInterface;

namespace
{
constexpr double kEps = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

double rad(double degrees) { return degrees * kPi / 180.0; }
bool isFinite(double value) { return std::isfinite(value); }

struct Candidate
{
  int id {-1};
  std::string type;
  Eigen::Vector3d approach {Eigen::Vector3d::Zero()};
  double closing_sign {1.0};
};

struct PlanResult
{
  int candidate_id {-1};
  std::string grasp_type;
  std::string failure_reason;
  geometry_msgs::msg::PoseStamped pre_pose;
  geometry_msgs::msg::PoseStamped grasp_pose;
  moveit_msgs::msg::RobotState start_state;
  moveit_msgs::msg::RobotTrajectory to_pre;
  moveit_msgs::msg::RobotTrajectory approach;
  double cartesian_fraction {0.0};
};

struct ShelfBox
{
  std::string id;
  Eigen::Vector3d center {Eigen::Vector3d::Zero()};
  Eigen::Vector3d size {Eigen::Vector3d::Zero()};
};
}  // namespace

class Cobot2MiNode : public rclcpp::Node
{
public:
  Cobot2MiNode() : Node("cobot2_mi")
  {
    planning_group_ = declare_parameter<std::string>("planning_group", "manipulator");
    eef_link_ = declare_parameter<std::string>("eef_link", "rg2_tcp");
    shelf_yaml_ = declare_parameter<std::string>("shelf_yaml", "");
    input_topic_ = declare_parameter<std::string>("input_topic", "/grasp/target");
    output_topic_ = declare_parameter<std::string>("output_topic", "/moveit_grasp/planned");

    planning_time_ = declare_parameter<double>("planning_time", 3.0);
    planning_attempts_ = declare_parameter<int>("planning_attempts", 3);
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.5);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.5);
    pre_distance_ = declare_parameter<double>("pre_grasp_distance", 0.08);
    cartesian_step_ = declare_parameter<double>("cartesian_step", 0.005);
    min_cartesian_fraction_ = declare_parameter<double>("cartesian_min_fraction", 0.95);
    low_z_threshold_ = declare_parameter<double>("low_center_z_threshold", 0.23);
    front_x_ = declare_parameter<double>("front_direction_x", 0.0);
    front_y_ = declare_parameter<double>("front_direction_y", -1.0);
    closing_local_axis_ = declare_parameter<std::string>(
      "gripper_closing_local_axis", "local_y");
    approach_z_sign_ = declare_parameter<double>("tcp_approach_z_sign", 1.0);
    shelf_shrink_x_ = declare_parameter<double>("shelf_shrink_x", 0.0);
    shelf_shrink_y_ = declare_parameter<double>("shelf_shrink_y", 0.0);
    shelf_shrink_z_ = declare_parameter<double>("shelf_shrink_z", 0.0);
    shelf_offset_x_ = declare_parameter<double>("shelf_offset_x", 0.0);
    shelf_offset_y_ = declare_parameter<double>("shelf_offset_y", 0.0);
    shelf_offset_z_ = declare_parameter<double>("shelf_offset_z", 0.0);

    if (closing_local_axis_ != "local_x" && closing_local_axis_ != "local_y") {
      throw std::invalid_argument(
        "gripper_closing_local_axis must be local_x or local_y");
    }
    if (std::abs(std::abs(approach_z_sign_) - 1.0) > 1.0e-6) {
      throw std::invalid_argument("tcp_approach_z_sign must be +1 or -1");
    }
    if (shelf_shrink_x_ < 0.0 || shelf_shrink_y_ < 0.0 || shelf_shrink_z_ < 0.0) {
      throw std::invalid_argument("shelf shrink values must be non-negative");
    }

    result_pub_ = create_publisher<PlannedGrasp>(output_topic_, 10);
    display_pub_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", 10);

    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group_;
    target_sub_ = create_subscription<GraspTarget>(
      input_topic_, 10,
      std::bind(&Cobot2MiNode::onTarget, this, std::placeholders::_1), options);

    RCLCPP_INFO(
      get_logger(),
      "cobot2_mi 시작 | input=%s | output=%s | eef=%s | "
      "closing_axis=%s | physical_approach=TCP_%sZ | low_z=%.3fm | "
      "shelf_shrink=(%.3f, %.3f, %.3f)m | shelf_offset=(%.3f, %.3f, %.3f)m",
      input_topic_.c_str(), output_topic_.c_str(), eef_link_.c_str(),
      closing_local_axis_.c_str(), approach_z_sign_ > 0.0 ? "+" : "-",
      low_z_threshold_, shelf_shrink_x_, shelf_shrink_y_, shelf_shrink_z_,
      shelf_offset_x_, shelf_offset_y_, shelf_offset_z_);
  }

  bool initialize()
  {
    try {
      move_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), planning_group_);
      move_group_->setEndEffectorLink(eef_link_);
      move_group_->setPlanningTime(planning_time_);
      move_group_->setNumPlanningAttempts(planning_attempts_);
      move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
      move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);
      move_group_->setStartStateToCurrentState();

      if (!loadShelf() || !applyShelf()) {
        return false;
      }

      ready_ = true;
      RCLCPP_INFO(
        get_logger(), "MoveIt 준비 완료 | frame=%s | eef=%s | shelf_boxes=%zu",
        move_group_->getPlanningFrame().c_str(), move_group_->getEndEffectorLink().c_str(),
        shelf_boxes_.size());
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "MoveIt 초기화 실패: %s", e.what());
      return false;
    }
  }

private:
  void onTarget(const GraspTarget::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(plan_mutex_);

    RCLCPP_INFO(
      get_logger(),
      "GraspTarget | id=%d | class=%s | center=(%.4f, %.4f, %.4f) | "
      "top_z=%.4f | size=(%.4f, %.4f, %.4f) | width=%.4f | "
      "closing=(%.4f, %.4f, %.4f) | depth=%.4f",
      msg->target_id, msg->class_name.c_str(), msg->center.x, msg->center.y, msg->center.z,
      msg->top_z, msg->dimensions.x, msg->dimensions.y, msg->dimensions.z,
      msg->required_width, msg->closing_axis.x, msg->closing_axis.y,
      msg->closing_axis.z, msg->grasp_depth);

    if (!ready_) {
      publishFailure(*msg, "MOVE_GROUP_NOT_READY");
      return;
    }

    std::string error;
    if (!validate(*msg, error)) {
      publishFailure(*msg, error);
      return;
    }

    PlanResult result;
    if (!planTarget(*msg, result)) {
      publishFailure(*msg, result.failure_reason);
      return;
    }

    publishDisplay(result);
    publishSuccess(*msg, result);
  }

  bool validate(const GraspTarget & target, std::string & error) const
  {
    if (!isFinite(target.center.x) || !isFinite(target.center.y) || !isFinite(target.center.z)) {
      error = "INVALID_CENTER";
      return false;
    }
    if (!isFinite(target.dimensions.x) || !isFinite(target.dimensions.y) ||
      !isFinite(target.dimensions.z) || target.dimensions.x <= 0.0 ||
      target.dimensions.y <= 0.0 || target.dimensions.z <= 0.0)
    {
      error = "INVALID_DIMENSIONS";
      return false;
    }
    if (!isFinite(target.top_z) || !isFinite(target.required_width) ||
      target.required_width <= 0.0 || !isFinite(target.grasp_depth) || target.grasp_depth < 0.0)
    {
      error = "INVALID_GRASP_VALUES";
      return false;
    }

    const Eigen::Vector3d closing(
      target.closing_axis.x, target.closing_axis.y, target.closing_axis.z);
    if (!closing.allFinite() || closing.norm() < kEps) {
      error = "INVALID_CLOSING_AXIS";
      return false;
    }

    const std::string frame = move_group_->getPlanningFrame();
    if (target.header.frame_id != frame) {
      error = "FRAME_MISMATCH:" + target.header.frame_id + "!=" + frame;
      return false;
    }

    if (workspace_loaded_) {
      const bool inside =
        target.center.x >= workspace_min_.x() && target.center.x <= workspace_max_.x() &&
        target.center.y >= workspace_min_.y() && target.center.y <= workspace_max_.y() &&
        target.top_z >= workspace_min_.z() && target.top_z <= workspace_max_.z();
      if (!inside) {
        error = "TARGET_OUTSIDE_WORKSPACE";
        return false;
      }
    }
    return true;
  }

  std::vector<Candidate> makeCandidates(const GraspTarget & target) const
  {
    std::vector<Candidate> result;
    int id = 0;
    auto add = [&](const std::string & type, Eigen::Vector3d approach, double sign) {
        if (approach.norm() < kEps) {
          return;
        }
        result.push_back({id++, type, approach.normalized(), sign});
      };

    Eigen::Vector3d front(front_x_, front_y_, 0.0);
    if (front.norm() < kEps) {
      RCLCPP_ERROR(get_logger(), "front_direction XY가 0입니다");
      return result;
    }
    front.normalize();
    const Eigen::Vector3d right(-front.y(), front.x(), 0.0);

    if (target.center.z < low_z_threshold_) {
      RCLCPP_WARN(
        get_logger(), "파지 모드 | center_z=%.4f < %.4f | LOW_HORIZONTAL_ONLY",
        target.center.z, low_z_threshold_);
      for (double yaw_deg : std::array<double, 3>{-20.0, 0.0, 20.0}) {
        const double yaw = rad(yaw_deg);
        Eigen::Vector3d approach = std::cos(yaw) * front + std::sin(yaw) * right;
        approach.z() = 0.0;
        add("LOW_HORIZONTAL_PCA_SHORT_AXIS", approach, 1.0);
        add("LOW_HORIZONTAL_PCA_SHORT_AXIS", approach, -1.0);
      }
    } else {
      RCLCPP_INFO(
        get_logger(), "파지 모드 | center_z=%.4f >= %.4f | NORMAL_FLEXIBLE",
        target.center.z, low_z_threshold_);

      // 0-degree TOP means a true vertical descent in base_link.
      // Only the two equivalent PCA-axis signs are tested.
      const Eigen::Vector3d vertical_down(0.0, 0.0, -1.0);
      add("TOP_VERTICAL_PCA_SHORT_AXIS", vertical_down, 1.0);
      add("TOP_VERTICAL_PCA_SHORT_AXIS", vertical_down, -1.0);
    }

    RCLCPP_INFO(get_logger(), "후보 생성 완료 | count=%zu", result.size());
    return result;
  }

  geometry_msgs::msg::Quaternion makeOrientation(
    const Eigen::Vector3d & approach,
    const Eigen::Vector3d & closing,
    Eigen::Vector3d & local_x,
    Eigen::Vector3d & local_y,
    Eigen::Vector3d & local_z) const
  {
    // sign=+1: TCP +Z points along pre-grasp -> grasp.
    // sign=-1: TCP -Z points along pre-grasp -> grasp.
    local_z = approach_z_sign_ * approach.normalized();
    Eigen::Vector3d projected = closing - closing.dot(local_z) * local_z;
    if (projected.norm() < kEps) {
      throw std::runtime_error("closing axis parallel to approach");
    }
    projected.normalize();

    if (closing_local_axis_ == "local_x") {
      local_x = projected;
      local_y = local_z.cross(local_x).normalized();
      local_x = local_y.cross(local_z).normalized();
    } else {
      local_y = projected;
      local_x = local_y.cross(local_z).normalized();
      local_y = local_z.cross(local_x).normalized();
    }

    Eigen::Matrix3d rotation;
    rotation.col(0) = local_x;
    rotation.col(1) = local_y;
    rotation.col(2) = local_z;
    Eigen::Quaterniond q(rotation);
    q.normalize();

    geometry_msgs::msg::Quaternion output;
    output.x = q.x();
    output.y = q.y();
    output.z = q.z();
    output.w = q.w();
    return output;
  }

  bool makePoses(
    const GraspTarget & target, const Candidate & candidate,
    geometry_msgs::msg::PoseStamped & pre_pose,
    geometry_msgs::msg::PoseStamped & grasp_pose,
    std::string & error) const
  {
    const Eigen::Vector3d approach = candidate.approach.normalized();
    const Eigen::Vector3d closing = candidate.closing_sign * Eigen::Vector3d(
      target.closing_axis.x, target.closing_axis.y, target.closing_axis.z);

    Eigen::Vector3d local_x, local_y, local_z;
    geometry_msgs::msg::Quaternion orientation;
    try {
      orientation = makeOrientation(approach, closing, local_x, local_y, local_z);
    } catch (const std::exception & e) {
      error = std::string("ORIENTATION_FAILED:") + e.what();
      return false;
    }

    const Eigen::Vector3d center(target.center.x, target.center.y, target.center.z);
    const Eigen::Vector3d size(
      target.dimensions.x, target.dimensions.y, target.dimensions.z);
    const double radius = 0.5 * (
      std::abs(approach.x()) * size.x() +
      std::abs(approach.y()) * size.y() +
      std::abs(approach.z()) * size.z());

    Eigen::Vector3d surface = center - approach * radius;
    if (approach.z() < -0.85) {
      surface = Eigen::Vector3d(target.center.x, target.center.y, target.top_z);
    }

    const Eigen::Vector3d grasp = surface + approach * target.grasp_depth;
    const Eigen::Vector3d pre = grasp - approach * pre_distance_;

    grasp_pose.header = target.header;
    grasp_pose.pose.position.x = grasp.x();
    grasp_pose.pose.position.y = grasp.y();
    grasp_pose.pose.position.z = grasp.z();
    grasp_pose.pose.orientation = orientation;

    pre_pose.header = target.header;
    pre_pose.pose.position.x = pre.x();
    pre_pose.pose.position.y = pre.y();
    pre_pose.pose.position.z = pre.z();
    pre_pose.pose.orientation = orientation;

    RCLCPP_INFO(
      get_logger(),
      "후보 자세 | id=%d | pre=(%.4f, %.4f, %.4f) | grasp=(%.4f, %.4f, %.4f) | "
      "TCP_X=(%.3f, %.3f, %.3f) | TCP_Y=(%.3f, %.3f, %.3f) | "
      "TCP_Z=(%.3f, %.3f, %.3f) | physical_approach=(%.3f, %.3f, %.3f)",
      candidate.id, pre.x(), pre.y(), pre.z(), grasp.x(), grasp.y(), grasp.z(),
      local_x.x(), local_x.y(), local_x.z(), local_y.x(), local_y.y(), local_y.z(),
      local_z.x(), local_z.y(), local_z.z(),
      approach.x(), approach.y(), approach.z());
    return true;
  }

  bool planTarget(const GraspTarget & target, PlanResult & output)
  {
    const auto candidates = makeCandidates(target);
    if (candidates.empty()) {
      output.failure_reason = "NO_CANDIDATE_GENERATED";
      return false;
    }

    std::string last_error = "NO_VALID_CANDIDATE";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const auto & candidate = candidates[i];
      RCLCPP_INFO(
        get_logger(),
        "후보 검사 %zu/%zu | id=%d | type=%s | approach=(%.3f, %.3f, %.3f) | sign=%+.0f",
        i + 1, candidates.size(), candidate.id, candidate.type.c_str(),
        candidate.approach.x(), candidate.approach.y(), candidate.approach.z(),
        candidate.closing_sign);

      PlanResult result;
      result.candidate_id = candidate.id;
      result.grasp_type = candidate.type;
      if (planCandidate(target, candidate, result)) {
        output = std::move(result);
        return true;
      }

      last_error = result.failure_reason.empty() ? "UNKNOWN_FAILURE" : result.failure_reason;
      RCLCPP_WARN(
        get_logger(), "후보 탈락 | id=%d | type=%s | reason=%s",
        candidate.id, candidate.type.c_str(), last_error.c_str());
    }

    output.failure_reason = "NO_VALID_CANDIDATE:last=" + last_error;
    return false;
  }

  bool planCandidate(
    const GraspTarget & target, const Candidate & candidate, PlanResult & result)
  {
    if (!makePoses(target, candidate, result.pre_pose, result.grasp_pose,
      result.failure_reason))
    {
      return false;
    }

    MoveGroupInterface::Plan pre_plan;
    if (!planPreGrasp(result.pre_pose, pre_plan, result.failure_reason)) {
      return false;
    }

    moveit::core::RobotStatePtr pre_state;
    if (!stateAtPlanEnd(pre_plan, pre_state, result.failure_reason)) {
      return false;
    }

    if (!planApproach(
      pre_state, result.grasp_pose, result.approach,
      result.cartesian_fraction, result.failure_reason))
    {
      return false;
    }

    result.to_pre = pre_plan.trajectory_;
    result.start_state = pre_plan.start_state_;
    RCLCPP_INFO(
      get_logger(), "후보 성공 | id=%d | type=%s | Cartesian=%.1f%%",
      result.candidate_id, result.grasp_type.c_str(), result.cartesian_fraction * 100.0);
    return true;
  }

  bool planPreGrasp(
    const geometry_msgs::msg::PoseStamped & pose,
    MoveGroupInterface::Plan & plan, std::string & error)
  {
    move_group_->stop();
    move_group_->clearPoseTargets();
    move_group_->setStartStateToCurrentState();
    if (!move_group_->setPoseTarget(pose.pose, eef_link_)) {
      error = "PREGRASP_POSE_TARGET_REJECTED";
      return false;
    }

    const auto code = move_group_->plan(plan);
    move_group_->clearPoseTargets();
    if (code != moveit::core::MoveItErrorCode::SUCCESS) {
      error = "PLAN_TO_PREGRASP_FAILED";
      return false;
    }
    if (plan.trajectory_.joint_trajectory.points.empty()) {
      error = "EMPTY_PREGRASP_TRAJECTORY";
      return false;
    }

    RCLCPP_INFO(
      get_logger(), "pre-grasp 계획 성공 | points=%zu",
      plan.trajectory_.joint_trajectory.points.size());
    return true;
  }

  bool stateAtPlanEnd(
    const MoveGroupInterface::Plan & plan,
    moveit::core::RobotStatePtr & state, std::string & error)
  {
    const auto & trajectory = plan.trajectory_.joint_trajectory;
    if (trajectory.joint_names.empty() || trajectory.points.empty() ||
      trajectory.points.back().positions.size() != trajectory.joint_names.size())
    {
      error = "INVALID_PREGRASP_TRAJECTORY";
      return false;
    }

    state = std::make_shared<moveit::core::RobotState>(move_group_->getRobotModel());
    state->setToDefaultValues();
    if (!plan.start_state_.joint_state.name.empty()) {
      moveit::core::robotStateMsgToRobotState(plan.start_state_, *state);
    }
    for (std::size_t i = 0; i < trajectory.joint_names.size(); ++i) {
      state->setVariablePosition(trajectory.joint_names[i], trajectory.points.back().positions[i]);
    }
    state->update();

    if (!state->satisfiesBounds()) {
      error = "PREGRASP_END_STATE_OUT_OF_BOUNDS";
      return false;
    }
    return true;
  }

  bool planApproach(
    const moveit::core::RobotStatePtr & start,
    const geometry_msgs::msg::PoseStamped & grasp_pose,
    moveit_msgs::msg::RobotTrajectory & trajectory,
    double & fraction, std::string & error)
  {
    move_group_->stop();
    move_group_->clearPoseTargets();
    move_group_->setStartState(*start);
    fraction = move_group_->computeCartesianPath(
      std::vector<geometry_msgs::msg::Pose>{grasp_pose.pose},
      cartesian_step_, 0.0, trajectory, true);
    move_group_->setStartStateToCurrentState();

    RCLCPP_INFO(
      get_logger(), "Cartesian 접근 | fraction=%.1f%% | min=%.1f%% | points=%zu",
      fraction * 100.0, min_cartesian_fraction_ * 100.0,
      trajectory.joint_trajectory.points.size());

    if (!isFinite(fraction) || fraction < min_cartesian_fraction_) {
      std::ostringstream stream;
      stream << "CARTESIAN_APPROACH_FAILED:" << fraction;
      error = stream.str();
      return false;
    }
    if (trajectory.joint_trajectory.points.empty()) {
      error = "EMPTY_APPROACH_TRAJECTORY";
      return false;
    }
    return true;
  }

  bool loadShelf()
  {
    if (shelf_yaml_.empty()) {
      RCLCPP_ERROR(get_logger(), "shelf_yaml parameter is empty");
      return false;
    }

    try {
      const YAML::Node root = YAML::LoadFile(shelf_yaml_);
      shelf_frame_ = root["frame_id"] ? root["frame_id"].as<std::string>() :
        move_group_->getPlanningFrame();

      if (root["workspace"]) {
        const auto w = root["workspace"];
        workspace_min_ = Eigen::Vector3d(
          w["x_min"].as<double>(), w["y_min"].as<double>(), w["z_min"].as<double>());
        workspace_max_ = Eigen::Vector3d(
          w["x_max"].as<double>(), w["y_max"].as<double>(), w["z_max"].as<double>());
        workspace_loaded_ = true;
        move_group_->setWorkspace(
          workspace_min_.x(), workspace_min_.y(), workspace_min_.z(),
          workspace_max_.x(), workspace_max_.y(), workspace_max_.z());
        RCLCPP_INFO(
          get_logger(), "workspace | min=(%.3f, %.3f, %.3f) | max=(%.3f, %.3f, %.3f)",
          workspace_min_.x(), workspace_min_.y(), workspace_min_.z(),
          workspace_max_.x(), workspace_max_.y(), workspace_max_.z());
      }

      const YAML::Node objects = root["collision_objects"];
      if (!objects || !objects.IsSequence()) {
        throw std::runtime_error("collision_objects missing");
      }

      shelf_boxes_.clear();
      for (const auto & node : objects) {
        if (!node["id"] || !node["center"] || !node["size"] ||
          node["center"].size() != 3 || node["size"].size() != 3)
        {
          throw std::runtime_error("invalid collision object");
        }
        ShelfBox box;
        box.id = node["id"].as<std::string>();
        box.center = Eigen::Vector3d(
          node["center"][0].as<double>() + shelf_offset_x_,
          node["center"][1].as<double>() + shelf_offset_y_,
          node["center"][2].as<double>() + shelf_offset_z_);
        box.size = Eigen::Vector3d(
          node["size"][0].as<double>() - shelf_shrink_x_,
          node["size"][1].as<double>() - shelf_shrink_y_,
          node["size"][2].as<double>() - shelf_shrink_z_);
        if ((box.size.array() <= 0.0).any()) {
          throw std::runtime_error("shelf shrink made size non-positive: " + box.id);
        }
        shelf_boxes_.push_back(box);
      }

      RCLCPP_INFO(
        get_logger(), "shelf yaml 로드 | path=%s | frame=%s | boxes=%zu",
        shelf_yaml_.c_str(), shelf_frame_.c_str(), shelf_boxes_.size());
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "shelf yaml 오류: %s", e.what());
      return false;
    }
  }

  bool applyShelf()
  {
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    for (const auto & box : shelf_boxes_) {
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = shelf_frame_;
      object.id = box.id;
      object.operation = moveit_msgs::msg::CollisionObject::ADD;

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
      primitive.dimensions = {box.size.x(), box.size.y(), box.size.z()};

      geometry_msgs::msg::Pose pose;
      pose.position.x = box.center.x();
      pose.position.y = box.center.y();
      pose.position.z = box.center.z();
      pose.orientation.w = 1.0;
      object.primitives = {primitive};
      object.primitive_poses = {pose};
      objects.push_back(object);
    }

    if (!planning_scene_.applyCollisionObjects(objects)) {
      RCLCPP_ERROR(get_logger(), "선반 collision object 적용 실패");
      return false;
    }
    RCLCPP_INFO(get_logger(), "선반 collision object 적용 요청 완료: %zu개", objects.size());
    return true;
  }

  void publishDisplay(const PlanResult & result)
  {
    moveit_msgs::msg::DisplayTrajectory display;
    display.model_id = move_group_->getRobotModel()->getName();
    display.trajectory_start = result.start_state;
    display.trajectory = {result.to_pre, result.approach};
    display_pub_->publish(display);
  }

  void publishSuccess(const GraspTarget & target, const PlanResult & result)
  {
    PlannedGrasp out;
    out.header = target.header;
    out.header.stamp = now();
    out.success = true;
    out.target_id = target.target_id;
    out.candidate_id = result.candidate_id;
    out.class_name = target.class_name;
    out.grasp_type = result.grasp_type;
    out.pre_grasp_pose = result.pre_pose;
    out.grasp_pose = result.grasp_pose;
    out.to_pre_grasp = result.to_pre;
    out.approach = result.approach;
    out.cartesian_fraction = static_cast<float>(result.cartesian_fraction);
    out.required_width = target.required_width;
    out.closing_axis = target.closing_axis;
    out.grasp_depth = target.grasp_depth;
    out.score = static_cast<float>(result.cartesian_fraction);
    result_pub_->publish(out);

    RCLCPP_INFO(
      get_logger(),
      "PlannedGrasp 성공 | target=%d | candidate=%d | class=%s | type=%s | Cartesian=%.1f%%",
      out.target_id, out.candidate_id, out.class_name.c_str(), out.grasp_type.c_str(),
      out.cartesian_fraction * 100.0);
  }

  void publishFailure(const GraspTarget & target, const std::string & reason)
  {
    PlannedGrasp out;
    out.header = target.header;
    out.header.stamp = now();
    out.success = false;
    out.failure_reason = reason;
    out.target_id = target.target_id;
    out.candidate_id = -1;
    out.class_name = target.class_name;
    out.required_width = target.required_width;
    out.closing_axis = target.closing_axis;
    out.grasp_depth = target.grasp_depth;
    result_pub_->publish(out);
    RCLCPP_ERROR(
      get_logger(), "PlannedGrasp 실패 | target=%d | class=%s | reason=%s",
      out.target_id, out.class_name.c_str(), reason.c_str());
  }

  rclcpp::Subscription<GraspTarget>::SharedPtr target_sub_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Publisher<PlannedGrasp>::SharedPtr result_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;
  std::shared_ptr<MoveGroupInterface> move_group_;
  PlanningSceneInterface planning_scene_;
  std::mutex plan_mutex_;
  bool ready_ {false};

  std::string planning_group_;
  std::string eef_link_;
  std::string shelf_yaml_;
  std::string input_topic_;
  std::string output_topic_;
  std::string shelf_frame_ {"base_link"};

  double planning_time_ {3.0};
  int planning_attempts_ {3};
  double velocity_scaling_ {0.5};
  double acceleration_scaling_ {0.5};
  double pre_distance_ {0.08};
  double cartesian_step_ {0.005};
  double min_cartesian_fraction_ {0.95};
  double low_z_threshold_ {0.23};
  double front_x_ {0.0};
  double front_y_ {-1.0};
  std::string closing_local_axis_ {"local_y"};
  double approach_z_sign_ {1.0};
  double shelf_shrink_x_ {0.0};
  double shelf_shrink_y_ {0.0};
  double shelf_shrink_z_ {0.0};
  double shelf_offset_x_ {0.0};
  double shelf_offset_y_ {0.0};
  double shelf_offset_z_ {0.0};

  bool workspace_loaded_ {false};
  Eigen::Vector3d workspace_min_ {Eigen::Vector3d::Zero()};
  Eigen::Vector3d workspace_max_ {Eigen::Vector3d::Zero()};
  std::vector<ShelfBox> shelf_boxes_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Cobot2MiNode>();
  if (!node->initialize()) {
    RCLCPP_ERROR(node->get_logger(), "MoveIt initialization failed");
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}