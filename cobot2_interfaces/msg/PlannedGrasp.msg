std_msgs/Header header

bool success
string failure_reason

int32 target_id
int32 candidate_id
string class_name
string grasp_type

geometry_msgs/PoseStamped pre_grasp_pose
geometry_msgs/PoseStamped grasp_pose

moveit_msgs/RobotTrajectory to_pre_grasp
moveit_msgs/RobotTrajectory approach

float32 cartesian_fraction

# PCA 단축 폭 [m]
float32 required_width

# perception에서 받은 PCA 단축
geometry_msgs/Vector3 closing_axis

# top_z 아래로 내려가는 깊이 [m]
float32 grasp_depth

float32 score
