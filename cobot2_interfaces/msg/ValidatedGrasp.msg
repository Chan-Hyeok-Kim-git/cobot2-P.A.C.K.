std_msgs/Header header

int32 candidate_id
string grasp_type

geometry_msgs/PoseStamped pre_grasp_pose
geometry_msgs/PoseStamped grasp_pose

float64[6] pre_grasp_joints
float64[6] grasp_joints

float32 required_width
float32 local_score