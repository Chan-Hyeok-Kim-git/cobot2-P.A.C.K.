uint8 GRASP_TOP=0
uint8 GRASP_FRONT=1
uint8 GRASP_SIDE=2

std_msgs/Header header
int64 scene_id
int32 candidate_id

uint8 grasp_type

geometry_msgs/Pose grasp_pose
geometry_msgs/Pose pre_grasp_pose

geometry_msgs/Vector3 approach_direction
geometry_msgs/Vector3 closing_direction
geometry_msgs/Vector3 finger_direction

float32 required_width
float32 approach_distance
float32 initial_score
