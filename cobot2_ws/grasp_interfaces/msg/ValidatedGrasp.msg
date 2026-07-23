uint16 VALID=0
uint16 WIDTH_TOO_LARGE=1
uint16 WIDTH_TOO_SMALL=2
uint16 CONTACT_MISSING=3
uint16 FINGER_COLLISION=4
uint16 BODY_COLLISION=5
uint16 SHELF_COLLISION=6
uint16 APPROACH_COLLISION=7
uint16 IK_FAILED=8
uint16 SINGULARITY=9
uint16 JOINT_JUMP=10

GraspCandidate candidate

bool valid
uint16 failure_code
string failure_reason

float32 measured_width
float32 left_contact_score
float32 right_contact_score
float32 min_clearance
uint32 collision_count
float32 local_score

bool ik_checked
sensor_msgs/JointState pre_grasp_joint_state
sensor_msgs/JointState grasp_joint_state
