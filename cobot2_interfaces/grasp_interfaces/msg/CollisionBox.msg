uint8 TYPE_DYNAMIC=0
uint8 TYPE_FIXED=1
uint8 TYPE_TARGET=2

std_msgs/Header header
int64 scene_id

string object_id
uint8 object_type

geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions

float32 confidence
float32 safety_margin
