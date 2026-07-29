std_msgs/Header header

int32 target_id
string class_name

# 선택된 object point-cloud 클러스터의 robust 3D 중심 [m]
geometry_msgs/Point center

# robust AABB 크기 [m]
geometry_msgs/Vector3 dimensions

# point cloud의 robust 상단 높이 [m]
float32 top_z

# PCA 단축으로 측정한 물체 폭 [m]
float32 required_width

# base_link 기준 PCA 단축 단위벡터.
geometry_msgs/Vector3 closing_axis

# top_z에서 물체 안쪽으로 내려갈 깊이 [m]
float32 grasp_depth

int32 shelf_level





