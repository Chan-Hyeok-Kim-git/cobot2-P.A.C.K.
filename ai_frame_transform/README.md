# ai_frame_transform

ROS 2 Humble package that transforms:

- `/ai/background_points` (`sensor_msgs/msg/PointCloud2`)
- `/ai/object_points` (`sensor_msgs/msg/PointCloud2`)
- `/ai/objects_3d/json` (`std_msgs/msg/String`)

into `base_link` by default.

The JSON parser explicitly reads each object's:

```json
"position_camera_xyz_m": [x, y, z]
```

and adds:

```json
"position_base_xyz_m": [x, y, z]
```

The original camera coordinate is preserved.

## Build

```bash
cd ~/cobot_ws
colcon build --packages-select ai_frame_transform
source install/setup.bash
```

## Run

```bash
ros2 launch ai_frame_transform ai_frame_transform.launch.py
```

Before running, this must succeed:

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```
