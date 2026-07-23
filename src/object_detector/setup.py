from glob import glob

from setuptools import find_packages, setup


package_name = "object_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml") + glob("config/*.rviz"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Rokey Team",
    maintainer_email="rokey@example.com",
    description="Real-time YOLO detector for ROS 2 camera images",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "yolo_detector = object_detector.yolo_detector_node:main",
            "yolo_pointcloud = object_detector.yolo_pointcloud_node:main",
            "yolo_pointcloud_mp = object_detector.yolo_sam_pointcloud_mp_node:main",
            "yolo_seg_pointcloud_mp = object_detector.yolo_seg_pointcloud_mp_node:main",
        ],
    },
)
