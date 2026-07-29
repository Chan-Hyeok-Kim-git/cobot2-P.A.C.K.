import os
from glob import glob
from setuptools import find_packages, setup

package_name = "ai_frame_transform"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Transform AI point clouds and position_camera_xyz_m JSON into base_link.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ai_frame_transform_node = "
            "ai_frame_transform.ai_frame_transform_node:main",
        ],
    },
)
