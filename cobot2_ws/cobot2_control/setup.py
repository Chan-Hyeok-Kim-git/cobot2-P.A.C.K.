from setuptools import find_packages, setup

package_name = 'cobot2_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='swyc',
    maintainer_email='a01048068152@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cobot2_grasp = cobot2_control.cobot2_grasp:main',
            'cobot2_move = cobot2_control.cobot2_move:main',
            "fake_grasp_cloud_publisher = cobot2_control.fake_grasp_cloud_publisher:main",
            "onrobot = cobot2_control.cobot2_move_modbus:main",
            "fake_motion_plan_publisher = cobot2_control.fake_motion_plan_publisher:main",
            "cobot2_task_manager = cobot2_control.cobot2_task_manager:main",
            
        ],
    },
)
