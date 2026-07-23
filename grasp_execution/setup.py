from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'grasp_execution'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='swyc',
    maintainer_email='a01048068152@gmail.com',
    description='로컬검증 -> MoveIt검증/계획 -> 실제 로봇 실행 -> 그리퍼 제어를 잇는 실행 조정 노드',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'execution_coordinator_node = grasp_execution.execution_coordinator_node:main',
        ],
    },
)
