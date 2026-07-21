from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'grasp_local_validation'

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
    description='재난 비상용품 그립 후보 생성 및 로컬(점군 기반) 검증 노드',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'local_validator_node = grasp_local_validation.local_validator_node:main',
            'candidate_generator_node = grasp_local_validation.candidate_generator_node:main',
            'mock_scene_publisher = grasp_local_validation.mock_scene_publisher:main',
        ],
    },
)
