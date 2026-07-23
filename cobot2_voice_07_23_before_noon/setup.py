# 이 파일은 ROS 2가 이 Python 패키지를 설치하고 실행 명령을 만들 때 읽는다.
from setuptools import find_packages, setup
from glob import glob
import os

# 패키지 이름은 package.xml의 name과 같아야 한다.
package_name = 'cobot2_voice'

setup(
    name=package_name,
    version='0.0.0',
    # cobot2_voice 폴더 안의 Python module들을 설치 대상에 넣는다.
    packages=find_packages(exclude=['test']),
    data_files=[
        # ROS 2가 이 패키지를 찾을 수 있도록 resource 표시 파일을 설치한다.
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # ROS 2 메타데이터가 담긴 package.xml도 함께 설치한다.
        ('share/' + package_name, ['package.xml']),

        # resource 폴더 내의 모델(.tflite) 및 .env 등 실제 필요한 파일만 설치 경로로 복사
        (os.path.join('share', package_name, 'resource'), ["resource/soundclassifier_with_metadata.tflite",
            "resource/hello_rokey_8332_32.tflite",
            "resource/.env",
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    # ros2 run cobot2_voice voice_command 명령이 호출할 Python 함수이다.
    entry_points={
        'console_scripts': [
            'voice_command = cobot2_voice.voice_command:main',
        ],
    },
)
