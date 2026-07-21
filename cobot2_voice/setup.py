from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'cobot2_voice'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # resource 폴더 내의 모델(.tflite) 및 .env 등 실제 필요한 파일만 설치 경로로 복사
        (os.path.join('share', package_name, 'resource'), [
            'resource/soundclassifier_with_metadata.tflite',  # 사용하는 모델 파일 명시
            'resource/.env',                         # .env 파일 명시
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
    entry_points={
        'console_scripts': [
            'voice_command = cobot2_voice.voice_command:main',
        ],
    },
)
