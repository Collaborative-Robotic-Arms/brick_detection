from setuptools import find_packages, setup

package_name = 'brick_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tarek',
    maintainer_email='tarek@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		# 'brick_detector_node = brick_detection.brick_detector_node:main',
		'yolo_detector = brick_detection.yolo_detector_node:main',
        ],
    },
)
