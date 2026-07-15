from setuptools import find_packages
from setuptools import setup

setup(
    name='joint_msgs',
    version='0.1.0',
    packages=find_packages(
        include=('joint_msgs', 'joint_msgs.*')),
)
