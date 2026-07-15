from setuptools import find_packages
from setuptools import setup

setup(
    name='mujoco_sim_msgs',
    version='0.1.0',
    packages=find_packages(
        include=('mujoco_sim_msgs', 'mujoco_sim_msgs.*')),
)
