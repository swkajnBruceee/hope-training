"""A3 embodiment binding for the isolated locomotion replica."""

from isaaclab.utils import configclass

from training.robots.agibot_a3 import AGIBOT_A3_CFG
from training.tasks.replica_locomotion.a3_replica_env_cfg import A3BaseStandReplicaEnvCfg as BaseReplicaEnvCfg


@configclass
class A3BaseStandReplicaEnvCfg(BaseReplicaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = AGIBOT_A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
