"""Extracted model_21800 Isaac executor with a physical-command-only boundary.

This module must be imported after Isaac's application has been launched.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

WBT_ROOT=Path("/home/a104/hope_training_repo/hope-model21800-isaac/hope_training/whole_body_tracking")
sys.path.insert(0,str(WBT_ROOT))

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import AgibotA3TableTennisEnvCfg
from training.tasks.table_tennis import geometry
from training.robots.agibot_a3 import AGIBOT_A3_JOINT_NAMES

DEPLOY_ROOT=Path("/home/a104/hope_training_repo/hope-deploy-baseline/a3_deploy/a3_deploy_example")
sys.path.insert(0,str(DEPLOY_ROOT/"reference"))
sys.path.insert(0,"/home/a104/hope_training_repo/hope-model21800-isaac/.local_deps")
from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter
from a3_deploy_onnx_ref_pingpong.lifecycle import LifecycleConfig, SwingLifecycle
from a3_deploy_onnx_ref_pingpong.observation import RobotState, build_observation
from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
from a3_deploy_onnx_ref_pingpong.racket_command import RacketCommand

MODEL=DEPLOY_ROOT/"models/model_21800/policy/exported/policy.onnx"
DEPLOY_YAML=DEPLOY_ROOT/"models/model_21800/policy/params/deploy.yaml"
RUNTIME_YAML=DEPLOY_ROOT/"config/hope_pingpong_runtime.yaml"
PHYSICS_DT=1/400
DECIMATION=8
CONTROL_DT=PHYSICS_DT*DECIMATION

@dataclass
class ExecutionResult:
    phases: set
    inference_calls: int
    nonfinite_count: int
    min_base_height: float
    max_base_tilt_deg: float
    max_root_xy_drift: float
    stable: bool
    failure_reason: str=""

class Model21800IsaacExecutor:
    """Consumes only world position/velocity, TTS and side."""
    def __init__(self, device="cuda:0"):
        with open(RUNTIME_YAML,encoding="utf-8") as f: runtime=yaml.safe_load(f)
        lc=runtime["lifecycle"]
        self.lifecycle_cfg=LifecycleConfig(dt=CONTROL_DT,follow_through_s=float(lc["follow_through_s"]),recovery_s=float(lc["recovery_s"]),ready_time_to_strike=float(lc["ready_time_to_strike"]),ready_reach_x=float(lc["ready_reach_x"]),ready_reach_y=float(lc["ready_reach_y"]),ready_reach_z=float(lc["ready_reach_z"]))
        self.policy=OnnxPolicy(MODEL,providers=["CPUExecutionProvider"])
        self.adapter=ActionAdapter.from_yaml(DEPLOY_YAML)
        cfg=AgibotA3TableTennisEnvCfg(); cfg.scene.num_envs=1; cfg.scene.env_spacing=4.; cfg.scene.robot.spawn.fix_base=False
        cfg.sim.dt=PHYSICS_DT; cfg.sim.render_interval=1; cfg.sim.device=device; cfg.sim.physx.enable_ccd=True
        self.sim=sim_utils.SimulationContext(cfg.sim); self.scene=InteractiveScene(cfg.scene); self.sim.reset(); self.scene.reset()
        self.robot=self.scene["robot"]
        ids,names=self.robot.find_joints(list(AGIBOT_A3_JOINT_NAMES),preserve_order=True)
        if len(ids)!=31 or tuple(names)!=tuple(AGIBOT_A3_JOINT_NAMES): raise RuntimeError("canonical joint order failure")
        self.joint_ids=ids; self.device=self.robot.device; self.dtype=self.robot.data.joint_pos.dtype
        self.q0=torch.as_tensor(self.adapter.default_q,device=self.device,dtype=self.dtype).reshape(1,31)
        self.qd0=torch.zeros_like(self.q0); self.root0=self.robot.data.default_root_state.clone(); self.root0[:,:3]+=self.scene.env_origins
        self.task_id=0; self.reset()

    def reset(self):
        self.robot.write_root_pose_to_sim(self.root0[:,:7]); self.robot.write_root_velocity_to_sim(self.root0[:,7:])
        self.robot.write_joint_state_to_sim(self.q0,self.qd0,joint_ids=self.joint_ids); self.robot.set_joint_position_target(self.q0,joint_ids=self.joint_ids)
        self.lifecycle=SwingLifecycle(self.lifecycle_cfg); self.last_action=np.zeros(31,np.float32); self.target=None
        # Exact Stage5 reset parity: settle for 0.20 s (80 physics steps),
        # writing scene state before every step.
        for _ in range(int(round(0.20 / PHYSICS_DT))):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
        self.base_target_xy=self.robot.data.root_pos_w[0,:2].detach().cpu().numpy().astype(np.float64)
        self.initial_base=self.robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)

    def set_target_command(self,position_world,velocity_world,time_to_strike_s,swing_sign):
        p=np.asarray(position_world,dtype=np.float64); v=np.asarray(velocity_world,dtype=np.float64)
        if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(np.r_[p,v,time_to_strike_s]).all() or time_to_strike_s<=0 or swing_sign not in (-1,1): raise ValueError("invalid target command")
        self.task_id+=1; self.target=(p,v,float(time_to_strike_s),int(swing_sign))

    def _state(self):
        r=self.robot.data
        return RobotState(r.root_pos_w[0].detach().cpu().numpy(),r.root_quat_w[0].detach().cpu().numpy(),r.root_ang_vel_b[0].detach().cpu().numpy(),r.joint_pos[0,self.joint_ids].detach().cpu().numpy(),r.joint_vel[0,self.joint_ids].detach().cpu().numpy())

    def run_full_lifecycle(self):
        if self.target is None: raise RuntimeError("target not set")
        p,v,tts,sign=self.target; phases={"ready"}; calls=0; bad=0; min_z=1e9; max_tilt=0.; max_drift=0.; tick=0
        while tick<300:
            state=self._state(); phase_before=self.lifecycle.phase.value
            cmd=RacketCommand(task_id=self.task_id,task_revision=tick,swing_sign=sign,position=p,velocity=v,time_to_strike=max(tts-tick*CONTROL_DT,1e-6)) if tick==0 else None
            target=self.lifecycle.update(cmd,state); phases.add(self.lifecycle.phase.value)
            obs=build_observation(state,target,self.last_action,self.adapter.default_q,self.base_target_xy)
            if obs.shape!=(110,) or not np.isfinite(obs).all(): bad+=1; break
            raw=np.asarray(self.policy.infer_target(obs,target.time_to_strike,self.lifecycle.swing_sign,CONTROL_DT),dtype=np.float32).reshape(31); calls+=1
            raw[3:5]=0.; q=self.adapter.decode(raw); q[3:5]=self.adapter.default_q[3:5]
            if not np.isfinite(np.r_[raw,q]).all(): bad+=1; break
            self.robot.set_joint_position_target(torch.as_tensor(q,device=self.device,dtype=self.dtype).reshape(1,31),joint_ids=self.joint_ids); self.last_action=raw.copy()
            for _ in range(DECIMATION): self.scene.write_data_to_sim(); self.sim.step(render=False); self.scene.update(PHYSICS_DT)
            pos=state.base_pos_w; min_z=min(min_z,float(pos[2])); max_drift=max(max_drift,float(np.linalg.norm(pos[:2]-self.initial_base[:2])))
            g=obs[96:99]; max_tilt=max(max_tilt,math.degrees(math.acos(float(np.clip(-g[2],-1,1)))))
            tick+=1
            if self.lifecycle.phase.value=="ready" and "recovery" in phases: break
        # Stage5 measures pelvis clearance above the HOPE floor, not raw
        # simulator world-Z. Keep its exact engineering sanity thresholds.
        min_root_clearance = min_z - float(geometry.FLOOR_Z)
        stable=(bad==0 and min_root_clearance>0.75 and max_tilt<35.0
                and max_drift<0.30 and {"ready","swing","follow_through","recovery"}<=phases)
        reason="" if stable else (f"phases={sorted(phases)}, bad={bad}, world_z={min_z:.3f}, "
                                  f"floor_clearance={min_root_clearance:.3f}, tilt={max_tilt:.2f}, drift={max_drift:.3f}")
        return ExecutionResult(phases,calls,bad,min_z,max_tilt,max_drift,stable,reason)
