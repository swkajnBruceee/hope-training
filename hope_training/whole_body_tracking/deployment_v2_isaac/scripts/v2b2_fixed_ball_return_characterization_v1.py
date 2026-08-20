"""V2-B2 deterministic post-contact return characterization; no learning/search."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]; WBT_ROOT=ROOT/"hope_training"/"whole_body_tracking"
sys.path[:0]=[str(ROOT),str(WBT_ROOT),str(ROOT/".local_deps")]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args=p.parse_args(); app=AppLauncher(args).app

import torch
from isaaclab.utils.math import quat_apply, quat_rotate_inverse
from training.robots.agibot_a3 import A3_MOUNT_OFFSET, A3_WRIST_BODY
from training.tasks.table_tennis import geometry
from training.tasks.table_tennis.ball import BallAerodynamicsCfg, compute_aero_wrench
from hope_training.whole_body_tracking.deployment_v2_isaac.fixed_ball_smoke import BALL_INITIAL_POSITION,BALL_INITIAL_VELOCITY,BALL_INITIAL_SPIN,predict_fixed_intercept
from hope_training.whole_body_tracking.deployment_v2_isaac.low_level_isaac_executor import Model21800IsaacExecutor,CONTROL_DT,PHYSICS_DT,DECIMATION,RacketCommand
from hope_training.whole_body_tracking.deployment_v2_isaac.predictor_observation import PredictorCommand
from hope_training.whole_body_tracking.deployment_v2_isaac.return_metrics import detect_opponent_table_bounce,detect_outgoing_net_crossing,legal_return
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import OneDecisionCommandAssembler

LEAD_TIME=1.20; ACTION=(0.,0.,0.); EXPECTED_SIDE=-1
CONTACT_FORCE_THRESHOLD=.05; FACE_LATERAL_THRESHOLD=.10; FACE_NORMAL_THRESHOLD=.10

def main():
    print("B1_PHASE=EXECUTOR_CREATE_BEGIN",flush=True); ex=Model21800IsaacExecutor(device=args.device)
    print("B1_PHASE=EXECUTOR_CREATED",flush=True); ball=ex.scene["ball"]; sensor=ex.scene.sensors["racket_ball_contact"]
    wrist_ids,_=ex.robot.find_bodies(A3_WRIST_BODY,preserve_order=True); wrist_id=int(wrist_ids[0]); mount=torch.tensor(A3_MOUNT_OFFSET,device=ex.device)
    print("B1_PHASE=BALL_SENSOR_RESOLVED",flush=True)
    aero_cfg=BallAerodynamicsCfg(); force_b=torch.zeros((1,1,3),device=ball.device); torque_b=torch.zeros_like(force_b)
    def aero(_dt):
        fw,tw=compute_aero_wrench(ball.data.root_lin_vel_w,ball.data.root_ang_vel_w,float(geometry.BALL_MASS),aero_cfg)
        force_b[:,0]=quat_rotate_inverse(ball.data.root_quat_w,fw); torque_b[:,0]=quat_rotate_inverse(ball.data.root_quat_w,tw)
        ball.set_external_force_and_torque(force_b,torque_b); ball.write_data_to_sim()
    ex.sim.add_physics_callback("v2b2_hope_ball_aerodynamics",aero); print("B1_PHASE=AERO_CALLBACK_REGISTERED",flush=True)
    pred=predict_fixed_intercept(); launch_delay=LEAD_TIME-pred.flight_time_s; print("B1_PHASE=PREDICTOR_COMPLETE",flush=True)
    assembler=OneDecisionCommandAssembler(); pc=PredictorCommand(tuple(pred.position),tuple(pred.velocity),10.,LEAD_TIME)
    built=assembler.build(flight_id=1,revision_id=1,command_seq=1,predictor_command=pc,source_now_s=10.,producer_wall_s=100.,current_base_xy=ex.initial_base[:2],normalized_action=ACTION)
    assert built.swing_sign==EXPECTED_SIDE and np.allclose(built.velocity_world,[2.035,.055,.86])
    ex.set_target_command(built.position_world,built.velocity_world,built.control_tts_s,built.swing_sign); print("B1_PHASE=COMMAND_BUILT",flush=True)
    pose=torch.zeros((1,7),device=ball.device); pose[:,3]=1.; vel=torch.zeros((1,6),device=ball.device)
    def set_ball(position,linear,angular=(0.,0.,0.)):
        pose[:,:3]=ex.scene.env_origins+torch.tensor(position,device=ball.device); vel[:,:3]=torch.tensor(linear,device=ball.device); vel[:,3:]=torch.tensor(angular,device=ball.device)
        ball.write_root_pose_to_sim(pose); ball.write_root_velocity_to_sim(vel)
    parked=(2.22,-.5375,.43); set_ball(parked,(0,0,0)); print("B1_PHASE=BALL_PARKED",flush=True)
    phases={"ready"}; calls=bad=0; launched=contact=net_seen=cross_net=bounce=False; contact_data={}; net_data={}; bounce_data={}
    min_world_z=float("inf"); max_tilt=max_drift=max_force=0.; previous_pos=previous_vel=None
    print("B1_PHASE=CONTROL_LOOP_ENTER",flush=True)
    for tick in range(220):
        if tick==0: print("B1_PHASE=FIRST_CONTROL_TICK",flush=True)
        elapsed=tick*CONTROL_DT; state=ex._state(); cmd=RacketCommand(task_id=ex.task_id,task_revision=0,swing_sign=built.swing_sign,position=built.position_world,velocity=built.velocity_world,time_to_strike=built.control_tts_s) if tick==0 else None
        target=ex.lifecycle.update(cmd,state); phases.add(ex.lifecycle.phase.value)
        from a3_deploy_onnx_ref_pingpong.observation import build_observation
        obs=build_observation(state,target,ex.last_action,ex.adapter.default_q,ex.base_target_xy)
        min_world_z=min(min_world_z,float(state.base_pos_w[2])); max_drift=max(max_drift,float(np.linalg.norm(state.base_pos_w[:2]-ex.initial_base[:2]))); max_tilt=max(max_tilt,math.degrees(math.acos(float(np.clip(-obs[98],-1,1)))))
        raw=np.asarray(ex.policy.infer_target(obs,target.time_to_strike,ex.lifecycle.swing_sign,CONTROL_DT),dtype=np.float32).reshape(31); calls+=1; raw[3:5]=0
        q=ex.adapter.decode(raw); q[3:5]=ex.adapter.default_q[3:5]; ex.robot.set_joint_position_target(torch.tensor(q,device=ex.device,dtype=ex.dtype).reshape(1,31),joint_ids=ex.joint_ids); ex.last_action=raw.copy()
        if not np.isfinite(np.r_[obs,raw,q]).all(): bad+=1; break
        for sub in range(DECIMATION):
            now=elapsed+sub*PHYSICS_DT
            if not launched and now+1e-12>=launch_delay: set_ball(BALL_INITIAL_POSITION,BALL_INITIAL_VELOCITY,BALL_INITIAL_SPIN); launched=True; print("B1_PHASE=BALL_LAUNCHED",flush=True)
            elif not launched: set_ball(parked,(0,0,0))
            ex.scene.write_data_to_sim();
            if tick==0 and sub==0: print("B1_PHASE=FIRST_PHYSICS_STEP",flush=True)
            ex.sim.step(render=False); ex.scene.update(PHYSICS_DT)
            bp=ball.data.root_pos_w[0]; bv=ball.data.root_lin_vel_w[0]; local_p=(bp-ex.scene.env_origins[0]).detach().cpu().numpy(); local_v=bv.detach().cpu().numpy()
            wpos=ex.robot.data.body_pos_w[0,wrist_id]; wquat=ex.robot.data.body_quat_w[0,wrist_id]; rp=wpos+quat_apply(wquat,mount)
            wrist_v=ex.robot.data.body_lin_vel_w[0,wrist_id]; wrist_w=ex.robot.data.body_ang_vel_w[0,wrist_id]; racket_v=wrist_v+torch.linalg.cross(wrist_w,quat_apply(wquat,mount))
            normal_local=torch.tensor([0.,1.,0.],device=ex.device); normal=quat_apply(wquat,normal_local); delta=bp-rp; signed=torch.dot(delta,normal); lateral=torch.linalg.vector_norm(delta-signed*normal)
            force=float(torch.linalg.vector_norm(sensor.data.net_forces_w,dim=-1).reshape(1,-1).amax(dim=1)[0]); max_force=max(max_force,force)
            true_contact=force>CONTACT_FORCE_THRESHOLD and float(lateral)<FACE_LATERAL_THRESHOLD and abs(float(signed))<FACE_NORMAL_THRESHOLD
            if true_contact and not contact:
                contact=True; contact_data={"time":now,"ball_position":local_p.tolist(),"ball_velocity_pre":None if previous_vel is None else previous_vel.tolist(),"ball_velocity_post":local_v.tolist(),"racket_position":(rp-ex.scene.env_origins[0]).detach().cpu().tolist(),"racket_velocity":racket_v.detach().cpu().tolist()}; print("B2_PHASE=CONTACT_DETECTED",flush=True)
            if previous_pos is not None:
                event=detect_outgoing_net_crossing(previous_pos,local_p,local_v,contact_seen=contact,net_seen=net_seen)
                if event is not None:
                    net_seen=True; cross_net=event.clears_net; net_data={"time":now-PHYSICS_DT+event.alpha*PHYSICS_DT,"position":event.position.tolist(),"clearance":event.clearance}; print("B2_PHASE=NET_CROSSED",flush=True)
            if detect_opponent_table_bounce(local_p,local_v,contact_seen=contact,bounce_seen=bounce):
                bounce=True; bounce_data={"time":now,"position":local_p.tolist()}; print("B2_PHASE=POST_CONTACT_BOUNCE_DETECTED",flush=True)
            previous_pos=local_p.copy(); previous_vel=local_v.copy()
        if (bounce and ex.lifecycle.phase.value=="ready" and "recovery" in phases) or (tick==219): break
    print("B1_PHASE=CONTROL_LOOP_EXIT",flush=True); print("B2_PHASE=RETURN_CHARACTERIZATION_COMPLETE",flush=True)
    min_clearance=min_world_z-float(geometry.FLOOR_Z); stable=bad==0 and min_clearance>.75 and max_tilt<35 and max_drift<.30 and {"ready","swing","follow_through","recovery"}<=phases
    legal=legal_return(contact,cross_net,bounce); failure="NONE" if legal else ("NO_CROSS_NET" if not cross_net else "CROSS_NET_BUT_MISSED_TABLE")
    values={"BALL_INITIAL_POSITION":BALL_INITIAL_POSITION.tolist(),"BALL_INITIAL_VELOCITY":BALL_INITIAL_VELOCITY.tolist(),"BALL_INITIAL_SPIN":BALL_INITIAL_SPIN.tolist(),"SELECTED_ACTION":"BH_CENTER","NORMALIZED_ACTION":list(ACTION),"EXPECTED_INTERCEPT_POSITION":pred.position.tolist(),"EXPECTED_INTERCEPT_TIME":pred.flight_time_s,"COMMAND_POSITION":built.position_world.tolist(),"COMMAND_VELOCITY":built.velocity_world.tolist(),"COMMAND_TTS":built.control_tts_s,"SWING_SIGN":built.swing_sign,"MODEL21800_INFERENCE_CALLS":calls,"RACKET_CONTACT":contact,"CONTACT_TIME":contact_data.get("time","N/A"),"BALL_POSITION_AT_CONTACT":contact_data.get("ball_position","N/A"),"BALL_VELOCITY_PRE_CONTACT":contact_data.get("ball_velocity_pre","N/A"),"BALL_VELOCITY_POST_CONTACT":contact_data.get("ball_velocity_post","N/A"),"RACKET_POSITION_AT_CONTACT":contact_data.get("racket_position","N/A"),"RACKET_VELOCITY_AT_CONTACT":contact_data.get("racket_velocity","NOT_AVAILABLE"),"MAX_CONTACT_FORCE":max_force,"CROSS_NET":cross_net,"NET_CROSSING_TIME":net_data.get("time","N/A"),"BALL_POSITION_AT_NET_CROSSING":net_data.get("position","N/A"),"BALL_Z_AT_NET_CROSSING":net_data.get("position",["N/A"]*3)[2],"NET_CLEARANCE":net_data.get("clearance","N/A"),"FIRST_POST_CONTACT_BOUNCE":bounce,"FIRST_POST_CONTACT_BOUNCE_TIME":bounce_data.get("time","N/A"),"FIRST_POST_CONTACT_BOUNCE_POSITION":bounce_data.get("position","N/A"),"OPPONENT_TABLE_LANDING":bounce,"LEGAL_RETURN":legal,"RETURN_FAILURE_CLASS":failure,"SIMULATION_NONFINITE_COUNT":bad,"SIMULATION_STABILITY":"PASS" if stable else "FAIL"}
    for k,v in values.items(): print(f"{k}="+(json.dumps(v) if isinstance(v,(list,dict)) else str(v)))
    for phase in ("ready","swing","follow_through","recovery"): print(f"{phase.upper()}_LIFECYCLE={'PASS' if phase in phases else 'FAIL'}")
    print("B1_PHASE=APP_CLOSE_BEGIN",flush=True); app.close(); print("B1_PHASE=APP_CLOSE_END",flush=True)
if __name__=="__main__": main()
