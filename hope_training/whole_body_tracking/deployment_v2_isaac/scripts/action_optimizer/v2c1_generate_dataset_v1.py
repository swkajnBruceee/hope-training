"""Predetermined finite-grid legal-return feasibility search; no learning."""
from __future__ import annotations
import argparse,json,math,sys,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
WBT_ROOT=ROOT/"hope_training"/"whole_body_tracking"

sys.path[:0]=[
    str(ROOT),
    str(WBT_ROOT),
    str(ROOT/".local_deps")
]
sys.path[:0]=[str(ROOT),str(WBT_ROOT),str(ROOT/".local_deps")]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument("--output-dir",default=None); AppLauncher.add_app_launcher_args(p); args=p.parse_args(); app=AppLauncher(args).app

import torch
from isaaclab.utils.math import quat_apply,quat_rotate_inverse
from training.robots.agibot_a3 import A3_MOUNT_OFFSET,A3_WRIST_BODY
from training.tasks.table_tennis import geometry
from training.tasks.table_tennis.ball import BallAerodynamicsCfg,compute_aero_wrench
from hope_training.whole_body_tracking.deployment_v2_isaac.action_space_feasibility import first_legal_result,negative_search_semantics,repeatability_pass,tier1_actions,tier2_additional_actions
from hope_training.whole_body_tracking.deployment_v2_isaac.fixed_ball_smoke import BALL_INITIAL_POSITION,BALL_INITIAL_SPIN,BALL_INITIAL_VELOCITY,predict_fixed_intercept
from hope_training.whole_body_tracking.deployment_v2_isaac.low_level_isaac_executor import CONTROL_DT,DECIMATION,PHYSICS_DT,Model21800IsaacExecutor,RacketCommand
from hope_training.whole_body_tracking.deployment_v2_isaac.predictor_observation import PredictorCommand
from hope_training.whole_body_tracking.deployment_v2_isaac.return_metrics import detect_opponent_table_bounce,detect_outgoing_net_crossing,legal_return
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import OneDecisionCommandAssembler

LEAD_TIME=1.20; SIDE=-1; PARKED=(2.22,-.5375,.43)
CONTACT_FORCE_THRESHOLD=.05; FACE_LATERAL_THRESHOLD=.10; FACE_NORMAL_THRESHOLD=.10
SCIENTIFIC_COMPLETE=False

def main():
    global SCIENTIFIC_COMPLETE
    print("B21_PHASE=MAIN_ENTER",flush=True)
    out = Path("hope_training/whole_body_tracking/deployment_v2_isaac/scripts/action_optimizer/datasets/C1_action_dataset_v1") if args.output_dir else ROOT/"results"/"model21800_v2B"/f"B21_action_feasibility_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True,exist_ok=False)
    ex=Model21800IsaacExecutor(device=args.device); ball=ex.scene["ball"]; sensor=ex.scene.sensors["racket_ball_contact"]
    lifecycle_type=type(ex.lifecycle)
    print("B21_PHASE=EXECUTOR_CREATED",flush=True)
    wrist_ids,_=ex.robot.find_bodies(A3_WRIST_BODY,preserve_order=True); wrist_id=int(wrist_ids[0]); mount=torch.tensor(A3_MOUNT_OFFSET,device=ex.device)
    pose=torch.zeros((1,7),device=ball.device); pose[:,3]=1.; velocity=torch.zeros((1,6),device=ball.device)
    def set_ball(position,linear,angular=(0.,0.,0.)):
        pose[:,:3]=ex.scene.env_origins+torch.tensor(position,device=ball.device); velocity[:,:3]=torch.tensor(linear,device=ball.device); velocity[:,3:]=torch.tensor(angular,device=ball.device)
        ball.write_root_pose_to_sim(pose); ball.write_root_velocity_to_sim(velocity)
    aero_cfg=BallAerodynamicsCfg(); force_b=torch.zeros((1,1,3),device=ball.device); torque_b=torch.zeros_like(force_b)
    def aero(_dt):
        fw,tw=compute_aero_wrench(ball.data.root_lin_vel_w,ball.data.root_ang_vel_w,float(geometry.BALL_MASS),aero_cfg)
        force_b[:,0]=quat_rotate_inverse(ball.data.root_quat_w,fw); torque_b[:,0]=quat_rotate_inverse(ball.data.root_quat_w,tw)
        ball.set_external_force_and_torque(force_b,torque_b); ball.write_data_to_sim()
    ex.sim.add_physics_callback("v2b21_hope_ball_aerodynamics",aero)
    pred=predict_fixed_intercept(); launch_delay=LEAD_TIME-pred.flight_time_s

    def fresh_reset():
        # Stage5 exact persistent-case reset: full robot reset, park before every
        # 400-Hz settle step, final no-step repark, then capture base targets.
        ex.robot.write_root_pose_to_sim(ex.root0[:,:7]); ex.robot.write_root_velocity_to_sim(ex.root0[:,7:])
        ex.robot.write_joint_state_to_sim(ex.q0,ex.qd0,joint_ids=ex.joint_ids); ex.robot.set_joint_position_target(ex.q0,joint_ids=ex.joint_ids)
        ex.lifecycle=lifecycle_type(ex.lifecycle_cfg); ex.last_action=np.zeros(31,np.float32); ex.target=None; ex.task_id=0
        for _ in range(int(round(.20/PHYSICS_DT))):
            set_ball(PARKED,(0,0,0),(0,0,0)); ex.scene.write_data_to_sim(); ex.sim.step(render=False); ex.scene.update(PHYSICS_DT)
        set_ball(PARKED,(0,0,0),(0,0,0)); ex.scene.write_data_to_sim()
        ex.base_target_xy=ex.robot.data.root_pos_w[0,:2].detach().cpu().numpy().astype(np.float64)
        ex.initial_base=ex.robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)

    def run_case(index,tier,action,repeat_trial=None):
        fresh_reset(); assembler=OneDecisionCommandAssembler(); pc=PredictorCommand(tuple(pred.position),tuple(pred.velocity),10.,LEAD_TIME)
        built=assembler.build(flight_id=1,revision_id=1,command_seq=1,predictor_command=pc,source_now_s=10.,producer_wall_s=100.,current_base_xy=ex.initial_base[:2],normalized_action=action)
        assert built.swing_sign==SIDE and np.allclose(built.position_world,pred.position) and abs(built.control_tts_s-LEAD_TIME)<1e-5
        ex.set_target_command(built.position_world,built.velocity_world,built.control_tts_s,built.swing_sign)
        phases={"ready"}; calls=bad=0; launched=contact=net_seen=cross_net=bounce=False; ctime=None; post_vel=None; net_clearance=None; bounce_pos=None
        min_z=float("inf"); max_tilt=max_drift=0.; previous_pos=None
        for tick in range(220):
            elapsed=tick*CONTROL_DT; state=ex._state(); cmd=RacketCommand(task_id=ex.task_id,task_revision=0,swing_sign=SIDE,position=built.position_world,velocity=built.velocity_world,time_to_strike=built.control_tts_s) if tick==0 else None
            target=ex.lifecycle.update(cmd,state); phases.add(ex.lifecycle.phase.value)
            from a3_deploy_onnx_ref_pingpong.observation import build_observation
            obs=build_observation(state,target,ex.last_action,ex.adapter.default_q,ex.base_target_xy)
            min_z=min(min_z,float(state.base_pos_w[2])); max_drift=max(max_drift,float(np.linalg.norm(state.base_pos_w[:2]-ex.initial_base[:2]))); max_tilt=max(max_tilt,math.degrees(math.acos(float(np.clip(-obs[98],-1,1)))))
            raw=np.asarray(ex.policy.infer_target(obs,target.time_to_strike,ex.lifecycle.swing_sign,CONTROL_DT),dtype=np.float32).reshape(31); calls+=1; raw[3:5]=0
            q=ex.adapter.decode(raw); q[3:5]=ex.adapter.default_q[3:5]; ex.robot.set_joint_position_target(torch.tensor(q,device=ex.device,dtype=ex.dtype).reshape(1,31),joint_ids=ex.joint_ids); ex.last_action=raw.copy()
            if not np.isfinite(np.r_[obs,raw,q]).all(): bad+=1; break
            for sub in range(DECIMATION):
                now=elapsed+sub*PHYSICS_DT
                if not launched and now+1e-12>=launch_delay: set_ball(BALL_INITIAL_POSITION,BALL_INITIAL_VELOCITY,BALL_INITIAL_SPIN); launched=True
                elif not launched: set_ball(PARKED,(0,0,0),(0,0,0))
                ex.scene.write_data_to_sim(); ex.sim.step(render=False); ex.scene.update(PHYSICS_DT)
                bp=ball.data.root_pos_w[0]; bv=ball.data.root_lin_vel_w[0]; local_p=(bp-ex.scene.env_origins[0]).detach().cpu().numpy(); local_v=bv.detach().cpu().numpy()
                wpos=ex.robot.data.body_pos_w[0,wrist_id]; wquat=ex.robot.data.body_quat_w[0,wrist_id]; rp=wpos+quat_apply(wquat,mount); normal=quat_apply(wquat,torch.tensor([0.,1.,0.],device=ex.device)); delta=bp-rp; signed=torch.dot(delta,normal); lateral=torch.linalg.vector_norm(delta-signed*normal)
                force=float(torch.linalg.vector_norm(sensor.data.net_forces_w,dim=-1).reshape(1,-1).amax(dim=1)[0])
                if force>CONTACT_FORCE_THRESHOLD and float(lateral)<FACE_LATERAL_THRESHOLD and abs(float(signed))<FACE_NORMAL_THRESHOLD and not contact: contact=True; ctime=now; post_vel=local_v.tolist()
                if previous_pos is not None:
                    event=detect_outgoing_net_crossing(previous_pos,local_p,local_v,contact_seen=contact,net_seen=net_seen)
                    if event is not None: net_seen=True; cross_net=event.clears_net; net_clearance=event.clearance
                if detect_opponent_table_bounce(local_p,local_v,contact_seen=contact,bounce_seen=bounce): bounce=True; bounce_pos=local_p.tolist()
                previous_pos=local_p.copy()
            if (bounce and ex.lifecycle.phase.value=="ready" and "recovery" in phases) or tick==219: break
        stable=bad==0 and min_z-float(geometry.FLOOR_Z)>.75 and max_tilt<35 and max_drift<.30 and {"ready","swing","follow_through","recovery"}<=phases
        legal=legal_return(contact,cross_net,bounce)
        reward = (
	    100.0 if legal else 0.0
            + 30.0 if cross_net else 0.0
            + 20.0 if contact else 0.0
        )
        row={"REWARD":reward,"CASE_INDEX":index,"SEARCH_TIER":tier,"REPEAT_TRIAL":repeat_trial,"NORMALIZED_ACTION":list(action),"COMMAND_VELOCITY":built.velocity_world.tolist(),"RACKET_CONTACT":contact,"CONTACT_TIME":ctime,"BALL_VELOCITY_POST_CONTACT":post_vel,"CROSS_NET":cross_net,"NET_CLEARANCE":net_clearance,"OPPONENT_TABLE_LANDING":bounce,"FIRST_POST_CONTACT_BOUNCE_POSITION":bounce_pos,"LEGAL_RETURN":legal,"RETURN_FAILURE_CLASS":"NONE" if legal else ("NO_CROSS_NET" if not cross_net else "CROSS_NET_BUT_MISSED_TABLE"),"MODEL21800_INFERENCE_CALLS":calls,"SIMULATION_NONFINITE_COUNT":bad,"SIMULATION_STABILITY":stable,"READY_LIFECYCLE":"ready" in phases,"SWING_LIFECYCLE":"swing" in phases,"FOLLOW_THROUGH_LIFECYCLE":"follow_through" in phases,"RECOVERY_LIFECYCLE":"recovery" in phases}
        print("CASE_RESULT="+json.dumps(row,separators=(",",":")),flush=True); return row

    search=[]
    print("TIER1_CASES=27",flush=True)
    print("TIER2_ADDITIONAL_CASES=98",flush=True)
    print("MAX_UNIQUE_SEARCH_CASES=125",flush=True)
    print("WITNESS_SELECTION_RULE=FIRST_LEGAL_IN_PREDEFINED_GRID_ORDER",flush=True)
    print("WITNESS_REPEAT_REQUIRED=5",flush=True)
    print("B21_PHASE=SEARCH_BEGIN",flush=True)
    for action in tier1_actions():
        search.append(
            run_case(len(search),1,action)
        )
    witness=first_legal_result(search)
    if witness is None:
        for action in tier2_additional_actions():
            search.append(
                run_case(len(search),2,action)
            )
        witness=first_legal_result(search)
    repeats=[]
    if witness is not None:
        print("B21_PHASE=WITNESS_FOUND",flush=True)
        print("B21_PHASE=WITNESS_REPEAT_BEGIN",flush=True)
        for trial in range(1,6): repeats.append(run_case(len(search)+trial-1,"WITNESS_REPEAT",tuple(witness["NORMALIZED_ACTION"]),trial))
    summary={"SEARCH_CASES_EXECUTED":len(search),"CONTACT_COUNT":sum(x["RACKET_CONTACT"] for x in search),"CROSS_NET_COUNT":sum(x["CROSS_NET"] for x in search),"OPPONENT_TABLE_LANDING_COUNT":sum(x["OPPONENT_TABLE_LANDING"] for x in search),"LEGAL_RETURN_COUNT":sum(x["LEGAL_RETURN"] for x in search),"UNSTABLE_COUNT":sum(not x["SIMULATION_STABILITY"] for x in search),"NONFINITE_CASE_COUNT":sum(x["SIMULATION_NONFINITE_COUNT"]>0 for x in search),**negative_search_semantics(search),"WITNESS_SELECTION_RULE":"FIRST_LEGAL_IN_PREDEFINED_GRID_ORDER","WITNESS_NORMALIZED_ACTION":None if witness is None else witness["NORMALIZED_ACTION"],"WITNESS_COMMAND_VELOCITY":None if witness is None else witness["COMMAND_VELOCITY"],"WITNESS_NET_CLEARANCE":None if witness is None else witness["NET_CLEARANCE"],"WITNESS_LANDING_POSITION":None if witness is None else witness["FIRST_POST_CONTACT_BOUNCE_POSITION"],"WITNESS_REPEAT_TRIALS":5 if witness else 0,"WITNESS_REPEAT_LEGAL":sum(x["LEGAL_RETURN"] for x in repeats),"WITNESS_REPEAT_CONTACT":sum(x["RACKET_CONTACT"] for x in repeats),"WITNESS_REPEAT_STABLE":sum(x["SIMULATION_STABILITY"] for x in repeats),"DETERMINISTIC_FEASIBILITY_WITNESS":"PASS" if repeatability_pass(repeats) else "FAIL"}
    (out/"search_cases.json").write_text(json.dumps(search,indent=2)+"\n"); (out/"witness_repeats.json").write_text(json.dumps(repeats,indent=2)+"\n"); (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("SEARCH_SUMMARY="+json.dumps(summary,separators=(",",":")),flush=True)
    print("B21_PHASE=SEARCH_COMPLETE",flush=True)
    SCIENTIFIC_COMPLETE=True
    print("B21_SCIENTIFIC_COMPLETE=TRUE",flush=True)
    print("B21_PHASE=APP_CLOSE_BEGIN",flush=True)
    app.close()
    print("B21_PHASE=APP_CLOSE_END",flush=True)
if __name__=="__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        if SCIENTIFIC_COMPLETE:
            print("B21_TEARDOWN_EXCEPTION_AFTER_SCIENTIFIC_COMPLETE=TRUE",flush=True)
        else:
            print("B21_RUNTIME_EXCEPTION=TRUE",flush=True)
            print("B21_SCIENTIFIC_COMPLETE=FALSE",flush=True)
        raise
