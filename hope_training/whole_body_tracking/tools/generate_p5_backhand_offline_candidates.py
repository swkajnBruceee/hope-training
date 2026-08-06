#!/usr/bin/env python3
"""Generate P5 position-conditioned *offline* backhand trajectory candidates.

Requires the AimSim/MuJoCo Python environment.  This tool is intentionally
not a teacher approver: it emits low-dimensional reference candidates plus
offline evidence, all marked ``physics_qualified=false``.  Formal P1 PhysX
replay remains the only path to A/B/C teacher grading.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

import numpy as np

from repair_canonical_motion_prior import (
    DEFAULT_METADATA, DEFAULT_MJCF, IK_JOINTS, PolicyTcpKinematics,
    _collision_audit, _finite_difference, _fit_deformation, _trajectory_task_states,
)
from build_v27_bent_ready_motion import MujocoAudit
from build_upper_momentum_library import UrdfModel
from materialize_p4b_repaired_canonical_prior import (
    DEFAULT_URDF, _regenerate_body_arrays, _relative_body_velocity_from_joint_state, _tcp_hit_state,
)

ROOT = Path(__file__).resolve().parents[1]

def _motion_path(manifest: Path, row: dict[str, Any]) -> Path:
    for key in ("repaired_motion_npz", "canonical_motion_npz", "motion_npz"):
        if key in row:
            path = Path(row[key])
            return path if path.is_absolute() else manifest.parent / path
    raise ValueError(f"motion {row.get('motion_id')} has no NPZ field")

def _solve_hit(kin: PolicyTcpKinematics, q0: np.ndarray, pos: np.ndarray, normal: np.ndarray,
               iterations: int, margin: float) -> tuple[np.ndarray, bool]:
    q = q0.copy(); ids = [kin.index[name] for name in IK_JOINTS]
    target = pos + np.array((0.0, 0.0, 1.04))
    for _ in range(iterations):
        p, n, jac = kin.state(q, IK_JOINTS); assert jac is not None
        e = np.r_[target-p, 0.15*(normal-n)]
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 3e-3: break
        j = jac.copy(); j[3:] *= .15
        dq = j.T @ np.linalg.solve(j @ j.T + 3e-4*np.eye(6), e)
        q[ids] += np.clip(dq, -.04, .04); kin.project_soft(q, IK_JOINTS, margin)
    p, n, _ = kin.state(q)
    angle = np.degrees(np.arccos(np.clip(np.dot(n, normal), -1., 1.)))
    return q, bool(np.linalg.norm(p-target) <= .003 and angle <= 2.)

def _deform(seed: np.ndarray, hit: int, qhit: np.ndarray, rate_delta: np.ndarray,
            velocity_indices: list[int], window: int, fps: float) -> np.ndarray:
    """C2 local bump with independent, exact centre position and velocity terms."""
    out = seed.copy(); lo=max(0,hit-window); hi=min(len(seed)-1,hit+window)
    delta=qhit-seed[hit]
    for frame in range(lo,hi+1):
        r=(frame-hit)/max(1,window); w=max(0.,1-r*r)**3
        out[frame] += w*delta
        # h(hit)=0 and dh/dt(hit)=1; it changes strike velocity without
        # changing the IK hit pose or either window endpoint.
        h=(window/fps)*r*max(0.,1-r*r)**3
        out[frame, velocity_indices] += h*rate_delta
    return out

def _p1_hit_state(model: UrdfModel, names: list[str], bodies: list[str], source: dict[str,np.ndarray],
                  q: np.ndarray, qd: np.ndarray, hit: int) -> dict[str,list[float]]:
    """The exact body/velocity reconstruction used by the P1 materializer."""
    root_pos=np.asarray(source['body_pos_b0'][:,0],dtype=float); root_quat=np.asarray(source['body_quat_b0_wxyz'][:,0],dtype=float)
    pos, quat=_regenerate_body_arrays(model,names,bodies,q,root_pos,root_quat,float(np.asarray(source['fps']).reshape(-1)[0]))
    old_lin,old_ang=_relative_body_velocity_from_joint_state(model,names,bodies,np.asarray(source['joint_pos'],dtype=float),np.asarray(source['joint_vel'],dtype=float),root_quat)
    new_lin,new_ang=_relative_body_velocity_from_joint_state(model,names,bodies,q,qd,root_quat)
    arrays={'body_pos_b0':pos,'body_quat_b0_wxyz':quat,'body_lin_vel_b0':np.asarray(source['body_lin_vel_b0'],dtype=float)+new_lin-old_lin,'body_ang_vel_b0':np.asarray(source['body_ang_vel_b0'],dtype=float)+new_ang-old_ang}
    return _tcp_hit_state(arrays,bodies.index('right_wrist_yaw_Link'),hit)

def _candidate(kin: PolicyTcpKinematics, audit: MujocoAudit, p1_model: UrdfModel, names: list[str], bodies: list[str], source: dict[str,np.ndarray], q: np.ndarray, qd: np.ndarray,
               hit: int, fps: float, goal: dict[str, Any], degree: int, margin: float, window: int,
               velocity_limit: float, time_limit: float) -> tuple[dict[str,Any], np.ndarray, np.ndarray, np.ndarray]:
    qhit, ik = _solve_hit(kin, q[hit], np.asarray(goal['position_b0_m']), np.asarray(goal['normal_b0']), 60, margin)
    if not ik: return {"qualification":"IK_FAILED"}, q, qd, np.empty((0,0))
    p=np.asarray(goal['position_b0_m'])+np.array((0.,0.,1.04)); n=np.asarray(goal['normal_b0']); v=np.asarray(goal['linear_velocity_b0_mps'])
    _, _, jac=kin.state(qhit, IK_JOINTS); assert jac is not None
    velocity_indices=[kin.index[name] for name in IK_JOINTS]
    base_qd=qd[hit, velocity_indices]
    velocity_residual=v-jac[:3]@base_qd
    rate_delta=jac[:3].T @ np.linalg.solve(jac[:3]@jac[:3].T+2e-3*np.eye(3),velocity_residual)
    rate_delta=np.clip(rate_delta,-4.,4.)
    scale=float(np.sum(qd*_finite_difference(q,fps))/max(np.sum(_finite_difference(q,fps)**2),1e-12))
    # Reconcile the MuJoCo Jacobian proposal with the P1 materializer oracle.
    # The oracle is authoritative; the Jacobian only supplies a bounded local
    # direction for the next correction.
    iterations=[]
    for iteration in range(4):
        direct=_deform(q,hit,qhit,rate_delta,velocity_indices,window,fps)
        coeff, fit=_fit_deformation(direct-q,degree); fitted=q+fit
        fitted_qd=qd+scale*_finite_difference(fitted-q,fps)
        p1=_p1_hit_state(p1_model,names,bodies,source,fitted,fitted_qd,hit)
        oracle_residual=v-np.asarray(p1['racket_velocity_b0_mps'])
        iterations.append(float(np.linalg.norm(oracle_residual)))
        if np.linalg.norm(oracle_residual) <= velocity_limit:
            break
        correction=jac[:3].T @ np.linalg.solve(jac[:3]@jac[:3].T+2e-3*np.eye(3),oracle_residual)
        rate_delta=np.clip(rate_delta+0.6*np.clip(correction,-2.,2.),-6.,6.)
    p1p=np.asarray(p1['racket_position_b0_m']); p1n=np.asarray(p1['racket_normal_b0']); p1v=np.asarray(p1['racket_velocity_b0_mps'])
    pe=float(np.linalg.norm(p1p-np.asarray(goal['position_b0_m']))); ne=float(np.degrees(np.arccos(np.clip(p1n@n,-1.,1.)))); ve=float(np.linalg.norm(p1v-v))
    soft=min(kin.soft_margin_detail(frame)[0] for frame in fitted)
    collision=_collision_audit(audit,fitted)
    time_error=abs(hit/fps-float(goal['time_to_hit_s']))
    gates={"reference_position_le_3mm":pe<=.003,"reference_normal_le_2deg":ne<=2.,"reference_velocity_within_limit":ve<=velocity_limit,"reference_time_within_limit":time_error<=time_limit,"positive_soft_margin":soft>=margin,"collision_nonnegative":collision['minimum_distance_m']>=0.}
    return {"qualification":"PENDING_PHYSX" if all(gates.values()) else "OFFLINE_REJECTED","offline_gates":gates,"reference_hit":{"position_error_m":pe,"normal_error_deg":ne,"velocity_error_mps":ve,"time_error_s":time_error,"velocity_oracle":"p1_materializer_urdf_directional_fk/v1"},"velocity_optimization":{"joint_names":list(IK_JOINTS),"joint_rate_delta_radps":rate_delta.tolist(),"initial_mujoco_residual_norm_mps":float(np.linalg.norm(velocity_residual)),"p1_oracle_residual_norms_mps":iterations},"minimum_soft_margin_rad":soft,"collision":collision,"physics_qualified":False}, fitted, fitted_qd, coeff

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--workspace',type=Path,required=True); ap.add_argument('--source-manifest',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seed-motion-ids',type=int,nargs='+',default=[0,2,3,4,5]); ap.add_argument('--max-samples',type=int,default=32); ap.add_argument('--splits',nargs='+',default=['training','validation']); ap.add_argument('--basis-degree',type=int,default=12); ap.add_argument('--window-frames',type=int,default=12); ap.add_argument('--soft-margin-rad',type=float,default=.02); ap.add_argument('--reference-velocity-error-max-mps',type=float,default=.2); ap.add_argument('--reference-time-error-max-s',type=float,default=.02); ap.add_argument('--metadata',type=Path,default=DEFAULT_METADATA); ap.add_argument('--mjcf',type=Path,default=DEFAULT_MJCF); args=ap.parse_args()
    workspace=json.loads(args.workspace.read_text()); source=json.loads(args.source_manifest.read_text()); metadata=json.loads(args.metadata.read_text()); names=metadata['joint_names']; bodies=metadata['body_names']; rows={int(x['motion_id']):x for x in source['motions']}; p1_model=UrdfModel(DEFAULT_URDF)
    samples=[x for x in workspace['samples'] if x['split'] in args.splits][:args.max_samples]
    kin=PolicyTcpKinematics(args.mjcf,names); audit=MujocoAudit(args.mjcf,names); args.output_dir.mkdir(parents=True,exist_ok=True); out=[]
    for sample in samples:
        attempts=[]
        for mid in args.seed_motion_ids:
            if mid not in rows: continue
            path=_motion_path(args.source_manifest,rows[mid])
            with np.load(path) as data: source_arrays={key:np.asarray(data[key]).copy() for key in data.files}; q=np.asarray(source_arrays['joint_pos'],dtype=float); qd=np.asarray(source_arrays['joint_vel'],dtype=float); fps=float(np.asarray(source_arrays['fps']).reshape(-1)[0])
            hit=int(rows[mid].get('hit_frame',30)); result,cq,cqd,coeff=_candidate(kin,audit,p1_model,names,bodies,source_arrays,q,qd,hit,fps,sample['canonical_goal_10d'],args.basis_degree,args.soft_margin_rad,args.window_frames,args.reference_velocity_error_max_mps,args.reference_time_error_max_s)
            stem=f"{sample['sample_id']}_seed{mid:02d}"; result.update({"seed_motion_id":mid,"source_npz":str(path),"hit_frame":hit})
            if result['qualification']=='PENDING_PHYSX':
                goal=sample['canonical_goal_10d']
                np.savez_compressed(args.output_dir/f'{stem}.npz',joint_pos=cq.astype(np.float32),joint_vel=cqd.astype(np.float32),basis_coefficients=coeff.astype(np.float32),hit_frame=np.asarray([hit]),canonical_goal_position_b0_m=np.asarray(goal['position_b0_m'],dtype=np.float64),canonical_goal_normal_b0=np.asarray(goal['normal_b0'],dtype=np.float64),canonical_goal_linear_velocity_b0_mps=np.asarray(goal['linear_velocity_b0_mps'],dtype=np.float64),canonical_goal_time_to_hit_s=np.asarray([goal['time_to_hit_s']],dtype=np.float64),physics_qualified=np.asarray([False]))
                result['candidate_npz']=f'{stem}.npz'
            attempts.append(result)
        out.append({**sample,"seed_attempts":attempts,"teacher_quality":None})
    payload={"schema_version":"p5_backhand_offline_candidates/v1","teacher_data":False,"physics_qualified":False,"workspace":str(args.workspace),"source_manifest":str(args.source_manifest),"candidate_count":len(out),"samples":out}
    (args.output_dir/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({"samples":len(out),"output":str(args.output_dir/'manifest.json')}))
if __name__=='__main__': main()
