from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/".local_deps"))
from isaaclab.app import AppLauncher
CASE_NAMES=("FH_LOW","FH_CENTER","FH_HIGH","BH_LOW","BH_CENTER","BH_HIGH")
p=argparse.ArgumentParser()
p.add_argument("--case", choices=CASE_NAMES, default=None,
               help="Run one immutable diagnostic case; default runs all six cases.")
AppLauncher.add_app_launcher_args(p); args=p.parse_args(); app=AppLauncher(args).app

from hope_training.whole_body_tracking.deployment_v2_isaac.low_level_isaac_executor import Model21800IsaacExecutor, MODEL, geometry
from hope_training.whole_body_tracking.deployment_v2_isaac.predictor_observation import PredictorCommand
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import OneDecisionCommandAssembler

CASES=(("FH_LOW",(-1,-1,-1),1),("FH_CENTER",(0,0,0),1),("FH_HIGH",(1,1,1),1),("BH_LOW",(-1,-1,-1),-1),("BH_CENTER",(0,0,0),-1),("BH_HIGH",(1,1,1),-1))
def main():
    ex=Model21800IsaacExecutor(device=args.device); assembler=OneDecisionCommandAssembler(); results=[]
    selected=CASES if args.case is None else tuple(case for case in CASES if case[0]==args.case)
    for i,(name,a,sign) in enumerate(selected,1):
        ex.reset(); base=ex.initial_base.copy(); yoff=-.44 if sign==1 else -.09
        target=(float(base[0]+.58),float(base[1]+yoff),1.0)
        pc=PredictorCommand(target,(-3.,.2,-.5),10.,.8)
        built=assembler.build(flight_id=i,revision_id=1,command_seq=i,predictor_command=pc,source_now_s=10.,producer_wall_s=100.+i,current_base_xy=base[:2],normalized_action=a)
        if built.swing_sign!=sign: raise RuntimeError(f"side mismatch {name}: {built.swing_sign}")
        ex.set_target_command(built.position_world,built.velocity_world,built.control_tts_s,built.swing_sign)
        r=ex.run_full_lifecycle()
        min_root_clearance=float(r.min_base_height)-float(geometry.FLOOR_Z)
        height_gate=min_root_clearance>0.75
        tilt_gate=float(r.max_base_tilt_deg)<35.0
        drift_gate=float(r.max_root_xy_drift)<0.30
        lifecycle_gate={"ready","swing","follow_through","recovery"}<=set(r.phases)
        row=dict(case=name,side=sign,action=a,velocity=built.velocity_world.tolist(),phases=sorted(r.phases),inference_calls=r.inference_calls,stable=r.stable,failure_reason=r.failure_reason,initial_base_z=float(base[2]),min_base_height=r.min_base_height,min_root_clearance=min_root_clearance,max_base_tilt_deg=r.max_base_tilt_deg,max_root_xy_drift=r.max_root_xy_drift,stability_height_gate=height_gate,stability_tilt_gate=tilt_gate,stability_drift_gate=drift_gate,stability_lifecycle_gate=lifecycle_gate,nonfinite=r.nonfinite_count)
        results.append(row)
        print(f"INITIAL_BASE_Z={row['initial_base_z']}")
        print(f"MIN_WORLD_Z={row['min_base_height']}")
        print(f"MIN_ROOT_CLEARANCE={row['min_root_clearance']}")
        print(f"MAX_BASE_TILT_DEG={row['max_base_tilt_deg']}")
        print(f"MAX_ROOT_XY_DRIFT={row['max_root_xy_drift']}")
        print(f"STABILITY_HEIGHT_GATE={row['stability_height_gate']}")
        print(f"STABILITY_TILT_GATE={row['stability_tilt_gate']}")
        print(f"STABILITY_DRIFT_GATE={row['stability_drift_gate']}")
        print(f"STABILITY_LIFECYCLE_GATE={row['stability_lifecycle_gate']}")
        print(f"STABLE={row['stable']}")
        print("CASE_RESULT="+json.dumps(row))
    calls=sum(x["inference_calls"] for x in results); passed=sum(x["stable"] for x in results); phases=set.intersection(*(set(x["phases"]) for x in results))
    print(f"MODEL21800_INFERENCE_CALLS={calls}"); print(f"INTEGRATED_CASES={passed}/{len(selected)}")
    for phase in ("ready","swing","follow_through","recovery"): print(f"{phase.upper()}_LIFECYCLE={'PASS' if phase in phases else 'FAIL'}")
    print(f"SIMULATION_NONFINITE_COUNT={sum(x['nonfinite'] for x in results)}"); print(f"SIMULATION_STABILITY={'PASS' if passed==len(selected) else 'FAIL'}"); print(f"V2B0={'PASS' if len(selected)==6 and passed==6 and calls>0 else 'FAIL'}")
    app.close()
if __name__=="__main__": main()
