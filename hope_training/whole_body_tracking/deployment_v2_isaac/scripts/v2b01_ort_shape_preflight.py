"""Real model_21800 ORT calls without claiming Isaac execution."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
DEPLOY=Path("/home/a104/hope_training_repo/hope-deploy-baseline/a3_deploy/a3_deploy_example")
sys.path[:0]=[str(ROOT/".local_deps"),str(DEPLOY/"reference")]
from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter
from a3_deploy_onnx_ref_pingpong.observation import RobotState, ObsTarget, build_observation
from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy

def main():
    model=DEPLOY/"models/model_21800/policy/exported/policy.onnx"
    adapter=ActionAdapter.from_yaml(DEPLOY/"models/model_21800/policy/params/deploy.yaml")
    policy=OnnxPolicy(model,providers=["CPUExecutionProvider"])
    state=RobotState(np.array([-.58,0,.8]),np.array([1.,0,0,0]),np.zeros(3),adapter.default_q.copy(),np.zeros(31))
    last=np.zeros(31,np.float32); calls=0
    for sign in (1,-1):
        for velocity in ((1.57,.10,.41),(2.06,.31,.88),(2.55,.52,1.35)) if sign==1 else ((1.55,-.18,.40),(2.035,.055,.86),(2.52,.29,1.32)):
            target=ObsTarget(np.array([0.,-.44 if sign==1 else -.09,1.]),np.asarray(velocity),.8)
            obs=build_observation(state,target,last,adapter.default_q,state.base_pos_w[:2])
            raw=policy.infer_target(obs,.8,sign,.02); calls+=1
            assert obs.shape==(110,) and raw.shape==(31,) and np.isfinite(np.r_[obs,raw]).all()
            raw[3:5]=0.; last=raw
    print(f"MODEL21800_INFERENCE_CALLS={calls}")
    print("MODEL21800_INPUT_CONTRACT=PASS")
    print("MODEL21800_OUTPUT_SHAPE=PASS")
    print("ISAAC_STATE_SOURCE=FALSE")
if __name__=="__main__": main()
