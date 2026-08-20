"""B0 preflight. Run with IsaacLab Python; it performs no training."""
from pathlib import Path
import numpy as np

from hope_training.whole_body_tracking.deployment_v2_isaac.predictor_observation import PredictorCommand
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import OneDecisionCommandAssembler, MODEL
from hope_training.whole_body_tracking.deployment_v2_isaac.model21800_executor import bind_target_to_observation

CASES=((1,(-1,-1,-1),-.44,(-.58,0)),(1,(0,0,0),-.44,(-.58,0)),(1,(1,1,1),-.44,(-.58,0)),
       (-1,(-1,-1,-1),-.09,(-.58,.04)),(-1,(0,0,0),-.09,(-.58,.04)),(-1,(1,1,1),-.09,(-.58,.04)))

def main():
    assembler=OneDecisionCommandAssembler(); passed=0
    for i,(expected,action,y,base) in enumerate(CASES,1):
        p=PredictorCommand((0.,y,1.),(-3.,.2,-.5),10.,.6)
        c=assembler.build(flight_id=i,revision_id=1,command_seq=i,predictor_command=p,source_now_s=10.1,producer_wall_s=100.+i,current_base_xy=base,normalized_action=action)
        obs=bind_target_to_observation(np.zeros(110,np.float32),target_position_world=c.position_world,base_position_world=(base[0],base[1],0),target_velocity_world=c.velocity_world,time_to_strike_s=c.control_tts_s)
        assert c.swing_sign==expected and obs.shape==(110,) and np.isfinite(obs).all()
        passed+=1
    print(f"V2B0_PURE_CONTRACT_CASES={passed}/6")
    print("CONTRACT_TO_110D_BINDING=PASS")
    print("MODEL21800_EXECUTION=NOT_RUN")
    print("SIMULATION_STABILITY=NOT_RUN")
    print("V2B0=FAIL")
    print("NO_TRAINING=TRUE")

if __name__ == "__main__": main()
