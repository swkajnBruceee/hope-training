"""Deterministic B1 fixed-ball definition traced to HOPE open-source sources."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml

SOLVER_YAML=Path("/home/a104/hope_training_repo/hope-model21800-isaac/hope_ws/src/solver/config/solver.yaml")
BALL_INITIAL_POSITION=np.array([2.22,-0.5375,0.43],dtype=np.float64)
BALL_INITIAL_VELOCITY=np.array([-6.0,0.0,0.08],dtype=np.float64)
BALL_INITIAL_SPIN=np.zeros(3,dtype=np.float64)
BALL_RADIUS=0.02
BALL_MASS=0.0027
PREDICTOR_DT=0.001
GRAVITY=np.array([0.0,0.0,-9.81],dtype=np.float64)

@dataclass(frozen=True)
class PredictedIntercept:
    position: np.ndarray
    velocity: np.ndarray
    flight_time_s: float
    bounce_count: int

def load_solver_physics():
    p=yaml.safe_load(SOLVER_YAML.read_text())["solver"]["ros__parameters"]
    return {k:float(p[k]) for k in ("x_hit","drag_k","restitution_h","restitution_v")}

def predict_fixed_intercept() -> PredictedIntercept:
    """Literal numeric parity port of HOPE BallTrajectoryPredictor::predict."""
    cfg=load_solver_physics(); p=BALL_INITIAL_POSITION.copy(); v=BALL_INITIAL_VELOCITY.copy(); t=0.; bounces=0
    for _ in range(round(2.0/PREDICTOR_DT)):
        p_prev=p.copy(); a=-cfg["drag_k"]*np.linalg.norm(v)*v+GRAVITY
        vn=v+a*PREDICTOR_DT; pn=p+v*PREDICTOR_DT+0.5*a*PREDICTOR_DT**2; t+=PREDICTOR_DT
        bounced=False; pb=p.copy(); vp=v.copy(); remaining=PREDICTOR_DT
        if pn[2]<0 and vn[2]<0 and -BALL_RADIUS<=pn[0]<=2.74+BALL_RADIUS and -1.525-BALL_RADIUS<=pn[1]<=BALL_RADIUS:
            frac=float(np.clip(p[2]/max(p[2]-pn[2],1e-12),0,1)); pb=p+frac*(pn-p); pb[2]=0
            vb=v+a*(frac*PREDICTOR_DT); vp=np.array([cfg["restitution_h"]*vb[0],cfg["restitution_h"]*vb[1],-cfg["restitution_v"]*vb[2]])
            remaining=(1-frac)*PREDICTOR_DT; ap=-cfg["drag_k"]*np.linalg.norm(vp)*vp+GRAVITY
            pn=pb+vp*remaining+0.5*ap*remaining**2; vn=vp+ap*remaining; bounces+=1; bounced=True
        if p_prev[0]>cfg["x_hit"] and pn[0]<=cfg["x_hit"] and vn[0]<0:
            start=pb if bounced else p; start_v=vp if bounced else v; span=remaining if bounced else PREDICTOR_DT
            frac=float(np.clip((start[0]-cfg["x_hit"])/max(start[0]-pn[0],1e-12),0,1))
            pc=start+frac*(pn-start); vc=start_v+frac*(vn-start_v); pc[0]=cfg["x_hit"]
            tc=(t-remaining+frac*remaining) if bounced else (t-PREDICTOR_DT+frac*PREDICTOR_DT)
            return PredictedIntercept(pc,vc,tc,bounces)
        p,v=pn,vn
    raise RuntimeError("fixed incoming did not cross HOPE x_hit")
