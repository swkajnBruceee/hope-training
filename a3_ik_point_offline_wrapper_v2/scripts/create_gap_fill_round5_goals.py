#!/usr/bin/env python3
"""Create internal-quantile targets for the remaining sparse dimensions."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'whole_space_gap_fill_20260807/gap_fill_manifest.json'; COVER=ROOT/'whole_space_gap_fill_round4_20260807/coverage_after_round4.json'; OUT=ROOT/'whole_space_gap_fill_round5_20260807'
def main():
    src=json.loads(SRC.read_text()); cov=json.loads(COVER.read_text()); by={s:[g for g in src['goals'] if g['swing_type']==s] for s in ('backhand','forehand')}; dims={'backhand':['velocity_x_mps','velocity_z_mps','normal_z','time_to_hit_s'],'forehand':['velocity_x_mps','velocity_y_mps','velocity_z_mps','normal_x','normal_y','normal_z','time_to_hit_s']}; goals=[]
    for stroke, names in dims.items():
        for di,name in enumerate(names):
            r=cov['by_stroke'][stroke]['ranges'][name]; vals=[r['min']+(r['max']-r['min'])*q for q in (.25,.5,.75)]
            for qi,val in enumerate(vals):
                base=by[stroke][(len(goals)*13+di)%len(by[stroke])]; vel=[float(x) for x in base['linear_velocity_mps']]; normal=np.array(base['racket_normal'],float); t=float(base['time_to_strike_s'])
                if name.startswith('velocity_'): vel[{'velocity_x_mps':0,'velocity_y_mps':1,'velocity_z_mps':2}[name]] = val
                elif name.startswith('normal_'): normal[{'normal_x':0,'normal_y':1,'normal_z':2}[name]]=val; normal/=np.linalg.norm(normal)
                else: t=val
                gid=f"r5_{'ba' if stroke=='backhand' else 'fo'}_{len(goals):03d}"; goals.append({'goal_id':gid,'goal_path':f'goals/{gid}.yaml','swing_type':stroke,'split':'validation' if len(goals)%6==0 else 'training','sequence':650000+len(goals),'position_m':[float(x) for x in base['position_m']],'linear_velocity_mps':vel,'racket_normal':normal.tolist(),'pitch_deg':float(base.get('pitch_deg',0)),'yaw_deg':float(base.get('yaw_deg',0)),'time_to_strike_s':t,'generation_role':'whole_space_internal_quantile_gap_fill_round5'})
    (OUT/'goals').mkdir(parents=True,exist_ok=True)
    for g in goals:
        (OUT/g['goal_path']).write_text('\n'.join(['schema_version: a3_canonical_strike_goal/v1',f"goal_id: {g['goal_id']}",'frame: initial_base_heading',f"swing_type: {g['swing_type']}",f"position_m: {[round(x,8) for x in g['position_m']]}",f"linear_velocity_mps: {[round(x,8) for x in g['linear_velocity_mps']]}",f"racket_normal: {[round(x,8) for x in g['racket_normal']]}",f"time_to_strike_s: {g['time_to_strike_s']:.6f}",f"sequence: {g['sequence']}"])+'\n')
    (OUT/'gap_fill_round5_manifest.json').write_text(json.dumps({'schema_version':'a3_whole_space_gap_fill_goals/v5','status':'raw_ik_generation_pending','source_coverage':str(COVER.resolve()),'goals':goals},ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'goals':len(goals),'forehand':sum(g['swing_type']=='forehand' for g in goals),'backhand':sum(g['swing_type']=='backhand' for g in goals)}))
if __name__=='__main__': main()
