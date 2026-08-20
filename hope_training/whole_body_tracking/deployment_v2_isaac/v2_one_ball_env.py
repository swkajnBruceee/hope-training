"""One-high-level-decision-per-flight command assembly (no trainer)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from hope_training.whole_body_tracking.deployment_v2 import (
    AdapterStationMirror, build_schema2_packet, load_canonical_metadata,
    map_normalized_velocity, velocity_inside_planner_box,
)
from .predictor_observation import PredictorCommand, build_v2_observation

MODEL = Path("/home/a104/hope_training_repo/hope-deploy-baseline/a3_deploy/a3_deploy_example/models/model_21800/policy/exported/policy.onnx")
POSITION_CONTRACT = "HOPE_OPEN_SOURCE_SOLVER_POSITION"
FRAME_CONTRACT = "HOPE_OPEN_SOURCE_WORLD_TABLE_FRAME_CODE_0"
MID_SWING_RL_RESAMPLING = False

@dataclass(frozen=True)
class TargetCommand:
    observation: np.ndarray
    normalized_action: np.ndarray
    position_world: np.ndarray
    velocity_world: np.ndarray
    control_tts_s: float
    swing_sign: int
    station_xy: np.ndarray
    schema2: np.ndarray

class OneDecisionCommandAssembler:
    def __init__(self, model_path=MODEL):
        self.metadata = load_canonical_metadata(model_path)
        self.station = AdapterStationMirror(self.metadata)
        self._actions = {}

    def build(self, *, flight_id, revision_id, command_seq, predictor_command,
              source_now_s, producer_wall_s, current_base_xy, normalized_action):
        sign, station = self.station.candidate_for(flight_id, predictor_command.position_world[:2], current_base_xy)
        action = np.asarray(normalized_action, dtype=np.float64)
        if flight_id in self._actions and not np.array_equal(action, self._actions[flight_id]):
            raise ValueError("mid-swing RL resampling is forbidden")
        self._actions.setdefault(flight_id, action.copy())
        velocity = map_normalized_velocity(action, sign, self.metadata)
        if not velocity_inside_planner_box(velocity, sign, self.metadata):
            raise RuntimeError("ACTION_CONTRACT_VIOLATION")
        obs = build_v2_observation(predictor_command, source_now_s, sign)
        tts = float(obs[5])
        packet = build_schema2_packet(valid=True, swing_sign=sign,
            position=predictor_command.position_world, velocity=velocity,
            control_tts_s=tts, producer_wall_s=producer_wall_s,
            command_seq=command_seq, flight_id=flight_id, revision_id=revision_id,
            estimator_sample_count=0, estimator_span_s=0.0)
        self.station.accept_candidate(flight_id, sign, station)
        return TargetCommand(obs, action, np.asarray(predictor_command.position_world), velocity, tts, sign, station, packet)
