# v13b training-contract source

These files record the training-side contract used by the runtime package. They
are included as source, not as a claim that IsaacLab and MuJoCo are interchangeable.

Important entry points:

- `training/tasks/base_locomotion/mdp/actions.py`: actor loading,
  normalizer application, prior reconstruction, alpha blending, READY bridge,
  lookahead and velocity feed-forward.
- `training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py`: v13b
  98D/26D public actor and private 126D/56D prior configuration.
- `training/tasks/tracking/mdp/hope_observations.py` and `observations.py`:
  frame and target observation functions.
- `training/robots/agibot_a3.py`: canonical A3 joint lists.
- `training/utils/v13b_ready_stance.py`: right-front READY construction.

This snapshot still imports IsaacLab and the rest of the training project.
Use `runtime/` for portable checkpoint inference; use this snapshot when
implementing the MuJoCo state/trajectory adapter.
