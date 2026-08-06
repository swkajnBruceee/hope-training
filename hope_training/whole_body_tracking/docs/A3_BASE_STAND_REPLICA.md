# A3 Base Stand Replica

`A3BaseStandReplica-v0` is an isolated diagnostic line that adapts the Isaac Lab H1 flat velocity-locomotion MDP to the existing A3 physics asset. It is not a replacement for `A3BaseStand-v0`, and it has no authority to alter the Base locomotion contract or deployment gates.

The replica retains the same A3 asset, reset posture, contact bodies, 200 Hz physics / 50 Hz policy transport, leg PD gains, and per-leg residual scales. It changes only the experiment-local MDP and runner:

- 12 leg actions; waist and all upper-body joints are held at their reset posture.
- 45-dimensional policy observation: base angular velocity, projected gravity, zero velocity command, leg joint position/velocity, and previous action.
- H1-flat-style core reward terms: zero-command velocity tracking, termination penalty, flat orientation, leg acceleration/action-rate, foot sliding, ankle limits, and hip posture.
- No randomization, terrain curriculum, pushes, strike reference, Composer, 925-dimensional history, or Base Stand checkpoint reuse.

Run a short, fixed-seed smoke independently of the main line:

```bash
source setup_train_env.sh
hope_isaac_py scripts/train.py task=A3BaseStandReplica algo=a3_base_stand_replica_ppo headless=true seed=0 max_iterations=100
```

Use a separate output/checkpoint run for this task. Compare it with `A3BaseStand-v0` only after matching the A3 asset, reset posture, physics timestep, decimation, seed, environment count, and iteration count.
