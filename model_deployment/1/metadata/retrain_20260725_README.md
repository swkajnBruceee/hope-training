# 20260725 Floating-Base Retraining Package

This directory is the immutable entry point for the current-contract
retraining chain. It is separate from the historical `model_3396` experiment.
The historical logs and checkpoints are archived at:

```text
/workspace/hopetmp/reproducibility_archives/model_3396_lineage_20260725
```

## Scope

The current target is the six-motion backhand strike-only set:

```text
sample_motions/p2_data260708_backhand_strike_only_v1/manifest.json
```

Each source motion is retained through its hit frame and has an explicit
zero-velocity tail. The tail is not a follow-through motion and must not be
treated as a learned post-hit action.

The current contract is:

```text
floating base: enabled
root frame: corrected 180-degree frame, quaternion (0, 0, 0, 1)
ready pose: validated slight knee/hip flexion
root work point: (3.15, -0.35, 1.04) m
upper policy: frozen fixed-base backhand model_900
lower policy: 12 leg residual channels in the public 14-D Base contract
target-driven root movement: disabled
random start phase: disabled
PD-gain randomization: disabled for migration
```

## Stage Order

The functional sequence is the same as the historical successful stabilizer
training, but the old manifests and old root contract are not reused.

| Stage | Purpose | Initial condition | PPO config |
|---|---|---|---|
| Fresh Stage-A | Learn leg stability during the swing and final hold | zero actor mean | `ppo_retrain_stage_a` |
| Return-C1 | Add smooth return-to-ready after the swing | Fresh checkpoint | `ppo_retrain_return` |
| Robust-B | Add reset perturbations and hard cases | Return checkpoint | `ppo_retrain_robust_b` |
| F1 | Preserve fixed-base strike accuracy on the floating base | Robust-B checkpoint | `ppo_retrain_f1` |

Fresh, Return and Robust-B are the required current six-backhand chain. The
historical K8/K17 expansion remains archived evidence; it must not be inserted
before the current corrected six-motion chain passes its gates.

## Exact Historical Settings Preserved

The first three stages retain the historical logged values:

```text
num_steps_per_env: 64
actor/critic: [256, 128, 64], ELU
init_noise_std: 0.08
learning_rate: 0.0003
adaptive schedule, desired_kl=0.01
entropy_coef=0.002
epochs=3, minibatches=8
gamma=0.99, lambda=0.95
clip=0.2, value_coef=1.0, max_grad_norm=1.0
normalize_advantage_per_mini_batch=false
```

Stage lengths and motion schedules are:

```text
Fresh: 2000 iterations, 128 envs, save every 25;
       prelude 50, hold 150, return 0
Return: 600 iterations, 128 envs, save every 25;
        prelude 50, hold 25, return 50
Robust-B: 300 iterations, 128 envs, save every 50;
          same return schedule + 0.50 reset perturbation + 0.40 hard case
F1: 3000 iterations, 128 envs, save every 100;
    prelude 50, no post-hit hold/return, strike-preservation rewards enabled
```

## Launch

Run from `hope_training/whole_body_tracking`:

```bash
bash retrain_20260725/run_stage.sh fresh
bash retrain_20260725/run_stage.sh return /absolute/path/to/fresh/model_1000.pt
bash retrain_20260725/run_stage.sh robust_b /absolute/path/to/return/model_1599.pt
bash retrain_20260725/run_stage.sh f1 /absolute/path/to/robust_b/model_1898.pt
```

The wrapper records the exact command, Git status, current diff, input hashes,
and checkpoint hash before every run. It refuses a missing checkpoint and does
not overwrite an existing run directory.

## Promotion Gates

Do not continue automatically after a stage. Preserve the stage output and
check:

```text
Fresh: no fall, all six motions complete, leg action is finite and bounded
Return: no fall and return-to-ready is stable
Robust-B: no fall under perturbation, no persistent root drift or foot slip
F1: floating strike error improves or stays below the Float-zero baseline;
    no target-driven root translation, stepping, or action saturation
```

Every run must retain `params/env.yaml`, `params/agent.yaml`, TensorBoard
events, checkpoint files, and the wrapper metadata directory. Never delete an
earlier stage to save disk space.

## Historical Lineage

The exact old `model_3396` weight chain is recorded in
`lineage/model_3396_lineage.json`. It is preserved for diagnosis only and is
not a warm start for this retraining package because its root/observation/data
contract is different.
