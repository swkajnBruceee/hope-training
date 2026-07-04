"""Headless probe: verify the corrected exact-strike success metric vs the RAW conditional pass rate.

Loads a trained checkpoint + local motion clip (no wandb), steps the env, and each step pairs the
per-env exact-strike mask with the instantaneous racket errors to recompute the RAW conditional
exact-strike pass rates. It then compares those to the LOGGED metric tensor the curriculum reads:

    hope_isaac_py scripts/probe_metric.py task=HOPEPingPong algo=ppo headless=true num_envs=512 \
        checkpoint=logs/rsl_rl/agibot_a3_hope/2026-06-25_08-10-54_pathA_basecouple/model_4400.pt \
        motion_file=artifacts/hope_forehand:v0/motion.npz steps=800

Confirms: (1) logged strike_composite_success_exact ~= raw exact composite, (2) pos/vel/normal logged
separately, (3) the success-gated ref_perturb_scale can move off its start value.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Also make ``training`` (and the ``trajectory`` overlay, resolvable as ``from show.trajectory``) importable regardless
# of how this script was launched. Paths are relative to THIS FILE so the
# script is independent of PYTHONPATH / cwd / checkout location.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (
    _REPO_ROOT,
    os.path.normpath(os.path.join(_REPO_ROOT, "show")),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _REPO_ROOT, _p

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _run(cfg, simulation_app):
    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import training.tasks  # noqa: F401  -- registers the gym tasks
    from training.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.commands.motion.motion_file = str(cfg.motion_file)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(cfg.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd = env.unwrapped.command_manager.get_term("racket_target")
    pos_thr = float(cmd.cfg.strike_success_pos_thresh)
    vel_thr = float(cmd.cfg.strike_success_vel_thresh)
    nrm_thr = float(cmd.cfg.strike_success_normal_thresh_deg)
    step_dt = float(env.unwrapped.step_dt)
    motion_term = cmd._motion()
    total = max(int(motion_term.motion.time_step_total), 1)
    strike_step = round(cmd.cfg.strike_phase * (total - 1))

    # accumulators over exact-strike samples, for TWO masks:
    #   stale = the in-code exact_strike (uses self.time_to_strike, 1 step lagged)
    #   fresh = recomputed from the CURRENT motion phase (aligned with the current racket FK)
    acc = {k: dict(n=0, pos=0, vel=0, nrm=0, comp=0) for k in ("stale", "fresh")}
    scale_start = float(cmd._curr_perturb_scale)
    n_steps = int(cfg.get("steps", 800))

    def tally(which, mask, m):
        k = int(mask.sum())
        if not k:
            return
        pp = m["racket_pos_error"][mask] < pos_thr
        vp = m["racket_vel_error"][mask] < vel_thr
        npass = m["racket_normal_error_deg"][mask] < nrm_thr
        a = acc[which]
        a["n"] += k
        a["pos"] += int(pp.sum()); a["vel"] += int(vp.sum()); a["nrm"] += int(npass.sum())
        a["comp"] += int((pp & vp & npass).sum())

    obs = env.get_observations().to(agent_cfg.device)
    step = 0
    while simulation_app.is_running() and step < n_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
        m = cmd.metrics
        tally("stale", m["exact_strike_hit_rate"] > 0.5, m)
        fresh_tts = (strike_step - motion_term.time_steps).float() * step_dt
        tally("fresh", fresh_tts.abs() <= (0.5 * step_dt + 1e-6), m)
        step += 1

    def rate(a, key):
        return acc[a][key] / acc[a]["n"] if acc[a]["n"] else float("nan")

    logged_comp = float(cmd.metrics["strike_composite_success_exact"][0])
    logged_pos = float(cmd.metrics["strike_pos_pass_exact"][0])
    logged_vel = float(cmd.metrics["strike_vel_pass_exact"][0])
    logged_nrm = float(cmd.metrics["strike_normal_pass_exact"][0])
    logged_decN = float(cmd.metrics["exact_strike_sample_count_decayed"][0])
    scale_end = float(cmd._curr_perturb_scale)

    print("\n" + "=" * 72, flush=True)
    print(f"PROBE over {step} steps, {num_envs} envs  |  exact samples stale={acc['stale']['n']} "
          f"fresh={acc['fresh']['n']}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'metric':<22}{'RAW stale':>12}{'RAW fresh':>12}{'LOGGED':>12}", flush=True)
    print(f"{'pos < %.3g m' % pos_thr:<22}{rate('stale','pos'):>12.4f}{rate('fresh','pos'):>12.4f}{logged_pos:>12.4f}", flush=True)
    print(f"{'vel < %.3g m/s' % vel_thr:<22}{rate('stale','vel'):>12.4f}{rate('fresh','vel'):>12.4f}{logged_vel:>12.4f}", flush=True)
    print(f"{'normal < %.3g deg' % nrm_thr:<22}{rate('stale','nrm'):>12.4f}{rate('fresh','nrm'):>12.4f}{logged_nrm:>12.4f}", flush=True)
    print(f"{'composite (all 3)':<22}{rate('stale','comp'):>12.4f}{rate('fresh','comp'):>12.4f}{logged_comp:>12.4f}", flush=True)
    print("-" * 72, flush=True)
    print(f"decayed exact-strike sample count (logged): {logged_decN:.1f}", flush=True)
    print(f"ref_perturb_scale: start={scale_start:.4f}  end={scale_end:.4f}  "
          f"moved={'YES' if scale_end > scale_start + 1e-9 else 'no'}", flush=True)
    print(f"advance_threshold={cmd.cfg.ref_perturb_advance_threshold}  "
          f"success_gated={cmd.cfg.ref_perturb_success_gated}", flush=True)
    print("=" * 72 + "\n", flush=True)

    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        _run(cfg, simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
