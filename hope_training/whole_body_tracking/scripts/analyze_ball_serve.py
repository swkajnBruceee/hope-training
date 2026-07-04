"""Analyze scripted table-tennis ball serves in Isaac Sim.

Run after ``source setup_train_env.sh``:

    hope_isaac_py scripts/analyze_ball_serve.py --task HitFixedBaseTouch --num_envs 32 --steps 220 --headless

The script steps the environment with zero robot action and reports whether the
served ball bounces once on the table and then crosses the P1-side table edge
(``x <= 0``). This is a quick pre-training check for the incoming-ball
distribution.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``training`` importable regardless of how this script was launched.
# Paths are resolved relative to THIS FILE so the script is independent of the
# caller's PYTHONPATH / cwd / checkout location.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (
    _REPO_ROOT,
    os.path.normpath(os.path.join(_REPO_ROOT, "show")),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _HERE, _REPO_ROOT, _p

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Analyze table-tennis serve trajectories.")
parser.add_argument("--task", choices=["Base", "HitFixedBase", "HitFixedBaseTouch"], default="HitFixedBaseTouch")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=220)
parser.add_argument("--fix_base", action="store_true", default=True)
parser.add_argument("--hit_x", type=float, default=-0.25, help="P1-side x plane used as a nominal hit point.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _make_env_cfg(task: str):
    from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import (
        AgibotA3HitFixedBaseEnvCfg,
        AgibotA3HitFixedBaseTouchEnvCfg,
        AgibotA3TableTennisEnvCfg,
    )

    if task == "HitFixedBaseTouch":
        return "HOPE-TableTennis-AgibotA3-HitFixedBaseTouch-v0", AgibotA3HitFixedBaseTouchEnvCfg()
    if task == "HitFixedBase":
        return "HOPE-TableTennis-AgibotA3-HitFixedBase-v0", AgibotA3HitFixedBaseEnvCfg()
    return "HOPE-TableTennis-AgibotA3-v0", AgibotA3TableTennisEnvCfg()


def main() -> None:
    import gymnasium as gym
    import torch

    import training.tasks  # noqa: F401
    from training.tasks.table_tennis import geometry

    task_id, env_cfg = _make_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.fix_base:
        env_cfg.scene.robot.spawn.fix_base = True

    env = gym.make(task_id, cfg=env_cfg)
    obs, _ = env.reset()
    del obs

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    zero_action = torch.zeros(env.action_space.shape, device=device)

    ball = env.unwrapped.scene["ball"]
    env_origins = env.unwrapped.scene.env_origins

    radius = float(geometry.BALL_RADIUS)
    table_z = 0.0
    near_edge_x = 0.0
    far_edge_x = float(geometry.TABLE_LENGTH)
    y_min, y_max = -float(geometry.TABLE_WIDTH), 0.0

    prev_pos = ball.data.root_pos_w - env_origins
    prev_vel = ball.data.root_lin_vel_w.clone()

    first_bounce_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_bounce_x = torch.full((num_envs,), float("nan"), device=device)
    first_bounce_y = torch.full((num_envs,), float("nan"), device=device)
    crossed_near_edge = torch.zeros(num_envs, dtype=torch.bool, device=device)
    near_edge_z = torch.full((num_envs,), float("nan"), device=device)
    crossed_hit_plane = torch.zeros(num_envs, dtype=torch.bool, device=device)
    hit_plane_z = torch.full((num_envs,), float("nan"), device=device)
    hit_plane_vx = torch.full((num_envs,), float("nan"), device=device)
    min_x_after_bounce = torch.full((num_envs,), float("inf"), device=device)
    max_z_after_bounce = torch.full((num_envs,), float("-inf"), device=device)

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            env.step(zero_action)

        pos = ball.data.root_pos_w - env_origins
        vel = ball.data.root_lin_vel_w

        on_table_xy = (pos[:, 0] >= near_edge_x) & (pos[:, 0] <= far_edge_x) & (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max)
        table_contact_height = pos[:, 2] <= table_z + radius + 0.015
        bounce_now = (~first_bounce_seen) & on_table_xy & table_contact_height & (prev_vel[:, 2] < -0.15) & (vel[:, 2] > 0.15)

        first_bounce_x = torch.where(bounce_now, pos[:, 0], first_bounce_x)
        first_bounce_y = torch.where(bounce_now, pos[:, 1], first_bounce_y)
        first_bounce_seen |= bounce_now

        min_x_after_bounce = torch.where(first_bounce_seen, torch.minimum(min_x_after_bounce, pos[:, 0]), min_x_after_bounce)
        max_z_after_bounce = torch.where(first_bounce_seen, torch.maximum(max_z_after_bounce, pos[:, 2]), max_z_after_bounce)

        cross_now = first_bounce_seen & (~crossed_near_edge) & (prev_pos[:, 0] > near_edge_x) & (pos[:, 0] <= near_edge_x)
        denom = torch.clamp(prev_pos[:, 0] - pos[:, 0], min=1.0e-6)
        alpha = torch.clamp(prev_pos[:, 0] / denom, 0.0, 1.0)
        z_cross = prev_pos[:, 2] + alpha * (pos[:, 2] - prev_pos[:, 2])
        near_edge_z = torch.where(cross_now, z_cross, near_edge_z)
        crossed_near_edge |= cross_now

        hit_x = float(args_cli.hit_x)
        hit_now = first_bounce_seen & (~crossed_hit_plane) & (prev_pos[:, 0] > hit_x) & (pos[:, 0] <= hit_x)
        hit_denom = torch.clamp(prev_pos[:, 0] - pos[:, 0], min=1.0e-6)
        hit_alpha = torch.clamp((prev_pos[:, 0] - hit_x) / hit_denom, 0.0, 1.0)
        hit_z_cross = prev_pos[:, 2] + hit_alpha * (pos[:, 2] - prev_pos[:, 2])
        hit_vx_cross = prev_vel[:, 0] + hit_alpha * (vel[:, 0] - prev_vel[:, 0])
        hit_plane_z = torch.where(hit_now, hit_z_cross, hit_plane_z)
        hit_plane_vx = torch.where(hit_now, hit_vx_cross, hit_plane_vx)
        crossed_hit_plane |= hit_now

        prev_pos = pos.clone()
        prev_vel = vel.clone()

    valid_bounce = first_bounce_seen
    valid_cross = crossed_near_edge
    valid_hit = crossed_hit_plane
    p1_half_bounce = valid_bounce & (first_bounce_x < geometry.NET_X)
    returnable_cross = valid_cross & (near_edge_z > 0.12) & (near_edge_z < 1.20)
    returnable_hit = valid_hit & (hit_plane_z > 0.18) & (hit_plane_z < 1.20)

    def _mean(t: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(t[mask].mean().item())

    def _min(t: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(t[mask].min().item())

    def _max(t: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(t[mask].max().item())

    print(f"[serve] task={args_cli.task} envs={num_envs} steps={args_cli.steps}", flush=True)
    print(
        "[serve] first_bounce="
        f"{int(valid_bounce.sum().item())}/{num_envs}, "
        f"p1_half_bounce={int(p1_half_bounce.sum().item())}/{num_envs}, "
        f"crossed_near_edge={int(valid_cross.sum().item())}/{num_envs}, "
        f"returnable_cross={int(returnable_cross.sum().item())}/{num_envs}, "
        f"returnable_hit_plane={int(returnable_hit.sum().item())}/{num_envs}",
        flush=True,
    )
    print(
        "[serve] first_bounce_x "
        f"mean={_mean(first_bounce_x, valid_bounce):.3f}, "
        f"min={_min(first_bounce_x, valid_bounce):.3f}, "
        f"max={_max(first_bounce_x, valid_bounce):.3f}",
        flush=True,
    )
    print(
        "[serve] near_edge_z "
        f"mean={_mean(near_edge_z, valid_cross):.3f}, "
        f"min={_min(near_edge_z, valid_cross):.3f}, "
        f"max={_max(near_edge_z, valid_cross):.3f}",
        flush=True,
    )
    print(
        "[serve] after_bounce "
        f"min_x_mean={_mean(min_x_after_bounce, valid_bounce):.3f}, "
        f"max_z_mean={_mean(max_z_after_bounce, valid_bounce):.3f}",
        flush=True,
    )
    print(
        f"[serve] hit_plane_x={args_cli.hit_x:.3f} "
        f"z_mean={_mean(hit_plane_z, valid_hit):.3f}, "
        f"z_min={_min(hit_plane_z, valid_hit):.3f}, "
        f"z_max={_max(hit_plane_z, valid_hit):.3f}, "
        f"vx_mean={_mean(hit_plane_vx, valid_hit):.3f}",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
