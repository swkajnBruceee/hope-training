"""Hydra eval/export entry for HOPE Agibot A3 WBC.

    python scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
        checkpoint=logs/rsl_rl/agibot_a3_hope/<run>/model_*.pt

Loads a trained policy from a local checkpoint or optional WandB run, runs it,
and exports policy.onnx next to the checkpoint.
"""

import os
import sys

# allow `from train import _apply_task_overrides` (sibling script; no isaaclab imported at its top)
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

from train import _apply_task_overrides, _as_bool


def _run_play(cfg, simulation_app):
    import pathlib
    import copy
    import dataclasses
    import json

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import training.tasks  # noqa: F401  -- registers the gym tasks
    from training.tasks.table_tennis.mdp.racket import racket_normal_w, racket_state_w
    from training.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
    from training.utils.ppo_cfg import runner_kwargs

    def _obs_to_device(obs, device):
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs.to(device)

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)
    # Keep visual replay aligned with train.py's deterministic paired audits.
    env_cfg.seed = int(cfg.seed)

    # Optional diagnostic table: extend the *original training scene* with a
    # static tabletop.  Do not replace the training scene with the match scene,
    # because that changes the floor height, reset contract, and physics setup.
    table_offset_x = cfg.get("table_offset_x", None)
    table_z_offset = float(cfg.get("table_z_offset", 0.76) or 0.0)
    full_table = bool(cfg.get("full_table", False))
    ball_enabled = bool(cfg.get("ball", False))
    if table_offset_x is not None or full_table or ball_enabled:
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils import configclass
        from training.tasks.table_tennis import geometry
        from training.tasks.table_tennis.table_tennis_env_cfg import (
            _MATS,
            build_center_line_cfg,
            build_net_cfg,
            build_net_post_cfg,
            build_table_top_cfg,
        )

        base_scene = env_cfg.scene
        base_scene_type = type(base_scene)

        racket_proxy_enabled = ball_enabled and bool(cfg.get("ball_racket_proxy", False))
        if ball_enabled:
            from training.tasks.table_tennis.table_tennis_env_cfg import build_ball_cfg

            @configclass
            class TrainingSceneWithTableCfg(base_scene_type):
                table: AssetBaseCfg = build_table_top_cfg(_MATS)
                ball: RigidObjectCfg = build_ball_cfg(_MATS)
                if racket_proxy_enabled:
                    racket_proxy: RigidObjectCfg = RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/RacketProxy",
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 10.0, 5.0)),
                        spawn=sim_utils.CuboidCfg(
                            # The imported A3 blade is approximately 160 mm
                            # square and only 2.9 mm thick.  A slightly thicker
                            # collider makes PhysX CCD robust while remaining
                            # faithful to the blade face.
                            size=(0.1604, 0.0060, 0.1604),
                            visible=False,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                kinematic_enabled=True,
                                disable_gravity=True,
                                max_depenetration_velocity=10.0,
                            ),
                            collision_props=sim_utils.CollisionPropertiesCfg(
                                collision_enabled=True,
                                contact_offset=0.002,
                                rest_offset=0.0,
                            ),
                            physics_material=sim_utils.RigidBodyMaterialCfg(
                                # The reverse solver assumes that the face
                                # does not brake tangential motion.  The ball
                                # material combines multiplicatively, so use
                                # ~0.93 here to obtain an effective
                                # ball/blade restitution near 0.842.
                                static_friction=0.0,
                                dynamic_friction=0.0,
                                restitution=0.93,
                                friction_combine_mode="multiply",
                                restitution_combine_mode="multiply",
                            ),
                        ),
                    )
                racket_ball_contact: ContactSensorCfg = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Ball",
                    update_period=0.0,
                    history_length=4,
                    track_air_time=True,
                    force_threshold=0.05,
                    debug_vis=False,
                )
        else:
            @configclass
            class TrainingSceneWithTableCfg(base_scene_type):
                table: AssetBaseCfg = build_table_top_cfg(_MATS)

        scene = TrainingSceneWithTableCfg(
            num_envs=int(base_scene.num_envs),
            env_spacing=float(base_scene.env_spacing),
        )
        for field in dataclasses.fields(base_scene):
            if field.name not in ("num_envs", "env_spacing"):
                setattr(scene, field.name, copy.deepcopy(getattr(base_scene, field.name)))
        table_cfg = build_table_top_cfg(_MATS)
        table_pos = list(table_cfg.init_state.pos)
        table_pos[0] += float(table_offset_x or 0.0)
        table_pos[2] += table_z_offset
        table_cfg.init_state.pos = tuple(table_pos)
        scene.table = table_cfg
        if ball_enabled:
            ball_cfg = build_ball_cfg(_MATS)
            ball_pos = list(ball_cfg.init_state.pos)
            ball_pos[0] += float(table_offset_x or 0.0)
            ball_pos[2] += table_z_offset
            ball_cfg.init_state.pos = tuple(ball_pos)
            scene.ball = ball_cfg
            if racket_proxy_enabled:
                proxy_cfg = copy.deepcopy(scene.racket_proxy)
                proxy_pos = list(proxy_cfg.init_state.pos)
                proxy_pos[0] += float(table_offset_x or 0.0)
                proxy_pos[2] += table_z_offset
                proxy_cfg.init_state.pos = tuple(proxy_pos)
                scene.racket_proxy = proxy_cfg
        if full_table:
            for name, builder in (
                ("net", lambda: build_net_cfg(_MATS)),
                ("center_line", build_center_line_cfg),
            ):
                asset_cfg = builder()
                asset_pos = list(asset_cfg.init_state.pos)
                asset_pos[0] += float(table_offset_x or 0.0)
                asset_pos[2] += table_z_offset
                asset_cfg.init_state.pos = tuple(asset_pos)
                setattr(scene, name, asset_cfg)
            for name, y in (
                ("net_post_left", geometry.NET_OVERHANG),
                ("net_post_right", -geometry.TABLE_WIDTH - geometry.NET_OVERHANG),
            ):
                asset_cfg = build_net_post_cfg(f"{{ENV_REGEX_NS}}/{name}", y)
                asset_pos = list(asset_cfg.init_state.pos)
                asset_pos[0] += float(table_offset_x or 0.0)
                asset_pos[2] += table_z_offset
                asset_cfg.init_state.pos = tuple(asset_pos)
                setattr(scene, name, asset_cfg)
        env_cfg.scene = scene
        print(
            f"[INFO] added {'full ' if full_table else ''}training-scene table "
            f"with x offset={float(table_offset_x or 0.0):+.3f} m, "
            f"z offset={table_z_offset:+.3f} m, ball={'on' if ball_enabled else 'off'}",
            flush=True,
        )
    has_motion_command = hasattr(env_cfg.commands, "motion")

    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name)))
    agent_cfg.device = str(cfg.device)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # resolve the checkpoint + reference motion
    wandb_path = cfg.wandb_path
    checkpoint = cfg.get("checkpoint", None)
    if wandb_path and not checkpoint:
        import wandb

        wandb_path = str(wandb_path)
        run_path = "/".join(wandb_path.split("/")[:-1]) if "model" in wandb_path else wandb_path
        api = wandb.Api()
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        fname = wandb_path.split("/")[-1] if "model" in wandb_path else max(
            files, key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        wandb_run.file(str(fname)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{fname}"
        print(f"[INFO] Loading model checkpoint from: {run_path}/{fname}")
        if has_motion_command and cfg.get("motion_manifest", None) is not None:
            env_cfg.commands.motion.motion_manifest = str(cfg.motion_manifest)
            env_cfg.commands.motion.motion_file = None
            if cfg.get("manifest_subset_size", None) is not None:
                env_cfg.commands.motion.manifest_subset_size = int(cfg.manifest_subset_size)
            if cfg.get("manifest_frame_z_offset", None) is not None:
                env_cfg.commands.motion.manifest_frame_z_offset = float(cfg.manifest_frame_z_offset)
            if _as_bool(cfg.get("validate_stance_contract", False)):
                env_cfg.commands.motion.validate_stance_contract = True
                stance_mode = cfg.get("stance_contract_mode", None)
                if stance_mode is not None:
                    env_cfg.commands.motion.stance_contract_mode = str(stance_mode)
        elif has_motion_command and cfg.motion_file is not None:
            env_cfg.commands.motion.motion_file = str(cfg.motion_file)
        elif has_motion_command:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is not None:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
            else:
                print("[WARN] No motion artifact in the run; pass motion_file=... if replay fails.")
    else:
        if checkpoint:
            resume_path = str(checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        reg = cfg.registry_name if cfg.registry_name is not None else cfg.task.get("registry_name")
        motion_manifest = cfg.get("motion_manifest", None)
        if motion_manifest is None:
            motion_manifest = cfg.task.get("motion_manifest")
        if has_motion_command and motion_manifest is not None:
            manifest_path = pathlib.Path(str(motion_manifest)).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = pathlib.Path.cwd() / manifest_path
            env_cfg.commands.motion.motion_manifest = str(manifest_path)
            env_cfg.commands.motion.motion_file = None
            subset_size = cfg.get("manifest_subset_size", None)
            if subset_size is None:
                subset_size = cfg.task.get("manifest_subset_size")
            if subset_size is not None:
                env_cfg.commands.motion.manifest_subset_size = int(subset_size)
            frame_z_offset = cfg.get("manifest_frame_z_offset", None)
            if frame_z_offset is None:
                frame_z_offset = cfg.task.get("manifest_frame_z_offset")
            if frame_z_offset is not None:
                env_cfg.commands.motion.manifest_frame_z_offset = float(frame_z_offset)
            if _as_bool(cfg.get("validate_stance_contract", False)):
                env_cfg.commands.motion.validate_stance_contract = True
                stance_mode = cfg.get("stance_contract_mode", None)
                if stance_mode is not None:
                    env_cfg.commands.motion.stance_contract_mode = str(stance_mode)
            print(
                f"[INFO] using local motion_manifest: {manifest_path} "
                f"(subset_size={env_cfg.commands.motion.manifest_subset_size}, "
                f"frame_z_offset={env_cfg.commands.motion.manifest_frame_z_offset:.4f}m)",
                flush=True,
            )
        elif has_motion_command and cfg.motion_file is not None:
            env_cfg.commands.motion.motion_file = str(cfg.motion_file)
        elif has_motion_command and reg is not None:
            import wandb

            reg = str(reg)
            if ":" not in reg:
                reg += ":latest"
            art = wandb.Api().artifact(reg)
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
        elif not has_motion_command:
            print("[INFO] env has no motion command; replaying pure RL policy without motion source.")

    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    if bool(cfg.get("ball_racket_proxy", False)) and "racket_proxy" in env.unwrapped.scene.rigid_objects:
        # The proxy overlaps the imported wrist/racket assembly by design.  A
        # filtered pair prevents the proxy from pushing the articulated robot,
        # while keeping its collision with the independent dynamic ball.
        try:
            from pxr import UsdPhysics

            stage = env.unwrapped.sim.stage
            filtered = 0
            for env_id in range(int(env.unwrapped.num_envs)):
                proxy_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/RacketProxy")
                ball_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Ball")
                robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
                if proxy_prim.IsValid() and robot_prim.IsValid():
                    api = UsdPhysics.FilteredPairsAPI.Apply(proxy_prim)
                    api.CreateFilteredPairsRel().AddTarget(robot_prim.GetPath())
                    filtered += 1
                if ball_prim.IsValid() and robot_prim.IsValid():
                    api = UsdPhysics.FilteredPairsAPI.Apply(ball_prim)
                    api.CreateFilteredPairsRel().AddTarget(robot_prim.GetPath())
                    filtered += 1
            print(f"[INFO] filtered ball/proxy<->Robot collision pairs: {filtered}", flush=True)

            # The imported URDF blade is a very thin mesh and its edge
            # collider can intercept the ball before the proxy face.  In
            # proxy mode make the proxy the sole racket collider by disabling
            # only the original blade collision meshes; hand/arm and foot
            # collisions remain untouched.
            disabled_blade_collisions = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if "/World/envs/" not in path or "/Robot/" not in path:
                    continue
                if "pingpang_red_Link" not in path and "pingpang_black_Link" not in path:
                    continue
                collision_attr = prim.GetAttribute("physics:collisionEnabled")
                if not collision_attr:
                    collision_attr = UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr()
                collision_attr.Set(False)
                disabled_blade_collisions += 1
            print(
                f"[INFO] disabled original racket blade collision meshes: {disabled_blade_collisions}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError("failed to install RacketProxy<->Robot collision filter") from exc
    camera_eye = cfg.get("camera_eye", None)
    camera_lookat = cfg.get("camera_lookat", None)
    if camera_eye is not None and camera_lookat is not None:
        env.unwrapped.sim.set_camera_view(eye=list(camera_eye), target=list(camera_lookat))
    elif cfg.video:
        viewer_cfg = getattr(env_cfg, "viewer", None)
        if viewer_cfg is not None:
            env.unwrapped.sim.set_camera_view(eye=viewer_cfg.eye, target=viewer_cfg.lookat)
    log_dir = os.path.dirname(resume_path)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    forced_motion_id = cfg.get("motion_id", None)
    if has_motion_command and forced_motion_id is not None:
        motion = env.unwrapped.command_manager.get_term("motion")
        motion_id = int(forced_motion_id)
        if not 0 <= motion_id < motion.motion.num_motions:
            raise ValueError(
                f"motion_id={motion_id} outside manifest range [0, {motion.motion.num_motions - 1}]"
            )
        # The wrapped env has already performed the task's physical
        # strike-ready reset.  Only replace the future reference and reset its
        # phase bookkeeping; do not teleport the floating base into a motion
        # frame, which would create an artificial reset seam in the video.
        motion.motion_ids.fill_(motion_id)
        motion.time_steps.zero_()
        motion.tail_steps.zero_()
        motion.prelude_elapsed_steps.zero_()
        env.unwrapped.strike_stabilizer_handoff_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.unwrapped.device
        )
        env_ids = torch.arange(env.num_envs, device=env.unwrapped.device)
        racket = env.unwrapped.command_manager.get_term("racket_target")
        racket._resample_command(env_ids)
        racket._compute_strike_timing()
        # A forced-motion replay must synchronize the physical robot to the
        # selected motion's frame zero.  Merely changing motion_ids after the
        # environment reset leaves the robot at the default ready pose while
        # the command/reference jumps to the selected clip, producing a large
        # artificial racket-position error and an invalid policy observation.
        with torch.inference_mode():
            motion_cmd = env.unwrapped.command_manager.get_term("motion")
            motion_cmd._prev_motion_steps = motion_cmd.time_steps.clone()
            joint_pos = motion_cmd.joint_pos.clone()
            joint_vel = motion_cmd.joint_vel.clone()
            root_pos, root_ori, root_lin_vel, root_ang_vel = motion_cmd._motion_root_state_w()
            motion_cmd.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            motion_cmd.robot.write_root_state_to_sim(
                torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
            )
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()
            motion_cmd._update_command()
        print(f"[INFO] forced manifest motion_id={motion_id} for deterministic replay", flush=True)

    # Optional dynamic-ball validation.  This deliberately runs inside the
    # original tracking environment: same floor, frame offset, reset, PD and
    # policy contract as training.  First calibrate the actual racket state at
    # the selected strike time, then rewind and launch the ball toward it.
    ball_plan = None
    racket_proxy_enabled = False
    if ball_enabled:
        if not has_motion_command or forced_motion_id is None:
            raise ValueError("ball=true requires a motion command and an explicit motion_id")
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        ball_motion_id = int(forced_motion_id)
        motion_manifest_path = pathlib.Path(str(env_cfg.commands.motion.motion_manifest)).expanduser()
        if not motion_manifest_path.is_absolute():
            motion_manifest_path = pathlib.Path.cwd() / motion_manifest_path
        manifest_data = json.loads(motion_manifest_path.read_text(encoding="utf-8"))
        manifest_entry = manifest_data.get("motions", [])[ball_motion_id]
        hit_time_override = cfg.get("ball_hit_time", None)
        source_hit_time = float(
            hit_time_override
            if hit_time_override is not None
            else manifest_entry.get("hit_event", {}).get("hit_time_from_start_s", 0.6)
        )
        prelude_steps = int(getattr(motion_cmd, "prelude_steps", env_cfg.commands.motion.prelude_steps))
        control_dt = float(env.unwrapped.step_dt)
        effective_hit_time = source_hit_time + prelude_steps * control_dt
        # In natural-contact mode the replay metrics are sampled after the
        # policy step.  Calibrate the racket at that post-step strike frame.
        if bool(cfg.get("ball_natural_racket_return", False)):
            effective_hit_time += control_dt
        flight_time = min(float(cfg.get("ball_flight_time", 0.10)), effective_hit_time - control_dt)
        incoming_velocity = torch.tensor(
            [
                float(cfg.get("ball_incoming_vx", 2.85075)),
                float(cfg.get("ball_incoming_vy", -0.51961)),
                float(cfg.get("ball_incoming_vz", 0.22545)),
            ],
            device=env.unwrapped.device,
            dtype=torch.float32,
        )

        def _sync_ball_motion():
            with torch.inference_mode():
                motion_cmd.motion_ids.fill_(ball_motion_id)
                motion_cmd.time_steps.zero_()
                motion_cmd.tail_steps.zero_()
                motion_cmd.prelude_elapsed_steps.zero_()
                motion_cmd._prev_motion_steps = motion_cmd.time_steps.clone()
                ids = torch.arange(env.num_envs, device=env.unwrapped.device)
                joint_pos = motion_cmd.joint_pos.clone()
                joint_vel = motion_cmd.joint_vel.clone()
                root_pos, root_ori, root_lin_vel, root_ang_vel = motion_cmd._motion_root_state_w()
                motion_cmd.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
                motion_cmd.robot.write_root_state_to_sim(
                    torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=ids
                )
                racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
                racket_cmd._resample_command(ids)
                racket_cmd._compute_strike_timing()
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.forward()
                motion_cmd._update_command()

        def _park_ball():
            """Keep the diagnostic ball off-court until its one-shot launch step."""
            ball = env.unwrapped.scene["ball"]
            park_pose = torch.tensor(
                [[10.0, 10.0, 5.0, 1.0, 0.0, 0.0, 0.0]],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            park_velocity = torch.zeros((1, 6), device=env.unwrapped.device, dtype=torch.float32)
            ball.write_root_pose_to_sim(park_pose)
            ball.write_root_velocity_to_sim(park_velocity)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()

        racket_proxy_enabled = "racket_proxy" in env.unwrapped.scene.rigid_objects

        def _park_racket_proxy():
            if not racket_proxy_enabled:
                return
            proxy = env.unwrapped.scene["racket_proxy"]
            park_pose = torch.tensor(
                [[10.0, 10.0, 5.0, 1.0, 0.0, 0.0, 0.0]],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            park_velocity = torch.zeros((1, 6), device=env.unwrapped.device, dtype=torch.float32)
            proxy.write_root_pose_to_sim(park_pose)
            proxy.write_root_velocity_to_sim(park_velocity)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()

        # Calibration pass without the ball.
        env.reset()
        _sync_ball_motion()
        _park_ball()
        _park_racket_proxy()
        obs_cal = _obs_to_device(env.get_observations(), agent_cfg.device)
        hit_step = int(round(effective_hit_time / control_dt))
        actual_racket_pos = None
        actual_racket_vel = None
        actual_racket_normal = None
        for cal_step in range(hit_step + 1):
            if cal_step == hit_step:
                actual_racket_pos, actual_racket_vel, _ = racket_state_w(env.unwrapped)
                actual_racket_normal = racket_normal_w(env.unwrapped, normal_axis=1, normal_sign=1.0)
                actual_racket_pos = actual_racket_pos[0].detach().clone()
                actual_racket_vel = actual_racket_vel[0].detach().clone()
                actual_racket_normal = actual_racket_normal[0].detach().clone()
            actions_cal = policy(obs_cal)
            obs_cal, _, _, _ = env.step(actions_cal.to(env.unwrapped.device))
            obs_cal = _obs_to_device(obs_cal, agent_cfg.device)

        gravity = torch.tensor([0.0, 0.0, -9.81], device=env.unwrapped.device, dtype=torch.float32)
        hit_pos = actual_racket_pos
        # The calibrated FK point is the racket center, not the contact point
        # on its front face.  For a physical ball/racket collision the ball
        # center must arrive one ball-radius (plus a small blade-thickness
        # margin) on the incoming side of that face.  Aiming at the center
        # leaves the ball several centimeters away from the actual collider.
        racket_contact_offset = float(
            cfg.get("ball_racket_contact_offset", 0.017 if racket_proxy_enabled else 0.024)
        )
        racket_contact_pos = hit_pos - actual_racket_normal * racket_contact_offset
        scripted_racket_return = bool(cfg.get("ball_scripted_racket_return", False))
        natural_racket_return = bool(cfg.get("ball_natural_racket_return", False))
        return_time = float(cfg.get("ball_return_time", 0.45))
        return_target = torch.tensor(
            [
                float(cfg.get("ball_return_target_x", 0.85)),
                float(cfg.get("ball_return_target_y", -0.7625)),
                table_z_offset + 0.02,
            ],
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        return_velocity = (
            return_target - racket_contact_pos - 0.5 * gravity * return_time * return_time
        ) / return_time
        incoming_contact_velocity = None
        if natural_racket_return:
            # Reverse the simplified ball/racket impact model.  Let n point
            # from the incoming side through the racket face.  Relative
            # normal speed reverses with restitution; tangential speed keeps a
            # calibrated fraction.  This gives the required pre-impact ball
            # velocity from a desired outgoing flight.
            racket_restitution = float(cfg.get("ball_racket_restitution", 0.842))
            tangential_retention = float(cfg.get("ball_racket_tangential_retention", 0.649))
            out_rel = return_velocity - actual_racket_vel
            out_normal = torch.dot(out_rel, actual_racket_normal)
            out_tangent = out_rel - out_normal * actual_racket_normal
            incoming_contact_velocity = (
                actual_racket_vel
                + out_tangent / max(tangential_retention, 1.0e-3)
                - (out_normal / max(racket_restitution, 1.0e-3)) * actual_racket_normal
            )
        bounce_once = bool(cfg.get("ball_bounce_once", False))
        bounce_pos = None
        if bounce_once:
            # Construct a two-segment ballistic path: one controlled bounce on
            # the robot's half, then an ascending flight into the end-of-swing
            # racket pose.  This avoids the invalid diagnostic where the ball
            # appears on the table long before the intended serve.
            post_bounce_time = float(cfg.get("ball_bounce_time_before_hit", 0.33))
            pre_bounce_time = float(cfg.get("ball_pre_bounce_time", 0.30))
            if post_bounce_time <= 0.05 or pre_bounce_time <= 0.05:
                raise ValueError("bounce times must be > 0.05 s")
            # The table/ball drag and contact friction reduce the horizontal
            # travel after the bounce.  Place the first bounce closer to the
            # net so the simulated post-bounce flight reaches the end-swing
            # racket pose instead of stopping short of it.
            bounce_x = float(cfg.get("ball_bounce_x", float(hit_pos[0].item() - 0.62)))
            bounce_y = float(cfg.get("ball_bounce_y", float(hit_pos[1].item() - 0.22)))
            if natural_racket_return:
                # Choose the rising root that places the computed incoming
                # velocity exactly at the racket after one table bounce.
                bounce_z = table_z_offset + 0.02
                vertical_gap = float((racket_contact_pos[2] - bounce_z).item())
                vin_z = float(incoming_contact_velocity[2].item())
                # ``vin_z`` is the vertical velocity required *at the
                # racket*, not immediately after the table bounce.  For a
                # rising segment with gravity g=-9.81, the displacement is
                #
                #     dz = vin_z * t + 0.5 * 9.81 * t^2
                #
                # because the velocity at the bounce is
                # ``vin_z + 9.81*t``.  The previous expression solved the
                # equation for an initial velocity and therefore made the
                # ball overshoot the racket vertically by roughly 10 cm.
                discriminant = vin_z * vin_z + 2.0 * 9.81 * vertical_gap
                if discriminant <= 0.0:
                    raise ValueError(
                        f"natural racket solve has no rising bounce root: vin_z={vin_z:.3f}, "
                        f"vertical_gap={vertical_gap:.3f}"
                    )
                post_bounce_time = (-vin_z + discriminant**0.5) / 9.81
                if post_bounce_time <= 0.05:
                    # The second root is negative for the usual rising
                    # racket geometry.  Keep the guard explicit so an
                    # unusual configuration fails loudly instead of creating
                    # a backwards-time launch.
                    raise ValueError(
                        f"natural racket solve produced too-short bounce flight: "
                        f"vin_z={vin_z:.3f}, vertical_gap={vertical_gap:.3f}, "
                        f"post_bounce_time={post_bounce_time:.4f}s"
                    )
                bounce_x = float(
                    (racket_contact_pos[0] - incoming_contact_velocity[0] * post_bounce_time).item()
                )
                bounce_y = float(
                    (racket_contact_pos[1] - incoming_contact_velocity[1] * post_bounce_time).item()
                )
                print(
                    f"[INFO] natural racket solve: racket_vel={actual_racket_vel.detach().cpu().numpy().round(4).tolist()} "
                    f"incoming_at_racket={incoming_contact_velocity.detach().cpu().numpy().round(4).tolist()} "
                    f"post_bounce_time={post_bounce_time:.4f}s "
                    f"bounce_xy={[round(bounce_x, 4), round(bounce_y, 4)]}",
                    flush=True,
                )
            bounce_pos = torch.tensor(
                [bounce_x, bounce_y, table_z_offset + 0.02],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            # Backward ballistic solve for the velocity immediately after the
            # bounce: hit_pos = bounce_pos + v*t + 0.5*g*t^2.
            post_bounce_velocity = (
                incoming_contact_velocity - gravity * post_bounce_time
                if natural_racket_return
                else (
                    racket_contact_pos
                    - bounce_pos
                    - 0.5 * gravity * post_bounce_time * post_bounce_time
                )
                / post_bounce_time
            )
            # Table/ball effective normal restitution is 0.906.
            # First compute the velocity immediately before table impact,
            # then integrate it backward to the actual launch time.  Using the
            # impact velocity directly as launch velocity would over-accelerate
            # the ball during the pre-bounce segment.
            pre_bounce_impact_velocity = post_bounce_velocity.clone()
            pre_bounce_impact_velocity[2] = -post_bounce_velocity[2] / 0.906
            launch_vel = pre_bounce_impact_velocity - gravity * pre_bounce_time
            launch_pos = (
                bounce_pos
                - launch_vel * pre_bounce_time
                - 0.5 * gravity * pre_bounce_time * pre_bounce_time
            )
            flight_time = pre_bounce_time + post_bounce_time
        else:
            launch_pos = racket_contact_pos - incoming_velocity * flight_time + 0.5 * gravity * flight_time * flight_time
            launch_vel = incoming_velocity - gravity * flight_time
        launch_step = int(round((effective_hit_time - flight_time) / control_dt))
        ball_plan = {
            "ball": env.unwrapped.scene["ball"],
            "launch_step": launch_step,
            "hit_step": hit_step,
            "hit_pos": hit_pos,
            "racket_contact_pos": racket_contact_pos,
            "scripted_racket_return": scripted_racket_return,
            "natural_racket_return": natural_racket_return,
            "incoming_contact_velocity": incoming_contact_velocity,
            "return_target": return_target,
            "return_time": return_time,
            "return_velocity": return_velocity,
            "racket_normal": actual_racket_normal,
            "launch_pos": launch_pos,
            "launch_vel": launch_vel,
            "effective_hit_time": effective_hit_time,
            "flight_time": flight_time,
            "incoming_velocity": incoming_velocity,
            "bounce_once": bounce_once,
            "bounce_pos": bounce_pos,
            "post_bounce_velocity": post_bounce_velocity if bounce_once else None,
            "bounce_step": (
                launch_step + int(round(pre_bounce_time / control_dt))
                if bounce_once
                else None
            ),
        }
        print(
            f"[INFO] ball calibrated in original training scene: "
            f"hit_time={effective_hit_time:.3f}s flight={flight_time:.3f}s "
            f"racket_pos={actual_racket_pos.detach().cpu().numpy().round(4).tolist()} "
            f"contact_target={racket_contact_pos.detach().cpu().numpy().round(4).tolist()} "
            f"scripted_return={scripted_racket_return} "
            f"natural_return={natural_racket_return} "
            f"return_target={return_target.detach().cpu().numpy().round(4).tolist()} "
            f"racket_normal={actual_racket_normal.detach().cpu().numpy().round(4).tolist()} "
            f"launch_pos={launch_pos.detach().cpu().numpy().round(4).tolist()} "
            f"launch_vel={launch_vel.detach().cpu().numpy().round(4).tolist()} "
            f"bounce_once={bounce_once} "
            f"bounce_pos={None if bounce_pos is None else bounce_pos.detach().cpu().numpy().round(4).tolist()}",
            flush=True,
        )
        # Rewind to the same physical state before the actual ball pass.
        env.reset()
        _sync_ball_motion()
        _park_ball()
        _park_racket_proxy()

    # Export is convenient for deployment, but it must never prevent an
    # interactive/video replay.  Some stateful custom observation terms are
    # not serializable by the generic metadata helper.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        if has_motion_command:
            export_motion_policy_as_onnx(
                env.unwrapped, ppo_runner.alg.policy,
                normalizer=getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None),
                path=export_model_dir, filename="policy.onnx",
            )
        else:
            export_policy_as_onnx(
                ppo_runner.alg.policy,
                normalizer=getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None),
                path=export_model_dir,
            )
        attach_onnx_metadata(env.unwrapped, str(wandb_path) if wandb_path else "none", export_model_dir)
        print(f"[INFO] Exported ONNX policy to: {export_model_dir}")
    except Exception as exc:
        print(f"[WARN] ONNX export/metadata skipped; replay continues: {exc}", flush=True)

    # Manual video capture: grab env.render() each step and encode to mp4 with imageio
    # (imageio-ffmpeg). Avoids gym RecordVideo's vec-env / flush quirks and reports exactly
    # how many frames were captured so a black/empty render is obvious instead of silent.
    frames = []
    # IsaacLab/rsl_rl versions differ here: some return obs directly, others return (obs, extras).
    # Normalize before passing observations to the inference policy.
    obs = _obs_to_device(env.get_observations(), agent_cfg.device)
    touch_term = getattr(getattr(env_cfg, "terminations", None), "touch_success", None)
    touch_params = getattr(touch_term, "params", {}) or {}
    touch_distance_threshold = float(touch_params.get("distance_threshold", 0.07))
    # Contact-based touch tasks use ball_contacted_by_racket(), whose default geometry gate is 0.10 m.
    # Older distance-based tasks used 0.07 m. Keep both paths aligned with their termination terms.
    contact_gate_default = 0.10 if "racket_ball_contact" in env.unwrapped.scene.sensors else touch_distance_threshold
    face_lateral_threshold = float(touch_params.get("lateral_threshold", contact_gate_default))
    face_normal_threshold = float(touch_params.get("normal_threshold", contact_gate_default))
    normal_axis = int(touch_params.get("normal_axis", 1))
    normal_sign = float(touch_params.get("normal_sign", 1.0))
    touch_forward_velocity = float(touch_params.get("min_forward_velocity", 0.2))
    min_racket_ball_distance = float("inf")
    min_face_lateral_distance = float("inf")
    min_face_normal_distance = float("inf")
    max_face_contact_score = float("-inf")
    max_ball_forward_velocity = float("-inf")
    max_contact_force = 0.0
    close_count = 0
    face_close_count = 0
    forward_touch_like_count = 0
    face_forward_count = 0
    first_contact_count = 0
    contact_forward_count = 0
    has_ball = "ball" in env.unwrapped.scene.rigid_objects
    command_metric_sums = {}
    command_metric_count = 0
    robot = env.unwrapped.scene["robot"]
    min_root_height = float("inf")
    max_root_speed = 0.0
    max_root_ang_speed = 0.0
    terminated_count = 0
    truncated_count = 0
    first_termination_step = None
    ball_launched = False
    ball_bounce_applied = False
    ball_racket_return_applied = False
    ball_contact_candidate = False
    ball_returned_minus_x = False
    ball_landed_opponent = False
    first_opponent_landing_pos = None
    first_opponent_landing_step = None
    ball_min_distance = float("inf")
    ball_return_vx = float("nan")
    final_ball_pos = None
    final_ball_vel = None
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if ball_plan is not None and racket_proxy_enabled:
                # Kinematic blade proxy: it follows the same FK pose and
                # velocity used by the racket diagnostics.  In natural mode
                # the ball is never teleported or assigned a post-impact
                # velocity; PhysX resolves the ball/proxy collision.
                proxy = env.unwrapped.scene["racket_proxy"]
                proxy_pos, proxy_vel, proxy_quat = racket_state_w(env.unwrapped)
                proxy_normal = racket_normal_w(
                    env.unwrapped, normal_axis=normal_axis, normal_sign=normal_sign
                )
                proxy_pos = proxy_pos + proxy_normal * float(cfg.get("ball_racket_proxy_offset", 0.008))
                proxy.write_root_pose_to_sim(torch.cat([proxy_pos, proxy_quat], dim=-1))
                proxy.write_root_velocity_to_sim(
                    torch.cat([proxy_vel, torch.zeros_like(proxy_vel)], dim=-1)
                )
                env.unwrapped.scene.write_data_to_sim()
            if ball_plan is not None and not ball_launched and timestep == ball_plan["launch_step"]:
                ball = ball_plan["ball"]
                launch_pos = ball_plan["launch_pos"]
                launch_vel = ball_plan["launch_vel"]
                ball.write_root_pose_to_sim(
                    torch.cat(
                        [
                            launch_pos.view(1, 3),
                            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device),
                        ],
                        dim=-1,
                    )
                )
                ball.write_root_velocity_to_sim(
                    torch.cat([launch_vel.view(1, 3), torch.zeros((1, 3), device=env.unwrapped.device)], dim=-1)
                )
                ball_launched = True
                print(
                    f"[INFO] ball launched at step={timestep} "
                    f"t={timestep * float(env.unwrapped.step_dt):.3f}s",
                    flush=True,
                )
            actions = policy(obs)
            obs, _, terminated, truncated = env.step(actions.to(env.unwrapped.device))
            obs = _obs_to_device(obs, agent_cfg.device)
            if (
                ball_plan is not None
                and ball_plan["bounce_once"]
                and ball_launched
                and not ball_bounce_applied
                and timestep == ball_plan["bounce_step"]
            ):
                # Planned single-bounce diagnostic: place the ball exactly at
                # the calibrated table contact point and continue with the
                # solved post-bounce ballistic velocity.  This keeps the
                # validation focused on the end-of-swing receive/strike timing
                # while avoiding repeated table-friction bounces.
                ball = ball_plan["ball"]
                bounce_pos = ball_plan["bounce_pos"]
                ball.write_root_pose_to_sim(
                    torch.cat(
                        [
                            bounce_pos.view(1, 3),
                            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device),
                        ],
                        dim=-1,
                    )
                )
                ball.write_root_velocity_to_sim(
                    torch.cat(
                        [
                            ball_plan["post_bounce_velocity"].view(1, 3),
                            torch.zeros((1, 3), device=env.unwrapped.device),
                        ],
                        dim=-1,
                    )
                )
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.forward()
                ball_bounce_applied = True
                print(
                    f"[INFO] planned single table bounce applied at step={timestep} "
                    f"pos={bounce_pos.detach().cpu().numpy().round(4).tolist()} "
                    f"post_vel={ball_plan['post_bounce_velocity'].detach().cpu().numpy().round(4).tolist()}",
                    flush=True,
                )
            if (
                ball_plan is not None
                and ball_plan["scripted_racket_return"]
                and ball_launched
                and not ball_racket_return_applied
                and timestep == ball_plan["hit_step"]
            ):
                # Diagnostic-only impact node.  The imported URDF racket mesh
                # does not reliably report a PhysX contact force, so this
                # explicitly places the ball at the calibrated front face and
                # gives it the planned outgoing velocity.  Keep this separate
                # from the natural-contact metric.
                ball = ball_plan["ball"]
                contact_pos = ball_plan["racket_contact_pos"]
                ball.write_root_pose_to_sim(
                    torch.cat(
                        [
                            contact_pos.view(1, 3),
                            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device),
                        ],
                        dim=-1,
                    )
                )
                ball.write_root_velocity_to_sim(
                    torch.cat(
                        [
                            ball_plan["return_velocity"].view(1, 3),
                            torch.zeros((1, 3), device=env.unwrapped.device),
                        ],
                        dim=-1,
                    )
                )
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.forward()
                ball_racket_return_applied = True
                ball_contact_candidate = True
                print(
                    f"[INFO] planned end-swing racket impact applied at step={timestep} "
                    f"contact_pos={contact_pos.detach().cpu().numpy().round(4).tolist()} "
                    f"return_vel={ball_plan['return_velocity'].detach().cpu().numpy().round(4).tolist()} "
                    f"target={ball_plan['return_target'].detach().cpu().numpy().round(4).tolist()}",
                    flush=True,
                )
            terminated_tensor = terminated if torch.is_tensor(terminated) else torch.zeros(
                env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
            )
            truncated_tensor = truncated if torch.is_tensor(truncated) else torch.zeros_like(terminated_tensor)
            root_pos = robot.data.root_pos_w
            root_lin_vel = robot.data.root_lin_vel_w
            root_ang_vel = robot.data.root_ang_vel_w
            min_root_height = min(min_root_height, float(torch.min(root_pos[:, 2]).item()))
            max_root_speed = max(max_root_speed, float(torch.linalg.norm(root_lin_vel, dim=-1).max().item()))
            max_root_ang_speed = max(max_root_ang_speed, float(torch.linalg.norm(root_ang_vel, dim=-1).max().item()))
            terminated_count += int(torch.sum(terminated_tensor).item())
            truncated_count += int(torch.sum(truncated_tensor).item())
            if first_termination_step is None and bool(torch.any(terminated_tensor | truncated_tensor).item()):
                first_termination_step = timestep + 1
            if hasattr(env.unwrapped.command_manager, "get_term"):
                try:
                    racket_cmd = env.unwrapped.command_manager.get_term("racket_target")
                except Exception:
                    racket_cmd = None
                if racket_cmd is not None:
                    for name, value in getattr(racket_cmd, "metrics", {}).items():
                        if name.startswith(("racket_", "strike_", "exact_", "action_", "joint_")):
                            command_metric_sums[name] = command_metric_sums.get(name, 0.0) + float(value.mean().item())
                    command_metric_count += 1
            if has_ball:
                ball = env.unwrapped.scene["ball"]
                racket_pos_w, _, _ = racket_state_w(env.unwrapped)
                normal_w = racket_normal_w(env.unwrapped, normal_axis=normal_axis, normal_sign=normal_sign)
                racket_ball_distance = torch.norm(ball.data.root_pos_w - racket_pos_w, dim=-1)
                rel = ball.data.root_pos_w - racket_pos_w
                signed_normal_dist = torch.sum(rel * normal_w, dim=-1)
                lateral = rel - signed_normal_dist.unsqueeze(-1) * normal_w
                lateral_dist = torch.norm(lateral, dim=-1)
                face_contact_score = torch.exp(
                    -(torch.square(lateral_dist) / max(face_lateral_threshold, 1.0e-6) ** 2)
                    - (torch.square(signed_normal_dist) / max(face_normal_threshold, 1.0e-6) ** 2)
                )
                ball_forward_velocity = ball.data.root_lin_vel_w[:, 0]
                min_racket_ball_distance = min(min_racket_ball_distance, float(torch.min(racket_ball_distance).item()))
                min_face_lateral_distance = min(min_face_lateral_distance, float(torch.min(lateral_dist).item()))
                min_face_normal_distance = min(
                    min_face_normal_distance, float(torch.min(torch.abs(signed_normal_dist)).item())
                )
                ball_min_distance = min(ball_min_distance, float(torch.min(racket_ball_distance).item()))
                max_face_contact_score = max(max_face_contact_score, float(torch.max(face_contact_score).item()))
                max_ball_forward_velocity = max(max_ball_forward_velocity, float(torch.max(ball_forward_velocity).item()))
                close = racket_ball_distance < touch_distance_threshold
                face_close = (lateral_dist < face_lateral_threshold) & (
                    torch.abs(signed_normal_dist) < face_normal_threshold
                )
                close_count += int(torch.sum(close).item())
                face_close_count += int(torch.sum(face_close).item())
                forward_touch_like_count += int(
                    torch.sum(close & (ball_forward_velocity > touch_forward_velocity)).item()
                )
                face_forward_count += int(torch.sum(face_close & (ball_forward_velocity > touch_forward_velocity)).item())
                contact_sensor = None
                if "racket_ball_contact" in env.unwrapped.scene.sensors:
                    contact_sensor = env.unwrapped.scene.sensors["racket_ball_contact"]
                if contact_sensor is not None:
                    contact_force = torch.linalg.norm(contact_sensor.data.net_forces_w, dim=-1).amax(dim=-1)
                    racket_contact = (contact_force > 0.05) & face_close
                    max_contact_force = max(max_contact_force, float(torch.max(contact_force).item()))
                    first_contact_count += int(torch.sum(racket_contact).item())
                    contact_forward_count += int(
                        torch.sum(racket_contact & (ball_forward_velocity > touch_forward_velocity)).item()
                    )
                if ball_plan is not None and ball_launched:
                    current_vx = float(ball.data.root_lin_vel_w[0, 0].item())
                    if (not ball_contact_candidate) and float(torch.min(racket_ball_distance).item()) < 0.08:
                        ball_contact_candidate = True
                    if ball_contact_candidate and current_vx < -0.2:
                        ball_returned_minus_x = True
                        ball_return_vx = current_vx
                    ball_pos = ball.data.root_pos_w[0]
                    ball_vel = ball.data.root_lin_vel_w[0]
                    table_x0 = float(table_offset_x or 0.0)
                    table_x_mid = table_x0 + 1.37
                    table_y0, table_y1 = -1.525, 0.0
                    table_z = table_z_offset
                    # A ball crossing the opponent half in the air is not a landing.
                    # Require its center to be near table-top + ball radius so the
                    # diagnostic cannot report a false positive before the bounce.
                    table_ball_center_z = table_z + 0.02
                    if (
                        ball_returned_minus_x
                        and table_x0 + 0.03 <= float(ball_pos[0]) <= table_x_mid
                        and table_y0 + 0.03 <= float(ball_pos[1]) <= table_y1 - 0.03
                        and table_ball_center_z - 0.025 <= float(ball_pos[2]) <= table_ball_center_z + 0.04
                    ):
                        if not ball_landed_opponent:
                            first_opponent_landing_pos = ball_pos.detach().clone()
                            first_opponent_landing_step = timestep
                        ball_landed_opponent = True
                    if timestep in (ball_plan["hit_step"] - 1, ball_plan["hit_step"], ball_plan["hit_step"] + 1):
                        print(
                            f"[INFO] ball-hit snapshot step={timestep}: "
                            f"ball={ball_pos.detach().cpu().numpy().round(4).tolist()} "
                            f"racket={racket_pos_w[0].detach().cpu().numpy().round(4).tolist()} "
                            f"distance={float(racket_ball_distance[0].item()):.4f} "
                            f"vel={ball.data.root_lin_vel_w[0].detach().cpu().numpy().round(4).tolist()} "
                            f"vx={current_vx:.4f}",
                            flush=True,
                        )
                    final_ball_pos = ball.data.root_pos_w[0].detach().clone()
                    final_ball_vel = ball.data.root_lin_vel_w[0].detach().clone()
        if cfg.video:
            frame = env.unwrapped.render()
            if frame is not None:
                frames.append(frame)
            if timestep >= int(cfg.video_length):
                break
        timestep += 1
        max_steps = cfg.get("max_steps", None)
        if max_steps is not None and timestep >= int(max_steps):
            break
        # non-video: keep stepping until the Isaac Sim window is closed (live viewing)

    if cfg.video:
        import numpy as np

        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        video_name = str(cfg.get("video_name", "play"))
        if not video_name or os.path.basename(video_name) != video_name:
            raise ValueError("video_name must be a plain filename without a path")
        if not video_name.endswith(".mp4"):
            video_name += ".mp4"
        video_path = os.path.join(video_dir, video_name)
        valid = [np.asarray(f) for f in frames if f is not None and getattr(f, "size", 0) > 0]
        print(f"[INFO] captured {len(frames)} frames ({len(valid)} non-empty)", flush=True)
        if valid:
            import imageio

            # Simulation/render frames are produced at the 50 Hz control rate.
            # Saving them at 30 FPS makes the replay appear ~1.67x slower.
            imageio.mimsave(video_path, valid, fps=50)
            print(f"[INFO] wrote video -> {video_path}", flush=True)
        else:
            print(
                "[ERROR] env.render() returned no usable frames. Check that AppLauncher got "
                "enable_cameras=True (it ties to video) and render_mode='rgb_array'.",
                flush=True,
            )
    if has_ball:
        print(
            "[INFO] replay metrics: "
            f"min_racket_ball_distance={min_racket_ball_distance:.4f} m, "
            f"min_face_lateral_distance={min_face_lateral_distance:.4f} m, "
            f"min_face_normal_distance={min_face_normal_distance:.4f} m, "
            f"max_face_contact_score={max_face_contact_score:.4f}, "
            f"max_ball_forward_velocity={max_ball_forward_velocity:.4f} m/s, "
            f"max_contact_force={max_contact_force:.4f} N, "
            f"close_count={close_count}, "
            f"face_close_count={face_close_count}, "
            f"forward_touch_like_count={forward_touch_like_count}, "
            f"face_forward_count={face_forward_count}, "
            f"first_contact_count={first_contact_count}, "
            f"contact_forward_count={contact_forward_count} "
            f"(thresholds: distance<{touch_distance_threshold:.3f} m, "
            f"face_lateral<{face_lateral_threshold:.3f} m, "
            f"face_normal<{face_normal_threshold:.3f} m, vx>{touch_forward_velocity:.3f} m/s)",
            flush=True,
        )
    else:
        print("[INFO] replay metrics: scene has no ball entity; skipped table-tennis contact metrics.", flush=True)
    print(
        "[INFO] stability metrics: "
        f"min_root_height={min_root_height:.4f} m, "
        f"max_root_speed={max_root_speed:.4f} m/s, "
        f"max_root_ang_speed={max_root_ang_speed:.4f} rad/s, "
        f"terminated_count={terminated_count}, truncated_count={truncated_count}, "
        f"first_termination_step={first_termination_step}",
        flush=True,
    )
    if ball_plan is not None:
        print(
            "[INFO] dynamic-ball validation: "
            f"contact_candidate={ball_contact_candidate}, "
            f"returned_toward_minus_x={ball_returned_minus_x}, "
            f"planned_racket_impact={ball_racket_return_applied}, "
            f"opponent_table_landing={ball_landed_opponent}, "
            f"closest_racket_distance={ball_min_distance:.4f} m, "
            f"return_vx={ball_return_vx:.4f} m/s",
            flush=True,
        )
        if final_ball_pos is not None and final_ball_vel is not None:
            print(
                f"[INFO] final_ball_state: pos={final_ball_pos.detach().cpu().numpy().round(4).tolist()} "
                f"vel={final_ball_vel.detach().cpu().numpy().round(4).tolist()}",
                flush=True,
            )
        if first_opponent_landing_pos is not None:
            print(
                f"[INFO] first_opponent_landing: step={first_opponent_landing_step} "
                f"pos={first_opponent_landing_pos.detach().cpu().numpy().round(4).tolist()}",
                flush=True,
            )
    if command_metric_count:
        print("[INFO] manifest command metrics:", flush=True)
        for name in sorted(command_metric_sums):
            print(f"[INFO]   {name}={command_metric_sums[name] / command_metric_count:.4f}", flush=True)

    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    try:
        _run_play(cfg, simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
