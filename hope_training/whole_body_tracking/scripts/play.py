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

def _run_play(cfg, simulation_app):
    import pathlib
    import copy
    import dataclasses
    import json

    import gymnasium as gym
    import torch

    # ``train`` imports task utilities that eventually import IsaacLab.  It
    # must be imported only after AppLauncher created omni::timeline; doing it
    # at module import time makes standalone headless replay fail before its
    # own launcher runs.  This is evaluation-only and does not alter training.
    from train import _apply_task_overrides, _as_bool

    from rsl_rl.runners import OnPolicyRunner
    import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import training.tasks  # noqa: F401  -- registers the gym tasks
    from training.tasks.table_tennis.mdp.racket import (
        racket_normal_w,
        racket_spatial_state_w,
        racket_state_w,
    )
    from training.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
    from training.utils.external_hit_schedule import schedule_external_hit_time
    from training.utils.external_strike_request import load_external_strike_request
    from training.utils.strike_goal import (
        BASE_HEADING_RECEIPT_FRAME_V1,
        LatchedStrikeGoal,
        PlannerRacketCommand,
        StrikeGoalFrameTransform,
        isaac_diagnostic_proxy_contact_calibration,
    )
    from training.utils.strike_goal_shadow import (
        RacketFaceState,
        StrikeGoalShadowPipeline,
    )
    from training.utils.ppo_cfg import runner_kwargs
    from training.utils.stagger_support_actor_critic import (
        BentReadyRecoveryActorCritic,
        StaggerSupportActorCritic,
        WideStaggerRecoveryActorCritic,
        WideStaggerSupportActorCritic,
    )
    from training.utils.target_conditioned_recovery_actor_critic import (
        TargetConditionedRecoveryActorCritic,
    )

    def _obs_to_device(obs, device):
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs.to(device)

    def _target_rows(value, name):
        """Normalize one 3-vector or an N-by-3 list without accepting ragged input."""
        if value is None:
            return None
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape == (3,):
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError(f"{name} must have shape [3] or [N,3], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor.tolist()

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)
    strike_goal_shadow_command_path = cfg.get(
        "strike_goal_shadow_command_path", None
    )
    strike_goal_shadow_enabled = strike_goal_shadow_command_path is not None
    strike_goal_shadow_report = cfg.get("strike_goal_shadow_report", None)
    if strike_goal_shadow_enabled:
        if num_envs != 1:
            raise ValueError("strike-goal shadow currently requires num_envs=1")
        if strike_goal_shadow_report is None:
            raise ValueError(
                "strike_goal_shadow_command_path requires strike_goal_shadow_report"
            )
        if cfg.get("strike_goal_shadow_hope_world_to_sim_rotation", None) is None or cfg.get(
            "strike_goal_shadow_hope_world_to_sim_translation", None
        ) is None:
            raise ValueError(
                "strike-goal shadow requires explicit HOPE-world-to-sim rotation and translation"
            )
    adapter_policy_scale = float(cfg.get("adapter_policy_scale", 1.0))
    if not torch.isfinite(torch.tensor(adapter_policy_scale)) or adapter_policy_scale < 0.0:
        raise ValueError("adapter_policy_scale must be a non-negative finite value")
    adapter_jacobian_step = cfg.get("adapter_jacobian_step", None)
    adapter_jacobian_enabled = adapter_jacobian_step is not None
    if adapter_jacobian_enabled:
        adapter_jacobian_step = float(adapter_jacobian_step)
        if not torch.isfinite(torch.tensor(adapter_jacobian_step)) or adapter_jacobian_step <= 0.0:
            raise ValueError("adapter_jacobian_step must be a positive finite raw-action perturbation")
    coordinator_jacobian_step = cfg.get("coordinator_jacobian_step", None)
    coordinator_jacobian_enabled = coordinator_jacobian_step is not None
    if coordinator_jacobian_enabled:
        coordinator_jacobian_step = float(coordinator_jacobian_step)
        if not torch.isfinite(torch.tensor(coordinator_jacobian_step)) or coordinator_jacobian_step <= 0.0:
            raise ValueError("coordinator_jacobian_step must be a positive finite raw-action perturbation")
    if adapter_jacobian_enabled and coordinator_jacobian_enabled:
        raise ValueError("choose at most one of adapter_jacobian_step or coordinator_jacobian_step")
    p4_recovery_action_offset = cfg.get("p4_recovery_action_offset", None)
    if p4_recovery_action_offset is not None:
        if OmegaConf.is_config(p4_recovery_action_offset):
            p4_recovery_action_offset = OmegaConf.to_container(
                p4_recovery_action_offset, resolve=True
            )
        p4_recovery_action_offset = torch.as_tensor(
            p4_recovery_action_offset, dtype=torch.float32
        ).flatten()
        if p4_recovery_action_offset.shape != (22,):
            raise ValueError(
                "p4_recovery_action_offset must be a finite 22-D coordinator action"
            )
        if not torch.isfinite(p4_recovery_action_offset).all():
            raise ValueError("p4_recovery_action_offset must contain only finite values")
    external_strike_request = None
    external_strike_request_path = cfg.get("external_strike_request_path", None)
    if external_strike_request_path is not None:
        if any(
            value is not None
            for value in (
                cfg.get("external_target_position_b", None),
                cfg.get("external_target_offset_b", None),
                cfg.get("target_offset_grid_cm", None),
                cfg.get("external_hit_time_s", None),
            )
        ):
            raise ValueError(
                "external_strike_request_path is mutually exclusive with direct target and time flags"
            )
        external_strike_request = load_external_strike_request(
            external_strike_request_path
        )
    external_target_position_rows = _target_rows(
        (
            external_strike_request["target_position_b"]
            if external_strike_request is not None
            else cfg.get("external_target_position_b", None)
        ),
        "external_target_position_b",
    )
    external_target_offset_rows = _target_rows(
        cfg.get("external_target_offset_b", None), "external_target_offset_b"
    )
    target_grid_raw = cfg.get("target_offset_grid_cm", None)
    target_grid_cm = None
    if target_grid_raw is not None:
        if OmegaConf.is_config(target_grid_raw):
            target_grid_raw = OmegaConf.to_container(target_grid_raw, resolve=True)
        target_grid_cm = [float(value) for value in target_grid_raw]
        if not target_grid_cm or any(
            (not torch.isfinite(torch.tensor(value))) or value <= 0.0
            for value in target_grid_cm
        ):
            raise ValueError("target_offset_grid_cm must contain positive finite radii")
        external_target_offset_rows = [[0.0, 0.0, 0.0]]
        for radius_cm in target_grid_cm:
            radius_m = radius_cm / 100.0
            for axis in range(3):
                negative = [0.0, 0.0, 0.0]
                positive = [0.0, 0.0, 0.0]
                negative[axis] = -radius_m
                positive[axis] = radius_m
                external_target_offset_rows.extend([negative, positive])

    if adapter_jacobian_enabled:
        if any(value is not None for value in (external_target_position_rows, external_target_offset_rows)):
            raise ValueError("adapter_jacobian_step cannot be combined with an external target grid")
        # nominal plus +/- finite differences for seven adapter outputs; pad to
        # a multiple of seven because P0's paired command contract requires it.
        external_target_offset_rows = [[0.0, 0.0, 0.0] for _ in range(21)]
    if coordinator_jacobian_enabled:
        if any(value is not None for value in (external_target_position_rows, external_target_offset_rows)):
            raise ValueError("coordinator_jacobian_step cannot be combined with an external target grid")
        # nominal plus +/- finite differences for 22 coordinator dimensions.
        # Pad 45 trials to seven target-pair groups required by the command.
        external_target_offset_rows = [[0.0, 0.0, 0.0] for _ in range(49)]

    target_input_count = sum(
        value is not None
        for value in (
            external_target_position_rows,
            cfg.get("external_target_offset_b", None),
            target_grid_cm,
        )
    )
    if target_input_count > 1:
        raise ValueError(
            "choose exactly one of external_target_position_b, "
            "external_target_offset_b, or target_offset_grid_cm"
        )
    target_audit_enabled = (
        external_target_position_rows is not None
        or external_target_offset_rows is not None
    )
    target_audit_post_hit_steps = int(cfg.get("target_audit_post_hit_steps", 15))
    if target_audit_post_hit_steps < 0:
        raise ValueError("target_audit_post_hit_steps must be non-negative")
    external_hit_time_s = (
        external_strike_request["hit_time_s"]
        if external_strike_request is not None
        else cfg.get("external_hit_time_s", None)
    )
    if external_hit_time_s is not None:
        external_hit_time_s = float(external_hit_time_s)
        if (
            not torch.isfinite(torch.tensor(external_hit_time_s))
            or external_hit_time_s <= 0.0
        ):
            raise ValueError("external_hit_time_s must be a positive finite number")
        if not target_audit_enabled:
            raise ValueError(
                "external_hit_time_s requires one external target position or offset"
            )
    external_hit_max_added_delay_s = float(
        cfg.get("external_hit_max_added_delay_s", 0.50)
    )
    if (
        not torch.isfinite(torch.tensor(external_hit_max_added_delay_s))
        or external_hit_max_added_delay_s < 0.0
    ):
        raise ValueError(
            "external_hit_max_added_delay_s must be a non-negative finite number"
        )
    auto_select_motion = _as_bool(cfg.get("auto_select_motion", False))
    auto_select_max_anchor_distance_m = float(
        cfg.get("auto_select_max_anchor_distance_m", 0.02)
    )
    if not torch.isfinite(torch.tensor(auto_select_max_anchor_distance_m)) or (
        auto_select_max_anchor_distance_m <= 0.0
    ):
        raise ValueError("auto_select_max_anchor_distance_m must be positive and finite")
    auto_select_local_range_tolerance_m = float(
        cfg.get("auto_select_local_range_tolerance_m", 1.0e-6)
    )
    if not torch.isfinite(torch.tensor(auto_select_local_range_tolerance_m)) or not (
        0.0 <= auto_select_local_range_tolerance_m <= 1.0e-4
    ):
        raise ValueError(
            "auto_select_local_range_tolerance_m must be finite and in [0, 1e-4]"
        )
    auto_select_local_half_ranges = cfg.get("auto_select_local_half_range_by_motion", None)
    if auto_select_local_half_ranges is not None:
        if OmegaConf.is_config(auto_select_local_half_ranges):
            auto_select_local_half_ranges = OmegaConf.to_container(
                auto_select_local_half_ranges, resolve=True
            )
        auto_select_local_half_ranges = torch.as_tensor(
            auto_select_local_half_ranges, dtype=torch.float32
        )
        if (
            auto_select_local_half_ranges.ndim != 2
            or auto_select_local_half_ranges.shape[1] != 3
            or not torch.isfinite(auto_select_local_half_ranges).all()
            or torch.any(auto_select_local_half_ranges <= 0.0)
        ):
            raise ValueError(
                "auto_select_local_half_range_by_motion must have finite positive shape [M, 3]"
            )
    if auto_select_motion:
        if external_target_position_rows is None:
            raise ValueError(
                "auto_select_motion requires external_target_position_b; "
                "a relative offset has no anchor-selection meaning"
            )
        if cfg.get("motion_id", None) is not None:
            raise ValueError("auto_select_motion and motion_id are mutually exclusive")
    if external_strike_request is not None and not auto_select_motion:
        raise ValueError(
            "external_strike_request_path requires auto_select_motion=true"
        )
    target_audit_synchronize_siblings = bool(
        cfg.get("target_audit_synchronize_siblings", True)
    )
    if target_audit_enabled:
        rows = (
            external_target_position_rows
            if external_target_position_rows is not None
            else external_target_offset_rows
        )
        num_envs = len(rows)
    if external_hit_time_s is not None and num_envs != 1:
        raise ValueError(
            "external_hit_time_s is currently a single-shot, single-target contract"
        )

    def _synchronize_target_audit_siblings(raw_env) -> None:
        """Clone env0's physical strike-ready state across audit siblings.

        External-target grids and finite differences identify millimetre-scale
        effects.  A normal vectorized reset is allowed to sample independent
        handoff/contact states, which can otherwise create centimetre-scale
        trial differences even when every target and policy action is equal.
        Keep each sibling in its own scene tile by translating only root
        positions by its environment-origin delta; orientations, velocities,
        and all joint states are copied exactly from env0.
        """
        if raw_env.num_envs <= 1:
            return
        robot = raw_env.scene["robot"]
        env_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
        root_state = robot.data.root_state_w[0:1].expand(raw_env.num_envs, -1).clone()
        root_state[:, :3] += raw_env.scene.env_origins - raw_env.scene.env_origins[0:1]
        joint_pos = robot.data.joint_pos[0:1].expand(raw_env.num_envs, -1).clone()
        joint_vel = robot.data.joint_vel[0:1].expand(raw_env.num_envs, -1).clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        raw_env.scene.write_data_to_sim()
        raw_env.sim.forward()

    multi_shot_raw = cfg.get("multi_shot_sequence", None)
    multi_shot_sequence = None
    multi_shot_max_steps = None
    if multi_shot_raw is not None:
        if isinstance(multi_shot_raw, str):
            multi_shot_sequence = [
                int(token.strip())
                for token in multi_shot_raw.split(",")
                if token.strip()
            ]
        else:
            multi_shot_sequence = [int(value) for value in multi_shot_raw]
        if not multi_shot_sequence:
            raise ValueError("multi_shot_sequence must contain at least one motion ID")
        if num_envs != 1:
            raise ValueError("multi_shot_sequence requires num_envs=1")
        if bool(cfg.get("ball", False)):
            raise ValueError("multi_shot_sequence does not yet support dynamic-ball replay")
    if auto_select_motion and multi_shot_sequence is not None:
        raise ValueError("auto_select_motion is currently a single-shot executor feature")
    if target_audit_enabled:
        if multi_shot_sequence is not None:
            raise ValueError("external target audit is single-shot and cannot use multi_shot_sequence")
        if bool(cfg.get("ball", False)):
            raise ValueError("external target audit deliberately runs without a dynamic ball")
        if cfg.get("motion_id", None) is None and not auto_select_motion:
            raise ValueError(
                "external target audit requires motion_id or auto_select_motion=true"
            )

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    if (
        (
            target_audit_enabled
            or bool(cfg.get("ball", False))
            or strike_goal_shadow_enabled
        )
        and num_envs % 7 != 0
        and bool(getattr(env_cfg.commands.racket_target, "adapter_external_paired", False))
    ):
        # Explicit external commands, one-ball diagnostics and read-only
        # Planner shadow runs define a one-environment audit.  The P0 paired sampler is a
        # training-data contract and would reject a valid one-shot command
        # solely because its vectorized batch is not a complete
        # [0,+x,-x,+y,-y,+z,-z] group.
        env_cfg.commands.racket_target.adapter_external_paired = False
    if target_audit_enabled and target_audit_synchronize_siblings:
        # ``physics_material`` is a startup domain-randomization event.  It
        # gives each vectorized environment a different contact model, which
        # invalidates a paired millimetre-scale response measurement even if
        # robot state and policy actions have been copied exactly.  Keep DR
        # for training/robustness evaluation; target-identification audits use
        # the deterministic nominal contact contract first.
        if hasattr(env_cfg.events, "physics_material"):
            env_cfg.events.physics_material = None
    env_cfg.sim.device = str(cfg.device)
    # Keep visual replay aligned with train.py's deterministic paired audits.
    env_cfg.seed = int(cfg.seed)
    scene_root_position = cfg.get("scene_root_position_w_m", None)
    scene_root_quaternion = cfg.get("scene_root_quaternion_wxyz", None)
    if scene_root_position is not None or scene_root_quaternion is not None:
        if scene_root_position is None or scene_root_quaternion is None:
            raise ValueError(
                "scene_root_position_w_m and scene_root_quaternion_wxyz must be set together"
            )
        scene_root_position = tuple(float(value) for value in scene_root_position)
        scene_root_quaternion = tuple(float(value) for value in scene_root_quaternion)
        if len(scene_root_position) != 3 or len(scene_root_quaternion) != 4:
            raise ValueError(
                "scene root placement requires position shape [3] and quaternion shape [4]"
            )
        values = torch.tensor(
            (*scene_root_position, *scene_root_quaternion), dtype=torch.float64
        )
        quaternion_norm = float(torch.linalg.norm(values[3:]).item())
        if not torch.isfinite(values).all() or abs(quaternion_norm - 1.0) > 1.0e-5:
            raise ValueError(
                "scene root placement must be finite and the wxyz quaternion unit length"
            )
        env_cfg.scene.robot.init_state.pos = scene_root_position
        env_cfg.scene.robot.init_state.rot = scene_root_quaternion
        print(
            "[INFO] evaluation-only scene root placement: "
            f"pos={scene_root_position}, quat_wxyz={scene_root_quaternion}",
            flush=True,
        )
    if external_hit_time_s is not None:
        # Preserve the requested post-hit audit tail even when the READY hold
        # intentionally pushes impact past the training episode horizon.
        control_dt = float(env_cfg.decimation * env_cfg.sim.dt)
        required_episode_length_s = external_hit_time_s + (
            target_audit_post_hit_steps + 5
        ) * control_dt
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s), required_episode_length_s
        )
    if target_audit_enabled:
        # Paired finite differences are meaningless if each parallel
        # environment draws different observation noise.  This is an
        # evaluation-only override; training noise remains unchanged.
        for group_name in ("policy", "critic", "upper", "stage_a"):
            group = getattr(env_cfg.observations, group_name, None)
            if group is not None and hasattr(group, "enable_corruption"):
                group.enable_corruption = False
    if multi_shot_sequence is not None:
        control_dt = float(env_cfg.decimation * env_cfg.sim.dt)
        multi_shot_max_steps = cfg.get("multi_shot_max_steps", None)
        if multi_shot_max_steps is None:
            multi_shot_max_steps = 350 * len(multi_shot_sequence)
        multi_shot_max_steps = int(multi_shot_max_steps)
        if multi_shot_max_steps < 1:
            raise ValueError("multi_shot_max_steps must be positive")
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s),
            multi_shot_max_steps * control_dt,
        )
        print(
            "[INFO] multi-shot no-reset audit: "
            f"sequence={multi_shot_sequence}, max_steps={multi_shot_max_steps}, "
            f"episode_length_s={env_cfg.episode_length_s:.3f}",
            flush=True,
        )

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
    legacy_stagger_support_task = (
        task_id == "HOPE-FloatingJointCoordinatorV8StaggerSupport-AgibotA3-v0"
    )
    wide_stagger_support_task = (
        task_id
        == "HOPE-FloatingJointCoordinatorV9WideStaggerSupport-AgibotA3-v0"
    )
    wide_stagger_recovery_task = (
        task_id
        == "HOPE-FloatingJointCoordinatorV10WideStaggerRecovery-AgibotA3-v0"
    )
    bent_ready_recovery_task = (
        task_id
        == "HOPE-FloatingJointCoordinatorV11BentReadyRecovery-AgibotA3-v0"
    )
    target_conditioned_recovery_task = (
        task_id
        in {
            "HOPE-FloatingTargetConditionedRecovery-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryYComp-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion0Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion2Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion4Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion5Calibrated-AgibotA3-v0",
            "HOPE-FloatingTargetConditionedRecoveryMotion1Train-AgibotA3-v0",
        }
    )
    stagger_support_task = (
        legacy_stagger_support_task
        or wide_stagger_support_task
        or wide_stagger_recovery_task
        or bent_ready_recovery_task
    )
    if target_conditioned_recovery_task:
        rsl_on_policy_runner.TargetConditionedRecoveryActorCritic = (
            TargetConditionedRecoveryActorCritic
        )
        agent_cfg.policy.class_name = "TargetConditionedRecoveryActorCritic"
        print(
            "[play.py] P4 policy=TargetConditionedRecoveryActorCritic "
            "(213-D observation; frozen P3 plus lower-body brace/residual adapter)",
            flush=True,
        )
    elif bent_ready_recovery_task:
        rsl_on_policy_runner.BentReadyRecoveryActorCritic = (
            BentReadyRecoveryActorCritic
        )
        agent_cfg.policy.class_name = "BentReadyRecoveryActorCritic"
        print(
            "[play.py] V28 policy=BentReadyRecoveryActorCritic "
            "(235-D observation; bounded post-hit bent-READY adapter)",
            flush=True,
        )
    elif wide_stagger_recovery_task:
        rsl_on_policy_runner.WideStaggerRecoveryActorCritic = (
            WideStaggerRecoveryActorCritic
        )
        agent_cfg.policy.class_name = "WideStaggerRecoveryActorCritic"
        print(
            "[play.py] V23 policy=WideStaggerRecoveryActorCritic "
            "(229-D observation; frozen V22 plus gated recovery adapter)",
            flush=True,
        )
    elif wide_stagger_support_task:
        rsl_on_policy_runner.WideStaggerSupportActorCritic = (
            WideStaggerSupportActorCritic
        )
        agent_cfg.policy.class_name = "WideStaggerSupportActorCritic"
        print(
            "[play.py] V22 policy=WideStaggerSupportActorCritic "
            "(227-D 2-D support observation; frozen legacy arm contract)",
            flush=True,
        )
    elif legacy_stagger_support_task:
        rsl_on_policy_runner.StaggerSupportActorCritic = StaggerSupportActorCritic
        agent_cfg.policy.class_name = "StaggerSupportActorCritic"
        print(
            "[play.py] V21 policy=StaggerSupportActorCritic "
            "(223-D stance observation; frozen legacy arm contract)",
            flush=True,
        )

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

    # IsaacLab's default viewport camera controller follows ``viewer.asset_name``
    # and can keep firing callbacks after a headless replay environment has
    # been torn down.  Evaluation callers that only need an encoded camera
    # stream can disable that tracker while retaining RGB rendering.
    if _as_bool(cfg.get("disable_viewer_tracking", False)):
        viewer_cfg = getattr(env_cfg, "viewer", None)
        if viewer_cfg is not None and hasattr(viewer_cfg, "asset_name"):
            viewer_cfg.asset_name = None
            if hasattr(viewer_cfg, "origin_type"):
                viewer_cfg.origin_type = "world"
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    # A V1.3B checkpoint is saved at a particular point on the shared
    # curriculum/prior schedule.  Evaluation normally has no runner to
    # advance that clock, so allow a caller to latch the exact saved progress
    # before the first policy action and resample the one episode goal under
    # that same distribution.  This is replay-only and cannot alter PPO.
    v13b_policy_progress = cfg.get("v13b_policy_progress", None)
    if v13b_policy_progress is not None:
        if "ReferenceFreeV13B" not in task_id:
            raise ValueError("v13b_policy_progress is only valid for a V1.3B task")
        v13b_policy_progress = float(v13b_policy_progress)
        if not torch.isfinite(torch.tensor(v13b_policy_progress)) or not 0.0 <= v13b_policy_progress <= 1.0:
            raise ValueError("v13b_policy_progress must be finite and in [0, 1]")
        env.unwrapped.v13b_policy_progress = v13b_policy_progress
        racket_target = env.unwrapped.command_manager.get_term("racket_target")
        racket_target._v13b_policy_progress = v13b_policy_progress
        replay_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
        racket_target._resample_command(replay_env_ids)
        racket_target._compute_strike_timing()
        print(
            "[V1.3B] replay latched training-progress snapshot: "
            f"{v13b_policy_progress:.6f}",
            flush=True,
        )
    # Ordered V1.3B admission smokes use these *evaluation-only* switches to
    # isolate zero-action/lower-prior/upper-prior paths.  They are runtime
    # attributes, never task configuration, and are ignored by normal PPO.
    for prior_name in ("lower", "upper"):
        value = cfg.get(f"v13b_force_{prior_name}_prior_alpha", None)
        if value is None:
            continue
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"v13b_force_{prior_name}_prior_alpha must be finite and in [0, 1]"
            )
        setattr(env.unwrapped, f"v13b_force_{prior_name}_prior_alpha", value)
        print(f"[V1.3B][smoke] forced {prior_name} prior alpha={value:.3f}", flush=True)
    p4c_upper_execution_mode = str(
        cfg.get("p4c_upper_execution_mode", "policy")
    )
    if p4c_upper_execution_mode not in {"policy", "reference_only"}:
        raise ValueError(
            "p4c_upper_execution_mode must be 'policy' or 'reference_only'"
        )
    env.unwrapped.p4c_upper_execution_mode = p4c_upper_execution_mode
    # Evaluation-only lower-support scaling.  This runtime hook is installed
    # only on the replay environment, so a concurrently running PPO process
    # and its training contract remain unchanged.
    replay_lower_output_scale = float(cfg.get("replay_lower_output_scale", 1.0))
    if not torch.isfinite(torch.tensor(replay_lower_output_scale)) or replay_lower_output_scale <= 0.0:
        raise ValueError("replay_lower_output_scale must be finite and positive")
    lower_action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if hasattr(lower_action_term, "_replay_lower_output_scale"):
        lower_action_term._replay_lower_output_scale = replay_lower_output_scale
        env.unwrapped.p5u_replay_lower_output_scale = replay_lower_output_scale
        print(
            "[INFO] replay-only lower-support output scale: "
            f"{replay_lower_output_scale:.3f}x (training contract remains 1.000x)",
            flush=True,
        )
    elif replay_lower_output_scale != 1.0:
        raise ValueError(
            "replay_lower_output_scale != 1 requires the unified P5U action term"
        )
    if p4c_upper_execution_mode != "policy":
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        if not hasattr(action_term, "upper_reference_actions"):
            raise ValueError(
                "p4c reference_only requires an upper-reference composite action"
            )
        print(
            "[INFO] P4C evaluation-only upper execution mode: "
            f"{p4c_upper_execution_mode}",
            flush=True,
        )
    if multi_shot_sequence is not None and not bool(
        getattr(
            env.unwrapped.action_manager.get_term("joint_pos").cfg,
            "stage_a_sagittal_rearm_enabled",
            False,
        )
    ):
        raise RuntimeError(
            "multi_shot_sequence requires a task with Stage-A sagittal re-arm enabled"
        )
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
    if stagger_support_task or target_conditioned_recovery_task:
        original_obs_normalizer_forward = ppo_runner.obs_normalizer.forward

        def v21_actor_observation_normalizer(observation):
            normalized = original_obs_normalizer_forward(observation)
            normalized[..., 204:] = observation[..., 204:]
            return normalized

        ppo_runner.obs_normalizer.forward = v21_actor_observation_normalizer
        support_end = (
            213
            if target_conditioned_recovery_task
            else 235
            if bent_ready_recovery_task
            else 227
            if wide_stagger_support_task or wide_stagger_recovery_task
            else 223
        )
        print(
            "[play.py] recovery/support actor normalizer preserves physical columns "
            f"204:{support_end}",
            flush=True,
        )
    # A semantic P5U -> V1.3B warm-start deliberately contains actor weights
    # only: critic and optimizer are fresh by contract.  RSL's generic
    # ``load`` is strict and would reject that valid admission-test artifact,
    # so load only the actor/actor-normalizer in this narrow replay case.
    checkpoint_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if bool(checkpoint_payload.get("v13b_migrated_from_p5u", False)):
        if "ReferenceFreeV13B" not in task_id:
            raise ValueError("V1.3B migrated warm-start may only be replayed by a V1.3B task")
        migrated_state = checkpoint_payload["model_state_dict"]
        expected_state = ppo_runner.alg.policy.state_dict()
        unexpected = tuple(key for key in migrated_state if key not in expected_state)
        missing = tuple(
            key
            for key in expected_state
            if key not in migrated_state and not key.startswith("critic.")
        )
        if unexpected or missing:
            raise RuntimeError(
                "V1.3B migrated warm-start actor contract mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        # IsaacLab's ActorCritic override returns a boolean rather than
        # PyTorch's IncompatibleKeys object, hence the explicit key audit
        # above and no reliance on a return value here.
        ppo_runner.alg.policy.load_state_dict(migrated_state, strict=False)
        ppo_runner.obs_normalizer.load_state_dict(checkpoint_payload["obs_norm_state_dict"])
        print(
            "[V1.3B] replay loaded semantic actor warm-start; critic remains fresh by contract",
            flush=True,
        )
    else:
        ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    forced_motion_id = (
        multi_shot_sequence[0]
        if multi_shot_sequence is not None
        else cfg.get("motion_id", None)
    )
    # Forced single-motion replays historically synchronized the physical
    # robot to motion frame 0.  That is unsuitable for P5U: its contract
    # requires the configured forked, flexed READY stance, followed by the
    # runtime prelude into frame 0.  Keep the legacy behavior available, but
    # make the physically correct P5U path explicit and opt-in.
    sync_initial_motion_state = _as_bool(cfg.get("sync_initial_motion_state", True))
    auto_motion_selection = None
    auto_selected_motion_ids = None
    hit_schedule = None
    if auto_select_motion:
        if not has_motion_command:
            raise ValueError("auto_select_motion requires an environment with a motion command")
        motion = env.unwrapped.command_manager.get_term("motion")
        target_position_b = torch.tensor(
            external_target_position_rows,
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        selected, distances, anchors_b = motion.select_nearest_strike_motion_ids(
            target_position_b
        )
        if selected.numel() != env.num_envs:
            raise RuntimeError("nearest-anchor selector returned the wrong environment count")
        if torch.any(distances > auto_select_max_anchor_distance_m):
            offending = torch.where(distances > auto_select_max_anchor_distance_m)[0]
            raise ValueError(
                "external target lies outside every verified local anchor range: "
                f"env_ids={offending.detach().cpu().tolist()}, "
                f"nearest_distance_m={distances[offending].detach().cpu().tolist()}, "
                f"limit_m={auto_select_max_anchor_distance_m}"
            )
        control_half_ranges = torch.as_tensor(
            getattr(motion.cfg, "external_control_local_half_range_by_motion", ()),
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        if control_half_ranges.numel() > 0:
            if (
                control_half_ranges.shape
                != (motion.motion.num_motions, 3)
                or not torch.isfinite(control_half_ranges).all()
                or torch.any(control_half_ranges < 0.0)
            ):
                raise ValueError(
                    "external_control_local_half_range_by_motion must have finite "
                    "non-negative shape [num_manifest_motions, 3]"
                )
        if auto_select_local_half_ranges is not None or control_half_ranges.numel() > 0:
            if (
                auto_select_local_half_ranges is not None
                and auto_select_local_half_ranges.shape[0] != motion.motion.num_motions
            ):
                raise ValueError(
                    "auto_select_local_half_range_by_motion must provide one row per "
                    f"manifest motion ({motion.motion.num_motions})"
                )
            if auto_select_local_half_ranges is None:
                candidate_half_ranges = control_half_ranges
            else:
                candidate_half_ranges = auto_select_local_half_ranges.to(
                    env.unwrapped.device
                )
            if control_half_ranges.numel() > 0 and auto_select_local_half_ranges is not None:
                candidate_half_ranges = torch.minimum(
                    candidate_half_ranges, control_half_ranges
                )
            half_ranges = candidate_half_ranges[selected]
            selected_anchors_b = anchors_b[selected]
            selected_delta_b = target_position_b - selected_anchors_b
            outside = torch.any(
                torch.abs(selected_delta_b)
                > half_ranges + auto_select_local_range_tolerance_m,
                dim=-1,
            )
            if torch.any(outside):
                offending = torch.where(outside)[0]
                raise ValueError(
                    "external target exceeds the verified per-motion local range: "
                    f"env_ids={offending.detach().cpu().tolist()}, "
                    f"motion_ids={selected[offending].detach().cpu().tolist()}, "
                    f"delta_b_m={selected_delta_b[offending].detach().cpu().tolist()}, "
                    f"half_range_m={half_ranges[offending].detach().cpu().tolist()}"
                )
        else:
            selected_delta_b = target_position_b - anchors_b[selected]
            half_ranges = None
        enabled_by_motion = torch.as_tensor(
            getattr(motion.cfg, "external_control_anchor_enabled_by_motion", ()),
            dtype=torch.bool,
            device=env.unwrapped.device,
        )
        if enabled_by_motion.numel() == 0:
            enabled_by_motion = torch.ones(
                motion.motion.num_motions, dtype=torch.bool, device=env.unwrapped.device
            )
        auto_selected_motion_ids = selected
        auto_motion_selection = {
            "selected_motion_ids": selected.detach().cpu().tolist(),
            "nearest_anchor_distance_m": distances.detach().cpu().tolist(),
            "candidate_anchor_position_b_m": anchors_b.detach().cpu().tolist(),
            "selected_delta_b_m": selected_delta_b.detach().cpu().tolist(),
            "candidate_motion_enabled_by_id": enabled_by_motion.detach().cpu().tolist(),
            "local_half_range_by_motion_m": None
            if auto_select_local_half_ranges is None and control_half_ranges.numel() == 0
            else candidate_half_ranges.detach().cpu().tolist(),
            "max_anchor_distance_m": auto_select_max_anchor_distance_m,
            "local_range_tolerance_m": auto_select_local_range_tolerance_m,
        }
    if has_motion_command and (
        forced_motion_id is not None or auto_selected_motion_ids is not None
    ):
        motion = env.unwrapped.command_manager.get_term("motion")
        if auto_selected_motion_ids is not None:
            selected_motion_ids = auto_selected_motion_ids
        else:
            motion_id = int(forced_motion_id)
            if not 0 <= motion_id < motion.motion.num_motions:
                raise ValueError(
                    f"motion_id={motion_id} outside manifest range [0, {motion.motion.num_motions - 1}]"
                )
            selected_motion_ids = torch.full(
                (env.num_envs,), motion_id, dtype=torch.long, device=env.unwrapped.device
            )
        if external_hit_time_s is not None:
            if not hasattr(motion.motion, "hit_frame"):
                raise RuntimeError(
                    "external_hit_time_s requires a motion manifest with hit frames"
                )
            if selected_motion_ids.numel() != 1:
                raise RuntimeError(
                    "external_hit_time_s requires exactly one selected motion"
                )
            control_dt = float(env.unwrapped.step_dt)
            initial_prelude_steps = int(motion.prelude_steps)
            hit_frame = int(
                motion.motion.hit_frame[selected_motion_ids[0]].item()
            )
            # The target is latched after the forced-motion phase priming
            # below. One command update always occurs; synchronized target
            # audits take one additional update after sibling state copy. The
            # schedule is defined from that actual COMMIT/latch boundary, not
            # from the earlier reference rewind.
            precommit_phase_steps = 1 + int(
                target_audit_enabled and target_audit_synchronize_siblings
            )
            hit_schedule = schedule_external_hit_time(
                requested_time_s=external_hit_time_s,
                control_dt_s=control_dt,
                initial_prelude_steps=initial_prelude_steps,
                motion_hit_frame=hit_frame,
                precommit_phase_steps=precommit_phase_steps,
                max_added_delay_s=external_hit_max_added_delay_s,
            )
            motion.prelude_steps = initial_prelude_steps + int(
                hit_schedule["added_ready_hold_steps"]
            )
        # The wrapped env has already performed the task's physical
        # strike-ready reset.  Only replace the future reference and reset its
        # phase bookkeeping; do not teleport the floating base into a motion
        # frame, which would create an artificial reset seam in the video.
        motion.motion_ids.copy_(selected_motion_ids)
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
        if multi_shot_sequence is None and sync_initial_motion_state:
            # Legacy forced-motion replay synchronizes the initial physical
            # state.  Multi-shot evaluation deliberately skips this path:
            # every shot starts from the actual settled state, never a
            # reference teleport.
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
                if target_audit_enabled and target_audit_synchronize_siblings:
                    _synchronize_target_audit_siblings(env.unwrapped)
                    motion_cmd._update_command()
                    # Rebuild the anchor after physical synchronization so
                    # every external target is latched from the same state.
                    racket._resample_command(env_ids)
                    racket._compute_strike_timing()
        elif multi_shot_sequence is None:
            print(
                "[INFO] preserving task-configured physical READY stance; "
                "motion id/phase updated without frame-0 teleport",
                flush=True,
            )
            # Keep the post-branch diagnostics valid when the replay
            # deliberately preserves the configured READY pose instead of
            # entering the legacy frame-0 physical synchronization branch.
            motion_cmd = env.unwrapped.command_manager.get_term("motion")
        if auto_motion_selection is None:
            print(
                f"[INFO] forced manifest motion_id={int(selected_motion_ids[0].item())} "
                "for deterministic replay",
                flush=True,
            )
        else:
            print(
                "[INFO] auto-selected manifest motions: "
                f"ids={auto_motion_selection['selected_motion_ids']}, "
                f"nearest_anchor_distance_m={auto_motion_selection['nearest_anchor_distance_m']}",
                flush=True,
            )
        if hit_schedule is not None:
            print(
                "[INFO] external hit schedule: "
                f"requested={hit_schedule['request_time_from_commit_s']:.3f}s, "
                f"native={hit_schedule['native_hit_time_s']:.3f}s, "
                f"ready_hold={hit_schedule['added_ready_hold_s']:.3f}s",
                flush=True,
            )

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
            # Command metrics are reset in-place by Isaac Lab.  Creating them
            # inside inference_mode makes them immutable outside that context
            # and causes the calibration rewind below to fail during reset.
            # This synchronization writes simulator state only and does not
            # construct an autograd graph, so keep its persistent command
            # tensors as ordinary tensors.
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

    target_audit_context = None
    target_audit_report_path = None
    if target_audit_enabled:
        from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

        raw_env = env.unwrapped
        env_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
        racket_cmd = raw_env.command_manager.get_term("racket_target")
        robot = raw_env.scene["robot"]
        # Forced-motion replay has just written and forwarded a new physical
        # state.  Refresh the FK cache before model_900 consumes it; otherwise
        # the first upper observation still contains the pre-teleport racket
        # state (and parallel env spacing leaks into the paired audit).
        racket_cmd._compute_racket_state()
        racket_cmd._compute_strike_timing()
        # P0 may itself sample a paired external displacement during reset.
        # An audit target must be specified relative to the frozen manifest
        # anchor, not relative to that already-offset runtime command.  Using
        # ``racket_target_*`` here lets the audit grid cancel the P0 pair rows
        # for env ids 1..6 (and makes every observed adapter delta zero).
        anchor_position_b = racket_cmd.racket_anchor_target_pos_b().detach().clone()
        anchor_position_w = racket_cmd.racket_anchor_target_pos_w.detach().clone()
        command_base_position_w = robot.data.root_pos_w.detach().clone()
        command_base_heading_w = yaw_quat(robot.data.root_quat_w).detach().clone()

        if external_target_position_rows is not None:
            target_position_b = torch.tensor(
                external_target_position_rows,
                dtype=torch.float32,
                device=raw_env.device,
            )
        else:
            offset_b = torch.tensor(
                external_target_offset_rows,
                dtype=torch.float32,
                device=raw_env.device,
            )
            target_position_b = anchor_position_b + offset_b
        racket_cmd.set_external_target_position_b(env_ids, target_position_b)

        requested_offset_b = target_position_b - anchor_position_b
        target_velocity_b = quat_rotate_inverse(
            command_base_heading_w, racket_cmd.racket_target_vel_w
        ).detach().clone()
        target_normal_b = quat_rotate_inverse(
            command_base_heading_w, racket_cmd.racket_target_normal_w
        ).detach().clone()
        target_audit_context = {
            "anchor_position_b": anchor_position_b,
            "anchor_position_w": anchor_position_w,
            "target_position_b": target_position_b.detach().clone(),
            "target_position_w": racket_cmd.racket_target_pos_w.detach().clone(),
            "requested_offset_b": requested_offset_b.detach().clone(),
            "target_velocity_b": target_velocity_b,
            "target_normal_b": target_normal_b,
            "command_base_position_w": command_base_position_w,
            "command_base_heading_w": command_base_heading_w,
        }
        report_path = cfg.get("target_audit_report", None)
        if report_path is None:
            report_path = cfg.get("single_shot_report", None)
        if report_path is None:
            raise ValueError(
                "external target audit requires target_audit_report=<path> "
                "(single_shot_report is accepted as a fallback)"
            )
        target_audit_report_path = pathlib.Path(str(report_path)).expanduser()
        if not target_audit_report_path.is_absolute():
            target_audit_report_path = pathlib.Path.cwd() / target_audit_report_path
        target_audit_report_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "[INFO] latched external racket targets: "
            f"motion_ids={motion_cmd.motion_ids.detach().cpu().tolist()}, "
            f"trials={raw_env.num_envs}, "
            f"offset_b_m={requested_offset_b.detach().cpu().numpy().round(4).tolist()}",
            flush=True,
        )

    strike_goal_shadow_pipeline = None
    strike_goal_shadow_context = None
    if strike_goal_shadow_enabled:
        from isaaclab.utils.math import matrix_from_quat, yaw_quat

        raw_env = env.unwrapped
        command_path = pathlib.Path(str(strike_goal_shadow_command_path)).expanduser()
        if not command_path.is_absolute():
            command_path = pathlib.Path.cwd() / command_path
        message_payload = json.loads(command_path.read_text(encoding="utf-8"))
        if "racket_command" in message_payload:
            message_payload = message_payload["racket_command"]
        planner_command = PlannerRacketCommand.from_ros_message(message_payload)

        rotation_cfg = cfg.get("strike_goal_shadow_hope_world_to_sim_rotation")
        translation_cfg = cfg.get("strike_goal_shadow_hope_world_to_sim_translation")
        if OmegaConf.is_config(rotation_cfg):
            rotation_cfg = OmegaConf.to_container(rotation_cfg, resolve=True)
        if OmegaConf.is_config(translation_cfg):
            translation_cfg = OmegaConf.to_container(translation_cfg, resolve=True)
        hope_world_to_sim = StrikeGoalFrameTransform(
            source_frame=planner_command.goal.frame_id,
            target_frame="isaac_tracking_world/v1",
            rotation=rotation_cfg,
            translation=translation_cfg,
        )

        robot = raw_env.scene["robot"]
        receipt_base_position_w = robot.data.root_pos_w[0].detach().clone()
        receipt_base_heading_w = yaw_quat(
            robot.data.root_quat_w[0:1]
        ).detach().clone()
        receipt_heading_matrix_w = matrix_from_quat(receipt_base_heading_w)[0]
        sim_to_receipt_rotation = receipt_heading_matrix_w.transpose(0, 1)
        sim_to_receipt_translation = -torch.mv(
            sim_to_receipt_rotation, receipt_base_position_w
        )
        sim_to_receipt = StrikeGoalFrameTransform(
            source_frame="isaac_tracking_world/v1",
            target_frame=BASE_HEADING_RECEIPT_FRAME_V1,
            rotation=sim_to_receipt_rotation.detach().cpu().tolist(),
            translation=sim_to_receipt_translation.detach().cpu().tolist(),
        )
        source_to_policy = hope_world_to_sim.followed_by(sim_to_receipt)
        verified_delay = float(
            cfg.get("strike_goal_shadow_verified_pre_receipt_delay_s", 0.0)
        )
        latched_goal = LatchedStrikeGoal.from_planner_command(
            planner_command,
            received_control_time_s=0.0,
            control_clock_domain="isaac_sim",
            verified_pre_receipt_delay_s=verified_delay,
        )
        calibration = isaac_diagnostic_proxy_contact_calibration()
        strike_goal_shadow_pipeline = StrikeGoalShadowPipeline(
            latched_goal=latched_goal,
            source_to_policy_transform=source_to_policy,
            contact_calibration=calibration,
        )
        shadow_report_path = pathlib.Path(str(strike_goal_shadow_report)).expanduser()
        if not shadow_report_path.is_absolute():
            shadow_report_path = pathlib.Path.cwd() / shadow_report_path
        shadow_report_path.parent.mkdir(parents=True, exist_ok=True)
        strike_goal_shadow_context = {
            "report_path": shadow_report_path,
            "command_path": command_path,
            "hope_world_to_sim": hope_world_to_sim,
            "sim_to_receipt": sim_to_receipt,
            "sim_to_receipt_rotation": sim_to_receipt_rotation,
            "sim_to_receipt_translation": sim_to_receipt_translation,
            "receipt_base_position_w": receipt_base_position_w,
            "receipt_base_heading_w": receipt_base_heading_w[0],
            "verified_pre_receipt_delay_s": verified_delay,
            "policy_action_trace": [],
        }
        print(
            "[INFO] strike-goal P2 shadow latched: "
            f"contract={planner_command.goal.contract_version}, "
            f"source_frame={planner_command.goal.frame_id}, "
            f"policy_frame={BASE_HEADING_RECEIPT_FRAME_V1}, "
            f"tts={latched_goal.goal_at_receipt.time_to_hit_s:.3f}s, "
            "action_effect=false",
            flush=True,
        )

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
    # Optional diagnostic mode: keep the last pre-step frame when a vector
    # environment auto-resets a terminated row.  The normal replay preserves
    # its historical post-step capture behavior; this mode is used when the
    # user needs to see the actual fall rather than the reset pose.
    video_stop_on_termination = bool(cfg.get("video_stop_on_termination", False))
    video_capture_pre_step_on_termination = bool(
        cfg.get("video_capture_pre_step_on_termination", False)
    )
    def _overlay_fall_audit(frame, raw_env):
        """Annotate replay frames with the same unified physical audit state."""
        if frame is None:
            return frame
        try:
            import numpy as np
            from PIL import Image, ImageDraw
            from training.tasks.tracking.mdp.fall_state import FallLevel, FallReason, unified_fall_state

            state = unified_fall_state(raw_env)
            index = 0
            level = FallLevel(int(state.risk_level[index].item())).name
            reason = FallReason(int(state.fall_reason[index].item())).name
            margins = state.support_margins[index].detach().cpu().tolist()
            lines = [
                f"fall={level} risk={float(state.risk_score[index]):.3f} reason={reason}",
                f"tilt F/L={float(state.forward_tilt_rad[index]):+.2f}/{float(state.lateral_tilt_rad[index]):+.2f} rad",
                f"torso F/L={float(state.torso_forward_tilt_rad[index]):+.2f}/{float(state.torso_lateral_tilt_rad[index]):+.2f} rad",
                f"h_root={float(state.relative_root_height_m[index]):.3f}m margins=" + ",".join(f"{v:+.3f}" for v in margins),
                f"pred_unrec={bool(state.predicted_unrecoverable[index])} confirmed={bool(state.confirmed_fall[index])} ready={bool(state.recovery_ready[index])} hold={int(state.recovery_stable_steps[index])}",
            ]
            image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            line_h = 18
            box_h = line_h * len(lines) + 10
            draw.rectangle((8, 8, min(image.width - 8, 900), 8 + box_h), fill=(0, 0, 0, 170))
            color = (255, 80, 80, 255) if bool(state.confirmed_fall[index]) else (255, 230, 120, 255)
            for row, line in enumerate(lines):
                draw.text((14, 12 + row * line_h), line, fill=color)
            return np.asarray(image)
        except Exception:
            # Overlay is diagnostic-only; never hide the raw replay if a
            # renderer/PIL backend is unavailable. The JSON trace remains the
            # authoritative state evidence.
            return frame
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
    min_base_upright = 1.0
    max_root_tilt_rad = 0.0
    min_torso_height = float("inf")
    max_torso_tilt_rad = 0.0
    strict_fall_candidate_steps = 0
    strict_fall_first_candidate_step = None
    torso_fall_candidate_steps = 0
    torso_fall_first_candidate_step = None
    strict_fall_consecutive = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.unwrapped.device
    )
    torso_fall_consecutive = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.unwrapped.device
    )
    strict_fall_max_tilt_rad = float(cfg.get("strict_fall_max_tilt_rad", 0.785398))
    strict_fall_min_height = float(cfg.get("strict_fall_min_height", 0.82))
    strict_fall_max_torso_tilt_rad = float(cfg.get("strict_fall_max_torso_tilt_rad", 0.95))
    strict_fall_min_torso_height = float(cfg.get("strict_fall_min_torso_height", 0.70))
    strict_fall_required_steps = int(cfg.get("strict_fall_required_steps", 2))
    try:
        torso_body_ids, torso_body_names = robot.find_bodies(["torso_Link"], preserve_order=True)
        torso_body_id = int(torso_body_ids[0]) if torso_body_names else 0
        has_torso_probe = bool(torso_body_names)
    except Exception:
        torso_body_id = 0
        has_torso_probe = False
    terminated_count = 0
    truncated_count = 0
    unified_predicted_unrecoverable_steps = 0
    unified_confirmed_fall_steps = 0
    unified_recovery_ready_steps = 0
    first_termination_step = None
    multi_shot_records = []
    multi_shot_transitions = []
    multi_shot_index = 0
    multi_shot_hit_recorded = False
    multi_shot_complete = False
    multi_shot_failure = None
    multi_shot_last_state = None
    multi_shot_start_state = None
    multi_shot_last_pre_step_state = None
    multi_shot_trace = []
    multi_shot_recovery_milestones = {}
    multi_shot_report_path = None
    record_trace = bool(cfg.get("record_trace", False))
    single_shot_trace = []
    single_shot_termination = None
    single_shot_hit_record = None
    target_audit_hit_records = (
        [None for _ in range(env.num_envs)] if target_audit_enabled else None
    )
    # A target-response audit used to expose only an aggregate termination
    # count.  That is sufficient for an all-siblings-stable pass, but it
    # cannot identify which finite-difference action direction delayed or
    # accelerated a fall.  Keep each sibling's first physical terminal event
    # so controllability and support audits can be evaluated causally.
    target_audit_first_physical_termination_step = (
        torch.full((env.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device)
        if target_audit_enabled
        else None
    )
    target_audit_first_physical_termination_reasons = (
        [None for _ in range(env.num_envs)] if target_audit_enabled else None
    )
    target_audit_initial_actor = None
    target_audit_all_hit_control_step = None
    def _target_actor_snapshot(
        raw_env, coordinator_actions, policy_observation
    ):
        """Return the exact model_900 input/output used for the current step."""
        action_term = raw_env.action_manager.get_term("joint_pos")
        if hasattr(action_term, "upper_last_observation"):
            upper_observation = action_term.upper_last_observation.detach().clone()
            upper_action = action_term.upper_raw_actions.detach().clone()
            upper_policy = getattr(action_term, "_upper_policy", None)
            if upper_policy is None:
                raise RuntimeError(
                    "external target audit could not access the frozen upper policy"
                )
            normalized = torch.clamp(
                (upper_observation - upper_policy.mean) / upper_policy.std,
                -100.0,
                100.0,
            )
        elif hasattr(action_term, "_annealed_upper_prior_last_observation"):
            # CompletePriors keeps the frozen model_900 branch under the
            # annealed-prior names rather than the legacy _upper_policy
            # names.  External target-grid replay is evaluation-only, so use
            # the exact cached model_900 observation/output from this action
            # term instead of rejecting an otherwise valid CompletePriors
            # target audit.
            upper_policy = getattr(action_term, "_annealed_upper_prior_policy", None)
            if upper_policy is None:
                raise RuntimeError(
                    "external target audit could not access the CompletePriors model_900 policy"
                )
            upper_observation = action_term._annealed_upper_prior_last_observation.detach().clone()
            upper_action = getattr(raw_env, "f0_upper_last_action", None)
            if upper_action is None:
                raise RuntimeError(
                    "external target audit could not access the CompletePriors model_900 output"
                )
            upper_action = upper_action.detach().clone()
            normalized = torch.clamp(
                (upper_observation - upper_policy.mean) / upper_policy.std,
                -100.0,
                100.0,
            )
        elif (
            policy_observation.shape[-1] == 56
            and coordinator_actions.shape[-1] == 10
        ):
            # Fixed-base model_900 is the top-level runner policy rather than
            # a frozen child action.  ``policy_observation`` and ``actions``
            # are exactly the tensors used immediately before env.step().
            upper_observation = policy_observation.detach().clone()
            upper_action = coordinator_actions.detach().clone()
            normalized = ppo_runner.obs_normalizer(
                upper_observation
            ).detach().clone()
        else:
            raise RuntimeError(
                "external target audit requires either the frozen model_900 "
                "composite action or a direct 56-D/10-D model_900 policy"
            )
        return {
            "upper_observation": upper_observation,
            "upper_observation_normalized": normalized,
            "upper_actor_output": upper_action,
            "coordinator_action": coordinator_actions.detach().clone(),
            "coordinator_target_feedforward": (
                action_term.coordinator_target_feedforward_last_action.detach().clone()
                if hasattr(action_term, "coordinator_target_feedforward_last_action")
                else None
            ),
        }

    def _multi_shot_state_snapshot(raw_env):
        from training.tasks.tracking.mdp.observations import stagger_support_state
        from training.tasks.tracking.mdp.fall_state import unified_fall_state, FallLevel, FallReason
        from training.robots.agibot_a3 import A3_FEET_BODIES
        from isaaclab.utils.math import quat_rotate_inverse

        support = stagger_support_state(raw_env)
        fall_state = unified_fall_state(raw_env)
        robot = raw_env.scene["robot"]
        gravity_b = robot.data.projected_gravity_b[0]
        tilt = torch.acos(torch.clamp(-gravity_b[2], -1.0, 1.0))
        torso_ids, torso_names = robot.find_bodies(["torso_Link"], preserve_order=True)
        torso_id = int(torso_ids[0]) if torso_names else 0
        torso_pos = robot.data.body_pos_w[0, torso_id]
        gravity_w = torch.zeros_like(torso_pos)
        gravity_w[2] = -1.0
        torso_gravity_b = quat_rotate_inverse(
            robot.data.body_quat_w[0, torso_id], gravity_w
        )
        torso_tilt = torch.acos(torch.clamp(-torso_gravity_b[2], -1.0, 1.0))
        actual_q = robot.data.joint_pos[0]
        soft_limits = robot.data.soft_joint_pos_limits[0]
        hard_limits = robot.data.joint_pos_limits[0]
        actual_soft_margin = torch.minimum(
            actual_q - soft_limits[:, 0], soft_limits[:, 1] - actual_q
        )
        actual_hard_margin = torch.minimum(
            actual_q - hard_limits[:, 0], hard_limits[:, 1] - actual_q
        )
        actual_soft_index = int(torch.argmin(actual_soft_margin).item())
        actual_hard_index = int(torch.argmin(actual_hard_margin).item())
        waist_roll_id = robot.joint_names.index("waist_roll_joint")
        sensor = raw_env.scene.sensors["contact_forces"]
        contact_force = torch.linalg.vector_norm(
            sensor.data.net_forces_w[0], dim=-1
        )
        max_contact_index = int(torch.argmax(contact_force).item())
        foot_sensor_ids, resolved_sensor_feet = sensor.find_bodies(
            A3_FEET_BODIES, preserve_order=True
        )
        foot_robot_ids, resolved_robot_feet = robot.find_bodies(
            A3_FEET_BODIES, preserve_order=True
        )
        if resolved_sensor_feet != A3_FEET_BODIES or resolved_robot_feet != A3_FEET_BODIES:
            raise RuntimeError("P4A trace could not resolve the configured A3 feet")
        foot_force = contact_force[
            torch.as_tensor(foot_sensor_ids, device=raw_env.device, dtype=torch.long)
        ]
        foot_speed = torch.linalg.vector_norm(
            robot.data.body_lin_vel_w[
                0,
                torch.as_tensor(foot_robot_ids, device=raw_env.device, dtype=torch.long),
                :2,
            ],
            dim=-1,
        )
        loaded_foot_speed = torch.where(
            foot_force >= 20.0, foot_speed, torch.zeros_like(foot_speed)
        )
        torque_ratio = torch.abs(robot.data.applied_torque[0]) / torch.clamp(
            robot.data.joint_effort_limits[0], min=1.0e-6
        )
        # P4D needs a time-resolved strike-window trace.  The previous
        # snapshot contained the joint/action chain but only emitted racket
        # errors at the single tagged hit frame, which cannot distinguish a
        # global phase error from an unsuitable trajectory shape.  Read the
        # same FK and target buffers used by the task; this is diagnostics
        # only and must not feed back into the controller.
        racket_command = raw_env.command_manager.get_term("racket_target")
        racket_pos_w, racket_vel_w, _ = racket_state_w(raw_env)
        racket_normal = racket_normal_w(raw_env, normal_axis=1, normal_sign=1.0)
        target_pos_w = racket_command.racket_target_pos_w[0]
        target_vel_w = racket_command.racket_target_vel_w[0]
        target_normal_w = racket_command.racket_target_normal_w[0]
        normal_alignment = torch.clamp(
            torch.dot(racket_normal[0], target_normal_w), -1.0, 1.0
        )
        velocity_speed = torch.linalg.vector_norm(racket_vel_w[0])
        target_speed = torch.linalg.vector_norm(target_vel_w)
        velocity_direction_alignment = torch.tensor(
            float("nan"), device=raw_env.device
        )
        if float(velocity_speed.item()) > 1.0e-6 and float(target_speed.item()) > 1.0e-6:
            velocity_direction_alignment = torch.clamp(
                torch.dot(racket_vel_w[0], target_vel_w) / (velocity_speed * target_speed),
                -1.0,
                1.0,
            )
        snapshot = {
            "root_position_w_m": [
                float(value)
                for value in robot.data.root_pos_w[0].tolist()
            ],
            "root_tilt_deg": float(torch.rad2deg(tilt).item()),
            "root_tilt_rad": float(tilt.item()),
            "torso_height_m": float(torso_pos[2].item()),
            "torso_tilt_deg": float(torch.rad2deg(torso_tilt).item()),
            "torso_tilt_rad": float(torso_tilt.item()),
            "root_forward_velocity_mps": float(
                raw_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
            ),
            "root_pitch_rate_radps": float(
                raw_env.scene["robot"].data.root_ang_vel_b[0, 1].item()
            ),
            "capture_rel_support_x_m": float(
                support["capture_rel_support_x_b"][0].item()
            ),
            "capture_front_margin_m": float(
                support["capture_front_margin"][0].item()
            ),
            "capture_rear_margin_m": float(
                support["capture_rear_margin"][0].item()
            ),
            "both_feet_contact": bool(support["contacts"][0].all().item()),
            "fall_state": {
                "risk_score": float(fall_state.risk_score[0].item()),
                "risk_level": int(fall_state.risk_level[0].item()),
                "risk_level_name": FallLevel(int(fall_state.risk_level[0].item())).name,
                "fall_reason": int(fall_state.fall_reason[0].item()),
                "fall_reason_name": FallReason(int(fall_state.fall_reason[0].item())).name,
                "forward_tilt_rad_signed": float(fall_state.forward_tilt_rad[0].item()),
                "lateral_tilt_rad_signed": float(fall_state.lateral_tilt_rad[0].item()),
                "torso_forward_tilt_rad_signed": float(fall_state.torso_forward_tilt_rad[0].item()),
                "torso_lateral_tilt_rad_signed": float(fall_state.torso_lateral_tilt_rad[0].item()),
                "relative_root_height_m": float(fall_state.relative_root_height_m[0].item()),
                "relative_torso_height_m": float(fall_state.relative_torso_height_m[0].item()),
                "root_linear_velocity_b_mps": [float(value) for value in fall_state.root_linear_velocity_b[0].tolist()],
                "root_angular_velocity_b_radps": [float(value) for value in fall_state.root_angular_velocity_b[0].tolist()],
                "torso_linear_velocity_b_mps": [float(value) for value in fall_state.torso_linear_velocity_b[0].tolist()],
                "torso_angular_velocity_b_radps": [float(value) for value in fall_state.torso_angular_velocity_b[0].tolist()],
                "com_position_b": [float(value) for value in fall_state.com_position_b[0].tolist()],
                "com_velocity_b": [float(value) for value in fall_state.com_velocity_b[0].tolist()],
                "capture_point_b": [float(value) for value in fall_state.capture_point_b[0].tolist()],
                "cop_position_b": [float(value) for value in fall_state.cop_position_b[0].tolist()],
                "support_margins_m": [float(value) for value in fall_state.support_margins[0].tolist()],
                "predicted_min_support_margin_m": float(fall_state.predicted_support_margins[0].min().item()),
                "foot_slip_mps": [float(value) for value in fall_state.foot_slip_mps[0].tolist()],
                "illegal_body_contact": bool(fall_state.illegal_body_contact[0].item()),
                "actuator_saturation": float(fall_state.actuator_saturation[0].item()),
                "safety_projection": float(fall_state.safety_projection[0].item()),
                "predicted_unrecoverable": bool(fall_state.predicted_unrecoverable[0].item()),
                "confirmed_fall": bool(fall_state.confirmed_fall[0].item()),
                "recovery_ready": bool(fall_state.recovery_ready[0].item()),
                "recovery_progress": float(fall_state.recovery_progress[0].item()),
                "recovery_stable_steps": int(fall_state.recovery_stable_steps[0].item()),
                "cycle_phase": int(getattr(raw_env, "fall_cycle_phase", torch.zeros_like(fall_state.risk_level))[0].item()),
                "cycle_guard_steps": int(getattr(raw_env, "fall_cycle_guard_steps", torch.zeros_like(fall_state.risk_level))[0].item()),
                "cycle_recovery_steps": int(getattr(raw_env, "fall_cycle_recovery_steps", torch.zeros_like(fall_state.risk_level))[0].item()),
                "risk_components": {
                    name: float(value[0].item()) for name, value in fall_state.risk_components.items()
                },
                "prediction": {
                    "horizons_s": [float(value) for value in fall_state.prediction["horizons_s"].tolist()],
                    "predicted_tilt_rad": [float(value) for value in fall_state.prediction["tilt_rad"][0].tolist()],
                    "predicted_relative_height_m": [float(value) for value in fall_state.prediction["relative_height_m"][0].tolist()],
                    "predicted_time_to_hit_s": [float(value) for value in fall_state.prediction["time_to_hit_s"][0].tolist()],
                    "future_reference_available": bool(fall_state.prediction["future_reference_available"][0].item()),
                },
            },
            "minimum_actual_soft_joint_margin_rad": float(
                actual_soft_margin[actual_soft_index].item()
            ),
            "minimum_actual_soft_joint_margin_joint": robot.joint_names[
                actual_soft_index
            ],
            "minimum_actual_hard_joint_margin_rad": float(
                actual_hard_margin[actual_hard_index].item()
            ),
            "minimum_actual_hard_joint_margin_joint": robot.joint_names[
                actual_hard_index
            ],
            "waist_roll_position_rad": float(actual_q[waist_roll_id].item()),
            "waist_roll_soft_margin_rad": float(
                actual_soft_margin[waist_roll_id].item()
            ),
            "waist_roll_hard_margin_rad": float(
                actual_hard_margin[waist_roll_id].item()
            ),
            "foot_contact_force_n": [float(value) for value in foot_force.tolist()],
            "foot_tangential_speed_mps": [
                float(value) for value in foot_speed.tolist()
            ],
            "loaded_foot_tangential_speed_max_mps": float(
                loaded_foot_speed.max().item()
            ),
            "max_robot_contact_force_n": float(contact_force[max_contact_index].item()),
            "max_robot_contact_body": sensor.body_names[max_contact_index],
            "max_effort_limit_ratio": float(torque_ratio.max().item()),
            "racket_state": {
                "actual_position_w_m": [float(value) for value in racket_pos_w[0].tolist()],
                "actual_velocity_w_mps": [float(value) for value in racket_vel_w[0].tolist()],
                "actual_normal_w": [float(value) for value in racket_normal[0].tolist()],
                "target_position_w_m": [float(value) for value in target_pos_w.tolist()],
                "target_velocity_w_mps": [float(value) for value in target_vel_w.tolist()],
                "target_normal_w": [float(value) for value in target_normal_w.tolist()],
                "position_error_m": float(torch.linalg.vector_norm(racket_pos_w[0] - target_pos_w).item()),
                "velocity_error_mps": float(torch.linalg.vector_norm(racket_vel_w[0] - target_vel_w).item()),
                "normal_error_deg": float(torch.rad2deg(torch.acos(normal_alignment)).item()),
                "actual_speed_mps": float(velocity_speed.item()),
                "target_speed_mps": float(target_speed.item()),
                "speed_error_mps": float((velocity_speed - target_speed).item()),
                "velocity_direction_error_deg": float(
                    torch.rad2deg(torch.acos(velocity_direction_alignment)).item()
                ) if bool(torch.isfinite(velocity_direction_alignment).item()) else None,
            },
        }
        if hasattr(raw_env, "stage_a_sagittal_exit_state"):
            snapshot.update(
                {
                    "stage_a_sagittal_state": int(
                        raw_env.stage_a_sagittal_exit_state[0].item()
                    ),
                    "stage_a_sagittal_scale": float(
                        raw_env.stage_a_sagittal_exit_scale[0].item()
                    ),
                    "stage_a_rearm_ready": bool(
                        raw_env.stage_a_sagittal_rearm_ready[0].item()
                    ),
                    "stage_a_rearm_stable": bool(
                        raw_env.stage_a_sagittal_rearm_stable[0].item()
                    ),
                }
            )
        try:
            motion = raw_env.command_manager.get_term("motion")
            arm_ids, _ = raw_env.scene["robot"].find_joints(
                [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                preserve_order=True,
            )
            arm_ids_tensor = torch.as_tensor(
                arm_ids, device=raw_env.device, dtype=torch.long
            )
            arm_position_error = torch.abs(
                raw_env.scene["robot"].data.joint_pos[0, arm_ids_tensor]
                - motion.ready_joint_pos[0, arm_ids_tensor]
            )
            arm_velocity = torch.abs(
                raw_env.scene["robot"].data.joint_vel[0, arm_ids_tensor]
            )
            leg_ids, _ = raw_env.scene["robot"].find_joints(
                [
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "right_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "right_hip_yaw_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                    "left_ankle_pitch_joint",
                    "right_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_ankle_roll_joint",
                ],
                preserve_order=True,
            )
            leg_ids_tensor = torch.as_tensor(
                leg_ids, device=raw_env.device, dtype=torch.long
            )
            leg_position_error = torch.abs(
                raw_env.scene["robot"].data.joint_pos[0, leg_ids_tensor]
                - motion.ready_joint_pos[0, leg_ids_tensor]
            )
            leg_velocity = torch.abs(
                raw_env.scene["robot"].data.joint_vel[0, leg_ids_tensor]
            )
            snapshot.update(
                {
                    "motion_id": int(motion.motion_ids[0].item()),
                    "motion_frame": int(motion.time_steps[0].item()),
                    "prelude_elapsed_steps": int(
                        motion.prelude_elapsed_steps[0].item()
                    ),
                    "tail_steps": int(motion.tail_steps[0].item()),
                    "right_arm_ready_error_max_rad": float(
                        arm_position_error.max().item()
                    ),
                    "right_arm_velocity_max_radps": float(
                        arm_velocity.max().item()
                    ),
                    "leg_ready_error_max_rad": float(
                        leg_position_error.max().item()
                    ),
                    "leg_velocity_max_radps": float(leg_velocity.max().item()),
                }
            )
            reference_q = motion.joint_pos[0]
            reference_soft_margin = torch.minimum(
                reference_q - soft_limits[:, 0], soft_limits[:, 1] - reference_q
            )
            reference_hard_margin = torch.minimum(
                reference_q - hard_limits[:, 0], hard_limits[:, 1] - reference_q
            )
            reference_soft_index = int(torch.argmin(reference_soft_margin).item())
            reference_hard_index = int(torch.argmin(reference_hard_margin).item())
            snapshot.update(
                {
                    "minimum_reference_soft_joint_margin_rad": float(
                        reference_soft_margin[reference_soft_index].item()
                    ),
                    "minimum_reference_soft_joint_margin_joint": robot.joint_names[
                        reference_soft_index
                    ],
                    "minimum_reference_hard_joint_margin_rad": float(
                        reference_hard_margin[reference_hard_index].item()
                    ),
                    "minimum_reference_hard_joint_margin_joint": robot.joint_names[
                        reference_hard_index
                    ],
                    "waist_roll_reference_rad": float(
                        reference_q[waist_roll_id].item()
                    ),
                    "waist_roll_reference_soft_margin_rad": float(
                        reference_soft_margin[waist_roll_id].item()
                    ),
                    "waist_roll_reference_hard_margin_rad": float(
                        reference_hard_margin[waist_roll_id].item()
                    ),
                }
            )
            action_term = raw_env.action_manager.get_term("joint_pos")
            if hasattr(action_term, "v13b_startup_diagnostics"):
                startup = action_term.v13b_startup_diagnostics
                def _startup_row(value):
                    return [float(item) for item in value[0].detach().cpu().tolist()]

                snapshot["v13b_startup_decomposition"] = {
                    "episode_step": int(startup["step"][0].item()),
                    "joint_names": list(robot.joint_names),
                    "q_reset_rad": _startup_row(startup["q_reset"]),
                    "q_actual_rad": _startup_row(startup["q_actual"]),
                    "q_ready_rad": _startup_row(action_term._ready_full),
                    "lower_prior_delta_rad": _startup_row(startup["lower_prior_delta"]),
                    "lower_student_delta_rad": _startup_row(startup["lower_student_delta"]),
                    "microstep_delta_rad": _startup_row(startup["microstep_delta"]),
                    "upper_prior_delta_rad": _startup_row(startup["upper_prior_delta"]),
                    "upper_student_delta_rad": _startup_row(startup["upper_student_delta"]),
                    "final_q_target_rad": _startup_row(startup["q_target"]),
                    "first_command_jump_rad": _startup_row(startup["first_command_jump"]),
                    "first_command_jump_rms_rad": float(startup["first_command_jump_rms"][0].item()),
                    "first_command_jump_abs_max_rad": float(startup["first_command_jump_abs_max"][0].item()),
                    "previous_action_zero": bool(
                        torch.all(raw_env.action_manager.prev_action[0].abs() < 1.0e-8).item()
                    ),
                }
            if hasattr(action_term, "upper_reference_actions"):
                upper_names = list(action_term.cfg.upper_joint_names)
                upper_ids = action_term._upper_joint_ids_tensor
                upper_actual = actual_q[upper_ids]
                upper_actual_velocity = robot.data.joint_vel[0, upper_ids]
                upper_soft_margin = actual_soft_margin[upper_ids]
                upper_hard_margin = actual_hard_margin[upper_ids]
                command = action_term.full_joint_targets[0, upper_ids]
                command_velocity = action_term.full_joint_velocity_targets[
                    0, upper_ids
                ]
                command_limits = soft_limits[upper_ids]
                command_soft_margin = torch.minimum(
                    command - command_limits[:, 0],
                    command_limits[:, 1] - command,
                )
                zeros = torch.zeros_like(action_term.upper_reference_actions)
                target_adapter = getattr(
                    action_term, "upper_target_adapter_contribution", zeros
                )
                def _upper_row(value):
                    return [float(item) for item in value[0].tolist()]

                # Linearize the current TCP around the actual PhysX state.
                # This is trace-only evidence for P4D responsibility audits:
                # it ranks which safe-reference tracking errors can explain
                # the observed racket-position error without changing any
                # action, reference, or safety filter.
                jacobians = robot.root_physx_view.get_jacobians()
                if racket_command._racket_mode == "body":
                    jacobian_body = racket_command._racket_body_index
                    tcp_jacobian = jacobians[
                        0,
                        # This PhysX backend keeps the root row for a
                        # floating articulation, but omits it for a fixed
                        # one.  Match IsaacLab's own task-space-action
                        # convention: only fixed-base body ids are shifted.
                        jacobian_body - 1 if robot.is_fixed_base else jacobian_body,
                        # PhysX spatial Jacobians are ordered [angular;
                        # linear].  The trace reports TCP *linear* position
                        # sensitivity, therefore use rows 3:6 here.
                        3:6,
                        :,
                    ]
                else:
                    from isaaclab.utils.math import quat_apply

                    wrist_index = racket_command._wrist_body_index
                    wrist_jacobian = jacobians[
                        0,
                        wrist_index - 1 if robot.is_fixed_base else wrist_index,
                        :,
                        :,
                    ]
                    offset_w = quat_apply(
                        robot.data.body_quat_w[0, wrist_index],
                        racket_command._mount_offset[0],
                    )
                    skew = torch.zeros((3, 3), dtype=torch.float32, device=raw_env.device)
                    skew[0, 1], skew[0, 2] = -offset_w[2], offset_w[1]
                    skew[1, 0], skew[1, 2] = offset_w[2], -offset_w[0]
                    skew[2, 0], skew[2, 1] = -offset_w[1], offset_w[0]
                    # PhysX spatial Jacobians are [angular; linear].  For a
                    # TCP at offset r from the wrist, v_tcp = v_wrist +
                    # omega x r = J_linear qdot - [r]_x J_angular qdot.
                    tcp_jacobian = wrist_jacobian[3:, :] - skew @ wrist_jacobian[:3, :]
                # Floating-base PhysX Jacobians prepend the six generalized
                # base coordinates.  Articulation joint ids address only the
                # robot's actuated joints, so derive (rather than hard-code)
                # the required column offset from the live tensor contract.
                jacobian_joint_offset = tcp_jacobian.shape[-1] - robot.num_joints
                if jacobian_joint_offset < 0:
                    raise RuntimeError(
                        "PhysX Jacobian has fewer columns than articulation joints: "
                        f"{tcp_jacobian.shape[-1]} < {robot.num_joints}"
                    )
                upper_jacobian_ids = upper_ids + jacobian_joint_offset
                upper_tcp_jacobian = tcp_jacobian[:, upper_jacobian_ids]
                upper_safe_reference_tracking_delta = (
                    action_term.upper_reference_actions[0] - upper_actual
                )
                upper_processed_command_tracking_delta = command - upper_actual
                upper_safe_reference_tcp_contribution = (
                    upper_tcp_jacobian * upper_safe_reference_tracking_delta.unsqueeze(0)
                )
                upper_processed_command_tcp_contribution = (
                    upper_tcp_jacobian * upper_processed_command_tracking_delta.unsqueeze(0)
                )

                snapshot["upper_action_chain"] = {
                    "execution_mode": str(
                        getattr(raw_env, "p4c_upper_execution_mode", "policy")
                    ),
                    "joint_names": upper_names,
                    "reference_lookahead_steps": [
                        float(value) for value in action_term._upper_lead.tolist()
                    ],
                    "articulation_upper_joint_ids": [
                        int(index) for index in upper_ids.detach().cpu().tolist()
                    ],
                    "physx_jacobian_joint_column_offset": int(jacobian_joint_offset),
                    "physx_upper_jacobian_column_ids": [
                        int(index) for index in upper_jacobian_ids.detach().cpu().tolist()
                    ],
                    # Kept for one-off P4D mapping validation.  It makes a
                    # backend joint-order mismatch observable instead of
                    # silently assigning a wrong Jacobian column to a joint.
                    "physx_tcp_linear_jacobian_all_columns_xyz_m_per_rad": [
                        [float(value) for value in tcp_jacobian[:, index].tolist()]
                        for index in range(tcp_jacobian.shape[1])
                    ],
                    "physx_jacobian_shape": list(jacobians.shape),
                    "physx_jacobian_body_index": int(
                        racket_command._racket_body_index
                        if racket_command._racket_mode == "body"
                        else racket_command._wrist_body_index
                    ),
                    "physx_jacobian_body_row": int(
                        (
                            racket_command._racket_body_index
                            if racket_command._racket_mode == "body"
                            else racket_command._wrist_body_index
                        )
                        - (1 if robot.is_fixed_base else 0)
                    ),
                    "safe_reference_position_rad": _upper_row(
                        action_term.upper_reference_actions
                    ),
                    "frozen_actor_raw": _upper_row(action_term.upper_raw_actions),
                    "frozen_actor_contribution_rad": _upper_row(
                        action_term.upper_primary_contribution
                    ),
                    "coordinator_contribution_rad": _upper_row(
                        action_term.upper_coordinator_contribution
                    ),
                    "target_adapter_contribution_rad": _upper_row(target_adapter),
                    "safety_override_rad": _upper_row(
                        action_term.upper_safety_override
                    ),
                    "velocity_safety_override_radps": _upper_row(
                        action_term.upper_velocity_safety_override
                    ),
                    "dynamic_safety_override_rad": _upper_row(
                        getattr(action_term, "upper_dynamic_safety_override", zeros)
                    ),
                    "dynamic_velocity_safety_override_radps": _upper_row(
                        getattr(
                            action_term,
                            "upper_dynamic_velocity_safety_override",
                            zeros,
                        )
                    ),
                    "processed_command_position_rad": [
                        float(item) for item in command.tolist()
                    ],
                    "processed_command_velocity_radps": [
                        float(item) for item in command_velocity.tolist()
                    ],
                    "processed_command_soft_margin_rad": [
                        float(item) for item in command_soft_margin.tolist()
                    ],
                    "actual_position_rad": [
                        float(item) for item in upper_actual.tolist()
                    ],
                    "actual_velocity_radps": [
                        float(item) for item in upper_actual_velocity.tolist()
                    ],
                    "actual_soft_margin_rad": [
                        float(item) for item in upper_soft_margin.tolist()
                    ],
                    "actual_hard_margin_rad": [
                        float(item) for item in upper_hard_margin.tolist()
                    ],
                    "tcp_linear_jacobian_xyz_m_per_rad": [
                        [float(value) for value in upper_tcp_jacobian[:, index].tolist()]
                        for index in range(upper_tcp_jacobian.shape[1])
                    ],
                    "linearized_safe_reference_minus_actual_tcp_xyz_m": [
                        [
                            float(value)
                            for value in upper_safe_reference_tcp_contribution[:, index].tolist()
                        ]
                        for index in range(upper_safe_reference_tcp_contribution.shape[1])
                    ],
                    "linearized_processed_command_minus_actual_tcp_xyz_m": [
                        [
                            float(value)
                            for value in upper_processed_command_tcp_contribution[:, index].tolist()
                        ]
                        for index in range(upper_processed_command_tcp_contribution.shape[1])
                    ],
                }
                # The prior-guided P5D action owns a frozen 3396 support path
                # in addition to the shared upper-chain fields above.  Record
                # it explicitly so a stable/unstable rollout can never be
                # misattributed to a missing lower-body prior.
                if hasattr(action_term, "_legacy_raw"):
                    base_names = list(action_term.cfg.base_joint_names)
                    base_ids = action_term._base_joint_ids_tensor
                    snapshot["frozen_stage_a_support_chain"] = {
                        "checkpoint_role": "frozen_model_3396_leg_support",
                        "base_joint_names": base_names,
                        "stage_a_raw_action": _upper_row(action_term._legacy_raw),
                        "stage_a_masked_bounded_action": _upper_row(
                            action_term._legacy_bounded
                        ),
                        "stage_a_action_mask": [
                            float(value) for value in action_term._mask[0].tolist()
                        ],
                        "processed_base_target_position_rad": [
                            float(value)
                            for value in action_term.full_joint_targets[0, base_ids].tolist()
                        ],
                    }
            # P5D deliberately does not use the historical upper-body
            # composite action.  Keep a separate chain so a tracker residual
            # (the learned control contribution) can never be mistaken for a
            # safety projection.  This is evidence only; it does not alter
            # the command reaching PhysX.
            if hasattr(action_term, "safe_reference_actions"):
                tracker_names = list(action_term._joint_names)
                tracker_ids = action_term._joint_index_tensor
                safe_reference = action_term.safe_reference_actions[0]
                processed = action_term._processed_actions[0]
                tracker_actual = actual_q[tracker_ids]
                tracker_actual_velocity = robot.data.joint_vel[0, tracker_ids]
                safety_override = action_term.safety_override[0]
                # ``processed - safe_reference`` contains both the PPO
                # residual and any projection.  The action term records the
                # latter explicitly, so the effective learned contribution is
                # observable even at a joint limit.
                effective_residual = processed - safe_reference - safety_override
                tracker_limits = soft_limits[tracker_ids]
                processed_soft_margin = torch.minimum(
                    processed - tracker_limits[:, 0],
                    tracker_limits[:, 1] - processed,
                )

                jacobians = robot.root_physx_view.get_jacobians()
                if racket_command._racket_mode == "body":
                    jacobian_body = racket_command._racket_body_index
                    tcp_jacobian = jacobians[
                        0,
                        jacobian_body - 1 if robot.is_fixed_base else jacobian_body,
                        3:6,
                        :,
                    ]
                else:
                    from isaaclab.utils.math import quat_apply

                    wrist_index = racket_command._wrist_body_index
                    wrist_jacobian = jacobians[
                        0,
                        wrist_index - 1 if robot.is_fixed_base else wrist_index,
                        :,
                        :,
                    ]
                    offset_w = quat_apply(
                        robot.data.body_quat_w[0, wrist_index],
                        racket_command._mount_offset[0],
                    )
                    skew = torch.zeros((3, 3), dtype=torch.float32, device=raw_env.device)
                    skew[0, 1], skew[0, 2] = -offset_w[2], offset_w[1]
                    skew[1, 0], skew[1, 2] = offset_w[2], -offset_w[0]
                    skew[2, 0], skew[2, 1] = -offset_w[1], offset_w[0]
                    tcp_jacobian = wrist_jacobian[3:, :] - skew @ wrist_jacobian[:3, :]
                jacobian_joint_offset = tcp_jacobian.shape[-1] - robot.num_joints
                if jacobian_joint_offset < 0:
                    raise RuntimeError(
                        "PhysX Jacobian has fewer columns than articulation joints: "
                        f"{tcp_jacobian.shape[-1]} < {robot.num_joints}"
                    )
                tracker_jacobian_ids = tracker_ids + jacobian_joint_offset
                tracker_tcp_jacobian = tcp_jacobian[:, tracker_jacobian_ids]

                def _tracker_row(value):
                    return [float(item) for item in value.tolist()]

                snapshot["reference_tracker_action_chain"] = {
                    "joint_names": tracker_names,
                    "articulation_joint_ids": [
                        int(index) for index in tracker_ids.detach().cpu().tolist()
                    ],
                    "physx_jacobian_joint_column_offset": int(jacobian_joint_offset),
                    "physx_jacobian_joint_column_ids": [
                        int(index) for index in tracker_jacobian_ids.detach().cpu().tolist()
                    ],
                    "safe_reference_position_rad": _tracker_row(safe_reference),
                    "processed_command_position_rad": _tracker_row(processed),
                    "actual_position_rad": _tracker_row(tracker_actual),
                    "actual_velocity_radps": _tracker_row(tracker_actual_velocity),
                    "effective_tracker_residual_rad": _tracker_row(effective_residual),
                    "safety_override_rad": _tracker_row(safety_override),
                    "processed_command_soft_margin_rad": _tracker_row(processed_soft_margin),
                    "actual_soft_margin_rad": _tracker_row(actual_soft_margin[tracker_ids]),
                    "actual_hard_margin_rad": _tracker_row(actual_hard_margin[tracker_ids]),
                    "tcp_linear_jacobian_xyz_m_per_rad": [
                        [float(value) for value in tracker_tcp_jacobian[:, index].tolist()]
                        for index in range(tracker_tcp_jacobian.shape[1])
                    ],
                    "linearized_safe_reference_minus_actual_tcp_xyz_m": [
                        [
                            float(value)
                            for value in (
                                tracker_tcp_jacobian[:, index]
                                * (safe_reference[index] - tracker_actual[index])
                            ).tolist()
                        ]
                        for index in range(tracker_tcp_jacobian.shape[1])
                    ],
                    "linearized_processed_command_minus_actual_tcp_xyz_m": [
                        [
                            float(value)
                            for value in (
                                tracker_tcp_jacobian[:, index]
                                * (processed[index] - tracker_actual[index])
                            ).tolist()
                        ]
                        for index in range(tracker_tcp_jacobian.shape[1])
                    ],
                }
        except Exception:
            # Non-motion tasks still benefit from the base stability snapshot.
            pass
        return snapshot

    def _fall_termination_audit(pre_step_state, termination_reasons, truncated):
        """Classify a terminal event from the last pre-reset unified state.

        The post-step vector environment may already have reset the robot, so
        this helper intentionally consumes the pre-step snapshot.  It keeps
        physical fall, prediction, recovery timeout and ordinary timeout as
        separate labels instead of treating every ``done`` as a fall.
        """
        fall = (pre_step_state or {}).get("fall_state", {})
        cycle_phase = int(fall.get("cycle_phase", 0))
        confirmed = bool(fall.get("confirmed_fall", False))
        predicted = bool(fall.get("predicted_unrecoverable", False))
        if confirmed or "strict_fall" in termination_reasons:
            label = "confirmed_fall"
        elif predicted:
            label = "predicted_unrecoverable"
        elif truncated and cycle_phase == 5:  # RECOVERY_MONITOR
            label = "recovery_timeout"
        elif truncated:
            label = "ordinary_timeout"
        else:
            label = "termination"
        return {
            "label": label,
            "confirmed_fall": confirmed,
            "predicted_unrecoverable": predicted,
            "risk_level": fall.get("risk_level_name"),
            "fall_reason": fall.get("fall_reason_name"),
            "cycle_phase": cycle_phase,
            "termination_reasons": list(termination_reasons),
        }

    if multi_shot_sequence is not None:
        report_path = cfg.get("multi_shot_report", None)
        if report_path is None:
            report_path = os.path.join(
                os.path.dirname(resume_path), "multi_shot_report.json"
            )
        multi_shot_report_path = pathlib.Path(str(report_path)).expanduser()
        if not multi_shot_report_path.is_absolute():
            multi_shot_report_path = pathlib.Path.cwd() / multi_shot_report_path
        multi_shot_report_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_multi_shot_report(progress_status: str):
        """Atomically persist partial multi-shot evidence after every key event."""
        if multi_shot_report_path is None:
            return
        report = {
            "checkpoint": str(resume_path),
            "task": str(cfg.task.name),
            "sequence": multi_shot_sequence,
            "complete": multi_shot_complete,
            "failure": multi_shot_failure,
            "control_steps": timestep,
            "progress_status": progress_status,
            "shots": multi_shot_records,
            "state_transitions": multi_shot_transitions,
            # Keep the final 100 pre-reset states so a delayed fall cannot be
            # hidden by vector-env auto-reset. This is intentionally bounded
            # to keep partial reports small during long audits.
            "pre_reset_trace_last_100": multi_shot_trace[-100:],
            "state_contract": {
                "0": "SUPPORT_ACTIVE",
                "1": "DECAYING",
                "2": "SETTLED",
                "3": "READY",
                "4": "RAMPING",
            },
        }
        temporary_path = multi_shot_report_path.with_suffix(
            multi_shot_report_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_path.replace(multi_shot_report_path)

    def _update_recovery_milestones(control_step: int, snapshot: dict):
        """Record the first post-hit step satisfying each re-arm prerequisite."""
        if not multi_shot_hit_recorded or not multi_shot_records:
            return False
        shot_index = multi_shot_records[-1]["shot_index"]
        milestones = multi_shot_recovery_milestones.setdefault(
            shot_index,
            {"hit": multi_shot_records[-1]["hit_control_step"]},
        )
        conditions = {
            "arm_pose_ready": snapshot.get("right_arm_ready_error_max_rad", float("inf")) <= 0.15,
            "arm_velocity_ready": snapshot.get("right_arm_velocity_max_radps", float("inf")) <= 0.15,
            "capture_centered": abs(snapshot["capture_rel_support_x_m"]) <= 0.05,
            "root_velocity_quiet": abs(snapshot["root_forward_velocity_mps"]) <= 0.06,
            "pitch_rate_quiet": abs(snapshot["root_pitch_rate_radps"]) <= 0.10,
            "tilt_quiet": snapshot["root_tilt_rad"] <= 0.10,
            "both_feet_contact": snapshot["both_feet_contact"],
            "leg_velocity_quiet": snapshot.get("leg_velocity_max_radps", float("inf")) <= 0.15,
        }
        updated = False
        for name, condition in conditions.items():
            if condition and name not in milestones:
                milestones[name] = control_step
                updated = True
        multi_shot_records[-1]["recovery_milestones"] = dict(milestones)
        return updated
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

    def _capture_strike_goal_shadow(control_step: int, control_time_s: float):
        if strike_goal_shadow_pipeline is None:
            return None
        raw_env = env.unwrapped
        link_pos_w, link_vel_w, _, link_ang_vel_w = racket_spatial_state_w(raw_env)
        face_normal_w = racket_normal_w(raw_env, normal_axis=1, normal_sign=1.0)
        rotation = strike_goal_shadow_context["sim_to_receipt_rotation"]
        translation = strike_goal_shadow_context["sim_to_receipt_translation"]

        def _point_to_receipt(value):
            return torch.mv(rotation, value[0]) + translation

        def _vector_to_receipt(value):
            return torch.mv(rotation, value[0])

        actual = RacketFaceState.from_link_state(
            link_origin_position=_point_to_receipt(link_pos_w).detach().cpu().tolist(),
            link_origin_linear_velocity=_vector_to_receipt(link_vel_w).detach().cpu().tolist(),
            link_angular_velocity=_vector_to_receipt(link_ang_vel_w).detach().cpu().tolist(),
            face_normal=_vector_to_receipt(face_normal_w).detach().cpu().tolist(),
            frame_id=BASE_HEADING_RECEIPT_FRAME_V1,
            calibration=strike_goal_shadow_pipeline.contact_calibration,
        )
        return strike_goal_shadow_pipeline.capture(
            control_step=control_step,
            current_control_time_s=control_time_s,
            actual=actual,
        )

    timestep = 0
    with torch.inference_mode():
        initial_shadow_sample = _capture_strike_goal_shadow(0, 0.0)
    if initial_shadow_sample is not None:
        print(
            "[INFO] strike-goal shadow initial target: "
            f"ball_b={list(initial_shadow_sample.target.ball_center_position)}, "
            f"face_b={list(initial_shadow_sample.target.face_contact_position)}, "
            f"link_b={list(initial_shadow_sample.target.link_origin_position)}",
            flush=True,
        )
    # Kit's ``is_running()`` may become false after its internal frame budget
    # in headless mode, even though the vector environment can still advance.
    # A multi-shot audit has its own explicit control-step budget and terminal
    # contract, so keep it alive until one of those conditions is reached.
    # Single-shot interactive/video playback retains the normal Kit lifetime.
    while simulation_app.is_running() or multi_shot_sequence is not None:
        with torch.inference_mode():
            pre_step_video_frame = None
            if (
                cfg.video
                and video_stop_on_termination
                and video_capture_pre_step_on_termination
            ):
                # Render before env.step().  IsaacLab may reset a terminated
                # vector row inside env.step(), so a post-step render would
                # show READY rather than the terminal physical state.
                pre_step_video_frame = env.unwrapped.render()
                if pre_step_video_frame is not None:
                    pre_step_video_frame = _overlay_fall_audit(
                        pre_step_video_frame, env.unwrapped
                    )
            # The unified A3 fall contract is only valid for motion tasks with
            # the configured torso/foot/contact scene.  Do not make generic
            # play.py tasks fail merely because they do not expose a motion
            # command; A3 replay and multi-shot audits remain fail-closed.
            pre_fall_state = None
            if has_motion_command:
                from training.tasks.tracking.mdp.fall_state import unified_fall_state
                pre_fall_state = unified_fall_state(env.unwrapped)
                unified_predicted_unrecoverable_steps += int(pre_fall_state.predicted_unrecoverable.sum().item())
                unified_confirmed_fall_steps += int(pre_fall_state.confirmed_fall.sum().item())
                unified_recovery_ready_steps += int(pre_fall_state.recovery_ready.sum().item())
            single_shot_pre_state = None
            if record_trace and multi_shot_sequence is None:
                single_shot_pre_state = _multi_shot_state_snapshot(env.unwrapped)
            if multi_shot_sequence is not None:
                multi_shot_last_pre_step_state = _multi_shot_state_snapshot(
                    env.unwrapped
                )
                multi_shot_trace.append(
                    {
                        "control_step": timestep,
                        "state": multi_shot_last_pre_step_state,
                    }
                )
                if multi_shot_start_state is None:
                    multi_shot_start_state = dict(multi_shot_last_pre_step_state)
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
            policy_observation = obs.detach().clone()
            actions = policy(obs)
            if _as_bool(cfg.get("force_zero_action", False)):
                actions = torch.zeros_like(actions)
            # P5D baseline: replay the exact same safe-reference trajectory
            # with a zero tracker residual.  This is not a policy result; it
            # isolates reference -> safety -> actual before attributing any
            # improvement to PPO.  The flag is intentionally rejected for
            # other tasks so it cannot silently alter historical evaluations.
            if _as_bool(cfg.get("zero_tracker_residual", False)):
                if task_id not in {
                    "HOPE-FloatingReferenceTracker-AgibotA3-v0",
                    "HOPE-FloatingPriorGuidedReferenceTracker-AgibotA3-v0",
                }:
                    raise ValueError(
                        "zero_tracker_residual is only valid for a P5D reference-tracker task"
                    )
                actions = torch.zeros_like(actions)
            recovery_policy_diagnostics = None
            if target_conditioned_recovery_task:
                recovery_policy = ppo_runner.alg.policy
                normalized_recovery_obs = ppo_runner.obs_normalizer(
                    policy_observation
                )
                recovery_base_action = recovery_policy.base_action_mean(
                    normalized_recovery_obs[..., : recovery_policy.BASE_OBS_DIM]
                )
                recovery_delta = actions - recovery_base_action
                recovery_policy_diagnostics = {
                    "gate": float(policy_observation[0, -1].item()),
                    "action_delta": [
                        float(value) for value in recovery_delta[0].tolist()
                    ],
                    "action_delta_linf": float(recovery_delta[0].abs().max().item()),
                }
                if p4_recovery_action_offset is not None:
                    # Evaluation-only authority probe.  The fixed raw offset
                    # uses precisely the same predictive gate as P4's learned
                    # lower-body branch; it is never available during training
                    # or exported for deployment.  This separates missing
                    # control authority from an under-trained recovery head.
                    evaluation_offset = p4_recovery_action_offset.to(
                        device=actions.device, dtype=actions.dtype
                    ).unsqueeze(0).expand_as(actions)
                    gate = policy_observation[..., -1:].clamp(0.0, 1.0)
                    actions = actions + gate * evaluation_offset
                    recovery_policy_diagnostics["evaluation_offset_linf"] = float(
                        (gate * evaluation_offset)[0].abs().max().item()
                    )
            if actions.shape[-1] == 7:
                actions = actions * adapter_policy_scale
            if adapter_jacobian_enabled:
                offsets = torch.zeros_like(actions)
                if actions.shape[-1] == 7:
                    # P0 exposes the adapter directly as the public action.
                    for joint in range(7):
                        offsets[1 + 2 * joint, joint] = -adapter_jacobian_step
                        offsets[2 + 2 * joint, joint] = adapter_jacobian_step
                    actions = actions + offsets
                else:
                    # Floating coordinator replay keeps its public 22-D
                    # checkpoint action.  Inject the same paired perturbation
                    # through the internal seven-joint adapter hook.
                    adapter_state = getattr(env.unwrapped, "target_adapter_last_action", None)
                    if adapter_state is None or adapter_state.shape[-1] != 7:
                        raise ValueError(
                            "adapter_jacobian_step requires either a 7-D public "
                            "adapter action or an internal seven-joint target adapter"
                        )
                    internal_offsets = torch.zeros_like(adapter_state)
                    for joint in range(7):
                        internal_offsets[1 + 2 * joint, joint] = -adapter_jacobian_step
                        internal_offsets[2 + 2 * joint, joint] = adapter_jacobian_step
                    env.unwrapped.target_adapter_override = internal_offsets
            if coordinator_jacobian_enabled:
                if actions.shape[-1] != 22:
                    raise ValueError(
                        "coordinator_jacobian_step requires a 22-D public coordinator action"
                    )
                coordinator_offsets = torch.zeros_like(actions)
                for joint in range(22):
                    coordinator_offsets[1 + 2 * joint, joint] = -coordinator_jacobian_step
                    coordinator_offsets[2 + 2 * joint, joint] = coordinator_jacobian_step
                env.unwrapped.coordinator_action_offset_override = coordinator_offsets
            if strike_goal_shadow_pipeline is not None:
                strike_goal_shadow_context["policy_action_trace"].append(
                    actions[0].detach().cpu().tolist()
                )
            obs, _, terminated, truncated = env.step(actions.to(env.unwrapped.device))
            obs = _obs_to_device(obs, agent_cfg.device)
            raw_env = env.unwrapped
            _capture_strike_goal_shadow(
                timestep + 1,
                (timestep + 1) * float(raw_env.step_dt),
            )
            done_tensor = torch.as_tensor(
                terminated, device=raw_env.device, dtype=torch.bool
            )
            if torch.is_tensor(truncated):
                timeout_tensor = torch.as_tensor(
                    truncated, device=raw_env.device, dtype=torch.bool
                )
                termination_log = {}
            else:
                time_outs = truncated.get("time_outs", None)
                timeout_tensor = (
                    torch.as_tensor(
                        time_outs, device=raw_env.device, dtype=torch.bool
                    )
                    if time_outs is not None
                    else torch.zeros_like(done_tensor)
                )
                termination_log = dict(truncated.get("log", {}) or {})
            terminated_only_tensor = done_tensor & (~timeout_tensor)
            termination_reasons = sorted(
                key.removeprefix("Episode_Termination/")
                for key, value in termination_log.items()
                if key.startswith("Episode_Termination/")
                and float(torch.as_tensor(value).sum().item()) > 0.0
            )
            # The environment resets terminated rows immediately after step().
            # Preserve the terminal reason in the audit counters before reset
            # wipes the term's persistence state.
            if "strict_fall" in termination_reasons:
                strict_fall_candidate_steps += 1
                if strict_fall_first_candidate_step is None:
                    strict_fall_first_candidate_step = timestep + 1
                torso_fall_candidate_steps += 1
                if torso_fall_first_candidate_step is None:
                    torso_fall_first_candidate_step = timestep + 1
            if target_audit_enabled:
                newly_terminated = torch.where(
                    terminated_only_tensor
                    & (target_audit_first_physical_termination_step < 0)
                )[0]
                if newly_terminated.numel() > 0:
                    target_audit_first_physical_termination_step[
                        newly_terminated
                    ] = timestep + 1
                    for env_id in newly_terminated.detach().cpu().tolist():
                        target_audit_first_physical_termination_reasons[env_id] = list(
                            termination_reasons
                        )
                actor_snapshot = _target_actor_snapshot(
                    raw_env, actions, policy_observation
                )
                if target_audit_initial_actor is None:
                    target_audit_initial_actor = {
                        name: None if value is None else value.detach().clone()
                        for name, value in actor_snapshot.items()
                    }
                motion_cmd = raw_env.command_manager.get_term("motion")
                racket_cmd = raw_env.command_manager.get_term("racket_target")
                in_swing = (
                    motion_cmd.prelude_elapsed_steps
                    >= int(motion_cmd.prelude_steps)
                )
                exact_mask = in_swing & (
                    racket_cmd.metrics["exact_strike_hit_rate"] > 0.5
                )
                new_hit_ids = [
                    int(env_id)
                    for env_id in torch.where(exact_mask)[0].detach().cpu().tolist()
                    if target_audit_hit_records[int(env_id)] is None
                ]
                if new_hit_ids:
                    heading = target_audit_context["command_base_heading_w"]
                    base_position = target_audit_context["command_base_position_w"]
                    actual_position_b = quat_rotate_inverse(
                        heading, racket_cmd.racket_pos_w - base_position
                    )
                    actual_velocity_b = quat_rotate_inverse(
                        heading, racket_cmd.racket_lin_vel_w
                    )
                    actual_normal_b = quat_rotate_inverse(
                        heading, racket_cmd.racket_normal_w
                    )
                    hit_root_position_w = raw_env.scene[
                        "robot"
                    ].data.root_pos_w
                    root_displacement_b = quat_rotate_inverse(
                        heading, hit_root_position_w - base_position
                    )
                    racket_relative_root_b = quat_rotate_inverse(
                        heading, racket_cmd.racket_pos_w - hit_root_position_w
                    )

                    def _row(tensor, env_id):
                        return [
                            float(value)
                            for value in tensor[env_id].detach().cpu().tolist()
                        ]

                    for env_id in new_hit_ids:
                        requested_offset = target_audit_context[
                            "requested_offset_b"
                        ][env_id]
                        actual_offset = (
                            actual_position_b[env_id]
                            - target_audit_context["anchor_position_b"][env_id]
                        )
                        target_audit_hit_records[env_id] = {
                            "trial_id": env_id,
                            "control_step": timestep + 1,
                            "motion_id": int(motion_cmd.motion_ids[env_id].item()),
                            "anchor_position_b_m": _row(
                                target_audit_context["anchor_position_b"], env_id
                            ),
                            "anchor_position_w_m": _row(
                                target_audit_context["anchor_position_w"], env_id
                            ),
                            "requested_offset_b_m": _row(
                                target_audit_context["requested_offset_b"], env_id
                            ),
                            "control_offset_from_calibrated_anchor_b_m": _row(
                                raw_env.target_adapter_control_delta_b, env_id
                            )
                            if hasattr(raw_env, "target_adapter_control_delta_b")
                            else _row(
                                target_audit_context["requested_offset_b"], env_id
                            ),
                            "target_position_b_m": _row(
                                target_audit_context["target_position_b"], env_id
                            ),
                            "target_position_w_m": _row(
                                target_audit_context["target_position_w"], env_id
                            ),
                            "actual_position_b_m": _row(actual_position_b, env_id),
                            "actual_position_w_m": _row(
                                racket_cmd.racket_pos_w, env_id
                            ),
                            "root_displacement_b_m": _row(
                                root_displacement_b, env_id
                            ),
                            "racket_relative_root_b_m": _row(
                                racket_relative_root_b, env_id
                            ),
                            "actual_offset_from_anchor_b_m": _row(
                                actual_offset.unsqueeze(0), 0
                            ),
                            "anchor_referenced_offset_error_b_m": _row(
                                (actual_offset - requested_offset).unsqueeze(0), 0
                            ),
                            "target_velocity_b_mps": _row(
                                target_audit_context["target_velocity_b"], env_id
                            ),
                            "actual_velocity_b_mps": _row(
                                actual_velocity_b, env_id
                            ),
                            "target_normal_b": _row(
                                target_audit_context["target_normal_b"], env_id
                            ),
                            "actual_normal_b": _row(actual_normal_b, env_id),
                            "position_error_m": float(
                                racket_cmd.metrics[
                                    "racket_pos_error_exact_strike"
                                ][env_id].item()
                            ),
                            "velocity_error_mps": float(
                                racket_cmd.metrics[
                                    "racket_vel_error_exact_strike"
                                ][env_id].item()
                            ),
                            "normal_error_deg": float(
                                racket_cmd.metrics[
                                    "racket_normal_error_deg_exact_strike"
                                ][env_id].item()
                            ),
                            "initial_upper_observation": _row(
                                target_audit_initial_actor["upper_observation"],
                                env_id,
                            ),
                            "initial_upper_observation_normalized": _row(
                                target_audit_initial_actor[
                                    "upper_observation_normalized"
                                ],
                                env_id,
                            ),
                            "initial_upper_actor_output": _row(
                                target_audit_initial_actor["upper_actor_output"],
                                env_id,
                            ),
                            "initial_coordinator_action": _row(
                                target_audit_initial_actor["coordinator_action"],
                                env_id,
                            ),
                            "hit_upper_observation": _row(
                                actor_snapshot["upper_observation"], env_id
                            ),
                            "hit_upper_observation_normalized": _row(
                                actor_snapshot["upper_observation_normalized"],
                                env_id,
                            ),
                            "hit_upper_actor_output": _row(
                                actor_snapshot["upper_actor_output"], env_id
                            ),
                            "hit_coordinator_action": _row(
                                actor_snapshot["coordinator_action"], env_id
                            ),
                            "hit_coordinator_target_feedforward": (
                                _row(
                                    actor_snapshot["coordinator_target_feedforward"],
                                    env_id,
                                )
                                if actor_snapshot["coordinator_target_feedforward"] is not None
                                else None
                            ),
                            "terminated_at_hit": bool(
                                terminated_only_tensor[env_id].item()
                            ),
                            "truncated_at_hit": bool(
                                timeout_tensor[env_id].item()
                            ),
                        }
                if (
                    target_audit_all_hit_control_step is None
                    and all(record is not None for record in target_audit_hit_records)
                ):
                    target_audit_all_hit_control_step = timestep + 1
            if record_trace and multi_shot_sequence is None:
                done_now = bool(done_tensor[0].item())
                motion_cmd = raw_env.command_manager.get_term("motion")
                racket_cmd = raw_env.command_manager.get_term("racket_target")
                in_swing = (
                    int(motion_cmd.prelude_elapsed_steps[0].item())
                    >= int(motion_cmd.prelude_steps)
                )
                exact_hit = (
                    in_swing
                    and float(
                        racket_cmd.metrics["exact_strike_hit_rate"][0].item()
                    )
                    > 0.5
                )
                if exact_hit and single_shot_hit_record is None:
                    single_shot_hit_record = {
                        "control_step": timestep + 1,
                        "motion_id": int(motion_cmd.motion_ids[0].item()),
                        "position_error_m": float(
                            racket_cmd.metrics[
                                "racket_pos_error_exact_strike"
                            ][0].item()
                        ),
                        "velocity_error_mps": float(
                            racket_cmd.metrics[
                                "racket_vel_error_exact_strike"
                            ][0].item()
                        ),
                        "normal_error_deg": float(
                            racket_cmd.metrics[
                                "racket_normal_error_deg_exact_strike"
                            ][0].item()
                        ),
                        "state": _multi_shot_state_snapshot(raw_env),
                    }
                trace_record = {
                    "control_step": timestep + 1,
                    "pre_step_state": single_shot_pre_state,
                    "done": done_now,
                    "terminated": bool(terminated_only_tensor[0].item()),
                    "truncated": bool(timeout_tensor[0].item()),
                    "termination_reasons": termination_reasons,
                }
                if recovery_policy_diagnostics is not None:
                    trace_record["recovery_policy"] = recovery_policy_diagnostics
                if not done_now:
                    trace_record["post_step_state"] = _multi_shot_state_snapshot(
                        raw_env
                    )
                single_shot_trace.append(trace_record)
                if done_now and single_shot_termination is None:
                    single_shot_termination = {
                        "control_step": timestep + 1,
                        "terminated": bool(terminated_only_tensor[0].item()),
                        "truncated": bool(timeout_tensor[0].item()),
                        "termination_reasons": termination_reasons,
                        "fall_audit": _fall_termination_audit(
                            single_shot_pre_state, termination_reasons,
                            bool(timeout_tensor[0].item()),
                        ),
                        "last_pre_step_state": single_shot_pre_state,
                    }
            if multi_shot_sequence is not None:
                motion_cmd = raw_env.command_manager.get_term("motion")
                racket_cmd = raw_env.command_manager.get_term("racket_target")
                state = int(raw_env.stage_a_sagittal_exit_state[0].item())
                scale = float(raw_env.stage_a_sagittal_exit_scale[0].item())
                if state != multi_shot_last_state:
                    multi_shot_transitions.append(
                        {
                            "control_step": timestep + 1,
                            "shot_index": multi_shot_index,
                            "motion_id": int(motion_cmd.motion_ids[0].item()),
                            "state": state,
                            "scale": scale,
                        }
                    )
                    multi_shot_last_state = state
                    _write_multi_shot_report("state_transition")

                in_swing = (
                    int(motion_cmd.prelude_elapsed_steps[0].item())
                    >= int(motion_cmd.prelude_steps)
                )
                exact_hit = (
                    in_swing
                    and float(
                        racket_cmd.metrics["exact_strike_hit_rate"][0].item()
                    )
                    > 0.5
                )
                if exact_hit and not multi_shot_hit_recorded:
                    multi_shot_records.append(
                        {
                            "shot_index": multi_shot_index,
                            "motion_id": int(motion_cmd.motion_ids[0].item()),
                            "hit_control_step": timestep + 1,
                            "position_error_m": float(
                                racket_cmd.metrics[
                                    "racket_pos_error_exact_strike"
                                ][0].item()
                            ),
                            "velocity_error_mps": float(
                                racket_cmd.metrics[
                                    "racket_vel_error_exact_strike"
                                ][0].item()
                            ),
                            "normal_error_deg": float(
                                racket_cmd.metrics[
                                    "racket_normal_error_deg_exact_strike"
                                ][0].item()
                            ),
                            "start_state": multi_shot_start_state,
                            "hit_state": _multi_shot_state_snapshot(raw_env),
                        }
                    )
                    multi_shot_hit_recorded = True
                    _update_recovery_milestones(
                        timestep + 1, multi_shot_records[-1]["hit_state"]
                    )
                    _write_multi_shot_report("exact_hit")

                if multi_shot_hit_recorded:
                    recovery_snapshot = _multi_shot_state_snapshot(raw_env)
                    if _update_recovery_milestones(timestep + 1, recovery_snapshot):
                        _write_multi_shot_report("recovery_milestone")

                done_now = bool(done_tensor[0].item())
                truncated_now = bool(timeout_tensor[0].item())
                terminated_now = bool(terminated_only_tensor[0].item())
                if done_now:
                    multi_shot_failure = {
                        "control_step": timestep + 1,
                        "shot_index": multi_shot_index,
                        "terminated": terminated_now,
                        "truncated": truncated_now,
                        "termination_reasons": termination_reasons,
                        "fall_audit": _fall_termination_audit(
                            multi_shot_last_pre_step_state, termination_reasons,
                            truncated_now,
                        ),
                        "last_pre_step_state": multi_shot_last_pre_step_state,
                    }
                    _write_multi_shot_report("physical_termination")
                elif bool(raw_env.stage_a_sagittal_rearm_rejected[0].item()):
                    multi_shot_failure = {
                        "control_step": timestep + 1,
                        "shot_index": multi_shot_index,
                        "reason": "rearm_rejected_unstable",
                    }
                    _write_multi_shot_report("rearm_rejected")
                elif bool(raw_env.stage_a_sagittal_rearm_ready[0].item()):
                    # The legacy Stage-A rearm flag is only a coordinator
                    # signal.  A new strike is admitted only after the
                    # unified physical recovery gate has held continuously;
                    # this prevents a fixed-time rearm from hiding a delayed
                    # torso/root fall.
                    from training.tasks.tracking.mdp.fall_state import unified_fall_state
                    recovery_state = unified_fall_state(raw_env)
                    next_action_allowed = motion_cmd.can_begin_next_shot([0])
                    if not bool(next_action_allowed[0].item()):
                        multi_shot_failure = {
                            "control_step": timestep + 1,
                            "shot_index": multi_shot_index,
                            "reason": "unified_recovery_not_ready",
                            "recovery_progress": float(recovery_state.recovery_progress[0].item()),
                            "risk_score": float(recovery_state.risk_score[0].item()),
                            "confirmed_fall": bool(recovery_state.confirmed_fall[0].item()),
                            "predicted_unrecoverable": bool(recovery_state.predicted_unrecoverable[0].item()),
                        }
                        _write_multi_shot_report("unified_recovery_not_ready")
                        continue
                    if not multi_shot_hit_recorded:
                        multi_shot_failure = {
                            "control_step": timestep + 1,
                            "shot_index": multi_shot_index,
                            "reason": "ready_before_exact_hit",
                        }
                        _write_multi_shot_report("ready_before_hit")
                    elif multi_shot_index + 1 >= len(multi_shot_sequence):
                        multi_shot_complete = True
                        _write_multi_shot_report("complete")
                    else:
                        multi_shot_records[-1]["ready_control_step"] = timestep + 1
                        multi_shot_records[-1]["ready_hold_steps"] = int(
                            raw_env.stage_a_sagittal_rearm_stable_steps[0].item()
                        )
                        multi_shot_records[-1]["rearm_stable_start_control_step"] = (
                            multi_shot_records[-1]["ready_control_step"]
                            - multi_shot_records[-1]["ready_hold_steps"]
                            + 1
                        )
                        multi_shot_records[-1]["ready_state"] = (
                            _multi_shot_state_snapshot(raw_env)
                        )
                        next_motion = int(
                            multi_shot_sequence[multi_shot_index + 1]
                        )
                        motion_cmd.begin_next_shot([0], [next_motion])
                        racket_cmd._resample_command([0])
                        racket_cmd._compute_strike_timing()
                        multi_shot_index += 1
                        multi_shot_hit_recorded = False
                        multi_shot_start_state = _multi_shot_state_snapshot(raw_env)
                        # Refresh the observation after changing the reference
                        # so the next coordinator action is not computed from
                        # the previous shot's READY command.
                        obs = _obs_to_device(
                            env.get_observations(), agent_cfg.device
                        )
                        print(
                            "[INFO] multi-shot re-arm accepted: "
                            f"shot={multi_shot_index}, motion_id={next_motion}, "
                            f"step={timestep + 1}",
                            flush=True,
                        )
                        _write_multi_shot_report("rearm_accepted")
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
            root_pos = robot.data.root_pos_w
            root_lin_vel = robot.data.root_lin_vel_w
            root_ang_vel = robot.data.root_ang_vel_w
            min_root_height = min(min_root_height, float(torch.min(root_pos[:, 2]).item()))
            max_root_speed = max(max_root_speed, float(torch.linalg.norm(root_lin_vel, dim=-1).max().item()))
            max_root_ang_speed = max(max_root_ang_speed, float(torch.linalg.norm(root_ang_vel, dim=-1).max().item()))
            upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], min=-1.0, max=1.0)
            tilt_rad = torch.acos(upright)
            min_base_upright = min(min_base_upright, float(upright.min().item()))
            max_root_tilt_rad = max(max_root_tilt_rad, float(tilt_rad.max().item()))
            if has_torso_probe:
                torso_pos = robot.data.body_pos_w[:, torso_body_id]
                torso_quat = robot.data.body_quat_w[:, torso_body_id]
                gravity_w = torch.zeros_like(torso_pos)
                gravity_w[:, 2] = -1.0
                from isaaclab.utils.math import quat_rotate_inverse

                torso_gravity_b = quat_rotate_inverse(torso_quat, gravity_w)
                torso_tilt_rad = torch.acos(
                    torch.clamp(-torso_gravity_b[:, 2], min=-1.0, max=1.0)
                )
                torso_height = torso_pos[:, 2]
            else:
                torso_tilt_rad = tilt_rad
                torso_height = root_pos[:, 2]
            min_torso_height = min(min_torso_height, float(torch.min(torso_height).item()))
            max_torso_tilt_rad = max(max_torso_tilt_rad, float(torch.max(torso_tilt_rad).item()))
            strict_bad = (
                (tilt_rad > strict_fall_max_tilt_rad)
                | (root_pos[:, 2] < strict_fall_min_height)
                | (torso_tilt_rad > strict_fall_max_torso_tilt_rad)
                | (torso_height < strict_fall_min_torso_height)
            )
            strict_fall_consecutive[:] = torch.where(
                strict_bad,
                strict_fall_consecutive + 1,
                torch.zeros_like(strict_fall_consecutive),
            )
            strict_now = strict_fall_consecutive >= strict_fall_required_steps
            strict_fall_candidate_steps += int(strict_now.sum().item())
            if strict_fall_first_candidate_step is None and bool(strict_now.any().item()):
                strict_fall_first_candidate_step = timestep + 1
            torso_bad = (torso_tilt_rad > strict_fall_max_torso_tilt_rad) | (
                torso_height < strict_fall_min_torso_height
            )
            torso_fall_consecutive[:] = torch.where(
                torso_bad,
                torso_fall_consecutive + 1,
                torch.zeros_like(torso_fall_consecutive),
            )
            torso_now = torso_fall_consecutive >= strict_fall_required_steps
            torso_fall_candidate_steps += int(torso_now.sum().item())
            if torso_fall_first_candidate_step is None and bool(torso_now.any().item()):
                torso_fall_first_candidate_step = timestep + 1
            terminated_count += int(torch.sum(terminated_only_tensor).item())
            truncated_count += int(torch.sum(timeout_tensor).item())
            if first_termination_step is None and bool(torch.any(done_tensor).item()):
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
            if (
                cfg.video
                and video_stop_on_termination
                and bool(torch.any(done_tensor).item())
            ):
                if pre_step_video_frame is not None:
                    frames.append(pre_step_video_frame)
                # Do not append the ordinary post-step frame: the environment
                # has already auto-reset the terminated row at this point.
                break
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
            if timestep >= int(cfg.get("video_start_step", 0)):
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(_overlay_fall_audit(frame, env.unwrapped))
            if multi_shot_sequence is None and timestep >= int(cfg.video_length):
                break
        timestep += 1
        if (
            target_audit_all_hit_control_step is not None
            and timestep
            >= target_audit_all_hit_control_step + target_audit_post_hit_steps
        ):
            break
        if multi_shot_sequence is not None and (
            multi_shot_complete or multi_shot_failure is not None
        ):
            break
        max_steps = (
            multi_shot_max_steps
            if multi_shot_sequence is not None
            else (
                cfg.get("max_steps", None)
                if not target_audit_enabled
                else cfg.get("target_audit_max_steps", 350)
            )
        )
        if max_steps is not None and timestep >= int(max_steps):
            break
        # non-video: keep stepping until the Isaac Sim window is closed (live viewing)

    if strike_goal_shadow_pipeline is not None:
        import hashlib

        shadow_report = strike_goal_shadow_pipeline.to_mapping()
        shadow_report["command_path"] = str(
            strike_goal_shadow_context["command_path"]
        )
        shadow_report["verified_pre_receipt_delay_s"] = strike_goal_shadow_context[
            "verified_pre_receipt_delay_s"
        ]
        shadow_report["hope_world_to_sim_transform"] = {
            "source_frame": strike_goal_shadow_context[
                "hope_world_to_sim"
            ].source_frame,
            "target_frame": strike_goal_shadow_context[
                "hope_world_to_sim"
            ].target_frame,
            "rotation": [
                list(row)
                for row in strike_goal_shadow_context[
                    "hope_world_to_sim"
                ].rotation
            ],
            "translation": list(
                strike_goal_shadow_context["hope_world_to_sim"].translation
            ),
        }
        shadow_report["receipt_base_position_w"] = strike_goal_shadow_context[
            "receipt_base_position_w"
        ].detach().cpu().tolist()
        shadow_report["receipt_base_heading_w"] = strike_goal_shadow_context[
            "receipt_base_heading_w"
        ].detach().cpu().tolist()
        policy_action_trace = strike_goal_shadow_context["policy_action_trace"]
        policy_action_bytes = json.dumps(
            policy_action_trace, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        shadow_report["policy_action_trace"] = policy_action_trace
        shadow_report["policy_action_trace_sha256"] = hashlib.sha256(
            policy_action_bytes
        ).hexdigest()
        shadow_report_path = strike_goal_shadow_context["report_path"]
        shadow_temporary_path = shadow_report_path.with_suffix(
            shadow_report_path.suffix + ".tmp"
        )
        shadow_temporary_path.write_text(
            json.dumps(shadow_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        shadow_temporary_path.replace(shadow_report_path)
        print(
            "[INFO] wrote strike-goal P2 shadow report -> "
            f"{shadow_report_path} ({len(strike_goal_shadow_pipeline.samples)} samples)",
            flush=True,
        )

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
        f"min_base_upright={min_base_upright:.4f}, "
        f"max_root_tilt={max_root_tilt_rad:.4f} rad "
        f"({max_root_tilt_rad * 57.2957795:.2f} deg), "
        f"min_torso_height={min_torso_height:.4f} m, "
        f"max_torso_tilt={max_torso_tilt_rad:.4f} rad "
        f"({max_torso_tilt_rad * 57.2957795:.2f} deg), "
        f"max_root_speed={max_root_speed:.4f} m/s, "
        f"max_root_ang_speed={max_root_ang_speed:.4f} rad/s, "
        f"terminated_count={terminated_count}, truncated_count={truncated_count}, "
        f"first_termination_step={first_termination_step}, "
        f"strict_fall_candidate_steps={strict_fall_candidate_steps}, "
        f"strict_fall_first_candidate_step={strict_fall_first_candidate_step}, "
        f"torso_fall_candidate_steps={torso_fall_candidate_steps}, "
        f"torso_fall_first_candidate_step={torso_fall_first_candidate_step}",
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

    def _goal_state_layers_for_report(motion_id, hit_record):
        """Attach the P4C four-layer audit contract without changing actions."""
        manifest_value = (
            getattr(env_cfg.commands.motion, "motion_manifest", None)
            if has_motion_command
            else None
        )
        if manifest_value is None:
            return None
        source_path = pathlib.Path(str(manifest_value)).expanduser()
        if not source_path.is_absolute():
            source_path = pathlib.Path.cwd() / source_path
        source = json.loads(source_path.read_text(encoding="utf-8"))
        entries = source.get("motions", [])
        if not 0 <= int(motion_id) < len(entries):
            return None
        layers = copy.deepcopy(entries[int(motion_id)].get("goal_state_layers"))
        if layers is None:
            return None
        layers["source_manifest"] = str(source_path.resolve())
        layers["runtime_frame"] = "base_yaw_heading_at_command_receipt"
        if hit_record is not None:
            runtime_target = {
                "position_b_m": hit_record.get("target_position_b_m"),
                "normal_b": hit_record.get("target_normal_b"),
                "linear_velocity_b_mps": hit_record.get("target_velocity_b_mps"),
                "control_step": hit_record.get("control_step"),
            }
            actual_state = {
                "position_b_m": hit_record.get("actual_position_b_m"),
                "normal_b": hit_record.get("actual_normal_b"),
                "linear_velocity_b_mps": hit_record.get("actual_velocity_b_mps"),
                "position_error_m": hit_record.get("position_error_m"),
                "normal_error_deg": hit_record.get("normal_error_deg"),
                "velocity_error_mps": hit_record.get("velocity_error_mps"),
                "control_step": hit_record.get("control_step"),
            }
            layers["runtime_policy_target_hit_state_b"] = runtime_target
            layers["actual_execution_hit_state"] = actual_state

            reference_state = layers.get("adapted_reference_hit_state_b0")
            def _state_error(source, destination):
                if source is None or destination is None:
                    return None
                def _vector(state, runtime_key, reference_key):
                    value = state.get(runtime_key, state.get(reference_key))
                    if value is None:
                        # P5 R-class references may carry only the canonical
                        # planner/trajectory provenance, rather than P4's
                        # historical adapted-reference label.  That is not a
                        # malformed reference and must not prevent a dynamic
                        # tracker audit from being written.
                        return None
                    return torch.tensor(value)

                source_position = _vector(
                    source, "position_b_m", "racket_position_b0_m"
                )
                destination_position = _vector(
                    destination, "position_b_m", "racket_position_b0_m"
                )
                source_velocity = _vector(
                    source, "linear_velocity_b_mps", "racket_velocity_b0_mps"
                )
                destination_velocity = _vector(
                    destination,
                    "linear_velocity_b_mps",
                    "racket_velocity_b0_mps",
                )
                source_normal = _vector(source, "normal_b", "racket_normal_b0")
                destination_normal = _vector(
                    destination, "normal_b", "racket_normal_b0"
                )
                if any(
                    value is None
                    for value in (
                        source_position,
                        destination_position,
                        source_velocity,
                        destination_velocity,
                        source_normal,
                        destination_normal,
                    )
                ):
                    return None
                normal_cosine = torch.dot(source_normal, destination_normal) / (
                    torch.linalg.vector_norm(source_normal)
                    * torch.linalg.vector_norm(destination_normal)
                ).clamp_min(1.0e-12)
                return {
                    "position_error_m": float(
                        torch.linalg.vector_norm(
                            destination_position - source_position
                        ).item()
                    ),
                    "normal_error_deg": float(
                        torch.rad2deg(
                            torch.acos(normal_cosine.clamp(-1.0, 1.0))
                        ).item()
                    ),
                    "velocity_error_mps": float(
                        torch.linalg.vector_norm(
                            destination_velocity - source_velocity
                        ).item()
                    ),
                }

            layers["error_decomposition"] = {
                "runtime_target_to_adapted_reference": _state_error(
                    runtime_target, reference_state
                ),
                "adapted_reference_to_actual_execution": _state_error(
                    reference_state, actual_state
                ),
                "runtime_target_to_actual_execution": _state_error(
                    runtime_target, actual_state
                ),
            }
        return layers

    if target_audit_enabled:
        complete = all(record is not None for record in target_audit_hit_records)
        missing_trial_ids = [
            index
            for index, record in enumerate(target_audit_hit_records)
            if record is None
        ]
        completed_trials = [
            record for record in target_audit_hit_records if record is not None
        ]
        for record in completed_trials:
            trial_id = int(record["trial_id"])
            terminal_step = int(
                target_audit_first_physical_termination_step[trial_id].item()
            )
            record["first_physical_termination_control_step"] = (
                None if terminal_step < 0 else terminal_step
            )
            record["physical_termination_after_hit_steps"] = (
                None
                if terminal_step < 0
                else terminal_step - int(record["control_step"])
            )
            record["physical_termination_reasons"] = (
                target_audit_first_physical_termination_reasons[trial_id]
            )
        def _paired_control_offset(record) -> list[float]:
            # P4-C may calibrate the physical hit centre away from the
            # manifest anchor.  Pairing then has to use the same small local
            # offset that reaches the adapter, rather than the potentially
            # decimetre-scale external-minus-manifest value.  Before P4-C the
            # two vectors are exactly equal, preserving all earlier audits.
            return record.get(
                "control_offset_from_calibrated_anchor_b_m",
                record["requested_offset_b_m"],
            )

        baseline = next(
            (
                record
                for record in completed_trials
                # Absolute auto-selected targets are represented in float32.
                # Treat sub-micrometre residuals as nominal so their paired
                # response remains measurable instead of being silently lost
                # to anchor subtraction round-off.
                if max(abs(value) for value in _paired_control_offset(record))
                <= 1.0e-6
            ),
            None,
        )
        axis_pairs = []
        if baseline is not None:
            baseline_position = torch.tensor(
                baseline["actual_position_b_m"], dtype=torch.float64
            )
            baseline_actor = torch.tensor(
                baseline["initial_upper_actor_output"], dtype=torch.float64
            )
            baseline_root = torch.tensor(
                baseline["root_displacement_b_m"], dtype=torch.float64
            )
            baseline_reach = torch.tensor(
                baseline["racket_relative_root_b_m"], dtype=torch.float64
            )
            for record in completed_trials:
                expected = torch.tensor(
                    _paired_control_offset(record), dtype=torch.float64
                )
                actual_response = (
                    torch.tensor(record["actual_position_b_m"], dtype=torch.float64)
                    - baseline_position
                )
                actor_response = (
                    torch.tensor(
                        record["initial_upper_actor_output"], dtype=torch.float64
                    )
                    - baseline_actor
                )
                root_response = (
                    torch.tensor(
                        record["root_displacement_b_m"], dtype=torch.float64
                    )
                    - baseline_root
                )
                reach_response = (
                    torch.tensor(
                        record["racket_relative_root_b_m"],
                        dtype=torch.float64,
                    )
                    - baseline_reach
                )
                expected_norm = float(torch.linalg.norm(expected).item())
                response_norm = float(torch.linalg.norm(actual_response).item())
                paired_response = {
                    "actual_response_from_nominal_b_m": actual_response.tolist(),
                    "actual_response_norm_m": response_norm,
                    "root_response_from_nominal_b_m": root_response.tolist(),
                    "racket_relative_root_response_from_nominal_b_m": (
                        reach_response.tolist()
                    ),
                    "initial_actor_response_l2": float(
                        torch.linalg.norm(actor_response).item()
                    ),
                }
                if expected_norm > 0.0:
                    dot = float(torch.dot(actual_response, expected).item())
                    gain = dot / (expected_norm * expected_norm)
                    directional_cosine = (
                        dot / (response_norm * expected_norm)
                        if response_norm > 0.0
                        else 0.0
                    )
                    along = gain * expected
                    paired_response.update(
                        {
                            "commanded_offset_norm_m": expected_norm,
                            "along_command_gain": gain,
                            "directional_cosine": directional_cosine,
                            "cross_axis_response_norm_m": float(
                                torch.linalg.norm(actual_response - along).item()
                            ),
                            "response_error_m": float(
                                torch.linalg.norm(actual_response - expected).item()
                            ),
                        }
                    )
                record["nominal_paired_response"] = paired_response

            for positive in completed_trials:
                offset = torch.tensor(
                    _paired_control_offset(positive), dtype=torch.float64
                )
                # The calibrated anchor is represented in float32, so a
                # nominal local-zero can retain a few 1e-7 m components.
                # Use the same micrometre tolerance as baseline detection;
                # otherwise every intended axis sample appears three-axis.
                nonzero = torch.where(torch.abs(offset) > 1.0e-6)[0]
                if (
                    nonzero.numel() != 1
                    or float(offset[nonzero[0]].item()) <= 0.0
                ):
                    continue
                negative = next(
                    (
                        candidate
                        for candidate in completed_trials
                        if max(
                            abs(a + b)
                            for a, b in zip(
                                _paired_control_offset(candidate),
                                _paired_control_offset(positive),
                            )
                        )
                        <= 1.0e-6
                    ),
                    None,
                )
                if negative is None:
                    continue
                axis = int(nonzero[0].item())
                radius = float(offset[axis].item())
                plus_position = torch.tensor(
                    positive["actual_position_b_m"], dtype=torch.float64
                )
                minus_position = torch.tensor(
                    negative["actual_position_b_m"], dtype=torch.float64
                )
                plus_actor = torch.tensor(
                    positive["initial_upper_actor_output"], dtype=torch.float64
                )
                minus_actor = torch.tensor(
                    negative["initial_upper_actor_output"], dtype=torch.float64
                )
                plus_root = torch.tensor(
                    positive["root_displacement_b_m"], dtype=torch.float64
                )
                minus_root = torch.tensor(
                    negative["root_displacement_b_m"], dtype=torch.float64
                )
                plus_reach = torch.tensor(
                    positive["racket_relative_root_b_m"], dtype=torch.float64
                )
                minus_reach = torch.tensor(
                    negative["racket_relative_root_b_m"], dtype=torch.float64
                )
                axis_pairs.append(
                    {
                        "axis": "xyz"[axis],
                        "radius_m": radius,
                        "negative_trial_id": negative["trial_id"],
                        "positive_trial_id": positive["trial_id"],
                        "position_jacobian_column": (
                            (plus_position - minus_position) / (2.0 * radius)
                        ).tolist(),
                        "root_position_jacobian_column": (
                            (plus_root - minus_root) / (2.0 * radius)
                        ).tolist(),
                        "racket_relative_root_jacobian_column": (
                            (plus_reach - minus_reach) / (2.0 * radius)
                        ).tolist(),
                        "actor_central_difference_l2_per_m": float(
                            torch.linalg.norm(
                                (plus_actor - minus_actor) / (2.0 * radius)
                            ).item()
                        ),
                    }
                )

            if adapter_jacobian_enabled and len(completed_trials) >= 15:
                # The internal-adapter audit keeps the Cartesian command at
                # zero and perturbs one raw adapter joint per paired env.  Its
                # trial rows therefore cannot use requested_offset_b_m for
                # pairing; recover the deterministic nominal,-,+ layout made
                # above and report the effective hit-point Jacobian per raw
                # adapter unit.
                axis_pairs = []
                for joint in range(7):
                    negative = completed_trials[1 + 2 * joint]
                    positive = completed_trials[2 + 2 * joint]
                    plus_position = torch.tensor(
                        positive["actual_position_b_m"], dtype=torch.float64
                    )
                    minus_position = torch.tensor(
                        negative["actual_position_b_m"], dtype=torch.float64
                    )
                    plus_root = torch.tensor(
                        positive["root_displacement_b_m"], dtype=torch.float64
                    )
                    minus_root = torch.tensor(
                        negative["root_displacement_b_m"], dtype=torch.float64
                    )
                    plus_reach = torch.tensor(
                        positive["racket_relative_root_b_m"], dtype=torch.float64
                    )
                    minus_reach = torch.tensor(
                        negative["racket_relative_root_b_m"], dtype=torch.float64
                    )
                    denominator = 2.0 * float(adapter_jacobian_step)
                    axis_pairs.append(
                        {
                            "adapter_joint_index": joint,
                            "negative_trial_id": negative["trial_id"],
                            "positive_trial_id": positive["trial_id"],
                            "raw_action_step": float(adapter_jacobian_step),
                            "position_jacobian_column": (
                                (plus_position - minus_position) / denominator
                            ).tolist(),
                            "root_position_jacobian_column": (
                                (plus_root - minus_root) / denominator
                            ).tolist(),
                            "racket_relative_root_jacobian_column": (
                                (plus_reach - minus_reach) / denominator
                            ).tolist(),
                        }
                    )
            if coordinator_jacobian_enabled and len(completed_trials) >= 45:
                # Same finite-difference convention as the arm-adapter audit,
                # but across the public 12-leg + 3-waist + 7-arm coordinator.
                axis_pairs = []
                for joint in range(22):
                    negative = completed_trials[1 + 2 * joint]
                    positive = completed_trials[2 + 2 * joint]
                    plus_position = torch.tensor(
                        positive["actual_position_b_m"], dtype=torch.float64
                    )
                    minus_position = torch.tensor(
                        negative["actual_position_b_m"], dtype=torch.float64
                    )
                    plus_root = torch.tensor(
                        positive["root_displacement_b_m"], dtype=torch.float64
                    )
                    minus_root = torch.tensor(
                        negative["root_displacement_b_m"], dtype=torch.float64
                    )
                    plus_reach = torch.tensor(
                        positive["racket_relative_root_b_m"], dtype=torch.float64
                    )
                    minus_reach = torch.tensor(
                        negative["racket_relative_root_b_m"], dtype=torch.float64
                    )
                    denominator = 2.0 * float(coordinator_jacobian_step)
                    axis_pairs.append(
                        {
                            "coordinator_action_index": joint,
                            "negative_trial_id": negative["trial_id"],
                            "positive_trial_id": positive["trial_id"],
                            "raw_action_step": float(coordinator_jacobian_step),
                            "position_jacobian_column": (
                                (plus_position - minus_position) / denominator
                            ).tolist(),
                            "root_position_jacobian_column": (
                                (plus_root - minus_root) / denominator
                            ).tolist(),
                            "racket_relative_root_jacobian_column": (
                                (plus_reach - minus_reach) / denominator
                            ).tolist(),
                        }
                    )

        if hit_schedule is not None and target_audit_all_hit_control_step is not None:
            actual_hit_time_s = target_audit_all_hit_control_step * float(
                env.unwrapped.step_dt
            )
            hit_schedule["actual_hit_time_s"] = actual_hit_time_s
            hit_schedule["actual_minus_requested_s"] = (
                actual_hit_time_s - hit_schedule["request_time_from_commit_s"]
            )
        report = {
            "schema_version": 1,
            "audit": "external_racket_position_conditioning",
            "checkpoint": str(resume_path),
            "task": str(cfg.task.name),
            "motion_id": int(
                env.unwrapped.command_manager.get_term("motion").motion_ids[0].item()
            ),
            "motion_ids": env.unwrapped.command_manager.get_term("motion")
            .motion_ids.detach()
            .cpu()
            .tolist(),
            "auto_motion_selection": auto_motion_selection,
            "external_strike_request": None
            if external_strike_request is None
            else {
                "request_path": external_strike_request["request_path"],
                "request_id": external_strike_request["request_id"],
            },
            "hit_schedule": hit_schedule,
            "seed": int(cfg.seed),
            "p4c_upper_execution_mode": p4c_upper_execution_mode,
            "complete": complete,
            "control_steps": timestep,
            "all_hit_control_step": target_audit_all_hit_control_step,
            "post_hit_steps_requested": target_audit_post_hit_steps,
            "missing_trial_ids": missing_trial_ids,
            "coordinate_contract": {
                "frame": "base_yaw_heading_at_command_receipt",
                "world_target_is_fixed_during_swing": True,
                "position_only_override": True,
                "local_adapter_pairing": "calibrated_control_anchor_when_configured",
                "velocity_source": "selected_manifest_anchor",
                "normal_source": "selected_manifest_anchor",
                "strike_time_source": "selected_manifest_anchor",
                "siblings_share_physical_strike_ready_state": bool(
                    target_audit_synchronize_siblings
                ),
                "startup_physics_domain_randomization_disabled": bool(
                    target_audit_synchronize_siblings
                ),
            },
            "goal_state_layers": _goal_state_layers_for_report(
                int(env.unwrapped.command_manager.get_term("motion").motion_ids[0].item()),
                target_audit_hit_records[0] if target_audit_hit_records else None,
            ),
            "grid_radii_cm": target_grid_cm,
            "trials": target_audit_hit_records,
            "axis_pairs": axis_pairs,
            "physical_termination_count": terminated_count,
            "timeout_count": truncated_count,
            "first_termination_step": first_termination_step,
            "stability": {
                "min_root_height_m": min_root_height,
                "max_root_speed_mps": max_root_speed,
                "max_root_angular_speed_radps": max_root_ang_speed,
                "unified_predicted_unrecoverable_steps": unified_predicted_unrecoverable_steps,
                "unified_confirmed_fall_steps": unified_confirmed_fall_steps,
                "unified_recovery_ready_steps": unified_recovery_ready_steps,
            },
        }
        if record_trace:
            report.update(
                {
                    "termination": single_shot_termination,
                    "trace": single_shot_trace,
                }
            )
        temporary_path = target_audit_report_path.with_suffix(
            target_audit_report_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(target_audit_report_path)
        print(
            "[INFO] external target audit: "
            f"complete={complete}, hits={len(completed_trials)}/{env.num_envs}, "
            f"axis_pairs={len(axis_pairs)}",
            flush=True,
        )
        print(
            f"[INFO] wrote external target audit -> {target_audit_report_path}",
            flush=True,
        )
    elif multi_shot_sequence is not None:
        if multi_shot_complete and multi_shot_records:
            multi_shot_records[-1]["ready_control_step"] = timestep
            multi_shot_records[-1]["ready_hold_steps"] = int(
                env.unwrapped.stage_a_sagittal_rearm_stable_steps[0].item()
            )
            multi_shot_records[-1]["rearm_stable_start_control_step"] = (
                multi_shot_records[-1]["ready_control_step"]
                - multi_shot_records[-1]["ready_hold_steps"]
                + 1
            )
            multi_shot_records[-1]["ready_state"] = _multi_shot_state_snapshot(
                env.unwrapped
            )
        _write_multi_shot_report(
            "complete" if multi_shot_complete else "stopped_without_terminal_event"
        )
        print(
            "[INFO] multi-shot audit: "
            f"complete={multi_shot_complete}, shots={len(multi_shot_records)}/"
            f"{len(multi_shot_sequence)}, failure={multi_shot_failure}",
            flush=True,
        )
        print(f"[INFO] wrote multi-shot report -> {multi_shot_report_path}", flush=True)
    elif record_trace:
        selected_motion_id = int(cfg.get("motion_id", 0) or 0)
        report = {
            "checkpoint": str(resume_path),
            "task": str(cfg.task.name),
            "motion_id": selected_motion_id,
            "seed": int(cfg.seed),
            "p4c_upper_execution_mode": p4c_upper_execution_mode,
            "zero_tracker_residual": _as_bool(
                cfg.get("zero_tracker_residual", False)
            ),
            "control_steps": timestep,
            "physical_termination_count": terminated_count,
            "timeout_count": truncated_count,
            "termination": single_shot_termination,
            "stability": {
                "min_root_height_m": min_root_height,
                "max_root_speed_mps": max_root_speed,
                "max_root_angular_speed_radps": max_root_ang_speed,
                "unified_predicted_unrecoverable_steps": unified_predicted_unrecoverable_steps,
                "unified_confirmed_fall_steps": unified_confirmed_fall_steps,
                "unified_recovery_ready_steps": unified_recovery_ready_steps,
            },
            "strike": single_shot_hit_record,
            "goal_state_layers": _goal_state_layers_for_report(
                selected_motion_id, single_shot_hit_record
            ),
            "trace": single_shot_trace,
        }
        report_path = cfg.get("single_shot_report", None)
        if report_path is None:
            raise ValueError(
                "record_trace=true requires single_shot_report=<path> for "
                "single-shot evaluation"
            )
        report_path = pathlib.Path(str(report_path)).expanduser()
        if not report_path.is_absolute():
            report_path = pathlib.Path.cwd() / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[INFO] wrote single-shot report -> {report_path}", flush=True)

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
