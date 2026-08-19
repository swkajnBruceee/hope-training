"""Export a trained HOPE checkpoint to a deployable ONNX policy + manifest.

Loads a local checkpoint, rebuilds the policy, and writes:

* ``policy.onnx``            — single-output actor graph, observation[1, 110] -> raw_action[1, 31]
* ``policy_manifest.json``   — the contract (name, dims, control rate, joint order, obs
                               normalization = none, ActionAdapter config path)

The contract name and dims are NOT hardcoded: they come from the actor observation
contract registry entry the environment actually implements (the shipped
HitterPingPong task uses ``hitter_pure``, 110-D). The canonical joint order is
embedded in the ONNX metadata (key ``joint_order``) together with the contract
name (key ``contract``) so the deploy loader can reject a permuted or
wrong-contract model at load time.

Usage:
    python scripts/export_onnx.py --checkpoint logs/rsl_rl/agibot_a3_hitter_pingpong/<run>/model_<iter>.pt

By default the files are written to ``<checkpoint_dir>/exported/``.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
from types import SimpleNamespace

MANIFEST_NAME = "policy_manifest.json"
# Repo-root-relative path of the shared adapter config recorded in the manifest.
ACTION_ADAPTER_RELPATH = "a3_deploy/a3_deploy_example/config/action_adapter.yaml"


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cfg").is_dir() and (parent / "source" / "whole_body_tracking").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    return str(rooted.resolve()) if rooted.is_file() else str(rooted)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_native_runtime_metadata(onnx_path: str, policy, checkpoint: str) -> None:
    """Attach the native runner contract while preserving trained-policy metadata.

    The native graph exporter produces the correct multi-output graph, but the compact
    deploy exporter does not carry the large historical runtime metadata contract.  Use
    the verified deployment ONNX as the contract template and replace only
    provenance/model-variant fields for the new checkpoint.
    """
    import onnx

    project_root = next(
        (
            parent
            for parent in (_repo_root(), *_repo_root().parents)
            if (parent / "a3_deploy" / "a3_deploy_example").is_dir()
        ),
        None,
    )
    if project_root is None:
        raise FileNotFoundError("could not locate the HOPETableTennis project root")
    template_path = (
        project_root
        / "a3_deploy"
        / "a3_deploy_example"
        / "models"
        / "model_21800"
        / "policy"
        / "exported"
        / "policy.onnx"
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            "native export requires the verified metadata template: "
            f"{template_path}"
        )
    model = onnx.load(onnx_path)
    template = onnx.load(str(template_path))
    metadata = {item.key: item.value for item in template.metadata_props}
    policy_metadata = getattr(policy, "get_model_metadata", None)
    if callable(policy_metadata):
        for key, value in policy_metadata().items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True, separators=(",", ":"))
            metadata[str(key)] = str(value)
    metadata["hitter_pure_checkpoint_sha256"] = _sha256_file(checkpoint)
    metadata["export_source"] = "native_motion_export_with_runtime_contract"
    del model.metadata_props[:]
    for key in sorted(metadata):
        item = model.metadata_props.add()
        item.key = str(key)
        item.value = str(metadata[key])
    onnx.save(model, onnx_path)


def assert_canonical_joint_order(joint_names, expected_order) -> None:
    """HARD GATE: the articulation enumeration must equal the canonical deploy joint order.

    The articulation's joint enumeration fixes the obs joint_pos/joint_vel/actions
    slices and all 31 action columns of the exported ONNX. If the asset enumerates
    differently, every column would be silently permuted at deploy.
    """
    if list(joint_names) != list(expected_order):
        raise RuntimeError(
            "Articulation joint order does not match the canonical deploy joint order "
            "(hope_training/config/joint_order_agibot_a3.yaml).\n"
            f"  articulation: {list(joint_names)}\n"
            f"  canonical:    {list(expected_order)}\n"
            "Fix your A3 URDF/USD so its joint enumeration matches the canonical order "
            "(or update the canonical order everywhere: training, planner, deploy runner)."
        )


def build_manifest(
    contract,
    joint_names,
    control_rate_hz: int = 50,
    onnx_name: str = "policy.onnx",
    model_metadata: dict | None = None,
) -> dict:
    """Build the deploy manifest from a registry contract entry (duck-typed).

    ``contract`` needs ``.name``, ``.total_dim`` and ``.layout`` (name, dim) pairs —
    the shape of :class:`ActorObservationContract` from the contract registry.
    """
    action_dim = len(list(joint_names))
    layout = []
    cursor = 0
    for term_name, term_dim in contract.layout:
        layout.append({"name": str(term_name), "dim": int(term_dim), "slice": [cursor, cursor + int(term_dim)]})
        cursor += int(term_dim)
    if cursor != int(contract.total_dim):
        raise ValueError(
            f"contract layout dims sum to {cursor}, expected total_dim {contract.total_dim}"
        )
    manifest = {
        "contract_name": str(contract.name),
        "obs_dim": int(contract.total_dim),
        "action_dim": action_dim,
        "control_rate_hz": int(control_rate_hz),
        "observation_normalization": "none",
        "observation_layout": layout,
        "onnx_file": str(onnx_name),
        "onnx_signature": {
            "input": {"name": "observation", "shape": [1, int(contract.total_dim)]},
            "output": {"name": "raw_action", "shape": [1, action_dim]},
        },
        "joint_order": list(joint_names),
        "action_adapter_config": ACTION_ADAPTER_RELPATH,
    }
    if model_metadata is not None:
        manifest["model_metadata"] = model_metadata
    return manifest


def _joint_reorder_for_contract(contract, articulation_joint_names, canonical_joint_names):
    """Build explicit PhysX-articulation <-> deploy-canonical permutations."""
    articulation_joint_names = list(articulation_joint_names)
    canonical_joint_names = list(canonical_joint_names)
    if len(articulation_joint_names) != 31 or set(articulation_joint_names) != set(canonical_joint_names):
        raise ValueError("articulation and canonical joint names must be the same 31-joint set")
    canonical_index = {name: index for index, name in enumerate(canonical_joint_names)}
    articulation_index = {name: index for index, name in enumerate(articulation_joint_names)}
    observation_perm = list(range(int(contract.total_dim)))
    cursor = 0
    for term_name, term_dim in contract.layout:
        term_dim = int(term_dim)
        if str(term_name) in {"joint_pos", "joint_vel", "actions"}:
            if term_dim != 31:
                raise ValueError(f"expected 31D {term_name} field, got {term_dim}")
            observation_perm[cursor : cursor + 31] = [
                cursor + canonical_index[name] for name in articulation_joint_names
            ]
        cursor += term_dim
    output_perm = [articulation_index[name] for name in canonical_joint_names]
    return observation_perm, output_perm


def _collect_hitter_pure_deploy_metadata(env) -> dict[str, str]:
    """Collect geometry metadata without changing the canonical ONNX joint contract.

    The compact Residual export intentionally embeds only one observation input and one
    action output.  The MuJoCo/deploy loader nevertheless needs the same
    hitter-pure sampling envelopes as the reference export.  Do not copy ``joint_names``
    from Isaac here: the wrapper already converts the trained articulation order to the
    canonical deployment order, and adding articulation ``joint_names`` would make the
    loader permute the input a second time.
    """
    metadata: dict[str, str] = {}
    try:
        command = env.unwrapped.command_manager.get_term("racket_target")
        cfg = command.cfg
    except Exception:
        return metadata
    if str(getattr(cfg, "target_mode", "")) != "hitter_pure":
        return metadata

    def _clip_rows(value) -> str | None:
        if value is None:
            return None
        return ";".join(
            ",".join(f"{float(lo):.4f},{float(hi):.4f}" for lo, hi in clip)
            for clip in value
        )

    for attr, key in (
        ("racket_pos_range_per_clip", "hitter_pure_pos_range_per_clip"),
        ("racket_vel_range_per_clip", "hitter_pure_vel_range_per_clip"),
        ("racket_vel_start_range_per_clip", "hitter_pure_vel_core_range_per_clip"),
        ("racket_vel_planner_range_per_clip", "hitter_pure_vel_planner_range_per_clip"),
    ):
        value = _clip_rows(getattr(cfg, attr, None))
        if value is not None:
            metadata[key] = value

    planner_mix = getattr(cfg, "racket_vel_planner_mix_prob", None)
    if planner_mix is not None:
        metadata["hitter_pure_vel_planner_mix_prob"] = f"{float(planner_mix):.4f}"
    ramp_steps = getattr(cfg, "racket_vel_range_ramp_steps", None)
    if ramp_steps is not None:
        metadata["hitter_pure_vel_range_ramp_steps"] = str(int(ramp_steps))
    base_x = getattr(cfg, "base_target_x_range", None)
    base_y = getattr(cfg, "base_target_y_range", None)
    if base_x is not None and base_y is not None:
        metadata["hitter_pure_base_target_range"] = ",".join(
            f"{float(value):.4f}" for value in (*base_x, *base_y)
        )

    pos_ranges = getattr(cfg, "racket_pos_range_per_clip", None)
    if pos_ranges and all(
        abs(float(clip[0][0]) - float(clip[0][1])) < 1.0e-8 for clip in pos_ranges
    ):
        metadata["hitter_pure_fixed_plane_x"] = f"{float(pos_ranges[0][0][0]):.4f}"

    try:
        command._ensure_reference_strike_state()
        reach = getattr(command, "_ref_reach_offset_xy_per_clip", None)
        if reach is not None:
            metadata["ref_reach_offset_xy"] = ",".join(
                f"{float(value):.6f}" for value in reach.reshape(-1).detach().cpu().tolist()
            )
    except Exception:
        # Reach offsets are an optional deploy aid.  The core geometry above remains valid
        # for compact actors even when an older command implementation lacks this helper.
        pass
    return metadata


def export_deploy_policy(
    policy,
    contract,
    articulation_joint_names,
    canonical_joint_names,
    output_dir: str,
    onnx_name: str,
    control_rate_hz: int,
    extra_metadata: dict[str, str] | None = None,
):
    """Trace the actor into a single-input/single-output ONNX + write the manifest.

    Returns ``(onnx_path, manifest_path)``. Embeds the canonical ``joint_order``
    and the ``contract`` name into the ONNX metadata for the loader-side gate.
    """
    import onnx
    import torch

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, onnx_name)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)

    observation_perm, output_perm = _joint_reorder_for_contract(
        contract, articulation_joint_names, canonical_joint_names
    )

    class _ActorWrapper(torch.nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor_critic = actor_critic
            self.register_buffer("observation_perm", torch.tensor(observation_perm, dtype=torch.long))
            self.register_buffer("output_perm", torch.tensor(output_perm, dtype=torch.long))

        def forward(self, observation):
            # Deploy input/output are canonical by name; the trained policy itself
            # remains in Isaac's articulation order.
            policy_observation = observation.index_select(1, self.observation_perm)
            articulation_action = self.actor_critic.act_inference(policy_observation)
            return articulation_action.index_select(1, self.output_perm)

    module = _ActorWrapper(policy).to("cpu").eval()
    dummy = torch.zeros(1, int(contract.total_dim))
    torch.onnx.export(
        module,
        (dummy,),
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["observation"],
        output_names=["raw_action"],
        dynamic_axes={},
    )

    model = onnx.load(onnx_path)
    model_metadata_fn = getattr(policy, "get_model_metadata", None)
    model_metadata = model_metadata_fn() if callable(model_metadata_fn) else None
    for key, value in (
        ("joint_order", ",".join(canonical_joint_names)),
        ("trained_articulation_joint_order", ",".join(articulation_joint_names)),
        ("contract", str(contract.name)),
        *(
            [("model_variant", str(model_metadata["model_variant"]))]
            if model_metadata is not None and "model_variant" in model_metadata
            else []
        ),
    ):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    if extra_metadata:
        existing = {entry.key for entry in model.metadata_props}
        for key, value in extra_metadata.items():
            if key in existing:
                continue
            entry = model.metadata_props.add()
            entry.key = str(key)
            entry.value = str(value)
    onnx.save(model, onnx_path)

    manifest = build_manifest(
        contract,
        canonical_joint_names,
        control_rate_hz=control_rate_hz,
        onnx_name=onnx_name,
        model_metadata=model_metadata,
    )
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return onnx_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint (.pt) to export.")
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Explicitly allow historical A5/Residual/non-scratch checkpoints; current exports must omit this.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <ckpt_dir>/exported).")
    parser.add_argument("--task", default="HOPE-HitterPingPong-AgibotA3-v0", help="Gym task id.")
    parser.add_argument(
        "--task-config",
        default="cfg/task/HOPEPingPong.yaml",
        help=(
            "Task YAML merged on top of the base HOPEPingPong recipe; use "
            "cfg/task/HOPEPingPongStanceCurriculum.yaml for Curriculum-FT exports."
        ),
    )
    parser.add_argument("--onnx-name", default="policy.onnx", help="Exported ONNX filename.")
    parser.add_argument(
        "--native-format",
        action="store_true",
        help="Export the native runner graph (obs + time_step and motion side outputs).",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to build (1 is enough to export).")
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--motion-file",
        default="motions/preprocessed/hope_forehand.npz",
        help="Forehand clip (only needed so the env instantiates).",
    )
    parser.add_argument(
        "--motion-file-2",
        default="motions/preprocessed/hope_backhand.npz",
        help="Backhand clip (only needed so the env instantiates).",
    )
    parser.add_argument("--experiment-name", default="agibot_a3_hitter_pingpong", help="rsl_rl experiment name.")
    parser.add_argument(
        "--algo-config",
        default=None,
        help="Optional PPO YAML; default uses the current standard cfg/algo/ppo.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    import torch

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not args.allow_legacy_checkpoint:
        from whole_body_tracking.utils.scratch_contract import validate_scratch_checkpoint

        validate_scratch_checkpoint(
            checkpoint_payload,
            amp_enabled=None,
            path=checkpoint,
        )
    else:
        print("[export_onnx] legacy checkpoint compatibility explicitly enabled", flush=True)
    output_dir = args.output_dir or os.path.join(os.path.dirname(checkpoint), "exported")

    # Launch Isaac (headless) before importing isaaclab modules; clear argv so Kit ignores our args.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app

    status = 0
    try:
        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import importlib

        importlib.import_module("whole_body_tracking.tasks")  # registers the gym tasks
        from whole_body_tracking.tasks.tracking.actor_observation_contract import (
            infer_actor_observation_contract,
        )
        from whole_body_tracking.utils.action_adapter_config import load_joint_order
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs
        from whole_body_tracking.utils.scratch_contract import validate_scratch_algorithm
        from whole_body_tracking.utils.exporter import export_motion_policy_as_onnx
        from train import _apply_task_overrides
        from omegaconf import OmegaConf

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        # Export must instantiate the exact same task recipe as train/evaluate.
        # The registered Gym defaults alone omit the full-pose mocap and ping-pong
        # command overrides required by one_way_strike_gate_v1.
        base_task_cfg_path = _repo_root() / "cfg" / "task" / "HOPEPingPong.yaml"
        task_cfg_path = pathlib.Path(args.task_config).expanduser()
        if not task_cfg_path.is_absolute():
            task_cfg_path = (_repo_root() / task_cfg_path).resolve()
        if not task_cfg_path.is_file():
            raise FileNotFoundError(f"task config not found: {task_cfg_path}")
        task_cfg = OmegaConf.load(str(base_task_cfg_path))
        if task_cfg_path.resolve() != base_task_cfg_path.resolve():
            task_cfg = OmegaConf.merge(task_cfg, OmegaConf.load(str(task_cfg_path)))
        applied_overrides = []
        _apply_task_overrides(env_cfg, SimpleNamespace(task=task_cfg), applied_overrides)
        clips = [_resolve_motion_path(c) for c in (args.motion_file, args.motion_file_2) if c]
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        articulation_joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
        canonical_joint_names = list(load_joint_order())
        if len(articulation_joint_names) != len(canonical_joint_names) or set(articulation_joint_names) != set(canonical_joint_names):
            raise RuntimeError(
                "Articulation joint set does not match the canonical deploy joint set.\n"
                f"  articulation: {articulation_joint_names}\n"
                f"  canonical:    {canonical_joint_names}"
            )
        if articulation_joint_names == canonical_joint_names:
            print("[export_onnx] articulation order matches canonical deploy order", flush=True)
        else:
            print("[export_onnx] inserting explicit articulation/canonical joint permutations", flush=True)

        # The deploy contract comes from the registry entry the built env implements
        # (the shipped HitterPingPong task -> hitter_pure, 110-D). Refuse to export a
        # policy whose observation layout matches no registered contract.
        contract = infer_actor_observation_contract(env.unwrapped)
        if contract is None:
            raise RuntimeError(
                "The environment's policy observation layout matches no registered actor "
                "observation contract (see tasks/tracking/actor_observation_contract.py). "
                "Refusing to export an unidentifiable policy."
            )
        control_rate_hz = int(round(1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))))
        print(
            f"[export_onnx] contract={contract.name} obs_dim={contract.total_dim} "
            f"control_rate={control_rate_hz} Hz",
            flush=True,
        )

        env = RslRlVecEnvWrapper(env)

        algo_params = load_ppo_params(args.algo_config)
        validate_scratch_algorithm(algo_params, enabled=not args.allow_legacy_checkpoint)
        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo_params, args.experiment_name))
        agent_cfg.device = args.device
        runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(checkpoint)

        if args.native_format:
            # The native MuJoCo/AimRT runner consumes the same multi-output graph
            # as the model_21800 deployment: obs + time_step inputs,
            # followed by actions and motion-reference side outputs.  This path
            # also calls ResidualMeanActorCritic.act_inference(), so the exported
            # action is HOPE mean + bounded residual rather than HOPE alone.
            export_motion_policy_as_onnx(
                env.unwrapped,
                runner.alg.policy,
                path=output_dir,
                normalizer=getattr(runner.alg.policy, "actor_obs_normalizer", None),
                filename=args.onnx_name,
            )
            onnx_path = os.path.join(output_dir, args.onnx_name)
            _restore_native_runtime_metadata(onnx_path, runner.alg.policy, checkpoint)
            manifest_path = None
        else:
            onnx_path, manifest_path = export_deploy_policy(
                runner.alg.policy,
                contract,
                articulation_joint_names,
                canonical_joint_names,
                output_dir,
                onnx_name=args.onnx_name,
                control_rate_hz=control_rate_hz,
                extra_metadata=_collect_hitter_pure_deploy_metadata(env),
            )
        print(f"[export_onnx] wrote {onnx_path}", flush=True)
        if manifest_path is not None:
            print(f"[export_onnx] wrote {manifest_path}", flush=True)
        env.close()
    except Exception:
        import traceback

        print("\n[export_onnx] ERROR:", flush=True)
        traceback.print_exc()
        status = 1
    finally:
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
