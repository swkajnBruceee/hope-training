#!/usr/bin/env python3
"""Pair-specific PhysX racket contact calibration for the V1.3B runtime contract.

The 48-case local-axis/distance sweep is retained for every sampled wrist
orientation.  Contact qualification is based on PhysX contact-report headers
whose actor/collider paths contain both the ball and the runtime wrist/racket
collider, not on the ball's unfiltered net force.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any

import torch
from isaaclab.app import AppLauncher

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BALL_RADIUS_M = 0.020
MOUNT_OFFSET = (0.21021, 0.032078, 0.032036)
BODY_NAME = "right_wrist_yaw_Link"
MIN_PAIR_CONTACTS = 10
MIN_ORIENTATIONS = 3
NORMAL_AXIS_TOL = math.radians(5.0)
NORMAL_SIGN_TOL = math.radians(5.0)
STD_NORMAL_MAX_M = 0.002
MAX_LATERAL_MAX_M = 0.003


def vec(x: Any) -> list[float]:
    if torch.is_tensor(x):
        return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]
    return [float(v) for v in x]


def _path(value: Any) -> str:
    try:
        from omni.physx.scripts.physicsUtils import PhysicsSchemaTools

        return str(PhysicsSchemaTools.intToSdfPath(value))
    except Exception:
        return str(value)


def _contact_to_dict(header: Any, datum: Any, ball_center: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    position = vec(datum.position)
    normal = vec(datum.normal)
    impulse = vec(datum.impulse)
    d = ball_center - reference
    return {
        "actor0": _path(header.actor0),
        "actor1": _path(header.actor1),
        "collider0": _path(header.collider0),
        "collider1": _path(header.collider1),
        "position_w": position,
        "normal_w": normal,
        "impulse_w": impulse,
        "separation_m": float(datum.separation),
        "face_index0": int(datum.face_index0),
        "face_index1": int(datum.face_index1),
        "ball_center_w": vec(ball_center),
        "reference_point_w": vec(reference),
        "d_ball_minus_reference_w": vec(d),
    }


def _is_pair_event(event: dict[str, Any]) -> bool:
    joined = " ".join(event[k].lower() for k in ("actor0", "actor1", "collider0", "collider1"))
    return "/ball" in joined and ("right_wrist_yaw_link" in joined or "racket" in joined or "pingpang" in joined)


def _pair_normal_sign(event: dict[str, Any], expected: torch.Tensor) -> float:
    n = torch.tensor(event["normal_w"], dtype=torch.float32, device=expected.device)
    return float(torch.dot(n / torch.linalg.vector_norm(n).clamp_min(1.0e-8), expected).detach().cpu())


def _qualification(rows: list[dict[str, Any]], pair_events: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [r for r in rows if r["pair_contact_count"] > 0]
    valid_events = [e for e in pair_events if e.get("local_axis") == 1 and e.get("sign") == 1.0]
    normals = [float(e["d_normal_m"]) for e in valid_events]
    laterals = [float(e["lateral_residual_m"]) for e in valid_events]
    signs = [float(e["event_normal_alignment"]) for e in valid_events]
    orientation_count = len({r["orientation"] for r in valid_rows})
    normal_std = float(torch.tensor(normals).std(unbiased=False)) if normals else None
    max_lateral = max(laterals, default=None)
    axis_consistent = all(abs(int(e["local_axis"]) - 1) == 0 for e in valid_events)
    sign_consistent = all(abs(abs(s) - 1.0) <= math.sin(NORMAL_SIGN_TOL) for s in signs)
    passed = (
        len(valid_events) >= MIN_PAIR_CONTACTS
        and orientation_count >= MIN_ORIENTATIONS
        and axis_consistent
        and sign_consistent
        and normal_std is not None
        and max_lateral is not None
        and normal_std < STD_NORMAL_MAX_M
        and max_lateral < MAX_LATERAL_MAX_M
    )
    return {
        "pass": bool(passed),
        "thresholds": {
            "min_pair_specific_contacts": MIN_PAIR_CONTACTS,
            "min_orientations": MIN_ORIENTATIONS,
            "normal_axis_tolerance_deg": math.degrees(NORMAL_AXIS_TOL),
            "normal_sign_tolerance_deg": math.degrees(NORMAL_SIGN_TOL),
            "std_d_normal_max_m": STD_NORMAL_MAX_M,
            "max_lateral_residual_max_m": MAX_LATERAL_MAX_M,
        },
        "metrics": {
            "pair_specific_contact_count": len(valid_events),
            "valid_row_count": len(valid_rows),
            "orientation_count": orientation_count,
            "normal_axis_consistent": axis_consistent,
            "normal_sign_consistent": sign_consistent,
            "d_normal_std_m": normal_std,
            "max_lateral_residual_m": max_lateral,
        },
    }


def main():
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False).app
    try:
        import gymnasium as gym
        import training.tasks  # noqa: F401
        from isaaclab.utils.math import matrix_from_quat, quat_apply
        from omni.physx import get_physx_simulation_interface
        from training.tasks.table_tennis.config.agibot_a3.table_tennis_env_cfg import AgibotA3HitFixedBaseTouchEnvCfg

        cfg = AgibotA3HitFixedBaseTouchEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.robot.init_state.pos = (-0.5, -0.7625, 1.04)
        cfg.sim.device = "cuda:0"
        print("[contact-probe] making env", flush=True)
        env = gym.make("HOPE-TableTennis-AgibotA3-HitFixedBaseTouch-v0", cfg=cfg, render_mode=None)
        contact_sub = None
        try:
            print("[contact-probe] env made; resetting", flush=True)
            env.reset()
            raw = env.unwrapped
            robot = raw.scene["robot"]
            ball = raw.scene["ball"]
            print("[contact-probe] assets acquired", flush=True)
            # URDF importers may leave the PhysX contact-report threshold at a
            # nonzero default.  Set it explicitly on both members of the
            # candidate pair before subscribing to the report stream.
            import omni.usd
            from pxr import PhysxSchema
            stage = omni.usd.get_context().get_stage()
            for prim_path in ("/World/envs/env_0/Ball", f"/World/envs/env_0/Robot/{BODY_NAME}"):
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim_path)
                    api.CreateThresholdAttr().Set(0.0)
                    print(f"[contact-probe] contact-report threshold=0 path={prim_path}", flush=True)
            hold_joint_pos = robot.data.joint_pos.detach().clone()
            body_index = list(robot.body_names).index(BODY_NAME)
            mount_offset = torch.tensor(MOUNT_OFFSET, device=robot.data.body_pos_w.device)
            joint_names = list(robot.joint_names)
            yaw_index = joint_names.index("right_wrist_yaw_joint")
            pitch_index = joint_names.index("right_wrist_pitch_joint")
            roll_index = joint_names.index("right_wrist_roll_joint")

            orientation_specs = [
                ("nominal", 0.0, 0.0, 0.0),
                ("yaw_plus", 0.35, 0.0, 0.0),
                ("yaw_minus", -0.35, 0.0, 0.0),
                ("pitch_plus", 0.0, 0.25, 0.0),
            ]
            state = ball.data.default_root_state.clone()
            rows: list[dict[str, Any]] = []
            pair_events: list[dict[str, Any]] = []
            all_report_headers: list[dict[str, Any]] = []
            active_context: dict[str, Any] = {}

            def on_contact_report(headers, data):
                offset = 0
                for header in headers:
                    if len(all_report_headers) < 20:
                        all_report_headers.append({
                            "actor0": _path(header.actor0),
                            "actor1": _path(header.actor1),
                            "collider0": _path(header.collider0),
                            "collider1": _path(header.collider1),
                            "num_contact_data": int(header.num_contact_data),
                        })
                    for idx in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                        datum = data[idx]
                        raw_event = {
                            "actor0": _path(header.actor0),
                            "actor1": _path(header.actor1),
                            "collider0": _path(header.collider0),
                            "collider1": _path(header.collider1),
                            "position_w": vec(datum.position),
                            "normal_w": vec(datum.normal),
                            "impulse_w": vec(datum.impulse),
                            "separation_m": float(datum.separation),
                            "face_index0": int(datum.face_index0),
                            "face_index1": int(datum.face_index1),
                        }
                        if _is_pair_event(raw_event):
                            pair_events.append({**raw_event, **active_context})

            contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)
            for orientation, yaw_delta, pitch_delta, roll_delta in orientation_specs:
                variant = hold_joint_pos.clone()
                variant[0, yaw_index] += yaw_delta
                variant[0, pitch_index] += pitch_delta
                variant[0, roll_index] += roll_delta
                robot.write_joint_state_to_sim(variant, torch.zeros_like(variant))
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                wrist_pos = robot.data.body_pos_w[0, body_index].detach().clone()
                body_quat = robot.data.body_quat_w[0, body_index].detach().clone()
                reference = wrist_pos + quat_apply(body_quat.unsqueeze(0), mount_offset.unsqueeze(0))[0]
                basis = matrix_from_quat(body_quat.unsqueeze(0))[0]
                print(f"[contact-probe] orientation={orientation} reference={vec(reference)}", flush=True)

                for axis in range(3):
                    for sign in (-1.0, 1.0):
                        normal = basis[:, axis] * sign
                        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1.0e-8)
                        for distance in (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060):
                            robot.write_joint_state_to_sim(variant, torch.zeros_like(variant))
                            raw.scene.write_data_to_sim()
                            raw.sim.forward()
                            center = reference + normal * distance
                            state[0, :3] = center
                            state[0, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=state.device)
                            state[0, 7:10] = -normal * 1.0
                            state[0, 10:13] = 0.0
                            ball.write_root_state_to_sim(state)
                            raw.scene.write_data_to_sim()
                            raw.sim.forward()
                            start_event_count = len(pair_events)
                            for _ in range(8):
                                robot.write_joint_state_to_sim(variant, torch.zeros_like(variant))
                                raw.scene.write_data_to_sim()
                                raw.sim.step()
                                raw.scene.update(raw.sim.get_rendering_dt())
                            case_events = pair_events[start_event_count:]
                            expected = basis[:, 1]
                            local_events = []
                            for event in case_events:
                                d = ball.data.root_pos_w[0] - reference
                                d_normal = float(torch.dot(d, expected).detach().cpu())
                                lateral = d - d_normal * expected
                                align = _pair_normal_sign(event, expected)
                                event.update({
                                    "local_axis": axis,
                                    "sign": sign,
                                    "d_normal_m": d_normal,
                                    "lateral_residual_m": float(torch.linalg.vector_norm(lateral).detach().cpu()),
                                    "event_normal_alignment": align,
                                })
                                local_events.append(event)
                            pair_events[start_event_count:] = local_events
                            rows.append({
                                "orientation": orientation,
                                "local_axis": axis,
                                "sign": sign,
                                "initial_distance_m": distance,
                                "pair_contact_count": len(local_events),
                                "pair_contact_event_indices": list(range(start_event_count, start_event_count + len(local_events))),
                                "reference_point_w": vec(reference),
                                "body_quat_wxyz": vec(body_quat),
                            })

            qualification = _qualification(rows, pair_events)
            result = {
                "status": "qualified" if qualification["pass"] else "physx_contact_probe_failed",
                "qualification": qualification,
                "robot_body_name": BODY_NAME,
                "robot_body_index": body_index,
                "policy_reference_point": "right_wrist_yaw_Link_origin_plus_mount_offset_v1",
                "local_offset_xyz_m": list(MOUNT_OFFSET),
                "ball_radius_m": BALL_RADIUS_M,
                "orientation_specs": [list(x) for x in orientation_specs],
                "rows": rows,
                "pair_events": pair_events,
                "all_report_headers_sample": all_report_headers,
            }
            out_dir = ROOT / "eval_outputs" / "target_conditioned_v13b_contact_probe"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2)
            pathlib.Path("/tmp/v13b_racket_contact_physx_probe.json").write_text(payload, encoding="utf-8")
            (out_dir / "physx_contact_probe_raw.json").write_text(payload, encoding="utf-8")
            (out_dir / "physx_contact_probe_summary.json").write_text(
                json.dumps({"status": result["status"], "qualification": qualification}, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(result, indent=2), flush=True)
        finally:
            contact_sub = None
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
