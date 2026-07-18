#!/usr/bin/env python3
"""Extract A3 T2D5 model parameters without modifying the training model.

The report keeps URDF, MJCF, and user-provided motor notes separate. This is
intentional: an official SIL model value is not automatically a real-robot
calibration value, and parallel-joint equivalent values must not be applied a
second time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import yaml


def number(value: str | None) -> float | None:
    return None if value is None else float(value)


def numbers(attrs: dict[str, str], names: tuple[str, ...]) -> list[float] | None:
    value = attrs.get(names[0])
    if value is None:
        return None
    return [float(x) for x in value.split()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_yaml_file(path: Path) -> dict[str, Any]:
    """Load a small official runtime YAML while preserving its source boundary."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def extract_official_runtime_config(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"status": "not_requested", "files": {}}

    relative_paths = {
        "pd_stand": "pd_stand/default.yaml",
        "damping": "damping/default.yaml",
        "action_setting": "action_setting/default.yaml",
        "aimrt": "aimrt/default.yaml",
        "model_info": "model_info/default.yaml",
        "ik": "ik/default.yaml",
    }
    files: dict[str, Any] = {}
    for key, relative in relative_paths.items():
        path = root / relative
        if not path.is_file():
            files[key] = {"path": str(path), "missing": True}
            continue
        files[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "data": parse_yaml_file(path),
        }
    missing = [key for key, value in files.items() if value.get("missing")]
    return {
        "status": "complete" if not missing else "partial",
        "root": str(root),
        "files": files,
        "interpretation": (
            "These are official A3 T2D5 SIL/MOTION runtime configuration values. "
            "They are not per-unit encoder zero, torque constant, friction, or IMU calibration."
        ),
    }


def parse_urdf(data: bytes) -> dict[str, Any]:
    root = ET.fromstring(data)
    links: dict[str, Any] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        links[name] = {
            "origin_xyz_m": numbers(origin.attrib, ("xyz",)) if origin is not None else None,
            "mass_kg": number(mass.attrib.get("value")) if mass is not None else None,
            "inertia_kg_m2": {
                key: number(inertia.attrib.get(key)) for key in
                ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            } if inertia is not None else None,
        }

    joints: dict[str, Any] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        axis = joint.find("axis")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        joints[name] = {
            "type": joint.attrib.get("type"),
            "axis": numbers(axis.attrib, ("xyz",)) if axis is not None else None,
            "limit": {
                "lower_rad": number(limit.attrib.get("lower")),
                "upper_rad": number(limit.attrib.get("upper")),
                "effort_nm": number(limit.attrib.get("effort")),
                "velocity_rad_s": number(limit.attrib.get("velocity")),
            } if limit is not None else None,
            "dynamics": dict(dynamics.attrib) if dynamics is not None else None,
        }
    return {"links": links, "joints": joints}


def parse_mjcf(data: bytes) -> dict[str, Any]:
    root = ET.fromstring(data)
    bodies: dict[str, Any] = {}
    joints: dict[str, Any] = {}
    for body in root.findall(".//body"):
        body_name = body.attrib.get("name")
        inertial = body.find("inertial")
        if body_name and inertial is not None:
            bodies[body_name] = {
                "pos_m": numbers(inertial.attrib, ("pos",)),
                "mass_kg": number(inertial.attrib.get("mass")),
                "diaginertia_kg_m2": numbers(inertial.attrib, ("diaginertia",)),
                "inertial_quat_wxyz": numbers(inertial.attrib, ("quat",)),
            }
        for joint in body.findall("joint"):
            name = joint.attrib.get("name")
            if not name:
                continue
            joints[name] = {
                "type": joint.attrib.get("type"),
                "axis": numbers(joint.attrib, ("axis",)),
                "range_rad": numbers(joint.attrib, ("range",)),
                "actuator_force_range_nm": numbers(joint.attrib, ("actuatorfrcrange",)),
                "actuator_force_limited": joint.attrib.get("actuatorfrclimited"),
                "damping": number(joint.attrib.get("damping")),
                "frictionloss_nm": number(joint.attrib.get("frictionloss")),
            }

    actuators: dict[str, Any] = {}
    for motor in root.findall("./actuator/motor"):
        name = motor.attrib.get("name")
        if name:
            actuators[name] = dict(motor.attrib)

    option = root.find("option")
    default_geom = root.find("./default/geom")
    return {
        "model": root.attrib.get("model"),
        "option": dict(option.attrib) if option is not None else {},
        "default_geom": dict(default_geom.attrib) if default_geom is not None else {},
        "bodies": bodies,
        "joints": joints,
        "actuators": actuators,
    }


def compare(urdf: dict[str, Any], mjcf: dict[str, Any]) -> dict[str, Any]:
    joint_diffs = []
    for name in sorted(set(urdf["joints"]) & set(mjcf["joints"])):
        u = urdf["joints"][name]
        m = mjcf["joints"][name]
        limit = u.get("limit") or {}
        rng = m.get("range_rad") or []
        force = m.get("actuator_force_range_nm") or []
        if len(rng) == 2 and len(force) == 2:
            if abs(limit.get("lower_rad", rng[0]) - rng[0]) > 1e-5 or abs(limit.get("upper_rad", rng[1]) - rng[1]) > 1e-5:
                joint_diffs.append({"joint": name, "kind": "range", "urdf": [limit.get("lower_rad"), limit.get("upper_rad")], "mjcf": rng})
            mjcf_effort = max(abs(force[0]), abs(force[1]))
            if abs(limit.get("effort_nm", mjcf_effort) - mjcf_effort) > 1e-5:
                joint_diffs.append({"joint": name, "kind": "effort", "urdf": limit.get("effort_nm"), "mjcf": force})

    mass_diffs = []
    for name in sorted(set(urdf["links"]) & set(mjcf["bodies"])):
        u = urdf["links"][name].get("mass_kg")
        m = mjcf["bodies"][name].get("mass_kg")
        if u is not None and m is not None and abs(u - m) > 1e-6:
            mass_diffs.append({"link": name, "urdf_kg": u, "mjcf_kg": m, "difference_kg": m - u})

    return {
        "joint_range_or_effort_differences": joint_diffs,
        "link_mass_differences": mass_diffs,
        "mjcf_has_explicit_kp_kv": False,
        "note": "MJCF motor elements are present, but explicit controller gains are not encoded in this file.",
    }


MANUAL_SCREENSHOT_NOTES = {
    "status": "user_provided_screenshot_estimate_not_calibration",
    "motor_families": [
        {"family": "PFP-110-75", "placement": "lower J4", "motor_inertia_kg_m2": 300.8507171e-6, "gear_ratio": 20.0, "rated_torque_nm": 70.0, "peak_torque_nm": 320.0, "rated_speed_rpm": 140.0, "peak_speed_rpm": 190.0},
        {"family": "PFP-93-65", "placement": "lower J1-J3 and waist yaw", "motor_inertia_kg_m2": 138.5069e-6, "gear_ratio": 21.906, "rated_torque_nm": 45.0, "peak_torque_nm": 220.0, "rated_speed_rpm": 115.0, "peak_speed_rpm": 206.0},
        {"family": "PFP-78-58", "placement": "upper J1-J2, waist roll/pitch, lower J5-J6", "motor_inertia_kg_m2": 30.118e-6, "gear_ratio": 20.03, "rated_torque_nm": 17.0, "peak_torque_nm": 60.0, "rated_speed_rpm": 130.0, "peak_speed_rpm": 240.0},
        {"family": "PFP-59-60", "placement": "upper J3-J5", "motor_inertia_kg_m2": 10.1374e-6, "gear_ratio": 22.136, "rated_torque_nm": 10.0, "peak_torque_nm": 36.0, "rated_speed_rpm": 140.0, "peak_speed_rpm": 170.0},
        {"family": "PFP-41-48", "placement": "upper J6-J7 and head yaw/pitch", "motor_inertia_kg_m2": 1.359e-6, "gear_ratio": 24.415, "rated_torque_nm": 2.0, "peak_torque_nm": 6.0, "rated_speed_rpm": 150.0, "peak_speed_rpm": 200.0},
    ],
    "parallel_equivalent_estimates": [
        {"joint": "ankle_pitch", "factor_on_output_inertia": 5.333, "equivalent_peak_torque_nm": 118.2, "equivalent_speed_limit_rad_s": 10.8},
        {"joint": "ankle_roll", "factor_on_output_inertia": 1.66562, "equivalent_peak_torque_nm": 54.75, "equivalent_speed_limit_rad_s": 19.37},
        {"joint": "waist_pitch", "factor_on_output_inertia": 7.3, "equivalent_peak_torque_nm": 115.0, "equivalent_speed_limit_rad_s": 9.24785},
        {"joint": "parallel_joint_label_in_screenshot", "factor_on_output_inertia": 1.21, "equivalent_peak_torque_nm": 46.0, "equivalent_speed_limit_rad_s": 22.7, "verification_note": "Screenshot label appears as ankle_roll under waist section; verify whether this means waist_roll."},
    ],
    "rules": [
        "These values are approximate notes from screenshots, not measured robot calibration.",
        "The supplied URDF README says waist and ankle parallel-joint limits, speed, and torque are already serial-equivalent for training.",
        "Do not apply the gear ratio or parallel-joint equivalent factor a second time.",
    ],
}


def markdown(report: dict[str, Any]) -> str:
    urdf = report["urdf"]
    mjcf = report["mjcf"]
    lines = [
        "# A3 T2D5 Parameter Sources",
        "",
        "> This report separates official model parameters, official SIL parameters, and user-provided estimates. It is not a real-robot calibration file.",
        "",
        "## Source Files",
        "",
        "| Source | SHA256 | Role |",
        "| --- | --- | --- |",
    ]
    for source in report["sources"]:
        lines.append(f"| `{source['path']}` | `{source['sha256']}` | {source['role']} |")
    lines += [
        "",
        "## Extraction Status",
        "",
        f"- URDF links: {len(urdf['links'])}; URDF joints: {len(urdf['joints'])}",
        f"- MJCF bodies: {len(mjcf['bodies'])}; MJCF joints: {len(mjcf['joints'])}; motors: {len(mjcf['actuators'])}",
        f"- URDF/MJCF joint range or effort differences: {len(report['comparison']['joint_range_or_effort_differences'])}",
        f"- URDF/MJCF link mass differences: {len(report['comparison']['link_mass_differences'])}",
        "- Explicit low-level PD gains: not present in the MJCF; do not infer them from the motor tags.",
        "",
        "## Joint Limits And SIL Actuator Fields",
        "",
        "| Joint | URDF range rad | URDF effort Nm | URDF velocity rad/s | MJCF force range Nm | MJCF damping | MJCF friction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(urdf["joints"]):
        u = urdf["joints"][name]
        m = mjcf["joints"].get(name, {})
        limit = u.get("limit") or {}
        rng = m.get("range_rad") or []
        force = m.get("actuator_force_range_nm") or []
        lines.append(f"| `{name}` | `{rng or [limit.get('lower_rad'), limit.get('upper_rad')]}` | `{limit.get('effort_nm')}` | `{limit.get('velocity_rad_s')}` | `{force}` | `{m.get('damping')}` | `{m.get('frictionloss_nm')}` |")
    lines += [
        "",
        "## Link Mass And Inertia",
        "",
        "The URDF table is the primary articulated-model source for Isaac alignment. The MJCF values are retained separately because the official SIL model has small mass/inertia differences for some bodies.",
        "",
        "| Link | URDF mass kg | URDF inertia `[ixx, ixy, ixz, iyy, iyz, izz]` kg m2 | MJCF mass kg | MJCF diagonal inertia kg m2 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for name in sorted(urdf["links"]):
        u = urdf["links"][name]
        m = mjcf["bodies"].get(name, {})
        lines.append(f"| `{name}` | `{u.get('mass_kg')}` | `{u.get('inertia_kg_m2')}` | `{m.get('mass_kg')}` | `{m.get('diaginertia_kg_m2')}` |")
    lines += [
        "",
        "## User-Provided Screenshot Notes",
        "",
        "These are recorded for later comparison only. They are estimates from screenshots and must not override URDF/MJCF values without a source file or measurement.",
        "",
        "| Motor family | Placement | Motor inertia kg m2 | Gear ratio | Rated Nm | Peak Nm | Rated rpm | Peak rpm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in MANUAL_SCREENSHOT_NOTES["motor_families"]:
        lines.append(f"| `{item['family']}` | {item['placement']} | `{item['motor_inertia_kg_m2']}` | `{item['gear_ratio']}` | `{item['rated_torque_nm']}` | `{item['peak_torque_nm']}` | `{item['rated_speed_rpm']}` | `{item['peak_speed_rpm']}` |")
    lines += [
        "",
        "### Parallel Equivalent Notes",
        "",
        "| Joint label | Inertia factor | Peak torque Nm | Speed limit rad/s | Note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in MANUAL_SCREENSHOT_NOTES["parallel_equivalent_estimates"]:
        lines.append(f"| `{item['joint']}` | `{item['factor_on_output_inertia']}` | `{item['equivalent_peak_torque_nm']}` | `{item['equivalent_speed_limit_rad_s']}` | {item.get('verification_note', '')} |")
    runtime = report.get("official_runtime_config", {})
    if runtime.get("status") not in (None, "not_requested"):
        files = runtime.get("files", {})
        pd = files.get("pd_stand", {}).get("data", {})
        action_cfg = files.get("action_setting", {}).get("data", {})
        action = action_cfg.get("ACTION_SETTING", {})
        action_data = files.get("aimrt", {}).get("data", {})
        model = files.get("model_info", {}).get("data", {})
        publishers = action_data.get("MotionControlModule", {}).get("PublisherManager", {}).get("publisher_cfg_list", [])
        publisher_summary = "; ".join(
            f"{item.get('topic')}={item.get('frequency')}Hz"
            for item in publishers
            if isinstance(item, dict) and item.get("topic")
        )
        lines += [
            "",
            "## Official MOTION Runtime Configuration",
            "",
            "> These values describe the packaged A3 T2D5 SIL/MOTION controller configuration. They are separate from Isaac actuator gains and from per-unit hardware calibration.",
            "",
            "| Item | Official value | Source |",
            "| --- | --- | --- |",
            f"| `PD_STAND` loop period | `{action.get('PD_STAND', {}).get('TASK1', {}).get('period')}` s | `pd_stand/default.yaml`, `action_setting/default.yaml` |",
            f"| `MOTION` loop period | `{action.get('MOTION', {}).get('TASK1', {}).get('period')}` s | `action_setting/default.yaml` |",
            f"| published command/state rates | `{publisher_summary}` | `aimrt/default.yaml` |",
            f"| `PD_STAND.kp_limb` | `{pd.get('kp_limb')}` | `pd_stand/default.yaml` |",
            f"| `PD_STAND.kd_limb` | `{pd.get('kd_limb')}` | `pd_stand/default.yaml` |",
            f"| active joint groups | `{model.get('active_joint_name')}` | `model_info/default.yaml` |",
            "",
            "The complete parsed YAML and source hashes are stored in `a3_t2d5_parameters.json`. The official `ik/default.yaml` also lists native-controller locked joints; those locks must not be silently copied into the learned strike action space.",
            "",
            "### Application Rule",
            "",
            "- Use official URDF/MJCF for model geometry, mass/inertia, limits, SIL damping/friction and actuator force range.",
            "- Use official runtime YAML to reproduce official MOTION/PD_STAND timing and controller behavior in AimSim.",
            "- Keep the current Isaac `AGIBOT_A3_CFG` gains as a separately auditable simulation approximation until each value is intentionally aligned and validated.",
            "- Do not claim real hardware calibration until encoder offsets, motor/current calibration, friction/backlash, IMU alignment and measured delay are supplied.",
        ]
    lines += [
        "",
        "## Rules",
        "",
        "- The ZIP README says waist pitch/roll and ankle roll are parallel mechanisms whose URDF speed/torque values are already serial-equivalent for training.",
        "- Do not apply gear ratios or parallel-joint factors again when configuring Isaac.",
        "- MJCF `<motor>` elements do not provide the real controller's PD gains, friction compensation, delays, or calibration offsets.",
        "- Real-robot calibration remains a separate missing source and must be kept distinct from this model report.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--official-urdf", type=Path, required=True)
    parser.add_argument("--official-mjcf", type=Path, required=True)
    parser.add_argument("--official-runtime-config-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as archive:
        zip_urdf_name = next(name for name in archive.namelist() if name.endswith("/urdf/model.urdf"))
        zip_urdf = archive.read(zip_urdf_name)

    urdf_bytes = args.official_urdf.read_bytes()
    mjcf_bytes = args.official_mjcf.read_bytes()
    report = {
        "model": "A3_T2D5",
        "status": "official_model_sources_extracted;_real_robot_calibration_not_present",
        "sources": [
            {"path": str(args.zip_path), "sha256": sha256_file(args.zip_path), "role": "user-provided URDF and mesh archive"},
            {"path": f"{args.zip_path}::{zip_urdf_name}", "sha256": sha256_bytes(zip_urdf), "role": "URDF from user archive"},
            {"path": str(args.official_urdf), "sha256": sha256_file(args.official_urdf), "role": "official AimSim packaged URDF"},
            {"path": str(args.official_mjcf), "sha256": sha256_file(args.official_mjcf), "role": "official AimSim SIL MJCF"},
        ],
        "zip_readme": "The supplied README states that waist pitch/roll and ankle roll parallel-joint speed/torque values are serial-equivalent for training.",
        "urdf_from_zip": parse_urdf(zip_urdf),
        "urdf": parse_urdf(urdf_bytes),
        "mjcf": parse_mjcf(mjcf_bytes),
        "official_runtime_config": extract_official_runtime_config(args.official_runtime_config_root),
        "comparison": {},
        "manual_screenshot_notes": MANUAL_SCREENSHOT_NOTES,
    }
    report["comparison"] = compare(report["urdf"], report["mjcf"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "joint_differences": len(report["comparison"]["joint_range_or_effort_differences"]), "mass_differences": len(report["comparison"]["link_mass_differences"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
