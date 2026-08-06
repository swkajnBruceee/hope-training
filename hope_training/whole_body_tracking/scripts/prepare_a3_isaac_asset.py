#!/usr/bin/env python3
"""Prepare the Agibot A3 ping-pong URDF for Isaac Lab.

The source URDF package is kept under agibot/ so users can inspect the original
materials. Isaac Lab loads a derived copy under the whole_body_tracking Python
package asset directory, where mesh paths are relative to model.urdf.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "agibot" / "URDF" / "A3T2.5-URDF-std-pingpang"
DEFAULT_MUJOCO_SOURCE = (
    REPO_ROOT
    / "agibot"
    / "A3_MuJoCo_Sim"
    / "aimrt_mujoco_sim"
    / "src"
    / "models"
    / "bin"
    / "cfg"
    / "model"
    / "a3_pingpong"
)
DEFAULT_DEST = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "training"
    / "assets"
    / "agibot_a3"
)
SOURCE_URDF = "URDF-JOINT-LINK.urdf"
DEST_URDF = "model.urdf"
RACKET_FACE_COLLISION_MESH = "right_racket_face_collision.STL"
REQUIRED_MESHES = (
    "pelvis_link.STL",
    "torso_Link.STL",
    "right_hand_pingpang_Link.STL",
    "pingpang_red_Link.STL",
    "pingpang_black_Link.STL",
    "pingbang_ball_Link.STL",
    RACKET_FACE_COLLISION_MESH,
)


def _rewrite_mesh_paths(text: str) -> str:
    """Rewrite package://<pkg>/meshes/foo.STL to ../meshes/foo.STL."""
    return re.sub(r"package://[^/]+/meshes/", "../meshes/", text)


def _sanitize_fixed_joint_axes(text: str) -> str:
    """Replace malformed/empty/zero axes on fixed joints with a valid placeholder.

    Isaac Sim's URDF importer validates the axis string even for fixed joints,
    and the Agibot-provided URDF contains empty, single-value, or zero-vector
    axes that cause import failures. A fixed joint has no rotational DOF, so
    the axis value is arbitrary as long as it parses.
    """

    def _fix_joint(match: re.Match) -> str:
        joint_block = match.group(0)
        # Only touch fixed joints.
        if 'type="fixed"' not in joint_block:
            return joint_block
        # Replace any <axis xyz="..."/> with a valid unit vector.
        joint_block = re.sub(r'<axis\s+xyz="[^"]*"\s*/?>', '<axis xyz="0 0 1"/>', joint_block)
        # Some axes in the source are not self-closed.
        joint_block = re.sub(r'<axis\s+xyz="[^"]*">', '<axis xyz="0 0 1">', joint_block)
        return joint_block

    # Match each <joint ...>...</joint> block (non-greedy inner match).
    return re.sub(r"<joint\b[^>]*>.*?</joint>", _fix_joint, text, flags=re.DOTALL)


def _patch_racket_collision(text: str) -> str:
    """Use a single stable racket-face collision mesh for Isaac PhysX.

    The original URDF uses two very thin visual face meshes plus an orange marker sphere as collision
    meshes. In Isaac/PhysX this makes racket-ball contact unreliable and can let the ball pass through
    a visually aligned paddle. The MuJoCo asset ships a dedicated centered racket-face collision mesh;
    use it on the red paddle link and remove the overlapping black/marker collisions.
    """

    def _replace_link_collision(match: re.Match) -> str:
        block = match.group(0)
        return re.sub(
            r"<collision>.*?</collision>",
            (
                "<collision>\n"
                '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
                "      <geometry>\n"
                f'        <mesh filename="../meshes/{RACKET_FACE_COLLISION_MESH}"/>\n'
                "      </geometry>\n"
                "    </collision>"
            ),
            block,
            count=1,
            flags=re.DOTALL,
        )

    def _remove_link_collision(match: re.Match) -> str:
        return re.sub(r"\n    <collision>.*?</collision>", "", match.group(0), count=1, flags=re.DOTALL)

    text = re.sub(
        r'<link name="pingpang_red_Link">.*?</link>',
        _replace_link_collision,
        text,
        count=1,
        flags=re.DOTALL,
    )
    for link_name in ("pingpang_black_Link", "pingbang_ball_Link"):
        text = re.sub(
            rf'<link name="{link_name}">.*?</link>',
            _remove_link_collision,
            text,
            count=1,
            flags=re.DOTALL,
        )
    return text


def prepare(source: Path, dest: Path, force: bool) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"source package not found: {source}")

    source_urdf = source / "urdf" / SOURCE_URDF
    if not source_urdf.exists():
        raise FileNotFoundError(f"source URDF not found: {source_urdf}")

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in ("meshes", "config"):
        src = source / name
        if src.exists():
            shutil.copytree(src, dest / name, dirs_exist_ok=True)

    racket_collision_src = (
        DEFAULT_MUJOCO_SOURCE / "meshes" / "collision_optimized" / RACKET_FACE_COLLISION_MESH
    )
    if racket_collision_src.exists():
        shutil.copy2(racket_collision_src, dest / "meshes" / RACKET_FACE_COLLISION_MESH)

    if (source / "package.xml").exists():
        shutil.copy2(source / "package.xml", dest / "package.xml")

    urdf_dest_dir = dest / "urdf"
    urdf_dest_dir.mkdir(parents=True, exist_ok=True)
    text = _rewrite_mesh_paths(source_urdf.read_text(encoding="utf-8"))
    text = _sanitize_fixed_joint_axes(text)
    text = _patch_racket_collision(text)
    model_urdf = urdf_dest_dir / DEST_URDF
    model_urdf.write_text(text, encoding="utf-8")
    return model_urdf


def check(dest: Path) -> None:
    model_urdf = dest / "urdf" / DEST_URDF
    if not model_urdf.exists():
        raise FileNotFoundError(f"prepared URDF not found: {model_urdf}")

    text = model_urdf.read_text(encoding="utf-8")
    if "package://" in text:
        raise RuntimeError(f"stale package:// mesh path remains in {model_urdf}")

    referenced_meshes = set(re.findall(r"\.\./meshes/([^\"'<>\s]+)", text))
    required_meshes = sorted(set(REQUIRED_MESHES).union(referenced_meshes))
    missing = [name for name in required_meshes if not (dest / "meshes" / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required mesh(es): {', '.join(missing)}")

    print(f"[OK] prepared A3 Isaac asset: {model_urdf}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="source A3 URDF package")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="prepared Isaac asset directory")
    parser.add_argument("--force", action="store_true", help="replace the destination directory first")
    parser.add_argument("--check", action="store_true", help="only verify the prepared asset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()

    if not args.check:
        model_urdf = prepare(source, dest, force=args.force)
        print(f"[OK] wrote {model_urdf}")

    check(dest)


if __name__ == "__main__":
    main()
