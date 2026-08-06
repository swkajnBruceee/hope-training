"""Standalone regression tests for the table-tennis geometry and ball-aerodynamics math.

These cover the pure-Python / pure-torch pieces that do NOT need Isaac Sim, so they can run on a plain
dev machine. Modules are loaded by file path to avoid importing ``training.tasks`` (which
pulls in Isaac Lab). The ball-aerodynamics tests are skipped automatically if ``torch`` is unavailable.

Run:  python tests/test_table_tennis_geometry.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

_PKG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # whole_body_tracking/ (parent of tests/)
    "training",
    "tasks",
    "table_tennis",
)


def _load(name: str, filename: str):
    path = os.path.join(_PKG, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required by dataclasses on py3.12+ to resolve annotations
    spec.loader.exec_module(mod)
    return mod


geometry = _load("tt_geometry", "geometry.py")


def test_table_dimensions_match_ittf():
    assert geometry.TABLE_LENGTH == 2.74
    assert geometry.TABLE_WIDTH == 1.525
    assert geometry.TABLE_HEIGHT == 0.76
    assert geometry.NET_X == 1.37
    assert geometry.NET_HEIGHT == 0.1525
    assert geometry.BALL_RADIUS == 0.02
    assert geometry.BALL_MASS == 0.0027


def test_hope_frame_landmarks():
    # Origin is the near-side left corner; the table occupies x in [0,2.74], y in [-1.525,0], z=0.
    assert geometry.ORIGIN == (0.0, 0.0, 0.0)
    assert geometry.TABLE_CENTER == (1.37, -0.7625, 0.0)
    assert geometry.NET_CENTER == (1.37, -0.7625, 0.0)
    assert geometry.P1_HALF_CENTER == (0.685, -0.7625, 0.0)
    assert geometry.P2_HALF_CENTER == (2.055, -0.7625, 0.0)
    # Floor is one table-height below the surface (z=0 is the surface).
    assert geometry.FLOOR_Z == -0.76


def test_table_top_slab_top_face_at_surface():
    cx, cy, cz = geometry.table_top_center()
    sx, sy, sz = geometry.table_top_size()
    assert (cx, cy) == (1.37, -0.7625)
    # Top face of the slab sits exactly at HOPE z = 0.
    assert math.isclose(cz + sz / 2.0, 0.0, abs_tol=1e-9)
    assert (sx, sy) == (2.74, 1.525)


def test_net_spans_table_width_plus_overhang():
    nx, ny, nz = geometry.net_size()
    assert math.isclose(ny, geometry.TABLE_WIDTH + 2 * geometry.NET_OVERHANG)
    cx, cy, cz = geometry.net_center()
    assert cx == 1.37
    # Net slab spans z in [0, NET_HEIGHT].
    assert math.isclose(cz - nz / 2.0, 0.0, abs_tol=1e-9)
    assert math.isclose(cz + nz / 2.0, geometry.NET_HEIGHT, abs_tol=1e-9)


def test_robot_stands_on_p1_side_centered():
    assert geometry.P1_STAND_X < 0.0  # behind the near table end
    assert math.isclose(geometry.P1_STAND_Y, -geometry.TABLE_WIDTH / 2.0)


def test_out_of_bounds_box_contains_table():
    box = geometry.OutOfBoundsBox().as_dict()
    assert box["x"][0] < 0.0 and box["x"][1] > geometry.TABLE_LENGTH
    assert box["y"][0] < -geometry.TABLE_WIDTH and box["y"][1] > 0.0


def _run_ball_tests() -> int:
    try:
        import torch  # noqa: F401
    except Exception:
        print("[skip] torch unavailable -> skipping ball-aerodynamics tests")
        return 0

    import torch

    ball = _load("tt_ball", "ball.py")

    try:
        # Pure drag: a = -k |v| v  =>  F = -m k |v| v, exactly opposing velocity.
        cfg = ball.BallAerodynamicsCfg(enabled=True, drag_coefficient=0.5, magnus_coefficient=0.0)
        v = torch.tensor([[3.0, 0.0, 0.0], [0.0, -4.0, 0.0]])
        w = torch.zeros_like(v)
        m = 0.0027
        f, t = ball.compute_aero_wrench(v, w, m, cfg)
        assert torch.allclose(t, torch.zeros_like(t))
        # First sample: |v|=3 -> F_x = -m*k*3*3 = -m*4.5
        assert torch.allclose(f[0], torch.tensor([-m * 0.5 * 9.0, 0.0, 0.0]), atol=1e-9)
        # Drag must point opposite velocity.
        assert torch.all(torch.sum(f * v, dim=-1) <= 0.0)

        # Magnus: F_magnus = m k_m (w x v), perpendicular to both w and v.
        cfg_m = ball.BallAerodynamicsCfg(enabled=True, drag_coefficient=0.0, magnus_coefficient=0.1)
        v2 = torch.tensor([[5.0, 0.0, 0.0]])
        w2 = torch.tensor([[0.0, 0.0, 10.0]])  # topspin about +z
        f2, _ = ball.compute_aero_wrench(v2, w2, m, cfg_m)
        expected = m * 0.1 * torch.cross(w2, v2, dim=-1)
        assert torch.allclose(f2, expected, atol=1e-9)
    except AssertionError as e:
        print(f"[FAIL] ball-aerodynamics tests: {e}")
        return 1

    print("[ok] ball-aerodynamics tests passed")
    return 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    failed += _run_ball_tests()
    print(f"\n{len(tests) - failed}/{len(tests)} geometry tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
