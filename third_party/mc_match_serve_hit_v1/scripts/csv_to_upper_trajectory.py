#!/usr/bin/env python3
"""Convert a named A3 arm CSV into the compact A3UPTRJ1 runtime format."""

import csv
import math
import struct
import sys
from pathlib import Path


ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HEADER = struct.Struct("<8sIId")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT_CSV OUTPUT_BIN", file=sys.stderr)
        return 2
    source, target = map(Path, sys.argv[1:])
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or not rows[0]:
        raise ValueError("CSV has fewer than two samples")
    missing = [name for name in ARM_JOINTS if name not in rows[0]]
    if missing:
        raise ValueError("missing arm columns: " + ", ".join(missing))
    q = []
    times = []
    for row in rows:
        times.append(float(row["time_s"]))
        q.extend(float(row[name]) for name in ARM_JOINTS)
    dt = times[1] - times[0]
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid CSV time step")
    if any(abs((times[i + 1] - times[i]) - dt) > 1e-7 for i in range(len(times) - 1)):
        raise ValueError("CSV time step is not uniform")
    qd = [0.0] * len(q)
    dof = len(ARM_JOINTS)
    for frame in range(len(rows)):
        for joint in range(dof):
            if frame == 0:
                derivative = (q[dof + joint] - q[joint]) / dt
            elif frame == len(rows) - 1:
                derivative = (q[joint + (frame * dof)] - q[joint + ((frame - 1) * dof)]) / dt
            else:
                derivative = (q[(frame + 1) * dof + joint] - q[(frame - 1) * dof + joint]) / (2.0 * dt)
            qd[frame * dof + joint] = derivative
    if not all(math.isfinite(value) for value in q + qd):
        raise ValueError("CSV contains non-finite arm data")
    output = HEADER.pack(b"A3UPTRJ1", len(rows), dof, dt)
    output += struct.pack(f"<{len(q)}d", *q)
    output += struct.pack(f"<{len(qd)}d", *qd)
    target.write_bytes(output)
    print(f"frames={len(rows)} dt={dt:.9f} duration_s={(len(rows)-1)*dt:.9f}")
    print(f"left_elbow_first_rad={q[3]:.12f} left_elbow_first_deg={math.degrees(q[3]):.6f}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
