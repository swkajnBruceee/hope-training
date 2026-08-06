#!/usr/bin/env python3
"""Advance the right seven-joint arm while preserving the left-arm timeline."""

import math
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("<8sIId")
DOF = 14
RIGHT_START = 7


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} INPUT OUTPUT ADVANCE_S", file=sys.stderr)
        return 2
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    advance_s = float(sys.argv[3])
    data = source.read_bytes()
    magic, frames, dof, dt = HEADER.unpack_from(data, 0)
    if magic != b"A3UPTRJ1" or dof != DOF or frames < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid upper trajectory header")
    advance_frames = int(round(advance_s / dt))
    if advance_frames < 0 or abs(advance_frames * dt - advance_s) > 1e-8:
        raise ValueError("advance must be a non-negative whole number of samples")
    count = frames * dof
    if len(data) != HEADER.size + 2 * count * 8:
        raise ValueError("unexpected upper trajectory size")
    values = struct.unpack_from(f"<{2 * count}d", data, HEADER.size)
    old_q = values[:count]
    old_qd = values[count:]
    new_q = [0.0] * count
    new_qd = [0.0] * count
    for frame in range(frames):
        right_source = min(frame + advance_frames, frames - 1)
        for joint in range(7):
            left_index = frame * DOF + joint
            right_index = frame * DOF + RIGHT_START + joint
            right_source_index = right_source * DOF + RIGHT_START + joint
            new_q[left_index] = old_q[left_index]
            new_qd[left_index] = old_qd[left_index]
            new_q[right_index] = old_q[right_source_index]
            new_qd[right_index] = 0.0 if frame + advance_frames >= frames else old_qd[right_source_index]
    output = HEADER.pack(b"A3UPTRJ1", frames, DOF, dt)
    output += struct.pack(f"<{len(new_q)}d", *new_q)
    output += struct.pack(f"<{len(new_qd)}d", *new_qd)
    target.write_bytes(output)
    print(f"frames={frames} dt={dt:.9f} advance_frames={advance_frames}")
    print(f"duration_s={(frames - 1) * dt:.9f}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
