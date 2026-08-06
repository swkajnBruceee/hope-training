#!/usr/bin/env python3
"""Delay the right seven-joint arm while preserving the left-arm timeline."""

import math
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("<8sIId")
DOF = 14
RIGHT_START = 7
RIGHT_DOF = 7


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} INPUT OUTPUT DELAY_S", file=sys.stderr)
        return 2
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    delay_s = float(sys.argv[3])
    data = source.read_bytes()
    magic, frames, dof, dt = HEADER.unpack_from(data, 0)
    if magic != b"A3UPTRJ1" or dof != DOF or frames < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid upper trajectory header")
    delay_frames = int(round(delay_s / dt))
    if delay_frames < 0 or abs(delay_frames * dt - delay_s) > 1e-8:
        raise ValueError("delay must be a non-negative whole number of samples")
    count = frames * dof
    if len(data) != HEADER.size + 2 * count * 8:
        raise ValueError("unexpected upper trajectory size")
    values = struct.unpack_from(f"<{2 * count}d", data, HEADER.size)
    old_q = values[:count]
    old_qd = values[count:]
    new_frames = frames + delay_frames
    new_q = [0.0] * (new_frames * dof)
    new_qd = [0.0] * (new_frames * dof)
    for frame in range(new_frames):
        left_source = min(frame, frames - 1)
        right_source = min(max(frame - delay_frames, 0), frames - 1)
        for joint in range(7):
            left_dst = frame * dof + joint
            left_src = left_source * dof + joint
            new_q[left_dst] = old_q[left_src]
            new_qd[left_dst] = old_qd[left_src] if frame < frames else 0.0
            right_dst = frame * dof + RIGHT_START + joint
            right_src = right_source * dof + RIGHT_START + joint
            new_q[right_dst] = old_q[right_src]
            new_qd[right_dst] = old_qd[right_src] if frame >= delay_frames else 0.0
    output = HEADER.pack(b"A3UPTRJ1", new_frames, DOF, dt)
    output += struct.pack(f"<{len(new_q)}d", *new_q)
    output += struct.pack(f"<{len(new_qd)}d", *new_qd)
    target.write_bytes(output)
    print(f"old_frames={frames} new_frames={new_frames} dt={dt:.9f} delay_frames={delay_frames}")
    print(f"old_duration_s={(frames - 1) * dt:.9f} new_duration_s={(new_frames - 1) * dt:.9f}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
