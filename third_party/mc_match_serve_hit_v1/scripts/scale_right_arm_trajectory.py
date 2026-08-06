#!/usr/bin/env python3
"""Scale right-arm motion about its first pose without changing timing."""

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
        print(f"usage: {sys.argv[0]} INPUT OUTPUT SCALE", file=sys.stderr)
        return 2
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    scale = float(sys.argv[3])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a finite positive number")

    data = source.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("trajectory is shorter than its header")
    magic, frames, dof, dt = HEADER.unpack_from(data)
    if magic != b"A3UPTRJ1" or dof != DOF or frames < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid A3 upper trajectory header")
    count = frames * dof
    expected = HEADER.size + 2 * count * 8
    if len(data) != expected:
        raise ValueError(f"unexpected trajectory size: {len(data)} != {expected}")

    values = list(struct.unpack_from(f"<{2 * count}d", data, HEADER.size))
    q = values[:count]
    qd = values[count:]
    reference = q[RIGHT_START:RIGHT_START + RIGHT_DOF]
    for frame in range(frames):
        for joint in range(RIGHT_DOF):
            index = frame * DOF + RIGHT_START + joint
            q[index] = reference[joint] + scale * (q[index] - reference[joint])
            qd[index] *= scale

    output = bytearray(data[:HEADER.size])
    output.extend(struct.pack(f"<{2 * count}d", *(q + qd)))
    target.write_bytes(output)
    print(f"frames={frames} dt={dt:.9f} scale={scale:.9f}")
    print(f"duration_s={(frames - 1) * dt:.9f}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
