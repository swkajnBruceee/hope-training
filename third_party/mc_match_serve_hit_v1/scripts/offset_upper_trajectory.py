#!/usr/bin/env python3
"""Apply a fixed joint-position offset to the compact A3 upper trajectory."""

import math
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("<8sIId")
DOF = 14
LEFT_ELBOW_INDEX = 3


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} INPUT OUTPUT OFFSET_DEG", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    offset = math.radians(float(sys.argv[3]))
    data = source.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("trajectory is shorter than its header")
    magic, frames, dof, dt = HEADER.unpack_from(data)
    if magic != b"A3UPTRJ1" or dof != DOF or frames < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid A3 upper trajectory header")
    q_count = frames * dof
    expected = HEADER.size + 2 * q_count * 8
    if len(data) != expected:
        raise ValueError(f"unexpected trajectory size: {len(data)} != {expected}")
    values = list(struct.unpack_from(f"<{2 * q_count}d", data, HEADER.size))
    for frame in range(frames):
        index = frame * dof + LEFT_ELBOW_INDEX
        values[index] += offset
    result = bytearray(data[:HEADER.size])
    result.extend(struct.pack(f"<{2 * q_count}d", *values))
    target.write_bytes(result)
    print(f"frames={frames} dt={dt:.9f} left_elbow_offset_rad={offset:.12f}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
