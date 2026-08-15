#!/usr/bin/env python3
"""Record an X11 root window to MP4 using xwd and the local ffmpeg.

The bundled ffmpeg in this workspace does not provide the x11grab input
device.  ``xwd`` is available on the deployment image, so capture one XWD
frame at a time and feed decoded BGR frames to ffmpeg's rawvideo input.
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def capture_xwd(display: str) -> tuple[np.ndarray, int, int]:
    env = os.environ.copy()
    env["DISPLAY"] = display
    completed = subprocess.run(
        ["xwd", "-root", "-silent"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    blob = completed.stdout
    if len(blob) < 100:
        raise RuntimeError("xwd returned a truncated header")
    fields = struct.unpack(">25I", blob[:100])
    header_size = fields[0]
    width = fields[4]
    height = fields[5]
    bits_per_pixel = fields[11]
    bytes_per_line = fields[12]
    ncolors = fields[19]
    if bits_per_pixel != 24 or bytes_per_line < width * 4:
        raise RuntimeError(
            f"unsupported XWD layout: bpp={bits_per_pixel} "
            f"bytes_per_line={bytes_per_line} width={width}"
        )
    pixel_offset = header_size + ncolors * 12
    expected = pixel_offset + bytes_per_line * height
    if len(blob) < expected:
        raise RuntimeError(
            f"xwd returned {len(blob)} bytes, expected at least {expected}"
        )
    # The current X server exposes 24-bit BGR pixels padded to four bytes.
    rows = np.frombuffer(
        blob, dtype=np.uint8, count=bytes_per_line * height, offset=pixel_offset
    ).reshape(height, bytes_per_line)
    frame = rows[:, : width * 4].reshape(height, width, 4)[:, :, :3]
    return np.ascontiguousarray(frame), width, height


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-width", type=int, default=1280)
    args = parser.parse_args()
    if args.fps <= 0 or args.scale_width <= 0:
        parser.error("--fps and --scale-width must be positive")

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    first, width, height = capture_xwd(args.display)
    out_height = int(round(height * args.scale_width / width))
    out_height -= out_height % 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(args.fps),
            "-i",
            "-",
            "-vf",
            f"scale={args.scale_width}:{out_height}",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    period = 1.0 / args.fps
    next_frame = time.monotonic()
    frames = 0
    try:
        while not stop_requested:
            frame = first if frames == 0 else capture_xwd(args.display)[0]
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(frame.tobytes())
            frames += 1
            next_frame += period
            time.sleep(max(0.0, next_frame - time.monotonic()))
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        ffmpeg.wait()
    print(f"recorded {frames} frames to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
