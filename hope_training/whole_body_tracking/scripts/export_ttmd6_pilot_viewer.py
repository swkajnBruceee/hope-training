#!/usr/bin/env python3
"""Export a browser-only viewer for raw TTMD6 pilot clips.

This viewer is intentionally source-space only. It does not apply A3 frame
conversions, infer a hit frame, or write training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EDGES = [
    [0, 1],
    [0, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7],
    [0, 8], [8, 9], [9, 10],
    [0, 11], [11, 12], [12, 13],
]


def load_points(path: Path, point_count: int) -> list[list[list[float]]]:
    frames: list[list[list[float]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            values = [float(value) for value in row]
            if not any(values):
                break
            if len(values) != point_count * 3:
                raise ValueError(f"{path}: expected {point_count * 3} values, got {len(values)}")
            frames.append([values[i : i + 3] for i in range(0, len(values), 3)])
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot_json.read_text(encoding="utf-8"))
    clips = []
    for record in pilot["records"]:
        human_path = Path(record["human_path"])
        bat_path = Path(record["bat_path"])
        human = load_points(human_path, 14)
        bat = load_points(bat_path, 1)
        frame_count = min(len(human), len(bat))
        human = human[:frame_count]
        bat = bat[:frame_count]
        clips.append(
            {
                "id": f"class{record['class_id']}_sample{record['sample_id']}",
                "class_id": record["class_id"],
                "label": record["class_label"],
                "group_id": record["group_id"],
                "fps": record["fps"],
                "source_length": record["source_length_declared"],
                "stored_frames": frame_count,
                "human": human,
                "bat": bat,
            }
        )

    payload = json.dumps({"edges": EDGES, "clips": clips}, ensure_ascii=True, separators=(",", ":"))
    html = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TTMD6 pilot source viewer</title>
<style>
body { margin: 0; background: #101317; color: #e9edf2; font: 14px sans-serif; }
header { padding: 12px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; background: #181d23; }
select, button, input { font: inherit; }
button, select { padding: 5px 8px; background: #252d36; color: #e9edf2; border: 1px solid #4a5663; }
main { display: grid; grid-template-columns: minmax(520px, 1fr) 300px; gap: 12px; padding: 12px; }
canvas { width: 100%; max-width: 900px; aspect-ratio: 1.25; background: #20262d; border: 1px solid #3b4652; }
aside { line-height: 1.6; background: #181d23; padding: 12px; border: 1px solid #303944; }
.muted { color: #9ca8b5; }
code { color: #b9e6ff; }
</style>
</head>
<body>
<header>
  <label>Clip <select id="clip"></select></label>
  <button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button>
  <label>Frame <input id="frame" type="range" min="0" value="0"></label>
</header>
<main>
  <canvas id="view" width="1000" height="800"></canvas>
  <aside>
    <div><strong id="title"></strong></div>
    <div id="meta"></div>
    <hr>
    <div>Blue: human skeleton. Red: paddle center of gravity.</div>
    <div class="muted">Projection uses source X/Z axes only. No A3 transform or hit-frame inference is applied.</div>
    <hr>
    <div id="state"></div>
  </aside>
</main>
<script>
const DATA = __PAYLOAD__;
const edges = DATA.edges;
const clips = DATA.clips;
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const clipSelect = document.getElementById('clip');
const frameInput = document.getElementById('frame');
const title = document.getElementById('title');
const meta = document.getElementById('meta');
const state = document.getElementById('state');
let clipIndex = 0, frameIndex = 0, playing = false, timer = null;
for (const [i, clip] of clips.entries()) {
  const option = document.createElement('option');
  option.value = i; option.textContent = `${clip.id} | ${clip.label}`;
  clipSelect.appendChild(option);
}
function setClip(i) {
  clipIndex = (i + clips.length) % clips.length;
  frameIndex = 0;
  clipSelect.value = clipIndex;
  frameInput.max = Math.max(0, clips[clipIndex].stored_frames - 1);
  draw();
}
function draw() {
  const clip = clips[clipIndex], human = clip.human[frameIndex], bat = clip.bat[frameIndex];
  frameInput.value = frameIndex;
  title.textContent = `${clip.id} | ${clip.label}`;
  meta.innerHTML = `group=${clip.group_id}<br>source_length=${clip.source_length}<br>stored_frames=${clip.stored_frames}<br>fps=${clip.fps}`;
  state.textContent = `frame ${frameIndex + 1}/${clip.stored_frames}, t=${(frameIndex / clip.fps).toFixed(3)} s`;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#20262d'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const all = human.concat(bat);
  const xs = all.map(p => p[0]), zs = all.map(p => p[2]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const sx = (canvas.width - 140) / Math.max(1, maxX - minX);
  const sz = (canvas.height - 120) / Math.max(1, maxZ - minZ);
  const scale = Math.min(sx, sz);
  const ox = 70 - minX * scale, oy = canvas.height - 60 + minZ * scale;
  const xy = p => [ox + p[0] * scale, oy - p[2] * scale];
  ctx.strokeStyle = '#4b5968'; ctx.lineWidth = 2;
  for (const [a,b] of edges) { const p=xy(human[a]), q=xy(human[b]); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke(); }
  for (const p of human) { const q=xy(p); ctx.fillStyle='#6eb6ff'; ctx.beginPath(); ctx.arc(q[0],q[1],8,0,Math.PI*2); ctx.fill(); }
  const bp=xy(bat); ctx.fillStyle='#ff5b5b'; ctx.beginPath(); ctx.arc(bp[0],bp[1],11,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#c7d0d9'; ctx.fillText('X horizontal / Z vertical; depth Y omitted', 20, 26);
}
function tick() { if (!playing) return; frameIndex = (frameIndex + 1) % clips[clipIndex].stored_frames; draw(); timer=setTimeout(tick, 1000 / 30); }
clipSelect.onchange = e => setClip(Number(e.target.value));
frameInput.oninput = e => { frameIndex=Number(e.target.value); draw(); };
document.getElementById('prev').onclick = () => setClip(clipIndex - 1);
document.getElementById('next').onclick = () => setClip(clipIndex + 1);
document.getElementById('play').onclick = e => { playing=!playing; e.target.textContent=playing?'Pause':'Play'; if(playing) tick(); };
setClip(0);
</script>
</body>
</html>
""".replace("__PAYLOAD__", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
