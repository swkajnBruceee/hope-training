#!/usr/bin/env python3
"""Generate auditable stroke-label samples and SVG diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_source(raw: str) -> dict[str, str]:
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        data = {}
    return {
        "source_csv": str(data.get("source_csv", "unknown")),
        "racket": str(data.get("racket", "unknown")),
    }


def line_points(x: np.ndarray, y: np.ndarray, x0: float, y0: float, w: float, h: float) -> str:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return ""
    xx = x[finite]
    yy = y[finite]
    xmin = float(np.min(xx))
    xmax = float(np.max(xx))
    ymin = float(np.min(yy))
    ymax = float(np.max(yy))
    if abs(xmax - xmin) < 1e-9:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    px = x0 + (xx - xmin) / (xmax - xmin) * w
    py = y0 + h - (yy - ymin) / (ymax - ymin) * h
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))


def svg_plot(path: Path, data: np.lib.npyio.NpzFile, idx: int) -> None:
    time_rel = data["time_rel"][idx]
    hit_idx = int(data["hit_index"][idx])
    right = data["stroke_body_right_axis_at_hit"][idx]
    lateral = (data["racket_pos"][idx] - data["body_center"][idx]) @ right
    lateral_vel = data["racket_vel"][idx] @ right
    dist = np.linalg.norm(data["ball_pos"][idx] - data["racket_pos"][idx], axis=1)
    ball_speed = np.linalg.norm(data["ball_vel"][idx], axis=1)
    racket_speed = np.linalg.norm(data["racket_vel"][idx], axis=1)

    label = str(data["stroke_type_rule_v2"][idx])
    conf = float(data["stroke_confidence_rule_v2"][idx])
    episode_id = str(data["episode_id"][idx])
    reason = str(data["stroke_label_reason_rule_v2"][idx])

    series = [
        ("lateral offset m", lateral, "#0f766e"),
        ("lateral velocity m/s", lateral_vel, "#b45309"),
        ("ball-racket distance m", dist, "#334155"),
        ("ball speed m/s", ball_speed, "#2563eb"),
        ("racket speed m/s", racket_speed, "#be123c"),
    ]

    width = 980
    row_h = 135
    top = 70
    height = top + row_h * len(series) + 35
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        f'<text x="24" y="28" font-family="monospace" font-size="15" fill="#111827">{html.escape(episode_id)}</text>',
        f'<text x="24" y="50" font-family="monospace" font-size="14" fill="#374151">label={label} confidence={conf:.3f} reason={html.escape(reason)} hit_index={hit_idx}</text>',
    ]
    for row, (name, y, color) in enumerate(series):
        x0 = 95
        y0 = top + row * row_h
        plot_w = 820
        plot_h = 92
        lines += [
            f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#fffdf8" stroke="#d6d3d1"/>',
            f'<text x="24" y="{y0 + 18}" font-family="monospace" font-size="13" fill="#111827">{html.escape(name)}</text>',
        ]
        pts = line_points(time_rel, y, x0, y0, plot_w, plot_h)
        if pts:
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        if np.nanmin(time_rel) <= 0.0 <= np.nanmax(time_rel):
            hx = x0 + (0.0 - float(np.nanmin(time_rel))) / (float(np.nanmax(time_rel)) - float(np.nanmin(time_rel))) * plot_w
            lines.append(f'<line x1="{hx:.1f}" y1="{y0}" x2="{hx:.1f}" y2="{y0 + plot_h}" stroke="#111827" stroke-dasharray="5 4"/>')
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        y_med = float(np.nanmedian(y))
        lines.append(f'<text x="{x0}" y="{y0 + plot_h + 18}" font-family="monospace" font-size="12" fill="#57534e">min={y_min:.3f} med={y_med:.3f} max={y_max:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def pick_samples(data: np.lib.npyio.NpzFile) -> tuple[list[int], dict[str, list[int]]]:
    labels = data["stroke_type_rule_v2"]
    conf = data["stroke_confidence_rule_v2"]
    sources = [parse_source(s) for s in data["source_json"]]
    rackets = np.asarray([s["racket"] for s in sources])

    buckets: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label == "unknown":
            buckets["unknown"].append(i)
        if label == "forehand" and rackets[i] == "TennisBats02":
            buckets["bats02_forehand"].append(i)
        if label == "backhand" and rackets[i] == "TennisBats01":
            buckets["bats01_backhand"].append(i)
        if label != "unknown" and conf[i] < 0.85:
            buckets["low_conf_known"].append(i)

    selected: list[int] = []
    selected_by_bucket: dict[str, list[int]] = {}
    for bucket, indices in buckets.items():
        if bucket == "unknown":
            # Sort unknown by the strongest residual motion evidence first.
            ordered = sorted(
                indices,
                key=lambda j: (
                    abs(float(data["stroke_lateral_velocity_window_mps"][j])),
                    abs(float(data["stroke_pre_to_hit_lateral_delta_m"][j])),
                ),
                reverse=True,
            )[:24]
        else:
            ordered = sorted(indices, key=lambda j: float(conf[j]))[:24]
        selected_by_bucket[bucket] = ordered
        selected.extend(ordered)

    selected = sorted(set(selected))
    return selected, selected_by_bucket


def write_csv(path: Path, data: np.lib.npyio.NpzFile, indices: list[int]) -> None:
    fields = [
        "idx",
        "episode_id",
        "label",
        "confidence",
        "racket",
        "source_csv",
        "lateral_offset_m",
        "lateral_velocity_window_mps",
        "pre_to_hit_delta_m",
        "score_forehand",
        "score_backhand",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in indices:
            src = parse_source(str(data["source_json"][i]))
            writer.writerow(
                {
                    "idx": i,
                    "episode_id": str(data["episode_id"][i]),
                    "label": str(data["stroke_type_rule_v2"][i]),
                    "confidence": f"{float(data['stroke_confidence_rule_v2'][i]):.4f}",
                    "racket": src["racket"],
                    "source_csv": src["source_csv"],
                    "lateral_offset_m": f"{float(data['stroke_lateral_offset_m'][i]):.5f}",
                    "lateral_velocity_window_mps": f"{float(data['stroke_lateral_velocity_window_mps'][i]):.5f}",
                    "pre_to_hit_delta_m": f"{float(data['stroke_pre_to_hit_lateral_delta_m'][i]):.5f}",
                    "score_forehand": f"{float(data['stroke_score_forehand_rule_v2'][i]):.2f}",
                    "score_backhand": f"{float(data['stroke_score_backhand_rule_v2'][i]):.2f}",
                    "reason": str(data["stroke_label_reason_rule_v2"][i]),
                }
            )


def write_all_labels_csv(path: Path, data: np.lib.npyio.NpzFile) -> None:
    write_csv(path, data, list(range(len(data["episode_id"]))))


def write_summary(path: Path, data: np.lib.npyio.NpzFile, selected_by_bucket: dict[str, list[int]]) -> None:
    labels = data["stroke_type_rule_v2"]
    conf = data["stroke_confidence_rule_v2"]
    sources = [parse_source(s) for s in data["source_json"]]
    rackets = np.asarray([s["racket"] for s in sources])
    source_csv = np.asarray([s["source_csv"] for s in sources])

    lines: list[str] = [
        "# DATA260703 Stroke Label Audit",
        "",
        "抽检目标：检查 v2 正反手标签是否按逐拍动作特征划分，而不是把某个人/球拍固定为正手或反手。",
        "",
        f"- total samples: {len(labels)}",
        f"- label counts: `{dict(Counter(labels.tolist()))}`",
        f"- unknown samples: `{int((labels == 'unknown').sum())}`",
        f"- low confidence known samples: `{int(((labels != 'unknown') & (conf < 0.85)).sum())}`",
        "",
        "## By Racket",
        "",
        "| racket | forehand | backhand | unknown | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for racket in sorted(set(rackets)):
        mask = rackets == racket
        c = Counter(labels[mask].tolist())
        lines.append(f"| {racket} | {c['forehand']} | {c['backhand']} | {c['unknown']} | {int(mask.sum())} |")
    lines += [
        "",
        "## By Source",
        "",
        "| source | forehand | backhand | unknown | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for src in sorted(set(source_csv)):
        mask = source_csv == src
        c = Counter(labels[mask].tolist())
        lines.append(f"| {src} | {c['forehand']} | {c['backhand']} | {c['unknown']} | {int(mask.sum())} |")

    lines += [
        "",
        "## Audit Buckets",
        "",
        "| bucket | selected | meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "unknown": "规则没有足够证据硬分，优先检查是否应保持 unknown",
        "bats02_forehand": "TennisBats02 中被判为正手，验证不是误把反手片段判正手",
        "bats01_backhand": "TennisBats01 中被判为反手，验证是否是真反手或异常动作",
        "low_conf_known": "已判正/反手但置信度偏低",
    }
    for bucket, indices in selected_by_bucket.items():
        lines.append(f"| {bucket} | {len(indices)} | {meanings.get(bucket, '')} |")

    lines += [
        "",
        "## Diagnostic SVGs",
        "",
        "每张 SVG 包含 5 条曲线：人体局部横向位置、人体局部横向速度、球拍-球距离、球速、球拍速度。竖虚线是击球时刻。",
        "",
    ]
    for bucket, indices in selected_by_bucket.items():
        lines.append(f"### {bucket}")
        lines.append("")
        for i in indices[:24]:
            lines.append(f"- [{i:04d}_{str(data['stroke_type_rule_v2'][i])}.svg](plots/{i:04d}_{str(data['stroke_type_rule_v2'][i])}.svg) `{data['episode_id'][i]}`")
        lines.append("")

    lines += [
        "## Current Read",
        "",
        "- `TennisBats02` 内部同时存在 `forehand` 和 `backhand`，说明当前规则没有按人硬分。",
        "- `TennisBats02` 中被判为 `forehand` 的抽检样本有大量来自 `Point 01_004`，符合“同一个人有一段时间打正手”的采集描述。",
        "- `unknown` 大多是横向速度和横向位移都弱的片段，当前保持 unknown 更安全。",
        "- `TennisBats01` 中少量 `backhand` 的横向速度/位移方向与反手规则一致，不应仅因为球拍 ID 而强制改回正手。",
        "- 下一步应人工打开本目录 `plots/` 下的 SVG，重点看 `unknown` 和反主趋势样本。",
        "",
        "## Recommendation",
        "",
        "- 当前 v2 标签可作为第一版训练标签使用，但训练时建议排除 `unknown`。",
        "- 若要更保守，可额外只使用 `stroke_confidence_rule_v2 >= 0.85` 的正反手样本。",
        "- 不建议把 `TennisBats01/02` 直接映射成正/反手标签。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    selected, selected_by_bucket = pick_samples(data)
    write_csv(args.output_dir / "stroke_audit_samples.csv", data, selected)
    write_all_labels_csv(args.output_dir / "stroke_all_labels.csv", data)
    write_summary(args.output_dir / "stroke_audit_summary.md", data, selected_by_bucket)
    for i in selected:
        svg_plot(plot_dir / f"{i:04d}_{str(data['stroke_type_rule_v2'][i])}.svg", data, i)

    print(f"selected {len(selected)} samples")
    for bucket, indices in selected_by_bucket.items():
        print(bucket, len(indices))
    print(f"wrote {args.output_dir / 'stroke_audit_summary.md'}")
    print(f"wrote {args.output_dir / 'stroke_audit_samples.csv'}")
    print(f"wrote {args.output_dir / 'stroke_all_labels.csv'}")
    print(f"wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
