"""Loader for Motive/OptiTrack CSV exports used by DATA260703."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from analysis.mocap_cleaning.schemas import EntityColumns, PoseSeries, RawTrial


HEADER_ROWS = 8


def _meta_value(meta: list[str], key: str, default: str = "") -> str:
    try:
        return meta[meta.index(key) + 1]
    except ValueError:
        return default


def read_motive_header(path: str | Path) -> list[list[str]]:
    path = Path(path)
    with path.open("r", errors="replace") as f:
        return [next(f).rstrip("\n\r").split(",") for _ in range(HEADER_ROWS)]


def parse_motive_metadata(header: list[list[str]]) -> dict[str, str]:
    meta = header[0]
    return {
        "format_version": _meta_value(meta, "Format Version"),
        "take_name": _meta_value(meta, "Take Name"),
        "capture_fps": _meta_value(meta, "Capture Frame Rate"),
        "export_fps": _meta_value(meta, "Export Frame Rate"),
        "capture_start_time": _meta_value(meta, "Capture Start Time"),
        "capture_start_frame": _meta_value(meta, "Capture Start Frame"),
        "total_frames": _meta_value(meta, "Total Frames in Take"),
        "total_exported_frames": _meta_value(meta, "Total Exported Frames"),
        "rotation_type": _meta_value(meta, "Rotation Type"),
        "length_units": _meta_value(meta, "Length Units"),
        "coordinate_space": _meta_value(meta, "Coordinate Space"),
    }


def list_entities(header: list[list[str]]) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = defaultdict(list)
    for kind, name in zip(header[2], header[3]):
        if kind in ("Rigid Body", "Bone", "Marker") and name and name not in entities[kind]:
            entities[kind].append(name)
    return dict(entities)


def find_entity_columns(header: list[list[str]], kind: str, name: str) -> EntityColumns | None:
    pos_axes: dict[str, int] = {}
    quat_axes: dict[str, int] = {}

    for idx, (col_kind, col_name, prop, axis) in enumerate(zip(header[2], header[3], header[6], header[7])):
        if col_kind != kind or col_name != name:
            continue
        if prop == "Position":
            pos_axes[axis] = idx
        elif prop == "Rotation":
            quat_axes[axis] = idx

    pos = None
    quat = None
    if {"X", "Y", "Z"} <= set(pos_axes):
        pos = (pos_axes["X"], pos_axes["Y"], pos_axes["Z"])
    if {"X", "Y", "Z", "W"} <= set(quat_axes):
        quat = (quat_axes["X"], quat_axes["Y"], quat_axes["Z"], quat_axes["W"])
    if pos is None and quat is None:
        return None
    return EntityColumns(kind=kind, name=name, pos=pos, quat_xyzw=quat)


def _read_selected_columns(path: Path, column_indexes: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    indexes = sorted(set(column_indexes))
    index_to_out = {idx: out_idx for out_idx, idx in enumerate(indexes)}
    time_values: list[float] = []
    data: list[list[float]] = []

    with path.open("r", errors="replace") as f:
        for _ in range(HEADER_ROWS):
            next(f)
        for line in f:
            row = line.rstrip("\n\r").split(",")
            if len(row) < 2:
                continue
            try:
                time_values.append(float(row[1]))
            except ValueError:
                continue

            values = [np.nan] * len(indexes)
            for idx in indexes:
                try:
                    values[index_to_out[idx]] = float(row[idx])
                except (ValueError, IndexError):
                    values[index_to_out[idx]] = np.nan
            data.append(values)

    return np.asarray(time_values, dtype=float), np.asarray(data, dtype=float)


def load_motive_csv(
    path: str | Path,
    rigid_bodies: Iterable[str] = (),
    bones: Iterable[str] = (),
    markers: Iterable[str] = (),
) -> RawTrial:
    path = Path(path)
    header = read_motive_header(path)
    metadata = parse_motive_metadata(header)

    requested: list[EntityColumns] = []
    for name in rigid_bodies:
        cols = find_entity_columns(header, "Rigid Body", name)
        if cols is not None:
            requested.append(cols)
    for name in bones:
        cols = find_entity_columns(header, "Bone", name)
        if cols is not None:
            requested.append(cols)
    for name in markers:
        cols = find_entity_columns(header, "Marker", name)
        if cols is not None:
            requested.append(cols)

    selected_columns: list[int] = []
    for entity in requested:
        if entity.pos:
            selected_columns.extend(entity.pos)
        if entity.quat_xyzw:
            selected_columns.extend(entity.quat_xyzw)

    if selected_columns:
        time, raw_data = _read_selected_columns(path, selected_columns)
        sorted_columns = sorted(set(selected_columns))
        col_to_data = {col: raw_data[:, idx] for idx, col in enumerate(sorted_columns)}
    else:
        time = np.asarray([], dtype=float)
        col_to_data = {}

    rigid_out: dict[str, PoseSeries] = {}
    bone_out: dict[str, PoseSeries] = {}
    marker_out: dict[str, np.ndarray] = {}

    for entity in requested:
        pos = None
        quat = None
        if entity.pos:
            pos = np.stack([col_to_data[col] for col in entity.pos], axis=1)
        if entity.quat_xyzw:
            quat = np.stack([col_to_data[col] for col in entity.quat_xyzw], axis=1)
        if entity.kind == "Rigid Body":
            if pos is not None:
                rigid_out[entity.name] = PoseSeries(pos=pos, quat_xyzw=quat)
        elif entity.kind == "Bone":
            if pos is not None:
                bone_out[entity.name] = PoseSeries(pos=pos, quat_xyzw=quat)
        elif entity.kind == "Marker":
            if pos is not None:
                marker_out[entity.name] = pos

    return RawTrial(
        source_path=str(path),
        take_name=metadata["take_name"],
        fps=float(metadata["export_fps"]),
        time=time,
        position_unit=metadata["length_units"].lower(),
        coordinate_space=metadata["coordinate_space"].lower(),
        rigid_bodies=rigid_out,
        bones=bone_out,
        markers=marker_out,
        metadata={
            **metadata,
            "entities": list_entities(header),
            "quat_order": "xyzw",
            "header_rows": HEADER_ROWS,
        },
    )

