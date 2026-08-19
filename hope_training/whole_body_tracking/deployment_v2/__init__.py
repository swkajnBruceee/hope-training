"""Deployment-Aligned High-Level SAC V2 software-contract prototype."""

from .hope_open_source_contract import (
    FRAME_CONTRACT,
    POSITION_CONTRACT,
    ContractMetadata,
    load_canonical_metadata,
    map_normalized_velocity,
    select_nearest_station_side,
    velocity_inside_native_component_support,
    velocity_inside_planner_box,
)
from .schema2_adapter import (
    AdapterStationMirror,
    FlightRevisionManager,
    IdentityError,
    TimingError,
    age_command_timing,
    build_schema2_packet,
    reanchor_to_wall,
)

__all__ = [
    "FRAME_CONTRACT", "POSITION_CONTRACT", "ContractMetadata", "AdapterStationMirror",
    "FlightRevisionManager", "IdentityError", "TimingError",
    "age_command_timing", "build_schema2_packet", "load_canonical_metadata",
    "map_normalized_velocity", "reanchor_to_wall", "select_nearest_station_side",
    "velocity_inside_native_component_support", "velocity_inside_planner_box",
]
