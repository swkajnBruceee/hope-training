"""Pure deterministic V2-B2.1 search protocol; contains no learning."""
from __future__ import annotations
from itertools import product

TIER1_LEVELS = (-1.0, 0.0, 1.0)
TIER2_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def tier1_actions():
    """vx-major, then vy, then vz (vz varies fastest)."""
    return tuple(product(TIER1_LEVELS, repeat=3))


def full_grid_actions():
    return tuple(product(TIER2_LEVELS, repeat=3))


def tier2_additional_actions():
    already = set(tier1_actions())
    return tuple(action for action in full_grid_actions() if action not in already)


def first_legal_result(results):
    return next((result for result in results if bool(result["LEGAL_RETURN"])), None)


def negative_search_semantics(results) -> dict:
    found = first_legal_result(results) is not None
    return {
        "FEASIBILITY_WITNESS_FOUND": found,
        "NO_WITNESS_FOUND_WITHIN_SEARCH_PROTOCOL": not found,
        "ACTION_SPACE_INFEASIBLE": False,
    }


def repeatability_pass(results) -> bool:
    return bool(
        len(results) == 5
        and all(bool(x["LEGAL_RETURN"]) for x in results)
        and all(bool(x["SIMULATION_STABILITY"]) for x in results)
        and all(int(x["SIMULATION_NONFINITE_COUNT"]) == 0 for x in results)
    )

