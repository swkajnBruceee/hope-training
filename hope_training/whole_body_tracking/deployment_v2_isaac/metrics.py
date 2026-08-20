"""Metric-only accumulator; prediction truth must never enter actor observations."""
from dataclasses import dataclass

PRIVILEGED_METRIC_ONLY = True

@dataclass
class SmokeMetrics:
    total: int = 0; contract_valid: int = 0; contact: int = 0; cross_net: int = 0
    legal: int = 0; stable: int = 0; obs_finite: int = 0; planner_box: int = 0
    def rates(self):
        d = max(self.total, 1)
        return {k: getattr(self, k) / d for k in ("contract_valid", "contact", "cross_net", "legal", "stable", "obs_finite", "planner_box")}
