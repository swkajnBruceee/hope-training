# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""HOPE-compatible 50 Hz MuJoCo control loop.

This is the project-side implementation of the deploy parts of HOPE's native
``main.cpp``/``pp_policy.hpp`` contract.  Hardware transport, ROS and AimRT are
intentionally outside this module; the state machine and command realization are
the same ones needed by the in-process MuJoCo verifier.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from . import quaternion as quat
from .config import RuntimeConfig
from .joint_order import HEAD_INDICES, NUM_JOINTS
from .lifecycle import Phase, SwingLifecycle
from .observation import build_observation
from .onnx_policy import OnnxPolicy
from .racket_command import RacketCommandSource
from .sim_bridge import SimBridge

_HEAD_IDX = list(HEAD_INDICES)
_WAIST_IDX = np.arange(0, 3)
_ARM_IDX = np.arange(5, 19)
_LEG_IDX = np.arange(19, 31)
_ANKLE_IDX = np.array([23, 24, 29, 30])


def _official_stand_gains() -> tuple[np.ndarray, np.ndarray]:
    """Return HOPE's a3_pd_stand gains in the 31-DOF SDK order."""
    kp = np.zeros(NUM_JOINTS, dtype=np.float64)
    kd = np.zeros(NUM_JOINTS, dtype=np.float64)
    kp[0:3] = [400.0, 500.0, 500.0]
    kd[0:3] = [4.0, 4.0, 4.0]
    kp[3:5] = [40.0, 40.0]
    kd[3:5] = [2.0, 2.0]
    kp[5:19] = [200, 200, 100, 200, 100, 50, 50] * 2
    kd[5:19] = [2, 2, 1, 1, 1, 1, 1] * 2
    kp[19:31] = [1500, 400, 300, 2000, 500, 500] * 2
    kd[19:31] = [8, 7, 7, 8, 5, 5] * 2
    return kp, kd


class PingPongReferenceRunner:
    """Run model_21800 with fixed-station lifecycle and native safety handoff."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        bridge: SimBridge,
        command_source: RacketCommandSource,
        policy: OnnxPolicy | None = None,
    ) -> None:
        self.cfg = cfg
        self.bridge = bridge
        self.source = command_source
        self.policy = policy or OnnxPolicy(cfg.onnx_path)
        self.lifecycle = SwingLifecycle(cfg.lifecycle)
        self.lifecycle.configure_geometry(
            getattr(self.policy, "hitter_pure_pos_boxes", None),
            getattr(self.policy, "hitter_pure_vel_boxes", None),
            getattr(self.policy, "reach_offsets", None),
            cfg.lifecycle.velocity_gate_margin,
        )

        self.default_q = cfg.action_adapter.default_q.copy()
        # Observations remain centered on the training default_q.  Deployed
        # neutral/hold targets include the fixed Curriculum-FT stance offset.
        self.stance_q = cfg.action_adapter.stance_q.copy()
        self.kp = cfg.sim_kp.copy()
        self.kd = cfg.sim_kd.copy()
        self.stand_kp, self.stand_kd = _official_stand_gains()
        if not cfg.official_stand:
            # Native pingpong's hoist-safe fallback when --official-stand is not
            # selected.  The project MuJoCo verifier defaults to official ground
            # gains because the model is evaluated on its feet.
            self.stand_kp = np.full(NUM_JOINTS, 60.0, dtype=np.float64)
            self.stand_kd = np.full(NUM_JOINTS, 4.0, dtype=np.float64)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.base_target_xy: np.ndarray | None = None
        self._last_q_des = self.stance_q.copy()
        self._blend_from_q: np.ndarray | None = None
        self._blend_tick = cfg.motion_blend_ticks
        self._had_swing = False
        self._was_active = False
        self._post_swing_s = 0.0
        self._fallen_ticks = 0
        self._safety_stopped = False

    def run(self, max_ticks: int | None = None, realtime: bool = False,
            status_every: int = 100) -> None:
        dt = self.cfg.control_dt
        self.bridge.reset()
        state = self.bridge.read_state()
        self.lifecycle.set_initial_station(state.base_pos_w[:2])
        self.base_target_xy = self.lifecycle.station_xy

        # Native main.cpp bypasses ONNX while the plant settles.  It uses the
        # production PD_STAND gains, then seeds the first policy target from the
        # measured posture for a continuous handoff.
        warmup = max(0, int(self.cfg.warmup_ticks))
        for warm_tick in range(warmup):
            self.bridge.write_targets(self.stance_q, self.stand_kp, self.stand_kd)
            self.bridge.step()
            self.bridge.sync_viewer()
            self.bridge.record_frame()
            if not self.bridge.is_viewer_running():
                return
        state = self.bridge.read_state()
        self._blend_from_q = state.q.copy()
        self._blend_tick = 0
        if warmup:
            print(f"[ref] PD_STAND warmup complete: {warmup} ticks ({warmup * dt:.2f}s)", file=sys.stderr)

        tick = 0
        try:
            while max_ticks is None or tick < max_ticks:
                loop_start = time.perf_counter()
                state = self.bridge.read_state()
                cmd = self.source.poll()
                previous_phase = self.lifecycle.phase
                target = self.lifecycle.update(cmd, state)
                active = self.lifecycle.phase in (Phase.SWING, Phase.FOLLOW_THROUGH)
                if active:
                    self._had_swing = True
                    self._post_swing_s = 0.0
                elif self._was_active and self.lifecycle.phase == Phase.READY:
                    self._post_swing_s = 0.0
                elif self._had_swing and self.lifecycle.pending_station_xy is None:
                    self._post_swing_s += dt
                self._was_active = active

                pending_station = self.lifecycle.pending_station_xy
                station = pending_station if pending_station is not None else self.lifecycle.station_xy
                if station is not None:
                    self.base_target_xy = np.asarray(station, dtype=np.float64)

                # Safety fallback after a persistent fall.  Native HOPE drops to
                # PASSIVE rather than continuing to slam stiff commands into the floor.
                grav = quat.projected_gravity_body(state.base_quat_w)
                if grav[2] > -0.5:
                    self._fallen_ticks += 1
                else:
                    self._fallen_ticks = 0
                if self._fallen_ticks >= 25:
                    self._safety_stopped = True

                static_hold = (
                    self._had_swing and not active and
                    self.lifecycle.pending_station_xy is None and
                    self._post_swing_s >= self.cfg.hold_recover_s
                )
                if self._safety_stopped:
                    q_des = state.q.copy()
                    kp, kd = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
                elif static_hold:
                    q_des = self.stance_q.copy()
                    kp, kd = self.stand_kp.copy(), self.stand_kd.copy()
                else:
                    obs = build_observation(
                        state, target, self.last_action, self.default_q, self.base_target_xy
                    )
                    infer_target = getattr(self.policy, "infer_target", None)
                    if callable(infer_target):
                        raw_action = infer_target(
                            obs, target.time_to_strike, self.lifecycle.swing_sign, dt
                        )
                    else:
                        raw_action = self.policy.infer(obs)
                    raw_action = np.nan_to_num(np.asarray(raw_action, dtype=np.float64), nan=0.0)
                    applied_action = raw_action.copy()
                    if self.cfg.passive_neck:
                        applied_action[_HEAD_IDX] = 0.0
                    self.last_action = np.clip(applied_action, -20.0, 20.0)
                    q_des = self.cfg.action_adapter.decode(self.last_action)
                    if self.cfg.passive_neck:
                        q_des[_HEAD_IDX] = self.default_q[_HEAD_IDX]

                    # Native runner's optional level-0 auto leg/waist hold.
                    hold_ground = self.cfg.auto_leg_hold and not active
                    if hold_ground:
                        q_des[_WAIST_IDX] = self.stance_q[_WAIST_IDX]
                        q_des[_LEG_IDX] = self.stance_q[_LEG_IDX]

                    if self.cfg.leg_clamp_rad > 0.0 and not hold_ground:
                        r = float(self.cfg.leg_clamp_rad)
                        q_des[_LEG_IDX] = np.clip(
                            q_des[_LEG_IDX], self.stance_q[_LEG_IDX] - r,
                            self.stance_q[_LEG_IDX] + r,
                        )

                    # Leg EMA is applied before the pose blend, matching the
                    # native low-pass that protects weight-bearing joints.
                    alpha = min(1.0, max(0.02, float(self.cfg.leg_smooth_alpha)))
                    if alpha < 1.0:
                        q_des[_LEG_IDX] = (
                            alpha * q_des[_LEG_IDX] + (1.0 - alpha) * self._last_q_des[_LEG_IDX]
                        )

                    # Policy gains are scaled by group; held ground joints use
                    # official stand gains exactly as in a3_pingpong_main.cpp.
                    kp, kd = self.kp.copy(), self.kd.copy()
                    kp[_WAIST_IDX] *= self.cfg.gain_scale
                    kd[_WAIST_IDX] *= self.cfg.gain_scale
                    kp[_ARM_IDX] *= self.cfg.gain_scale
                    kd[_ARM_IDX] *= self.cfg.gain_scale
                    kp[_LEG_IDX] *= self.cfg.leg_gain_scale
                    kd[_LEG_IDX] *= self.cfg.leg_gain_scale
                    kp[_ANKLE_IDX] *= self.cfg.ankle_gain_scale
                    kd[_ANKLE_IDX] *= self.cfg.ankle_gain_scale
                    if hold_ground:
                        kp[_WAIST_IDX], kd[_WAIST_IDX] = self.stand_kp[_WAIST_IDX], self.stand_kd[_WAIST_IDX]
                        kp[_LEG_IDX], kd[_LEG_IDX] = self.stand_kp[_LEG_IDX], self.stand_kd[_LEG_IDX]

                    # 0.5 s policy-entry blend prevents the official stand from
                    # snapping to the learned windup pose.
                    if self._blend_from_q is not None and self._blend_tick < self.cfg.motion_blend_ticks:
                        a = (self._blend_tick + 1) / max(1, self.cfg.motion_blend_ticks)
                        q_des = (1.0 - a) * self._blend_from_q + a * q_des
                        self._blend_tick += 1
                    else:
                        self._blend_from_q = None

                self._last_q_des = q_des.copy()
                self.bridge.write_targets(q_des, kp, kd)
                self.bridge.step()
                self.bridge.sync_viewer()
                self.bridge.record_frame()

                if not self.bridge.is_viewer_running():
                    break
                if status_every and tick % status_every == 0:
                    self._print_status(tick, target, static_hold, grav)
                tick += 1
                if realtime:
                    self._sleep_to_rate(loop_start, dt)
        finally:
            self.bridge.close()

    def _print_status(self, tick: int, target, static_hold: bool, grav: np.ndarray) -> None:
        side = "forehand" if self.lifecycle.swing_sign >= 0 else "backhand"
        station = self.lifecycle.station_xy
        station_s = "none" if station is None else f"({station[0]:+.2f},{station[1]:+.2f})"
        print(
            f"[ref] t={tick * self.cfg.control_dt:6.2f}s "
            f"phase={self.lifecycle.phase.value:<14} "
            f"task={self.lifecycle.active_task_id} side={side} "
            f"tts={target.time_to_strike:+.2f} station={station_s} "
            f"grav=({grav[0]:+.2f},{grav[1]:+.2f},{grav[2]:+.2f}) "
            f"static={static_hold} safety={self._safety_stopped}",
            file=sys.stderr,
        )

    @staticmethod
    def _sleep_to_rate(loop_start: float, dt: float) -> None:
        remaining = dt - (time.perf_counter() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
