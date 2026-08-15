# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Continuous multi-rally swing lifecycle.

Per tick the runner advances one state machine that turns the latest
``RacketCommand`` and the robot state into the strike goal fed to the observation:

    ready -> swing -> follow-through -> recovery -> ready -> (next task_id)

Contract points enforced here:
  * A new ``task_id`` engages a swing (only from ready or recovery). ``swing_sign``
    is locked for the whole task. The sign shapes the ready reach only — it is
    NEVER part of the 110-D observation (deploy infers the side outside the policy).
  * A higher ``task_revision`` under the active ``task_id`` updates the target and
    time-to-strike, but only before contact.
  * There is exactly one swing per ``task_id``.
  * Between balls the robot pose, joint state, and policy history are NOT reset --
    the lifecycle never touches them, it only chooses what goal to observe.
  * Recovery is in-place recentring only (a fixed ready reach); it is never
    locomotion or footstep planning.

The reference clock: ``time_to_strike`` is seeded from the command and counts down
by ``dt`` each tick, passing through zero at contact and continuing negative through
the follow-through (matching the training reference clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .observation import ObsTarget, RobotState
from .racket_command import BACKHAND, FOREHAND, RacketCommand


class Phase(Enum):
    READY = "ready"
    SWING = "swing"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class LifecycleConfig:
    dt: float = 0.02
    follow_through_s: float = 0.6   # goal held after contact so the swing completes
    recovery_s: float = 0.8         # in-place recentring window before the next ball
    ready_time_to_strike: float = 1.0
    # Fallback ready geometry.  For model_21800 these values are replaced by the
    # exported hitter_pure metadata in configure_geometry().
    ready_reach_x: float = 0.40
    ready_reach_y: float = 0.20
    ready_reach_z: float = -0.05    # relative to the pelvis height
    fixed_world_station: bool = True
    station_step_max: float = 0.85
    station_x_max: float = 0.15
    station_ready_xy: float = 0.10
    station_ready_speed: float = 0.20
    station_ready_hold_s: float = 0.12
    # Online intercept predictors refine the same station by a few millimetres
    # every tick. Such revisions must not restart the settle dwell.
    station_revision_tolerance: float = 0.01
    velocity_gate_margin: float = 0.30


class SwingLifecycle:
    def __init__(self, cfg: LifecycleConfig | None = None) -> None:
        self.cfg = cfg or LifecycleConfig()
        self.phase = Phase.READY
        self.active_task_id: int | None = None
        self.swing_sign: int = FOREHAND         # last locked side (default forehand)
        self._target_pos_w = np.zeros(3)
        self._target_vel_w = np.zeros(3)
        self._tts = self.cfg.ready_time_to_strike
        self._follow_t = 0.0
        self._recover_t = 0.0
        # task_id of the most recently engaged ball; task_ids increase monotonically
        # (one ball, one increasing id), so we only ever engage a strictly newer id --
        # this enforces exactly one swing per task_id.
        self._last_engaged_task_id: int = -1
        # highest task_revision already applied to the active task (pre-contact only).
        self._applied_revision: int = -1
        self._station_xy: np.ndarray | None = None
        self._pending_station_xy: np.ndarray | None = None
        self._pending_cmd: RacketCommand | None = None
        self._station_ready_time = 0.0
        self._last_seen_task_id = -1
        self._engaged_tts = float(self._tts)
        self._pos_boxes = None
        self._vel_boxes = None
        self._reach_offsets = {
            FOREHAND: np.array([self.cfg.ready_reach_x, self.cfg.ready_reach_y]),
            BACKHAND: np.array([self.cfg.ready_reach_x, -self.cfg.ready_reach_y]),
        }
        self._ready_pos = {
            FOREHAND: None,
            BACKHAND: None,
        }
        self._ready_vel = {
            FOREHAND: np.zeros(3),
            BACKHAND: np.zeros(3),
        }

    @property
    def station_xy(self) -> np.ndarray | None:
        return None if self._station_xy is None else self._station_xy.copy()

    @property
    def pending_station_xy(self) -> np.ndarray | None:
        return None if self._pending_station_xy is None else self._pending_station_xy.copy()

    @property
    def just_engaged(self) -> bool:
        return self.phase == Phase.SWING and self._tts == self._engaged_tts

    def set_initial_station(self, base_xy: np.ndarray) -> None:
        if self._station_xy is None:
            self._station_xy = np.asarray(base_xy, dtype=np.float64).reshape(2).copy()

    def configure_geometry(
        self,
        pos_boxes=None,
        vel_boxes=None,
        reach_offsets=None,
        velocity_gate_margin: float | None = None,
    ) -> None:
        """Install model-specific hitter_pure geometry from ONNX metadata."""
        if pos_boxes and len(pos_boxes) >= 2:
            self._pos_boxes = tuple(tuple(float(x) for x in row) for row in pos_boxes)
            for sign, c in ((FOREHAND, 0), (BACKHAND, 1)):
                b = np.asarray(pos_boxes[c], dtype=np.float64)
                self._ready_pos[sign] = np.array(
                    [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
                )
                self._reach_offsets[sign] = self._ready_pos[sign][:2].copy()
        if vel_boxes and len(vel_boxes) >= 2:
            self._vel_boxes = tuple(tuple(float(x) for x in row) for row in vel_boxes)
            for sign, c in ((FOREHAND, 0), (BACKHAND, 1)):
                b = np.asarray(vel_boxes[c], dtype=np.float64)
                self._ready_vel[sign] = np.array(
                    [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
                )
        if reach_offsets and len(reach_offsets) >= 4:
            self._reach_offsets[FOREHAND] = np.asarray(reach_offsets[:2], dtype=np.float64)
            self._reach_offsets[BACKHAND] = np.asarray(reach_offsets[2:4], dtype=np.float64)
        if velocity_gate_margin is not None:
            self.cfg.velocity_gate_margin = float(velocity_gate_margin)

    # -- helpers ------------------------------------------------------------
    def _ready_target_pos_w(self, state: RobotState) -> np.ndarray:
        sign = FOREHAND if self.swing_sign >= 0 else BACKHAND
        base = self._station_xy if self._station_xy is not None else state.base_pos_w[:2]
        if self._ready_pos[sign] is not None:
            # Native pp_policy holds a fixed world station and reconstructs the
            # racket target as station + the side-specific trained reach.  The
            # target therefore stays fixed when the robot drifts, instead of
            # following the live pelvis.
            return np.array(
                [base[0] + self._reach_offsets[sign][0],
                 base[1] + self._reach_offsets[sign][1],
                 self._ready_pos[sign][2]],
                dtype=np.float64,
            )
        side = 1.0 if sign == FOREHAND else -1.0
        return np.asarray(
            [base[0] + self.cfg.ready_reach_x,
             base[1] + side * self.cfg.ready_reach_y,
             state.base_pos_w[2] + self.cfg.ready_reach_z],
            dtype=np.float64,
        )

    def _ready_target_vel_w(self) -> np.ndarray:
        sign = FOREHAND if self.swing_sign >= 0 else BACKHAND
        return self._ready_vel[sign].copy()

    def _candidate_station(self, cmd: RacketCommand) -> np.ndarray:
        sign = FOREHAND if cmd.swing_sign >= 0 else BACKHAND
        return np.asarray(cmd.position[:2], dtype=np.float64) - self._reach_offsets[sign]

    def _inside_geometry(self, cmd: RacketCommand) -> bool:
        sign = FOREHAND if cmd.swing_sign >= 0 else BACKHAND
        c = 0 if sign == FOREHAND else 1
        # Geometry gates are only active when the model exported its boxes.
        # This keeps older compatible actors usable while failing safely for the
        # model_21800 contract.
        pos_boxes = getattr(self, "_pos_boxes", None)
        vel_boxes = getattr(self, "_vel_boxes", None)
        if pos_boxes:
            p = np.asarray(pos_boxes[c], dtype=np.float64)
            m = 0.015
            if np.any(cmd.position < p[[0, 2, 4]] - m) or np.any(cmd.position > p[[1, 3, 5]] + m):
                return False
        if vel_boxes:
            v = np.asarray(vel_boxes[c], dtype=np.float64)
            m = self.cfg.velocity_gate_margin
            if np.any(cmd.velocity < v[[0, 2, 4]] - m) or np.any(cmd.velocity > v[[1, 3, 5]] + m):
                return False
        return True

    def _station_ready(self, state: RobotState) -> bool:
        if self._pending_station_xy is None:
            return True
        delta = self._pending_station_xy - np.asarray(state.base_pos_w[:2])
        speed = float(np.linalg.norm(np.asarray(state.base_lin_vel_w[:2])))
        ready = float(np.max(np.abs(delta))) <= self.cfg.station_ready_xy and speed <= self.cfg.station_ready_speed
        if ready:
            self._station_ready_time += self.cfg.dt
        else:
            self._station_ready_time = 0.0
        return self._station_ready_time >= self.cfg.station_ready_hold_s

    def _can_engage(self) -> bool:
        return self.phase in (Phase.READY, Phase.RECOVERY)

    # -- main tick ----------------------------------------------------------
    def update(self, cmd: RacketCommand | None, state: RobotState) -> ObsTarget:
        c = self.cfg

        # 1) Engage a new ball, or refine the active one before contact.
        if cmd is not None:
            if cmd.task_id > self._last_seen_task_id and self._can_engage():
                if self._inside_geometry(cmd):
                    candidate = self._candidate_station(cmd)
                    current = self._station_xy
                    step = 0.0 if current is None else float(np.linalg.norm(candidate - current))
                    x_step = 0.0 if current is None else abs(float(candidate[0] - current[0]))
                    if step <= c.station_step_max and x_step <= c.station_x_max:
                        # Do not consume a task id until its command has passed
                        # the model-geometry and station-step gates.  A live
                        # planner refines the same ball over several ticks; if
                        # the first prediction is temporarily OOD, the next
                        # refinement must still be allowed to engage it.
                        self._last_seen_task_id = cmd.task_id
                        self._pending_cmd = cmd
                        self._pending_station_xy = candidate
                        self._station_ready_time = 0.0
            elif (
                cmd.task_id == self._last_seen_task_id
                and self._pending_cmd is not None
                and self._can_engage()
                and cmd.task_revision > self._pending_cmd.task_revision
                and self._inside_geometry(cmd)
            ):
                # Keep a pending, not-yet-engaged ball synchronized with newer
                # planner revisions while preserving the same task identity.
                candidate = self._candidate_station(cmd)
                current = self._station_xy
                step = 0.0 if current is None else float(np.linalg.norm(candidate - current))
                x_step = 0.0 if current is None else abs(float(candidate[0] - current[0]))
                if step <= c.station_step_max and x_step <= c.station_x_max:
                    self._pending_cmd = cmd
                    # Planner revisions normally update position/velocity while
                    # keeping the same world station.  Do not restart the dwell
                    # timer on every such revision, otherwise a continuously
                    # streaming command can never become engageable.
                    previous_station = self._pending_station_xy
                    self._pending_station_xy = candidate
                    if (
                        previous_station is None
                        or float(np.linalg.norm(candidate - previous_station))
                        > c.station_revision_tolerance
                    ):
                        self._station_ready_time = 0.0
            elif (
                cmd.task_id == self.active_task_id
                and self.phase == Phase.SWING
                and self._tts > 0.0
                and cmd.task_revision > self._applied_revision
            ):
                # Pre-contact revision (must be newer): update where/when, never the
                # locked side.
                self._applied_revision = cmd.task_revision
                self._target_pos_w = np.asarray(cmd.position, dtype=np.float64).copy()
                self._target_vel_w = np.asarray(cmd.velocity, dtype=np.float64).copy()
                self._tts = float(cmd.time_to_strike)

        # A planner target is accepted only after the station has settled.  This
        # is the native move-to-station -> dwell -> strike lifecycle; it avoids
        # releasing a swing while the actor still sees an OOD base error.
        if self._pending_cmd is not None and self._can_engage() and self._station_ready(state):
            cmd = self._pending_cmd
            self._station_xy = self._pending_station_xy.copy()
            self._pending_cmd = None
            self._pending_station_xy = None
            self.active_task_id = cmd.task_id
            self._last_engaged_task_id = cmd.task_id
            self._applied_revision = cmd.task_revision
            self.swing_sign = cmd.swing_sign
            self._target_pos_w = np.asarray(cmd.position, dtype=np.float64).copy()
            self._target_vel_w = np.asarray(cmd.velocity, dtype=np.float64).copy()
            self._tts = float(cmd.time_to_strike)
            self._engaged_tts = self._tts
            self.phase = Phase.SWING

        # 2) Advance the reference clock and step the phase machine.
        if self.phase == Phase.SWING:
            self._tts -= c.dt
            if self._tts <= 0.0:
                self.phase = Phase.FOLLOW_THROUGH
                self._follow_t = 0.0
        elif self.phase == Phase.FOLLOW_THROUGH:
            self._tts -= c.dt
            self._follow_t += c.dt
            if self._follow_t >= c.follow_through_s:
                self.phase = Phase.RECOVERY
                self._recover_t = 0.0
        elif self.phase == Phase.RECOVERY:
            self._recover_t += c.dt
            if self._recover_t >= c.recovery_s:
                self.phase = Phase.READY
                self.active_task_id = None

        # 3) Emit the goal to observe this tick (the locked side stays internal —
        #    it shapes the ready reach but never appears in the observation).
        if self.phase in (Phase.SWING, Phase.FOLLOW_THROUGH):
            return ObsTarget(
                pos_w=self._target_pos_w,
                vel_w=self._target_vel_w,
                time_to_strike=self._tts,
            )
        # READY / RECOVERY -> in-place ready reach, clock pinned.
        return ObsTarget(
            pos_w=self._ready_target_pos_w(state),
            vel_w=self._ready_target_vel_w(),
            time_to_strike=c.ready_time_to_strike,
        )
