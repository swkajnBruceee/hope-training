"""Table-tennis environment that adds ball aerodynamics on top of the manager-based RL env.

Everything except aerodynamics is handled by the standard :class:`~isaaclab.envs.ManagerBasedRLEnv`
machinery configured in :mod:`.table_tennis_env_cfg`. PhysX does not model air drag, so this subclass
registers a **physics-step callback** that, every physics substep, reads the ball velocity, computes the
aerodynamic wrench (:func:`.ball.compute_aero_wrench`) and writes it to the ball as an external force.

Using a physics callback (rather than overriding ``step()``) keeps this robust across Isaac Lab versions
and applies the force at the full physics rate (here 400 Hz) for accurate flight. If the callback cannot
be registered for any reason, the environment still runs — the ball just flies on PhysX gravity +
contacts alone, which is a valid (drag-free) scene.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_rotate_inverse

from . import geometry
from .ball import compute_aero_wrench
from .table_tennis_env_cfg import TableTennisEnvCfg


class BallTruthPointRos2Publisher:
    """Best-effort ROS 2 publisher for a single ball position truth stream."""

    def __init__(self, topic: str, frame_id: str):
        self._enabled = False
        self._owns_rclpy = False
        self._node = None
        self._publisher = None
        self._frame_id = frame_id

        if not topic:
            return

        try:
            import rclpy
            from geometry_msgs.msg import PointStamped
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            print(f"[TableTennisEnv] ROS 2 ball truth publisher disabled: {exc!r}")
            return

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self._rclpy = rclpy
        self._msg_type = PointStamped
        self._node = rclpy.create_node("hope_ball_truth_publisher")
        self._publisher = self._node.create_publisher(PointStamped, topic, 10)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def publish(self, position_w: torch.Tensor) -> None:
        if not self._enabled or self._node is None or self._publisher is None:
            return

        position = position_w.detach().to("cpu").tolist()
        msg = self._msg_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.point.x = float(position[0])
        msg.point.y = float(position[1])
        msg.point.z = float(position[2])
        self._publisher.publish(msg)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            self._publisher = None
        if self._owns_rclpy:
            try:
                getattr(self, "_rclpy").shutdown()
            except Exception:
                pass
        self._enabled = False


class TableTennisEnv(ManagerBasedRLEnv):
    """Manager-based table-tennis env with a per-substep ball aerodynamic force field."""

    cfg: TableTennisEnvCfg

    def __init__(self, cfg: TableTennisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._aero_active = False
        self._ball_truth_active = False
        self._ball_truth_publisher: BallTruthPointRos2Publisher | None = None
        self._ball_truth_env_index = 0
        self._setup_ball_aerodynamics()
        self._setup_ball_truth_publisher()

    def _setup_ball_aerodynamics(self) -> None:
        self._ball = self.scene["ball"]
        self._aero_cfg = self.cfg.ball_aerodynamics
        self._ball_mass = float(geometry.BALL_MASS)
        # Reusable zeroed external-wrench buffers: (num_envs, num_bodies=1, 3).
        self._aero_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._aero_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        if not self._aero_cfg.enabled:
            return
        try:
            # isaaclab.sim.SimulationContext inherits add_physics_callback from the Isaac Sim core
            # SimulationContext; the callback fires once per physics step with the step size.
            self.sim.add_physics_callback("hope_ball_aerodynamics", self._apply_ball_aerodynamics)
            self._aero_active = True
        except Exception as exc:  # pragma: no cover - defensive: never block the sim on aero setup
            import omni.log

            omni.log.warn(
                f"[TableTennisEnv] could not register the ball aerodynamics physics callback "
                f"({exc!r}); the ball will fly on PhysX gravity + contacts only."
            )

    def _setup_ball_truth_publisher(self) -> None:
        if not getattr(self.cfg, "publish_ball_truth", False):
            return

        topic = str(getattr(self.cfg, "ball_truth_topic", "")).strip()
        frame_id = str(getattr(self.cfg, "ball_truth_frame_id", "world"))
        env_index = int(getattr(self.cfg, "ball_truth_env_index", 0))

        if self.num_envs > 1:
            import omni.log

            omni.log.warn(
                f"[TableTennisEnv] publish_ball_truth is enabled with num_envs={self.num_envs}; "
                f"only env index {env_index} will be published on '{topic}'."
            )

        if env_index < 0 or env_index >= self.num_envs:
            import omni.log

            omni.log.warn(
                f"[TableTennisEnv] ball_truth_env_index={env_index} is out of range for num_envs={self.num_envs}; "
                "falling back to env 0."
            )
            env_index = 0

        self._ball_truth_env_index = env_index
        self._ball_truth_publisher = BallTruthPointRos2Publisher(topic, frame_id)
        if not self._ball_truth_publisher.enabled:
            return

        try:
            self.sim.add_physics_callback("hope_ball_truth", self._publish_ball_truth)
            self._ball_truth_active = True
        except Exception as exc:  # pragma: no cover - defensive: never block sim setup
            import omni.log

            omni.log.warn(
                f"[TableTennisEnv] could not register the ball truth physics callback "
                f"({exc!r}); the truth stream is disabled."
            )
            self._ball_truth_publisher.close()
            self._ball_truth_publisher = None
            self._ball_truth_active = False

    def _publish_ball_truth(self, _dt: float) -> None:
        """Publish one ball center sample for downstream trajectory prediction."""
        if not self._ball_truth_active or self._ball_truth_publisher is None:
            return

        env_id = self._ball_truth_env_index
        ball_pos_w = self._ball.data.root_pos_w[env_id].detach()
        ball_pos_hope = ball_pos_w - self.scene.env_origins[env_id]
        self._ball_truth_publisher.publish(ball_pos_hope)

    def _apply_ball_aerodynamics(self, dt: float) -> None:
        """Physics-step callback: apply the aerodynamic wrench to the ball (world frame -> body frame)."""
        lin_vel_w = self._ball.data.root_lin_vel_w
        ang_vel_w = self._ball.data.root_ang_vel_w
        force_w, torque_w = compute_aero_wrench(lin_vel_w, ang_vel_w, self._ball_mass, self._aero_cfg)

        # set_external_force_and_torque applies the wrench in the body frame; rotate the world-frame
        # wrench into the ball body frame so the net effect is the intended world-frame force.
        quat_w = self._ball.data.root_quat_w
        self._aero_force[:, 0, :] = quat_rotate_inverse(quat_w, force_w)
        self._aero_torque[:, 0, :] = quat_rotate_inverse(quat_w, torque_w)

        self._ball.set_external_force_and_torque(self._aero_force, self._aero_torque)
        self._ball.write_data_to_sim()

    def close(self) -> None:
        if self._ball_truth_publisher is not None:
            self._ball_truth_publisher.close()
            self._ball_truth_publisher = None
        super().close()

    # --------------------------------------------------------------------- #
    # Pre-reset diagnostic hook (read-only). Activated only when HOPE_DIAG_RESET=1.
    #
    # Goal: when an episode is about to be reset, dump the LAST pre-reset state
    # (ball pos / vel, racket pos, joint pos / vel, last termination flags,
    # episode length, last action stats). Critically, this fires BEFORE
    # super()._reset_idx(env_ids), which is what manager-based envs use to apply
    # mode="reset" events (reset_robot, reset_joints, reset_ball_serve). After
    # super() returns, ball/racket pose have already been rewritten by reset
    # events, so reading them from env.step() return would be too late.
    #
    # Defaults OFF so that train.py / play_table_tennis.py behavior is unchanged.
    # --------------------------------------------------------------------- #
    def _reset_idx(self, env_ids):
        """Override ManagerBasedRLEnv._reset_idx to add a pre-reset diagnostic snapshot."""
        # 1) Capture the pre-reset state BEFORE the manager applies any reset events.
        self._hope_diag_reset_hook(env_ids)
        # 2) Run the canonical manager reset (parent class behaviour is unchanged).
        super()._reset_idx(env_ids)

    def _hope_diag_reset_hook(self, env_ids):
        """Read-only pre-reset diagnostic. No writes, no manager state touched.

        Only effective when the env var HOPE_DIAG_RESET=1.
        Logs one line per env in env_ids, format designed for grep/awk and clean
        comparison across many episodes.
        """
        import os

        if os.environ.get("HOPE_DIAG_RESET", "0") != "1":
            return
        if env_ids is None:
            return
        # Lazy imports keep this off the warm-path when the env var is unset.
        from isaaclab.managers import SceneEntityCfg

        from . import mdp
        from .geometry import OutOfBoundsBox
        from .mdp.racket import racket_normal_w, racket_state_w

        device = self.device
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        n = int(env_ids.numel())
        if n == 0:
            return

        robot = self.scene["robot"]
        ball = self.scene["ball"]

        # ---- last action (best-effort; not fatal if missing) ----
        last_act = None
        try:
            am = self.action_manager
            # rsl_rl / isaaclab action_manager exposes `.prev_action` or `.last_action`
            src = getattr(am, "prev_action", None)
            if src is None:
                src = getattr(am, "last_action", None)
            if src is not None:
                last_act = src[env_ids].detach().clone()
        except Exception:
            last_act = None

        # ---- ball & racket snapshots (in HOPE frame, i.e. env-local) ----
        ball_pos_w = ball.data.root_pos_w[env_ids].detach().clone()
        ball_vel_w = ball.data.root_lin_vel_w[env_ids].detach().clone()
        ball_pos_hope = ball_pos_w - self.scene.env_origins[env_ids]

        r_pos_w_all, _, _ = racket_state_w(self, SceneEntityCfg("robot"))
        r_pos_w = r_pos_w_all[env_ids].detach().clone()
        normal_w_all = racket_normal_w(self, SceneEntityCfg("robot"), 1, 1.0)
        normal_w = normal_w_all[env_ids].detach().clone()

        rel = ball_pos_w - r_pos_w
        signed_n = (rel * normal_w).sum(-1)
        lateral = rel - signed_n.unsqueeze(-1) * normal_w
        lateral_d = torch.linalg.norm(lateral, dim=-1)
        normal_d = signed_n.abs()
        ball_to_rkt = torch.linalg.norm(rel, dim=-1)

        # ---- 7 right-arm joints ----
        joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        body_idxs = torch.as_tensor(
            [robot.find_joints(n_, preserve_order=True)[0][0] for n_ in joint_names],
            device=device,
            dtype=torch.long,
        )
        jp = robot.data.joint_pos[env_ids][:, body_idxs].detach().clone()
        jv = robot.data.joint_vel[env_ids][:, body_idxs].detach().clone()

        # ---- episode length / max episode length ----
        ep_buf = self.episode_length_buf[env_ids].detach().clone()
        max_ep_len = int(self.max_episode_length)

        # ---- common step counter (best effort) ----
        common_step_v = None
        csc = getattr(self, "common_step_counter", None)
        if csc is not None:
            try:
                common_step_v = int(csc.item()) if torch.is_tensor(csc) else int(csc)
            except Exception:
                common_step_v = None

        # ---- last-value flags from manager (these reflect the most recent step's
        #     `_apply_terminations` write, which is the pre-reset condition). ----
        last_flags = {}
        try:
            terms = self.termination_manager._terms
            for k in terms.keys():
                v = getattr(terms[k], "last_value", None)
                last_flags[k] = v[env_ids].clone() if v is not None else None
        except Exception:
            pass

        # ---- redundant one-shot re-evaluation at this instant, using the
        #     SAME functions and SAME thresholds as the manager. Read-only. ----
        robot_cfg = SceneEntityCfg("robot")
        ball_cfg = SceneEntityCfg("ball")
        bounds = OutOfBoundsBox().as_dict()
        # NOTE: termination helper signatures vary a bit across helper versions
        # (kwargs differ). Try the standard kwarg names first, fall back if needed.
        try:
            face_now = mdp.terminations.ball_close_to_racket_face(
                self, lateral_threshold=0.10, normal_threshold=0.10,
                robot_cfg=robot_cfg, ball_cfg=ball_cfg,
                normal_axis=1, normal_sign=1.0,
            )[env_ids]
        except TypeError:
            face_now = mdp.terminations.ball_close_to_racket_face(
                self, 0.10, 0.10, robot_cfg, ball_cfg, 1, 1.0,
            )[env_ids]

        oob_now = mdp.terminations.ball_out_of_bounds(
            self, bounds, ball_cfg,
        )[env_ids]

        try:
            scraped_now = mdp.terminations.racket_too_low_over_table(
                self, 0.01, blade_radius=0.085, table_margin=0.03, robot_cfg=robot_cfg,
            )[env_ids]
        except TypeError:
            scraped_now = mdp.terminations.racket_too_low_over_table(
                self, 0.01, 0.085, 0.03, robot_cfg,
            )[env_ids]

        time_out_now = ep_buf >= max_ep_len

        # ---- global reset counter ----
        if not hasattr(self, "_hope_diag_reset_count"):
            self._hope_diag_reset_count = 0
        self._hope_diag_reset_count += n

        # ---- emit one line per env ----
        for k in range(n):
            eid = int(env_ids[k].item())
            reset_g = int(self._hope_diag_reset_count - n + k + 1)
            parts = [
                "[HOPE-DIAG-RESET]",
                f"reset#{reset_g}",
                f"env={eid}",
                f"ep_len={int(ep_buf[k].item())}/{max_ep_len}",
            ]
            if common_step_v is not None:
                parts.append(f"g_step={common_step_v}")
            # reset_buf / terminated / truncated current values on the env (pre-reset)
            try:
                parts.append(f"reset_buf={int(self.reset_buf[eid].item())}")
            except Exception:
                pass
            try:
                parts.append(f"terminated={int(self.terminated[eid].item())}")
            except Exception:
                pass
            try:
                parts.append(f"truncated={int(self.truncated[eid].item())}")
            except Exception:
                pass

            # manager-side last flags (true == the manager saw this term as
            # true in the most recent step's _apply_terminations)
            for kk in ("time_out", "ball_out_of_bounds", "racket_scraped_table", "touch_success"):
                f = last_flags.get(kk)
                if f is not None:
                    parts.append(f"{kk}(mgr)={int(f[k].item())}")

            # redundant re-eval at this snapshot (ground truth)
            parts += [
                f"touch_success(now)={int(face_now[k].item())}",
                f"ball_oob(now)={int(oob_now[k].item())}",
                f"scraped(now)={int(scraped_now[k].item())}",
                f"time_out(now)={int(time_out_now[k].item())}",
                f"ball=({ball_pos_hope[k,0].item():+.3f},{ball_pos_hope[k,1].item():+.3f},{ball_pos_hope[k,2].item():+.3f})",
                f"|v|={ball_vel_w[k].norm().item():.2f}",
                f"v=({ball_vel_w[k,0].item():+.2f},{ball_vel_w[k,1].item():+.2f},{ball_vel_w[k,2].item():+.2f})",
                f"rkt=({r_pos_w[k,0].item():+.3f},{r_pos_w[k,1].item():+.3f},{r_pos_w[k,2].item():+.3f})",
                f"b2r={ball_to_rkt[k].item():.3f}",
                f"lat={lateral_d[k].item():.3f}",
                f"norm={normal_d[k].item():.3f}",
                "jp=" + ",".join(f"{x:+.2f}" for x in jp[k].tolist()),
                "jv=" + ",".join(f"{x:+.2f}" for x in jv[k].tolist()),
            ]
            if last_act is not None:
                la = last_act[k]
                parts += [
                    f"act_min={la.min().item():+.2f}",
                    f"act_max={la.max().item():+.2f}",
                    f"act_mean={la.mean().item():+.2f}",
                    f"act_std={la.std().item():+.2f}",
                ]
            print(" ".join(parts), flush=True)
