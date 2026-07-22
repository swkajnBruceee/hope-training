"""Versioned actor and asymmetric-critic observations for A3 Base Stand."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import ContactSensor


HISTORY_LENGTH = 10
PROPRIO_DIM = 87
CURRENT_CONTEXT_DIM = 85
ACTOR_OBSERVATION_DIM = HISTORY_LENGTH * PROPRIO_DIM + CURRENT_CONTEXT_DIM
PRIVILEGED_DIM = 45
CRITIC_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM + PRIVILEGED_DIM


class _A3BaseObservationBuilder:
    """Stateful history builder shared by the actor and critic observation terms."""

    def __init__(
        self,
        env,
        *,
        policy_joint_names: Sequence[str],
        torso_body_name: str,
        nominal_body_height_m: float,
    ):
        self._env = env
        self._asset: Articulation = env.scene["robot"]
        self._policy_joint_ids, resolved_joint_names = self._asset.find_joints(
            list(policy_joint_names), preserve_order=True
        )
        if resolved_joint_names != list(policy_joint_names):
            raise ValueError(
                "A3 policy observation joint order mismatch: "
                f"expected={list(policy_joint_names)}, resolved={resolved_joint_names}"
            )
        torso_ids, torso_names = self._asset.find_bodies([torso_body_name], preserve_order=True)
        if torso_names != [torso_body_name]:
            raise ValueError(f"Could not resolve A3 torso body {torso_body_name!r}: {torso_names}")
        self._torso_body_id = torso_ids[0]
        self._policy_joint_ids_tensor = torch.tensor(
            self._policy_joint_ids, dtype=torch.long, device=env.device
        )
        self._nominal_body_height_m = float(nominal_body_height_m)
        self._history = torch.zeros(
            (env.num_envs, HISTORY_LENGTH, PROPRIO_DIM), device=env.device
        )
        self._needs_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_episode_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._needs_reset[:] = True
            self._last_episode_step[:] = -1
        else:
            self._needs_reset[env_ids] = True
            self._last_episode_step[env_ids] = -1

    def _last_base_action(self) -> torch.Tensor:
        action_manager = getattr(self._env, "action_manager", None)
        if action_manager is None:
            return torch.zeros((self._env.num_envs, 14), device=self._env.device)
        try:
            action_term = action_manager.get_term("base")
        except (KeyError, AttributeError):
            return torch.zeros((self._env.num_envs, 14), device=self._env.device)
        # Observe the effective bounded command, not an arbitrarily large raw
        # network sample that the actuator/composer never receives.
        return action_term.raw_actions

    def _proprio_frame(self) -> torch.Tensor:
        data = self._asset.data
        torso_quat_w = data.body_quat_w[:, self._torso_body_id]
        torso_ang_vel_b = math_utils.quat_rotate_inverse(
            torso_quat_w, data.body_ang_vel_w[:, self._torso_body_id]
        )
        torso_projected_gravity = math_utils.quat_rotate_inverse(torso_quat_w, data.GRAVITY_VEC_W)
        joint_ids = self._policy_joint_ids_tensor
        joint_pos_rel = data.joint_pos[:, joint_ids] - data.default_joint_pos[:, joint_ids]
        joint_vel = data.joint_vel[:, joint_ids]
        frame = torch.cat(
            (
                # Linear velocity in the body frame is essential for a
                # receive-ready controller: it must detect a forward/lateral
                # fall tendency before a large tilt has already developed.
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                torso_ang_vel_b,
                torso_projected_gravity,
                joint_pos_rel,
                joint_vel,
                self._last_base_action(),
            ),
            dim=-1,
        )
        if frame.shape[-1] != PROPRIO_DIM:
            raise RuntimeError(f"A3 Base proprio dimension is {frame.shape[-1]}, expected {PROPRIO_DIM}")
        return frame

    def actor_observation(self) -> torch.Tensor:
        frame = self._proprio_frame()
        # ObservationManager probes term dimensions before ManagerBasedRLEnv
        # creates episode_length_buf.  Treat that probe as reset step zero.
        episode_step = getattr(self._env, "episode_length_buf", None)
        if episode_step is None:
            episode_step = torch.zeros(
                self._env.num_envs, dtype=torch.long, device=self._env.device
            )

        # At reset repeat current deployable proprioception through the entire
        # history but force the previous-action field to zero.  This avoids a
        # synthetic all-zero IMU/joint history while preventing action leakage.
        reset_ids = self._needs_reset.clone()
        if reset_ids.any():
            reset_frame = frame[reset_ids].clone()
            reset_frame[:, -14:] = 0.0
            self._history[reset_ids] = reset_frame.unsqueeze(1).repeat(1, HISTORY_LENGTH, 1)
            self._needs_reset[reset_ids] = False

        advance_ids = (~reset_ids) & (episode_step != self._last_episode_step)
        if advance_ids.any():
            self._history[advance_ids] = torch.roll(self._history[advance_ids], shifts=-1, dims=1)
            self._history[advance_ids, -1] = frame[advance_ids]
        self._last_episode_step[:] = episode_step

        command = torch.zeros((self._env.num_envs, 5), device=self._env.device)
        command[:, 3] = self._nominal_body_height_m
        # Stand v0 has no intervention, phase, time-to-hit, or future strike
        # reference.  Keeping their schema slots makes future task checkpoints
        # contract-compatible without exposing unavailable simulator truth.
        context = torch.cat(
            (
                command,
                torch.zeros((self._env.num_envs, 1 + 2 + 1 + 36 + 36 + 4), device=self._env.device),
            ),
            dim=-1,
        )
        actor = torch.cat((self._history.reshape(self._env.num_envs, -1), context), dim=-1)
        if actor.shape[-1] != ACTOR_OBSERVATION_DIM:
            raise RuntimeError(
                f"A3 Base actor observation dimension is {actor.shape[-1]}, expected {ACTOR_OBSERVATION_DIM}"
            )
        if not torch.isfinite(actor).all():
            raise RuntimeError("A3 Base actor observation contains NaN or infinity")
        return actor


class A3BaseActorObservation(ManagerTermBase):
    """925-D deployable actor observation with explicit reset history semantics."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._builder = _A3BaseObservationBuilder(env, **cfg.params)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._builder.reset(env_ids)

    def __call__(
        self,
        env,
        policy_joint_names: Sequence[str],
        torso_body_name: str,
        nominal_body_height_m: float,
    ) -> torch.Tensor:
        return self._builder.actor_observation()


class A3BaseCriticObservation(ManagerTermBase):
    """970-D asymmetric critic observation; privileged fields stay critic-only."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        builder_params = {
            key: cfg.params[key]
            for key in ("policy_joint_names", "torso_body_name", "nominal_body_height_m")
        }
        self._builder = _A3BaseObservationBuilder(env, **builder_params)
        self._contact_sensor_name = cfg.params["contact_sensor_name"]
        self._foot_body_names = list(cfg.params["foot_body_names"])
        self._sensor: ContactSensor = env.scene.sensors[self._contact_sensor_name]
        sensor_body_names = list(self._sensor.body_names)
        missing = [name for name in self._foot_body_names if name not in sensor_body_names]
        if missing:
            raise ValueError(f"Contact sensor does not contain A3 foot bodies {missing}")
        self._foot_sensor_ids = [sensor_body_names.index(name) for name in self._foot_body_names]
        foot_body_ids, resolved_foot_names = self._builder._asset.find_bodies(
            self._foot_body_names, preserve_order=True
        )
        if resolved_foot_names != self._foot_body_names:
            raise ValueError(
                f"A3 critic foot body order mismatch: expected={self._foot_body_names}, "
                f"resolved={resolved_foot_names}"
            )
        self._foot_body_ids = foot_body_ids

    def reset(self, env_ids: Sequence[int] | None = None):
        self._builder.reset(env_ids)

    def __call__(
        self,
        env,
        policy_joint_names: Sequence[str],
        torso_body_name: str,
        nominal_body_height_m: float,
        contact_sensor_name: str,
        foot_body_names: Sequence[str],
    ) -> torch.Tensor:
        actor = self._builder.actor_observation()
        data = self._builder._asset.data
        foot_forces = self._sensor.data.net_forces_w[:, self._foot_sensor_ids]
        foot_contact = (torch.linalg.vector_norm(foot_forces, dim=-1) > 1.0).to(actor.dtype)
        foot_slip_velocity = data.body_lin_vel_w[:, self._foot_body_ids]
        zeros_1 = torch.zeros((env.num_envs, 1), device=env.device)
        privileged = torch.cat(
            (
                data.root_lin_vel_b,
                foot_forces.reshape(env.num_envs, 6),
                foot_contact,
                foot_slip_velocity.reshape(env.num_envs, 6),
                torch.ones((env.num_envs, 1), device=env.device),  # ground friction
                torch.ones((env.num_envs, 1), device=env.device),  # robot mass scale
                torch.zeros((env.num_envs, 3), device=env.device),  # COM offset
                torch.ones((env.num_envs, 14), device=env.device),  # actuator strength
                torch.zeros((env.num_envs, 6), device=env.device),  # external wrench
                zeros_1,  # action delay
                zeros_1,  # state delay
                torch.ones((env.num_envs, 1), device=env.device),  # simulation dt scale
            ),
            dim=-1,
        )
        if privileged.shape[-1] != PRIVILEGED_DIM:
            raise RuntimeError(
                f"A3 Base privileged dimension is {privileged.shape[-1]}, expected {PRIVILEGED_DIM}"
            )
        critic = torch.cat((actor, privileged), dim=-1)
        if critic.shape[-1] != CRITIC_OBSERVATION_DIM:
            raise RuntimeError(
                f"A3 Base critic observation dimension is {critic.shape[-1]}, expected {CRITIC_OBSERVATION_DIM}"
            )
        if not torch.isfinite(critic).all():
            raise RuntimeError("A3 Base critic observation contains NaN or infinity")
        return critic
