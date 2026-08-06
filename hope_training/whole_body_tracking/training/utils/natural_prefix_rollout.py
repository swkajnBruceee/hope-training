"""Natural on-trajectory recovery sampling for V29-P0.

This module deliberately contains no snapshot restore or physical state
writeback. It only decides which fresh-reset transitions are eligible for
PPO and provides compact rollout storage that never writes masked prefix
transitions.
"""

from __future__ import annotations

from typing import Any

import torch
import gymnasium as gym

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage


def natural_recovery_rollout_mask(
    env: Any,
    *,
    phase_mode: str = "phase_balanced",
    min_post_hit_steps: int = 10,
    early_max_post_hit_steps: int = 30,
    mid_max_post_hit_steps: int = 80,
) -> torch.Tensor:
    """Return the phase-entry predicate for eligible V29-P0 transitions.

    The wrapper latches this predicate once true and keeps collecting until
    READY or a real environment termination.  Thus phase balancing selects a
    recovery start rather than truncating the recovery segment at the end of
    the early/mid bucket.
    """
    if phase_mode not in {"phase_balanced", "all_recovery"}:
        raise ValueError(f"Unknown natural recovery phase mode: {phase_mode!r}")
    motion = env.command_manager.get_term("motion")
    action = env.action_manager.get_term("joint_pos")
    if motion._use_motion_library:
        hit = motion.motion.hit_frame[motion.motion_ids].long()
    else:
        hit = torch.full_like(motion.time_steps, int(motion.motion.hit_frame[0]))
    post_hit = (motion.time_steps.long() - hit).clamp(min=0)
    in_recovery = motion.prelude_elapsed_steps >= int(motion.prelude_steps)
    not_ready = ~action._stage_a_rearm_ready
    if phase_mode == "all_recovery":
        return in_recovery & not_ready & (post_hit >= int(min_post_hit_steps))

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    bucket = env_ids.remainder(4)
    early = (
        (bucket == 0)
        & (post_hit >= int(min_post_hit_steps))
        & (post_hit <= int(early_max_post_hit_steps))
        & (motion.shot_cycle == 0)
    )
    mid = (
        (bucket == 1)
        & (post_hit > int(early_max_post_hit_steps))
        & (post_hit <= int(mid_max_post_hit_steps))
        & (motion.shot_cycle == 0)
    )
    near_ready = (
        (bucket == 2)
        & (motion.tail_steps > 0)
        & (motion.shot_cycle == 0)
    )
    multi_shot = (
        (bucket == 3)
        & (motion.shot_cycle > 0)
        & (post_hit >= int(min_post_hit_steps))
    )
    return in_recovery & not_ready & (early | mid | near_ready | multi_shot)


class NaturalPrefixRolloutWrapper(gym.Wrapper):
    """Gym-compatible wrapper that publishes a pre-action recovery mask."""

    def __init__(
        self,
        env,
        *,
        phase_mode: str = "phase_balanced",
        min_post_hit_steps: int = 10,
        early_max_post_hit_steps: int = 30,
        mid_max_post_hit_steps: int = 80,
    ):
        super().__init__(env)
        self.phase_mode = phase_mode
        self.min_post_hit_steps = min_post_hit_steps
        self.early_max_post_hit_steps = early_max_post_hit_steps
        self.mid_max_post_hit_steps = mid_max_post_hit_steps
        self._recovery_active = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        raw = self.env.unwrapped
        raw.natural_prefix_recovery_enabled = True
        raw.natural_recovery_action_mask = torch.zeros(
            raw.num_envs, dtype=torch.bool, device=raw.device
        )
        self._recovery_active = torch.zeros(
            raw.num_envs, dtype=torch.bool, device=raw.device
        )
        return result

    def step(self, action):
        raw = self.env.unwrapped
        entry_mask = natural_recovery_rollout_mask(
            raw,
            phase_mode=self.phase_mode,
            min_post_hit_steps=self.min_post_hit_steps,
            early_max_post_hit_steps=self.early_max_post_hit_steps,
            mid_max_post_hit_steps=self.mid_max_post_hit_steps,
        )
        if self._recovery_active is None:
            raise RuntimeError("NaturalPrefixRolloutWrapper.step called before reset")
        not_ready = ~raw.action_manager.get_term("joint_pos")._stage_a_rearm_ready
        mask = (self._recovery_active | entry_mask) & not_ready
        raw.natural_prefix_recovery_enabled = True
        raw.natural_recovery_action_mask = mask
        result = self.env.step(action)
        if len(result) != 5:
            raise RuntimeError("NaturalPrefixRolloutWrapper requires a five-return Gym step")
        observation, reward, terminated, truncated, info = result
        info = dict(info) if isinstance(info, dict) else {}
        next_not_ready = ~raw.action_manager.get_term("joint_pos")._stage_a_rearm_ready
        next_mask = mask & next_not_ready
        if isinstance(terminated, torch.Tensor):
            next_mask = next_mask & ~terminated.to(device=next_mask.device, dtype=torch.bool)
        if isinstance(truncated, torch.Tensor):
            next_mask = next_mask & ~truncated.to(device=next_mask.device, dtype=torch.bool)
        self._recovery_active = next_mask
        info["natural_recovery_rollout_mask"] = mask.detach().clone()
        info["natural_recovery_next_mask"] = next_mask.detach().clone()
        info["natural_recovery_rollout_count"] = mask.float().sum().detach().clone()
        return observation, reward, terminated, truncated, info


class NaturalPrefixRolloutStorage(RolloutStorage):
    """Compact storage indexed by per-environment accepted transitions only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def clear(self):
        super().clear()
        self.write_index.zero_()

    @property
    def valid_count(self) -> int:
        return int(self.write_index.sum().item())

    def add_masked_transitions(self, transition, mask: torch.Tensor) -> None:
        ids = torch.nonzero(mask.to(device=self.device, dtype=torch.bool), as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        slots = self.write_index[ids]
        if torch.any(slots >= self.num_transitions_per_env):
            raise OverflowError("Natural-prefix rollout storage is full")
        self.observations[slots, ids] = transition.observations[ids]
        if self.privileged_observations is not None:
            self.privileged_observations[slots, ids] = transition.privileged_observations[ids]
        self.actions[slots, ids] = transition.actions[ids]
        self.rewards[slots, ids] = transition.rewards[ids].view(-1, 1)
        self.dones[slots, ids] = transition.dones[ids].view(-1, 1).to(dtype=self.dones.dtype)
        if self.training_type == "rl":
            self.values[slots, ids] = transition.values[ids]
            self.actions_log_prob[slots, ids] = transition.actions_log_prob[ids].view(-1, 1)
            self.mu[slots, ids] = transition.action_mean[ids]
            self.sigma[slots, ids] = transition.action_sigma[ids]
        if self.rnd_state_shape is not None:
            self.rnd_state[slots, ids] = transition.rnd_state[ids]
        self.write_index[ids] += 1

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        self.advantages.zero_()
        self.returns.zero_()
        for env_id in range(self.num_envs):
            length = int(self.write_index[env_id].item())
            if length == 0:
                continue
            advantage = torch.zeros((1,), device=self.device)
            for step in reversed(range(length)):
                if step == length - 1:
                    next_values = last_values[env_id]
                else:
                    next_values = self.values[step + 1, env_id]
                next_is_not_terminal = 1.0 - self.dones[step, env_id].float()
                delta = (
                    self.rewards[step, env_id]
                    + next_is_not_terminal * gamma * next_values
                    - self.values[step, env_id]
                )
                advantage = delta + next_is_not_terminal * gamma * lam * advantage
                self.advantages[step, env_id] = advantage
                self.returns[step, env_id] = advantage + self.values[step, env_id]
        valid = torch.arange(self.num_transitions_per_env, device=self.device)[:, None] < self.write_index[None, :]
        if normalize_advantage and bool(valid.any().item()):
            values = self.advantages[valid]
            self.advantages[valid] = (values - values.mean()) / (values.std() + 1.0e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("Natural-prefix storage is only implemented for RL")
        valid = torch.arange(self.num_transitions_per_env, device=self.device)[:, None] < self.write_index[None, :]
        indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        total = int(indices.numel())
        if total == 0:
            raise RuntimeError("Natural-prefix PPO iteration collected zero recovery transitions")
        batches = min(int(num_mini_batches), total)
        observations = self.observations.flatten(0, 1)[indices]
        privileged = (
            self.privileged_observations.flatten(0, 1)[indices]
            if self.privileged_observations is not None
            else observations
        )
        actions = self.actions.flatten(0, 1)[indices]
        values = self.values.flatten(0, 1)[indices]
        returns = self.returns.flatten(0, 1)[indices]
        old_log_prob = self.actions_log_prob.flatten(0, 1)[indices]
        mu = self.mu.flatten(0, 1)[indices]
        sigma = self.sigma.flatten(0, 1)[indices]
        advantages = self.advantages.flatten(0, 1)[indices]
        for _ in range(num_epochs):
            order = torch.randperm(total, device=self.device)
            for batch in range(batches):
                start = batch * total // batches
                stop = (batch + 1) * total // batches
                batch_ids = order[start:stop]
                yield (
                    observations[batch_ids], privileged[batch_ids], actions[batch_ids],
                    values[batch_ids], advantages[batch_ids], returns[batch_ids],
                    old_log_prob[batch_ids], mu[batch_ids], sigma[batch_ids],
                    (None, None), None, None,
                )


class NaturalPrefixPPO(PPO):
    """PPO variant whose storage excludes all masked natural-prefix steps."""

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        self.storage = NaturalPrefixRolloutStorage(
            training_type, num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, actions_shape,
            None, self.device,
        )

    def process_env_step(self, rewards, dones, infos):
        mask = infos.get("natural_recovery_rollout_mask")
        if mask is None:
            raise RuntimeError("NaturalPrefixPPO requires natural_recovery_rollout_mask in env infos")
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        next_mask = infos.get("natural_recovery_next_mask")
        if next_mask is None:
            raise RuntimeError("NaturalPrefixPPO requires natural_recovery_next_mask in env infos")
        # READY closes the recovery segment.  Treat that boundary as an
        # artificial terminal for GAE so the compact storage never bootstraps
        # through a disabled/non-collected prefix or a later fresh trajectory.
        self.transition.dones = dones | ~next_mask.to(device=dones.device, dtype=torch.bool)
        self.storage.add_masked_transitions(self.transition, mask)
        self.transition.clear()
        self.policy.reset(dones)
