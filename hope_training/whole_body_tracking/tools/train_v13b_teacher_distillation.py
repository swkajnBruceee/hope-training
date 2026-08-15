#!/usr/bin/env python3
"""Offline teacher-to-student behavior distillation for V1.3B.

The rollout's ``student_action_26d`` is the authoritative public action after
the composed teacher/prior execution contract. The 31-DOF target is retained
for audit only: inverting it to 26 direct channels is not valid because the
prior and direct scales are not one-to-one (most recovered channels saturate).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


LOWER_SCALE = np.asarray((0.192, 0.048, 0.192, 0.192, 0.144, 0.192,
                          0.192, 0.048, 0.096, 0.072, 0.144, 0.192), np.float32)
UPPER_SCALE = np.asarray((0.44, 0.022, 0.11, 0.44, 0.0132, 0.11,
                          0.44, 0.44, 0.44, 0.44), np.float32)
# V1.3B right-front READY, in A3 backend/articulation joint order.
READY_LOWER = np.asarray((-0.1611863933, 0.1462128429, -0.0348, 0.48,
                          -0.3138136067, -0.0740128429,
                          -0.3187263393, -0.1462128429, 0.0348, 0.48,
                          -0.1562736607, 0.0740128429), np.float32)
READY_UPPER = np.asarray((0.0, 0.0, 0.0, 0.3, -0.12, 0.0, 0.8, 0.0, 0.0, 0.0), np.float32)
# 31-D backend indices: waist(0:3), head(3:5), right arm(12:19), legs(19:31).
UPPER_Q_INDEX = np.asarray((0, 1, 2, 12, 13, 14, 15, 16, 17, 18), np.int64)
LOWER_Q_INDEX = np.arange(19, 31, dtype=np.int64)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(98, 512), nn.ELU(),
                                 nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(),
                                 nn.Linear(128, 26))

    def forward(self, x):
        return torch.tanh(self.net(x))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=20260812)
    p.add_argument('--min-valid-steps', type=int, default=500,
                   help='Keep only complete/near-complete episodes; 0 keeps all valid frames.')
    p.add_argument('--obs-normalizer-checkpoint', type=Path, default=None,
                   help='PPO checkpoint whose frozen 98D observation normalizer is authoritative.')
    return p.parse_args()


def action_labels(q31: np.ndarray, public_action: np.ndarray) -> np.ndarray:
    """Return the authoritative 26-D action emitted in the PhysX rollout."""
    del q31
    # ActionManager applies this contract before the PhysX target is formed;
    # the rollout stores the actor output before that internal clamp.
    return np.clip(np.asarray(public_action, dtype=np.float32), -1.0, 1.0)


def collect_shards(dirs):
    paths = []
    for d in dirs:
        paths.extend(sorted(glob.glob(str(Path(d).expanduser() / 'teacher_rollout_*.npz'))))
    if not paths:
        raise FileNotFoundError('no teacher_rollout_*.npz found')
    return paths


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    paths = collect_shards(args.data_dir)
    # Every tenth shard is held out deterministically; rescue shards are train
    # augmentation and never form the validation split.
    train_paths = [p for i, p in enumerate(paths) if i % 10 != 0]
    val_paths = [p for i, p in enumerate(paths) if i % 10 == 0]
    model = Actor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    def load(p):
        z = np.load(p)
        valid = z['valid_mask']
        if int(args.min_valid_steps) > 0:
            # Distill complete teacher trajectories only.  Early-fall prefixes
            # are valuable for a recovery learner, but poison a pure
            # behavior-cloning target with post-reset distribution shift.
            keep_rows = valid.sum(axis=1) >= int(args.min_valid_steps)
            valid = valid & keep_rows[:, None]
        mask = valid.reshape(-1)
        x = z['observation_98d'].reshape(-1, 98)[mask].astype(np.float32)
        y = action_labels(z['teacher_joint_target_31d'].reshape(-1, 31)[mask],
                          z['student_action_26d'].reshape(-1, 26)[mask])
        return x, y

    # The PPO teacher's frozen normalizer is the authoritative 98D contract.
    # Falling back to dataset statistics is retained for standalone datasets.
    if args.obs_normalizer_checkpoint is not None:
        ck = torch.load(args.obs_normalizer_checkpoint.expanduser().resolve(), map_location='cpu', weights_only=False)
        ns = ck.get('obs_norm_state_dict')
        if ns is None or '_mean' not in ns or '_std' not in ns:
            raise RuntimeError('obs-normalizer-checkpoint has no _mean/_std 98D state')
        mean = np.asarray(ns['_mean']).reshape(-1).astype(np.float32)
        std = np.asarray(ns['_std']).reshape(-1).astype(np.float32)
        if mean.size != 98 or std.size != 98:
            raise RuntimeError(f'authoritative observation normalizer must be 98D, got {mean.size}/{std.size}')
        std = np.maximum(std, 1e-6)
    else:
        sums = np.zeros(98, np.float64); sq = np.zeros(98, np.float64); nobs = 0
        for p in train_paths:
            x, _ = load(p); sums += x.sum(0); sq += (x.astype(np.float64) ** 2).sum(0); nobs += len(x)
        mean = (sums / max(nobs, 1)).astype(np.float32)
        var = np.maximum(sq / max(nobs, 1) - mean.astype(np.float64) ** 2, 1e-6).astype(np.float32)
        std = np.sqrt(var).astype(np.float32)

    def evaluate():
        model.eval(); total = count = 0
        with torch.no_grad():
            for p in val_paths:
                x, y = load(p); x = (x - mean) / std
                if len(y) == 0:
                    continue
                pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
                total += float(np.abs(pred - y).mean()) * len(y); count += len(y)
        return total / max(count, 1)

    best = float('inf'); history = []
    for epoch in range(args.epochs):
        model.train(); order = np.random.permutation(len(train_paths)); running = 0.0; seen = 0
        for oi in order:
            x, y = load(train_paths[oi]); x = (x - mean) / std
            perm = np.random.permutation(len(x))
            for start in range(0, len(x), args.batch_size):
                ids = perm[start:start + args.batch_size]
                xb = torch.from_numpy(x[ids]).to(device); yb = torch.from_numpy(y[ids]).to(device)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                running += float(loss.item()) * len(ids); seen += len(ids)
        val = evaluate(); train = running / max(seen, 1); history.append({'epoch': epoch + 1, 'train_loss': train, 'val_l1': val})
        print(f'[distill] epoch={epoch+1}/{args.epochs} train_smooth_l1={train:.6f} val_l1={val:.6f}', flush=True)
        if val < best:
            best = val
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'model_state_dict': {'actor.' + str(i): v for i, v in []},
                'obs_mean': mean, 'obs_std': std,
                'source_data_dirs': [str(Path(d).resolve()) for d in args.data_dir],
                'teacher_contract': 'model_3396 + model_900 + model_5000 composed PhysX targets',
                'action_contract': 'V1.3B 98D -> 26D direct action',
                'history': history,
            }
            # Save a clean actor state separately; deployment/training adapters
            # can load it without pretending a critic or PPO optimizer exists.
            torch.save({'actor': model.state_dict(), 'obs_mean': mean, 'obs_std': std,
                        'history': history, 'teacher_contract': payload['teacher_contract']}, args.output)
    print(f'[distill] complete best_val_l1={best:.6f} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
