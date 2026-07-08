"""IL pretraining for the placement policy (Phase A): imitate py3dbp's
placement decisions (which package, which extreme point, which orientation)
via cross-entropy over our own candidate-scoring network -- mirroring the
same IL-then-RL fix that worked for Phase B, since a direct comparison
showed py3dbp achieving much lower cost than our RL-only placement policy.

Also trains on realistic package-set sizes (30-90 packages per ULD) instead
of the smaller random subsets used in the original pure-RL run, to close
the train/eval distribution gap (real cluster-assigned loads are bigger).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "common", "src"))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from data import load_split
from placement_env import PlacementEnv
from placement_policy import PlacementPolicy, featurize_candidates
from py3dbp_teacher import run_py3dbp

IL_CKPT_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy_il.pt")


def collect_examples(uld_row, packages_df, max_pivots: int = 400):
    """Replay py3dbp's placement sequence through our own Heightmap. Since
    our pivot-point candidate generation mirrors py3dbp's own method, its
    exact (package, dims, x, y, z) choice should usually be directly among
    our candidates -- record it as a supervised (features, target_index)
    example. Falls back to package+dims-only matching (closest position) if
    an exact match is missing, and to a raw direct placement (skipping
    supervision) if not even that is achievable, to keep heightmap state in
    sync with the teacher's sequence."""
    steps, _unfit = run_py3dbp(uld_row, packages_df)
    env = PlacementEnv(uld_row, packages_df, max_pivots=max_pivots)
    examples = []
    n_matched = 0

    for step in steps:
        if env.done:
            break
        cands = env.candidates()
        if not cands:
            env.close_out()
            break
        pool_ids = env.pool["Package_ID"]
        same_pkg_dims = [
            (ci, c) for ci, c in enumerate(cands)
            if pool_ids.loc[c.pool_idx] == step.package_id and c.dx == step.dx
            and c.dy == step.dy and c.dz == step.dz
        ]
        exact = [(ci, c) for ci, c in same_pkg_dims if c.x == step.x and c.y == step.y and c.z == step.z]

        matches = exact or same_pkg_dims
        if matches:
            target_idx = min(matches, key=lambda ic: (ic[1].x - step.x) ** 2 + (ic[1].y - step.y) ** 2
                              + (ic[1].z - step.z) ** 2)[0]
            feats = featurize_candidates(cands, env.hm)
            examples.append((feats, target_idx))
            env.step(cands[target_idx])
            n_matched += 1
        else:
            match_rows = env.pool[env.pool["Package_ID"] == step.package_id]
            if len(match_rows) == 0:
                continue
            pool_idx = match_rows.index[0]
            weight = float(match_rows.iloc[0]["Weight"])
            if env.hm.fits(step.dx, step.dy, step.dz, step.x, step.y, step.z, weight):
                env.hm.place(step.package_id, step.dx, step.dy, step.dz, step.x, step.y, step.z, weight)
                env.placed_ids.append(step.package_id)
                env.pool = env.pool.drop(pool_idx)
                if len(env.pool) == 0:
                    env.done = True

    return examples, len(steps), n_matched


def sample_realistic_episode(instances, rng: np.random.Generator, min_n=30, max_n=90):
    inst = instances[rng.integers(len(instances))]
    uld_row = inst.ulds.sample(n=1, random_state=int(rng.integers(1 << 30))).iloc[0]
    n = int(rng.integers(min_n, max_n + 1))
    n = min(n, len(inst.packages))
    pkgs = inst.packages.sample(n=n, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    return uld_row, pkgs


def train(n_episodes: int = 400, batch_size: int = 64, lr: float = 1e-3, hidden: int = 96,
          seed: int = 0, n_train_instances: int = 200, log_every: int = 20, max_pivots: int = 400,
          resume_from: str | None = None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    instances = load_split("train")[:n_train_instances]

    policy = PlacementPolicy(hidden=hidden)
    if resume_from and os.path.exists(resume_from):
        policy.load_state_dict(torch.load(resume_from))
        print(f"resumed from {resume_from}")
    opt = optim.Adam(policy.parameters(), lr=lr)

    def flush(buffer):
        losses, correct = [], 0
        for feats, target in buffer:
            logits = policy.logits(torch.from_numpy(feats))
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])))
            correct += int(torch.argmax(logits).item() == target)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item()), correct / len(buffer)

    buffer, total_steps, total_matched = [], 0, 0
    t0 = time.time()
    last_loss = last_acc = 0.0

    for ep in range(1, n_episodes + 1):
        uld_row, pkgs = sample_realistic_episode(instances, rng)
        examples, n_steps, n_matched = collect_examples(uld_row, pkgs, max_pivots=max_pivots)
        total_steps += n_steps
        total_matched += n_matched
        buffer.extend(examples)

        if len(buffer) >= batch_size:
            last_loss, last_acc = flush(buffer)
            buffer = []

        if ep % log_every == 0 or ep == 1:
            elapsed = time.time() - t0
            match_rate = total_matched / total_steps if total_steps else 0.0
            print(f"episode {ep:4d} | loss={last_loss:.4f} | acc={last_acc:.4f} | "
                  f"match_rate={match_rate:.3f} | {elapsed:.1f}s")

    if buffer:
        last_loss, last_acc = flush(buffer)
        print(f"final flush | loss={last_loss:.4f} | acc={last_acc:.4f}")

    torch.save(policy.state_dict(), IL_CKPT_PATH)
    print(f"saved checkpoint to {IL_CKPT_PATH}")
    return policy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--n-train-instances", type=int, default=200)
    p.add_argument("--hidden", type=int, default=96)
    args = p.parse_args()
    train(n_episodes=args.n_episodes, n_train_instances=args.n_train_instances, hidden=args.hidden)
