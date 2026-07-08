"""Supervised pretraining of the assignment policy: imitate a greedy
heuristic (imitation_baseline.greedy_assign) via masked cross-entropy.

This exists purely to give the PPO fine-tuning stage (train_assignment.py)
a sane starting point instead of random init -- pure whole-instance-reward
REINFORCE was found to converge to a degenerate "leave almost everything
behind" policy because credit assignment across hundreds of per-package
decisions from one scalar reward is extremely weak.
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

from assignment_env import build_inputs
from assignment_policy import AssignmentPolicy, MAX_N_PKGS, N_CLASSES, NONE_CLASS
from data import load_split
from imitation_baseline import greedy_assign

IL_CKPT_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "model_b", "assignment_policy_il.pt")


def build_batch_tensors(instances):
    uld_feats_l, pkg_feats_l, context_l, masks_l, kpm_l, targets_l = [], [], [], [], [], []
    for inst in instances:
        uld_feats, pkg_feats, context, masks, kpm, n_ulds, n_pkgs = build_inputs(inst)
        assignment = greedy_assign(inst.packages, inst.ulds)
        pkgs = inst.packages.iloc[:n_pkgs]
        target = np.full(MAX_N_PKGS, -100, dtype=np.int64)
        for pos, pid in enumerate(pkgs["Package_ID"]):
            a = assignment.get(pid)
            target[pos] = NONE_CLASS if a is None else a
        uld_feats_l.append(uld_feats)
        pkg_feats_l.append(pkg_feats)
        context_l.append(context)
        masks_l.append(masks)
        kpm_l.append(kpm)
        targets_l.append(target)
    return (
        torch.from_numpy(np.stack(uld_feats_l)),
        torch.from_numpy(np.stack(pkg_feats_l)),
        torch.from_numpy(np.stack(context_l)),
        torch.from_numpy(np.stack(masks_l)),
        torch.from_numpy(np.stack(kpm_l)),
        torch.from_numpy(np.stack(targets_l)),
    )


def train(n_epochs: int = 8, batch_size: int = 16, lr: float = 1e-3,
          n_train_instances: int = 300, seed: int = 0, log_every: int = 1):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    instances = load_split("train")[:n_train_instances]
    policy = AssignmentPolicy()
    opt = optim.Adam(policy.parameters(), lr=lr)

    n_batches = (len(instances) + batch_size - 1) // batch_size
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        perm = rng.permutation(len(instances))
        total_loss, total_correct, total_count = 0.0, 0, 0
        total_prio_correct, total_prio_count = 0, 0

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch_instances = [instances[i] for i in idx]
            uld_feats, pkg_feats, context, masks, kpm, targets = build_batch_tensors(batch_instances)

            logits = policy(uld_feats, pkg_feats, context, kpm)
            logits = logits.masked_fill(~masks, -1e9)
            loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), targets.reshape(-1), ignore_index=-100)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(-1)
                valid = targets != -100
                total_correct += int((pred[valid] == targets[valid]).sum().item())
                total_count += int(valid.sum().item())
                prio_valid = valid & (targets != NONE_CLASS)
                total_prio_correct += int((pred[prio_valid] == targets[prio_valid]).sum().item())
                total_prio_count += int(prio_valid.sum().item())
            total_loss += float(loss.item()) * len(batch_instances)

        if epoch % log_every == 0 or epoch == 1:
            acc = total_correct / total_count if total_count else 0.0
            prio_acc = total_prio_correct / total_prio_count if total_prio_count else 0.0
            elapsed = time.time() - t0
            print(f"epoch {epoch:3d} | loss={total_loss/len(instances):.4f} | acc={acc:.4f} | "
                  f"non_none_acc={prio_acc:.4f} | {elapsed:.1f}s")

    torch.save(policy.state_dict(), IL_CKPT_PATH)
    print(f"saved checkpoint to {IL_CKPT_PATH}")
    return policy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-train-instances", type=int, default=300)
    args = p.parse_args()
    train(n_epochs=args.n_epochs, batch_size=args.batch_size, n_train_instances=args.n_train_instances)
