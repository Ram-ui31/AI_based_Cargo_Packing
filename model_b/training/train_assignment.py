"""PPO training for the assignment policy (Phase B), using the frozen Phase-A
placement policy as the feasibility/reward oracle.

Each instance-rollout is treated as a single structured action (log-prob =
sum over packages of their chosen-class log-prob); PPO's clipped ratio is
computed at that sequence level. Advantage = per-K EMA baseline of cost
minus this rollout's cost (lower cost is better), batch-normalized.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "common", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "rl_packer", "src"))

import numpy as np
import torch
import torch.optim as optim

from assignment_env import rollout, evaluate_assignment, feasibility_hinge_loss, soft_spread_loss
from assignment_policy import AssignmentPolicy, normalize_k
from data import load_split
from placement_policy import PlacementPolicy

CKPT_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "model_b", "assignment_policy.pt")
LOG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "logs", "model_b", "assignment_training_log.csv")
PLACEMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")
IL_CKPT_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "model_b", "assignment_policy_il.pt")


def load_frozen_placement_policy(hidden: int = 96):
    pp = PlacementPolicy(hidden=hidden)
    pp.load_state_dict(torch.load(PLACEMENT_CKPT))
    pp.eval()
    for p in pp.parameters():
        p.requires_grad_(False)
    return pp


def validate(policy: AssignmentPolicy, val_instances, placement_policy, n_episodes: int = 20,
             seed: int = 12345) -> tuple[float, int]:
    """Deterministic (fixed-seed, greedy-decode) mean cost on a held-out
    slice of synthetic_train that gradients never touch -- used purely for
    checkpoint selection / early stopping, never synthetic_test. Returns
    (mean_cost, total_priority_dropped)."""
    rng = random.Random(seed)
    was_training = policy.training
    policy.eval()
    costs, prio_dropped = [], 0
    with torch.no_grad():
        for _ in range(n_episodes):
            inst = rng.choice(val_instances)
            assignment, _, _, _, _ = rollout(policy, inst, greedy=True)
            result = evaluate_assignment(inst, assignment, placement_policy)
            costs.append(result["cost"])
            prio_dropped += len(result["priority_dropped"])
    if was_training:
        policy.train()
    return float(np.mean(costs)), prio_dropped


def train(n_updates: int = 100, episodes_per_update: int = 6, ppo_epochs: int = 4,
          clip_eps: float = 0.2, lr: float = 1e-4, entropy_coef: float = 0.01,
          seed: int = 0, n_train_instances: int = 100, n_val_instances: int = 40,
          val_every: int = 10, val_episodes: int = 20, patience: int = 10,
          log_every: int = 5, init_from_il: bool = True, resume_from: str | None = None,
          best_ckpt_path: str | None = None, hinge_coef: float = 0.1, hinge_margin: float = 1.0,
          spread_coef: float = 0.0):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    all_instances = load_split("train")
    instances = all_instances[:n_train_instances]
    val_instances = all_instances[n_train_instances:n_train_instances + n_val_instances]
    assert val_instances, "no instances left for validation slice -- reduce n_train_instances"
    placement_policy = load_frozen_placement_policy()

    policy = AssignmentPolicy()
    init_path = resume_from or (IL_CKPT_PATH if init_from_il else None)
    if init_path and os.path.exists(init_path):
        policy.load_state_dict(torch.load(init_path))
        print(f"initialized assignment policy from {init_path}")
    opt = optim.Adam(policy.parameters(), lr=lr)

    best_ckpt_path = best_ckpt_path or CKPT_PATH
    best_val_cost = float("inf")
    updates_since_improvement = 0
    stopped_early = False

    # Seed each K's baseline (mean AND std of cost) from real (greedy-decode)
    # rollouts of the starting policy before any gradient step -- an empty
    # baseline dict means the first rollout at each K sets baseline[K] = its
    # own cost (zero advantage, no learning signal), and the next few
    # rollouts at that K see a very noisy, low-sample-count EMA target. That
    # noise was found to destabilize training early (best checkpoint ended
    # up being ~update 10, right after init).
    baseline: dict[int, float] = {}
    baseline_std: dict[int, float] = {}
    k_values_seen = sorted({inst.K for inst in instances})
    print(f"warming up baseline for K values: {k_values_seen}")
    with torch.no_grad():
        for K in k_values_seen:
            k_instances = [inst for inst in instances if inst.K == K]
            warm_costs = []
            for inst in rng.sample(k_instances, min(8, len(k_instances))):
                assignment, _, _, _, _ = rollout(policy, inst, greedy=True)
                warm_costs.append(evaluate_assignment(inst, assignment, placement_policy)["cost"])
            baseline[K] = float(np.mean(warm_costs))
            baseline_std[K] = float(np.std(warm_costs)) if len(warm_costs) > 1 else max(baseline[K] * 0.1, 1.0)
            baseline_std[K] = max(baseline_std[K], 1.0)
            print(f"  K={K}: baseline={baseline[K]:.1f} +/- {baseline_std[K]:.1f} (from {len(warm_costs)} rollouts)")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "mean_cost", "mean_spread", "mean_delay_cost",
                          "priority_dropped_total", "policy_loss", "entropy", "hinge_loss", "spread_loss",
                          "val_cost", "val_prio_dropped", "best_val_cost"])
        t0 = time.time()

        for update in range(1, n_updates + 1):
            batch = []  # (instance, chosen_actions, old_logprob, advantage)
            costs, spreads, delays, prio_dropped = [], [], [], []

            for _ in range(episodes_per_update):
                inst = rng.choice(instances)
                with torch.no_grad():
                    assignment, old_lp, _, chosen, _ = rollout(policy, inst, greedy=False)
                result = evaluate_assignment(inst, assignment, placement_policy)
                cost = result["cost"]

                K = inst.K
                prev_baseline = baseline.get(K, cost)
                baseline[K] = cost if K not in baseline else 0.9 * prev_baseline + 0.1 * cost
                dev = cost - prev_baseline
                baseline_std[K] = (max(abs(dev), 1.0) if K not in baseline_std
                                   else max(0.9 * baseline_std[K] + 0.1 * abs(dev), 1.0))
                # per-K normalized advantage -- a raw (baseline - cost) would let K=5000's
                # naturally much larger cost swings dominate the gradient over K=100's,
                # since a single global batch-wide normalization can't undo that imbalance
                # (it was found to produce a policy that always minimizes spread regardless
                # of K, rather than a genuinely K-conditional spread-vs-delay tradeoff).
                advantage = (prev_baseline - cost) / baseline_std[K]

                batch.append((inst, chosen, old_lp.detach(), advantage))
                costs.append(cost)
                spreads.append(result["spread"])
                delays.append(result["delay_cost"])
                prio_dropped.append(len(result["priority_dropped"]))

            advs = np.array([b[3] for b in batch], dtype=np.float32)

            last_loss = last_entropy = last_hinge = last_spread_loss = 0.0
            for _ in range(ppo_epochs):
                losses = []
                entropies = []
                hinges = []
                spread_losses = []
                for (inst, chosen, old_lp, _), adv in zip(batch, advs):
                    _, new_lp, entropy, _, aux = rollout(policy, inst, greedy=False, replay_actions=chosen)
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * float(adv)
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * float(adv)
                    losses.append(-torch.min(surr1, surr2))
                    entropies.append(entropy)
                    # ground-truth-feasibility hinge loss -- see feasibility_hinge_loss()
                    # docstring: found the network's own static preference already rejects
                    # 30-46% of economy packages even when 100% dimensionally feasible
                    # somewhere, which the sparse per-rollout cost signal is too diffuse to
                    # fix quickly on its own. This is a direct, per-package, always-on
                    # supervised signal tied to ground truth (the dim-fit mask).
                    hinges.append(feasibility_hinge_loss(aux, margin=hinge_margin))
                    # differentiable K-scaled spread loss -- see soft_spread_loss() docstring:
                    # mirrors clusterer A's own dense spread_loss mechanism (which our sparse
                    # whole-rollout REINFORCE signal alone never had), computed straight from
                    # the network's own priority-package softmax, not the downstream cost.
                    if spread_coef:
                        spread_losses.append(normalize_k(inst.K) * soft_spread_loss(aux))
                loss = (torch.stack(losses).mean() - entropy_coef * torch.stack(entropies).mean()
                        + hinge_coef * torch.stack(hinges).mean())
                if spread_coef:
                    loss = loss + spread_coef * torch.stack(spread_losses).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                last_loss = float(loss.item())
                last_entropy = float(torch.stack(entropies).mean().item())
                last_hinge = float(torch.stack(hinges).mean().item())
                if spread_coef:
                    last_spread_loss = float(torch.stack(spread_losses).mean().item())

            val_cost, val_prio = "", ""
            if update % val_every == 0 or update == n_updates:
                val_cost, val_prio = validate(policy, val_instances, placement_policy, n_episodes=val_episodes)
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    updates_since_improvement = 0
                    torch.save(policy.state_dict(), best_ckpt_path)
                else:
                    updates_since_improvement += 1

            writer.writerow([update, np.mean(costs), np.mean(spreads), np.mean(delays),
                              sum(prio_dropped), last_loss, last_entropy, last_hinge, last_spread_loss,
                              val_cost, val_prio, best_val_cost])
            f.flush()

            if update % log_every == 0 or update == 1 or val_cost != "":
                elapsed = time.time() - t0
                val_str = f" | val_cost={val_cost:.1f} (best={best_val_cost:.1f}, prio_dropped={val_prio})" \
                    if val_cost != "" else ""
                print(f"update {update:4d} | cost={np.mean(costs):9.1f} | spread={np.mean(spreads):.2f} | "
                      f"delay={np.mean(delays):8.1f} | prio_dropped={sum(prio_dropped)} | "
                      f"loss={last_loss:.4f} | ent={last_entropy:.4f} | hinge={last_hinge:.4f} | "
                      f"spread_loss={last_spread_loss:.4f}{val_str} | {elapsed:.1f}s")

            if val_cost != "" and updates_since_improvement >= patience:
                print(f"early stop at update {update}: no val improvement for {patience} checks "
                      f"(best_val_cost={best_val_cost:.1f})")
                stopped_early = True
                break

    print(f"best checkpoint (val_cost={best_val_cost:.1f}) saved to {best_ckpt_path}"
          f"{' [stopped early]' if stopped_early else ''}")
    policy.load_state_dict(torch.load(best_ckpt_path))
    return policy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=6)
    p.add_argument("--n-train-instances", type=int, default=100)
    args = p.parse_args()
    train(n_updates=args.n_updates, episodes_per_update=args.episodes_per_update,
          n_train_instances=args.n_train_instances)
