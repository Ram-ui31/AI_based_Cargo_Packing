"""Custom PPO training for the single-ULD placement policy (Phase A).

Reward is per-step normalized volume placed (independent of K / delay cost,
per design) -- this policy just learns to pack a given box of packages into
a given ULD as densely as possible; it is reused, frozen, by the Phase B
assignment stage later.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "common", "src"))

import numpy as np
import torch
import torch.optim as optim

from data import load_split
from placement_env import PlacementEnv
from placement_policy import PlacementPolicy, featurize_candidates, featurize_global

CKPT_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")
LOG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "logs", "rl_packer", "placement_training_log.csv")


def compute_gae(rewards: list[float], values: list[float], gamma: float = 0.99, lam: float = 0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalue - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + np.array(values[:T], dtype=np.float32)
    return advantages, returns


def sample_episode_inputs(instances, rng: np.random.Generator, min_n: int = 30, max_n: int = 91):
    inst = instances[rng.integers(len(instances))]
    uld_row = inst.ulds.sample(n=1, random_state=int(rng.integers(1 << 30))).iloc[0]
    n = int(rng.integers(min_n, max_n))
    n = min(n, len(inst.packages))
    pkgs = inst.packages.sample(n=n, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    return uld_row, pkgs


def collect_episode(policy: PlacementPolicy, instances, rng: np.random.Generator, greedy: bool = False):
    uld_row, pkgs = sample_episode_inputs(instances, rng)
    env = PlacementEnv(uld_row, pkgs)

    cand_feats_list, global_feats_list, actions, old_logprobs, rewards, values = [], [], [], [], [], []
    while not env.done:
        cands = env.candidates()
        if not cands:
            env.close_out()
            break
        cf = featurize_candidates(cands, env.hm)
        gf = featurize_global(env.hm, env.pool)
        cf_t = torch.from_numpy(cf)
        gf_t = torch.from_numpy(gf)

        with torch.no_grad():
            logits = policy.logits(cf_t)
            dist = torch.distributions.Categorical(logits=logits)
            idx = int(torch.argmax(logits).item()) if greedy else int(dist.sample().item())
            logprob = dist.log_prob(torch.tensor(idx))
            value = policy.value(gf_t)

        r = env.step(cands[idx])

        cand_feats_list.append(cf)
        global_feats_list.append(gf)
        actions.append(idx)
        old_logprobs.append(float(logprob.item()))
        rewards.append(r)
        values.append(float(value.item()))

    return cand_feats_list, global_feats_list, actions, old_logprobs, rewards, values, env.hm.utilization()


def validate(policy: PlacementPolicy, val_instances, n_episodes: int = 20, seed: int = 12345) -> float:
    """Deterministic (fixed-seed) greedy-decode utilization on a held-out
    slice of synthetic_train that gradients never touch -- used purely for
    checkpoint selection / early stopping, never synthetic_test."""
    rng = np.random.default_rng(seed)
    was_training = policy.training
    policy.eval()
    utils_ = []
    with torch.no_grad():
        for _ in range(n_episodes):
            uld_row, pkgs = sample_episode_inputs(val_instances, rng)
            env = PlacementEnv(uld_row, pkgs)
            while not env.done:
                cands = env.candidates()
                if not cands:
                    env.close_out()
                    break
                idx, _, _ = policy.select(cands, env.hm, env.pool, greedy=True)
                env.step(cands[idx])
            utils_.append(env.hm.utilization())
    if was_training:
        policy.train()
    return float(np.mean(utils_))


def train(n_updates: int = 200, episodes_per_update: int = 8, ppo_epochs: int = 4,
          clip_eps: float = 0.2, lr: float = 3e-4, entropy_coef: float = 0.02,
          value_coef: float = 0.5, seed: int = 0, n_train_instances: int = 50,
          n_val_instances: int = 40, val_every: int = 10, val_episodes: int = 20,
          patience: int = 8, log_every: int = 5, hidden: int = 64, resume: bool = False,
          resume_from: str | None = None, best_ckpt_path: str | None = None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    all_instances = load_split("train")
    instances = all_instances[:n_train_instances]
    val_instances = all_instances[n_train_instances:n_train_instances + n_val_instances]
    assert val_instances, "no instances left for validation slice -- reduce n_train_instances"

    policy = PlacementPolicy(hidden=hidden)
    init_path = resume_from if resume_from else (CKPT_PATH if resume else None)
    if init_path and os.path.exists(init_path):
        policy.load_state_dict(torch.load(init_path))
        print(f"initialized from {init_path}")
    opt = optim.Adam(policy.parameters(), lr=lr)

    best_ckpt_path = best_ckpt_path or CKPT_PATH
    best_val_util = -1.0
    updates_since_improvement = 0
    stopped_early = False

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "mean_utilization", "mean_reward_sum", "policy_loss", "value_loss",
                          "entropy", "val_utilization", "best_val_utilization"])

        t0 = time.time()
        for update in range(1, n_updates + 1):
            batch_cf, batch_gf, batch_act, batch_oldlp, batch_ret, batch_adv = [], [], [], [], [], []
            utils_, reward_sums = [], []

            for _ in range(episodes_per_update):
                cf_list, gf_list, actions, old_lp, rewards, values, util = collect_episode(policy, instances, rng)
                if not rewards:
                    continue
                adv, ret = compute_gae(rewards, values)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
                batch_cf.extend(cf_list)
                batch_gf.extend(gf_list)
                batch_act.extend(actions)
                batch_oldlp.extend(old_lp)
                batch_ret.extend(ret.tolist())
                batch_adv.extend(adv.tolist())
                utils_.append(util)
                reward_sums.append(sum(rewards))

            if not batch_cf:
                continue

            gf_tensor = torch.from_numpy(np.stack(batch_gf))
            ret_tensor = torch.tensor(batch_ret, dtype=torch.float32)
            adv_tensor = torch.tensor(batch_adv, dtype=torch.float32)
            oldlp_tensor = torch.tensor(batch_oldlp, dtype=torch.float32)
            act_tensor = torch.tensor(batch_act, dtype=torch.int64)

            last_policy_loss = last_value_loss = last_entropy = 0.0
            for _ in range(ppo_epochs):
                new_logprobs, entropies = [], []
                for cf, a in zip(batch_cf, batch_act):
                    logits = policy.logits(torch.from_numpy(cf))
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprobs.append(dist.log_prob(torch.tensor(a)))
                    entropies.append(dist.entropy())
                new_lp = torch.stack(new_logprobs)
                entropy = torch.stack(entropies).mean()

                ratio = torch.exp(new_lp - oldlp_tensor)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_tensor
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = policy.value(gf_tensor)
                value_loss = ((values_pred - ret_tensor) ** 2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

                last_policy_loss, last_value_loss, last_entropy = (
                    float(policy_loss.item()), float(value_loss.item()), float(entropy.item())
                )

            mean_util = float(np.mean(utils_)) if utils_ else 0.0
            mean_rsum = float(np.mean(reward_sums)) if reward_sums else 0.0

            val_util = ""
            if update % val_every == 0 or update == n_updates:
                val_util = validate(policy, val_instances, n_episodes=val_episodes)
                if val_util > best_val_util:
                    best_val_util = val_util
                    updates_since_improvement = 0
                    torch.save(policy.state_dict(), best_ckpt_path)
                else:
                    updates_since_improvement += 1

            writer.writerow([update, mean_util, mean_rsum, last_policy_loss, last_value_loss,
                              last_entropy, val_util, best_val_util])
            f.flush()

            if update % log_every == 0 or update == 1 or val_util != "":
                elapsed = time.time() - t0
                val_str = f" | val_util={val_util:.3f} (best={best_val_util:.3f})" if val_util != "" else ""
                print(f"update {update:4d} | util={mean_util:.3f} | ret={mean_rsum:.3f} | "
                      f"pi_loss={last_policy_loss:.4f} | v_loss={last_value_loss:.4f} | "
                      f"ent={last_entropy:.4f}{val_str} | {elapsed:.1f}s")

            if val_util != "" and updates_since_improvement >= patience:
                print(f"early stop at update {update}: no val improvement for {patience} checks "
                      f"(best_val_util={best_val_util:.3f})")
                stopped_early = True
                break

    print(f"best checkpoint (val_util={best_val_util:.3f}) saved to {best_ckpt_path}"
          f"{' [stopped early]' if stopped_early else ''}")
    policy.load_state_dict(torch.load(best_ckpt_path))
    return policy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-updates", type=int, default=200)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--n-train-instances", type=int, default=50)
    p.add_argument("--n-val-instances", type=int, default=40)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--resume-from", type=str, default=None)
    args = p.parse_args()
    train(n_updates=args.n_updates, episodes_per_update=args.episodes_per_update,
          n_train_instances=args.n_train_instances, n_val_instances=args.n_val_instances,
          val_every=args.val_every, patience=args.patience, hidden=args.hidden, lr=args.lr,
          entropy_coef=args.entropy_coef, resume_from=args.resume_from)
