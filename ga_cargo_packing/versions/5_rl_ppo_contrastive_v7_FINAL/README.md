# Version 5: RL PPO Contrastive v7 — FINAL, deployed in production

**This is the checkpoint actually used everywhere downstream** — it's the
Priority Clusterer in the production pipeline and in the Eclipse, Halley,
and Cherry deliverable folders (`priority_clusterer.pt` in each of their
`checkpoints/` directories is a copy of this exact checkpoint).

PPO fine-tune from the Version 1 IL checkpoint, with an added contrastive
loss term that directly rewards the network for using K (the delay-cost
multiplier) as a causal control variable.

## Why the contrastive loss was necessary

This version supersedes an intermediate "multi-K" attempt: training on
data spanning many different K values. That removed a *correlational*
confound (K happening to line up with other instance features in the
training distribution) but didn't fix the real problem — a controlled
same-instance K-sweep (same packages/ULDs, only K varied) came back
**perfectly flat**. The network had learned to ignore K entirely; multi-K
training data alone gives no incentive to actually use it.

The fix: an explicit loss term that takes the *same* instance at *two*
different K values and penalizes the network if its predicted spread
doesn't differ between them. This directly rewards causal K-sensitivity
instead of hoping it emerges from data diversity alone.

## Files

- `train_ga_rl_ppo_contrastive.py` — training script. Monkey-patches
  `train_rl_ppo_contrastive`'s `_load_train_pool` for this process only —
  does not modify `train_rl_ppo.py`, `train_rl_ppo_contrastive.py`, or any
  other existing file.
- `rl_ppo_contrastive_training.log` — full training run log.
- `checkpoints/` — copy of the trained checkpoint (production reference:
  `../../checkpoints/rl_ppo_contrastive_v7/`):
  - `transformer_rl_ppo_contrastive.pt`
  - `rl_ppo_contrastive_training_log.csv` — per-epoch metrics.

The `_v7` in the production folder name reflects this being the 7th
iteration of this specific training run before the version that stuck.
