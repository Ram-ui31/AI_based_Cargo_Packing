# Version 2: RL Early (abandoned)

A REINFORCE-style RL fine-tune starting from the Version 1 IL checkpoint.
Abandoned — **no checkpoint file survives**, only the training script and
its log.

## Files

- `train_ga_rl.py` — training script (mirrors `cargoism/git/rl_over_il_h1h2/scripts/train_h1h2_rl.py`,
  defaults to `RLPackerAdapter` as the packer instead of `EPIPacker`).
- `train_ga_rl.log` — full training run log (kept for the experimental
  record even though the resulting checkpoint no longer exists).

Superseded by Version 4 (RL PPO) and Version 5 (RL PPO Contrastive), which
took a PPO-based approach instead of REINFORCE.
