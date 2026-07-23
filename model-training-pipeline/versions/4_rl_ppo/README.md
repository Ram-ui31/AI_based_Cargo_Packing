# Version 4: RL PPO

A PPO fine-tune from the Version 1 IL checkpoint — a deliberately
*separate* entry point from the REINFORCE line (Versions 2–3), explicitly
designed not to touch those files at all, so if this experiment didn't
pan out the REINFORCE pipeline would remain completely unaffected.

## Files

- `train_ga_rl_ppo.py` — training script.
- `rl_ppo_training.log` — full training run log.
- `checkpoints/` — copy of the trained checkpoint (production reference:
  `../../checkpoints/rl_ppo/`):
  - `transformer_rl_ppo.pt`
  - `rl_ppo_training_log.csv` — per-epoch metrics.

## Status

Superseded by Version 5 (RL PPO Contrastive v7), which added a
contrastive loss term on top of this same PPO approach to fix a real,
diagnosed causal-K-sensitivity problem this version still had.
