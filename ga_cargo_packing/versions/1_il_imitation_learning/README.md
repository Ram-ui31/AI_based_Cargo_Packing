# Version 1: IL (Imitation Learning)

The foundation checkpoint every other version fine-tunes from. Trains a
Transformer to imitate a Genetic Algorithm's package-to-ULD assignment
decisions (behavioral cloning), not any reward-based method.

## Files

- `train_ga_il.py` — main IL training script.
- `pretrain_priority_consolidation.py` — related pretraining stage; saves
  into the same checkpoint family (`transformer_imitation_priority_pretrained.pt`).
- `train_ga_il.log` — full training run log.
- `checkpoints/` — copy of the trained checkpoints (production reference:
  `../../checkpoints/il/`):
  - `transformer_imitation_ga.pt` — the main IL checkpoint; every
    downstream RL/PPO version warm-starts from this.
  - `transformer_imitation_priority_pretrained.pt` — pretraining-stage checkpoint.
  - `il_training_log.csv` — per-epoch metrics.

## Note

This checkpoint is K-blind (doesn't take the delay-cost multiplier K as
an input) — see version 5's README for why that mattered and how it was
eventually fixed downstream.
