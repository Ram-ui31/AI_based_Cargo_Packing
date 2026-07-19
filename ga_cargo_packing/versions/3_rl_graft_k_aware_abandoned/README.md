# Version 3: RL Graft, K-aware (abandoned)

A K-aware REINFORCE fine-tune, warm-started from the *old* (K-blind)
Version 1 IL checkpoint rather than retraining IL from scratch with K as
an input. Abandoned — **no checkpoint file survives**, only the training
script.

## Why grafting instead of retraining IL with K included

Retraining IL with K as an input (producing an `il_k_aware` checkpoint)
gave a meaningfully *worse* IL model (val_loss 0.797 @ epoch 146,
early-stopped) than the original K-blind IL run (val_loss 0.760 @ epoch
233 — Version 1's `transformer_imitation_ga.pt`). Two reasons: (1) K
carries no real signal for the IL objective — the GA labels it's
imitating don't depend on K — so adding it as an input just added noise
to imitation learning specifically; (2) grafting K-awareness onto the RL
fine-tuning stage instead (where it plausibly *does* matter, since reward
is K-weighted) sidesteps that problem entirely.

## Files

- `train_ga_rl_graft.py` — the only surviving artifact of this line.

Note: this script's own docstring references a `checkpoints/rl_modelb_balanced/`
checkpoint from "the existing, working REINFORCE pipeline" — that
checkpoint no longer exists on disk either (only a compiled `.pyc` cache
of the script remains as a trace). Version 4 (RL PPO) explicitly branched
off as a *separate, non-destructive* experiment from this same K-blind IL
checkpoint rather than continuing down this REINFORCE line, and that's the
line that ultimately led to the deployed model (Version 5).
