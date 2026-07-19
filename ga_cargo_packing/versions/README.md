# Training versions

Every distinct trained-model lineage attempted for the Priority-to-ULD
assignment network, in chronological/dependency order. Each folder holds
that version's training script(s), training log, and a **copy** of its
checkpoint(s) — the checkpoints here are for organizational/archival
completeness only; the production reference every downstream script
actually imports from is still `../checkpoints/<name>/`, untouched by
this reorganization (~20 scripts hardcode that path — moving the live
checkpoints would have broken all of them, so this folder copies rather
than relocates).

| # | Version | Status | Method | Depends on |
|---|---|---|---|---|
| 1 | [`1_il_imitation_learning`](1_il_imitation_learning/) | Base checkpoint | Imitation learning on GA-generated labels | — (trained from scratch) |
| 2 | [`2_rl_early_abandoned`](2_rl_early_abandoned/) | Abandoned, no surviving checkpoint | REINFORCE fine-tune | Version 1 |
| 3 | [`3_rl_graft_k_aware_abandoned`](3_rl_graft_k_aware_abandoned/) | Abandoned, no surviving checkpoint | K-aware REINFORCE fine-tune, grafted onto the K-blind IL checkpoint | Version 1 |
| 4 | [`4_rl_ppo`](4_rl_ppo/) | Superseded by v5 | PPO fine-tune (separate entry point from the REINFORCE line — explicitly does not touch it) | Version 1 |
| 5 | [`5_rl_ppo_contrastive_v7_FINAL`](5_rl_ppo_contrastive_v7_FINAL/) | **Deployed in production** (Eclipse/Halley/Cherry's Priority Clusterer) | PPO + a contrastive loss term that directly rewards the network for using K as a causal control variable (same-instance, two-K contrast) | Version 1, supersedes an intermediate "multi-K" attempt that removed the *correlational* K confound but not the *causal* one |

## The lineage in plain terms

1. **IL (imitation learning)** is the foundation everything else fine-tunes
   from: a Transformer trained to imitate a Genetic Algorithm's
   package-to-ULD assignment decisions. `pretrain_priority_consolidation.py`
   is a related pretraining stage feeding into the same checkpoint family.
2. **RL early** and **RL graft (k-aware)** are two REINFORCE-based fine-tuning
   attempts on top of the IL checkpoint. Both were abandoned before this
   reorganization — no checkpoint files for either survive, only their
   training scripts and (for the early version) a training log. Kept for
   completeness of the experimental record, not because they're usable.
3. **RL PPO** is a deliberately separate, non-destructive experiment
   (its own docstring notes it doesn't touch the REINFORCE pipeline's
   files at all) — PPO fine-tuning from the same IL checkpoint.
4. **RL PPO Contrastive v7** is the version actually in production. It
   adds a contrastive loss specifically to fix a real, diagnosed problem:
   earlier multi-K training data removed a *correlational* confound (K
   happening to correlate with other instance features) but a controlled
   same-instance K-sweep came back perfectly flat — the network wasn't
   causally using K at all. The contrastive term (same instance, two K
   values, penalize predictions that don't differ) fixed that directly.

See each folder's own README for specifics.
