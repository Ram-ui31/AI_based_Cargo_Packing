# ULD RL Fine-tuning

Reinforcement learning fine-tuning of a Transformer-based ULD (Unit Load Device) clustering model, initialized from an imitation-learning checkpoint.

## Structure

```
uld-rl-finetuning/
├── src/
│   ├── config.py       — all hyperparameters and constants
│   ├── model.py        — TransformerClusterer architecture
│   ├── data_utils.py   — feature extraction, build_tensors, chunking, IL sampling
│   ├── packer.py       — EPI and py3dbp 3-D bin-packing strategies
│   ├── reward.py       — compute_packing_cost, capacity-violation penalty
│   └── train_rl.py     — train_rl(), rl_sample_actions_safe(), rl_assign_argmax_safe()
│
├── notebooks/
│   ├── 01_setup.ipynb  — imports, path config, device setup
│   ├── 02_train.ipynb  — load IL weights, K-value map, call train_rl()
│   └── 03_evaluate.ipynb — argmax inference, RL vs IL comparison, visualization
│
├── requirements.txt
└── README.md
```

## Prerequisites

1. Train the IL model first (`imitation_cluster_v2.ipynb`) — the RL loop requires the IL checkpoint (`transformer_imitation_v2.pt`).
2. Prepare data: `good_data/synthetic_train/` and `good_data/synthetic_test/` with `metadata.csv`, `<tag>_ulds.csv`, and `<tag>_packages.csv` files.

## Quick start

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:

1. `notebooks/01_setup.ipynb` — set `DATA_DIR` and `CLUSTER_DIR`
2. `notebooks/02_train.ipynb` — runs `train_rl()`; saves checkpoint to `CLUSTER_DIR`
3. `notebooks/03_evaluate.ipynb` — loads best checkpoint, plots RL vs IL cost per K

## Key design decisions

**Chunking** (`src/data_utils.py`, `src/train_rl.py`): The model has fixed-shape inputs baked into the IL checkpoint. Instances with more than `MAX_N_PKGS` packages or `MAX_N_ULDS` ULDs are split into chunks instead of being truncated or crashing.

**Capacity penalty** (`src/reward.py`): `model.sample_actions()` hard-masks weight/volume before sampling, which prevents the network from ever seeing a gradient for preferring overflowing ULDs. `rl_capacity_violation_penalty()` adds a differentiable penalty computed from raw pre-mask logits so the policy is actively pushed toward respecting capacity.

**K-value spread penalty**: Each training instance is assigned one of `K ∈ {100, 500, 1000, 3000, 5000}`. The cost function is `delay_cost + K × n_priority_ulds`, so the model learns a policy that generalizes across different trade-offs between delay and spread.

**Packer strategies**: `EPIPacker` (built-in, no extra deps) and `pd3Packer` (requires `py3dbp`). Swap via the `packer=` argument to `train_rl()`.
