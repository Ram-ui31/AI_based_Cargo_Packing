# ULD Cargo Packing

End-to-end pipeline for the Unit Load Device (ULD) cargo-packing problem: assign packages to ULDs to minimize delay cost and priority spread, subject to weight/volume capacity. The repo covers four stages — synthetic data generation, a hand-tuned greedy heuristic baseline, an imitation-learning (IL) Transformer trained on that baseline, and RL fine-tuning of the IL model.

## Structure

```
.
├── good-data-generator/    — synthetic ULD/package dataset generator
├── h1_h2_cargo/            — greedy heuristic baseline (H1/H2 scorers, binary-search split, extreme-point packer)
├── good-il-over-greedy/    — imitation-learning Transformer clusterer, trained on greedy-heuristic labels
├── rl_fineuning_over_il/   — RL fine-tuning of the IL checkpoint
└── LICENSE
```

### Pipeline order

1. **`good-data-generator/`** — generate synthetic `{ulds, packages, metadata}.csv` instances.
2. **`h1_h2_cargo/`** — solve instances with the greedy heuristic pipeline; produces baseline assignments and acts as the label source for IL.
3. **`good-il-over-greedy/`** — train a TransformerClusterer to imitate the greedy heuristic's assignments.
4. **`rl_fineuning_over_il/`** — fine-tune the IL checkpoint with RL (policy gradient over packing cost) to improve past the heuristic.

---

## `good-data-generator/`

Generates synthetic ULD/package instances with controllable volume/weight fill ratios and priority-vs-economy package mix.

```
good-data-generator/
├── src/
│   ├── config.py      — dimension/weight pools, target ratios, instance-size bounds
│   ├── generators.py  — generate_ulds, generate_packages, generate_instance, generate_dataset
│   ├── sampling.py    — dimension/weight pools and scaling helpers
│   └── summary.py     — dataset-level summary statistics
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_generate.ipynb
│   └── 03_summary.ipynb
└── requirements.txt
```

Each instance independently samples ULD dimensions/weight limits, then generates Priority and Economy package pools scaled so the overall volume/weight fill ratio and the priority share both land near configured targets. Output per instance: `<tag>_ulds.csv`, `<tag>_packages.csv`, plus a dataset-level `metadata.csv`.

## `h1_h2_cargo/`

Greedy heuristic baseline and reference packing engine, used both as a standalone solver and as the label source for the IL model.

```
h1_h2_cargo/
├── geometry.py            — Package/ULD/PlacedBox primitives, coordinate conventions
├── extreme_points.py      — checkpoint/rollback extreme-point tracker for fast placement search
├── uld_partition.py       — Step 1: partitions ULDs into a priority bucket and an "other" bucket
├── h1_heuristic.py        — economy package scorer feeding the binary-search split
├── binary_search_split.py — splits economy packages into Set 1 / Set 2 via trial packing
├── selector.py            — candidate placement generator and scorer
├── greedy_pack.py         — single-pass greedy placer
├── h2_heuristic.py        — leftover economy package scorer for retroactive allocation
├── greedy_pipeline.py     — orchestrates partition → split → pack → retroactive H2 fill
├── dataset_io.py          — reads the cargo dataset layout (toy/generated splits, ULD catalogue)
├── run_greedy.py          — CLI batch runner: `python run_greedy.py --data-root data --split <split> --out <dir>`
├── run_dataset.py         — parallel batch runner (ProcessPoolExecutor) over a full split
└── k_results.csv          — per-K result summary
```

Pipeline (priority packages always packed first, economy packages split and scored by H1/H2 so high-value/expensive-to-delay items are placed earliest): `uld_partition` → `binary_search_split` (H1-scored) → `greedy_pack` → retroactive `h2_heuristic` fill of leftovers.

## `good-il-over-greedy/`

Imitation-learning Transformer that learns to reproduce the greedy heuristic's package→ULD assignments, providing a warm-start checkpoint for RL.

```
good-il-over-greedy/
├── src/
│   ├── config.py       — architecture dims (checkpoint-shape-defining — don't change post-training), training hyperparameters
│   ├── model.py         — TransformerClusterer architecture
│   ├── data_utils.py    — ClusteringDataset, collate_fn, build_tensors, chunking
│   ├── labeller.py       — Labeller strategy pattern; GreedyLabeller wraps the h1_h2_cargo heuristic for training labels
│   ├── losses.py         — capacity_violation_penalty
│   ├── inference.py      — chunked inference for instances of any size
│   └── train_il.py       — train_il()
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_train.ipynb
│   └── 03_evaluate.ipynb
└── requirements.txt
```

`Labeller` is a strategy pattern — swap in a different heuristic by subclassing `Labeller` and passing it to `ClusteringDataset()` without touching the model or training loop. `inference.py`'s chunked inference is safe to call unconditionally, including for instances within the model's normal capacity.

## `rl_fineuning_over_il/`

Reinforcement learning fine-tuning of the IL Transformer checkpoint.

```
rl_fineuning_over_il/
├── src/
│   ├── config.py       — all hyperparameters and constants
│   ├── model.py        — TransformerClusterer architecture
│   ├── data_utils.py   — feature extraction, build_tensors, chunking, IL sampling
│   ├── packer.py       — EPI and py3dbp 3-D bin-packing strategies
│   ├── reward.py       — compute_packing_cost, capacity-violation penalty
│   └── train_rl.py     — train_rl(), rl_sample_actions_safe(), rl_assign_argmax_safe()
├── notebooks/
│   ├── 01_setup.ipynb  — imports, path config, device setup
│   ├── 02_train.ipynb  — load IL weights, K-value map, call train_rl()
│   └── 03_evaluate.ipynb — argmax inference, RL vs IL comparison, visualization
└── requirements.txt
```

### Prerequisites

1. Train the IL model first (`good-il-over-greedy/`) — the RL loop requires the resulting IL checkpoint (`transformer_imitation_v2.pt`).
2. Prepare data: `good_data/synthetic_train/` and `good_data/synthetic_test/` with `metadata.csv`, `<tag>_ulds.csv`, and `<tag>_packages.csv` files (e.g. from `good-data-generator/`).

### Quick start

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:

1. `notebooks/01_setup.ipynb` — set `DATA_DIR` and `CLUSTER_DIR`
2. `notebooks/02_train.ipynb` — runs `train_rl()`; saves checkpoint to `CLUSTER_DIR`
3. `notebooks/03_evaluate.ipynb` — loads best checkpoint, plots RL vs IL cost per K

### Key design decisions

**Chunking** (`src/data_utils.py`, `src/train_rl.py`): The model has fixed-shape inputs baked into the IL checkpoint. Instances with more than `MAX_N_PKGS` packages or `MAX_N_ULDS` ULDs are split into chunks instead of being truncated or crashing.

**Capacity penalty** (`src/reward.py`): `model.sample_actions()` hard-masks weight/volume before sampling, which prevents the network from ever seeing a gradient for preferring overflowing ULDs. `rl_capacity_violation_penalty()` adds a differentiable penalty computed from raw pre-mask logits so the policy is actively pushed toward respecting capacity.

**K-value spread penalty**: Each training instance is assigned one of `K ∈ {100, 500, 1000, 3000, 5000}`. The cost function is `delay_cost + K × n_priority_ulds`, so the model learns a policy that generalizes across different trade-offs between delay and spread.

**Packer strategies**: `EPIPacker` (built-in, no extra deps) and `pd3Packer` (requires `py3dbp`). Swap via the `packer=` argument to `train_rl()`.

## License

MIT — see [LICENSE](LICENSE).
