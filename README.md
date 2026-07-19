# Cargoism — Optimal Cargo Management for Flights

An end-to-end system for the air cargo ULD (Unit Load Device) loading
problem: assign Priority and Economy packages to a fixed set of ULDs and
pack them in 3D, minimizing a delay-cost-and-priority-spread objective
subject to hard weight, volume, and non-overlap constraints, with every
Priority package guaranteed to ship.

**Best result: average cost 9,499** across 20 held-out synthetic
instances (4 each across K ∈ {100, 500, 1000, 3000, 5000}) — beating a
published academic online-3D-bin-packing baseline's 15,535 average on the
same instances by a wide margin, with zero Priority packages ever
dropped (the baseline drops at least one Priority package in 17 of the
20 instances).

## Where to start

| I want to... | Go to |
|---|---|
| See the final, best-performing system | [`cherry/`](cherry/) |
| Understand the three model variants that were compared head-to-head | [`cherry/`](cherry/), [`eclipse/`](eclipse/), [`halley/`](halley/) |
| See the research/development environment that produced them | [`ga_cargo_packing/`](ga_cargo_packing/), [`gnn_economy_selector/`](gnn_economy_selector/) |
| See how the project evaluated against a published academic baseline | `online-3d-bpp-benchmark/` *(sibling folder alongside this repo, not itself part of it)* |
| See the earlier project iterations that led here | [Project history](#project-history) below |

## The three final models

Three trained-model variants were built and evaluated on identical
terms — the same 20 held-out synthetic instances (4 each across K ∈
{100, 500, 1000, 3000, 5000}), grand average = mean of the 5 per-K
averages:

| Model | Role | Grand-average cost |
|---|---|---|
| **Eclipse** | RL-trained 3D placement policy, competing in a best-of-5 ensemble against heuristic strategies for every container | 10,631 |
| **Halley** | GRPO-trained economy-package ordering model, generalized across many synthetic instances | 10,540 |
| **Cherry** | Set-attention transformer that screens evict-and-recompact moves as a final refinement pass on top of Eclipse's result | **9,499** |

Cherry is named for its headline component, `CentrifugeEvictProposer` —
the most rigorously validated model in the project: held out on unseen
synthetic instances, then validated at production scale on a
structurally different, sparse-signal regime, where model-guided search
reached the identical result as exhaustive brute-force search at roughly
1/15th the number of expensive geometric checks.

Each of the three folders is a complete, standalone, GitHub-style
deliverable: full pipeline source, trained checkpoints, results,
comparison graphs, and architecture flowcharts, with an honest account of
what was tried, what worked, and what didn't (including negative results
for reinforcement learning on the package-ordering side and for an
AI-guided local-search variant that never demonstrated an improvement).

## Repository structure

```
cargoism/git/                    ← repository root
├── cherry/                — FINAL: best-performing pipeline (9,499 avg)
├── eclipse/                — RL placement-policy pipeline (10,631 avg)
├── halley/                 — GRPO economy-ordering pipeline (10,540 avg)
├── ga_cargo_packing/       — research environment: assignment, local
│                              search, packer ensemble, EMS geometry
├── gnn_economy_selector/   — research environment: economy-ranking
│                              and move-screening model training
├── rl_packer/              — shared 3D placement-policy source (geometry,
│                              environment, actor-critic network)
├── common/                  — shared evaluation/comparison utilities
│                              across earlier pipeline iterations
├── good-data-generator/    — synthetic ULD/package instance generator
├── h1_h2_cargo/             — hand-tuned greedy heuristic baseline
├── good-il-over-greedy(c)/ — imitation-learning Transformer trained
│                              on the greedy baseline's labels
├── model_b(c)/              — learned assignment-policy environment
├── rl_fineuning_over_il/   — early RL fine-tune of the IL checkpoint
├── rl_over_il_h1h2/         — RL fine-tune combined with the H1/H2
│                              heuristic packer
└── LICENSE
```

Trained placement-policy checkpoints referenced by `rl_packer` and by
every packer ensemble live one level up, at `cargoism/uld_heightmap_rl/`
— outside this repository, since checkpoint binaries are versioned
separately from source.

## The path to the final result

1. **Data generation** (`good-data-generator/`) produces synthetic ULD/
   package instances with controllable fill ratios and Priority/Economy
   mix, used throughout for training and generalization testing.
2. **Heuristic baseline** (`h1_h2_cargo/`) — a hand-tuned greedy pipeline
   (ULD partitioning, binary-search economy split, extreme-point packing)
   used both as a standalone solver and as the initial label source for
   imitation learning.
3. **Imitation learning → RL fine-tuning** (`good-il-over-greedy(c)/`,
   `rl_fineuning_over_il/`, `model_b(c)/`, `rl_over_il_h1h2/`) — a
   sequence of iterations training a Transformer to imitate, then improve
   past, the heuristic baseline, progressively adding K-awareness,
   3D-placement integration, and a dedicated RL placement policy
   (`rl_packer/`).
4. **Production research environment** (`ga_cargo_packing/`,
   `gnn_economy_selector/`) — where the final architecture took shape: a
   trained Priority Clusterer, a 5-way best-of-N packer ensemble
   (including the trained RL placement policy), a local search over
   Economy package assignment with real geometric evaluation at every
   step, and — the single largest lever found — a full geometric
   candidate-generation rewrite (Empty Maximal Space decomposition)
   fixing a structural bottleneck shared by every earlier packing
   strategy. `ga_cargo_packing/versions/` documents the complete training
   lineage of the Priority Clusterer, including abandoned branches, with
   an explanation of why each was superseded.
5. **Three final, independently packaged deliverables** — Cherry, Eclipse,
   and Halley — each isolating one trained model's real, measured
   contribution against a shared baseline pipeline.
6. **External validation** — the final pipeline was benchmarked against
   `Online-3D-BPP-DRL` (AAAI 2021), a standard sequential-placement
   online bin-packing policy in the same family as Packing Configuration
   Tree (PCT)-style methods, on the same 20 held-out synthetic instances
   used above. It averaged **15,535** — well behind Cherry's 9,499 — and,
   because it has no concept of a hard Priority constraint, dropped at
   least one Priority package in **17 of the 20 instances** (203
   Priority packages dropped in total). Every model in this repository
   dropped **zero** Priority packages, in every instance, at every K.
   Documented in the sibling `online-3d-bpp-benchmark/` folder.

## Key technical findings

- **Candidate-generation geometry, not model sophistication, was the
  single biggest lever.** Every packing strategy — trained and
  heuristic alike — originally generated placement candidates from
  corner-adjacent points of already-placed boxes only, structurally
  missing valid placements in gaps that don't touch an existing box's
  corner. Replacing this with a complete empty-space decomposition
  improved every strategy by 200–2,000 points before any model changes.
- **Reinforcement learning on the package-selection side consistently
  underperformed a hand-tuned formula** across every configuration tested
  (single-instance RL/GRPO, multi-instance generalized GRPO, IL-warm-start
  GRPO fine-tuning) — diagnosed as a genuinely jagged, discontinuous cost
  landscape rather than a tuning failure, and corroborated independently
  by a separate team's report of the same finding on the same problem
  statement.
- **The RL 3D placement policy (Eclipse) has a real, measured, positive
  contribution** — +339 points by controlled ablation (remove it from the
  ensemble, re-run, compare) — the clearest positive AI result in the
  project outside of Cherry's own model.
- **A "centrifuging" compaction technique** (slide placed boxes toward
  one corner to consolidate fragmented free space) was investigated in
  depth: mechanically valid and genuinely effective in isolation, of
  negligible value once naively added to an already-strong ensemble, and
  ultimately real and generalizable once correctly targeted — an evict
  → compact → refill move, screened by a trained model rather than
  exhaustive search. This became Cherry.

## Project history

The repository grew through several complete pipeline iterations before
arriving at the current architecture. Each earlier stage remains in the
repository, unmodified, as a record of what was tried:

- **`good-data-generator/`** — the synthetic instance generator, used
  unchanged from the earliest iteration through to the final one.
- **`h1_h2_cargo/`** — the original heuristic baseline: ULD partitioning,
  an H1-scored binary-search split of Economy packages, extreme-point
  greedy packing, and an H2-scored retroactive fill pass. Still used as a
  labelling source and a reference packer.
- **`good-il-over-greedy(c)/`** — the first learned model, trained purely
  to imitate `h1_h2_cargo`'s assignments.
- **`rl_fineuning_over_il/`** — the first attempt at improving past the
  imitation-learned checkpoint with policy-gradient RL.
- **`model_b(c)/`** — a from-scratch learned assignment policy and
  environment, exploring a different action/state representation from
  the IL-then-RL line.
- **`rl_over_il_h1h2/`** — RL fine-tuning integrated directly with the
  H1/H2 heuristic packer, the immediate predecessor to the final
  production environment.
- **`rl_packer/`** — the 3D placement-policy component (geometry,
  environment, actor-critic network) developed alongside the above and
  ultimately shared by every later packer ensemble, including Eclipse's.
- **`common/`** — evaluation and comparison utilities used across the
  above iterations to keep results comparable as the architecture changed.

The final architecture — a trained Priority Clusterer, a best-of-N packer
ensemble built around `rl_packer`'s placement policy, a real-geometry
local search, and the Cherry/Eclipse/Halley model comparisons — lives in
`ga_cargo_packing/` and `gnn_economy_selector/`.

## License

MIT — see [LICENSE](LICENSE).
