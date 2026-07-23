# GA-labelled ULD packing pipeline

> **Session 4 update**: the real-world stress instance's cost (see "Known
> limitation" below) has since been driven from 33,857 down to **29,564**,
> via a completely different technique family (real-packer-evaluated local
> search, not model training) — see
> [`economy-package-ranker/README.md`](../economy-package-ranker/README.md) for
> the full story. Short version: RL/GRPO hit a structural ceiling around
> 30,608-30,672 because
> it's a smooth (gradient-based) optimizer fighting a jaggy, discrete cost
> landscape; a local beam search directly on the real packer's output broke
> through to 29,656, and a follow-up reformulation as a direct
> package-to-ULD assignment search (not an ordering search) reached 29,564.
> An exact MILP relaxation then proved the remaining gap is now a **3D
> packing-efficiency problem**, not a package-selection problem — consistent
> with, and a sharper restatement of, this README's own "Known limitation"
> section below, written before that proof existed.

Same shape as `cargoism/git`'s H1H2-labelled pipeline (`good-il-over-greedy(c)`
+ `rl_over_il_h1h2`), but the label source is a **Genetic Algorithm** instead
of a hand-tuned greedy heuristic, and the RL fine-tuning stage packs with
`cargoism/git/rl_packer`'s learned single-ULD placement policy instead of the
built-in `EPIPacker`. Nothing under `cargoism/git/` is modified — everything
here reuses that repo's code via `sys.path` imports (same pattern as its own
`H1H2Labeller` / `RLPackerAdapter`).

The RL stage was later upgraded (session 2) to condition its assignment
strategy on K (the spread-cost coefficient) — the original RL model never
saw K as an input at all. See `NOTES.md` and `results/GA_pipeline_report.pdf`
§7 for the full story.

## Repository layout

```
model-training-pipeline/
├── src/
│   ├── ga/            — Genetic Algorithm solver, pipeline, labeller
│   ├── il/             — IL Transformer: model, training loop, data pipeline (K-blind by design)
│   └── rl/             — RL fine-tuning: model (dual K-injection), REINFORCE loop
│                          (train_rl.py — also the shared inference utilities),
│                          PPO loop (train_rl_ppo.py, ppo_rollout.py), packer,
│                          rl_packer adapter
├── scripts/            — CLI entrypoints (one per pipeline stage)
│   ├── precompute_ga_cache.py
│   ├── train_ga_il.py
│   ├── train_ga_rl.py           — original (K-blind) REINFORCE training
│   ├── train_ga_rl_graft.py     — K-aware REINFORCE, warm-started from the
│   │                               original IL checkpoint (also a dependency
│   │                               of train_ga_rl_ppo.py below)
│   ├── train_ga_rl_ppo.py       — K-aware PPO + per-K EMA baseline (current best)
│   ├── beam_search_economy.py   — [session 4] local beam search over Economy
│   │                               package ORDER, real packer as evaluator
│   ├── beam_search_guided.py    — [session 4] same, but candidate swaps are
│   │                               pre-screened by economy-package-ranker's
│   │                               SwapProposer (pairwise ranking model)
│   ├── knapsack_search_economy.py — [session 4] local beam search directly
│   │                               over the package -> ULD ASSIGNMENT
│   │                               (multiple-knapsack reformulation)
│   ├── milp_ceiling.py          — [session 4] exact volume+weight MILP
│   │                               (scipy/HiGHS) -- theoretical ceiling +
│   │                               proof that the remaining gap is a 3D
│   │                               packing-efficiency problem
│   ├── test_priority_allocation.py — Priority-to-ULD allocation sweep
│   └── generate_summary_plots.py   — [session 4] renders results/plots/*.png
├── eval/
│   ├── verify_pipeline.py             — asserts the 4 correctness conditions
│   ├── generate_comparison_data_ppo.py — GA/PPO cost+spread per test instance
│   ├── generate_plots.py               — cost_vs_k.png / spread_vs_k.png
│   └── generate_report.py              — results/GA_pipeline_report.pdf
├── results/
│   ├── comparison_ga_ppo.csv
│   ├── cost_vs_k.png
│   ├── spread_vs_k.png
│   └── GA_pipeline_report.pdf
├── checkpoints/
│   ├── il/             — transformer_imitation_ga.pt (K-blind IL, graft source), training log
│   ├── rl/              — transformer_rl_ga.pt — original K-blind RL model (baseline)
│   └── rl_ppo/           — transformer_rl_ppo.pt — current best, K-conditioned (USE THIS for inference)
├── cache/
│   └── ga_cache.pkl    — precomputed GA labels for all 1000+100 instances
├── logs/                — raw stdout from each training run
└── NOTES.md              — session-2 K-conditioning fix write-up
```

## Pipeline

```
good_data/{synthetic_train,synthetic_test}/   (existing, unmodified)
        │
        ▼
src/ga/              Genetic Algorithm: for each Economy package, decide
                     gene ∈ {2=priority-ULD bucket, 1=other-ULD bucket,
                     0=unallocated}. Priority packages are packed first
                     against the full ULD fleet (reusing h1_h2_cargo's
                     GreedyPipeline rescue/nuclear-eviction, unmodified) so
                     they're never affected by the GA's choices. Fitness =
                     sum(delay_cost of unallocated Economy). See
                     ga_solver.py for the encoding/crossover/mutation/
                     selection spec.
        │
        ▼
src/il/              IL Transformer (TransformerClusterer) trained to
                     imitate the GA's package→ULD assignments. Deliberately
                     K-blind — the GA labels it imitates never depend on K,
                     so a K input would only add noise at this stage (see
                     NOTES.md).
        │
        ▼
src/rl/              RL fine-tunes the IL checkpoint via a "graft" warm
                     start (zero-initialized new K-input layers grafted
                     onto the K-blind IL checkpoint, so training starts
                     behaviorally identical to it). Packer = rl_packer
                     (RLPackerAdapter). Two training loops available:
                       - train_rl.py            REINFORCE, frozen-IL-baseline
                                                 advantage, per-instance
                                                 normalized
                       - train_rl_ppo.py         PPO clipped surrogate +
                                                 online per-K EMA baseline
                                                 (current best, checkpoints/rl_ppo/)
                     Auxiliary losses (both loops): a K-scaled differentiable
                     soft-spread loss and a feasibility hinge loss (penalizes
                     rejecting a dimensionally-feasible Economy package) —
                     their weights must be comparable in magnitude to each
                     other or one silently dominates (see NOTES.md).
        │
        ▼
eval/verify_pipeline.py   Asserts: every Priority packed, no overlaps,
                          weight/volume respected, reports K*spread + delay
                          cost vs the H1H2+RL / IL-only baselines in
                          cargoism/git/common/results/.
```

## Usage

```bash
cd model-training-pipeline

# 1. Precompute GA labels (parallel across CPU cores -- this is the slow step)
python scripts/precompute_ga_cache.py --data-root ~/Desktop/good_data --out cache/ga_cache.pkl

# 2. Train IL on the GA labels
python scripts/train_ga_il.py --data-root ~/Desktop/good_data \
    --cache cache/ga_cache.pkl --save-dir checkpoints/il

# 3. RL fine-tune with rl_packer -- PPO (recommended, current best)
python scripts/train_ga_rl_ppo.py --data-root ~/Desktop/good_data \
    --old-il-weights checkpoints/il/transformer_imitation_ga.pt \
    --save-dir checkpoints/rl_ppo

# 4. Verify
python eval/verify_pipeline.py --data-root ~/Desktop/good_data \
    --checkpoint checkpoints/rl_ppo/transformer_rl_ppo.pt \
    --baselines-dir ~/Desktop/cargoism/git/common/results

# 5. Regenerate comparison plots + report
python eval/generate_comparison_data_ppo.py --data-root ~/Desktop/good_data \
    --ga-cache cache/ga_cache.pkl \
    --rl-checkpoint checkpoints/rl_ppo/transformer_rl_ppo.pt \
    --out results/comparison_ga_ppo.csv
python eval/generate_plots.py --data results/comparison_ga_ppo.csv --out-dir results
python eval/generate_report.py
```

K values (100/500/1000/3000/5000, 200 each across the 1000 train instances,
20 each across the 100 test instances) are read directly from
`good_data/synthetic_{train,test}/metadata_with_K.csv`, which already exists
— no re-derivation needed.

GA defaults (`pop_size=16, max_generations=20, patience=6,
time_budget_seconds=90`) are tuned for tractability across ~1300 instance-
chunks on a 10-core machine (took ~3.25h for the full precompute); override
via CLI flags on `precompute_ga_cache.py` for a more/less thorough search.

## Results (fair, full 83-instance non-chunked test set)

**Correction (session 3, late):** a real bug was found in
`rl_packer_adapter.py`'s cross-ULD rescue pass — when rescuing a stuck
Priority package into another ULD, it re-packed that ULD from only its
*currently-placed* packages plus the rescued one, silently discarding that
ULD's own previously-left-behind packages (they never got placed, and were
overwritten out of tracking, so `compute_packing_cost` never counted their
delay cost either). Fixed to carry those packages forward every time. This
was invisible on typical smaller test instances but large on the 400-package
real-world stress instance (91 of 400 packages were silently vanishing).
**All numbers below already reflect the fix.**

| Model | Mean cost (K·spread + delay) | vs original |
|---|---|---|
| **`checkpoints/rl_ppo/` + decode-order + Priority consolidation + Economy value-density (use this)** | **12,698.2** | **−4,147.8 (−24.6%)** |
| `checkpoints/rl/` — original, K-blind | 16,846.0 | — |
| `cargoism/git` H1H2+RL baseline | 24,513.8 | — |
| `cargoism/git` Hybrid baseline | 25,013.7 | — |

Zero priority drops, zero overlap/weight/volume violations across the full
test set (`eval/verify_pipeline.py`). The K-conditioning fix (session 2)
clearly wins at K=100/500/5000, is roughly tied at K=1000, and is slightly
behind at K=3000.

The step-by-step mean-cost figures in the walkthrough below (13,362.9 /
12,390.9 / etc.) were measured before the rescue-loop bug above was found —
they predate the fix, so are each mildly understated in absolute terms.
Each fix's *qualitative* conclusion (the K=500 gate, unseeded-beats-seeded)
was a same-bug-on-both-sides comparison, so those conclusions hold; only the
absolute mean-cost numbers in steps 1–3 do not reflect the final bugfix.
The final row above and the real-world-instance numbers after it are
correct as of the fix.

Session 3 is three inference-time fixes layered on the frozen session-2
model (no retraining) — worth being precise that this is decoding-order and
heuristic engineering around a fixed policy, not the model learning new
behavior:
1. **Decode order**: `rl_assign_argmax_safe` decoded packages in file order,
   letting Economy packages claim ULD capacity before Priority packages
   later in the file arrived. Reordering to Priority-first (desc. weight),
   then Economy (asc. volume) fixed this — 78/83 test instances improved
   (−19.0% mean cost).
2. **Capacity-aware Priority consolidation**: even with the order fixed, the
   *choice* of which ULDs to put Priority in was still left to the model's
   own per-package decisions, which don't have a global view of "which ULD
   combination leaves the most margin." A real-world stress instance showed
   this leaving 0 headroom against `rl_packer`'s real (~70%, not 100%)
   placement efficiency, forcing an extra ULD. `_consolidate_priority_by_capacity`
   deterministically bin-packs Priority into the fewest, largest-volume ULDs
   first. This only pays off when K is large enough to justify giving
   Economy the smaller ULDs, so it's gated to K ≥ 500
   (`PRIORITY_CONSOLIDATION_MIN_K`, `src/rl/config.py`) — a +443.8 mean-cost
   loss at K=100 if applied unconditionally (10/17 instances worse), a clear
   win at every K≥500 (e.g. −2,750.5 mean at K=5000).
3. **Economy value-density selection**: at low K, spread is nearly
   irrelevant (K=100: delay cost is ~98% of total cost), so the whole
   problem there is a knapsack — which Economy packages are worth keeping
   given more supply than capacity. The model's own per-package Economy
   choice shows *some* learned value-triage (kept packages average ~2x the
   value-density of dropped ones) but a simple greedy heuristic — sort by
   descending delay_cost÷volume, first-fit — still beats it at *every* K
   bucket (~9–12% lower mean cost, no gating needed). Economy's ULD choice
   is now always re-derived this way after Priority is placed, deliberately
   *not* seeded with Priority's nominal footprint (same real-packer-slack
   reasoning as #2) — kept as the default, though after the rescue-loop
   bugfix below, seeded vs. unseeded turned out to be nearly a wash on the
   real-world instance (33,846 vs 33,857); the dramatic gap reported earlier
   (27,474 vs 30,703) was itself partly an artifact of the bug in #4.
4. **Packer rescue-loop bugfix** (`src/rl/rl_packer_adapter.py`): when
   rescuing a stuck Priority package into another ULD, the rescue re-packed
   that ULD from only its *currently-placed* packages plus the rescued one —
   silently discarding that ULD's own previously-left-behind packages
   (`candidate_ids = other_placed_ids + [pid]` should have also included
   `left_behind_by_uld[other_uid]`). Those packages never got placed, and
   were overwritten out of tracking entirely — `compute_packing_cost` only
   sums over the returned placements list, so they contributed no delay
   cost either, silently understating every cost this session had reported
   for large/contested instances. Invisible on typical smaller test
   instances; on the 400-package real-world instance, 91 of 400 packages
   were vanishing this way. Also includes the rescue-order fix from before
   (try Priority-holding ULDs first, then by volume, rather than raw volume
   only) — real and useful on its own, though it alone didn't move this
   specific instance (those 3 ULDs held Priority almost exclusively, nothing
   to evict).

On the real-world stress instance (400 packages, K=5000, 103 Priority) that
originally surfaced all of this: cost went from 49,903 (file-order decode,
also understated by the same bug, likely by less since spread=6 meant less
rescue contention) to **33,857** (spread 6→3) after every fix above and the
bugfix, zero violations. This does **not** beat its 27,500 external
benchmark — closing that gap further remains open work. Breakdown of the
33,857: 15,000 spread cost, 3,960 delay cost from packages the assignment
stage itself dropped, and **14,897 delay cost from packages the assignment
stage assigned a ULD but the packer couldn't physically fit** — the
packer's own ~70%-of-nominal placement efficiency ceiling, priced out
directly (see "Known limitation" below) — now the dominant cost component.

See `NOTES.md` and `results/GA_pipeline_report.pdf` §7.5–7.8 for the full
story (including the packer-efficiency diagnosis and the bug correction)
and `results/*.png` for the per-K breakdown.

## Known limitation: rl_packer's placement efficiency

`src/rl/rl_packer_adapter.py` packs each ULD Priority-first (two episodes
into the same Heightmap, Priority packages get first claim on every extreme
point) and includes a bounded cross-ULD eviction-rescue pass for stuck
Priority packages (mirroring `h1_h2_cargo`'s own `EPIPacker` rescue
strategy, and since session 3 preferring ULDs that already hold Priority
over raw volume when choosing a rescue target) — added specifically because
the plain `rl_packer_adapter.py` in `cargoism/git/rl_over_il_h1h2` has
neither and was measurably worse on the priority-safety condition in
side-by-side testing.

The learned placement policy itself (`rl_packer`'s neural network,
`placement_policy.pt` — untouched all session) achieves only **~70% true
volumetric efficiency** in practice, not the ~100% the assignment stage's
aggregate weight/volume accounting assumes (verified order-invariant across
6 heuristic orderings + 15 random shuffles, so it's a genuine placement-
quality ceiling, not another ordering artifact; rotation is already in its
candidate search). This is now the largest identified remaining gap and the
dominant cost component: 44.0% of the real-world stress instance's final
cost (14,897 of 33,857) is Economy packages the assignment stage thought
would fit but the packer couldn't physically place. Closing it further
means improving the placement policy itself (better search — e.g.
beam/best-of-N instead of single greedy rollout, which recovered ~18% of
stuck packages in a quick test — or retraining it), a separately trained
component out of scope for this session. This is also the most direct lever
left for closing the real-world instance's remaining gap to its 27,500
external benchmark (currently 33,857, per the correction above).

## What's not in this pipeline (possible future work)

- **The per-K EMA baseline (PPO) is still a scalar mean/std**, not a full
  learned critic conditioned on the specific instance's features — a proper
  value head would likely sharpen the advantage estimate further,
  particularly at K=3000 (the one bucket the K-conditioning fix didn't fully
  resolve).
- **Per-instance rollouts, not batched.** Each training step processes one
  instance at a time; batching multiple instances' gradients before each
  `optimizer.step()` would reduce variance and probably wall-clock time too.
- **GA population/generation counts are tuned for tractability, not
  quality** — `pop_size=16, max_generations=20` was chosen to keep the full
  1300-chunk precompute under ~4 hours; a larger search budget (or a smarter
  fitness-evaluation shortcut) would likely produce better GA labels, which
  the IL model would then imitate more accurately.
- **The GA's fitness is K-agnostic** — it only optimizes delay cost, never
  spread. A K-aware GA fitness could give the IL model a better starting
  point, narrowing the gap to the final RL/PPO result before RL even enters.
- **`rl_packer`'s placement policy is the largest remaining gap** (see
  "Known limitation" above) — ~70% true volumetric efficiency vs the ~100%
  the assignment stage's capacity accounting assumes. A quick best-of-10
  stochastic-rollout test recovered ~18% of packages a single greedy rollout
  left stuck, suggesting real headroom from better search alone, before even
  considering retraining the policy.
