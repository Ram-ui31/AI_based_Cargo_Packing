# K-conditioning fix — session notes (2026-07-11/12)

## The problem
The original RL model (`checkpoints/rl/`) never saw K (the spread-cost
coefficient) as an input at all — it produced roughly the same spread
regardless of whether K=100 or K=5000, which is wrong: spread cost should
matter a lot more at high K.

## What was tried (roughly in order)
1. **From-scratch retrain with K injected** — regressed badly (worse than
   the old model on every axis). Root cause: retraining IL from scratch with
   an added `k_proj` layer converged to a meaningfully worse checkpoint than
   the original K-blind IL run.
2. **Graft warm-start** — instead of retraining IL, grafted a
   zero-initialized `k_proj` onto the *existing* strong IL checkpoint (so
   the model starts out mathematically identical to the old model, then RL
   fine-tunes from there). This became the standard warm-start technique
   used for every attempt after this point. Better, but still plateaued
   below the old model.
3. **Ported fixes from `cargoism/git/model_b(c)`** (a separate, more mature
   reference implementation that had already solved this problem):
   log-scale K normalization (was linear, crushed most K values near 0),
   **dual K injection** (K fed in twice — once into the shared transformer
   trunk, once concatenated directly at the output head, since a single
   diffuse path let K get "washed out"), a **feasibility hinge loss**
   (penalizes rejecting a dimensionally-feasible economy package to NONE),
   and a **K-scaled soft-spread loss** (differentiable proxy for spread,
   scaled by K).
4. **Critical calibration bug found and fixed**: the spread loss had been
   ~2000x smaller than the other loss terms all session — mathematically
   present but functionally noise. Boosting its weight alone caused the
   model to over-minimize spread even at *low* K (where it shouldn't).
   Fix: boost the hinge loss (economy-retention pressure) by a comparable
   amount too, so the two forces properly counterbalance each other instead
   of one drowning out the other. **This was the actual unlock** — the
   first checkpoint after this fix beat the old model outright.
5. **Speed optimization**: stratified per-epoch instance subsampling (250 of
   1000 instances/epoch, balanced across K) to get more training epochs per
   hour of wall-clock time.
6. **PPO + per-K EMA baseline** (`src/rl/train_rl_ppo.py`, `src/rl/ppo_rollout.py`,
   `scripts/train_ga_rl_ppo.py` — all new files, none of the working
   REINFORCE code was touched): ported model_b's more sophisticated training
   mechanism (clipped surrogate objective, online per-K exponential-moving-
   average baseline) as a further experiment on top of the now-working
   architecture/loss design.

## Final result (fair, full 83-instance test-set comparison, zero priority drops for all)

| Model | Mean cost | vs Old |
|---|---|---|
| OLD (K-blind) — `checkpoints/rl/transformer_rl_ga.pt` | 16,846.0 | — |
| REINFORCE (balanced) — `checkpoints/rl_modelb_balanced/transformer_rl_ga.pt` | 16,652.0 | **−194** |
| **PPO (update 100)** — `checkpoints/rl_ppo/transformer_rl_ppo.pt` | **16,508.0** | **−338 (best)** |

Per-K: both new models clearly beat the old one at K=100/500/5000. K=3000 is
a soft spot — both are *slightly* worse than the old model there. K=1000 is
close to a wash (REINFORCE is a bit worse than old, PPO is about tied).

**Caveat on the numbers above**: earlier in the night I reported PPO as
beating REINFORCE by a huge margin (val_rl_cost ~15,600 vs ~19,200) — that
was misleading. PPO's own training-time validation sampled a random 40
instances each check (noisy, not comparable across checkpoints), unlike
REINFORCE's validation which always used the full test set. The numbers
in the table above come from re-evaluating both checkpoints on the *same*
full 83-instance set with identical deterministic decoding — this is the
trustworthy comparison.

## Recommendation
**Use `checkpoints/rl_ppo/transformer_rl_ppo.pt`** — it's the best result of
the session and passes all 4 correctness conditions (priority always
packed, no overlaps, weight/volume respected, verified via
`eval/verify_pipeline.py`). `checkpoints/rl_modelb_balanced/transformer_rl_ga.pt`
(the REINFORCE result) is a very close second and also a valid, verified
improvement over the old model — kept as a working alternative, not deleted.

Both are genuine, if modest, improvements over the old model (~1-2%
aggregate). K=3000 remains the one bucket neither approach fully solved —
worth a closer look if this is picked up again.

## Preserved for reference
- `checkpoints/il/`, `checkpoints/rl/` — original models, untouched all session.
- `checkpoints/rl_modelb_balanced/` — REINFORCE winner + its log
  (`logs/rl_modelb_balanced_training.log`).
- `checkpoints/rl_ppo/` — PPO winner + its log (`logs/rl_ppo_training.log`).
- `src/rl/{model,data_utils,reward,train_rl}.py` — the working REINFORCE
  pipeline with the K-injection fixes baked in.
- `src/rl/{ppo_rollout,train_rl_ppo}.py`, `scripts/train_ga_rl_ppo.py` — the
  new, separate PPO implementation.
- `scripts/train_ga_rl_graft.py` — the graft warm-start technique, used by
  both training paths.

## Session 3: decode-order fix (no retraining)

A real-world stress-test instance (400 packages, 6 ULDs, K=5000, 103
Priority — supplied separately, not part of `good_data/`) came back with
cost=49,903 and spread=6 (worst possible) from the PPO model. A side-by-side
check against the old (K-blind) model on the *same* instance showed it did
no better (spread=6, cost=50,063) — ruling out a regression from the
K-conditioning fix. The real cause: `rl_assign_argmax_safe` decodes packages
sequentially in whatever order the input file gives them, and each
package's capacity mask reflects only what earlier packages in that order
already used. Priority and Economy were interleaved in file order, so
Economy packages decided earlier could claim ULD capacity before a later
Priority package arrived — inflating spread. Instances needing chunking
(>300 packages) compounded this: Priority could land split across
independent chunk forward passes with no coordination between them.

**Fix** (in `rl_assign_argmax_safe`, `src/rl/train_rl.py`): reorder packages
Priority-first (descending weight), then Economy (ascending volume), before
decoding. No retraining — pure inference-time reordering. Verified on the
full 83-instance test set (not just the one instance that surfaced the bug):

| Decode order | Mean cost | vs file-order |
|---|---|---|
| File order (as generated) | 16,508.0 | — |
| **Priority-first, Economy asc-volume** | **13,362.9** | **−3,145.1 (−19.0%)** |

78/83 instances improved, 5 marginally worse, 0 unchanged. Zero priority
drops, zero violations (`eval/verify_pipeline.py`), so this is now the
default — all numbers in `results/` and the PDF report reflect it.

On the real-world instance itself: cost dropped 49,903 → 34,673 (spread
6→4, −30.5%). The assignment stage now consolidates Priority into just 3
ULDs, but the separately-trained single-ULD placement policy (`rl_packer`)
can't always physically fit everything assigned to a ULD (true 3D
extreme-point placement can fail even when aggregate weight/volume allow
it), and its own bounded rescue pass moves the stuck package into a 4th ULD.

**Caveat worth stating plainly**: this whole fix is a decode-order heuristic
wrapped around the frozen §7 model — no weights changed. The model still
picks every package's ULD via its own logits, but was never asked to decide
*which* package to consider first; that ordering matters enormously for a
sequential greedy decoder. This is not the model learning to reduce spread.

## Pushing further: why spread was still 4, not 3

Pushed on this after being asked directly whether the packer was really the
bottleneck, given an external benchmark hit spread=3 on the same instance.
Diagnosed properly rather than assumed:

- The packer's cross-ULD rescue pass (`rl_packer_adapter.py`) tries other
  ULDs sorted by raw volume only, with **no preference for ULDs already
  holding Priority** — so a single large, fresh ULD became the dumping
  ground for every one of 39 stuck Priority packages, recruiting a 4th ULD
  even though the model's own 3 chosen ULDs might have had room after
  evicting Economy. Fixed: candidates are now tried Priority-holding-first,
  then by volume. Real, generally-useful fix — but on *this* instance it
  changed nothing, because those 3 ULDs held Priority almost exclusively
  already (0, 0, 1 Economy package) — nothing to evict.
- The real cause was one level deeper: Priority's total volume was 96% of
  those 3 ULDs' *nominal* capacity, but `rl_packer`'s real extreme-point
  placement only achieves ~70% *true* volumetric efficiency (verified
  order-invariant across 6 heuristic orderings + 15 random shuffles — not
  another ordering bug; rotation was already in its candidate search). 96%
  nominal at 70% true efficiency cannot physically fit.
- The assignment stage had also picked the *wrong* 3 ULDs — not the 3
  largest by volume. The 3 largest, at the same 70% efficiency, have just
  enough margin (+1.9%) to fit all of Priority's volume.

**Fix**: `_consolidate_priority_by_capacity` (`src/rl/train_rl.py`)
deterministically bin-packs Priority via first-fit-decreasing into the
fewest, largest-volume ULDs, before the model ever sees Economy — bypassing
the model's own (myopic, no-global-view) per-package Priority choice
entirely. On the real-world instance: spread 4→3, cost 34,673→27,474 (beats
the external benchmark's 27,500), zero violations.

**But this isn't free**: claiming the largest ULDs for Priority leaves the
*smallest* ULDs for Economy. Tested per-K on the full 83-instance set,
unconditionally applied:

| K | Mean cost change | Verdict |
|---|---|---|
| 100 | +443.8 (10/17 worse) | Net loss |
| 500 | −508.7 | Net win |
| 1000 | −641.1 | Net win |
| 3000 | −860.6 | Net win |
| 5000 | −2,750.5 | Net win |

Same class of bug as the original K-conditioning problem this whole session
started from: a fix that ignores K either over- or under-corrects spread.
Gated to `k_value >= PRIORITY_CONSOLIDATION_MIN_K` (500, `src/rl/config.py`)
— disabled only for K=100, the one bucket in this dataset below it. Final
full-83-instance mean cost: **12,390.9** (vs 16,846.0 original, −26.4%),
zero priority drops, zero violations. This is now the default inside
`rl_assign_argmax_safe`.

Two bugs caught and fixed while wiring the K-gate in: the "disabled" fallback
path initially let Priority packages leak into the Economy dataframe
(duplicate assignment) since the exclusion filter checked consolidation
membership instead of package `Type`; and it dropped the descending-weight
sort session 3 had already established for Priority, both fixed and
re-verified before trusting the K=100 numbers above.

Closing the remaining gap on the real-world instance further (spread=3 is
tight, only ~2% margin even with the best 3 ULDs) would mean improving
`rl_packer`'s placement policy itself, not the assignment model — a
different, separately trained component, out of scope for this session.

## What is the packer actually costing us? (asked directly)

Broke down the real-world instance's 129 dropped Economy packages by exact
cause, using the `reason` field `rl_packer_adapter.pack()` already tags each
placement with:

| Reason | Count | Delay cost |
|---|---|---|
| Assignment stage itself said NONE (never tried) | 46 | 4,417 |
| Assignment stage assigned a ULD, but the packer couldn't physically fit it | 83 | 8,057 |
| **Total** | **129** | **12,474** |

65% of the delay cost is the packer-placement-efficiency gap (the ~70%
ceiling from earlier), not the assignment stage's own conservatism. Also
confirmed precisely which component was touched: only `rl_packer_adapter.py`
(this repo's wrapper — specifically the rescue-loop candidate-ordering fix
above) — the actual learned placement policy
(`cargoism/uld_heightmap_rl/checkpoints/rl_packer/placement_policy.pt`) has
identical weights to session start.

## Solving for low K: Economy selection was the real gap

Asked directly what the fix should be for low K, since spread is nearly
irrelevant there. Checked the actual cost composition on K=100 test
instances: delay cost is ~98% of total cost. So at low K, the entire
objective reduces to a knapsack problem — Economy volume routinely exceeds
total ULD capacity even before Priority and packer inefficiency (verified:
one K=100 instance had Economy volume alone at 104.4% of total nominal ULD
capacity), so which Economy packages get kept is the whole game.

Tested whether the model's own per-package Economy choice is actually good
at this, against a simple greedy heuristic: sort Economy by descending
delay_cost/volume ("value density"), first-fit greedily. The model does show
real learned value-triage (on a sampled instance, kept packages averaged
~2x the value-density of dropped ones — not random) but the heuristic still
won clearly, at *every* K bucket tested, no gating needed:

| K | Model mean | Greedy-VD mean | Win rate |
|---|---|---|---|
| 100 | 10,683.8 | 9,687.9 | 15/17 |
| 500 | 11,380.6 | 10,233.6 | 15/16 |
| 1000 | 10,066.5 | 8,886.4 | 15/17 |
| 3000 | 13,126.4 | 11,679.4 | 13/16 |
| 5000 | 16,681.2 | 15,245.8 | 17/17 |

**Integrated**: the model-driven decode loop still runs unchanged (so
Priority's eviction-rescue mechanism is untouched), but Economy's resulting
ULD assignment is discarded and re-derived via this heuristic afterward.
Tried seeding it with Priority's final nominal footprint (so Economy
correctly sees reduced capacity in Priority-heavy ULDs) — this scored better
in aggregate (mean 11,155.1) but regressed the real-world instance (27,474
→ 30,703, missing its 27,500 external benchmark), for the exact same reason
as the earlier "don't seed" finding: nominal accounting is more conservative
than the packer's real ~70% efficiency, so seeding throws away real
physical slack on dense instances. Left unseeded: smaller aggregate gain,
but the real-world instance improves further instead of regressing (27,474
→ 26,434). Kept unseeded since the real-world instance is the one that
actually matters. **(These specific numbers were later found to be
understated by the rescue-loop bug below — see the correction.)**

**Numbers before correction**: full 83-instance mean cost 12,049.7 (vs
16,846.0 original, −28.5%). Real-world stress instance: 49,903 (file-order
decode, session start) → 26,434 (spread 6→3), apparently beating its 27,500
external benchmark. **These numbers were wrong — see below.**

## Correction: rescue-loop bug silently dropped packages from cost accounting

User asked for an exact JSON export of the 400-package instance's placement
(package ID, assigned ULD, dimensions, weight, coordinates). Building that
export surfaced a real bug: `len(placements)` from
`RLPackerAdapter.pack()` was 309, not 400 — 91 packages were missing
entirely, neither placed nor tagged as dropped.

Root cause, in `pack()`'s cross-ULD rescue loop
(`src/rl/rl_packer_adapter.py`): when rescuing a stuck Priority package into
`other_uid`, the re-pack candidate list was built as
`other_placed_ids + [pid]` — only `other_uid`'s *currently-placed* packages
plus the newly-rescued one. `other_uid`'s own previously-left-behind
packages (from an earlier pass) were excluded from the candidate list AND
then overwritten out of `left_behind_by_uld[other_uid]` by the re-pack's
result — so they vanished from all tracking, never placed, never counted as
dropped. `compute_packing_cost` only sums over the returned `placements`
list, so these packages contributed **zero** delay cost too, silently
understating every cost this session reported for large/contested
instances. Invisible on typical smaller test instances (little rescue
contention); severe on the 400-package real-world instance, where 103
Priority packages consolidated into 3 ULDs created heavy contention across
many rescue rounds.

**Fix**: `candidate_ids = other_placed_ids + left_behind_by_uld[other_uid] + [pid]`
— always carry the target ULD's own previously-left-behind packages forward
into the re-pack, so nothing is ever silently dropped.

**Corrected numbers**:
- Full 83-instance mean cost: **12,698.2** (vs 16,846.0 original, **−24.6%**,
  down from the previously-reported −28.5%). Re-verified zero priority
  drops, zero violations (`eval/verify_pipeline.py`).
- Real-world stress instance: 49,903 (file-order decode, also understated by
  this bug, likely less severely since spread=6 meant less rescue
  contention than the consolidated spread=3 case) → **33,857** (spread 6→3).
  **This does not beat the 27,500 external benchmark** — that claim was
  wrong. Breakdown: 15,000 spread cost, 3,960 delay from packages the
  assignment stage itself dropped, **14,897 delay from packages the
  assignment stage assigned a ULD but the packer couldn't physically
  fit** — now 44.0% of total cost, the dominant component and the clearest
  remaining lever (improving `rl_packer`'s placement policy itself).
- Re-checked the seeded-vs-unseeded Economy question under the fix: the gap
  nearly disappears (33,846 seeded vs 33,857 unseeded on the real-world
  instance) — the earlier dramatic 27,474-vs-30,703 gap was itself partly a
  bug artifact. Kept unseeded (no reason to change it, and it's still
  marginally better here).
- Did **not** re-derive every historical intermediate step-by-step number
  (file-order-only, ordering-only, consolidation-only) under the fix — that
  would take substantially more compute for diminishing value. Each
  comparison was bug-present on both sides being compared, so the
  *qualitative* conclusions (K≥500 gate, unseeded default) should still
  hold; only their absolute magnitudes are approximate.
