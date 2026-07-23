# Running Cherry, Eclipse, and Halley on your own instance

This folder documents how to run the three final models — [`01-cherry/`](../01-cherry/), [`02-eclipse/`](../02-eclipse/), [`03-halley/`](../03-halley/) — on a CSV instance file you provide, using each model's own already-trained checkpoints (no training or fine-tuning required).

## 1. Setup

Clone the **whole repository** (not just one model's folder) — each model's run script references a couple of sibling folders for shared code (details below), which must be present alongside it exactly as cloned.

```bash
git clone https://github.com/Ram-ui31/AI_based_Cargo_Packing.git
cd AI_based_Cargo_Packing
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install torch pandas numpy
```

Tested with Python 3.13, torch 2.9, pandas 2.3, numpy 2.3 — any reasonably recent version of each (Python ≥3.9, torch ≥2.0) should work; none of the scripts use exotic APIs.

## 2. Prepare your instance CSV

Each script expects the same plain-text format used throughout this project (matches `~/Downloads/input.csv` from the original problem statement):

```
5000

U1,224,318,162,2500
U2,224,318,162,2500
U3,244,318,244,2800
...

P-1,99,53,55,61,Economy,176
P-2,56,99,81,53,Priority,-
...
```

- **Line 1**: the K value (delay-cost multiplier), alone on its own line.
- **Blank line**, then one row per ULD: `ULD_ID,Length,Width,Height,Weight_Limit`.
- **Blank line**, then one row per package: `Package_ID,Length,Width,Height,Weight,Type,Delay_Cost` — `Type` is `Priority` or `Economy`; `Delay_Cost` is `-` for Priority packages (they have none — Priority is a hard constraint, not a cost).

Blank-line placement doesn't need to be exact — each parser classifies every non-blank line purely by its field count (5 fields = ULD, 7 fields = package), so it's robust to minor formatting differences.

## 3. Run a model

Each model's run script lives inside its own folder and is invoked the same way:

```bash
# Cherry (best result, includes the centrifuge-evict refinement)
python3 01-cherry/run_cherry.py --input /path/to/your_instance.csv

# Eclipse (RL placement policy, no refinement)
python3 02-eclipse/run_eclipse.py --input /path/to/your_instance.csv

# Halley (GRPO-trained economy ordering, no refinement)
python3 03-halley/run_halley.py --input /path/to/your_instance.csv
```

Each prints a live progress log, then a final summary:

```
Parsed 6 ULDs, 400 packages (103 Priority, 297 Economy), K=5,000
Running local search (15 rounds, real-evaluated)...
  round 0 (initial): cost=29,741
  ...
Running centrifuge-evict refinement (exhaustive)...
  cycle 1: evict P-80 from U1, compact, refill -- net gain 286
  ...

=== Cherry result ===
Total cost: 29,340  (delay=14,340, spread=15,000)
Priority ULDs used: 3
Priority placed: 103/103
Economy placed: 140/297
Wall time: 48.2s

Saved results_judge/final_metrics.json and final_placements.json
```

`final_metrics.json` has the cost breakdown; `final_placements.json` has every package's ULD assignment and, if placed, its exact 3D position (`x0,y0,z0,x1,y1,z1`) and orientation.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--input` | *(required)* | Path to your instance CSV. |
| `--output-dir` | `results_judge/` inside the model's own folder | Where the two output JSON files are saved. |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) if available — the RL placement policy and rankers are small enough that CPU is fine for a single instance. |
| `--search-rounds` | `15` (Cherry/Eclipse/Halley all support this) | Local-search rounds after the initial assignment. Each round is a real, full re-pack of the instance — more rounds costs more time but generally finds a better result. Set to `0` to skip local search entirely and just use the one-shot assignment. |

Cherry's centrifuge-evict refinement always runs after local search (it doesn't have a flag to disable it, since it's Cherry's whole reason for existing) and typically takes well under a minute even on the full 400-package benchmark instance.

## 4. What each script actually does

All three share the same first stage, using only that model's own bundled checkpoints (`checkpoints/priority_clusterer.pt`, `checkpoints/rl_placement_policy.pt`):

1. **Priority clustering** — the trained Priority Clusterer decides which ULDs hold Priority cargo.
2. **Priority packing** — exhaustive, guaranteed placement. All three scripts print a loud warning if this ever fails on your instance (it shouldn't, but if your instance has, e.g., a single Priority package that's physically larger than every ULD, no algorithm can place it — that's a genuinely infeasible instance, not a bug).
3. **Economy ordering** — a value-density formula for Eclipse and Cherry; the trained `PackageSetRanker` network for Halley (`checkpoints/halley_economy_ranker.pt`).
4. **Best-of-5 packing ensemble** — the trained RL placement policy plus four geometric heuristic strategies compete per-ULD; whichever packs a given container best wins it.
5. **Local search** — a simplified, self-contained hill-climbing search (swap/relocate moves, real-evaluated) over the Economy assignment. This is a faster, more compact version of the research-grade local search described in each model's own README — it will generally improve on the one-shot assignment but won't necessarily reach the exact numbers quoted in this project's writeups, which came from a much more extensive multi-hour search process. Increase `--search-rounds` for a better (slower) result.
6. **(Cherry only) Centrifuge-evict refinement** — exhaustively tests evicting each already-placed Economy package, compacting its container, and refilling from the unplaced pool, keeping any net-improving move. This script runs the exhaustive version (every placed package tried, not a trained model's top-K shortlist), so it's correct by construction, just slower than the shortlisted version on very large instances.

## 5. Why the whole repo, not just one folder

Each script is self-contained except for two small, shared, non-weight dependencies also in this repo:

- `rl_packer/src/` — the shared 3D placement-policy geometry and environment code (used by all three).
- `economy-package-ranker/src/` — only needed by Halley, for the `PackageSetRanker` class definition (the checkpoint alone isn't enough to load it without the class).

Neither of these carries model weights of its own — all trained weights live in each model's own `checkpoints/` folder — but the class/geometry code they define is imported by the run scripts, so it needs to be present at the expected relative path, which is automatic if you clone the full repository rather than downloading a single subfolder.

## 6. Expected runtime

On the real 400-package benchmark instance (6 ULDs) on a standard laptop CPU:

| Stage | Approx. time |
|---|---|
| Clustering + one-shot assignment + packing | ~20s |
| Local search, per round | ~15–20s |
| Centrifuge refinement (Cherry), per cycle | ~15s |

Runtime scales with instance size (more packages/ULDs → more per-round packing work) and with `--search-rounds`. For a quick sanity check on a new instance, `--search-rounds 0` gives the one-shot result in well under a minute.
