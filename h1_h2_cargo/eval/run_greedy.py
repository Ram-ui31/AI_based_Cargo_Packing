"""
run_greedy.py — batch runner for the Greedy Heuristic Pipeline.

Usage
-----
python run_greedy.py --data-root data --split generated_test --out results/test
python run_greedy.py --data-root data --split toy_example   --out results/toy
python run_greedy.py --data-root data --split generated_test --out results/test --limit 5
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dataset_io import (
    ProblemInstance,
    load_toy_example,
    load_split_instance,
    list_instance_names,
    load_metadata_csv,
)
from greedy_pack import PackResult
from greedy_pipeline import GreedyPipeline


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def format_solution(result: PackResult, packages, k_penalty: float) -> str:
    delay_costs = {p.id: p.delay_cost for p in packages}
    total_cost  = result.total_cost(delay_costs, k_penalty)
    lines = [
        f"{total_cost:g}, {len(result.placed_boxes)}, {result.uld_priority_count()}"
    ]
    for box in result.placed_boxes:
        x0, y0, z0, x1, y1, z1 = box.as_output_tuple()
        lines.append(
            f"{box.package_id}, {box.uld_id}, "
            f"{x0:g}, {y0:g}, {z0:g}, {x1:g}, {y1:g}, {z1:g}"
        )
    # left_behind = economy not placed; unplaceable = priority not placed
    for pid in result.left_behind + result.unplaceable:
        lines.append(f"{pid}, NONE, -1, -1, -1, -1, -1, -1")
    return "\n".join(lines) + "\n"


def write_solution(result: PackResult, packages, k_penalty: float, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_solution(result, packages, k_penalty))


def summarize(instance_id: str, result: PackResult,
              packages, k_penalty: float, elapsed: float) -> Dict:
    delay_costs  = {p.id: p.delay_cost for p in packages}
    placed_ids   = {b.package_id for b in result.placed_boxes}
    n_priority   = sum(1 for p in packages if p.is_priority)
    n_pri_placed = sum(1 for p in packages if p.is_priority and p.id in placed_ids)
    return {
        "instance":          instance_id,
        "n_packages":        len(packages),
        "n_priority":        n_priority,
        "n_priority_placed": n_pri_placed,
        "all_priority_ok":   n_pri_placed == n_priority,
        "n_placed":          len(result.placed_boxes),
        "n_left_behind":     len(result.left_behind),
        "n_unplaceable":     len(result.unplaceable),
        "n_priority_ulds":   result.uld_priority_count(),
        "total_cost":        result.total_cost(delay_costs, k_penalty),
        "infeasible":        result.infeasible,
        "elapsed_seconds":   round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# Single-instance solver
# ---------------------------------------------------------------------------

def run_one(
    instance:           ProblemInstance,
    out_dir:            Path,
    candidates_per_uld: int   = 3,
    fill_target:        float = 1.2,
    pack_threshold:     float = 0.80,
) -> Dict:
    t0 = time.time()
    pipeline = GreedyPipeline(
        ulds=instance.ulds,
        packages=instance.packages,
        k_penalty=instance.k_penalty,
        candidates_per_uld=candidates_per_uld,
        fill_target=fill_target,
        pack_threshold=pack_threshold,
    )
    result  = pipeline.solve()
    elapsed = time.time() - t0

    write_solution(
        result, instance.packages, instance.k_penalty,
        out_dir / f"{instance.instance_id}_solution.txt",
    )
    return summarize(instance.instance_id, result,
                     instance.packages, instance.k_penalty, elapsed)


# ---------------------------------------------------------------------------
# Worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _worker(args) -> Dict:
    instance, out_dir, cands, fill, thresh = args
    try:
        return run_one(instance, out_dir, cands, fill, thresh)
    except Exception as e:
        return {"instance": instance.instance_id,
                "error": str(e),
                "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Split runner
# ---------------------------------------------------------------------------

def run_split(
    data_root:          Path,
    split:              str,
    out_dir:            Path,
    candidates_per_uld: int           = 3,
    fill_target:        float         = 1.2,
    pack_threshold:     float         = 0.80,
    limit:              Optional[int] = None,
    n_workers:          Optional[int] = None,
) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or os.cpu_count() or 1

    names = list_instance_names(data_root, split)
    if limit:
        names = names[:limit]
    total = len(names)

    meta      = load_metadata_csv(data_root / split / "metadata.csv")
    instances = [load_split_instance(data_root, split, n, meta) for n in names]
    work      = [(inst, out_dir, candidates_per_uld, fill_target, pack_threshold)
                 for inst in instances]

    rows: List[Dict] = []
    done = 0

    if n_workers == 1:
        for args in work:
            row = _worker(args)
            done += 1
            _print_row(row, done, total)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, a): a[0].instance_id for a in work}
            for f in as_completed(futures):
                row = f.result()
                done += 1
                _print_row(row, done, total)
                rows.append(row)

    order = {n: i for i, n in enumerate(names)}
    rows.sort(key=lambda r: order.get(r.get("instance", ""), 9999))

    if rows:
        csv_path = out_dir / "results.csv"
        keys = [k for k in rows[0] if k != "traceback"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults → {csv_path}")

    return rows


def _print_row(row: Dict, done: int, total: int):
    if "error" in row:
        print(f"[{done}/{total}] {row['instance']}: ERROR — {row['error']}")
        return
    flags = []
    if row.get("infeasible"):
        flags.append("INFEASIBLE")
    if not row.get("all_priority_ok"):
        flags.append(f"PRIORITY MISSING {row['n_priority']-row['n_priority_placed']}")
    flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
    print(
        f"[{done}/{total}] {row['instance']}: "
        f"cost={row['total_cost']:g}  "
        f"priority {row['n_priority_placed']}/{row['n_priority']}  "
        f"placed {row['n_placed']}/{row['n_packages']}  "
        f"{row['elapsed_seconds']}s{flag_str}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Greedy Heuristic Pipeline — no tree search.")
    p.add_argument("--data-root",  type=Path, required=True)
    p.add_argument("--split",
                   choices=["generated_test", "synthetic_train", "toy_example"],
                   default="generated_test")
    p.add_argument("--out",        type=Path, required=True)
    p.add_argument("--candidates", type=int,  default=3,
                   help="Placements scored per ULD per package (default 3)")
    p.add_argument("--fill-target",    type=float, default=1.2)
    p.add_argument("--pack-threshold", type=float, default=0.80)
    p.add_argument("--limit",   type=int,  default=None)
    p.add_argument("--workers", type=int,  default=None,
                   help="Parallel processes (use 1 on Colab)")
    args = p.parse_args()

    if args.split == "toy_example":
        instance = load_toy_example(args.data_root)
        row = run_one(instance, args.out, args.candidates,
                      args.fill_target, args.pack_threshold)
        for k, v in row.items():
            print(f"  {k:<26} {v}")
    else:
        run_split(
            args.data_root, args.split, args.out,
            args.candidates, args.fill_target, args.pack_threshold,
            args.limit, args.workers,
        )


if __name__ == "__main__":
    main()
