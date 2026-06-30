"""
Batch runner — optimised edition.

Changes vs original
-------------------
1.  Parallel instance solving via concurrent.futures.ProcessPoolExecutor.
    Instances are independent, so they parallelize perfectly.  Wall-clock
    time for a 50-instance split drops from ~50 × T_single to
    ~T_single × ceil(50 / n_workers), where n_workers defaults to
    os.cpu_count().  A machine with 8 cores cuts wall time by ~8×.

    ProcessPoolExecutor (not ThreadPoolExecutor) is used because the engine
    is CPU-bound; threads cannot parallelize CPU work in Python due to the GIL.

2.  Ordered progress: results are printed in completion order (as_completed),
    not submission order, so you get live feedback rather than a long silence
    followed by a burst of output.

3.  Worker error isolation: a single instance crash is caught and logged
    without killing the whole run.  The summary CSV still includes that row
    with an "error" marker.

4.  run_one() is unchanged in signature so notebooks importing it directly
    are unaffected.
"""

from __future__ import annotations
import argparse
import csv
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from dataset_io import (
    ProblemInstance,
    load_toy_example,
    load_split_instance,
    iter_split,
    list_instance_names,
    load_metadata_csv,
)
from output_io import write_solution, summarize_result
from tree_search import PackingEngine
from greedy_pipeline import GreedyPipeline


# ---------------------------------------------------------------------------
# Single-instance solver (must be importable at module level for pickling)
# ---------------------------------------------------------------------------

def run_one(instance: ProblemInstance, out_dir: Path,
            beam_width: int = 8, candidates_per_uld: int = 5,
            priority_beam_factor: int = 2,
            strategy: str = "greedy") -> Dict:
    """
    Solves one instance, writes its solution file, returns a summary dict.

    strategy:
        "greedy"  — Greedy Heuristic Pipeline (H1 → Binary Search → Pack →
                    H2 → Pack). Recommended for large instances.
        "beam"    — Original beam-search engine only (no heuristic pre-sort).
    """
    t0 = time.time()

    if strategy == "greedy":
        solver = GreedyPipeline(
            ulds=instance.ulds,
            packages=instance.packages,
            k_penalty=instance.k_penalty,
            beam_width=beam_width,
            candidates_per_uld=candidates_per_uld,
            priority_beam_factor=priority_beam_factor,
        )
    else:
        solver = PackingEngine(
            ulds=instance.ulds,
            packages=instance.packages,
            k_penalty=instance.k_penalty,
            beam_width=beam_width,
            candidates_per_uld=candidates_per_uld,
            priority_beam_factor=priority_beam_factor,
        )

    best = solver.solve()
    elapsed = time.time() - t0

    write_solution(
        best, instance.packages, instance.k_penalty,
        out_dir / f"{instance.instance_id}_solution.txt",
    )
    return summarize_result(
        instance.instance_id, best, instance.packages,
        instance.k_penalty, elapsed,
    )


def _worker(args) -> Dict:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    instance, out_dir, beam_width, candidates_per_uld, priority_beam_factor, strategy = args
    try:
        return run_one(instance, out_dir, beam_width,
                       candidates_per_uld, priority_beam_factor, strategy)
    except Exception as e:
        return {
            "instance": instance.instance_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Split runner
# ---------------------------------------------------------------------------

def run_split(
    data_root: Path,
    split: str,
    out_dir: Path,
    beam_width: int = 8,
    candidates_per_uld: int = 5,
    priority_beam_factor: int = 2,
    limit: Optional[int] = None,
    n_workers: Optional[int] = None,
    strategy: str = "greedy",
) -> List[Dict]:
    """
    Solve every instance in a split in parallel.

    n_workers  : number of parallel processes (default = os.cpu_count()).
                 Pass n_workers=1 to disable parallelism (useful for debugging
                 or when the overhead of spawning processes outweighs the gain,
                 e.g. on a 2-core machine with very fast instances).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or os.cpu_count() or 1

    instance_names = list_instance_names(data_root, split)
    if limit is not None:
        instance_names = instance_names[:limit]
    total = len(instance_names)

    metadata_cache = load_metadata_csv(data_root / split / "metadata.csv")

    # Load all instances up-front (fast — just CSV parsing)
    instances = [
        load_split_instance(data_root, split, name, metadata_cache)
        for name in instance_names
    ]

    work = [
        (inst, out_dir, beam_width, candidates_per_uld, priority_beam_factor, strategy)
        for inst in instances
    ]

    rows: List[Dict] = []
    completed = 0

    if n_workers == 1:
        # Single-process path: simpler stack traces when debugging
        for args in work:
            row = _worker(args)
            completed += 1
            _print_row(row, completed, total)
            rows.append(row)
    else:
        future_to_name = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for args in work:
                future_to_name[pool.submit(_worker, args)] = args[0].instance_id
            for future in as_completed(future_to_name):
                row = future.result()
                completed += 1
                _print_row(row, completed, total)
                rows.append(row)

    # Sort output rows back into instance order for a stable CSV
    name_order = {name: i for i, name in enumerate(instance_names)}
    rows.sort(key=lambda r: name_order.get(r.get("instance", ""), 9999))

    results_path = out_dir / "results.csv"
    if rows:
        all_keys = list(rows[0].keys())
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {results_path}")

    return rows


def _print_row(row: Dict, completed: int, total: int) -> None:
    if "error" in row:
        print(f"[{completed}/{total}] {row['instance']}: ERROR — {row['error']}")
    else:
        print(
            f"[{completed}/{total}] {row['instance']}: "
            f"cost={row['total_cost']:g}  "
            f"priority {row['n_priority_placed']}/{row['n_priority']}  "
            f"placed {row['n_placed']}/{row['n_packages']}  "
            f"{row['elapsed_seconds']}s"
            + ("  [INFEASIBLE]" if row.get("infeasible") else "")
        )


# ---------------------------------------------------------------------------
# Toy example
# ---------------------------------------------------------------------------

def run_toy_example(data_root: Path, out_dir: Path,
                     beam_width: int = 8, candidates_per_uld: int = 6,
                     priority_beam_factor: int = 2) -> Dict:
    instance = load_toy_example(data_root)
    return run_one(instance, out_dir, beam_width,
                   candidates_per_uld, priority_beam_factor)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the packing engine over a dataset split (parallel).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split",
                        choices=["generated_test", "synthetic_train", "toy_example"],
                        default="generated_test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--candidates-per-uld", type=int, default=5)
    parser.add_argument("--priority-beam-factor", type=int, default=2,
                        help="Beam multiplier for Priority package steps (default 2)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strategy", choices=["greedy", "beam"], default="greedy",
                        help="'greedy' = H1→BinSearch→Pack→H2→Pack pipeline (default). 'beam' = original beam search only.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default = CPU count)")
    args = parser.parse_args()

    if args.split == "toy_example":
        row = run_toy_example(args.data_root, args.out,
                               args.beam_width, args.candidates_per_uld,
                               args.priority_beam_factor)
        print(row)
    else:
        run_split(
            args.data_root, args.split, args.out,
            args.beam_width, args.candidates_per_uld,
            args.priority_beam_factor,
            args.limit, args.workers,
            strategy=args.strategy,
        )


if __name__ == "__main__":
    main()
