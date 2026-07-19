"""
precompute_ga_cache.py — parallel precompute of GALabeller assignments.

The GA solver is the most expensive label source in this pipeline (population
x generations trial-packs per instance), so precompute in parallel across CPU
cores rather than solving live during epoch 1 of IL training. Pickles a
{(tag, uld_chunk_idx, pkg_chunk_idx): assignment} cache, same shape as
good-il-over-greedy(c)/scripts/precompute_h1h2_cache.py.

Usage:
    python precompute_ga_cache.py --data-root ~/Desktop/good_data --out ../cache/ga_cache.pkl
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_THIS_DIR, '..')
sys.path.insert(0, _ROOT_DIR)
from src.il.data_utils import chunk_dataframe             # noqa: E402
from src.il.config import MAX_SAFE_PKGS, MAX_SAFE_ULDS     # noqa: E402


def _solve_chunk(tag: str, uld_chunk_idx: int, pkg_chunk_idx: int,
                  pkgs_df: pd.DataFrame, ulds_df: pd.DataFrame,
                  pop_size: int, max_generations: int, patience: int,
                  time_budget_seconds: float):
    """Runs in a worker process: fresh GALabeller (fresh sys.path insert)."""
    ga_src = os.path.join(_THIS_DIR, '..', 'src', 'ga')
    if ga_src not in sys.path:
        sys.path.insert(0, ga_src)
    from ga_labeller import GALabeller

    labeller = _solve_chunk._labeller
    if labeller is None:
        labeller = GALabeller(pop_size=pop_size, max_generations=max_generations,
                               patience=patience, time_budget_seconds=time_budget_seconds)
        _solve_chunk._labeller = labeller
    key = (tag, uld_chunk_idx, pkg_chunk_idx)
    t0 = time.time()
    assignment = labeller.label(pkgs_df, ulds_df, tag=tag,
                                 pkg_chunk_idx=pkg_chunk_idx, uld_chunk_idx=uld_chunk_idx)
    return key, assignment, len(pkgs_df), time.time() - t0


_solve_chunk._labeller = None


def iter_chunks(data_dir: str, meta_path: str):
    # Namespace by data_dir: synthetic_train and synthetic_test both use their
    # own independent 'instance_000'... sequence, so the bare tag collides
    # across splits. Must match data_utils.ClusteringDataset._load's split_tag.
    split_tag_prefix = os.path.basename(os.path.normpath(data_dir))
    meta = pd.read_csv(meta_path)
    for _, row in meta.iterrows():
        raw_tag = row['instance']
        tag = f'{split_tag_prefix}/{raw_tag}'
        pkgs_df = pd.read_csv(os.path.join(data_dir, f'{raw_tag}_packages.csv'))
        ulds_df = pd.read_csv(os.path.join(data_dir, f'{raw_tag}_ulds.csv'))
        for ui, uld_chunk in enumerate(chunk_dataframe(ulds_df, MAX_SAFE_ULDS)):
            for pi, pkg_chunk in enumerate(chunk_dataframe(pkgs_df, MAX_SAFE_PKGS)):
                yield tag, ui, pi, pkg_chunk, uld_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='good_data root (has synthetic_train/, synthetic_test/)')
    ap.add_argument('--out', required=True, help='output pickle path')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--pop-size', type=int, default=16)
    ap.add_argument('--max-generations', type=int, default=20)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--time-budget-seconds', type=float, default=90.0,
                     help='hard wall-clock cap per chunk on the GA loop itself (ga_solver.run_ga); '
                          'guards against pathological instances regardless of pop-size/max-generations')
    ap.add_argument('--max-instances', type=int, default=None, help='cap instances per split (smoke-testing only)')
    ap.add_argument('--checkpoint-every', type=int, default=25,
                     help='flush partial cache to --out every N completions, so an interrupted run keeps its progress')
    args = ap.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    jobs = []
    for split in ['synthetic_train', 'synthetic_test']:
        data_dir  = os.path.join(data_root, split)
        meta_path = os.path.join(data_dir, 'metadata.csv')
        chunks = list(iter_chunks(data_dir, meta_path))
        if args.max_instances is not None:
            seen_tags_ordered = list(dict.fromkeys(c[0] for c in chunks))
            allowed_tags = set(seen_tags_ordered[:args.max_instances])
            chunks = [c for c in chunks if c[0] in allowed_tags]
        jobs.extend(chunks)

    print(f'Total chunks to solve: {len(jobs)} across {args.workers} workers '
          f'(pop_size={args.pop_size}, max_generations={args.max_generations}, '
          f'time_budget_seconds={args.time_budget_seconds})', flush=True)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _checkpoint():
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(cache, f)
        os.replace(tmp_path, out_path)

    cache = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_solve_chunk, tag, ui, pi, pkg_chunk, uld_chunk,
                      args.pop_size, args.max_generations, args.patience,
                      args.time_budget_seconds): tag
            for tag, ui, pi, pkg_chunk, uld_chunk in jobs
        }
        for fut in as_completed(futures):
            tag = futures[fut]
            try:
                key, assignment, n_pkgs, chunk_dt = fut.result()
                assert key not in cache, f'duplicate key {key}'
                cache[key] = assignment
                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(jobs) - done) / rate if rate > 0 else float('inf')
                print(f'  [{done}/{len(jobs)}] {tag}  n_pkgs={n_pkgs}  '
                      f'chunk_time={chunk_dt:.1f}s  elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)
            except Exception as e:
                done += 1
                print(f'  [{done}/{len(jobs)}] ERROR on {tag}: {e}', flush=True)

            if done % args.checkpoint_every == 0 or done == len(jobs):
                _checkpoint()

    print(f'Saved {len(cache)} cached assignments -> {out_path}  (total {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
