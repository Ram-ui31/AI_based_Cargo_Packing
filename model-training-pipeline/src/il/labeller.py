# ─────────────────────────────────────────────────────────────────────────────
# LABELLER STRATEGY PATTERN (same shape as good-il-over-greedy(c)/src/labeller.py)
#
# A Labeller produces {Package_ID: ULD_ID | 'NONE'} assignments used as
# training labels for the IL model. This package's only Labeller wraps
# src.ga.GALabeller (the Genetic Algorithm split, see
# model-training-pipeline/src/ga/ga_labeller.py) instead of a greedy heuristic.
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys


class Labeller:
    """Abstract base for label-generation strategies."""
    def label(self, packages_df, ulds_df, tag=None, pkg_chunk_idx=None, uld_chunk_idx=None):
        raise NotImplementedError


class GALabellerAdapter(Labeller):
    """
    Wraps src.ga.ga_labeller.GALabeller (the Genetic Algorithm economy-split
    solver) as a Labeller for this package's ClusteringDataset/train_il.

    cache : optional {(tag, uld_chunk_idx, pkg_chunk_idx): assignment}
        precomputed by scripts/precompute_ga_cache.py.
    """

    def __init__(self, cache=None, pop_size=16, max_generations=20,
                 patience=6, gene_contribution_ratio=0.65, seed=None,
                 time_budget_seconds=90.0, ga_src=None):
        if ga_src is None:
            ga_src = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'ga'
            )
        ga_src = os.path.abspath(ga_src)
        if ga_src not in sys.path:
            sys.path.insert(0, ga_src)

        from ga_labeller import GALabeller  # noqa: F401
        self._ga = GALabeller(
            cache=cache, pop_size=pop_size, max_generations=max_generations,
            patience=patience, gene_contribution_ratio=gene_contribution_ratio, seed=seed,
            time_budget_seconds=time_budget_seconds,
        )

    def label(self, packages_df, ulds_df, tag=None, pkg_chunk_idx=None, uld_chunk_idx=None):
        return self._ga.label(packages_df, ulds_df, tag=tag,
                               pkg_chunk_idx=pkg_chunk_idx, uld_chunk_idx=uld_chunk_idx)


# ── Default labeller ───────────────────────────────────────────────────────────
DEFAULT_LABELLER = GALabellerAdapter()
