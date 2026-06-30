# ─────────────────────────────────────────────────────────────────────────────
# LABELLER STRATEGY PATTERN
#
# A Labeller produces {Package_ID: ULD_ID | 'NONE'} assignments used as
# training labels for the IL model.
#
# To retrain on a different heuristic, subclass Labeller and pass it to
# ClusteringDataset(). The Transformer architecture and training loop are
# unchanged.
# ─────────────────────────────────────────────────────────────────────────────

class Labeller:
    """Abstract base for label-generation strategies."""
    def label(self, packages_df, ulds_df):
        """
        Returns:
            assignment : {Package_ID: ULD_ID | 'NONE'}
        """
        raise NotImplementedError


class GreedyLabeller(Labeller):
    """
    Greedy heuristic that assigns packages to ULDs respecting weight and volume limits.

    Sort order:
        1. Priority packages first
        2. Within each group: descending volume

    Assignment logic per package:
        - Try ULDs in order of remaining capacity (best-fit by remaining volume)
        - A ULD is eligible if:
            (a) package physically fits (sorted dim check)
            (b) adding it does not exceed weight_limit
            (c) adding it does not exceed volume capacity
        - If no eligible ULD exists:
            - Economy -> NONE (expected, priced into cost)
            - Priority -> NONE + log warning (should not happen in well-formed data)
    """

    def label(self, packages_df, ulds_df):
        uld_records   = ulds_df.to_dict('records')
        n_ulds        = len(uld_records)
        weight_used   = [0.0] * n_ulds
        volume_used   = [0.0] * n_ulds
        uld_volumes   = [u['Length']*u['Width']*u['Height'] for u in uld_records]
        uld_wt_limits = [u['Weight_Limit'] for u in uld_records]
        uld_ids       = ulds_df['ULD_ID'].tolist()

        # Sort: Priority first, then Economy; within each group largest volume first
        pkgs = packages_df.copy()
        pkgs['_vol']      = pkgs['Length'] * pkgs['Width'] * pkgs['Height']
        pkgs['_prio_key'] = (pkgs['Type'] == 'Priority').astype(int)
        pkgs = pkgs.sort_values(['_prio_key', '_vol'], ascending=[False, False])

        assignment = {}

        for _, pkg in pkgs.iterrows():
            pid    = pkg['Package_ID']
            pl, pw, ph = pkg['Length'], pkg['Width'], pkg['Height']
            pw_    = pkg['Weight']
            pvol   = pl * pw * ph
            pd_    = sorted([pl, pw, ph])

            # Find eligible ULDs sorted by remaining volume (best-fit)
            eligible = []
            for j, uld in enumerate(uld_records):
                ud_ = sorted([uld['Length'], uld['Width'], uld['Height']])
                fits_dim    = all(pd_[k] <= ud_[k] for k in range(3))
                fits_weight = weight_used[j] + pw_ <= uld_wt_limits[j]
                fits_volume = volume_used[j] + pvol <= uld_volumes[j]
                if fits_dim and fits_weight and fits_volume:
                    remaining = uld_volumes[j] - volume_used[j]
                    eligible.append((remaining, j))

            if eligible:
                # Best-fit: choose ULD with least remaining space that still fits
                eligible.sort(key=lambda x: x[0])
                _, chosen_j = eligible[0]
                assignment[pid]         = uld_ids[chosen_j]
                weight_used[chosen_j]  += pw_
                volume_used[chosen_j]  += pvol
            else:
                assignment[pid] = 'NONE'
                if pkg['Type'] == 'Priority':
                    print(f'  WARNING: Priority package {pid} assigned NONE '
                          f'- no eligible ULD found')

        return assignment


# ── Default labeller (swap this to retrain on a different heuristic) ──────────
DEFAULT_LABELLER = GreedyLabeller()
