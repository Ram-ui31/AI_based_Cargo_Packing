"""Cost evaluation using py3dbp as the packer, mirroring assignment_env.py's
evaluate_assignment() structure exactly (priority-first per ULD, fallback
retry for any dropped priority in whichever other ULD has the most free
volume, then economy) -- so the RL-packer vs py3dbp-packer comparison is
apples-to-apples: same assignment, same priority-preservation policy, only
the geometric packer differs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl_packer", "src"))

from py3dbp import Item

from py3dbp_teacher import _make_bin


def _volume_sorted_items(df) -> list:
    items = [
        Item(str(row["Package_ID"]), int(row["Length"]), int(row["Height"]),
             int(row["Width"]), float(row["Weight"]))
        for _, row in df.iterrows()
    ]
    for it in items:
        it.format_numbers(0)
    items.sort(key=lambda it: it.get_volume(), reverse=True)
    return items


def evaluate_assignment_py3dbp(instance, assignment: dict) -> dict:
    packages_df, ulds_df, K = instance.packages, instance.ulds, instance.K
    pkg_lookup = packages_df.set_index("Package_ID")
    uld_rows = list(ulds_df.itertuples())
    n_ulds = len(uld_rows)

    bins = [_make_bin(ulds_df.iloc[i]) for i in range(n_ulds)]
    for b in bins:
        b.format_numbers(0)

    import py3dbp
    helper = py3dbp.Packer()  # stateless; pack_to_bin only touches (bin, item)

    by_uld = [[] for _ in range(n_ulds)]
    left_behind = set()
    for pid in packages_df["Package_ID"]:
        a = assignment.get(pid)
        if a is None or not (0 <= a < n_ulds):
            left_behind.add(pid)
        else:
            by_uld[a].append(pid)

    def bin_free_volume(b) -> float:
        used = sum(float(it.get_volume()) for it in b.items)
        return float(b.width) * float(b.height) * float(b.depth) - used

    # Phase 1: priority packages first, per their assigned ULD
    dropped_priority = []
    priority_placed_uld: dict[str, int] = {}
    for uld_idx in range(n_ulds):
        prio_ids = [p for p in by_uld[uld_idx] if pkg_lookup.loc[p, "Type"] == "Priority"]
        if not prio_ids:
            continue
        sub = packages_df[packages_df["Package_ID"].isin(prio_ids)]
        for item in _volume_sorted_items(sub):
            helper.pack_to_bin(bins[uld_idx], item)
        placed_names = {it.name for it in bins[uld_idx].items}
        for pid in prio_ids:
            if pid in placed_names:
                priority_placed_uld[pid] = uld_idx
            else:
                dropped_priority.append(pid)

    # Fallback: retry dropped priority in whichever other ULD has the most free volume
    still_dropped_priority = []
    for pid in dropped_priority:
        row = packages_df[packages_df["Package_ID"] == pid]
        item = _volume_sorted_items(row)[0]
        order = sorted(range(n_ulds), key=lambda i: -bin_free_volume(bins[i]))
        placed_ok = False
        for uld_idx in order:
            helper.pack_to_bin(bins[uld_idx], item)  # appends to bin.items on success, else bin.unfitted_items
            if any(it.name == item.name for it in bins[uld_idx].items):
                priority_placed_uld[pid] = uld_idx
                placed_ok = True
                break
        if not placed_ok:
            still_dropped_priority.append(pid)

    # Phase 2: economy packages into their assigned (possibly already partly-filled) ULD
    for uld_idx in range(n_ulds):
        econ_ids = [p for p in by_uld[uld_idx] if pkg_lookup.loc[p, "Type"] == "Economy"]
        if not econ_ids:
            continue
        sub = packages_df[packages_df["Package_ID"].isin(econ_ids)]
        for item in _volume_sorted_items(sub):
            helper.pack_to_bin(bins[uld_idx], item)
        placed_names = {it.name for it in bins[uld_idx].items}
        for pid in econ_ids:
            if pid not in placed_names:
                left_behind.add(pid)

    delay_cost = float(pkg_lookup.loc[list(left_behind), "Delay_Cost"].sum()) if left_behind else 0.0
    spread = len(set(priority_placed_uld.values()))
    cost = K * spread + delay_cost

    utilization = []
    for b, u in zip(bins, uld_rows):
        vol = int(u.Length) * int(u.Width) * int(u.Height)
        used = sum(float(it.get_volume()) for it in b.items)
        utilization.append(used / vol if vol else 0.0)

    return dict(
        cost=cost, spread=spread, delay_cost=delay_cost,
        left_behind=sorted(left_behind), priority_dropped=still_dropped_priority,
        n_priority=int((packages_df["Type"] == "Priority").sum()),
        utilization=utilization,
    )
