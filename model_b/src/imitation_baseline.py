"""Greedy heuristic assignment used purely as an imitation-learning target
for pretraining the assignment policy -- not used at inference time."""

from __future__ import annotations

from assignment_policy import dim_fits


def greedy_assign(packages_df, ulds_df) -> dict:
    n_ulds = len(ulds_df)
    uld_rows = list(ulds_df.itertuples())
    uld_weight_limit = [float(u.Weight_Limit) for u in uld_rows]
    uld_volume = [int(u.Length * u.Width * u.Height) for u in uld_rows]
    running_weight = [0.0] * n_ulds
    running_volume = [0] * n_ulds

    pkgs = packages_df.copy()
    pkgs["volume"] = pkgs["Length"] * pkgs["Width"] * pkgs["Height"]
    prio = pkgs[pkgs["Type"] == "Priority"].sort_values("volume", ascending=False)
    econ = pkgs[pkgs["Type"] == "Economy"].sort_values("volume", ascending=False)

    def try_assign(row, force: bool):
        order = sorted(range(n_ulds), key=lambda i: -(uld_volume[i] - running_volume[i]))
        for i in order:
            u = uld_rows[i]
            if not dim_fits(row.Length, row.Width, row.Height, u.Length, u.Width, u.Height):
                continue
            if running_weight[i] + row.Weight > uld_weight_limit[i]:
                continue
            if running_volume[i] + row.volume > uld_volume[i]:
                continue
            running_weight[i] += row.Weight
            running_volume[i] += row.volume
            return i
        if force:
            i = max(range(n_ulds), key=lambda i: uld_weight_limit[i] - running_weight[i])
            running_weight[i] += row.Weight
            running_volume[i] += row.volume
            return i
        return None

    assignment: dict[str, int | None] = {}
    for row in prio.itertuples():
        assignment[row.Package_ID] = try_assign(row, force=True)
    for row in econ.itertuples():
        assignment[row.Package_ID] = try_assign(row, force=False)
    return assignment
