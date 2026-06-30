"""
uld_partition.py — Step 1 of the Greedy Heuristic pipeline.

FIX: previous version could consume ALL ULDs into the priority bucket when
priority volume was large, leaving other_ulds=[] and no space for Set 2
economy packages. Now always reserves at least one ULD for other packages
unless there is genuinely only one ULD in total.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from geometry import Package, ULD


@dataclass
class ULDPartition:
    priority_ulds: List[ULD]
    other_ulds: List[ULD]

    @property
    def all_ulds(self) -> List[ULD]:
        return self.priority_ulds + self.other_ulds


def partition_ulds(
    ulds: List[ULD],
    packages: List[Package],
    fill_target: float = 1.2,
    min_priority_ulds: int = 1,
    max_priority_fraction: float = 0.75,  # never take more than this share of ULDs
) -> ULDPartition:
    """
    Split ULDs into priority-reserved and open buckets.

    Parameters
    ----------
    fill_target :
        Reserve enough ULD volume to hold fill_target * priority_volume.
        1.2 = 20 % headroom.
    min_priority_ulds :
        Always reserve at least this many ULDs for priority packages.
    max_priority_fraction :
        Cap: never put more than this fraction of ULDs into the priority
        bucket, so there is always room for economy packages. Default 0.75
        means at least 25 % of ULDs remain as "other".
    """
    priority_pkgs = [p for p in packages if p.is_priority]

    if not priority_pkgs:
        return ULDPartition(priority_ulds=[], other_ulds=list(ulds))

    if len(ulds) == 1:
        # Only one ULD — it must serve as both; pipeline will pack everything in.
        return ULDPartition(priority_ulds=list(ulds), other_ulds=list(ulds))

    priority_vol   = sum(p.volume for p in priority_pkgs)
    target_cap     = priority_vol * fill_target
    max_pri_ulds   = max(min_priority_ulds, int(len(ulds) * max_priority_fraction))

    sorted_ulds = sorted(ulds, key=lambda u: u.volume, reverse=True)
    priority_bucket: List[ULD] = []
    covered = 0.0

    for uld in sorted_ulds:
        if len(priority_bucket) >= max_pri_ulds:
            break
        priority_bucket.append(uld)
        covered += uld.volume
        if covered >= target_cap and len(priority_bucket) >= min_priority_ulds:
            break

    priority_ids = {u.id for u in priority_bucket}
    other_bucket = [u for u in ulds if u.id not in priority_ids]

    # Safety: if other_bucket ended up empty (shouldn't with max_priority_fraction
    # guard, but just in case), move the smallest priority ULD across.
    if not other_bucket and len(priority_bucket) > 1:
        spill = priority_bucket.pop()
        other_bucket.append(spill)

    return ULDPartition(priority_ulds=priority_bucket, other_ulds=other_bucket)
