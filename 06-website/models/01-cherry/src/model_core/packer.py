"""
3-D bin-packing strategies.

Two implementations are provided:
  EPIPacker   — numpy-accelerated Extreme Point Insertion (Crainic et al. 2008)
  pd3Packer   — wrapper around the py3dbp library

Both expose the same interface:
    packer.pack(assignment, packages_df, ulds_df)
        -> (placements, total_unfit)

placements is a list of dicts with keys:
    Package_ID, ULD_ID, x0, y0, z0, x1, y1, z1
    (coordinates are -1 for unplaced packages)
total_unfit counts packages the packer physically couldn't place.
"""

import numpy as np
import pandas as pd
from itertools import permutations
from dataclasses import dataclass
from typing import List, Dict

from .config import MAX_EPS_PER_ULD, MAX_N_ULDS


# ── EPI packer internals ──────────────────────────────────────────────────────

def _orientations(l, w, h):
    return list(set(permutations([l, w, h])))


def _overlap(a, b):
    """AABB overlap test. Touching faces are NOT overlap."""
    return not (a[3] <= b[0] or b[3] <= a[0] or
                a[4] <= b[1] or b[4] <= a[1] or
                a[5] <= b[2] or b[5] <= a[2])


def _update_eps_np(eps, new_box, boxes_np, uld_dims, max_eps=MAX_EPS_PER_ULD):
    """
    Add 3 canonical EPs from newly placed box, prune with numpy.
    boxes_np : (N,6) float32 array of all placed boxes so far.
    """
    x0, y0, z0, x1, y1, z1 = new_box
    L, W, H = uld_dims
    candidates = [(x1, y0, z0), (x0, y1, z0), (x0, y0, z1)]

    for cx, cy, cz in candidates:
        if cx > L or cy > W or cz > H:
            continue
        if len(boxes_np) > 0:
            b      = boxes_np.reshape(-1, 6)
            inside = (
                (b[:, 0] < cx) & (cx < b[:, 3]) &
                (b[:, 1] < cy) & (cy < b[:, 4]) &
                (b[:, 2] < cz) & (cz < b[:, 5])
            ).any()
            if inside:
                continue
        eps.add((cx, cy, cz))

    if len(eps) > max_eps:
        eps_arr = np.array(sorted(eps), dtype=np.float32)
        dom     = np.all(eps_arr[None, :, :] <= eps_arr[:, None, :], axis=2)
        np.fill_diagonal(dom, False)
        dominated = dom.any(axis=1)
        eps_arr   = eps_arr[~dominated]
        if len(eps_arr) > max_eps:
            order   = np.lexsort((eps_arr[:, 1], eps_arr[:, 0], eps_arr[:, 2]))
            eps_arr = eps_arr[order[:max_eps]]
        eps = set(map(tuple, eps_arr.tolist()))

    return eps


def pack_uld_epi(packages_in_uld, uld):
    """
    Pack packages into one ULD using greedy EPI (numpy-accelerated).

    Returns:
        placed   : list of (Package_ID, x0, y0, z0, x1, y1, z1)
        unplaced : list of Package_IDs that didn't fit
    """
    # Priority packages get first pick of extreme points, then largest-volume
    # first within each group (matches h1_h2_cargo's own greedy sort). Without
    # the priority-first split, a small priority package could be packed last
    # -- after big Economy packages have already claimed the good extreme
    # points -- and fail to fit even when the ULD had room for it earlier.
    pkgs_sorted = sorted(
        packages_in_uld,
        key=lambda p: (0 if str(p.get('Type', '')).upper() == 'PRIORITY' else 1,
                       -(p['Length'] * p['Width'] * p['Height'])),
    )

    L, W, H           = uld['Length'], uld['Width'], uld['Height']
    uld_dims          = (L, W, H)
    placed_boxes_list = []
    boxes_np          = np.empty((0, 6), dtype=np.float32)
    placed            = []
    unplaced          = []
    eps               = {(0, 0, 0)}

    for pkg in pkgs_sorted:
        l, w, h    = pkg['Length'], pkg['Width'], pkg['Height']
        best_pos   = None
        best_score = (float('inf'),) * 3

        for ep in sorted(eps, key=lambda e: (e[2], e[0], e[1])):
            x0, y0, z0 = ep
            for ol, ow, oh in _orientations(l, w, h):
                x1, y1, z1 = x0 + ol, y0 + ow, z0 + oh
                if x0 < 0 or y0 < 0 or z0 < 0 or x1 > L or y1 > W or z1 > H:
                    continue
                if len(boxes_np) > 0:
                    no_overlap = (
                        (boxes_np[:, 3] <= x0) | (x1 <= boxes_np[:, 0]) |
                        (boxes_np[:, 4] <= y0) | (y1 <= boxes_np[:, 1]) |
                        (boxes_np[:, 5] <= z0) | (z1 <= boxes_np[:, 2])
                    )
                    if not no_overlap.all():
                        continue
                score = (z0, x0, y0)
                if score < best_score:
                    best_score = score
                    best_pos   = (x0, y0, z0, x1, y1, z1)
            if best_pos and best_score == (0, 0, 0):
                break

        if best_pos is not None:
            placed_boxes_list.append(best_pos)
            boxes_np = np.array(placed_boxes_list, dtype=np.float32).reshape(-1, 6)
            placed.append((pkg['Package_ID'],) + best_pos)
            eps = _update_eps_np(eps, best_pos, boxes_np, uld_dims)
        else:
            unplaced.append(pkg['Package_ID'])

    return placed, unplaced


def pack_uld_epi_with_rescue(packages_in_uld, uld, max_evictions=40, weight_limit=None):
    """
    pack_uld_epi(), then if any Priority package still doesn't fit -- or the
    placed total exceeds weight_limit -- evict already-placed Economy
    packages and re-pack from scratch, repeating until every Priority
    package fits within weight AND geometry, or there's no Economy package
    left to evict.

    pack_uld_epi only checks geometric fit; it has no notion of weight. A
    ULD can be well under its volume budget yet already at its weight limit
    (e.g. dense small packages), in which case a stuck Priority package
    geometrically "fits" as soon as one Economy package is evicted, but the
    resulting total is still over the weight limit -- evicting the biggest
    Economy package by VOLUME doesn't reliably fix a WEIGHT problem (a bulky
    but light item frees geometric space without freeing much weight), so
    weight overage is checked and evicted-for separately (biggest-by-WEIGHT)
    once geometric placement alone has already succeeded.

    Economy packages are geometrically fungible with priority-first ordering
    already giving Priority first pick of extreme points (see pack_uld_epi);
    this rescue only kicks in for the genuine leftover case -- too many
    packages assigned to one ULD for all of them to fit -- mirroring
    h1_h2_cargo's own rescue/nuclear-eviction pass for the same situation.
    Evicted Economy packages come back as unplaced (priced into delay cost,
    same as any other left-behind Economy package -- not a hard constraint).
    """
    def is_priority(pkg):
        return str(pkg.get('Type', '')).upper() == 'PRIORITY'

    pkg_by_id = {p['Package_ID']: p for p in packages_in_uld}
    working = list(packages_in_uld)
    placed, unplaced = pack_uld_epi(working, uld)
    evicted_ids = []

    for _ in range(max_evictions):
        priority_unplaced = [pid for pid in unplaced if is_priority(pkg_by_id[pid])]
        placed_ids = {p[0] for p in placed}
        weight_over = (weight_limit is not None and
                       sum(pkg_by_id[pid]['Weight'] for pid in placed_ids) > weight_limit + 1e-6)

        if not priority_unplaced and not weight_over:
            break

        economy_placed = [pkg for pkg in working
                          if pkg['Package_ID'] in placed_ids and not is_priority(pkg)]
        if not economy_placed:
            break  # nothing left to evict; genuinely infeasible (e.g. too big for the ULD)

        if priority_unplaced:
            biggest = max(economy_placed, key=lambda p: p['Length'] * p['Width'] * p['Height'])
        else:
            biggest = max(economy_placed, key=lambda p: p['Weight'])  # weight_over-only case
        working = [p for p in working if p['Package_ID'] != biggest['Package_ID']]
        evicted_ids.append(biggest['Package_ID'])
        placed, unplaced = pack_uld_epi(working, uld)

    final_unplaced = list(unplaced) + [pid for pid in evicted_ids
                                        if pid not in {p[0] for p in placed}]
    return placed, final_unplaced


def greedy_epi_pack(assignment, packages_df, ulds_df):
    """
    Run EPI packer on all ULDs given a clustering assignment.

    Returns:
        placements  : list of placement dicts
        total_unfit : int
    """
    def is_priority(pkg):
        return str(pkg.get('Type', '')).upper() == 'PRIORITY'

    pkg_lookup   = packages_df.set_index('Package_ID').to_dict('index')
    uld_lookup   = ulds_df.set_index('ULD_ID').to_dict('index')
    uld_objs     = {uid: dict(uld_lookup[uid], ULD_ID=uid) for uid in uld_lookup}
    uld_packages = {uid: [] for uid in uld_lookup}
    placements   = []
    total_unfit  = 0

    for pid, uid in assignment.items():
        if uid == 'NONE':
            placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                               'x0': -1, 'y0': -1, 'z0': -1,
                               'x1': -1, 'y1': -1, 'z1': -1,
                               'reason': 'clusterer_none'})
        elif uid in uld_packages:
            pkg = dict(pkg_lookup[pid])
            pkg['Package_ID'] = pid
            uld_packages[uid].append(pkg)

    # working[uid] = packages currently committed to that ULD (mutated by the
    # cross-ULD rescue below); result[uid] = (placed, unplaced) for `working[uid]`.
    working, result = {}, {}
    for uid, pkgs in uld_packages.items():
        if not pkgs:
            working[uid], result[uid] = [], ([], [])
            continue
        placed, unplaced = pack_uld_epi_with_rescue(
            pkgs, uld_objs[uid], weight_limit=uld_objs[uid]['Weight_Limit'])
        working[uid], result[uid] = pkgs, (placed, unplaced)

    # Cross-ULD rescue: a Priority package can still be stuck unplaced after
    # its own ULD's within-ULD eviction pass if that ULD is itself packed with
    # too many Priority packages to fit together (nothing Economy to evict).
    # Try moving it into each OTHER ULD instead -- with eviction there too --
    # before giving up on it. This is h1_h2_cargo's own rescue strategy
    # (try other ULDs, not just the assigned one), applied to EPIPacker.
    #
    # Run to a fixed point, not just one pass: rescuing package A into ULD B
    # can force an eviction that only makes room for a *different* stuck
    # package once B is revisited, so a single sweep can leave rescuable
    # packages behind purely due to processing order.
    for _round in range(15):
        moved_any = False
        for uid in list(uld_packages):
            placed, unplaced = result[uid]
            stuck_priority = [pid for pid in unplaced
                              if is_priority(pkg_lookup[pid])]
            for pid in stuck_priority:
                pkg = dict(pkg_lookup[pid]); pkg['Package_ID'] = pid
                for other_uid in sorted(
                    (u for u in uld_packages if u != uid),
                    key=lambda u: -(uld_objs[u]['Length'] * uld_objs[u]['Width'] * uld_objs[u]['Height']),
                ):
                    other_placed, other_unplaced = result[other_uid]
                    other_placed_ids = {p[0] for p in other_placed}
                    candidate = [p for p in working[other_uid] if p['Package_ID'] in other_placed_ids] + [pkg]
                    new_placed, new_unplaced = pack_uld_epi_with_rescue(
                        candidate, uld_objs[other_uid], weight_limit=uld_objs[other_uid]['Weight_Limit'])
                    # Belt-and-suspenders: pack_uld_epi_with_rescue now evicts for
                    # weight overage internally too, but re-verify here in case
                    # eviction couldn't fully resolve it (e.g. ran out of Economy).
                    candidate_by_id = {p['Package_ID']: p for p in candidate}
                    new_weight = sum(candidate_by_id[p[0]]['Weight'] for p in new_placed)
                    within_weight = new_weight <= uld_objs[other_uid]['Weight_Limit'] + 1e-6
                    if pid not in new_unplaced and within_weight:
                        working[other_uid] = candidate
                        result[other_uid]  = (new_placed, new_unplaced)
                        unplaced = [p for p in unplaced if p != pid]
                        result[uid] = (placed, unplaced)
                        moved_any = True
                        break
        if not moved_any:
            break

    for uid in uld_packages:
        placed, unplaced = result[uid]
        for pid, x0, y0, z0, x1, y1, z1 in placed:
            placements.append({'Package_ID': pid, 'ULD_ID': uid,
                               'x0': x0, 'y0': y0, 'z0': z0,
                               'x1': x1, 'y1': y1, 'z1': z1,
                               'reason': 'placed'})
        for pid in unplaced:
            placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                               'x0': -1, 'y0': -1, 'z0': -1,
                               'x1': -1, 'y1': -1, 'z1': -1,
                               'reason': 'packer_unfit'})
            total_unfit += 1

    return placements, total_unfit


# ── py3dbp wrapper ────────────────────────────────────────────────────────────

@dataclass
class _ULD:
    ULD_ID:       str
    Length:       int
    Width:        int
    Height:       int
    Weight_Limit: int


@dataclass
class _Package:
    Package_ID: str
    Length:     int
    Width:      int
    Height:     int
    Weight:     int
    Type:       str
    Delay_Cost: float


def pd3_pack(assignment, packages_df, ulds_df):
    """
    Run py3dbp packer on all ULDs given a clustering assignment.

    Returns:
        placements  : list of placement dicts
        total_unfit : int
    """
    from py3dbp import Packer as pacpac, Bin, Item

    pkg_dict = {}
    for _, row in packages_df.iterrows():
        pid = str(row['Package_ID'])
        pkg_dict[pid] = _Package(
            Package_ID=pid,
            Length=int(row['Length']),
            Width=int(row['Width']),
            Height=int(row['Height']),
            Weight=int(row['Weight']),
            Type=str(row['Type']),
            Delay_Cost=float(row.get('Delay_Cost', 0)),
        )

    ulds_list = []
    for _, row in ulds_df.iterrows():
        ulds_list.append(_ULD(
            ULD_ID=str(row['ULD_ID']),
            Length=int(row['Length']),
            Width=int(row['Width']),
            Height=int(row['Height']),
            Weight_Limit=int(row['Weight_Limit']),
        ))

    assignments_map = {uld.ULD_ID: [] for uld in ulds_list}
    placements  = []
    total_unfit = 0

    for pid, uid in assignment.items():
        if uid == 'NONE':
            placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                               'x0': -1, 'y0': -1, 'z0': -1,
                               'x1': -1, 'y1': -1, 'z1': -1,
                               'reason': 'clusterer_none'})
        elif uid in assignments_map and pid in pkg_dict:
            assignments_map[uid].append(pkg_dict[pid])

    for uld in ulds_list:
        pkgs = assignments_map.get(uld.ULD_ID, [])
        if not pkgs:
            continue
        priority_first = sorted(pkgs, key=lambda p: (0 if p.Type == 'Priority' else 1))
        packer = pacpac()
        packer.add_bin(Bin(uld.ULD_ID, uld.Length, uld.Width, uld.Height, uld.Weight_Limit))
        for pkg in priority_first:
            packer.add_item(Item(pkg.Package_ID, pkg.Length, pkg.Width, pkg.Height, pkg.Weight))
        packer.pack(bigger_first=True, distribute_items=False, number_of_decimals=0)

        for item in packer.bins[0].items:
            x0 = int(item.position[0])
            y0 = int(item.position[1])
            z0 = int(item.position[2])
            placements.append({'Package_ID': item.name, 'ULD_ID': uld.ULD_ID,
                               'x0': x0, 'y0': y0, 'z0': z0,
                               'x1': x0 + int(item.width),
                               'y1': y0 + int(item.depth),
                               'z1': z0 + int(item.height),
                               'reason': 'placed'})
        for item in packer.unfit_items:
            placements.append({'Package_ID': item.name, 'ULD_ID': 'NONE',
                               'x0': -1, 'y0': -1, 'z0': -1,
                               'x1': -1, 'y1': -1, 'z1': -1,
                               'reason': 'packer_unfit'})
            total_unfit += 1

    return placements, total_unfit


# ── Strategy base class and concrete implementations ──────────────────────────

class Packer:
    """Abstract base for packing strategies. Subclass and override `pack`."""
    def pack(self, assignment, packages_df, ulds_df):
        raise NotImplementedError


class EPIPacker(Packer):
    """Greedy Extreme Point Insertion packer (Crainic et al. 2008)."""
    def __init__(self, max_eps=MAX_EPS_PER_ULD):
        self.max_eps = max_eps

    def pack(self, assignment, packages_df, ulds_df):
        return greedy_epi_pack(assignment, packages_df, ulds_df)


class pd3Packer(Packer):
    """py3dbp-based 3-D bin packer."""
    def __init__(self, max_eps=MAX_EPS_PER_ULD):
        self.max_eps = max_eps

    def pack(self, assignment, packages_df, ulds_df):
        return pd3_pack(assignment, packages_df, ulds_df)


DEFAULT_PACKER = EPIPacker()
