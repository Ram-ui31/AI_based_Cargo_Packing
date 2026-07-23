"""
Independent implementations of three classical heuristics standard in the
3D bin-packing literature, applied to the ULD cargo-packing problem. No
code is reused from cargoism/git's EMS/RL packer -- placement geometry
here is a simple, classic corner-point method (bottom-left-back scan
order, all 6 orientations tried per candidate point), independent of the
EMS-based geometry used everywhere else in this project.

Heuristics:
  - FFD  (First-Fit Decreasing)       -- sort by decreasing VOLUME, place
    in the first ULD (in given order) where it geometrically fits.
  - LAFF (Largest Area Fit First)     -- sort by decreasing LARGEST FACE
    AREA, same first-fit placement rule. "Biggest first" by footprint
    rather than volume.
  - BFD  (Best Fit Decreasing)        -- sort by decreasing volume (like
    FFD), but place in whichever ULD leaves the LEAST leftover empty
    volume after placement, trying every ULD rather than stopping at the
    first fit.

All three share the same adaptation to this problem's constraints
(Priority must always ship, Economy is optional with a delay cost, spread
cost = K * distinct ULDs holding any Priority package):
  1. All Priority packages are sorted and placed first (guarantees the
     hard constraint whenever geometrically possible).
  2. Remaining Economy packages are sorted and placed into whatever space
     is left; anything that doesn't fit anywhere stays unplaced.
"""
import os
import json
import time

import pandas as pd

from packing_geometry import BinState


# ---------------------------------------------------------------------------
# Shared: cost + I/O
# ---------------------------------------------------------------------------

def compute_cost(placement, pkgs_df, k_value):
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    delay_cost = 0.0
    prio_ulds = set()
    unplaced_prio, unplaced_econ = [], []
    for pid, uid in placement.items():
        pkg = pkg_lookup[pid]
        if uid == 'NONE':
            if pkg['Type'] == 'Economy':
                delay_cost += pkg['Delay_Cost']
                unplaced_econ.append(pid)
            else:
                unplaced_prio.append(pid)
        elif pkg['Type'] == 'Priority':
            prio_ulds.add(uid)
    spread_cost = k_value * len(prio_ulds)
    total = delay_cost + spread_cost
    return {
        'total_cost': total, 'delay_cost': delay_cost, 'spread_cost': spread_cost,
        'n_priority_ulds': len(prio_ulds),
        'unplaced_priority': unplaced_prio, 'unplaced_economy_count': len(unplaced_econ),
    }


def parse_input_csv(path):
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f]
    k_value = float(lines[0].strip())
    uld_rows, pkg_rows = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0].startswith('U'):
            uid, length, width, height, weight_limit = parts
            uld_rows.append({'ULD_ID': uid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight_Limit': float(weight_limit)})
        else:
            pid, length, width, height, weight, ptype, delay = parts
            delay_cost = 0.0 if delay.strip() == '-' else float(delay)
            pkg_rows.append({'Package_ID': pid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight': float(weight), 'Type': ptype,
                              'Delay_Cost': delay_cost})
    return k_value, pd.DataFrame(uld_rows), pd.DataFrame(pkg_rows)


INSTANCES = [
    'instance_051', 'instance_050', 'instance_025', 'instance_097',
    'instance_039', 'instance_045', 'instance_076', 'instance_078',
    'instance_005', 'instance_003', 'instance_093', 'instance_075',
    'instance_098', 'instance_030', 'instance_048', 'instance_072',
    'instance_085', 'instance_006', 'instance_008', 'instance_019',
]
GOOD_DATA = os.path.expanduser('~/Desktop/good_data/synthetic_test')


# ---------------------------------------------------------------------------
# FFD -- First-Fit Decreasing (sort by volume, first ULD that fits)
# ---------------------------------------------------------------------------

def pack_ffd(pkgs_df, ulds_df):
    bins = {row['ULD_ID']: BinState(row['Length'], row['Width'], row['Height'], row['Weight_Limit'])
            for _, row in ulds_df.iterrows()}
    uld_order = list(ulds_df['ULD_ID'])

    pkgs = pkgs_df.copy()
    pkgs['_key'] = pkgs['Length'] * pkgs['Width'] * pkgs['Height']

    placement = {}
    for ptype in ['Priority', 'Economy']:
        subset = pkgs[pkgs['Type'] == ptype].sort_values('_key', ascending=False)
        for _, pkg in subset.iterrows():
            dims = (pkg['Length'], pkg['Width'], pkg['Height'])
            placed = False
            for uid in uld_order:
                if bins[uid].try_place(dims, pkg['Weight']):
                    placement[pkg['Package_ID']] = uid
                    placed = True
                    break
            if not placed:
                placement[pkg['Package_ID']] = 'NONE'
    return placement


# ---------------------------------------------------------------------------
# LAFF -- Largest Area Fit First (sort by largest face area, first-fit)
# ---------------------------------------------------------------------------

def pack_laff(pkgs_df, ulds_df):
    bins = {row['ULD_ID']: BinState(row['Length'], row['Width'], row['Height'], row['Weight_Limit'])
            for _, row in ulds_df.iterrows()}
    uld_order = list(ulds_df['ULD_ID'])

    pkgs = pkgs_df.copy()
    pkgs['_key'] = pkgs[['Length', 'Width', 'Height']].apply(
        lambda r: max(r['Length'] * r['Width'], r['Width'] * r['Height'], r['Length'] * r['Height']), axis=1)

    placement = {}
    for ptype in ['Priority', 'Economy']:
        subset = pkgs[pkgs['Type'] == ptype].sort_values('_key', ascending=False)
        for _, pkg in subset.iterrows():
            dims = (pkg['Length'], pkg['Width'], pkg['Height'])
            placed = False
            for uid in uld_order:
                if bins[uid].try_place(dims, pkg['Weight']):
                    placement[pkg['Package_ID']] = uid
                    placed = True
                    break
            if not placed:
                placement[pkg['Package_ID']] = 'NONE'
    return placement


# ---------------------------------------------------------------------------
# BFD -- Best Fit Decreasing (sort by volume, place in whichever ULD leaves
# the least leftover empty volume -- tries every ULD, not just the first
# that fits)
# ---------------------------------------------------------------------------

def _used_volume(bin_state):
    return sum((x1 - x0) * (y1 - y0) * (z1 - z0) for x0, y0, z0, x1, y1, z1 in bin_state.boxes)


def pack_bfd(pkgs_df, ulds_df):
    bins = {row['ULD_ID']: BinState(row['Length'], row['Width'], row['Height'], row['Weight_Limit'])
            for _, row in ulds_df.iterrows()}
    uld_order = list(ulds_df['ULD_ID'])

    pkgs = pkgs_df.copy()
    pkgs['_key'] = pkgs['Length'] * pkgs['Width'] * pkgs['Height']

    placement = {}
    for ptype in ['Priority', 'Economy']:
        subset = pkgs[pkgs['Type'] == ptype].sort_values('_key', ascending=False)
        for _, pkg in subset.iterrows():
            dims = (pkg['Length'], pkg['Width'], pkg['Height'])
            weight = pkg['Weight']

            best_uid, best_leftover = None, None
            for uid in uld_order:
                b = bins[uid]
                boxes_snap, corners_snap, weight_snap = list(b.boxes), list(b.corners), b.weight_used
                if b.try_place(dims, weight):
                    leftover = (b.L * b.W * b.H) - _used_volume(b)
                    if best_leftover is None or leftover < best_leftover:
                        best_leftover, best_uid = leftover, uid
                    # revert trial placement -- only the chosen ULD gets committed below
                    b.boxes, b.corners, b.weight_used = boxes_snap, corners_snap, weight_snap
                # on failure, try_place is guaranteed not to have mutated state

            if best_uid is not None:
                committed = bins[best_uid].try_place(dims, weight)
                assert committed, 'best-fit candidate must re-place identically on committed attempt'
                placement[pkg['Package_ID']] = best_uid
            else:
                placement[pkg['Package_ID']] = 'NONE'
    return placement


PACKERS = {'ffd': pack_ffd, 'laff': pack_laff, 'bfd': pack_bfd}
