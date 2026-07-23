"""
centrifuge_refiner.py -- post-processing pass applied AFTER any packer's
pack() call: for each ULD, reconstruct its Heightmap from the resulting
placements, "centrifuge" it (Heightmap.compact() -- slide every box toward
one corner to consolidate fragmented free space), then greedily try to
insert currently-unplaced Economy packages into the freed space. Repeats
for a few cycles since filling one ULD changes which packages remain
available for the next.

Why this is a genuinely different lever from everything else tried: EMS
candidate generation finds every gap that EXISTS, but doesn't change WHERE
boxes are positioned -- fragmented free space (many small scattered gaps)
stays fragmented even with perfect candidate generation. Centrifuging
repositions the already-accepted boxes (same set, same sizes) to
consolidate that free space, which can make previously-unfit packages fit
even without changing which packages were selected at all -- proven on the
real instance: a single centrifuge+refill pass with just one packing
strategy recovered 12 packages / 1,060 delay-cost points that plain EMS
candidate generation alone did not.
"""
from __future__ import annotations

import os

import pandas as pd

from .heuristic_packer import HeuristicPacker, _import_geometry


def centrifuge_refine(placements, packages_df, ulds_df, n_cycles=2, strategy='dblf',
                       rl_packer_src=None):
    """Returns an updated placements list (same shape as any Packer.pack()
    output) after centrifuging + refilling each ULD for n_cycles passes."""
    geometry = _import_geometry(rl_packer_src or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
        'rl_packer', 'src'))
    Heightmap = geometry.Heightmap

    pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    by_uld = {}
    none_entries = []
    for p in placements:
        if p['ULD_ID'] == 'NONE':
            none_entries.append(p)
        else:
            by_uld.setdefault(p['ULD_ID'], []).append(p)

    unplaced_pids = {p['Package_ID'] for p in none_entries
                      if pkg_lookup[p['Package_ID']]['Type'] != 'Priority'}

    # origin_source='ems' is not optional here -- pivot_points() (the
    # default) is exactly the corner-adjacent candidate-generation
    # bottleneck EMS was built to fix; using it for the refill step would
    # make centrifuging blind to most of the free space it just consolidated.
    filler = HeuristicPacker(rl_packer_src=rl_packer_src, strategy=strategy, origin_source='ems')

    for _cycle in range(n_cycles):
        any_change = False
        for uid, plist in list(by_uld.items()):
            uld_row = uld_lookup[uid]
            hm = Heightmap(length=int(uld_row['Length']), width=int(uld_row['Width']),
                            height=int(uld_row['Height']), weight_limit=float(uld_row['Weight_Limit']))
            for p in plist:
                pid = p['Package_ID']
                dx, dy, dz = p['x1'] - p['x0'], p['y1'] - p['y0'], p['z1'] - p['z0']
                hm.place(pid, dx, dy, dz, p['x0'], p['y0'], p['z0'], pkg_lookup[pid]['Weight'])

            weight_by_pid = {p['Package_ID']: pkg_lookup[p['Package_ID']]['Weight'] for p in plist}
            compacted = hm.compact(weight_by_pid)

            if unplaced_pids:
                candidates_df = pd.DataFrame([pkg_lookup[pid] for pid in unplaced_pids])
                candidates_df = candidates_df.assign(
                    _vol=lambda d: d['Length'] * d['Width'] * d['Height'],
                ).assign(
                    _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5),
                ).sort_values('_vdp', ascending=False)
                left_behind = filler._greedy_pack_into(compacted, candidates_df)
                newly_placed = unplaced_pids - set(left_behind)
                if newly_placed:
                    any_change = True
                    unplaced_pids -= newly_placed
            else:
                newly_placed = set()

            by_uld[uid] = [
                {'Package_ID': p.package_id, 'ULD_ID': uid,
                 'x0': p.x0, 'y0': p.y0, 'z0': p.z0, 'x1': p.x1, 'y1': p.y1, 'z1': p.z1,
                 'reason': 'placed'}
                for p in compacted.placements
            ]
        if not any_change:
            break

    new_placements = [entry for plist in by_uld.values() for entry in plist]
    all_placed_pids = {e['Package_ID'] for e in new_placements}
    for p in none_entries:
        if p['Package_ID'] not in all_placed_pids:
            new_placements.append(p)
    return new_placements
