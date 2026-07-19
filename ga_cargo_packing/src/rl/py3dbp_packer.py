"""
py3dbp_packer.py -- Packer-interface-compatible wrapper around py3dbp (a
mature, already-integrated 3D bin-packing library, currently only used as
an imitation-learning teacher for rl_packer's placement policy -- see
cargoism/git/rl_packer/src/py3dbp_teacher.py's docstring: "a mature, much
better-performing 3D packer per direct comparison"). Never tried as an
actual candidate in CombinedPacker's ensemble until now.

Unlike our own Heightmap.pivot_points() (corner-adjacent points of already-
placed boxes only, capped at 400, arbitrary truncation order), py3dbp
implements a more complete placement search internally -- worth testing
directly before committing to building Empty-Maximal-Space / Packing-
Configuration-Tree candidate generation from scratch.
"""
from __future__ import annotations

import os
import sys

CARGOISM_RL_PACKER_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'rl_packer', 'src',
)


def _import_py3dbp_teacher():
    src = os.path.abspath(CARGOISM_RL_PACKER_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import py3dbp_teacher
    return py3dbp_teacher


class Py3dbpPacker:
    """Packer-interface-compatible: pack(assignment, packages_df, ulds_df)."""

    def __init__(self, priority_first=True):
        self._teacher = _import_py3dbp_teacher()
        self.priority_first = priority_first

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
        pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
        for pid, row in pkg_lookup.items():
            row['Package_ID'] = pid
        uld_pkg_ids = {uid: [] for uid in uld_lookup}
        placements = []

        for pid, uid in assignment.items():
            if uid == 'NONE':
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                   'x0': -1, 'y0': -1, 'z0': -1,
                                   'x1': -1, 'y1': -1, 'z1': -1,
                                   'reason': 'clusterer_none'})
            elif uid in uld_pkg_ids:
                uld_pkg_ids[uid].append(pid)

        import pandas as pd
        run_fn = self._teacher.run_py3dbp_priority_first if self.priority_first else self._teacher.run_py3dbp
        total_unfit = 0
        for uid, pids in uld_pkg_ids.items():
            if not pids:
                continue
            pkgs_df_uld = pd.DataFrame([pkg_lookup[pid] for pid in pids])
            steps, unfit_ids = run_fn(uld_lookup[uid], pkgs_df_uld)
            for s in steps:
                placements.append({'Package_ID': s.package_id, 'ULD_ID': uid,
                                   'x0': s.x, 'y0': s.y, 'z0': s.z,
                                   'x1': s.x + s.dx, 'y1': s.y + s.dy, 'z1': s.z + s.dz,
                                   'reason': 'placed'})
            for pid in unfit_ids:
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                   'x0': -1, 'y0': -1, 'z0': -1,
                                   'x1': -1, 'y1': -1, 'z1': -1,
                                   'reason': 'packer_unfit'})
                total_unfit += 1

        return placements, total_unfit
