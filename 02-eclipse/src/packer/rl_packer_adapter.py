"""
rl_packer_adapter.py — wraps the OTHER pipeline's learned placement policy
(cargoism/git/rl_packer: true-3D pivot-point placement, actor-critic network)
as a Packer strategy, so it can be swapped in for EPIPacker/pd3Packer without
touching the assignment side (TransformerClusterer, rl_assign_argmax_safe)
at all -- this only replaces the packing stage.

Unlike EPIPacker, rl_packer's Heightmap.fits() checks the weight limit
natively (no separate weight-blindness patch needed here), and placement is
true pivot-point 3D (z is a free choice, not forced to rest on the floor).
It has no explicit priority-first ordering or eviction rescue, though --
package selection is entirely up to the learned policy's softmax over all
currently-valid (package, position, orientation) candidates, so whether
Priority packages get shipped depends on what the policy learned, same as
EPIPacker before the priority-first + eviction fixes in packer.py.
"""
from __future__ import annotations
import importlib
import os
import sys

import pandas as pd
import torch


def _import_rl_packer_modules(rl_packer_src):
    """
    rl_packer/src/geometry.py and h1_h2_cargo/src/geometry.py are two
    unrelated modules that happen to share the name 'geometry' -- if a
    caller (e.g. src.ga's GAPipeline, which needs h1_h2_cargo's geometry)
    already imported one under sys.modules['geometry'] in this same process,
    a bare `import geometry` here would silently reuse the wrong one instead
    of raising (rl_packer's placement_policy.py does `from geometry import
    Heightmap`, which h1_h2_cargo's geometry.py doesn't define -> ImportError,
    or worse, a same-named-but-wrong symbol on the reverse collision).

    Fix: temporarily evict any cached 'geometry'/'placement_env'/
    'placement_policy' entries, force a fresh import with rl_packer_src at
    the front of sys.path, then restore whatever was cached before -- the
    modules we just imported keep their own correctly-bound references
    regardless of what sys.modules['geometry'] points to afterward, since
    Python binds `from geometry import Heightmap` at import time into the
    importing module's own namespace, not lazily.
    """
    if rl_packer_src not in sys.path:
        sys.path.insert(0, rl_packer_src)
    else:
        sys.path.remove(rl_packer_src)
        sys.path.insert(0, rl_packer_src)

    conflict_names = ('geometry', 'placement_env', 'placement_policy')
    saved = {name: sys.modules.pop(name, None) for name in conflict_names}
    try:
        placement_policy = importlib.import_module('placement_policy')
        placement_env = importlib.import_module('placement_env')
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
    return placement_policy, placement_env


class RLPackerAdapter:
    """Packer-interface-compatible wrapper: pack(assignment, packages_df, ulds_df)."""

    def __init__(self, weights_path=None, rl_packer_src=None, greedy=True, device='cpu'):
        if rl_packer_src is None:
            # This file lives at model-training-pipeline/src/rl/ -- three
            # levels below Desktop/, not two like the original
            # cargoism/git/rl_over_il_h1h2/src/ copy this was adapted from --
            # so the walk back up to cargoism/git/rl_packer/src is longer here.
            rl_packer_src = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                'rl_packer', 'src'
            )
        rl_packer_src = os.path.abspath(rl_packer_src)
        placement_policy_mod, placement_env_mod = _import_rl_packer_modules(rl_packer_src)
        PlacementPolicy = placement_policy_mod.PlacementPolicy
        self._PlacementEnv = placement_env_mod.PlacementEnv

        if weights_path is None:
            # placement_policy_big.pt (hidden=256, trained on real-assignment-
            # derived tight episodes -- see rl_packer/training/
            # train_placement_tight_finetune.py) is the current best,
            # validated on both a held-out aggregate benchmark and the
            # real-world 400-package stress instance. placement_policy.pt
            # (hidden=96, the original) is kept alongside it, untouched, as
            # a reference/fallback -- pass weights_path= explicitly to use it.
            weights_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..',
                'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_big.pt',
            )
        weights_path = os.path.abspath(weights_path)

        state = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        # hidden size is read from the checkpoint's own weight shape rather
        # than hardcoded -- this file has shipped with hidden=96 (original),
        # hidden=256 (current default), and hidden=512 (tested, not kept)
        # checkpoints during this session; hardcoding one size here silently
        # breaks loading any checkpoint trained at a different width.
        hidden = state['actor.0.weight'].shape[0]
        self.policy = PlacementPolicy(hidden=hidden)
        self.policy.load_state_dict(state)
        self.policy.eval()
        self.greedy = greedy
        self.weights_path = weights_path

    def _pack_uld(self, uid, pids, uld_lookup, pkg_lookup):
        """Priority-first, two-phase packing of `pids` into ULD `uid` from
        scratch: Priority packages get their own episode with first claim on
        every extreme point, then Economy continues filling the SAME
        Heightmap (PlacementEnv supports resuming into an already-partially-
        filled hm= for exactly this). Plain RLPackerAdapter has no priority
        ordering at all -- candidates() mixes every remaining package
        regardless of type, so Economy could claim the only spot a Priority
        package needed purely by candidate-list order. Returns (hm,
        left_behind_ids)."""
        priority_ids = [pid for pid in pids if pkg_lookup[pid]['Type'] == 'Priority']
        economy_ids = [pid for pid in pids if pkg_lookup[pid]['Type'] != 'Priority']
        priority_df = pd.DataFrame([pkg_lookup[pid] for pid in priority_ids])
        economy_df = pd.DataFrame([pkg_lookup[pid] for pid in economy_ids])

        hm = None
        left_behind_ids = []
        if len(priority_df) > 0:
            env_p = self._PlacementEnv(uld_lookup[uid], priority_df)
            with torch.no_grad():
                env_p.run_episode(self.policy, greedy=self.greedy)
            hm = env_p.hm
            left_behind_ids.extend(env_p.left_behind_ids)
        if len(economy_df) > 0:
            env_e = self._PlacementEnv(
                uld_lookup[uid] if hm is None else None, economy_df, hm=hm,
            )
            with torch.no_grad():
                env_e.run_episode(self.policy, greedy=self.greedy)
            hm = env_e.hm
            left_behind_ids.extend(env_e.left_behind_ids)
        return hm, left_behind_ids

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        """
        max_rescue_rounds : cross-ULD rescue for stuck Priority packages,
            mirroring h1_h2_cargo/packer.py's own cross-ULD rescue loop (a
            stuck Priority package can still be unplaced after its own ULD's
            priority-first pass if that ULD is itself assigned too many
            Priority packages to fit together -- try moving it into each
            OTHER ULD instead, evicting Economy there if needed, before
            giving up on it). Bounded (not run-to-fixed-point like
            packer.py's uncapped loop) since each attempt re-packs a whole
            ULD from scratch and this runs inside the RL training loop once
            per instance per epoch -- unbounded cost here was exactly the
            failure mode that made the GA's own repair loop pathologically
            slow before it was capped (see ga_solver.py's max_repairs).

            Candidate ULDs are tried ULDs-already-holding-Priority first
            (largest such ULD first), falling back to Priority-free ULDs
            (largest first) only once no already-Priority ULD can fit the
            package. Sorting by volume alone (no Priority preference) let a
            single large empty ULD become the default rescue target for
            every stuck package -- recruiting a brand new ULD into the
            Priority set even when squeezing back into an already-Priority
            ULD (evicting Economy there) was possible, needlessly inflating
            spread beyond what the clusterer originally assigned.
        """
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

        hm_by_uld = {}
        left_behind_by_uld = {}
        for uid, pids in uld_pkg_ids.items():
            if not pids:
                hm_by_uld[uid] = None
                left_behind_by_uld[uid] = []
                continue
            hm_by_uld[uid], left_behind_by_uld[uid] = self._pack_uld(uid, pids, uld_lookup, pkg_lookup)

        uld_vol = {uid: row['Length'] * row['Width'] * row['Height'] for uid, row in uld_lookup.items()}

        def _has_priority(uid):
            hm = hm_by_uld[uid]
            return hm is not None and any(
                pkg_lookup[pl.package_id]['Type'] == 'Priority' for pl in hm.placements)

        for _round in range(max_rescue_rounds):
            moved_any = False
            for uid in list(uld_pkg_ids):
                stuck_priority = [pid for pid in left_behind_by_uld[uid]
                                  if pkg_lookup[pid]['Type'] == 'Priority']
                for pid in stuck_priority:
                    for other_uid in sorted((u for u in uld_pkg_ids if u != uid),
                                            key=lambda u: (not _has_priority(u), -uld_vol[u])):
                        other_placed_ids = ([p.package_id for p in hm_by_uld[other_uid].placements]
                                            if hm_by_uld[other_uid] is not None else [])
                        # Must include other_uid's OWN pre-existing left-behind
                        # packages here, not just what it currently has placed
                        # -- omitting them silently drops those packages from
                        # every downstream accounting (compute_packing_cost
                        # only sums over `placements`, so a dropped package
                        # contributes neither delay cost nor a spread credit,
                        # understating true cost). Found on a 400-package
                        # stress instance where 91 packages vanished this way
                        # across repeated rescue rounds.
                        candidate_ids = other_placed_ids + left_behind_by_uld[other_uid] + [pid]
                        new_hm, new_left_behind = self._pack_uld(other_uid, candidate_ids, uld_lookup, pkg_lookup)
                        if pid not in new_left_behind:
                            hm_by_uld[other_uid] = new_hm
                            left_behind_by_uld[other_uid] = new_left_behind
                            left_behind_by_uld[uid] = [p for p in left_behind_by_uld[uid] if p != pid]
                            moved_any = True
                            break
            if not moved_any:
                break

        total_unfit = 0
        for uid in uld_pkg_ids:
            hm = hm_by_uld[uid]
            if hm is not None:
                for p in hm.placements:
                    placements.append({'Package_ID': p.package_id, 'ULD_ID': uid,
                                       'x0': p.x0, 'y0': p.y0, 'z0': p.z0,
                                       'x1': p.x1, 'y1': p.y1, 'z1': p.z1,
                                       'reason': 'placed'})
            for pid in left_behind_by_uld[uid]:
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                   'x0': -1, 'y0': -1, 'z0': -1,
                                   'x1': -1, 'y1': -1, 'z1': -1,
                                   'reason': 'packer_unfit'})
                total_unfit += 1

        return placements, total_unfit
