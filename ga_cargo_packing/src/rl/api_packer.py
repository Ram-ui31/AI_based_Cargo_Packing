"""
api_packer.py -- Packer implementation wrapping the third-party
3dbinpacking.com API (https://www.3dbinpacking.com/en/api-doc), as an
alternative candidate to RLPackerAdapter/EPIPacker for
rl_assign_argmax_adaptive (or a standalone comparison against our own
packer's cost on a given assignment).

Full API research (request/response schema, confirmed limitations, open
items) lives in ~/Desktop/3dbinpacking_api/api_reference.md -- read that
before changing this file's request/response handling.

Two confirmed limitations shape this implementation (see that doc's
"Confirmed limitations" section for the direct evidence):
  1. No incremental/partial-bin packing -- every API call is stateless, so
     RLPackerAdapter's clean "Priority into a Heightmap, Economy continues
     into the SAME Heightmap" pattern isn't available here.
  2. No item-level priority/ordering field -- the API "optimizes item
     placement freely," so we can't just send Priority+Economy together and
     trust Priority gets first claim on space.

Workaround: two API calls per ULD. Priority packages first, against the
ULD's real dimensions. Then Economy, against a REDUCED bin (weight limit
lowered by Priority's actual placed weight -- exact, weight is additive;
height reduced by Priority's reported volume utilization fraction --
APPROXIMATE, since Priority's real placed shape isn't a clean sub-box in
general). This trades some packing efficiency for a guarantee that Economy
can never encroach on space the API itself reports Priority already used.

REQUIRES CREDENTIALS: reads THREEDBINPACKING_USERNAME / THREEDBINPACKING_API_KEY
from the environment, falling back to ~/Desktop/3dbinpacking_api/.credentials.json
(local, untracked, outside this git repo -- never commit credentials). See
_load_credentials() below.

NO COORDINATES IN THE RESPONSE (verified directly, not assumed --
scripts/verify_api_axis_mapping.py's actual live-call output): despite the
doc site's example response showing a per-item 'coordinates' field, the real
packIntoMany response only returns id/w/h/d/wg per packed item, plus
aggregate bin_data (used_space%, weight, stack_height) -- no x/y/z position
at all. This does NOT block cost comparison (compute_packing_cost only reads
Package_ID/ULD_ID per placement, never coordinates), so placements below use
placeholder (-1) coordinates. It WOULD block real geometry use (visualization,
overlap verification) -- no mode/endpoint that returns real coordinates has
been found yet.
"""
from __future__ import annotations

import json
import os
import time

import requests

API_URL = 'https://global-api.3dbinpacking.com/packer/packIntoMany'
# Local, untracked, outside this git repo -- never commit credentials.
# Env vars (THREEDBINPACKING_USERNAME / THREEDBINPACKING_API_KEY) take
# priority when set; this file is just a convenience fallback so scripts
# don't need the secret retyped into every shell command.
CREDENTIALS_FILE = os.path.expanduser('~/Desktop/3dbinpacking_api/.credentials.json')


def _load_credentials():
    username = os.environ.get('THREEDBINPACKING_USERNAME')
    api_key = os.environ.get('THREEDBINPACKING_API_KEY')
    if username and api_key:
        return username, api_key
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE) as f:
            creds = json.load(f)
        return creds.get('username'), creds.get('api_key')
    return None, None


class ThreeDBinPackingAPIPacker:
    """Packer-interface-compatible wrapper: pack(assignment, packages_df, ulds_df)."""

    def __init__(self, username=None, api_key=None, optimization_mode='bins_utilization', timeout=30):
        fallback_username, fallback_api_key = _load_credentials()
        self.username = username or fallback_username
        self.api_key = api_key or fallback_api_key
        if not self.username or not self.api_key:
            raise ValueError(
                'ThreeDBinPackingAPIPacker requires credentials -- set '
                'THREEDBINPACKING_USERNAME and THREEDBINPACKING_API_KEY env vars, '
                'or pass username=/api_key= explicitly. See '
                '~/Desktop/3dbinpacking_api/api_reference.md for how to obtain them.'
            )
        self.optimization_mode = optimization_mode
        self.timeout = timeout

    def _call_api(self, bin_row, pkg_rows):
        """One packIntoMany call: a single ULD (as one bin, q=1) + a list of
        package dict rows. Returns the parsed 'response' dict, or a
        zero-packed stub if pkg_rows is empty (avoid a pointless network call)."""
        if not pkg_rows:
            return {'bins_packed': [], 'not_packed_items': [], 'status': 1}

        bins = [{
            'id': bin_row['ULD_ID'],
            'w': bin_row['Length'], 'h': bin_row['Height'], 'd': bin_row['Width'],
            'max_wg': bin_row['Weight_Limit'], 'q': 1, 'type': 'box',
        }]
        items = [{
            'id': p['Package_ID'],
            'w': p['Length'], 'h': p['Height'], 'd': p['Width'],
            'wg': p['Weight'], 'q': 1, 'vr': 1,
        } for p in pkg_rows]

        payload = {
            'username': self.username, 'api_key': self.api_key,
            'bins': bins, 'items': items,
            'params': {'optimization_mode': self.optimization_mode},
        }
        # Transient 500s observed in practice -- retry a couple times with
        # backoff before giving up, since a single flaky call shouldn't
        # abort an entire multi-ULD comparison run.
        last_exc = None
        for attempt in range(3):
            try:
                resp = requests.post(API_URL, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                print(f'    [api_packer] attempt {attempt + 1}/3 failed: {exc} '
                      f'-- body: {resp.text[:300]}')
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        else:
            raise last_exc
        data = resp.json()['response']
        if data.get('status') == 0:
            raise RuntimeError(f'3dbinpacking.com API returned a critical error: {data.get("errors")}')
        return data

    @staticmethod
    def _extract_placements(resp, uid):
        # Verified directly (scripts/verify_api_axis_mapping.py): the live
        # packIntoMany response does NOT include a 'coordinates' field on
        # packed items at all (despite the doc site's example showing one) --
        # only id/w/h/d/wg per item, plus aggregate bin_data (used_space%,
        # weight, stack_height). compute_packing_cost() only reads
        # Package_ID and ULD_ID per placement, never x0/y0/z0/x1/y1/z1, so
        # this is NOT a blocker for cost comparison -- coordinates below are
        # placeholders (-1), not real geometry. If real coordinates are ever
        # needed (visualization, overlap verification), this API would need
        # a different mode/endpoint than the one used here -- not yet found.
        placements = []
        for bp in resp.get('bins_packed', []):
            for it in bp['items']:
                placements.append({
                    'Package_ID': it['id'], 'ULD_ID': uid,
                    'x0': -1, 'y0': -1, 'z0': -1,
                    'x1': -1, 'y1': -1, 'z1': -1,
                    'reason': 'placed',
                })
        return placements

    def _pack_uld(self, uid, pids, uld_lookup, pkg_lookup):
        priority_rows = [pkg_lookup[pid] for pid in pids if pkg_lookup[pid]['Type'] == 'Priority']
        economy_rows  = [pkg_lookup[pid] for pid in pids if pkg_lookup[pid]['Type'] != 'Priority']
        bin_row = uld_lookup[uid]

        placements = []
        left_behind_ids = []

        prio_used_weight = 0.0
        prio_used_space_frac = 0.0
        if priority_rows:
            resp = self._call_api(bin_row, priority_rows)
            for bp in resp.get('bins_packed', []):
                prio_used_weight = max(prio_used_weight, bp['bin_data'].get('weight', 0.0))
                prio_used_space_frac = max(prio_used_space_frac,
                                            bp['bin_data'].get('used_space', 0.0) / 100.0)
            placements.extend(self._extract_placements(resp, uid))
            left_behind_ids.extend(item['id'] for item in resp.get('not_packed_items', []))

        if economy_rows:
            # Shrink ALL THREE dimensions by the cube root of the remaining
            # volume fraction, rather than dumping the whole reduction into
            # Height alone. Found by direct testing: an all-into-Height
            # shrink can push Height below many Economy items' own height
            # (e.g. residual Height=52.8 vs items with h=90-110), and instead
            # of gracefully reporting those as not-packed (as the docs
            # describe), the live API returns a 500 Internal Server Error --
            # a server-side robustness bug in their handling of that shape,
            # not something fixable on our end. Cube-root scaling reduces
            # total volume by the same fraction while keeping every
            # individual axis reasonably sized, avoiding the crash.
            remaining_frac = max(1 - prio_used_space_frac, 0.0)
            shrink = remaining_frac ** (1 / 3)
            residual_bin = dict(bin_row)
            residual_bin['Weight_Limit'] = max(bin_row['Weight_Limit'] - prio_used_weight, 0.0)
            residual_bin['Length'] = max(bin_row['Length'] * shrink, 1.0)
            residual_bin['Width']  = max(bin_row['Width']  * shrink, 1.0)
            residual_bin['Height'] = max(bin_row['Height'] * shrink, 1.0)
            resp = self._call_api(residual_bin, economy_rows)
            placements.extend(self._extract_placements(resp, uid))
            left_behind_ids.extend(item['id'] for item in resp.get('not_packed_items', []))

        return placements, left_behind_ids

    def pack(self, assignment, packages_df, ulds_df):
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

        total_unfit = 0
        for uid, pids in uld_pkg_ids.items():
            if not pids:
                continue
            print(f'  [api_packer] packing {uid}: {len(pids)} packages...')
            uld_placements, left_behind_ids = self._pack_uld(uid, pids, uld_lookup, pkg_lookup)
            placements.extend(uld_placements)
            for pid in left_behind_ids:
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                    'x0': -1, 'y0': -1, 'z0': -1,
                                    'x1': -1, 'y1': -1, 'z1': -1,
                                    'reason': 'packer_unfit'})
                total_unfit += 1

        return placements, total_unfit


def rescue_unfit_economy_via_api(placements, assignment, packages_df, ulds_df, api_packer):
    """
    Targeted second pass: leaves Priority placement entirely alone (already
    correctly handled by whatever packer produced `placements` -- this
    function trusts that completely and never touches it), and only retries
    Economy packages that packer marked 'packer_unfit', giving the
    3dbinpacking.com API a shot at squeezing them into whatever REAL
    capacity is left in their originally-assigned ULD.

    Why this and not using ThreeDBinPackingAPIPacker as the whole packer:
    direct comparison on the real 400-package instance showed the API's own
    algorithm is WORSE than RLPackerAdapter at fitting Priority packages (8
    priority packages the API called not_packed_items that RLPackerAdapter
    placed successfully in the identical ULD/assignment) but BETTER at
    Economy specifically (150/249 Economy packages fit vs our 103/249). This
    keeps the proven-safe Priority placement and only asks the API to do the
    one thing it's actually shown to be better at.

    Residual capacity uses REAL placed weight/volume for reason='placed'
    entries (summed directly from `placements`' own Weight and coordinate
    deltas) -- more accurate than _pack_uld's own Priority-only-call
    estimate, since this reflects what's ACTUALLY there after a full real
    pack. reason='rescued_by_api' entries (from a PRIOR call to this same
    function -- safe to call again on its own output for a second rescue
    round) use NOMINAL package volume instead, since the API never returns
    real coordinates (see module docstring) -- slightly less accurate but
    the only information available for those.

    Returns a NEW placements list (input is not mutated) with any newly-
    rescued packages moved from packer_unfit/NONE to their real ULD,
    'reason': 'rescued_by_api'.
    """
    uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
    pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid

    used_weight = {uid: 0.0 for uid in uld_lookup}
    used_volume = {uid: 0.0 for uid in uld_lookup}
    unfit_by_uld = {}

    for p in placements:
        pid = p['Package_ID']
        if p['reason'] == 'placed':
            uid = p['ULD_ID']
            used_weight[uid] += pkg_lookup[pid]['Weight']
            used_volume[uid] += (max(p['x1'] - p['x0'], 0) * max(p['y1'] - p['y0'], 0)
                                  * max(p['z1'] - p['z0'], 0))
        elif p['reason'] == 'rescued_by_api':
            uid = p['ULD_ID']
            used_weight[uid] += pkg_lookup[pid]['Weight']
            used_volume[uid] += (pkg_lookup[pid]['Length'] * pkg_lookup[pid]['Width']
                                  * pkg_lookup[pid]['Height'])
        elif p['reason'] == 'packer_unfit' and pkg_lookup[pid]['Type'] != 'Priority':
            orig_uid = assignment.get(pid)
            if orig_uid is not None and orig_uid in uld_lookup:
                unfit_by_uld.setdefault(orig_uid, []).append(pid)

    rescued_ids = set()
    rescued_placements = []
    for uid, pids in unfit_by_uld.items():
        bin_row = uld_lookup[uid]
        uld_volume = bin_row['Length'] * bin_row['Width'] * bin_row['Height']
        remaining_frac = max(1 - used_volume[uid] / uld_volume, 0.0) if uld_volume else 0.0
        shrink = remaining_frac ** (1 / 3)
        residual_bin = dict(bin_row)
        residual_bin['Weight_Limit'] = max(bin_row['Weight_Limit'] - used_weight[uid], 0.0)
        residual_bin['Length'] = max(bin_row['Length'] * shrink, 1.0)
        residual_bin['Width']  = max(bin_row['Width']  * shrink, 1.0)
        residual_bin['Height'] = max(bin_row['Height'] * shrink, 1.0)

        print(f'  [rescue] {uid}: retrying {len(pids)} packer_unfit Economy packages '
              f'against residual capacity ({remaining_frac:.1%} volume, '
              f'{residual_bin["Weight_Limit"]:.0f}kg weight remaining)...')
        pkg_rows = [pkg_lookup[pid] for pid in pids]
        try:
            resp = api_packer._call_api(residual_bin, pkg_rows)
        except requests.exceptions.HTTPError as exc:
            # Server-side robustness bug (see module docstring): a residual
            # bin too small relative to remaining items' own dimensions can
            # 500 instead of gracefully reporting them not-packed. Treat as
            # "nothing rescued here" and move on -- one ULD's failure
            # shouldn't lose whatever other ULDs' calls already succeeded.
            print(f'    -> API call failed ({exc}), treating as 0 rescued for {uid}')
            continue
        for bp in resp.get('bins_packed', []):
            for it in bp['items']:
                rescued_ids.add(it['id'])
                rescued_placements.append({
                    'Package_ID': it['id'], 'ULD_ID': uid,
                    'x0': -1, 'y0': -1, 'z0': -1,
                    'x1': -1, 'y1': -1, 'z1': -1,
                    'reason': 'rescued_by_api',
                })
        print(f'    -> rescued {len(rescued_ids & set(pids))}/{len(pids)}')

    new_placements = [p for p in placements if p['Package_ID'] not in rescued_ids]
    new_placements.extend(rescued_placements)
    return new_placements
