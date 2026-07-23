"""
verify_api_axis_mapping.py -- one small, cheap API call to confirm (not
assume) how 3dbinpacking.com's response coordinates map to our own
(Length, Width, Height) axes, before trusting api_packer.py on the real
400-package instance. See ~/Desktop/3dbinpacking_api/api_reference.md's
"Axis Convention" open item -- the docs never confirmed this.

Method: build a bin JUST BARELY bigger than a single item with three
DISTINCT dimensions, in only ONE specific orientation. Any rotation that
swaps two of the item's axes would need the bin to be big enough for the
swapped dimensions too -- it isn't, by construction -- so if the item packs
at all, its placement is geometrically forced into exactly one orientation,
regardless of what the API's rotation settings do internally. That lets us
read the true axis correspondence directly off the returned coordinate
deltas, with no dependency on trusting the 'vr' parameter's exact semantics.

Item (our convention): Length=20, Width=10, Height=5.
Bin: just 1 unit bigger in each of our own axes: Length=21, Width=11, Height=6.
A 90-degree rotation on any axis pair (e.g. swap Length<->Height, needing a
bin with Height>=20) cannot fit in a bin with Height=6 -- so the ONLY way
this item fits at all is unrotated, in its declared orientation.

Expected result if the request/response axis correspondence is the
straightforward one assumed in api_packer.py (x<->w<->Length,
y<->h<->Height, z<->d<->Width):
    x2-x1 == 20  (Length)
    y2-y1 == 5   (Height)
    z2-z1 == 10  (Width)
If the deltas come back in a different arrangement, api_packer.py's
_extract_placements mapping needs to be corrected to match what's printed
here before the real run is trustworthy.

Usage:
    export THREEDBINPACKING_USERNAME=...
    export THREEDBINPACKING_API_KEY=...
    python scripts/verify_api_axis_mapping.py
"""
from __future__ import annotations
import os

import json as json_module

import requests

sys_path_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
import sys
sys.path.insert(0, sys_path_root)
from src.rl.api_packer import _load_credentials  # noqa: E402

API_URL = 'https://global-api.3dbinpacking.com/packer/packIntoMany'

# Our convention: Length, Width, Height -- all distinct so no ambiguity.
ITEM_LENGTH, ITEM_WIDTH, ITEM_HEIGHT, ITEM_WEIGHT = 20, 10, 5, 50
BIN_LENGTH, BIN_WIDTH, BIN_HEIGHT, BIN_WEIGHT_LIMIT = 21, 11, 6, 1000


def main():
    username, api_key = _load_credentials()
    if not (username and api_key):
        raise SystemExit(
            'Missing credentials. Set THREEDBINPACKING_USERNAME and '
            'THREEDBINPACKING_API_KEY in the environment -- see '
            '~/Desktop/3dbinpacking_api/api_reference.md for how to obtain them.'
        )

    payload = {
        'username': username, 'api_key': api_key,
        'bins': [{
            'id': 'TEST_BIN',
            'w': BIN_LENGTH, 'h': BIN_HEIGHT, 'd': BIN_WIDTH,
            'max_wg': BIN_WEIGHT_LIMIT, 'q': 1, 'type': 'box',
        }],
        'items': [{
            'id': 'TEST_ITEM',
            'w': ITEM_LENGTH, 'h': ITEM_HEIGHT, 'd': ITEM_WIDTH,
            'wg': ITEM_WEIGHT, 'q': 1, 'vr': 0,
        }],
        'params': {'optimization_mode': 'bins_utilization'},
    }
    print('Request bin  (our L,W,H) = '
          f'({BIN_LENGTH}, {BIN_WIDTH}, {BIN_HEIGHT})  ->  API (w,h,d) = '
          f'({BIN_LENGTH}, {BIN_HEIGHT}, {BIN_WIDTH})')
    print('Request item (our L,W,H) = '
          f'({ITEM_LENGTH}, {ITEM_WIDTH}, {ITEM_HEIGHT})  ->  API (w,h,d) = '
          f'({ITEM_LENGTH}, {ITEM_HEIGHT}, {ITEM_WIDTH})')
    print()

    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()['response']
    print('RAW RESPONSE:')
    print(json_module.dumps(data, indent=2))
    print()

    if data.get('status') == 0 or not data.get('bins_packed'):
        print('FAILED: item was not packed at all.')
        print('errors:', data.get('errors'))
        print('not_packed_items:', data.get('not_packed_items'))
        print()
        print('This likely means vr=0 does NOT mean "no rotation" the way assumed, '
              'or the bin-just-barely-bigger trick needs a larger margin than 1 unit. '
              'Try loosening the bin dimensions slightly and re-run, or inspect the '
              'raw response below.')
        print(data)
        return

    item = data['bins_packed'][0]['items'][0]
    print(f'Item keys returned: {list(item.keys())}')
    if 'coordinates' not in item:
        print('No "coordinates" key -- schema differs from api_reference.md\'s example. '
              'See RAW RESPONSE above to find the real field names, then fix this '
              'script and api_packer.py\'s _extract_placements to match.')
        return
    c = item['coordinates']
    dx = c['x2'] - c['x1']
    dy = c['y2'] - c['y1']
    dz = c['z2'] - c['z1']
    print(f'Returned coordinates: {c}')
    print(f'Deltas: dx={dx}  dy={dy}  dz={dz}')
    print()

    expected = {'x': ITEM_LENGTH, 'y': ITEM_HEIGHT, 'z': ITEM_WIDTH}
    actual = {'x': dx, 'y': dy, 'z': dz}
    if actual == expected:
        print('CONFIRMED: axis mapping in api_packer.py is correct as written.')
        print('  x <-> w <-> our Length')
        print('  y <-> h <-> our Height (our vertical axis)')
        print('  z <-> d <-> our Width')
    else:
        print('MISMATCH -- api_packer.py\'s _extract_placements needs correcting.')
        print(f'  expected (x,y,z) deltas = ({ITEM_LENGTH}, {ITEM_HEIGHT}, {ITEM_WIDTH})')
        print(f'  actual   (x,y,z) deltas = ({dx}, {dy}, {dz})')
        # Try to identify which of our axes each response axis actually matches.
        our_dims = {'Length': ITEM_LENGTH, 'Width': ITEM_WIDTH, 'Height': ITEM_HEIGHT}
        for axis_name, delta in actual.items():
            matches = [name for name, val in our_dims.items() if val == delta]
            print(f'  response {axis_name}-delta={delta}  ->  matches our {matches}')


if __name__ == '__main__':
    main()
