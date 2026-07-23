"""
ARGO desktop launcher -- the actual entry point PyInstaller freezes.

Starts the real FastAPI backend (backend/app.py, completely unmodified logic,
same models/checkpoints, same everything) on localhost, then opens it in the
user's default browser. No server, no install, no terminal commands -- the
user just double-clicks the built app.

Run directly with `python3 desktop/launcher.py` to test unfrozen (uses the
regular 06-website/ layout); once frozen by PyInstaller (see build.sh, which
passes --paths so PyInstaller's analyzer can find and bundle app.py's actual
code), backend/app.py detects sys.frozen and reads frontend/, demo_data/,
models/ from next to the built executable instead.
"""
from __future__ import annotations

# Dummy imports: run_<model>.py (loaded dynamically at runtime via
# importlib, as plain data files PyInstaller never statically analyzes)
# depends on these, but nothing PyInstaller *can* see statically-imports
# them -- without these lines here, PyInstaller won't bundle them at all
# and the frozen app fails at runtime with "No module named 'torch'" etc.
import numpy  # noqa: F401
import pandas  # noqa: F401
import torch  # noqa: F401
import tqdm  # noqa: F401
import tqdm.auto  # noqa: F401  -- train_rl.py imports specifically from here

import os
import socket
import sys
import threading
import time
import webbrowser

if not getattr(sys, 'frozen', False):
    # Only needed for `python3 desktop/launcher.py` (testing unfrozen) --
    # once frozen, PyInstaller bundles app.py's actual code directly (see
    # build.sh's --paths flag), so this branch never runs then.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_HERE, '..', 'backend'))

import app as backend_app  # the FastAPI `app` object + all its route handlers


def _find_free_port(preferred=8000, tries=10):
    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return preferred  # give up, let uvicorn raise a clear error


def _open_browser_when_ready(url, timeout=15):
    deadline = time.time() + timeout
    port = int(url.rsplit(':', 1)[1])
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.3)
    webbrowser.open(url)  # last resort -- open anyway


if __name__ == '__main__':
    port = _find_free_port(8000)
    url = f'http://127.0.0.1:{port}'
    print(f'\nARGO starting -- opening {url} in your browser...\n')

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    import uvicorn
    # Pin to the pure-Python loop/http backends -- uvicorn's optional
    # accelerators (uvloop, httptools) are picked dynamically at runtime,
    # which PyInstaller's static import analysis can miss entirely,
    # breaking only once frozen. Plenty fast for a single local user.
    uvicorn.run(backend_app.app, host='127.0.0.1', port=port, loop='asyncio', http='h11')
