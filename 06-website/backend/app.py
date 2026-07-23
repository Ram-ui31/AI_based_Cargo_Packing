"""
ARGO web app backend -- FastAPI server that:
  - serves the frontend (static HTML/CSS/JS + images)
  - serves precomputed demo results (real 400-package benchmark instance,
    one result per model, precomputed by demo_data/precompute_demo.py)
  - accepts a user-uploaded CSV instance file + model choice, actually runs
    that model's real run_cherry.py / run_eclipse.py / run_halley.py logic
    (imported directly, not shelled out to), tracking progress in-memory
    so the frontend can poll a real progress bar
  - serves the final result as downloadable JSON

Run with: python3 app.py   (then open http://localhost:8000)
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# On constrained/throttled hosts (e.g. a free-tier container), torch's default
# thread count follows the host's *reported* CPU count, which can be far more
# than the CPU share actually granted -- causing thread contention on top of
# the throttling. Only acts if TORCH_NUM_THREADS is explicitly set (e.g. in
# Render's dashboard), so local dev behavior (and the desktop app, which wants
# the user's full real CPU) is completely unchanged.
if os.environ.get('TORCH_NUM_THREADS'):
    import torch
    torch.set_num_threads(int(os.environ['TORCH_NUM_THREADS']))

FROZEN = getattr(sys, 'frozen', False)  # True inside a PyInstaller-built desktop app

if FROZEN:
    # Where frontend/, demo_data/, models/ (bundled as data, see desktop/build.sh)
    # actually land depends on PyInstaller's build mode: --onefile extracts them
    # to a temp dir at sys._MEIPASS; --onedir (what we use) puts them under an
    # _internal/ folder next to the executable as of PyInstaller 6+, or directly
    # beside it on older versions -- check all three so this isn't tied to one
    # specific PyInstaller version/mode. No sibling 01-cherry/etc. ever exists in
    # this layout, so MODEL_FOLDERS always resolves to the bundle either way.
    if hasattr(sys, '_MEIPASS'):
        _BASE = sys._MEIPASS
    else:
        _internal = os.path.join(os.path.dirname(sys.executable), '_internal')
        _BASE = _internal if os.path.isdir(_internal) else os.path.dirname(sys.executable)
    REPO_ROOT = _BASE
    BUNDLED_MODELS_DIR = os.path.join(_BASE, 'models')
    FRONTEND_DIR = os.path.join(_BASE, 'frontend')
    DEMO_DATA_DIR = os.path.join(_BASE, 'demo_data')
    # writes go to the OS temp dir, not next to the app bundle, since the
    # install location (Program Files, /Applications, ...) may not be writable
    UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'argo_uploads')
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.join(HERE, '..', '..')  # git/ -- parent of 06-website, sibling of 01-cherry etc.
    BUNDLED_MODELS_DIR = os.path.join(HERE, '..', 'models')  # self-contained copies, used when deployed standalone
    FRONTEND_DIR = os.path.join(HERE, '..', 'frontend')
    DEMO_DATA_DIR = os.path.join(HERE, 'demo_data')
    UPLOAD_DIR = os.path.join(HERE, 'uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _resolve_model_folder(sibling_name):
    """Prefers the live sibling folder (01-cherry/02-eclipse/03-halley next to
    06-website) used during local development, and falls back to the
    self-contained models/ bundle -- used when this app is deployed on its
    own (e.g. a Hugging Face Space), where the sibling repos don't exist."""
    sibling = os.path.join(REPO_ROOT, sibling_name)
    if os.path.isdir(sibling):
        return sibling
    return os.path.join(BUNDLED_MODELS_DIR, sibling_name)


MODEL_FOLDERS = {
    'cherry': _resolve_model_folder('01-cherry'),
    'eclipse': _resolve_model_folder('02-eclipse'),
    'halley': _resolve_model_folder('03-halley'),
}
MODEL_MODULES = {}  # lazily imported run_<model> modules, keyed by name


def _load_model_module(model):
    """Imports run_<model>.py from that model's own folder, isolated per
    model so their identically-named internal modules (src.model_core.*
    etc.) don't collide with each other in sys.modules."""
    if model in MODEL_MODULES:
        return MODEL_MODULES[model]
    folder = MODEL_FOLDERS[model]
    import importlib.util
    # Each model folder's run_*.py does `sys.path.insert(0, HERE)` itself
    # and imports absolute `src.*` -- to avoid the three models' `src`
    # packages colliding in sys.modules, we import each run script fresh
    # and immediately drop its `src` submodules from sys.modules after
    # load, keeping only the run_<model> module object itself (which has
    # already bound the correct classes/functions via closures at import
    # time).
    script_path = os.path.join(folder, f'run_{model}.py')
    spec = importlib.util.spec_from_file_location(f'run_{model}_isolated', script_path)
    mod = importlib.util.module_from_spec(spec)
    old_sys_path = list(sys.path)
    old_cwd = os.getcwd()
    try:
        os.chdir(folder)  # checkpoints are referenced relative to HERE inside the script, which is fine (uses its own __file__), but keep cwd sane anyway
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_sys_path
        os.chdir(old_cwd)
        for name in list(sys.modules):
            if name == 'src' or name.startswith('src.') or name in ('model', 'features', 'constants'):
                del sys.modules[name]
    MODEL_MODULES[model] = mod
    return mod


# ---------------------------------------------------------------------------
# In-memory job tracking for the real progress bar
# ---------------------------------------------------------------------------
JOBS = {}  # job_id -> {"status": ..., "progress": 0-100, "message": ..., "result": ...}
JOBS_LOCK = threading.Lock()


def _set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _run_job(job_id, model, input_path, search_rounds):
    try:
        _set_job(job_id, status='running', progress=5, message='Loading trained checkpoints...')
        mod = _load_model_module(model)

        def progress_cb(fraction, message):
            _set_job(job_id, progress=int(5 + fraction * 90), message=message)

        run_fn = getattr(mod, f'run_{model}')
        metrics, placements = run_fn(input_path, device='cpu', search_rounds=search_rounds,
                                      progress_cb=progress_cb)

        # re-parse the same CSV to hand the frontend the ULD dimensions and
        # per-package Type/dims it needs for the 3D view (not returned by
        # run_<model> itself, which only tracks placement geometry)
        _, ulds_df, pkgs_df = mod.parse_input_csv(input_path)
        ulds = ulds_df.to_dict('records')
        packages = pkgs_df.to_dict('records')

        _set_job(job_id, status='done', progress=100, message='Done',
                 result={'metrics': metrics, 'placements': placements,
                         'ulds': ulds, 'packages': packages})
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, status='error', progress=100, message=f'Error: {e}')


app = FastAPI(title='ARGO')


@app.get('/api/demo/{model}')
def get_demo(model: str):
    if model not in MODEL_FOLDERS:
        raise HTTPException(404, 'unknown model')
    metrics_path = os.path.join(DEMO_DATA_DIR, model, 'final_metrics.json')
    placements_path = os.path.join(DEMO_DATA_DIR, model, 'final_placements.json')
    uld_path = os.path.join(DEMO_DATA_DIR, 'ulds.json')
    packages_path = os.path.join(DEMO_DATA_DIR, 'packages.json')
    if not os.path.exists(metrics_path):
        raise HTTPException(503, 'demo results not precomputed yet -- run backend/demo_data/precompute_demo.py')
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(placements_path) as f:
        placements = json.load(f)
    with open(uld_path) as f:
        ulds = json.load(f)
    with open(packages_path) as f:
        packages = json.load(f)
    return {'metrics': metrics, 'placements': placements, 'ulds': ulds, 'packages': packages}


@app.post('/api/pack')
async def pack(model: str = Form(...), search_rounds: int = Form(10), file: UploadFile = File(...)):
    if model not in MODEL_FOLDERS:
        raise HTTPException(404, 'unknown model')
    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f'{job_id}.csv')
    with open(input_path, 'wb') as f:
        f.write(await file.read())

    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'queued', 'progress': 0, 'message': 'Queued...', 'result': None}

    t = threading.Thread(target=_run_job, args=(job_id, model, input_path, search_rounds), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.get('/api/status/{job_id}')
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, 'unknown job')
    # don't ship the (potentially large) result until it's actually done
    resp = {k: v for k, v in job.items() if k != 'result'}
    if job['status'] == 'done':
        resp['result'] = job['result']
    return resp


app.mount('/', StaticFiles(directory=FRONTEND_DIR, html=True), name='frontend')


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    print(f'\nARGO running at http://localhost:{port}\n')
    uvicorn.run(app, host='0.0.0.0', port=port)
