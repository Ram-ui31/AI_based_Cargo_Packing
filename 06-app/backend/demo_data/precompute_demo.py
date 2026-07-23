"""
Regenerates the precomputed "Get a demo" results for all 3 models by running
their real run_<model>.py pipelines against demo_instance.csv (the same real
400-package benchmark instance used throughout this project).

Run from anywhere:
    python3 backend/demo_data/precompute_demo.py

Writes {cherry,eclipse,halley}/final_metrics.json and final_placements.json
next to this script. ulds.json and packages.json (shared across all 3
models, since it's the same instance) are derived from demo_instance.csv
and only need to be regenerated if the instance itself changes -- this
script leaves them alone.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, '..', '..', '..')  # git/ -- parent of 06-website, sibling of 01-cherry etc.
INSTANCE_CSV = os.path.join(HERE, 'demo_instance.csv')

MODEL_FOLDERS = {
    'cherry': os.path.join(REPO_ROOT, '01-cherry'),
    'eclipse': os.path.join(REPO_ROOT, '02-eclipse'),
    'halley': os.path.join(REPO_ROOT, '03-halley'),
}


def _load_model_module(model):
    folder = MODEL_FOLDERS[model]
    script_path = os.path.join(folder, f'run_{model}.py')
    spec = importlib.util.spec_from_file_location(f'run_{model}_isolated', script_path)
    mod = importlib.util.module_from_spec(spec)
    old_sys_path = list(sys.path)
    old_cwd = os.getcwd()
    try:
        os.chdir(folder)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_sys_path
        os.chdir(old_cwd)
        for name in list(sys.modules):
            if name == 'src' or name.startswith('src.') or name in ('model', 'features', 'constants'):
                del sys.modules[name]
    return mod


def main():
    for model in ('cherry', 'eclipse', 'halley'):
        print(f'\n=== Precomputing demo for {model} ===')
        mod = _load_model_module(model)
        run_fn = getattr(mod, f'run_{model}')

        def progress_cb(fraction, message, _model=model):
            print(f'  [{_model}] {fraction*100:5.1f}%  {message}')

        metrics, placements = run_fn(INSTANCE_CSV, device='cpu', search_rounds=10, progress_cb=progress_cb)

        out_dir = os.path.join(HERE, model)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'final_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(out_dir, 'final_placements.json'), 'w') as f:
            json.dump(placements, f, indent=2, default=str)
        print(f'Saved {out_dir}/final_metrics.json and final_placements.json')


if __name__ == '__main__':
    main()
