# ARGO -- the showdown

A full-stack demo of the 3 ARGO models (Cherry, Eclipse, Halley): browse what
each model is, watch a precomputed 3D packing of the real 400-package
benchmark instance, or upload your own instance CSV and watch it get packed
live by any of the three models.

No external services and no CDN dependency — Three.js is vendored locally in
`frontend/vendor/`, and the three models' trained checkpoints + code are
bundled in `models/` so this folder runs standalone, whether that's on your
own machine or deployed as a web service.

## Run it locally

```bash
cd 06-website
pip install -r requirements.txt
python3 backend/app.py
```

Then open **http://localhost:8000** in your browser. That's it — the
backend serves the frontend directly, so there's nothing else to start.

(If you're running this from inside the full `cargoism/git` checkout, the
app actually prefers the live `01-cherry`/`02-eclipse`/`03-halley` folders
next to `06-website` over its own `models/` copies — see "Deploying" below.)

## Deploying it (Render, free tier)

Hugging Face now requires a paid PRO plan for any Space that runs real
compute (Docker/Gradio) — only static Spaces stay free there, which can't run
this app's backend. **Render**'s free web-service tier still works: it builds
straight from the `Dockerfile` already in this folder and reads the `$PORT`
env var the same way `app.py` already expects
(`os.environ.get('PORT', 8000)`) — no code changes needed.

Steps:

1. Push this repo (or at least the `06-website/` folder) to GitHub — it's
   already part of the `AI_based_Cargo_Packing` repo on `origin`.
2. At render.com: **New > Web Service** → connect that GitHub repo.
3. Set **Root Directory** to `06-website` (this is a monorepo — that tells
   Render to build only this subfolder's `Dockerfile`, ignoring the model
   research folders alongside it).
4. Plan: **Free**. Render auto-detects the Dockerfile and deploys.

First build takes a few minutes (installing the CPU torch wheel); after that
it's a live site at `https://<service-name>.onrender.com`.

Free-tier services on Render spin down after ~15 minutes of inactivity and
cold-start (~30-60s) on the next visit — normal for the free tier, not a bug.

## What each part of the site does

- **About our models** — a short write-up of Cherry, Eclipse and Halley's
  actual architectures (what's shared, what each one adds).
- **Get a demo** — a precomputed 3D visualization of all 6 ULDs from the real
  400-package benchmark instance, one result per model (switch with the
  dropdown). These results were generated once via
  `backend/demo_data/precompute_demo.py` and are served as static JSON.
- **Use our product** — upload your own instance CSV (format shown on the
  page itself, and documented in `../05-run-instructions/README.md`), pick a
  model, and click "Pack this shipment." This actually runs that model's real
  `run_<model>.py` pipeline in the background (RL placement ensemble +
  trained checkpoints), with a genuine progress bar polling the job's real
  stage-by-stage progress — not a simulated one. When it finishes you land on
  the same kind of 3D viewer, with a "Download .json" button for the full
  result.

## Notes on the demo numbers

The precomputed demo results use a lighter local-search budget
(`--search-rounds 10`) than the multi-hour research-grade search behind the
numbers in the competition report, so they won't exactly match those —
they're still real, valid, zero-overlap packings from the real models, just
computed under a demo-friendly time budget. The relative ordering
(Cherry < Eclipse < Halley on total cost) matches the report.

## Project layout

```
06-website/
├── Dockerfile                 CPU-only container build, used for Render deploys
├── requirements.txt
├── backend/
│   ├── app.py                 FastAPI server (frontend + demo API + live pack API)
│   └── demo_data/
│       ├── precompute_demo.py    regenerates the 3 precomputed demo results
│       ├── demo_instance.csv     the real 400-package benchmark instance
│       ├── ulds.json             the 6 real-instance ULD dimensions
│       ├── packages.json         Package_ID -> Type/dims for the demo instance
│       └── {cherry,eclipse,halley}/final_metrics.json, final_placements.json
├── models/                    self-contained copies used when 01-cherry/
│   ├── 01-cherry/, 02-eclipse/, 03-halley/    etc. aren't available next to
│   ├── rl_packer/                             06-website (e.g. deployed on
│   └── gnn_economy_selector/                  Render) -- see app.py's
│                                               _resolve_model_folder()
└── frontend/
    ├── index.html, style.css, app.js, viewer.js (Three.js packing renderer)
    ├── vendor/           three.min.js + OrbitControls.js (vendored, offline)
    └── assets/           background imagery
```
