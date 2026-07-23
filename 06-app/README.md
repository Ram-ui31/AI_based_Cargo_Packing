# ARGO -- the showdown

Browse what each of the 3 ARGO models (Cherry, Eclipse, Halley) is, watch a
precomputed 3D packing of the real 400-package benchmark instance, or get
the app to pack your own instance for real.

Three ways to actually use ARGO:

1. **The live site** — landing page, model write-ups, and the "Get a demo"
   3D viewer, all fully static (no server, no backend, just files) —
   see "Deploying (GitHub Pages)" below.
2. **The desktop app** — download it from the site's "Use our product" page
   (or straight from this repo's [Releases](../../releases)), and it runs
   the real models on your own computer, at your own computer's speed. No
   install, no Python.
3. **From source** — clone the repo and run the models directly via the
   commands in [`../05-run-instructions/README.md`](../05-run-instructions/README.md),
   for anyone who'd rather not download an app at all.

Only path 2 needs any compute — the live site (path 1) is 100% static
files, and no longer runs a live backend for real-time packing (that used
to be a "Use our product" upload flow on the site itself; it's now the
desktop app's job instead, since that runs on the user's own hardware for
free rather than a throttled/paid server).

## Run it locally (full dev version, with the live backend)

```bash
cd 06-app
pip install -r requirements.txt
python3 backend/app.py
```

Then open **http://localhost:8000**. This is the same server the desktop
app freezes — useful for local development/testing of the live-pack path,
but not what the deployed website runs (see below).

(If you're running this from inside the full `cargoism/git` checkout, the
app actually prefers the live `01-cherry`/`02-eclipse`/`03-halley` folders
next to `06-app` over its own `models/` copies.)

## Deploying the live site (GitHub Pages, free, static)

The `frontend/` folder is fully self-contained and static — no backend
needed. `.github/workflows/deploy-pages.yml` auto-deploys it on every push
to `main` that touches `06-app/frontend/`.

One-time setup: repo **Settings → Pages → Source → GitHub Actions**. After
that it's automatic — push a change, the workflow runs, the site updates.

## Optional: self-hosting the live-pack backend

The `Dockerfile` + `backend/app.py`'s `/api/pack` live-execution path still
work if you want to self-host real-time packing as a web service instead of
pointing people at the desktop app (e.g. Render's free tier, though it's
throttled — see the desktop app instead for real speed). Not the
recommended path anymore, but not removed either, in case it's useful.

## What each part of the site does

- **About our models** — a short write-up of Cherry, Eclipse and Halley's
  actual architectures (what's shared, what each one adds).
- **Get a demo** — a precomputed 3D visualization of all 6 ULDs from the
  real 400-package benchmark instance, one result per model (switch with
  the dropdown). Generated once via `backend/demo_data/precompute_demo.py`,
  fetched by the frontend as plain static JSON from `frontend/demo_data/`.
- **Use our product** — download links for the desktop app (macOS/Windows/
  Linux, auto-highlighting your platform), plus a fallback link to run the
  models from source instead.

## Notes on the demo numbers

The precomputed demo results use a lighter local-search budget
(`--search-rounds 10`) than the multi-hour research-grade search behind the
numbers in the competition report, so they won't exactly match those —
they're still real, valid, zero-overlap packings from the real models, just
computed under a demo-friendly time budget. The relative ordering
(Cherry < Eclipse < Halley on total cost) matches the report.

## Project layout

```
06-app/
├── Dockerfile                 CPU-only container build -- optional, see above
├── requirements.txt
├── backend/
│   ├── app.py                 FastAPI server (static frontend + live pack API)
│   └── demo_data/
│       ├── precompute_demo.py    regenerates the demo results (writes into
│       │                          ../../frontend/demo_data/, see below)
│       └── demo_instance.csv     the real 400-package benchmark instance
├── models/                     self-contained copies used when 01-cherry/
│   ├── 01-cherry/, 02-eclipse/, 03-halley/     etc. aren't available next
│   ├── rl_packer/                              to 06-app (e.g. the desktop
│   └── economy-package-ranker/                 app build) -- see app.py's
│                                                _resolve_model_folder()
├── desktop/                   PyInstaller launcher + build script -- see
│                               desktop/README.md
└── frontend/                  the entire live site -- fully static, this is
    ├── index.html, style.css, app.js, viewer.js (Three.js packing renderer)   what GitHub Pages deploys
    ├── vendor/           three.min.js + OrbitControls.js (vendored, offline)
    ├── assets/           background imagery
    └── demo_data/        precomputed results, fetched directly as static JSON
        ├── ulds.json, packages.json
        └── {cherry,eclipse,halley}/final_metrics.json, final_placements.json
```
