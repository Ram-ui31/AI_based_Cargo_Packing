# ARGO desktop app

Packages the exact same backend (`../backend/app.py`, same models, same
checkpoints, same frontend) into a standalone app that runs on the user's
own machine -- no server, no hosting cost, no install step beyond running
the built app. Runs entirely on the user's own CPU, so speed depends on
their hardware rather than a hosting plan's CPU allocation.

## Build it

```bash
cd desktop
bash build.sh
```

This creates an isolated build environment (`.buildenv/`, CPU-only PyTorch
only -- keeps the build from sweeping in unrelated packages from your normal
Python environment) and runs PyInstaller. Output lands in `dist/ARGO/`.

**Run the built app**: double-click `dist/ARGO/ARGO` (or `dist/ARGO/ARGO.exe`
on Windows) -- it starts a local server and opens your browser to it
automatically.

## Important: PyInstaller is not cross-platform

`build.sh` must be run **on** each OS you want to support -- a macOS build
only produces a macOS app; Windows and Linux each need their own build run
on that OS (or via CI, e.g. a GitHub Actions matrix with
`windows-latest`/`macos-latest`/`ubuntu-latest` runners, which are free for
public repos).

## Size

The build is roughly 550MB, almost entirely PyTorch's CPU runtime (~410MB)
-- normal for a bundled ML app, not a bug. Building inside a fresh
`.buildenv/` (rather than your regular dev Python environment) is what keeps
it from being several times larger; skipping that step and building with a
Python environment that has other ML libraries installed (transformers,
tensorflow, etc.) will sweep those in too even though nothing here uses them.

## How this works

`launcher.py` is the actual PyInstaller entry point -- it starts the same
FastAPI app as `python3 backend/app.py` would, then opens it in the default
browser. `backend/app.py` detects `sys.frozen` (set by PyInstaller) and
switches to reading `frontend/`, `demo_data/`, `models/` from next to the
built executable instead of the source-tree layout, and writes uploads to
the OS temp dir instead of next to the app (the install location may not be
writable).
