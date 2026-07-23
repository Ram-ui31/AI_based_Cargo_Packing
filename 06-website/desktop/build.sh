#!/usr/bin/env bash
# Builds the ARGO desktop app for the current OS using PyInstaller.
# Run from anywhere; output lands in 06-website/desktop/dist/ARGO/.
#
# NOTE: PyInstaller is NOT cross-platform -- this must be run ON each target
# OS (macOS build on macOS, Windows build on Windows, etc).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"   # desktop/
pip install --quiet pyinstaller

rm -rf build dist ARGO.spec

# macOS/Linux use ':' between src and dest in --add-data; Windows uses ';'.
SEP=":"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  SEP=";"
fi

pyinstaller launcher.py \
  --name ARGO \
  --paths ../backend \
  --add-data "../frontend${SEP}frontend" \
  --add-data "../backend/demo_data${SEP}demo_data" \
  --add-data "../models${SEP}models" \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops.asyncio \
  --hidden-import uvicorn.protocols.http.h11_impl \
  --hidden-import uvicorn.protocols.websockets.wsproto_impl \
  --hidden-import uvicorn.lifespan.on \
  --hidden-import tqdm.auto \
  --noconfirm

echo ""
echo "Built: desktop/dist/ARGO/  -- double-click the ARGO executable inside it."
