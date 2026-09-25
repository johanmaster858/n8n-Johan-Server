#!/bin/bash
# Render every frame with two worker processes (2 numba threads each), then encode.
set -e
cd "$(dirname "$0")"
mkdir -p build
python3 render.py assets
NUMBA_NUM_THREADS=2 python3 render.py frames 0 1200 2 > build/render_even.log 2>&1 &
NUMBA_NUM_THREADS=2 python3 render.py frames 1 1200 2 > build/render_odd.log 2>&1 &
wait
python3 audio.py build/soundtrack.wav
python3 render.py encode
python3 contact_sheet.py build/frames output/contact_sheet.jpg
