#!/bin/bash
# Mod: fst-ep-local -- per-node fastsafetensors loading for the two-node TP2+EP recipes (no cross-node broadcast, each node
# reads only its own experts). List it AFTER mods/flashnext-swift15-b12x (or flashnext-int4-b12x): it wraps the loader those
# mods already patched. Single-node launches are unaffected; FST_EP_LOCAL=0 in the recipe env turns it off. See patch_fst.py.
set -euo pipefail
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -c "import fastsafetensors.parallel_loader as p, inspect; assert 'all_local' in inspect.signature(p.ParallelLoader).parameters" \
  || { echo "FATAL $(hostname): fastsafetensors in this image has no ParallelLoader(all_local=...) (needs >= 0.4)" >&2; exit 1; }
OUT=$(python3 "$MOD_DIR/patch_fst.py") || { echo "FATAL $(hostname): fst-ep-local patch failed" >&2; exit 1; }
echo "$(hostname): $OUT"
