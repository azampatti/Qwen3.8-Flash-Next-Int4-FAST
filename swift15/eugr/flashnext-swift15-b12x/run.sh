#!/bin/bash
# Mod: flashnext-swift15-b12x
# Lets eugr's vllm-node-b12x image serve Qwen3.8-Flash-Next-Swift1.5 (azampatti): the published
# azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound with Swift1.5's attention and a shared expert healed on that trunk,
# on one node or two (recipes: recipes/qwen3.8-flash-next-swift15-b12x-solo.yaml / qwen3.8-flash-next-swift15-b12x.yaml).
# Same code as mods/flashnext-int4-b12x; only the model it looks for differs.
#
# The launcher copies this folder into EVERY node's container and runs run.sh before the launch script; a non-zero exit
# aborts the launch, so a node with a missing or incomplete model never starts serving. Nothing outside the container is
# written, the Hugging Face cache is read-only here, and nothing is copied: the two folders below are symlinks + small JSON.
#
#   1. check the composed model in the HF cache (what ~/Qwen3.8-Flash-Next-Int4-FAST/swift15/setup.sh builds on top of the base model)
#   2. build /workspace/flashnext/model     (served model view: PLE table indexed, config fields b12x needs)   build_views.py
#            /workspace/flashnext/draft-k10 (slim MTP draft folder, top-k 10, the head's own shared-expert width)
#   3. patch this container's vLLM: fp8 hybrid, int4 lm_head, loader index filter (required); draft x2, GDN fixes (optional)
#                                                                                                                 patch_b12x.py
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/root/.cache/huggingface/hub/models--local--Qwen3.8-Flash-Next-Swift1.5
OUT=/workspace/flashnext
die(){ echo "FATAL $(hostname): $*" >&2; exit 1; }

[ -f "$REPO/refs/main" ] || die "Swift1.5 variant not set up on this node -- run ~/Qwen3.8-Flash-Next-Int4-FAST/swift15/setup.sh on this node"
REV=$(cat "$REPO/refs/main"); SNAP="$REPO/snapshots/$REV"
[ -d "$SNAP/ple-table" ] || die "variant ${REV} has no ple-table (base model incomplete?) -- re-run ~/Qwen3.8-Flash-Next-Int4-FAST/swift15/setup.sh"
for f in config.json model.safetensors.index.json model-healed-shared-expert.safetensors model-swift15-attn.safetensors model_extra_tensors.safetensors \
         model-lmhead-int4.safetensors medium_chat_template.jinja; do
  [ -e "$SNAP/$f" ] || die "variant ${REV} is incomplete: $f is missing or a dangling link -- re-run setup.sh"
done
python3 -c "import b12x" 2>/dev/null || die "this is not the b12x image (no b12x package) -- the recipe needs vllm-node-b12x"

mkdir -p "$OUT"
VIEWS=$(python3 "$MOD_DIR/build_views.py" "$SNAP" "$OUT") || die "could not build the model folders"
PATCH=$(python3 "$MOD_DIR/patch_b12x.py" "$MOD_DIR") || die "could not patch vLLM: $PATCH"

echo "$(hostname): Qwen3.8-Flash-Next-Swift1.5 ${REV} | $VIEWS"
echo "$(hostname): $PATCH"
