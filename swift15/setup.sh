#!/usr/bin/env bash
# Qwen3.8-Flash-Next-Swift1.5 (azampatti) -- one-time setup on top of the base model.
#
# PREREQUISITE: the base model azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound in the Hugging Face cache, i.e.
#   git clone https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST ~/Qwen3.8-Flash-Next-Int4-FAST && ~/Qwen3.8-Flash-Next-Int4-FAST/setup.sh
# and eugr's spark-vllm-docker with the vllm-node-b12x image (the base setup pulls the pinned image when eugr's repo is there).
#
# What it does (idempotent, ~3 GB of new data, no GPU, no running containers touched):
#   1. gets the three Swift1.5 files: model/ next to this script if present, else downloads them from the Hugging Face repo
#      $OVERLAY_REPO into model/ (host 'hf' CLI if installed, else the one inside the b12x image); checks sha256 against MANIFEST.json
#   2. finds the base model's pinned snapshot in the HF cache and checks every file the index needs is there
#   3. builds the served model as an HF-cache repo, models--local--Qwen3.8-Flash-Next-Swift1.5: the three Swift1.5 files as blobs,
#      everything else a relative symlink into the base snapshot (nothing of the base is copied)
#   4. installs the mod + the solo and cluster recipes into eugr's spark-vllm-docker
#
#   ./setup.sh                 HF_HOME=... / EUGR_DIR=... override the defaults (~/.cache/huggingface, ~/spark-vllm-docker)
#   ./setup.sh --force         rebuild the served model even if it is already in place
# Two nodes: run this on BOTH (the cluster recipe checks the variant on every node).
set -euo pipefail
cd "$(dirname "$0")"; HERE="$(pwd)"
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"; HUB="$HF_HOME/hub"; EUGR_DIR="${EUGR_DIR:-$HOME/spark-vllm-docker}"
B12X_IMAGE="${B12X_IMAGE:-vllm-node-b12x}"; OVERLAY_REPO="${OVERLAY_REPO:-azampatti/Qwen3.8-Flash-Next-Swift1.5}"
read -r VERSION BASE_REPO BASE_REV < <(python3 -c "import json;m=json.load(open('MANIFEST.json'));print(m['version'],m['base_model']['repo'],m['base_model']['revision'])")
BASE_DIR="$HUB/models--${BASE_REPO//\//--}"; BASE_SNAP="$BASE_DIR/snapshots/$BASE_REV"
NAME=models--local--Qwen3.8-Flash-Next-Swift1.5; OUT="$HUB/$NAME"; SNAP="$OUT/snapshots/$VERSION"
MOD=flashnext-swift15-b12x; RECIPES=(qwen3.8-flash-next-swift15-b12x-solo.yaml qwen3.8-flash-next-swift15-b12x.yaml)
say(){ echo "[swift15-setup] $*"; }; die(){ echo "[swift15-setup] ERROR: $*" >&2; exit 1; }
OVERLAY=(model-swift15-attn.safetensors model-healed-shared-expert.safetensors model.safetensors.index.json)

# 1. the Swift1.5 files
missing=(); for f in "${OVERLAY[@]}"; do [ -s "model/$f" ] || missing+=("$f"); done
if [ ${#missing[@]} -gt 0 ]; then
  say "downloading ${missing[*]} from $OVERLAY_REPO (~3 GB) ..."
  mkdir -p model
  if command -v hf >/dev/null; then hf download "$OVERLAY_REPO" "${missing[@]}" --local-dir model
  elif command -v docker >/dev/null && docker image inspect "$B12X_IMAGE" >/dev/null 2>&1; then
    docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf -v "$HERE/model:/out" --entrypoint hf "$B12X_IMAGE" download "$OVERLAY_REPO" "${missing[@]}" --local-dir /out
  else die "no 'hf' CLI and no $B12X_IMAGE image to borrow it from -- pip install -U huggingface_hub, or put the files in $HERE/model/"; fi
  rm -rf model/.cache
fi
say "checking the Swift1.5 files (sha256, ~3 GB) ..."
python3 - <<'PY' || die "a Swift1.5 file does not match MANIFEST.json (incomplete download?)"
import hashlib, json, sys
m = json.load(open("MANIFEST.json"))["files"]
for name, meta in m.items():
    h = hashlib.sha256()
    with open(f"model/{name}", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    if h.hexdigest() != meta["sha256"]: print(f"  {name}: sha256 mismatch", file=sys.stderr); sys.exit(1)
    print(f"  {name}: ok")
PY

# 2. the base model
if [ ! -d "$BASE_SNAP" ]; then
  have=$(cat "$BASE_DIR/refs/main" 2>/dev/null || true)
  [ -n "$have" ] || die "the base model is not in $HUB. Install it first:
    git clone https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST ~/Qwen3.8-Flash-Next-Int4-FAST && ~/Qwen3.8-Flash-Next-Int4-FAST/setup.sh"
  die "the base model in your cache is revision ${have:0:8}, this variant was built on ${BASE_REV:0:8}. Fetch that revision too:
    hf download $BASE_REPO --revision $BASE_REV      (or: HF_HOME=$HF_HOME ~/Qwen3.8-Flash-Next-Int4-FAST/setup.sh, if it pins the same revision)"
fi
python3 - "$BASE_SNAP" <<'PY' || die "the base snapshot ${BASE_REV:0:8} is incomplete -- re-run the base model's setup.sh"
import json, os, sys
snap = sys.argv[1]; overlay = {"model-swift15-attn.safetensors", "model-healed-shared-expert.safetensors", "model.safetensors.index.json"}
need = sorted(set(json.load(open("model/model.safetensors.index.json"))["weight_map"].values()) - overlay)
need += ["config.json", "chat_template.jinja", "medium_chat_template.jinja", "model_extra_tensors.safetensors", "model-lmhead-int4.safetensors", "ple-table"]
missing = [f for f in need if not os.path.exists(os.path.join(snap, f))]
if missing: print(f"  missing in the base snapshot: {missing[:5]}{' ...' if len(missing) > 5 else ''}", file=sys.stderr); sys.exit(1)
print(f"  base snapshot ok: {len(need)} files the variant uses are present")
PY

# 3. the served model (HF-cache layout: blobs named by sha256, snapshot = symlinks)
if [ "$FORCE" = 1 ] || [ "$(cat "$OUT/refs/main" 2>/dev/null)" != "$VERSION" ] || [ ! -e "$SNAP/model-swift15-attn.safetensors" ]; then
  say "building $OUT (version $VERSION on base ${BASE_REV:0:8}) ..."
  rm -rf "$SNAP"; mkdir -p "$OUT/blobs" "$OUT/refs" "$SNAP"
  for f in "${OVERLAY[@]}"; do
    h=$(python3 -c "import json;print(json.load(open('MANIFEST.json'))['files']['$f']['sha256'])")
    [ -s "$OUT/blobs/$h" ] || ln "model/$f" "$OUT/blobs/$h" 2>/dev/null || { cp "model/$f" "$OUT/blobs/$h.part" && mv "$OUT/blobs/$h.part" "$OUT/blobs/$h"; }   # hard link when on the same disk (no second copy)
    ln -sfn "../../blobs/$h" "$SNAP/$f"
  done
  for p in "$BASE_SNAP"/*; do
    n=$(basename "$p"); [ -e "$SNAP/$n" ] && continue
    ln -sfn "../../../models--${BASE_REPO//\//--}/snapshots/$BASE_REV/$n" "$SNAP/$n"
  done
  echo -n "$VERSION" > "$OUT/refs/main"
else
  say "$OUT is already at $VERSION (--force rebuilds it)"
fi
python3 - "$SNAP" <<'PY' || die "the served model does not resolve (see above) -- re-run with --force"
import json, os, sys
snap = sys.argv[1]
files = set(json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"].values())
bad = [f for f in sorted(files) + ["config.json", "chat_template.jinja", "model_extra_tensors.safetensors", "ple-table"] if not os.path.exists(os.path.join(snap, f))]
if bad: print(f"  unresolved: {bad[:5]}", file=sys.stderr); sys.exit(1)
print(f"  served model ok: {len(files)} weight files + extras resolve ({snap})")
PY

# 4. eugr's spark-vllm-docker
if [ -d "$EUGR_DIR/recipes" ] && [ -d "$EUGR_DIR/mods" ]; then
  rm -rf "$EUGR_DIR/mods/$MOD"; cp -r "eugr/$MOD" "$EUGR_DIR/mods/$MOD"
  rm -rf "$EUGR_DIR/mods/fst-ep-local"; cp -r "../fst-ep-local" "$EUGR_DIR/mods/fst-ep-local"   # shared with the base recipes (cluster loader)
  for r in "${RECIPES[@]}"; do cp "eugr/$r" "$EUGR_DIR/recipes/"; done
  say "installed mods/$MOD + mods/fst-ep-local + ${RECIPES[*]} into $EUGR_DIR"
else
  say "eugr's spark-vllm-docker not found at $EUGR_DIR -- copy eugr/$MOD into its mods/ and eugr/*.yaml into its recipes/ yourself"
fi
command -v docker >/dev/null && ! docker image inspect "$B12X_IMAGE" >/dev/null 2>&1 && \
  say "WARNING: no '$B12X_IMAGE' image on this host -- run the base model's setup.sh (or eugr-setup.sh --pull-only) to get the pinned b12x image"

# colours only on a terminal (NO_COLOR=1 turns them off): bright white = titles, pink = commands to run
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then W=$'\e[1;97m'; K=$'\e[38;5;213m'; R=$'\e[0m'; else W=; K=; R=; fi
cat <<EOF

${W}Qwen3.8-Flash-Next-Swift1.5 is ready. Serve it from $EUGR_DIR:${R}
  ${K}./run-recipe.sh recipes/qwen3.8-flash-next-swift15-b12x-solo.yaml --solo${R}   # one DGX Spark
  ${K}./run-recipe.sh recipes/qwen3.8-flash-next-swift15-b12x.yaml${R}               # two nodes (run this setup on BOTH first; launch on the head)
${W}API:${R} http://<host>:8000/v1, model name azampatti/Qwen3.8-Flash-Next-Swift1.5. Reasoning effort is xhigh unless a request sets
"reasoning_effort": "medium" (or "low"). ${W}Stop:${R} ${K}docker rm -f vllm_node${R} (or the --name you gave run-recipe).
EOF
