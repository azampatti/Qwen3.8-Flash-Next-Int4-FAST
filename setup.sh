#!/usr/bin/env bash
# One-time setup: build the serving image and download the weights. Re-runnable; it skips whatever is already done.
#   ./setup.sh
# Everything happens inside Docker -- the only things you need on the host are docker and the NVIDIA container toolkit.
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env

step(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

step "Checking prerequisites"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running? are you in the docker group?)"
docker run --rm --gpus all --entrypoint true "${CUDA_PROBE_IMAGE:-nvidia/cuda:13.0.0-base-ubuntu24.04}" >/dev/null 2>&1 \
  || die "docker cannot see the GPU -- install the NVIDIA container toolkit"
free_gb=$(df -PBG "$MODELS_DIR" 2>/dev/null | awk 'NR==2{gsub("G","",$4);print $4}' || echo 0)
[ "${free_gb:-0}" -ge 130 ] || echo "  WARNING: only ${free_gb}G free in $MODELS_DIR; this needs ~130G (72G weights + 49G n-gram table)"
echo "  ok: docker, GPU, $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
mkdir -p "$MODELS_DIR"

step "Building the upstream image (Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)"
if docker image inspect "$UPSTREAM_IMAGE" >/dev/null 2>&1; then
  echo "  already built: $UPSTREAM_IMAGE"
else
  [ -d upstream ] || git clone --depth 1 "$UPSTREAM_REPO" upstream
  git -C upstream fetch --depth 1 origin "$UPSTREAM_COMMIT" 2>/dev/null || true
  git -C upstream checkout -q "$UPSTREAM_COMMIT" 2>/dev/null || echo "  note: could not pin $UPSTREAM_COMMIT, building the default branch"
  echo "  building $UPSTREAM_IMAGE (20-40 min, this compiles kernels)"
  docker build -t "$UPSTREAM_IMAGE" upstream
fi

step "Applying the two optional patches (fp8 KV cache, deterministic top-k)"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  already built: $IMAGE"
else
  docker build --build-arg BASE="$UPSTREAM_IMAGE" -t "$IMAGE" .
fi

dl(){   # dl <hf repo> : download with the official Hugging Face tool, into the standard cache unless DOWNLOAD_MODE=local
  local repo="$1"
  if [ "$DOWNLOAD_MODE" = cache ]; then
    if [ -n "$(hf_snapshot_dir "$repo" 2>/dev/null || true)" ]; then
      echo "  already in the Hugging Face cache: $(hf_snapshot_dir "$repo")"; return 0
    fi
    mkdir -p "$HF_HOME"
    if command -v hf >/dev/null 2>&1; then          # host tool: files stay owned by you
      echo "  downloading $repo into $HF_HOME (host 'hf')"
      HF_HOME="$HF_HOME" hf download "$repo"
    else                                            # no host tool: use the image, then hand the files back
      echo "  downloading $repo into $HF_HOME (via the container; no 'hf' on this host)"
      docker run --rm -v "$HF_HOME":"$HF_HOME" -e HF_HOME="$HF_HOME" ${HF_TOKEN:+-e HF_TOKEN} \
        --entrypoint hf "$IMAGE" download "$repo"
      docker run --rm -v "$HF_HOME":"$HF_HOME" --entrypoint sh "$IMAGE" \
        -c "chown -R $(id -u):$(id -g) '$(hf_repo_root "$repo")'"
    fi
    [ -n "$(hf_snapshot_dir "$repo" 2>/dev/null || true)" ] || { echo "ERROR: $repo did not land in the cache" >&2; exit 1; }
  else
    local target="$MODELS_DIR/$(basename "$repo")"
    if [ -f "$target/.download-complete" ]; then echo "  already downloaded: $target"; return 0; fi
    echo "  downloading $repo -> $target"
    mkdir -p "$MODELS_DIR"
    docker run --rm -v "$MODELS_DIR":/models ${HF_TOKEN:+-e HF_TOKEN} --entrypoint hf "$IMAGE" \
      download "$repo" --local-dir "/models/$(basename "$repo")"
    docker run --rm -v "$MODELS_DIR":/models --entrypoint sh "$IMAGE" \
      -c "chown -R $(id -u):$(id -g) /models/$(basename "$repo") && touch /models/$(basename "$repo")/.download-complete"
  fi
}
step "Downloading the model (~120 GB, the n-gram table is inside it)"
dl "$MODEL_REPO"
source ./config.env                      # re-resolve MODEL_DIR/TABLE_DIR now that the files exist
echo "  model: $MODEL_DIR"
if [ -n "$TABLE_DIR" ] && [ -n "$(ls "$TABLE_DIR" 2>/dev/null)" ]; then
  echo "  n-gram table: $TABLE_DIR"
else
  step "Downloading the n-gram table separately (~49 GB)"; dl "$TABLE_REPO"; source ./config.env; echo "  n-gram table: $TABLE_DIR"
fi

step "Creating the speculative-decoding draft directory (symlinks, no extra disk)"
if [ -f "$DRAFT_DIR/config.json" ] && [ -e "$DRAFT_DIR/model.safetensors.index.json" ]; then echo "  already there: $DRAFT_DIR"; else python3 tools/make_draft_dir.py "$MODEL_DIR" "$DRAFT_DIR"; fi

printf '\n\033[1mSetup complete.\033[0m  Start the server with:  ./serve.sh\n\n'
