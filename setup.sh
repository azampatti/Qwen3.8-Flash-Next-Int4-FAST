#!/usr/bin/env bash
# One-time setup: get the serving image and download the weights. Re-runnable; it skips whatever is already done.
#   ./setup.sh
#   BACKEND=classic ./setup.sh     build our own image even when Eugr's spark-vllm-docker is there (BACKEND=b12x: insist on it)
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

# Eugr's spark-vllm-docker checked out (EUGR_DIR, default ~/spark-vllm-docker) = serve with his b12x stack: pull the pinned b12x image instead
# of building ours. Anything short of a successful pull (repo missing, pull failed) falls back to building our image.
step "Looking for Eugr's spark-vllm-docker (b12x stack)"
USE_B12X=0
case "$BACKEND" in
  classic) echo "  BACKEND=classic: building our own image" ;;
  auto|b12x)
    if ! eugr_present; then
      [ "$BACKEND" = b12x ] && die "BACKEND=b12x, but Eugr's repo is not at $EUGR_DIR (git clone https://github.com/eugr/spark-vllm-docker $EUGR_DIR)"
      echo "  not found at $EUGR_DIR: building our own image"
    elif [ "$DOWNLOAD_MODE" != cache ]; then
      [ "$BACKEND" = b12x ] && die "the b12x recipe reads the model from the Hugging Face cache; it needs DOWNLOAD_MODE=cache"
      echo "  found at $EUGR_DIR, but the b12x recipe needs DOWNLOAD_MODE=cache (this is '$DOWNLOAD_MODE'): building our own image"
    elif { echo "  found at $EUGR_DIR"; ./eugr-setup.sh --pull-only 2>&1 | sed 's/^/  /'; }; then
      USE_B12X=1; IMAGE="$B12X_IMAGE"
      echo "  serving with $IMAGE: skipping our own image build"
    else
      [ "$BACKEND" = b12x ] && die "could not pull the pinned b12x image $B12X_IMAGE_PIN"
      echo "  WARNING: could not pull the pinned b12x image -- building our own image instead"
    fi ;;
  *) die "BACKEND must be auto, b12x or classic (got '$BACKEND')" ;;
esac

if [ "$USE_B12X" = 1 ]; then
  # The mod (flashnext-int4-b12x) patches vLLM for this checkpoint at every launch and checks the model there, so all
  # this image needs up front is the b12x stack itself -- and the 'hf' tool, which the download below may borrow.
  step "Checking the b12x image"
  docker run --rm --entrypoint sh "$IMAGE" -c \
    'python3 -c "import b12x, vllm, huggingface_hub; print(\"  ok: b12x + vLLM\", vllm.__version__)" && command -v hf >/dev/null' \
    || die "$IMAGE is not a usable b12x image. Delete it and re-run setup: docker rmi $IMAGE"
else
  step "Building the upstream image (Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)"
  if docker image inspect "$UPSTREAM_IMAGE" >/dev/null 2>&1; then
    echo "  already built: $UPSTREAM_IMAGE"
    if [ -d upstream/.git ] && [ "$(git -C upstream rev-parse HEAD 2>/dev/null)" != "$UPSTREAM_COMMIT" ]; then
      echo "  WARNING: that image was built from upstream $(git -C upstream rev-parse --short HEAD 2>/dev/null), not the pinned ${UPSTREAM_COMMIT:0:7}"
      echo "           (setup.sh before 2026-09-21 could not apply the pin). To rebuild from the pin:"
      echo "           docker rmi $IMAGE $UPSTREAM_IMAGE && rm -rf upstream && ./setup.sh"
    fi
  else
    [ -d upstream ] || git clone --depth 1 "$UPSTREAM_REPO" upstream
    # The pin is a HARD requirement: a short SHA cannot be fetched from a shallow clone, and the old fallback ("building the
    # default branch") silently produced an image from whatever upstream had that day. UPSTREAM_COMMIT must be the full 40-hex SHA.
    [[ "$UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
      || { echo "  ERROR: UPSTREAM_COMMIT must be a full 40-character commit SHA (got '$UPSTREAM_COMMIT')"; exit 1; }
    git -C upstream fetch --depth 1 origin "$UPSTREAM_COMMIT" \
      || { echo "  ERROR: could not fetch the pinned upstream commit $UPSTREAM_COMMIT -- refusing to build an unpinned image"; exit 1; }
    git -C upstream checkout -q "$UPSTREAM_COMMIT" \
      || { echo "  ERROR: could not check out the pinned upstream commit $UPSTREAM_COMMIT"; exit 1; }
    [ "$(git -C upstream rev-parse HEAD)" = "$UPSTREAM_COMMIT" ] \
      || { echo "  ERROR: upstream/ is at $(git -C upstream rev-parse HEAD), not the pinned $UPSTREAM_COMMIT"; exit 1; }
    echo "  pinned: upstream @ $UPSTREAM_COMMIT"
    echo "  building $UPSTREAM_IMAGE (20-40 min, this compiles kernels)"
    docker build -t "$UPSTREAM_IMAGE" upstream
  fi

  step "Applying the two optional patches (fp8 KV cache, deterministic top-k)"
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "  already built: $IMAGE"
  else
    docker build --build-arg BASE="$UPSTREAM_IMAGE" -t "$IMAGE" .
  fi

  # Everything serve.sh relies on, checked against the image that is actually going to run -- an image that was built
  # earlier is not evidence that it still carries the pieces this model needs. Hard failures stop setup; the draft-logit
  # patch only warns, because serve.sh falls back to unscaled drafting.
  step "Checking the image has what this model needs"
  docker run --rm --entrypoint sh "$IMAGE" -c '
    fail=0
    f=$(find /usr/local/lib -path "*vllm/config/speculative.py" 2>/dev/null | head -1)
    if [ -n "$f" ] && grep -q "\"block\"" "$f" && grep -q "\"probabilistic\"" "$f"; then
      echo "  ok: block verification + probabilistic drafting"
    else
      echo "  FAIL: no block/probabilistic speculative sampling (set REJECTION_SAMPLE=standard DRAFT_SAMPLE=greedy)"; fail=1
    fi
    m=$(find /usr/local/lib -path "*qwen3_8_flash_next/nvidia/mtp.py" 2>/dev/null | head -1)
    if [ -n "$m" ]; then
      echo "  ok: speculative-decoding head support"
      if grep -q "LogitsProcessor(config.vocab_size)$" "$m"; then
        echo "  ok: draft-logit scaling can be applied (DRAFT_SCALE)"
      else
        echo "  WARNING: this image builds the draft logits differently -- DRAFT_SCALE will be ignored"
      fi
    else
      echo "  FAIL: this image cannot serve the speculative head (set MTP=0)"; fail=1
    fi
    grep -rqs "FP8_HYBRID" /usr/local/lib/python3.12/dist-packages/vllm/ \
      && echo "  ok: fp8 hybrid loader (this checkpoint needs it)" \
      || { echo "  FAIL: no fp8 hybrid loader -- this image cannot serve this checkpoint"; fail=1; }
    grep -rqs "PLE_MMAP" /usr/local/lib/python3.12/dist-packages/vllm/ \
      && echo "  ok: memory-mapped n-gram table" \
      || { echo "  FAIL: no n-gram table support -- this image cannot serve this checkpoint"; fail=1; }
    exit $fail
  ' || die "the image is missing something this model needs (see above). Delete it and re-run setup to rebuild: docker rmi $IMAGE"
fi

dl(){   # dl <hf repo> : download with the official Hugging Face tool, into the standard cache unless DOWNLOAD_MODE=local
  local repo="$1"
  if [ "$DOWNLOAD_MODE" = cache ]; then
    local have; have="$(hf_snapshot_dir "$repo" 2>/dev/null || true)"
    if [ -n "$have" ]; then
      if [ "${OFFLINE:-0}" = 1 ]; then echo "  already in the cache, not contacting the hub (OFFLINE=1): $have"; return 0; fi
      echo "  already in the cache: $have"
      echo "  asking the hub for anything newer (unchanged files are reused, so this is usually seconds)"
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
    if [ -f "$target/.download-complete" ] && [ "${OFFLINE:-0}" = 1 ]; then echo "  already downloaded, not contacting the hub (OFFLINE=1): $target"; return 0; fi
    [ -f "$target/.download-complete" ] && echo "  already downloaded: $target -- asking the hub for anything newer"
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

# A download reports success when the revision looks complete in ITS bookkeeping. That is not the same as the files
# being the ones the hub serves now, so compare each large file's cached hash against the hub's before trusting it.
verify_cache(){ docker run --rm -v "$PWD":/repo:ro -v "$HF_HOME":"$HF_HOME" -e HF_HOME="$HF_HOME" ${HF_TOKEN:+-e HF_TOKEN} \
  --entrypoint python3 "$IMAGE" /repo/tools/verify_cache.py --repo "$MODEL_REPO" --hf-home "$HF_HOME" 2>/dev/null | grep -v "^Hint"; }
if [ "$DOWNLOAD_MODE" = cache ] && [ "${OFFLINE:-0}" != 1 ]; then
  step "Checking the cached files are the ones the hub has"
  out="$(verify_cache)"; printf '%s\n' "$out" | sed 's/^/  /'
  if printf '%s' "$out" | grep -qE '^(MISSING|STALE)'; then
    echo "  repairing: dropping the outdated entries so they download again"
    while read -r kind name rest; do
      case "$kind" in MISSING|STALE) rm -f "$MODEL_DIR/$name" ;; esac
    done <<< "$out"
    dl "$MODEL_REPO"; source ./config.env
    out="$(verify_cache)"; printf '%s\n' "$out" | sed 's/^/  /'
    printf '%s' "$out" | grep -qE '^(MISSING|STALE)' && die "the cache still does not match the hub (see above)"
  fi
fi
if [ -n "$TABLE_DIR" ] && [ -n "$(ls "$TABLE_DIR" 2>/dev/null)" ]; then
  echo "  n-gram table: $TABLE_DIR"
else
  step "Downloading the n-gram table separately (~49 GB)"; dl "$TABLE_REPO"; source ./config.env; echo "  n-gram table: $TABLE_DIR"
fi

if [ "$USE_B12X" = 1 ]; then
  # The mod builds the draft folder and applies the draft-logit scaling inside the container, so neither step is needed here.
  step "Installing the b12x recipes + mod into $EUGR_DIR"
  ./eugr-setup.sh | sed 's/^/  /'
  printf 'SETUP_BACKEND=b12x\nSETUP_EUGR_DIR=%q\n' "$EUGR_DIR" > "$BACKEND_FILE"
  printf '\n\033[1mSetup complete (b12x).\033[0m  Start the server with:  ./serve.sh   (it runs the single-node recipe above)\n\n'
else
  step "Creating the speculative-decoding draft directory (symlinks, no extra disk)"
  # The draft folder is symlinks INTO the model folder, and a hub update lands in a new snapshot folder, so links
  # made against the previous one must be rebuilt. Comparing timestamps cannot see this: every file in a snapshot is
  # itself a symlink to a content-addressed blob whose timestamp is the day it was first downloaded. So compare the
  # path the links actually name against the model folder in use.
  draft_fresh(){
    [ -f "$DRAFT_DIR/config.json" ] || return 1
    [ -e "$DRAFT_DIR/model.safetensors.index.json" ] && [ -e "$DRAFT_DIR/model_extra_tensors.safetensors" ] || return 1
    case "$(readlink "$DRAFT_DIR/model.safetensors.index.json" 2>/dev/null)" in
      *"$(basename "$MODEL_DIR")"/*) ;;
      *) return 1 ;;
    esac
    # it exists and points at the right copy of the model; it also has to actually say 10 experts, or it does nothing
    [ "$(python3 -c "import json;c=json.load(open('$DRAFT_DIR/config.json'));t=c.get('text_config',c);print(t['num_experts_per_tok'])" 2>/dev/null)" = 10 ]
  }
  if draft_fresh; then echo "  already there: $DRAFT_DIR"; else python3 tools/make_draft_dir.py "$MODEL_DIR" "$DRAFT_DIR"; fi

  step "Preparing the draft-logit scaling module (DRAFT_SCALE in serve.sh)"
  if [ "${DRAFT_SCALE:-2}" = 1 ]; then
    echo "  DRAFT_SCALE=1, nothing to build"
  elif python3 tools/draft_scale.py --image "$IMAGE" --out "$MODELS_DIR/.draft-scale/mtp.py"; then
    grep -q "VLLM_MTP_DRAFT_SCALE" "$MODELS_DIR/.draft-scale/mtp.py" \
      || die "the draft-scale module was written without the scaling hook -- please report this"
  else
    echo "  note: this image does not match the patcher; serve.sh will draft without scaling"
  fi
  echo "SETUP_BACKEND=classic" > "$BACKEND_FILE"
  printf '\n\033[1mSetup complete.\033[0m  Start the server with:  ./serve.sh\n\n'
fi
