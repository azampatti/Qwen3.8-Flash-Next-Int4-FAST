#!/usr/bin/env bash
# Start the server.    ./serve.sh        run in the foreground, Ctrl-C stops it
#                      ./serve.sh -d     run in the background
#
# EVERYTHING you might want to change is in the block below: one setting per line.
# Edit a line, save, re-run. Or override any of them for a single run without editing:
#     CTX=32768 CHAT_TEMPLATE=medium ./serve.sh

# ═══════════════════════════════ SETTINGS ═══════════════════════════════
# >>> SETTINGS >>>
MODEL_REPO="${MODEL_REPO:-azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound}"  # which model; setup.sh downloads this one
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next-a5b}"  # the name clients send as "model"
PORT="${PORT:-8000}"                                  # host port for the OpenAI-compatible API

CTX="${CTX:-262144}"                                  # max tokens in ONE request (prompt + output)
SEQS="${SEQS:-8}"                                     # how many requests may run at the same time
KV_BYTES="${KV_BYTES:-20g}"                           # KV cache size; 20g holds ~645k tokens total
KV_DTYPE="${KV_DTYPE:-auto}"                          # auto | fp8_e4m3  (fp8 = ~1.9x context, ~10% slower)

CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"                    # "" = the model's default | medium | xhigh | /path/to/file.jinja
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"             # how tool calls are parsed out of the reply
REASONING_PARSER="${REASONING_PARSER:-qwen3}"         # puts <think> into its own "reasoning" field

MTP="${MTP:-3}"                                       # speculative decoding depth; 0 = off. 3 is the measured optimum
DRAFT_K10="${DRAFT_K10:-1}"                           # 1 = draft over 10 experts (faster); 0 = draft like the model
DET_TOPK="${DET_TOPK:-0}"                             # 1 = deterministic expert top-k (slower, still not bit-exact)

IMAGE="${IMAGE:-qwen38-flash-dgx:a5b-int4}"           # the image setup.sh built
CONTAINER="${CONTAINER:-qwen38-flash}"                # docker container name
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.01}"                  # left tiny on purpose: KV_BYTES above sets the cache instead
BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"              # prefill chunk size
SHM="${SHM:-16g}"                                     # container shared memory

DOWNLOAD_MODE="${DOWNLOAD_MODE:-cache}"               # cache = the standard HF cache | local = a plain folder
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"        # where the HF cache lives (cache mode)
MODELS_DIR="${MODELS_DIR:-$HOME/models}"              # plain-folder location (local mode) + the small draft folder

EXTRA_ARGS="${EXTRA_ARGS:-}"                          # anything else to append to the vLLM command line
# <<< SETTINGS <<<
# ════════════════════════════════════════════════════════════════════════

# setup.sh and check.sh read the settings above by sourcing this file with SETTINGS_ONLY=1.
[ "${SETTINGS_ONLY:-0}" = 1 ] && return 0

set -euo pipefail
cd "$(dirname "$0")"; source ./config.env      # turns the settings above into concrete paths
DETACH=0; [ "${1:-}" = "-d" ] && DETACH=1

[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] || { echo "Model not found (DOWNLOAD_MODE=$DOWNLOAD_MODE) -- run ./setup.sh first" >&2; exit 1; }
[ -n "$TABLE_DIR" ] && [ -n "$(ls "$TABLE_DIR" 2>/dev/null)" ] || { echo "n-gram table not found -- run ./setup.sh first" >&2; exit 1; }

# Mount every directory at the SAME absolute path it has on the host. The Hugging Face cache stores a snapshot as
# symlinks into ../../blobs, so identical paths inside and outside the container are what keeps them resolving.
MOUNTS=(-v "$MODELS_DIR":"$MODELS_DIR":ro)
[ "$DOWNLOAD_MODE" = cache ] && MOUNTS+=(-v "$HF_HOME/hub":"$HF_HOME/hub":ro)
TOPK=$(python3 -c "import json;c=json.load(open('$MODEL_DIR/config.json'));print(c.get('text_config',c)['num_experts_per_tok'])")

# CHAT_TEMPLATE accepts a bare name (medium, xhigh) for a template shipped inside the model, or a full path.
TPL=(); TPL_MOUNT=(); TPL_NOTE=""
if [ -n "$CHAT_TEMPLATE" ]; then
  TPL_FILE="$CHAT_TEMPLATE"
  if [ "${CHAT_TEMPLATE#*/}" = "$CHAT_TEMPLATE" ] && [ ! -f "$CHAT_TEMPLATE" ]; then
    for cand in "$MODEL_DIR/${CHAT_TEMPLATE}_chat_template.jinja" "$MODEL_DIR/$CHAT_TEMPLATE"; do
      [ -f "$cand" ] && { TPL_FILE="$cand"; break; }
    done
  fi
  [ -f "$TPL_FILE" ] || { echo "CHAT_TEMPLATE '$CHAT_TEMPLATE' not found. Available inside the model:" >&2
                          ls "$MODEL_DIR"/*chat_template*.jinja 2>/dev/null | sed 's|.*/|  |' >&2; exit 1; }
  TPL_MOUNT=(-v "$TPL_FILE:/chat_template.jinja:ro"); TPL=(--chat-template /chat_template.jinja)
  TPL_NOTE=", template $(basename "$TPL_FILE")"
fi

SPEC=(); DRAFT_NOTE=""
if [ "$MTP" != 0 ]; then
  DRAFT="$MODEL_DIR"; DRAFT_NOTE=" (draft: same as the model)"
  if [ "$DRAFT_K10" = 1 ]; then
    if [ -f "$DRAFT_DIR/config.json" ]; then DRAFT="$DRAFT_DIR"; DRAFT_NOTE=" (draft k=10)"
    else echo "  note: DRAFT_K10=1 but $DRAFT_DIR does not exist -- drafting like the model. Create it with: python3 tools/make_draft_dir.py \"$MODEL_DIR\""; fi
  fi
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"model\":\"$DRAFT\"}")
fi
DET=(); [ "$DET_TOPK" = 1 ] && DET=(-e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so)
RUN=(--rm); [ "$DETACH" = 1 ] && RUN=(-d --rm) || { [ -t 0 ] && RUN=(--rm -it); }

# PLE table gathers are a CPU op + a host-to-device copy, so they must run outside CUDA graphs.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'

echo "Serving $MODEL_REPO -- top-k $TOPK, ${CTX} ctx, MTP ${MTP}${DRAFT_NOTE}, KV $KV_BYTES $KV_DTYPE$TPL_NOTE$([ "$DET_TOPK" = 1 ] && echo ', deterministic top-k') on port $PORT"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
exec docker run "${RUN[@]}" --name "$CONTAINER" \
  --gpus all --ipc=host --shm-size "$SHM" -p "${PORT}:8000" \
  "${MOUNTS[@]}" "${TPL_MOUNT[@]}" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=1 -e VLLM_PLE_MMAP_DIR="$TABLE_DIR" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_FLASHINFER_SAMPLER=1 "${DET[@]}" \
  "$IMAGE" "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8000 --load-format fastsafetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-memory-bytes "$KV_BYTES" \
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens "$BATCHED_TOKENS" \
    -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="$SPLIT" --no-enable-flashinfer-autotune \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER" \
    "${TPL[@]}" "${SPEC[@]}" $EXTRA_ARGS
