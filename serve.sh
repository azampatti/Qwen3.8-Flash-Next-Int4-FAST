#!/usr/bin/env bash
# Start the server.   ./serve.sh            foreground, Ctrl-C stops it
#                     ./serve.sh -d         background
# Settings live in config.env; you can also override any of them inline, e.g.  KV_BYTES=30g ./serve.sh
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
DETACH=0; [ "${1:-}" = "-d" ] && DETACH=1

[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] || { echo "Model not found (DOWNLOAD_MODE=$DOWNLOAD_MODE) -- run ./setup.sh first" >&2; exit 1; }
[ -n "$TABLE_DIR" ] && [ -n "$(ls "$TABLE_DIR" 2>/dev/null)" ] || { echo "n-gram table not found -- run ./setup.sh first" >&2; exit 1; }
# Mount every directory at the SAME absolute path it has on the host. The Hugging Face cache stores a snapshot as
# symlinks into ../../blobs, so identical paths inside and outside the container are what keeps them resolving.
MOUNTS=(-v "$MODELS_DIR":"$MODELS_DIR":ro)
[ "$DOWNLOAD_MODE" = cache ] && MOUNTS+=(-v "$HF_HOME/hub":"$HF_HOME/hub":ro)
TOPK=$(python3 -c "import json;c=json.load(open('$MODEL_DIR/config.json'));print(c.get('text_config',c)['num_experts_per_tok'])")

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
TPL=(); TPL_MOUNT=()
if [ -n "$CHAT_TEMPLATE" ]; then TPL_MOUNT=(-v "$CHAT_TEMPLATE:/chat_template.jinja:ro"); TPL=(--chat-template /chat_template.jinja); fi
RUN=(--rm); [ "$DETACH" = 1 ] && RUN=(-d --rm) || { [ -t 0 ] && RUN=(--rm -it); }

# PLE table gathers are a CPU op + a host-to-device copy, so they must run outside CUDA graphs.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'

echo "Serving $MODEL_REPO -- top-k $TOPK, ${CTX} ctx, MTP ${MTP}${DRAFT_NOTE}, KV $KV_BYTES $KV_DTYPE$([ "$DET_TOPK" = 1 ] && echo ', deterministic top-k') on port $PORT"
docker rm -f qwen38-flash >/dev/null 2>&1 || true
exec docker run "${RUN[@]}" --name qwen38-flash \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  "${MOUNTS[@]}" "${TPL_MOUNT[@]}" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=1 -e VLLM_PLE_MMAP_DIR="$TABLE_DIR" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_FLASHINFER_SAMPLER=1 "${DET[@]}" \
  "$IMAGE" "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8000 --load-format fastsafetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization 0.01 --kv-cache-memory-bytes "$KV_BYTES" \
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops="$SPLIT" --no-enable-flashinfer-autotune \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3 \
    "${TPL[@]}" "${SPEC[@]}"
