#!/usr/bin/env bash
# Is it up, and is it the model you think?   ./check.sh
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "Not answering on port $PORT (still loading? 'docker logs -f qwen38-flash')"; exit 1; }
curl -s "http://localhost:$PORT/v1/models" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'][0];print('served:',d['id'],'\nfrom:  ',d.get('root'),'\nmax context:',d.get('max_model_len'),'tokens')"
python3 -c "import json;c=json.load(open('$MODEL_DIR/config.json'));t=c.get('text_config',c);print('experts per token:',t['num_experts_per_tok'],'of',t['num_experts'])"
echo "local path:  $MODEL_DIR"
head_link="$(readlink "$MODEL_DIR/model_extra_tensors.safetensors" 2>/dev/null || true)"   # cache mode: the blob name IS the file's sha256
[ -n "$head_link" ] && echo "MTP head:    ${head_link##*/}" | cut -c1-30
if [ "$MTP" != 0 ] && [ "$DRAFT_SCALE" != 1 ] && [ -f "$MODELS_DIR/.draft-scale/mtp.py" ]; then
  echo "draft:       depth $MTP, logits x$DRAFT_SCALE"
else
  echo "draft:       depth $MTP, no scaling"
fi
echo; echo "test prompt:"
# Ask the server what it calls itself, so this works even when SERVED_NAME was overridden for the run.
NAME="$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")"
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word READY.\"}],\"max_tokens\":30,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c "import json,sys
d=json.load(sys.stdin)
if 'choices' in d: m=d['choices'][0]['message']; print(' ', m.get('content') or m.get('reasoning') or '(empty)')
else: print('  no answer:', str(d.get('error') or d)[:200])"
