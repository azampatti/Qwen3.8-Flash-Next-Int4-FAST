#!/usr/bin/env bash
# Is it up, and is it the model you think?   ./check.sh
set -euo pipefail
cd "$(dirname "$0")"; source ./config.env
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "Not answering on port $PORT (still loading? 'docker logs -f qwen38-flash')"; exit 1; }
curl -s "http://localhost:$PORT/v1/models" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'][0];print('served:',d['id'],'\nfrom:  ',d.get('root'),'\nmax context:',d.get('max_model_len'),'tokens')"
python3 -c "import json;c=json.load(open('$MODEL_DIR/config.json'));t=c.get('text_config',c);print('experts per token:',t['num_experts_per_tok'],'of',t['num_experts'])"
echo "local path:  $MODEL_DIR"
echo; echo "test prompt:"
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word READY.\"}],\"max_tokens\":30,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print(' ',m.get('content') or m.get('reasoning') or '(empty)')"
