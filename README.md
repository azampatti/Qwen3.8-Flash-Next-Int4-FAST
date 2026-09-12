# Qwen3.8-Flash-Next 125B-A5B INT4-AR

Qwen3.8-Flash-Next, quantized to int4 and cut to **5 routed experts per token instead of 10**, then healed
so it behaves as close to the original as we could get. It runs on **one DGX Spark (GB10, 128 GB)** at
around 64-70 tokens/s.

125B parameters in total, **4.8B active per token**. The original activates 6B.

---

## What you need

- A DGX Spark or another GB10 box, 128 GB unified memory
- Docker with the NVIDIA container toolkit
- About 130 GB of free disk (the model repository is ~120 GB: 72 GB of weights plus a 49 GB n-gram table).
  It goes into the standard Hugging Face cache, so it is shared with every other tool on your machine
- Roughly an hour for the first setup, most of it downloading

## Get it running

```bash
git clone https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST.git
cd Qwen3.8-Flash-Next-Int4-FAST
./setup.sh        # builds the image, downloads the weights. Once.
./serve.sh        # starts the server on port 8000
```

In another terminal:

```bash
./check.sh        # says what is being served and sends one test prompt
```

That is the whole thing. `setup.sh` is safe to re-run; it skips the work it has already done.

**Updating.** Both the scripts and the weights change from time to time:

```bash
git pull
./setup.sh        # keeps the image, asks the hub for changed model files, rebuilds the small draft folder
./serve.sh
```

`setup.sh` re-checks the hub every run. Files that have not changed are reused from your cache, so an
up-to-date machine finishes in seconds and a refreshed speculative-decoding head costs one 5 GB download
rather than the full 120 GB. Add `OFFLINE=1` to skip the hub check entirely.

Nothing is taken on trust. Every run also checks:

- **the image**, against what this checkpoint needs: the fp8 hybrid loader, the memory-mapped n-gram table,
  the speculative head, and the two sampling methods `serve.sh` defaults to. An image built weeks ago is not
  evidence that it still has them, so this runs even when the build step says "already built".
- **the cached weights**, against the hashes the hub reports right now. A download reports success when the
  revision looks complete in its own bookkeeping, which is not the same as the files being current. Anything
  outdated is dropped and fetched again, and setup stops rather than serve you a stale head.
- **the two small things setup generates**, the draft folder and the draft-logit module: both are rebuilt when
  they no longer match the model folder or the image in use.

The model goes into the **standard Hugging Face cache**, so if you already pulled it with `hf download` or your own
script, setup finds it and downloads nothing. Other tools on the machine share the same copy. If you would rather have
a plain folder of real files, run `DOWNLOAD_MODE=local ./setup.sh` instead.

## Using it

It speaks the OpenAI API, so any client works:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-a5b",
  "messages": [{"role": "user", "content": "Write a haiku about sparse attention."}]
}'
```

Tool calling and reasoning are on. To get an answer without the thinking step, send
`"chat_template_kwargs": {"enable_thinking": false}`.

## Options

Everything is in one block at the top of `serve.sh`, one setting per line. Edit a line and re-run,
or put any of them in front of the command for a single run.

| Setting | Default | What it does |
|---|---|---|
| `MODEL_REPO` | `azampatti/Qwen3.8-...-AutoRound` | Which model to download and serve |
| `SERVED_NAME` | `qwen3.8-flash-next-a5b` | The name clients send as `"model"` |
| `PORT` | 8000 | Port to serve on |
| `CTX` | 262144 | Longest single request |
| `SEQS` | 8 | How many requests at once |
| `KV_BYTES` | `20g` | Size of the KV cache. 20 GB holds about 645,000 tokens |
| `KV_DTYPE` | `auto` | `fp8_e4m3` fits ~1.9x more context per GB, but is slower and slightly worse |
| `CHAT_TEMPLATE` | — | `medium`, `xhigh`, or a path. Bare names resolve to the templates inside the model |
| `TOOL_PARSER` | `qwen3_coder` | How tool calls are parsed out of the reply |
| `REASONING_PARSER` | `qwen3` | Puts the thinking block in its own `reasoning` field |
| `MTP` | 3 | Speculative decoding depth. `0` turns it off; 3 is the measured optimum |
| `DRAFT_K10` | 1 | Let the speculator draft over 10 experts. Faster, costs nothing |
| `DET_TOPK` | 0 | `1` makes expert selection deterministic. Slower, and not enough on its own to make greedy output bit-exact |
| `REJECTION_SAMPLE` | `block` | How the drafted tokens are checked. `block` judges the three as a set (Sun et al.), `standard` one at a time. Both exact. Default is `block` because, with `DRAFT_SAMPLE=probabilistic`, it measured +2.4pp acceptance at T=0.5 (alone it was a tie) |
| `DRAFT_SAMPLE` | `probabilistic` | `probabilistic` keeps the draft's full logits for the accept test; `greedy` hands over a one-hot guess. Both exact. Neither knob changes anything at temperature 0 |
| `DRAFT_SCALE` | 2 | Multiplies the draft's logits before the accept test, so the draft commits instead of hedging. Verification still uses the model's own distribution, so the reply does not change; only the share of drafted tokens that survive. Measured +2.5pp acceptance and about +4% tok/s at temperature 0.5. `1` turns it off. Built from the image itself by `tools/draft_scale.py` |
| `IMAGE` | `qwen38-flash-dgx:a5b-int4` | The image `setup.sh` built |
| `CONTAINER` | `qwen38-flash` | Docker container name |
| `GPU_MEM_UTIL` | 0.01 | Deliberately tiny: `KV_BYTES` sizes the cache instead |
| `BATCHED_TOKENS` | 8192 | Prefill chunk size |
| `DOWNLOAD_MODE` | `cache` | `cache` uses `~/.cache/huggingface` like every other Hugging Face tool. `local` puts a plain folder of real files under `MODELS_DIR` instead |
| `HF_HOME` | `~/.cache/huggingface` | Where the cache lives |
| `MODELS_DIR` | `~/models` | Plain-folder location for `local` mode, and the small draft folder |
| `OFFLINE` | 0 | `1` skips every hub call in `setup.sh`, so nothing is checked or downloaded |
| `EXTRA_ARGS` | — | Anything else to append to the vLLM command line |
| `DOCKER_EXTRA_ARGS` | — | Extra flags for `docker run` itself (bind mounts, `-e` variables) |

Example: `KV_BYTES=30g DET_TOPK=1 ./serve.sh`

## How it measures up

Our own harness, so treat it as a relative comparison rather than a leaderboard entry. Capability is a
trimmed average over 6 runs; tool use is 3 trials.

| | This model (5 experts) | Original (10 experts) |
|---|---|---|
| Capability | 47.6 | 51.8 |
| Tool use | 85 | 86 |
| Speed | ~64-70 tok/s | ~57 tok/s |
| Active parameters | 4.8B | 6.0B |

You give up about four points of general capability and keep tool use intact, in exchange for a fifth
fewer active parameters and noticeably faster generation.

**Speculative decoding.** The model ships with its own retrained draft head. On our 60-row probe at
temperature 0.5, with the defaults in `serve.sh`, the share of drafted tokens the model accepts is:

| | Accepted |
|---|---|
| Everything | 76% |
| Code | 86% |
| Prose | 63% |
| Long context | 87% |

Prose is the hard case for any draft head, which is why `DRAFT_SCALE` exists: it is the cheapest two and
a half points of acceptance available, and it cannot change what the model says.

## What the image contains

Nothing of ours. It is the image built by **[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)**,
which is what makes this int4 checkpoint servable at all, plus two optional patches fetched from their
own repositories at pinned commits and verified by checksum:

- **fp8 KV cache** for the sparse-attention path, by [@Nanetnounou](https://github.com/Nanetnounou),
  from [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) (`KV_DTYPE=fp8_e4m3`)
- **deterministic top-k kernel**, by [@jschmied](https://github.com/jschmied)
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122)) (`DET_TOPK=1`)

Both are off by default, so out of the box the image behaves exactly like the upstream one. The one other
change, `DRAFT_SCALE`, is not vendored either: `tools/draft_scale.py` reads the speculative-decoding module
out of the image you are running, rewrites the single line that builds the draft's logits processor, and
`serve.sh` mounts that copy. If a future image does not match, it says so and serves without scaling. If you do not
need either, **use [Saren-Arterius's image directly** and skip this repo](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) —
`serve.sh` here is a thin wrapper around the same vLLM command.

## Credits

The heavy lifting is other people's work:

- **Qwen** for Qwen3.8-Flash-Next
- **Intel** for the AutoRound int4 quantization of it
- **[Saren-Arterius](https://github.com/Saren-Arterius)** for the vLLM build that serves it on a Spark
- **[blazux](https://github.com/blazux)** for the DGX Spark serving work these patches come from,
  **[@Nanetnounou](https://github.com/Nanetnounou)** and **[@jschmied](https://github.com/jschmied)** for the two patches

What is ours: cutting the routed experts from 10 to 5 and healing the model afterwards, plus these scripts.

## Troubleshooting

**It is taking forever to start.** First boot loads 72 GB and warms the n-gram table; two to five minutes
is normal. Watch it with `docker logs -f qwen38-flash`.

**Out of memory.** Lower `KV_BYTES`. The weights need about 72 GB and the rest of the box has to fit in
what is left.

**Docker cannot see the GPU.** Install the NVIDIA container toolkit and restart the docker daemon.

**Answers come back empty.** With thinking on, the answer is in the `reasoning` field until the model
finishes thinking. Send `"chat_template_kwargs": {"enable_thinking": false}` for a direct answer.
