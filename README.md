# Qwen3.8-Flash-Next 125B-A5B INT4-AR

Qwen3.8-Flash-Next, quantized to int4 and cut to **5 routed experts per token instead of 10**, then healed
so it behaves as close to the original as we could get. It runs on **one DGX Spark (GB10, 128 GB)** at
about 70-75 tokens/s. 125B parameters in total, **4.8B active per token** (the original activates 6B).

## What you need

- A DGX Spark or another GB10 box, 128 GB unified memory
- Docker with the NVIDIA container toolkit
- About 130 GB of free disk. The model goes into the standard Hugging Face cache, so other tools share it
- **Recommended:** [Eugr's spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), for the faster b12x serving stack:

  ```bash
  git clone https://github.com/eugr/spark-vllm-docker ~/spark-vllm-docker
  ```

  With it, setup pulls Eugr's b12x image (pinned to the build this model was validated on) and skips building
  ours: about 75 tok/s instead of 70, and no 20-40 minute compile. Without it, everything still works on our own image.

## Get it running

```bash
git clone https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST.git
cd Qwen3.8-Flash-Next-Int4-FAST
./setup.sh        # gets the image, downloads the weights (~1 hour the first time)
./serve.sh        # starts the server on port 8000 (-d to run it in the background)
./check.sh        # in another terminal: what is being served, plus one test prompt
```

`setup.sh` is safe to re-run: it skips what is already done and checks the image and the cached weights
every time. To update, `git pull && ./setup.sh`. Unchanged files are reused, so this usually takes seconds.

## Using it

It speaks the OpenAI API, so any client works:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-a5b",
  "messages": [{"role": "user", "content": "Write a haiku about sparse attention."}]
}'
```

Tool calling and reasoning are on. For an answer without the thinking step, send
`"chat_template_kwargs": {"enable_thinking": false}`.

## Settings

Every setting is one commented line at the top of `serve.sh`. Edit it, or set it for a single run:
`CTX=65536 KV_BYTES=30g ./serve.sh`. The ones you are most likely to touch are `PORT`, `CTX` (longest
request), `SEQS` (requests at once) and `KV_BYTES` (KV cache size; 20g holds ~645k tokens).

`BACKEND=classic` uses our own image even when Eugr's repo is there, for setup and serving alike. On the b12x
stack the speculative-decoding settings come from the recipe, not from `serve.sh`. For **two nodes**, `setup.sh`
also installs a cluster recipe into Eugr's repo and prints the command to run it.

## How it measures up

Our own harness, so read it as a relative comparison. Capability is a trimmed average over 6 runs; tool use is 3 trials.

| | This model (5 experts) | Original (10 experts) |
|---|---|---|
| Capability | 47.6 | 51.8 |
| Tool use | 85 | 86 |
| Speed | ~70-75 tok/s | ~57 tok/s |
| Active parameters | 4.8B | 6.0B |

The model ships its own retrained speculative-decoding head. At temperature 0.5 it gets 75% of its drafted tokens accepted overall
(85% code, 63% prose, 89% long context).

## Credits

The heavy lifting is other people's work:

- **Qwen** for Qwen3.8-Flash-Next, and **Intel** for its AutoRound int4 quantization
- **[vLLM](https://github.com/vllm-project/vllm)**, which everything here serves on
- **[Eugr](https://github.com/eugr/spark-vllm-docker)** (Eugene Rakhmatulin) for spark-vllm-docker, the b12x image, and the
  Qwen3.8-Flash-Next b12x recipe our recipes are built from
- **[lukealonso](https://github.com/lukealonso/b12x)** for b12x, the kernels and RoCE all-reduce behind that image
- **[Saren-Arterius](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)** for the vLLM build our own image is based on,
  and for the fp8 hybrid loader the b12x mod carries over from it
- **[blazux](https://github.com/blazux/qwen3.8-Flash-DGX)** for the DGX Spark serving work the fp8 KV cache patch comes from
- **[@Nanetnounou](https://github.com/Nanetnounou)** for that fp8 KV cache patch (`KV_DTYPE=fp8_e4m3`)
- **[@jschmied](https://github.com/jschmied/qwen38-flash-next-gb10)** for the deterministic top-k kernel
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122), `DET_TOPK=1`)
- **Sun et al.** for block verification, the speculative-decoding check `serve.sh` uses by default

What is ours: cutting the routed experts from 10 to 5 and healing the model afterwards, plus these scripts.

## Troubleshooting

- **Slow to start.** Loading 72 GB takes two to five minutes. Watch it with `docker logs -f qwen38-flash`.
- **Out of memory.** Lower `KV_BYTES`.
- **Docker cannot see the GPU.** Install the NVIDIA container toolkit and restart the docker daemon.
- **Empty answers.** With thinking on, the text sits in the `reasoning` field until the model finishes thinking.
