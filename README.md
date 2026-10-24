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

  With it, setup pulls Eugr's latest b12x image and skips building ours: about 75 tok/s instead of 70, and no 20-40
  minute compile. Without it, everything still works on our own image.

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

**Optional:** an Orca-abliterated variant (refusals removed) — see [the last section](#optional-orca-abliterated-variant) before using it.

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

The b12x recipes set a server-default `presence_penalty` of 0.1 (`presence_penalty:` in the recipe) to damp repeated
loops in thinking blocks; a request that sends its own `presence_penalty` keeps it. The mod patches vLLM so this default
applies at all: stock vLLM ignores `presence_penalty` in `--override-generation-config`.

The recipes also default `temperature` to 0.6 (clients that send their own keep it): with the speculative-decoding
head, 0.6 decodes 10-20 % faster than the model's 1.0 on real agent traffic. The two-node recipe loads with fastsafetensors
through `fst-ep-local/`, a second mod that `eugr-setup.sh` installs: each node reads only its own half of the experts from
its own disk (about 35 s instead of about 4 minutes, and no cross-node broadcast).

The mod also fixes a speculative-decoding bug that hit parallel sessions: with 3, 5, 6 or 7 requests running (any count
that is not 1, 2, 4, 8 or 16) the MTP drafter replayed a padded CUDA graph whose spare rows read stale request ids, and one
request would then draft at 1 token per step for the rest of its turn. The QSA metadata builder now clears those rows every step.
To check whether a running server has this bug: `sh tools/mtp_drafter_check.sh [http://host:8000]` (needs only sh and
python3; about a minute on an idle server; prints PASS or FAIL with the tokens per step of every parallel stream).
Eugr's 2026-10-15 build no longer has the bug, so the mod leaves it alone there.

**Which b12x image.** `eugr-setup.sh` (run by `setup.sh`) pulls Eugr's latest b12x build, the one his
`build-and-copy.sh --exp-b12x` pulls, and tags it `vllm-node-b12x-<build date>` (for example `vllm-node-b12x-20261015`).
It then sets the `container:` line of every recipe of ours in Eugr's `recipes/` to that tag, so the recipes always run
the image that was pulled. Re-run `./eugr-setup.sh` to move to a newer build; the older tag stays until you delete it.
Validated: the 2026-10-15 build (vLLM 0.1.dev21830) on one node. On the two-node cluster, run `eugr-setup.sh` on both
nodes so they share the tag.

## How it measures up

Our own harness, so read it as a relative comparison. Capability is the average of 6 runs for this model (the original:
median of 3). Tool use is Tool-Eval-Bench (69 scenarios, 3 trials). Both serving stacks score the same: 93 tool use on
ours, 94 on Eugr's b12x.

| | This model (5 experts) | Original (10 experts) |
|---|---|---|
| Capability | ~50 | 51.8 |
| Tool use | ~93 | 89 |
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

## Optional: Orca-abliterated variant

```bash
./setup.sh --orca     # on each node: the normal setup, then builds the variant on top of the base model
                      # (about a minute, ~7.5 GB, no download) and installs its solo and cluster recipes
```

Serve it with `B12X_RECIPE=qwen3.8-flash-next-int4-b12x-orca-solo.yaml ./serve.sh`, or with the `-orca` recipes in
`~/spark-vllm-docker`. What it is and how it measures: [orca/README.md](orca/README.md).

### ⚠️ Disclaimer — read before use

This variant has had its safety alignment substantially removed via abliteration (orthogonalizing the refusal direction
out of the residual stream). As a direct consequence:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the original Qwen3.8-Flash-Next would
  refuse. It has no meaningful built-in guardrails.
- It is released **strictly for legitimate research** — interpretability, AI-safety and refusal-mechanism study,
  red-teaming, robustness evaluation, and controlled experiments.
- **You assume full responsibility and liability** for how you use it and for everything it generates. Do not deploy it to
  end users or in production without adding your own safety, moderation, and abuse-prevention layers.
- Use must comply with the licenses inherited from the base model ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next))
  and from OrcaRouter's release ([orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)),
  and with all laws and regulations that apply to you.
- The authors and uploaders **accept no liability** for any misuse or harm. Its outputs do **not** reflect the views of the
  uploaders, of OrcaRouter, or of Qwen / Alibaba.

**By building or using this variant you acknowledge and accept the above.**
