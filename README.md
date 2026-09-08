# Qwen3.8-Flash-Next 125B-A5B INT4-AR

Qwen3.8-Flash-Next, quantized to int4 and cut to **5 routed experts per token instead of 10**, then healed
so it behaves as close to the original as we could get. It runs on **one DGX Spark (GB10, 128 GB)** at
around 64-70 tokens/s.

125B parameters in total, **4.8B active per token**. The original activates 6B.

---

## What you need

- A DGX Spark or another GB10 box, 128 GB unified memory
- Docker with the NVIDIA container toolkit
- About 130 GB of free disk (the model repository is ~120 GB: 72 GB of weights plus a 49 GB n-gram table)
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

That is the whole thing. `setup.sh` is safe to re-run; it skips anything already done.

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

Edit `config.env`, or put any of these in front of the command:

| Setting | Default | What it does |
|---|---|---|
| `PORT` | 8000 | Port to serve on |
| `KV_BYTES` | `20g` | Size of the KV cache. 20 GB holds about 645,000 tokens |
| `CTX` | 262144 | Longest single request |
| `SEQS` | 8 | How many requests at once |
| `MTP` | 3 | Speculative decoding depth. `0` turns it off |
| `DRAFT_K10` | 1 | Let the speculator draft over 10 experts. Faster, costs nothing |
| `DET_TOPK` | 0 | `1` makes greedy output identical run to run |
| `KV_DTYPE` | `auto` | `fp8_e4m3` fits ~1.9x more context per GB, but is slower and slightly worse |
| `CHAT_TEMPLATE` | — | Path to one of the alternative templates shipped with the model |

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

## What the image contains

Nothing of ours. It is the image built by **[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)**,
which is what makes this int4 checkpoint servable at all, plus two optional patches fetched from their
own repositories at pinned commits and verified by checksum:

- **fp8 KV cache** for the sparse-attention path, by [@Nanetnounou](https://github.com/Nanetnounou),
  from [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) (`KV_DTYPE=fp8_e4m3`)
- **deterministic top-k kernel**, by [@jschmied](https://github.com/jschmied)
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122)) (`DET_TOPK=1`)

Both are off by default, so out of the box the image behaves exactly like the upstream one. If you do not
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
