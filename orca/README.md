# Orca-abliterated variant (optional)

## ⚠️ Disclaimer — read before use

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

In agent use it is also more willing to follow injected or "authority" instructions: it loses tool-eval-bench TC-58 / TC-59.

## About

The same model as the base (INT4-AutoRound routed experts at top-k 5, healed shared expert, int4 output head, MTP head, n-gram
table, the b12x serving stack and every setting of its recipes) with the refusal-direction edit of
[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) (Apache-2.0)
applied to it — no re-quantization, no retraining, nothing new to download.

## Install

```bash
./setup.sh --orca          # on EACH node: the normal setup (downloads the base model if needed), then builds the variant
                           # and installs its recipes into Eugr's spark-vllm-docker (eugr-setup.sh --orca)
```

Then, from `~/spark-vllm-docker`:

```bash
./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x-orca-solo.yaml --solo   # one DGX Spark
./run-recipe.sh recipes/qwen3.8-flash-next-int4-b12x-orca.yaml               # two-node cluster (built on both nodes first)
```

or `B12X_RECIPE=qwen3.8-flash-next-int4-b12x-orca-solo.yaml ./serve.sh` from this repo. The served name is
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-Orca`.

## What it is

OrcaRouter's edit is one unit direction `r` (2560 values, `orca_direction.safetensors`) orthogonalized out of every matrix
that writes to the residual stream: `W' = W - r (rᵀW)` at strength 1.0. We measured it against Qwen's BF16 original on every
writer class (attention `o_proj`, GDN `out_proj`, routed and shared experts' `down_proj`, `ple.value_proj`, the MTP head,
the embedding rows): the same direction everywhere, rank-1, strength 1.000 ± 0.001; `lm_head`, the router, the
hyper-connection mixers and the input projections are untouched.

Here it is applied in two ways, both exact:

- **Folded into the bf16 weights** — embeddings, `ple.value_proj`, the MTP head (516 tensors): `build_orca.py` rewrites
  the three base files that hold them (≈7.5 GB) into a local Hugging Face cache repo,
  `models--local--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound-Orca`, whose other files are symlinks into the base snapshot.
- **At runtime** for the trunk — the mod's `orca-proj` patch projects `r` out of every decoder layer's attention and MLP
  block output, which is the same map as editing those writers. The int4 routed experts, the fp8 attention and the healed
  shared expert stay byte-identical: re-rounding an int4 weight that already sits on its grid would erase a 2 % edit.

## Measured (2026-10-24, one DGX Spark each, same image and settings, run in parallel)

| | base | Orca variant |
|---|---|---|
| run.py ×6 (thinking off) | 48.05 (sd 1.8) | 51.0 (sd 2.1) |
| run.py tokens / trial | 14,156 | 13,454 |
| decode, run.py | 77.5 tok/s | 76.6 tok/s |
| tool-eval-bench ×1 | 93 | 92 (FAILs TC-58, TC-59: the safety scenarios) |

The capability difference is within the noise of a six-run comparison (the base ran below its earlier 49.7): read it as no
measured loss.
