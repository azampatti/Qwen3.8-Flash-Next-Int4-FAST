# Qwen3.8-Flash-Next-Swift1.5 (azampatti)

A variant of [azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
that thinks about **40% fewer tokens** at about the same accuracy. It replaces two pieces of the base model:

- **Attention:** the attention weights of [ukisai/Swift1.5-Qwen3.8-Flash-Next](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next),
  the fine-tune that taught Qwen3.8-Flash-Next to reason tersely.
- **Shared expert:** a new one, healed on that trunk with the same KL recipe as the base model's, against a corpus Swift1.5 generated
  at xhigh. This is V11, epoch 2, stopped at step 2050.

Everything else is the base model: int4 AutoRound experts (top-k 5), int4 lm_head, the fp8 PLE table, the retrained MTP
speculative head, and the config. This package therefore holds only about 3 GB of new weights and links the rest from the base model you
already have.

## Quick start

**Prerequisites:** the base model and eugr's b12x stack, exactly as for the base model:

```bash
git clone https://github.com/eugr/spark-vllm-docker ~/spark-vllm-docker
git clone https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST ~/Qwen3.8-Flash-Next-Int4-FAST
~/Qwen3.8-Flash-Next-Int4-FAST/setup.sh          # downloads the base model and pulls the pinned vllm-node-b12x image
```

**Then this variant (optional, on top of the base):**

```bash
~/Qwen3.8-Flash-Next-Int4-FAST/swift15/setup.sh   # downloads the ~3 GB of Swift1.5 files, ~1-2 minutes after that, no GPU
cd ~/spark-vllm-docker
./run-recipe.sh recipes/qwen3.8-flash-next-swift15-b12x-solo.yaml --solo     # one DGX Spark
./run-recipe.sh recipes/qwen3.8-flash-next-swift15-b12x.yaml                 # two nodes: run setup.sh on BOTH, launch on the head
```

The API is at `http://<host>:8000/v1` and the model name is `azampatti/Qwen3.8-Flash-Next-Swift1.5`. The base model keeps working
side by side: its recipes, files and HF cache entry are untouched.

`setup.sh` does four things:
1. Downloads the three new weight files from [azampatti/Qwen3.8-Flash-Next-Swift1.5](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-Swift1.5)
   into `swift15/model/` (git-ignored; `OVERLAY_REPO` overrides the repo) and checks them against `MANIFEST.json`.
2. Checks that the base model's pinned revision (`1464274`) is in your HF cache.
3. Builds the served model as `models--local--Qwen3.8-Flash-Next-Swift1.5` in the HF cache. The three new files are hard-linked
   in as blobs, and everything else is a symlink into the base snapshot.
4. Installs the mod and the two recipes into eugr's repo.

It is safe to re-run. `--force` rebuilds the served model, and `HF_HOME` and `EUGR_DIR` override the default paths.

## Reasoning effort

The recipes serve the model's stock chat template, so the default effort is **xhigh**, where this variant is strongest. Set
`"reasoning_effort": "medium"` (or `"low"`) per request to change it, or `chat_template: medium_chat_template.jinja` in the recipe
to make medium the default.

The recipes also set a server-default `presence_penalty` of 0.5 against repeated loops in thinking blocks; a request
that sends its own value keeps it.

## Results

All results were measured against the base model (Production V8) on the same b12x stack, with the same questions, seeds and
sampling, and 8 requests in flight.

**Thinking suites** (GPQA-Diamond, AIME 2025, MATH-500; 688 paired questions):

| | Thinking tokens vs base | Accuracy (this / base) |
|---|---|---|
| xhigh (GPQA ×2 seeds, AIME) | **0.58x** | 88.3% / 88.9% |
| medium (GPQA ×2 seeds, MATH, AIME) | **0.61x** | 82.9% / 84.8% |
| All 688 | **0.59x** | 84.2% / 85.8% |

The whole run took 7.2 h of wall time against the base model's 12.5 h. No individual set differs significantly at these sample sizes.
The medium-effort gap comes from the healing corpus being xhigh-only.

**Battery** (thinking off unless noted):

| | This variant | Base model |
|---|---|---|
| Capability benchmark (run.py ×6) | 50.6% (sd 1.4) | 49.1–49.7% (V8 epoch 1 / epoch 2 s1675) |
| Tool-Eval-Bench ×3 (xhigh) | 91.3 ± 1.5, Pass@3 88.4 | 91.3 ± 0.6 (V8 s1675, medium) |
| Decode, single stream (run.py) | 77–79 tok/s | 77–79 tok/s |

## Files

| File | Content |
|---|---|
| `model/model-swift15-attn.safetensors` (downloaded) | Swift1.5 attention projections: 156 GDN and QSA tensors, block-quantized to fp8 e4m3 (128×128 scales), the same layout as the base model's attention |
| `model/model-healed-shared-expert.safetensors` | The healed 1280-wide shared expert (fp8) plus the 48 trained shared-expert gate rows (bf16). Same tensor names and shapes as the base file it replaces. |
| `model/model.safetensors.index.json` | The base index with the attention keys pointed at the Swift1.5 file |
| `MANIFEST.json` | sha256 of every shipped file, the pinned base revision, and provenance |
| `eugr/flashnext-swift15-b12x/` | The eugr mod: the base model's `flashnext-int4-b12x` code, pointed at this variant |
| `eugr/qwen3.8-flash-next-swift15-b12x{-solo,}.yaml` | The single-node and two-node recipes. Settings are the base model's measured b12x ones; only the model, served name and template change. |

## Licence and credits

- **[ukisai](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)** for Swift1.5, whose attention weights this variant ships
  (licence: *other*, inherited from Qwen; check its terms before redistributing).
- **Qwen** for Qwen3.8-Flash-Next.
- **[Eugr](https://github.com/eugr/spark-vllm-docker)** for spark-vllm-docker and the b12x image.
- **[lukealonso](https://github.com/lukealonso/b12x)** for b12x.

What is ours: the expert cut, the healing (base and this variant), the fp8 packaging and these scripts.
