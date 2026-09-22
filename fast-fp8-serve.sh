#!/usr/bin/env bash
# Start the server in FAST-FP8 mode.   ./fast-fp8-serve.sh      foreground, Ctrl-C stops it
#                                      ./fast-fp8-serve.sh -d   background
#
# Same model, same settings, same image as ./serve.sh -- plus two optional FP8 weight sets that ship in the model's
# fast-fp8/ folder on Hugging Face:
#   * the speculative (MTP) head's 512 draft experts in blockwise FP8        (4.86 -> 2.51 GB)
#   * the trunk's 96 hyper-connection mixer projections in blockwise FP8     (1.17 -> 0.59 GB), served by a W8A16
#     Triton kernel that keeps the activations in BF16
# Measured on one GB10 against the same model through ./serve.sh: 74.9 vs 71.7 tok/s (+4.5%, 6 runs), capability 50.5 vs
# 49.7 (a tie), tool-eval 91.0 vs 91.3 (a tie). Speculative decoding stays exact, so the draft head can only change speed.
#
# ./serve.sh and ./setup.sh are untouched: to go back, just run ./serve.sh. Every setting in serve.sh's SETTINGS block
# applies here too (CTX=32768 ./fast-fp8-serve.sh works the same way).
#
# What this script does, every run (cheap, a second or two):
#   1. finds fast-fp8/ inside your downloaded copy of the model (./setup.sh fetches it with everything else);
#   2. checks that its index matches that copy of the model tensor for tensor, then builds (or refreshes) a folder of
#      links, $MODELS_DIR/<model>-fast-fp8: every file links to your downloaded model, except the index, the speculative
#      head and the mixer file, which link to fast-fp8/. Nothing is copied, the download is never modified;
#   3. builds FP8-aware copies of two modules of the image you are running (read out of the image itself and patched on
#      a few anchor lines; stops with a message if the image does not match) plus the mixer kernel, and mounts them;
#   4. hands over to ./serve.sh with that folder, so everything else is exactly the normal launch.
set -euo pipefail
cd "$(dirname "$0")"
SETTINGS_ONLY=1 . ./serve.sh
source ./config.env
[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] || { echo "Model not found (DOWNLOAD_MODE=$DOWNLOAD_MODE) -- run ./setup.sh first" >&2; exit 1; }
BASE_DIR="$MODEL_DIR"
FP8_SRC="${FP8_SRC:-$BASE_DIR/fast-fp8}"          # where the three fast-fp8 files are (normally inside the downloaded model)
FAST_DIR="$MODELS_DIR/$(basename "$MODEL_REPO")-fast-fp8"
SD="$MODELS_DIR/.fast-fp8"                          # the generated modules
for f in model.safetensors.index.json model_extra_tensors.safetensors model-hc-fp8.safetensors; do
  [ -f "$FP8_SRC/$f" ] || { echo "fast-fp8: $FP8_SRC/$f not found. Your copy of the model predates the fast-fp8 files --" \
                                 "run ./setup.sh (it fetches the newest revision of the model), then run this again." >&2; exit 1; }
done
command -v python3 >/dev/null || { echo "fast-fp8: python3 is required on the host" >&2; exit 1; }
mkdir -p "$SD"

# --- 2. the link folder -----------------------------------------------------------------------------------------------
python3 - "$BASE_DIR" "$FP8_SRC" "$FAST_DIR" <<'PY'
import json, os, shutil, struct, sys
base, src, dst = (os.path.abspath(p) for p in sys.argv[1:4])
FP8_FILES = ("model.safetensors.index.json", "model_extra_tensors.safetensors", "model-hc-fp8.safetensors")
HEAD, HC = "model_extra_tensors.safetensors", "model-hc-fp8.safetensors"
def die(msg): sys.exit(f"fast-fp8: {msg}")
def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    h.pop("__metadata__", None); return h
stamp = json.dumps({"base": os.path.realpath(base), "fp8": {f: [os.path.realpath(os.path.join(src, f)),
                    os.path.getsize(os.path.join(src, f))] for f in FP8_FILES}}, sort_keys=True)
stamp_file = os.path.join(dst, ".fast-fp8-links")
if os.path.isfile(stamp_file) and open(stamp_file).read() == stamp and os.path.isfile(os.path.join(dst, "config.json")):
    sys.exit(0)                                              # up to date
# validate: the fp8 index must be the model's own index plus exactly the fp8 changes
bi = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
fi = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
head, hc = header(os.path.join(src, HEAD)), header(os.path.join(src, HC))
missing = [k for k in bi if k not in fi]
if missing: die(f"fast-fp8/model.safetensors.index.json does not match this copy of the model ({len(missing)} of its tensors are "
                f"missing, e.g. {missing[0]}). The fast-fp8 files belong to a different revision; run ./setup.sh to update.")
for k, f in fi.items():
    if k in bi and f != bi[k] and not (f == HC and k in hc):
        die(f"fast-fp8 index moves {k} from {bi[k]} to {f}: not an fp8 change -- refusing to build")
    if k not in bi and not k.endswith(".weight_scale_inv"):
        die(f"fast-fp8 index adds {k}, which is not an fp8 scale -- refusing to build")
    if f in (HEAD, HC) and k not in (head if f == HEAD else hc):
        die(f"fast-fp8 index maps {k} to {f}, but that file has no such tensor")
for name, h, f in (("speculative head", head, HEAD), ("mixer file", hc, HC)):
    if not any(v["dtype"] == "F8_E4M3" for v in h.values()): die(f"the fast-fp8 {name} holds no FP8 tensors")
    unlisted = [k for k in h if fi.get(k) != f]
    if unlisted: die(f"the fast-fp8 index does not list {len(unlisted)} tensor(s) of {f}, e.g. {unlisted[0]}")
# build: links only. Refuse to touch a folder this script did not create.
if os.path.lexists(dst):
    if not os.path.isfile(stamp_file): die(f"{dst} exists and was not made by fast-fp8-serve.sh -- move it away first")
    shutil.rmtree(dst)
os.makedirs(dst)
for name in sorted(os.listdir(base)):
    if name in FP8_FILES or name == "fast-fp8" or name.startswith(".download"):
        continue
    os.symlink(os.path.join(base, name), os.path.join(dst, name))
for name in FP8_FILES:
    os.symlink(os.path.join(src, name), os.path.join(dst, name))
absent = sorted({f for f in fi.values() if not os.path.exists(os.path.join(dst, f))})
if absent: shutil.rmtree(dst); die(f"the fast-fp8 index names file(s) this copy of the model does not have: {absent[:3]}")
open(stamp_file, "w").write(stamp)
print(f"  fast-fp8 link folder ready: {dst} ({len(os.listdir(dst)) - 1} links, fp8 index {len(fi)} tensors)")
PY
# A draft folder built for an earlier link folder would carry a stale config: rebuild it with the link folder.
DRAFT_FAST="$FAST_DIR-draft-k10"
[ -e "$DRAFT_FAST" ] && [ "$FAST_DIR/.fast-fp8-links" -nt "$DRAFT_FAST/config.json" ] && rm -rf "$DRAFT_FAST"

# The mixer kernel + modules, written verbatim into $SD/hc_fp8.py and mounted next to the image's hyperconnection.py.
write_hc_fp8_py(){ cat <<'HC_FP8_PY'
# SPDX-License-Identifier: Apache-2.0
"""L12 (2026-09-22; shipped via fast-fp8-serve.sh): the trunk's hyper-connection mixer projections served from blockwise FP8 weights.

Why a custom kernel: on GB10 (sm121) the mixers run plain cuBLAS BF16 (the tuned skinny GEMM in low_latency_gemm.py is
sm103-only), and the stock FP8 paths do not help at decode shapes: CUTLASS blockwise W8A8 is a wash on the K=10240 down
GEMM (no split-K), Triton blockwise is 5x slower, Marlin FP8 gains ~15%. Measured 2026-09-22 (heal_b/mtp_fp8/NOTES.md).
This W8A16 kernel (fp8 weight x 128x128 fp32 block scales, BF16 activations, fp32 accumulate) is bandwidth-bound:
per trunk forward over the 96 mixers, down 3.3 -> 2.0 ms and up 2.95 -> 1.6 ms. No activation quantization, so the only
error is the weight cast (rel RMS 0.0265, same as the trunk's other FP8 side layers).

Shapes: down [320, 10240] + the exact BF16 inject rows [4, 10240] (fused into the same launch), up [10240, 320].
320 is not a multiple of 128: the edge scale blocks are partial (scale grids [3,80] / [80,3]); N and K tails are masked.
Decode (M <= 32): down = split-K partials kernel + reduce kernel (deterministic, no atomics); up = one direct kernel.
Prefill (M > 32): dequantise + cuBLAS (the GEMM is compute-bound there; the dequant is a few MB per mixer).
Both entry points are registered as custom ops so torch.compile / CUDA graphs treat them as opaque, like the other
qwen3_8_flash_next ops.
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

BLOCK = 128
BN_DOWN = 32          # measured best for [320, 10240] (10 n-blocks + 1 inject block) x split-K 10
BN_UP = 64            # measured best for [10240, 320]
SPLITK_DOWN = 10
M_TRITON_MAX = 32     # above this the dequant + cuBLAS path wins


@triton.jit
def _hc_w8a16_partial(X, W, S, WI, P, M, N, NI, K, sxm, swn, ssn, swin, spk, spm,
                      K_PER_SPLIT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    """P[split, m, :] = x[m, K-slice] @ [dequant(W[N, K]) ; WI[NI, K]]^T.
    Programs with pid_n < n_blocks do fp8 rows (one 128x128 block scale per (n_block, k_chunk): BN | 128, BK == 128);
    the program with pid_n == n_blocks does the BF16 inject rows exactly, with the same tile shapes."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_blocks = tl.cdiv(N, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    k0 = pid_k * K_PER_SPLIT
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if pid_n < n_blocks:
        n_mask = offs_n < N
        for kk in range(0, K_PER_SPLIT, BK):
            k = k0 + kk + offs_k
            k_mask = k < K
            x = tl.load(X + offs_m[:, None] * sxm + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(W + offs_n[:, None] * swn + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            s = tl.load(S + ((pid_n * BN) // 128) * ssn + (k0 + kk) // 128)
            acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s
    else:
        offs_i = offs_n - n_blocks * BN
        n_mask = offs_i < NI
        for kk in range(0, K_PER_SPLIT, BK):
            k = k0 + kk + offs_k
            k_mask = k < K
            x = tl.load(X + offs_m[:, None] * sxm + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(WI + offs_i[:, None] * swin + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))
        offs_n = N + offs_i
    tl.store(P + pid_k * spk + offs_m[:, None] * spm + offs_n[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _hc_reduce(P, OUT, M, NT, spk, spm, som, SPLITK: tl.constexpr, BM: tl.constexpr, BNT: tl.constexpr):
    """out[m, :NT] (bf16) = sum over splits of P[split, m, :NT] (fp32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BNT + tl.arange(0, BNT)
    mask = (offs_m < M)[:, None] & (offs_n < NT)[None, :]
    acc = tl.zeros((BM, BNT), dtype=tl.float32)
    for s in range(SPLITK):
        acc += tl.load(P + s * spk + offs_m[:, None] * spm + offs_n[None, :], mask=mask, other=0.0)
    tl.store(OUT + offs_m[:, None] * som + offs_n[None, :], acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _hc_w8a16_direct(X, W, S, OUT, M, N, K, sxm, swn, ssn, som, BN: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    """out[m, N] (bf16) = x[m, K] @ dequant(W[N, K])^T, no split-K (short K)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK) * BK, BK):
        k = kk + offs_k
        k_mask = k < K
        x = tl.load(X + offs_m[:, None] * sxm + k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(W + offs_n[:, None] * swn + k[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        s = tl.load(S + ((pid_n * BN) // 128) * ssn + kk // 128)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s
    tl.store(OUT + offs_m[:, None] * som + offs_n[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def _dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    se = s.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return (w.float() * se).to(torch.bfloat16)


def _splitk_for(k: int) -> int:
    nchunks = triton.cdiv(k, BLOCK)
    for sk in (SPLITK_DOWN, 8, 5, 4, 2, 1):
        if nchunks % sk == 0:
            return sk
    return 1


def _hc_fp8_down_inject(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, wi: torch.Tensor, out_cols: int) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    m = x2.shape[0]
    n, k = w.shape
    ni = wi.shape[0]
    nt = n + ni
    if m > M_TRITON_MAX or m == 0:
        y = F.linear(x2, torch.cat([_dequant(w, s), wi], 0))
        y = F.pad(y, (0, out_cols - nt)) if out_cols > nt else y
        return y.reshape(*x.shape[:-1], out_cols)
    bm = 16 if m <= 16 else 32
    mb = triton.cdiv(m, bm)
    nb = triton.cdiv(n, BN_DOWN)
    sk = _splitk_for(k)
    kps = (triton.cdiv(k, BLOCK) // sk) * BLOCK
    p = torch.empty(sk, m, nt, device=x.device, dtype=torch.float32)
    _hc_w8a16_partial[(nb + 1, sk, mb)](
        x2, w, s, wi, p, m, n, ni, k,
        x2.stride(0), w.stride(0), s.stride(0), wi.stride(0), p.stride(0), p.stride(1),
        K_PER_SPLIT=kps, BN=BN_DOWN, BK=BLOCK, BM=bm, num_warps=4)
    if out_cols > nt:
        out = torch.zeros(m, out_cols, device=x.device, dtype=torch.bfloat16)
    else:
        out = torch.empty(m, nt, device=x.device, dtype=torch.bfloat16)
    _hc_reduce[(mb, triton.cdiv(nt, 64))](p, out, m, nt, p.stride(0), p.stride(1), out.stride(0),
                                          SPLITK=sk, BM=bm, BNT=64, num_warps=4)
    return out.reshape(*x.shape[:-1], out_cols)


def _hc_fp8_down_inject_fake(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, wi: torch.Tensor, out_cols: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], out_cols))


def _hc_fp8_up(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    m = x2.shape[0]
    n, k = w.shape
    if m > M_TRITON_MAX or m == 0:
        return F.linear(x2, _dequant(w, s)).reshape(*x.shape[:-1], n)
    bm = 16 if m <= 16 else 32
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    _hc_w8a16_direct[(triton.cdiv(n, BN_UP), triton.cdiv(m, bm))](
        x2, w, s, out, m, n, k, x2.stride(0), w.stride(0), s.stride(0), out.stride(0),
        BN=BN_UP, BK=BLOCK, BM=bm, num_warps=4)
    return out.reshape(*x.shape[:-1], n)


def _hc_fp8_up_fake(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w.shape[0]))


direct_register_custom_op(op_name="qwen3_8_flash_next_hc_fp8_down_inject", op_func=_hc_fp8_down_inject, fake_impl=_hc_fp8_down_inject_fake)
direct_register_custom_op(op_name="qwen3_8_flash_next_hc_fp8_up", op_func=_hc_fp8_up, fake_impl=_hc_fp8_up_fake)


class HcFp8DownInject(nn.Module):
    """Drop-in for the merged down+inject(+pad) MergedColumnParallelLinear: output [M, rank + hc_count + pad].
    Checkpoint: `<hc>.input_mix_weight_down.weight` (fp8, shard 0 via _HC_WEIGHTS_MAPPER) + `.weight_scale_inv` (fp32 [3,80],
    also shard 0 through the same substring mapping) + `<hc>.block_inject_weight.weight` (bf16 [4, K], shard 1) -> `inject`,
    a non-persistent buffer so the loader does not expect a tensor under that name."""

    def __init__(self, in_features: int, rank: int, n_inject: int, pad: int, prefix: str = "") -> None:
        super().__init__()
        self.in_features, self.rank, self.n_inject, self.pad = in_features, rank, n_inject, pad
        self.out_cols = rank + n_inject + pad
        self.prefix = prefix
        self.weight = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float8_e4m3fn), requires_grad=False)
        set_weight_attrs(self.weight, {"weight_loader": self._load_weight})
        self.weight_scale_inv = nn.Parameter(
            torch.empty(triton.cdiv(rank, BLOCK), triton.cdiv(in_features, BLOCK), dtype=torch.float32), requires_grad=False)
        set_weight_attrs(self.weight_scale_inv, {"weight_loader": self._load_scale})
        self.register_buffer("inject", torch.zeros(n_inject, in_features, dtype=torch.bfloat16), persistent=False)
        self._inject_loaded = self._fp8_loaded = self._scale_loaded = False

    def _load_weight(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        shard = getattr(loaded, "shard_id", None)
        if shard == 1 or (loaded.dtype != torch.float8_e4m3fn and tuple(loaded.shape) == tuple(self.inject.shape)):
            self.inject.copy_(loaded.to(self.inject.dtype))
            self._inject_loaded = True
            return
        if tuple(loaded.shape) != tuple(param.shape):
            raise ValueError(f"{self.prefix}: expected weight {tuple(param.shape)}, got {loaded.dtype} {tuple(loaded.shape)}")
        if loaded.dtype != torch.float8_e4m3fn:
            # The original BF16 copy still sits in the model's own shards; a loader without an index filter reads it too.
            # It is superseded by the fp8 copy (model-hc-fp8.safetensors): ignore it. forward() refuses to run if the
            # fp8 copy never arrived, so a missing file cannot pass silently.
            return
        param.data.copy_(loaded)
        self._fp8_loaded = True

    def _load_scale(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        if tuple(loaded.shape) != tuple(param.shape):
            raise ValueError(f"{self.prefix}: expected scale {tuple(param.shape)}, got {tuple(loaded.shape)}")
        param.data.copy_(loaded.to(param.dtype))
        self._scale_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self._fp8_loaded and self._scale_loaded and self._inject_loaded):
            raise RuntimeError(f"{self.prefix}: fp8 hyper-connection weights not loaded (fp8 {self._fp8_loaded}, scale "
                               f"{self._scale_loaded}, inject {self._inject_loaded}) -- the model folder must index model-hc-fp8.safetensors")
        return torch.ops.vllm.qwen3_8_flash_next_hc_fp8_down_inject(x, self.weight, self.weight_scale_inv, self.inject, self.out_cols)


class HcFp8Up(nn.Module):
    """Drop-in for the up ReplicatedLinear(rank -> hyper_hidden): `<hc>.input_mix_weight_up.weight` (fp8) + `.weight_scale_inv`."""

    def __init__(self, in_features: int, out_features: int, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn), requires_grad=False)
        set_weight_attrs(self.weight, {"weight_loader": self._load})
        self.weight_scale_inv = nn.Parameter(
            torch.empty(triton.cdiv(out_features, BLOCK), triton.cdiv(in_features, BLOCK), dtype=torch.float32), requires_grad=False)
        set_weight_attrs(self.weight_scale_inv, {"weight_loader": self._load})
        self._fp8_loaded = self._scale_loaded = False

    def _load(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        if tuple(loaded.shape) != tuple(param.shape):
            raise ValueError(f"{self.prefix}: expected {param.dtype} {tuple(param.shape)}, got {loaded.dtype} {tuple(loaded.shape)}")
        if param.dtype == torch.float8_e4m3fn:
            if loaded.dtype != torch.float8_e4m3fn:
                return            # the superseded BF16 original from the model's own shards (see HcFp8DownInject)
            self._fp8_loaded = True
        else:
            self._scale_loaded = True
        param.data.copy_(loaded.to(param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self._fp8_loaded and self._scale_loaded):
            raise RuntimeError(f"{self.prefix}: fp8 hyper-connection weights not loaded (fp8 {self._fp8_loaded}, scale "
                               f"{self._scale_loaded}) -- the model folder must index model-hc-fp8.safetensors")
        return torch.ops.vllm.qwen3_8_flash_next_hc_fp8_up(x, self.weight, self.weight_scale_inv)


def hc_fp8_enabled(prefix: str, use_combine: bool) -> bool:
    """FP8 mixers only for the trunk's per-layer mixers (use_combine=True, prefix under language_model); the MTP head's
    mixers and the final mixer stay BF16 (they are not in the converted file)."""
    return os.environ.get("VLLM_HC_FP8", "0").lower() in ("1", "true", "yes") and use_combine and "language_model" in prefix
HC_FP8_PY
}

# --- 3. the FP8-aware modules -----------------------------------------------------------------------------------------
IMG_ID="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)" || { echo "fast-fp8: image $IMAGE not found -- run ./setup.sh first" >&2; exit 1; }
if ! { [ "$(cat "$SD/image_id" 2>/dev/null)" = "$IMG_ID" ] && [ ! "$(basename "$0")" -nt "$SD/image_id" ] \
       && [ -s "$SD/hc_fp8.py" ] && [ -s "$SD/hyperconnection.py" ] && [ -s "$SD/vllm_fp8_hybrid.py" ] && [ -s "$SD/paths" ]; }; then
  echo "  building the fast-fp8 modules for image $IMAGE"
  rm -f "$SD/image_id"
  PATHS="$(docker run --rm --entrypoint sh "$IMAGE" -c \
    'find /usr/local/lib /usr/lib -path "*qwen3_8_flash_next/nvidia/hyperconnection.py" 2>/dev/null | head -1; find /usr/local/lib /usr/lib -name vllm_fp8_hybrid.py -path "*-packages/*" 2>/dev/null | head -1')"
  HC_PATH="$(sed -n 1p <<<"$PATHS")"; HY_PATH="$(sed -n 2p <<<"$PATHS")"
  [ -n "$HC_PATH" ] && [ -n "$HY_PATH" ] || { echo "fast-fp8: $IMAGE lacks the qwen3_8_flash_next hyper-connection module or the fp8 hybrid loader -- use ./serve.sh" >&2; exit 1; }
  docker run --rm --entrypoint cat "$IMAGE" "$HC_PATH" > "$SD/hyperconnection.orig.py"
  docker run --rm --entrypoint cat "$IMAGE" "$HY_PATH" > "$SD/vllm_fp8_hybrid.orig.py"
  write_hc_fp8_py > "$SD/hc_fp8.py"
  python3 - "$SD" <<'PY'
import os, sys
sd = sys.argv[1]
def patch(name, anchor, text, where):
    src = open(os.path.join(sd, name + ".orig.py")).read()
    if src.count(anchor) != 1:
        sys.exit(f"fast-fp8: {name}.py in this image does not match (expected exactly one '{anchor.strip()[:70]}', found "
                 f"{src.count(anchor)}). The image changed; use ./serve.sh, or update fast-fp8-serve.sh.")
    out = src.replace(anchor, anchor + text if where == "after" else text + anchor)
    open(os.path.join(sd, name + ".py"), "w").write(out)
patch("hyperconnection", "        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0\n", '''\
        # fast-fp8 (added by fast-fp8-serve.sh): with VLLM_HC_FP8=1 the trunk's per-layer mixers load blockwise-fp8 weights
        # and run the W8A16 kernel in hc_fp8.py. Output shapes are unchanged, so mix() / combine_and_mix() are untouched.
        from .hc_fp8 import HcFp8DownInject, HcFp8Up, hc_fp8_enabled, logger as _hc_fp8_logger
        self.hc_fp8 = hc_fp8_enabled(prefix, use_combine)
        if self.hc_fp8:
            _hc_fp8_logger.info_once("fast-fp8: hyper-connection mixers use blockwise-fp8 weights + the W8A16 kernel")
            self.input_mix_weight_down_block_inject = HcFp8DownInject(
                self.hyper_hidden_size, self.lora_rank, self.hc_count, self.pad_size,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"))
            self.input_mix_weight_up = HcFp8Up(
                self.lora_rank, self.hyper_hidden_size, prefix=maybe_prefix(prefix, "input_mix_weight_up"))
            return
''', "after")
patch("vllm_fp8_hybrid", "    setattr(cfg_cls, _SENTINEL, True)\n", '''\
    # --- fast-fp8 (added by fast-fp8-serve.sh): the speculative head's routed experts may be stored as blockwise fp8.
    # A RoutedExperts layer is ONE module for all its experts, so it is matched on the expert GROUP
    # ("mtp.layers.0.mlp.experts"), on the draft side only -- the trunk's GPTQ-int4 experts are never touched.
    # A no-op for any checkpoint whose experts are not fp8.
    import re as _re
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    _EXP = r"\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
    _ff_update, _ff_mapper, _ff_gqm = cfg_cls.maybe_update_config, cfg_cls.apply_vllm_mapper, cfg_cls.get_quant_method

    def _ff_maybe_update_config(self, model_name, hf_config=None, revision=None):
        _ff_update(self, model_name, hf_config=hf_config, revision=revision)
        self.fp8_expert_groups = {_re.sub(_EXP, ".experts", n) for n in (getattr(self, "fp8_layers", None) or ())
                                  if _re.search(_EXP, n)}
        if self.fp8_expert_groups:
            logger.info("fast-fp8: blockwise-fp8 expert groups: %s", sorted(self.fp8_expert_groups))

    def _ff_apply_vllm_mapper(self, hf_to_vllm_mapper):
        _ff_mapper(self, hf_to_vllm_mapper)
        if getattr(self, "fp8_expert_groups", None):
            self.fp8_expert_groups = set(hf_to_vllm_mapper.apply_list(list(self.fp8_expert_groups)))

    def _ff_key(n):
        return ("trunk" if "language_model" in n else "draft"), _re.sub(r"^.*?layers\\.\\d+\\.", "", n)

    def _ff_get_quant_method(self, layer, prefix):
        groups = getattr(self, "fp8_expert_groups", None)
        if groups and isinstance(layer, RoutedExperts) and _ff_key(prefix) in {_ff_key(g) for g in groups}:
            fp8_cfg = getattr(self, "_fp8_cfg", None)
            if fp8_cfg is None:
                fp8_cfg = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
                fp8_cfg.packed_modules_mapping = self.packed_modules_mapping
                self._fp8_cfg = fp8_cfg
            logger.info("fast-fp8: %s -> Fp8MoEMethod (blockwise fp8 experts)", prefix)
            return fp8_cfg.get_quant_method(layer, prefix)
        return _ff_gqm(self, layer, prefix)

    cfg_cls.maybe_update_config = _ff_maybe_update_config
    cfg_cls.apply_vllm_mapper = _ff_apply_vllm_mapper
    cfg_cls.get_quant_method = _ff_get_quant_method
''', "before")
PY
  printf '%s\n%s\n' "$HC_PATH" "$HY_PATH" > "$SD/paths"
  rm -f "$SD"/*.orig.py
  echo "$IMG_ID" > "$SD/image_id"
fi
HC_PATH="$(sed -n 1p "$SD/paths")"; HY_PATH="$(sed -n 2p "$SD/paths")"

# --- 4. hand over to serve.sh -----------------------------------------------------------------------------------------
HUB_MOUNT=""; [ "$DOWNLOAD_MODE" = cache ] && HUB_MOUNT="-v $HF_HOME/hub:$HF_HOME/hub:ro"
export DOWNLOAD_MODE=local MODELS_DIR MODEL_REPO="$MODEL_REPO-fast-fp8" DRAFT_DIR="$DRAFT_FAST" \
       DOCKER_EXTRA_ARGS="$HUB_MOUNT -v $SD/hyperconnection.py:$HC_PATH:ro -v $SD/hc_fp8.py:$(dirname "$HC_PATH")/hc_fp8.py:ro -v $SD/vllm_fp8_hybrid.py:$HY_PATH:ro -e VLLM_HC_FP8=1 ${DOCKER_EXTRA_ARGS:-}"
echo "fast-fp8: FP8 speculative head + FP8 hyper-connection mixers ($FAST_DIR)"
exec ./serve.sh "$@"
