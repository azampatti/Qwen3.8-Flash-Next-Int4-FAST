#!/usr/bin/env python3
"""fst-ep-local: per-node fastsafetensors loading for TP/EP clusters (applied AFTER mods/flashnext-int4-b12x or mods/flashnext-swift15-b12x).

Stock vLLM, world size > 1: fastsafetensors_weights_iterator loads with the WORLD group -> each rank reads some shards and
EVERY tensor is broadcast to every rank (fastsafetensors tensor_factory.shuffle: owner clones, receivers allocate a full copy),
and the EP weight filter (local_expert_ids) is not passed to it at all. On two GB10 nodes that boot runs out of memory.

Patched, world size > 1 (single rank: the stock function runs unchanged):
  * fastsafetensors 0.4 ParallelLoader(all_local=True): every rank reads its own local copy, no cross-rank broadcast
  * tensor_filter, applied BEFORE bytes are read (nogds copier: filtered tensors are never read, plan_chunks skips them):
      - drop routed-expert tensors of experts this rank does not own. vLLM's own should_skip_weight only skips names ending
        .weight/.weight_packed, which for this GPTQ checkpoint matches only the MTP head's 1,536 bf16 expert tensors; the
        221,184 int4 tensors (.qweight/.qzeros/.scales) would all be read. Suffixes: env FST_EP_SKIP_SUFFIXES.
      - for /workspace/flashnext/* folders with an index: drop tensors the index places in another file (same rule as the
        flashnext index filter, which still runs afterwards; this only saves reading them -- Swift1.5's 2.7 GB of shadowed
        attention tensors in the base shards).
  * chunked loads: max_batch_bytes (env FST_EP_MAX_BATCH_BYTES, default 2 GiB >= the largest tensor, embed_tokens 1.18 GiB)
    with use_chunk_budget_as_allocation_size=True: every chunk buffer is the same size, so the CUDA caching allocator reuses
    one block instead of fragmenting on 1-5 GB whole-file buffers; queue_size = VLLM_FASTSAFETENSORS_QUEUE_SIZE (default 0:
    one chunk copying + one being consumed -> <= 2 x 2 GiB in flight + one cloned tensor).
  * FST_EP_LOCAL=0 restores the stock behaviour without rebuilding the container.

Every edit is idempotent (marker) and anchored (exit 1 if an anchor is missing).  usage: patch_fst.py [vllm package dir]
"""
import os, py_compile, sys

V = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"
MARK = "fst-ep-local"
WU = f"{V}/model_executor/model_loader/weight_utils.py"
DL = f"{V}/model_executor/model_loader/default_loader.py"

ITER = f'''

# --- {MARK}:iterator ---
def _fst_ep_keep_factory(files, local_expert_ids, skip_suffixes):
    """Tensor-name predicate for one ParallelLoader call: own experts only + index placement (see mods/fst-ep-local)."""
    import json as _json
    from pathlib import Path as _Path

    from vllm.model_executor.model_loader.ep_weight_filter import parse_expert_id as _eid

    here = None
    folders = {{_Path(f).parent for f in files}}
    if len(files) == 1 and str(next(iter(folders))).startswith("/workspace/flashnext/"):
        index = next(iter(folders)) / "model.safetensors.index.json"
        if index.is_file():
            weight_map = _json.load(open(index))["weight_map"]
            here = (weight_map, _Path(files[0]).name)

    def keep(name):
        if here is not None and here[0].get(name) != here[1]:
            return False
        if local_expert_ids is not None and name.endswith(skip_suffixes):
            eid = _eid(name)
            if eid is not None and eid not in local_expert_ids:
                return False
        return True

    return keep


_fst_stock_weights_iterator = fastsafetensors_weights_iterator


def fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    weight_name_prefixes: Sequence[str] | None = None,
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    if (
        os.environ.get("FST_EP_LOCAL", "1") == "0"
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() <= 1
    ):
        yield from _fst_stock_weights_iterator(
            hf_weights_files, use_tqdm_on_load, weight_name_prefixes=weight_name_prefixes
        )
        return
    from fastsafetensors.parallel_loader import ParallelLoader

    files = sorted(hf_weights_files, key=_natural_sort_key)
    skip_suffixes = tuple(
        s
        for s in os.environ.get(
            "FST_EP_SKIP_SUFFIXES", ".weight,.weight_packed,.qweight,.qzeros,.scales,.g_idx"
        ).split(",")
        if s
    )
    logger.info_once(
        "{MARK}: rank %d loads its own files (all_local, no broadcast), %s experts, skip suffixes %s",
        torch.distributed.get_rank(),
        "all" if local_expert_ids is None else len(local_expert_ids),
        ",".join(skip_suffixes),
    )
    pl = ParallelLoader(
        pg=None,
        hf_weights_files=files,
        queue_size=envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE,
        use_tqdm_on_load=enable_tqdm(use_tqdm_on_load),
        device=f"cuda:{{current_platform.current_device()}}",
        nogds=True,
        tensor_filter=_fst_ep_keep_factory(files, local_expert_ids, skip_suffixes),
        all_local=True,
        max_batch_bytes=int(os.environ.get("FST_EP_MAX_BATCH_BYTES", str(2 << 30))),
        use_chunk_budget_as_allocation_size=True,
    )
    try:
        for name, tensor in pl.iterate_weights():
            if weight_name_prefixes and not _matches_weight_name_prefixes(name, weight_name_prefixes):
                continue
            yield name, tensor
    finally:
        pl.close()
'''

ANCHOR = """            return fastsafetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                weight_name_prefixes=source.weight_name_prefixes,
            )"""
PASS = f"""            return fastsafetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                weight_name_prefixes=source.weight_name_prefixes,
                local_expert_ids=self.local_expert_ids,  # {MARK}:pass-ids
            )"""

done = []


def edit(path, label, fn):
    src = open(path).read()
    if f"{MARK}:{label}" in src:
        done.append(f"{label} (already)")
        return
    new = fn(src)
    if new is None or new == src:
        sys.exit(f"FATAL {MARK}: anchor for '{label}' not found in {path} -- this vLLM build differs; launch aborted")
    tmp = path + f".{MARK}.tmp"
    open(tmp, "w").write(new)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        os.remove(tmp)
        sys.exit(f"FATAL {MARK}: '{label}' would break {path}: {e}")
    os.replace(tmp, path)
    done.append(label)


def add_iter(s):
    needed = ("def fastsafetensors_weights_iterator(", "def _natural_sort_key(", "def enable_tqdm(",
              "def _matches_weight_name_prefixes(", "from vllm.platforms import current_platform", "from vllm import envs")
    return s + ITER if all(n in s for n in needed) else None


edit(WU, "iterator", add_iter)
edit(DL, "pass-ids", lambda s: s.replace(ANCHOR, PASS) if s.count(ANCHOR) == 1 else None)
print(f"{MARK}: " + ", ".join(done))
