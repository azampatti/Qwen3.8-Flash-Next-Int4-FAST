#!/usr/bin/env python3
"""Build the Orca-abliterated variant of the base model as a local Hugging Face cache repo (runs inside the b12x image: torch).

    build_orca.py <hub dir> <base snapshot dir> <this folder>

Writes <hub>/<served_repo>: the three rewritten files as blobs (named by sha256), refs/main = version, and a snapshot whose
other entries are relative symlinks into the base snapshot (nothing of the base is copied). See MANIFEST.json.
"""
import hashlib, json, os, re, struct, sys
import torch

HUB, BASE, HERE = sys.argv[1:4]
M = json.load(open(os.path.join(HERE, "MANIFEST.json")))
OUT = os.path.join(HUB, M["served_repo"]); SNAP = os.path.join(OUT, "snapshots", M["version"])
base_repo_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(BASE))))
base_rev = os.path.basename(os.path.realpath(BASE))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def header(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; return f, json.loads(f.read(n)), 8 + n


# 0. inputs: the direction and the three base files must be exactly the ones the variant was built from
assert sha256(os.path.join(HERE, M["direction"]["file"])) == M["direction"]["sha256"], "orca_direction.safetensors does not match MANIFEST.json"
for name, meta in M["inputs"].items():
    p = os.path.join(BASE, name)
    blob = os.path.basename(os.path.realpath(p))      # HF cache blobs are named by their sha256
    ok = blob == meta["sha256"] or sha256(p) == meta["sha256"]
    assert ok and os.path.getsize(p) == meta["bytes"], f"{name} in the base snapshot is not the file this variant was built from"
f, h, b = header(os.path.join(HERE, M["direction"]["file"]))
s, e = h["orca_refusal_direction"]["data_offsets"]; f.seek(b + s)
r = torch.frombuffer(bytearray(f.read(e - s)), dtype=torch.float32).clone(); r /= r.norm()

WRITERS = re.compile(r"^(mtp\.layers\.0\.self_attn\.o_proj\.weight|mtp\.layers\.0\.mlp\.shared_expert\.down_proj\.weight|"
                     r"mtp\.layers\.0\.mlp\.experts\.\d+\.down_proj\.weight|model\.language_model\.layers\.1\.ple\.value_proj\.weight)$")
ROWSPACE = {"model.language_model.embed_tokens.weight"}
os.makedirs(os.path.join(OUT, "blobs"), exist_ok=True); os.makedirs(os.path.join(OUT, "refs"), exist_ok=True)
if os.path.isdir(SNAP):
    for n in os.listdir(SNAP):
        os.unlink(os.path.join(SNAP, n))
os.makedirs(SNAP, exist_ok=True)

# 1. fold the edit into the bf16 writers of the three files
n_fold, worst = 0, 0.0
for name in M["inputs"]:
    f, h, b = header(os.path.join(BASE, name)); meta = h.pop("__metadata__", None)
    out, off, parts = {}, 0, []
    for k in sorted(h):
        v = h[k]; s, e = v["data_offsets"]; f.seek(b + s); raw = f.read(e - s)
        if WRITERS.match(k) or k in ROWSPACE:
            assert v["dtype"] == "BF16", (k, v["dtype"])
            w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(v["shape"]).float()
            if k in ROWSPACE:
                w2 = torch.empty_like(w)
                for i in range(0, w.shape[0], 16384):
                    c = w[i:i + 16384]; w2[i:i + 16384] = c - torch.outer(c @ r, r)
                leak = float((w2.bfloat16().float() @ r).norm() / (w @ r).norm())
            else:
                w2 = w - torch.outer(r, r @ w)
                leak = float((r @ w2.bfloat16().float()).norm() / (r @ w).norm())
            rel = float((w2 - w).norm() / w.norm())
            assert 0.01 < rel < 0.03 and leak < 0.02, f"{k}: unexpected fold (rel change {rel:.4f}, leakage {leak:.4f})"
            worst = max(worst, leak); n_fold += 1
            raw = w2.bfloat16().contiguous().view(torch.uint8).numpy().tobytes()
        out[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + len(raw)]}; parts.append(raw); off += len(raw)
    if meta is not None:
        out["__metadata__"] = meta
    hdr = json.dumps(out, separators=(",", ":")).encode(); hdr += b" " * (-len(hdr) % 8)
    tmp = os.path.join(OUT, "blobs", name + ".part")
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hdr))); fo.write(hdr)
        for p in parts:
            fo.write(p)
    digest = sha256(tmp); os.replace(tmp, os.path.join(OUT, "blobs", digest))
    os.symlink(f"../../blobs/{digest}", os.path.join(SNAP, name))
    print(f"  {name}: rewritten ({off / 1e9:.2f} GB, blob {digest[:12]})", flush=True)
assert n_fold == 516, f"folded {n_fold} tensors, expected 516"

# 2. the direction (read by the orca-proj patch at runtime) + everything else as relative symlinks into the base snapshot
d = M["direction"]["sha256"]; dst = os.path.join(OUT, "blobs", d)
if not os.path.exists(dst):
    with open(os.path.join(HERE, M["direction"]["file"]), "rb") as fi, open(dst, "wb") as fo:
        fo.write(fi.read())
os.symlink(f"../../blobs/{d}", os.path.join(SNAP, "orca_direction.safetensors"))
for n in sorted(os.listdir(BASE)):
    if not os.path.lexists(os.path.join(SNAP, n)):
        os.symlink(f"../../../{base_repo_dir}/snapshots/{base_rev}/{n}", os.path.join(SNAP, n))
open(os.path.join(OUT, "refs", "main"), "w").write(M["version"])
print(f"  folded {n_fold} bf16 tensors (worst leakage after bf16 {worst:.4f}); snapshot {SNAP} on base {base_rev[:8]}")
