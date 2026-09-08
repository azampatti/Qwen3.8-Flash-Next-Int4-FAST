#!/usr/bin/env python3
"""Create the MTP draft directory: the same weights, one config value changed.

The MTP draft layer reads `num_experts_per_tok` from the model directory vLLM is told to load the
draft from. Pointing it at a directory whose config says 10 makes the draft route over 10 experts
while the model itself still runs at 5 -- the draft is the same tensors either way, so this costs
no extra disk (everything except config.json is a symlink) and measurably improves how many drafted
tokens the model accepts.

  usage: make_draft_dir.py <model dir> [draft dir]      (default: <model dir>-draft-k10)
"""
import json, os, sys, shutil
src = os.path.abspath(sys.argv[1].rstrip("/"))
dst = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else src + "-draft-k10"
cfg = json.load(open(os.path.join(src, "config.json")))
t = cfg.get("text_config", cfg)
if t["num_experts_per_tok"] == 10:
    sys.exit(f"{src} already routes 10 experts -- no draft directory needed")
if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
for name in sorted(os.listdir(src)):
    if name == "config.json":
        continue
    os.symlink(os.path.relpath(os.path.join(src, name), dst), os.path.join(dst, name))
t["num_experts_per_tok"] = 10
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print(f"draft directory ready: {dst}  ({len(os.listdir(dst)) - 1} symlinks + its own config.json, top-k 10)")
