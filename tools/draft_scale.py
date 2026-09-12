#!/usr/bin/env python3
"""Build a scale-aware copy of the serving image's OWN speculative-decoding module.

Nothing of vLLM is vendored in this repository. This reads
`vllm/models/qwen3_8_flash_next/nvidia/mtp.py` out of the image you are about to run, changes the single
line that builds the draft head's LogitsProcessor so it honours $VLLM_MTP_DRAFT_SCALE, and writes the
result to --out. serve.sh then mounts that one file over the module inside the container.

Why: multiplying the draft's logits before the acceptance test makes the draft commit to its best guess
instead of hedging. Verification still uses the target's own distribution, so outputs are unchanged; it
only raises how often a drafted token survives. Measured on our 60-row probe at temperature 0.5:
+2.5pp acceptance overall, +3.4pp on prose, about +4% tokens per second.

usage: draft_scale.py --image qwen38-flash-dgx:a5b-int4 --out ~/models/.draft-scale/mtp.py
Writes <out>, plus `module_path` and `image_id` beside it so serve.sh knows where to mount it and can
tell when the image changed. Exits non-zero (with a reason) if the expected line is not found.
"""
import argparse, os, subprocess, sys

ORIG = "        self.logits_processor = LogitsProcessor(config.vocab_size)"
PATCH = '''        # Draft-logit scaling, added by tools/draft_scale.py. The draft's logits are multiplied by
        # VLLM_MTP_DRAFT_SCALE before sampling and before the acceptance test (scale 2 behaves like a
        # draft temperature of 0.25 when you serve at 0.5). Exact: the verifier still uses the target
        # model's distribution, so the text you get out does not change.
        import os as _os
        _draft_scale = float(_os.environ.get("VLLM_MTP_DRAFT_SCALE", "1.0"))
        if _draft_scale != 1.0:
            from vllm.logger import init_logger as _il
            _il(__name__).warning("MTP draft logits scale = %.3f (VLLM_MTP_DRAFT_SCALE)", _draft_scale)
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=_draft_scale)'''


def run(args, what):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"draft_scale: could not {what}: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    find = ('f=$(find /usr/local/lib /usr/lib -path "*qwen3_8_flash_next/nvidia/mtp.py" 2>/dev/null | head -1);'
            ' [ -n "$f" ] && echo "$f"')
    path = run(["docker", "run", "--rm", "--entrypoint", "sh", a.image, "-c", find], "look inside " + a.image).strip()
    if not path:
        sys.exit("draft_scale: this image has no qwen3_8_flash_next MTP module -- leave DRAFT_SCALE=1")
    src = run(["docker", "run", "--rm", "--entrypoint", "cat", a.image, path], "read " + path)

    if src.count(ORIG) != 1:
        sys.exit(f"draft_scale: expected exactly one '{ORIG.strip()}' in {path}, found {src.count(ORIG)}. "
                 "The image changed; leave DRAFT_SCALE=1 or update this patcher.")
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(src.replace(ORIG, PATCH))
    d = os.path.dirname(out)
    open(os.path.join(d, "module_path"), "w").write(path + "\n")
    image_id = run(["docker", "image", "inspect", "-f", "{{.Id}}", a.image], "inspect " + a.image).strip()
    open(os.path.join(d, "image_id"), "w").write(image_id + "\n")
    print(f"draft-scale module ready: {out}  (mounts over {path})")


if __name__ == "__main__":
    main()
