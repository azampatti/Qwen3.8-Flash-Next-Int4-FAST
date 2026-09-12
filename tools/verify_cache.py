#!/usr/bin/env python3
"""Check the cached copy of a model against the hub, without downloading anything.

Every file in a Hugging Face cache is a symlink whose target is named after that file's sha256, so the
cache can be checked against what the hub currently serves with a single metadata call. This catches the
case a plain download cannot: a file that is present locally but is no longer the file the hub has, which
the downloader skips because, as far as its own bookkeeping goes, the revision is complete.

Prints one line per problem and a summary. Exit 0 = cache matches the hub, 2 = it does not, 1 = could not
check (no network, unknown repo). Run it inside the serving image, which ships huggingface_hub.

usage: verify_cache.py --repo <id> [--hf-home ~/.cache/huggingface]
"""
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    a = ap.parse_args()

    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(a.repo, files_metadata=True)
    except Exception as e:
        print(f"could not reach the hub: {str(e)[:160]}")
        return 1

    rev = info.sha
    snap = os.path.join(a.hf_home, "hub", "models--" + a.repo.replace("/", "--"), "snapshots", rev)
    print(f"hub revision: {rev[:12]}")
    if not os.path.isdir(snap):
        print(f"not cached at this revision (expected {snap})")
        return 2

    missing, stale, checked = [], [], 0
    for s in info.siblings:
        p = os.path.join(snap, s.rfilename)
        if not os.path.lexists(p):
            missing.append(s.rfilename)
            continue
        want = getattr(getattr(s, "lfs", None), "sha256", None)
        if not want:
            continue                      # small files are stored by git hash, nothing to compare
        checked += 1
        got = os.path.basename(os.path.realpath(p))
        if got != want:
            stale.append((s.rfilename, got, want))

    for f in missing:
        print(f"MISSING  {f}")
    for f, got, want in stale:
        print(f"STALE    {f}  (cached {got[:12]}, hub {want[:12]})")
    print(f"checked {checked} large files: {len(missing)} missing, {len(stale)} stale")
    return 2 if (missing or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
