#!/bin/sh
# mtp_drafter_check.sh -- does this server lose speculative decoding (MTP) for one stream when several run in parallel?
#
# The bug ("dead drafter"): on vLLM + b12x builds serving Qwen3.8-Flash-Next with MTP, when the number of running requests is
# 3, 5, 6 or 7 (not 1, 2, 4, 8, 16) right after a new prompt was processed, one request stops accepting ANY drafted token and
# generates at 1 token per engine step until it finishes (3-5x slower), while its output stays correct. Details and fix:
# https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST  (commit 73c4fa0, "BUG FIX").
#
# What this does: one stream alone (baseline), then 3 streams started a moment apart, then 6. For every stream it counts the
# tokens delivered per streamed chunk (= per engine step). A stream stuck at ~1.0 while the baseline is well above 1 = FAIL.
#
# Needs: sh + python3 (standard library only). Run it against an IDLE server; it sends about 1-2 minutes of test traffic.
#   sh mtp_drafter_check.sh                         # http://localhost:8000
#   sh mtp_drafter_check.sh http://host:8000 -k KEY # other server / API key (or env API_KEY)
#   options: -m MODEL  -n TOKENS(400)  --quick (skip the 6-stream phase)  --no-color
# Exit code: 0 PASS, 1 FAIL, 2 could not test (no speculative decoding, or an error).
PY=$(command -v python3 || command -v python) || { echo "python3 is required" >&2; exit 2; }
exec "$PY" - "$@" <<'PYEOF'
import argparse, json, os, sys, threading, time
import urllib.request, urllib.error

ap = argparse.ArgumentParser(prog="mtp_drafter_check.sh", description="Check for the MTP dead-drafter bug under parallel streams.")
ap.add_argument("base", nargs="?", default=os.environ.get("VLLM_BASE", "http://localhost:8000"), help="server URL (default http://localhost:8000)")
ap.add_argument("-k", "--api-key", default=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
ap.add_argument("-m", "--model", default="", help="model id (default: the first one the server lists)")
ap.add_argument("-n", "--tokens", type=int, default=400, help="tokens generated per stream (default 400)")
ap.add_argument("--stagger", type=float, default=0.7, help="seconds between stream starts (default 0.7)")
ap.add_argument("--quick", action="store_true", help="skip the 6-stream phase")
ap.add_argument("--no-color", action="store_true")
a = ap.parse_args()

BASE = a.base.rstrip("/")
if BASE.endswith("/v1"):
    BASE = BASE[:-3]
TTY = sys.stdout.isatty()
COLOR = TTY and not a.no_color and not os.environ.get("NO_COLOR")
def c(code, s):
    return "\033[%sm%s\033[0m" % (code, s) if COLOR else str(s)
BOLD, DIM, RED, GREEN, YELLOW, CYAN, WHITE = "1", "2", "1;31", "1;32", "1;33", "36", "1;97"

def http(path, body=None, timeout=600):
    hdr = {"Content-Type": "application/json"}
    if a.api_key:
        hdr["Authorization"] = "Bearer " + a.api_key
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data, hdr), timeout=timeout)

def die(msg, code=2):
    print(c(RED, "ERROR: ") + msg)
    sys.exit(code)

# ---- server ------------------------------------------------------------------------------------------------------------------
try:
    models = json.loads(http("/v1/models", timeout=20).read().decode())["data"]
except Exception as e:
    die("cannot reach %s/v1/models (%s)" % (BASE, e))
MODEL = a.model or models[0]["id"]
CTX = next((m.get("max_model_len") for m in models if m.get("id") == MODEL), None)

def metrics():
    """vLLM's own speculative-decoding counters, if /metrics is reachable (optional, only shown as extra values)."""
    try:
        text = http("/metrics", timeout=10).read().decode()
    except Exception:
        return None
    d = {}
    for ln in text.splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        name, val = ln.rsplit(" ", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        if name.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            d["acc"] = d.get("acc", 0) + v
        elif name.startswith("vllm:spec_decode_num_drafts_total"):
            d["drafts"] = d.get("drafts", 0) + v
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total") and 'position="0"' in name:
            d["pos0"] = d.get("pos0", 0) + v
        elif name.startswith("vllm:num_requests_running"):
            d["running"] = d.get("running", 0) + v
    return d

TOPICS = ["the history of professional tennis since 1968", "how container ships changed world trade", "the life cycle of stars",
          "the invention and spread of the printing press", "how the Roman road network was built and used",
          "the development of weather forecasting", "the history of the bicycle", "how coral reefs form and why they matter"]
def prompt(i):
    return ("Write a long, detailed essay of at least 1500 words about %s. Use plain flowing prose only: no headings, no lists, "
            "no tables. (essay %d, %d)" % (TOPICS[i % len(TOPICS)], i + 1, time.time_ns() % 100000))

EXTRA = {"min_tokens": a.tokens, "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}

def body_for(text, n):
    b = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": 0, "stream": True}
    for k, v in EXTRA.items():
        b[k] = n if k == "min_tokens" else v
    return b

def preflight():
    """Find the request options this server accepts (min_tokens, thinking switch and usage-in-stream are vLLM extensions)."""
    for drop in (None, "min_tokens", "chat_template_kwargs", "stream_options"):
        if drop:
            EXTRA.pop(drop, None)
        try:
            r = http("/v1/chat/completions", body_for("Say hello.", 8), timeout=120)
            for _ in r:
                pass
            return
        except urllib.error.HTTPError as e:
            if e.code != 400:
                die("the server answered HTTP %d to a test request: %s" % (e.code, e.read()[:300].decode(errors="replace")))
        except Exception as e:
            die("test request failed: %s" % e)
    die("the server rejects streaming chat requests (HTTP 400)")

class Stream(object):
    def __init__(self, label, text):
        self.label, self.text = label, text
        self.pieces = []          # characters per streamed chunk (one chunk = one engine step for this request)
        self.tokens = None; self.finish = None; self.error = None; self.done = False
        self.t0 = self.t_first = self.t_last = None

    def run(self, n):
        self.t0 = time.time()
        try:
            r = http("/v1/chat/completions", body_for(self.text, n))
            for raw in r:
                ln = raw.decode(errors="replace").strip()
                if not ln.startswith("data:"):
                    continue
                ln = ln[5:].strip()
                if ln == "[DONE]":
                    break
                try:
                    ev = json.loads(ln)
                except ValueError:
                    continue
                u = ev.get("usage")
                if u and u.get("completion_tokens") is not None:
                    self.tokens = int(u["completion_tokens"])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or "") + (d.get("reasoning") or "")
                    if piece:
                        now = time.time()
                        self.t_first = self.t_first or now
                        self.t_last = now
                        self.pieces.append(len(piece))
                    if ch.get("finish_reason"):
                        self.finish = ch["finish_reason"]
        except urllib.error.HTTPError as e:
            self.error = "HTTP %d %s" % (e.code, e.read()[:160].decode(errors="replace"))
        except Exception as e:
            self.error = str(e)[:160]
        self.done = True

    def stats(self, n):
        steps = len(self.pieces); chars = sum(self.pieces)
        approx = False
        tokens = self.tokens
        if tokens is None:
            if self.finish == "length":
                tokens = n
            else:
                tokens = int(round(chars / 4.0)); approx = True
        if not steps or not tokens:
            return None
        k = steps if steps < 40 else max(20, int(steps * 0.4))          # the last 40 % of the steps
        tail_tokens = tokens * (sum(self.pieces[-k:]) / float(chars or 1))
        dur = (self.t_last - self.t_first) if self.t_last and self.t_first else 0
        return {"tokens": tokens, "approx": approx, "steps": steps, "tps": tokens / float(steps), "tail": tail_tokens / float(k),
                "toks": (tokens - 1) / dur if dur > 0 else 0.0, "ttft": (self.t_first or self.t0) - self.t0}

def progress(title, streams, n, force=False, state={"last": 0.0}):
    now = time.time()
    if not TTY and not force and now - state["last"] < 10:
        return
    state["last"] = now
    parts = []
    for s in streams:
        got = min(n, int(sum(s.pieces) / 4.0))
        mark = c(GREEN, "done") if s.done and not s.error else (c(RED, "error") if s.error else "%3d%%" % (100 * got // max(n, 1)))
        parts.append("%s %s" % (c(CYAN, s.label), mark))
    line = "  %s  %s" % (c(DIM, title), "  ".join(parts))
    if TTY:
        sys.stdout.write("\r\033[K" + line); sys.stdout.flush()
    else:
        print(line)

def run_phase(title, count, n):
    streams = [Stream("R%d" % (i + 1), prompt(i)) for i in range(count)]
    m0 = metrics()
    threads = []
    for i, s in enumerate(streams):
        t = threading.Thread(target=s.run, args=(n,)); t.daemon = True; t.start(); threads.append(t)
        t_next = time.time() + (a.stagger if i < count - 1 else 0)
        while time.time() < t_next:
            progress(title, streams[: i + 1], n); time.sleep(0.1)
    while any(t.is_alive() for t in threads):
        progress(title, streams, n); time.sleep(0.2)
    progress(title, streams, n, force=True)
    if TTY:
        sys.stdout.write("\r\033[K"); sys.stdout.flush()
    m1 = metrics()
    srv = None
    if m0 and m1 and m1.get("drafts", 0) - m0.get("drafts", 0) > 0:
        dr = m1["drafts"] - m0["drafts"]
        srv = {"len": 1 + (m1.get("acc", 0) - m0.get("acc", 0)) / dr,
               "pos0": (m1.get("pos0", 0) - m0.get("pos0", 0)) / dr if "pos0" in m1 else None}
    return streams, srv

# ---- run ---------------------------------------------------------------------------------------------------------------------
print(c(WHITE, "MTP dead-drafter check") + c(DIM, "  (parallel streams vs speculative decoding)"))
print("  server  %s" % c(CYAN, BASE))
print("  model   %s%s" % (c(CYAN, MODEL), c(DIM, "   context %s" % CTX) if CTX else ""))
m = metrics()
if m and m.get("running", 0) > 0:
    print(c(YELLOW, "  note    %d other request(s) are running on this server: they change the number of parallel streams and can "
                    "hide or trigger the condition. Best run on an idle server." % int(m["running"])))
print(c(DIM, "  checking which request options the server accepts ..."))
preflight()

N = a.tokens
plan = [("baseline: 1 stream", 1), ("3 parallel streams", 3)] + ([] if a.quick else [("6 parallel streams", 6)])
results = []
for idx, (title, count) in enumerate(plan):
    label = "phase %d/%d  %s" % (idx + 1, len(plan), title)
    streams, srv = run_phase(label, count, N)
    results.append((title, streams, srv))
    errs = [s for s in streams if s.error]
    print("  %s %s%s" % (c(GREEN, "done ") if not errs else c(RED, "error"), label,
                         c(RED, "   " + errs[0].error) if errs else ""))
    if idx == 0 and (errs or not streams[0].stats(N)):
        die("the baseline request failed, cannot test: %s" % (errs[0].error if errs else "no tokens received"))

base = results[0][1][0].stats(N)
b = base["tps"]
dead_thr = 1.0 + min(0.20, (b - 1.0) * 0.4)

print("")
print(c(WHITE, "Report"))
hdr = "  %-7s %7s %6s %12s %10s %8s %12s  %s" % ("stream", "tokens", "steps", "tokens/step", "last 40%", "tok/s", "first token", "status")
dead_total = 0; tested = 0; worst = None; phase_errors = 0
for title, streams, srv in results:
    print("")
    print("  " + c(BOLD, title))
    print(c(DIM, hdr))
    for s in streams:
        st = s.stats(N)
        if s.error or not st:
            phase_errors += 1
            print("  %-7s %s" % (s.label, c(RED, "request failed: %s" % (s.error or "no tokens"))))
            continue
        is_dead = b >= 1.2 and st["tail"] < dead_thr and len(streams) > 1
        if len(streams) > 1:
            tested += 1
            worst = st["tail"] if worst is None else min(worst, st["tail"])
            dead_total += 1 if is_dead else 0
        status = c(RED, "DEAD DRAFTER") if is_dead else c(GREEN, "ok")
        tps = "%.2f" % st["tps"]; tail = "%.2f" % st["tail"]
        print("  %-7s %6d%s %6d %s %s %8.1f %10.2f s  %s" % (
            s.label, st["tokens"], "~" if st["approx"] else " ", st["steps"],
            c(RED if is_dead else WHITE, "%12s" % tps), c(RED if is_dead else WHITE, "%10s" % tail), st["toks"], st["ttft"], status))
    if srv:
        print(c(DIM, "  server counters: mean accepted length %.2f%s" % (
            srv["len"], ", first drafted token accepted %.0f %%" % (100 * srv["pos0"]) if srv["pos0"] is not None else "")))

print("")
print(c(WHITE, "Summary"))
print("  tokens per step, one stream alone      %s" % c(CYAN, "%.2f" % b))
if worst is not None:
    print("  lowest tokens per step in parallel     %s   %s" % (c(RED if dead_total else CYAN, "%.2f" % worst),
                                                            c(DIM, "(end of the stream; a dead drafter sits at 1.00)")))
    print("  streams that lost speculative decoding %s" % c(RED if dead_total else GREEN, "%d of %d" % (dead_total, tested)))
print("")
if b < 1.2:
    print(c(YELLOW, "  RESULT: NOT APPLICABLE") + "  speculative decoding does not appear to be active on this server "
          "(%.2f tokens per step with a single stream), so there is nothing to test." % b)
    sys.exit(2)
if dead_total:
    print(c(RED, "  RESULT: FAIL") + "  %d of %d parallel streams fell to 1 token per step and stayed there." % (dead_total, tested))
    print("  This server has the dead-drafter bug: with 3, 5, 6 or 7 requests in flight one of them loses speculative decoding")
    print("  for the rest of its answer (text stays correct, speed drops 3-5x).")
    print("  Fix for the b12x recipes of " + c(CYAN, "github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST") + " (commit 73c4fa0):")
    print("    " + c("38;5;213", "git pull && ./eugr-setup.sh") + "        (Swift1.5: also " + c("38;5;213", "./swift15/setup.sh") + "), then restart the server.")
    sys.exit(1)
if phase_errors:
    print(c(YELLOW, "  RESULT: INCONCLUSIVE") + "  %d request(s) failed, so not every stream could be checked (see above). "
          "No dead drafter was seen in the streams that completed." % phase_errors)
    sys.exit(2)
note = "" if b >= 1.5 else c(YELLOW, "  (low acceptance on this workload, %.2f tokens per step: the margin is thin)" % b)
print(c(GREEN, "  RESULT: PASS") + "  every parallel stream kept speculative decoding (%d streams checked).%s" % (tested, note))
sys.exit(0)
PYEOF
