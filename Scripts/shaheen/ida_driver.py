"""Run analyze.py over one shard of an input list with W parallel idalib processes.

One process per binary (crash/timeout isolation). Results are appended to
<out>/shard_<k>.jsonl by this single process, so the output is a handful of
files rather than one per binary. Inputs already present in the shard's
jsonl (ok or failed) are skipped, so resubmitting the same array resumes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def done_inputs(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["input"])
                except (ValueError, KeyError):
                    pass  # torn last line from a killed job
    return done


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def run_one(src, args, workroot):
    wd = tempfile.mkdtemp(dir=workroot)
    try:
        p = subprocess.run(
            [sys.executable, args.analyze, src],
            cwd=wd, capture_output=True, text=True, timeout=args.timeout,
        )
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode == 0 and lines:
            return lines[-1]
        err = (p.stderr or p.stdout)[-2000:]
        return json.dumps({"input": src, "ok": False, "error": f"exit {p.returncode}", "log": err})
    except subprocess.TimeoutExpired:
        return json.dumps({"input": src, "ok": False, "error": f"timeout {args.timeout}s"})
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="text file, one input path per line")
    ap.add_argument("--out", required=True, help="output dir (on /scratch)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per binary")
    ap.add_argument("--analyze", default=os.path.join(HERE, "analyze.py"))
    ap.add_argument("--workroot", default=os.environ.get("TMPDIR", "/tmp"))
    args = ap.parse_args()

    with open(args.list) as f:
        inputs = [l.strip() for l in f if l.strip()]
    shard = inputs[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"shard_{args.shard:05d}.jsonl")
    done = done_inputs(out_path)
    # Biggest first: a large binary can take minutes, so starting it last leaves a long tail.
    todo = sorted((s for s in shard if s not in done), key=_size, reverse=True)
    print(f"shard {args.shard}/{args.nshards}: {len(shard)} inputs, {len(done)} done, {len(todo)} to run, "
          f"{args.workers} workers, timeout {args.timeout}s", flush=True)

    lock = threading.Lock()
    n_ok = n_fail = 0
    t0 = time.time()
    with open(out_path, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run_one, s, args, args.workroot) for s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            line = fut.result()
            ok = json.loads(line).get("ok", False)
            with lock:
                out.write(line + "\n")
                out.flush()
            n_ok += ok
            n_fail += not ok
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ok={n_ok} fail={n_fail} {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
