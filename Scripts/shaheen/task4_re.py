"""Task 4 for one opt level: <bc_root>/<rel>.bc -> <re_root>/<rel>.re (out of place).

Reuses task4_recompile.recompile_file (clang -m32/-m64 -c -fno-inline -fno-pic -fno-pie, bitness from
the dataset file name via task2_reoptimize.resolve_arch): every .re is an x86 object file, i386 for
32-bit sources and x86_64 for 64-bit ones, whatever the source architecture.
Must run where LLVM 18 is on PATH (the regraph SIF).

    python Scripts/shaheen/task4_re.py <bc_root> <re_root> --list shard_000.txt --workers 384
"""
import argparse
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import task4_recompile as t4  # noqa: E402
import utils  # noqa: E402


def convert(bc, re_path, timeout):
    t0 = time.time()
    t4.run_command = functools.partial(utils.run_command, timeout=timeout)
    arch = t4.resolve_arch(bc, "auto")
    if arch is None:
        return {"input": bc, "ok": False, "error": "cannot infer -m32/-m64 from the file name"}
    os.makedirs(os.path.dirname(re_path), exist_ok=True)
    marker = re_path + ".marker"
    ok, _, err = t4.recompile_file(bc, re_path, arch, marker)
    if os.path.exists(marker):
        os.remove(marker)  # state lives in the jsonl manifest instead
    rec = {"input": bc, "ok": bool(ok and os.path.exists(re_path) and os.path.getsize(re_path) > 0),
           "arch": arch, "secs": round(time.time() - t0, 1)}
    if not rec["ok"]:
        rec["error"] = (err or "")[-2000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bc_root")
    ap.add_argument("re_root")
    ap.add_argument("--list", required=True, help=".ll paths relative to the lift root (a shard list)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=int, default=28800, help="seconds per clang command")
    ap.add_argument("--manifest-tag", default="")
    args = ap.parse_args()

    with open(args.list) as f:
        rels = [l.strip()[: -len(".ll")] for l in f if l.strip()]
    jobs = []
    for rel in rels:
        bc = os.path.join(args.bc_root, rel + ".bc")
        re_path = os.path.join(args.re_root, rel + ".re")
        if not (os.path.exists(re_path) and os.path.getsize(re_path) > 0):
            jobs.append((bc, re_path))
    jobs.sort(key=lambda j: os.path.getsize(j[0]) if os.path.exists(j[0]) else 0, reverse=True)
    os.makedirs(args.re_root, exist_ok=True)
    manifest = os.path.join(args.re_root, f"_task4{args.manifest_tag}.jsonl")
    print(f"task4: {len(rels)} modules, {len(jobs)} to compile, {args.workers} workers", flush=True)

    t0, n_ok, n_fail = time.time(), 0, 0
    with open(manifest, "a") as out, ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(convert, bc, re_path, args.timeout) for bc, re_path in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_ok += rec["ok"]
            n_fail += not rec["ok"]
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ok={n_ok} fail={n_fail} {time.time() - t0:.0f}s", flush=True)
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
