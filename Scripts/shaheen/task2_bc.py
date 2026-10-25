"""Task 2 for one opt level: every <ll_root>/<rel>.ll -> <bc_root>/<rel>.bc (out of place).

Reuses task2_reoptimize.reoptimize_file (clang -O* with per-file -m32/-m64 and
-fno-inline/-fno-pic/-fno-pie; Oc = opt canonicalize passes with the
illegal-bitcast retry). O0 is only assembled with llvm-as, not run through clang.
Out of place so several opt levels can share one lifted .ll tree. Next to each X.bc it hardlinks
X.ll and X.ll.lift.json, where Task 3's segment filter (split_functions) expects them.
Must run where LLVM 18 is on PATH (the regraph SIF).

    python Scripts/shaheen/task2_bc.py <ll_root> <bc_root> --opt-level O2 --workers 384
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
import task2_reoptimize as t2  # noqa: E402
import utils  # noqa: E402


def link_sources(ll, bc):
    """Hardlink X.ll and X.ll.lift.json next to X.bc (same filesystem: no extra space or data inodes)."""
    for src, dst in ((ll, bc[:-3] + ".ll"), (ll + ".lift.json", bc[:-3] + ".ll.lift.json")):
        if os.path.exists(src) and not (os.path.exists(dst) and os.path.samefile(src, dst)):
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)


def convert(ll, bc, opt_level, timeout):
    t0 = time.time()
    rec = _convert(ll, bc, opt_level, timeout)
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def _convert(ll, bc, opt_level, timeout):
    t2.run_command = functools.partial(utils.run_command, timeout=timeout)
    os.makedirs(os.path.dirname(bc), exist_ok=True)
    tmp = bc + ".part"
    if opt_level == "-O0":
        ok, _, err = t2.run_command(["llvm-as", ll, "-o", tmp], f"Assembling {ll}")
        tool = "llvm-as"
    else:
        arch = t2.resolve_arch(ll, "auto")
        if arch is None:
            return {"input": ll, "ok": False, "error": "cannot infer -m32/-m64 from path or IR header"}
        marker = bc + ".marker"
        ok, _, err = t2.reoptimize_file(ll, tmp, opt_level, arch, marker)
        if os.path.exists(marker):
            os.remove(marker)  # state lives in the jsonl manifest instead
        tool = "opt" if t2.canonicalize_passes_for_opt_level(opt_level) else "clang"
    if ok and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, bc)
        return {"input": ll, "ok": True, "tool": tool}
    if os.path.exists(tmp):
        os.remove(tmp)
    return {"input": ll, "ok": False, "tool": tool, "error": (err or "")[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ll_root")
    ap.add_argument("bc_root")
    ap.add_argument("--opt-level", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=int, default=28800, help="seconds per clang/opt/llvm-as command")
    ap.add_argument("--list", help="only these .ll paths (relative to ll_root, one per line); default: all")
    ap.add_argument("--manifest-tag", default="", help="suffix for the jsonl manifest, e.g. a shard id")
    args = ap.parse_args()
    opt_level = utils.normalize_clang_opt_level(args.opt_level)

    if args.list:
        with open(args.list) as f:
            lls = [os.path.join(args.ll_root, l.strip()) for l in f if l.strip()]
    else:
        lls = [os.path.join(d, n) for d, _, files in os.walk(args.ll_root) for n in files if n.endswith(".ll")]
    jobs = []
    for ll in lls:
        bc = os.path.join(args.bc_root, os.path.relpath(ll, args.ll_root))[:-3] + ".bc"
        os.makedirs(os.path.dirname(bc), exist_ok=True)
        link_sources(ll, bc)
        if not (os.path.exists(bc) and os.path.getsize(bc) > 0):
            jobs.append((ll, bc))
    jobs.sort(key=lambda j: os.path.getsize(j[0]), reverse=True)  # big files first, shorter tail
    os.makedirs(args.bc_root, exist_ok=True)
    manifest = os.path.join(args.bc_root, f"_task2_{opt_level.lstrip('-')}{args.manifest_tag}.jsonl")
    print(f"task2 {opt_level}: {len(jobs)} .ll to convert, {args.workers} workers", flush=True)

    t0, n_ok, n_fail = time.time(), 0, 0
    with open(manifest, "a") as out, ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(convert, ll, bc, opt_level, args.timeout) for ll, bc in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_ok += rec["ok"]
            n_fail += not rec["ok"]
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ok={n_ok} fail={n_fail} {time.time() - t0:.0f}s", flush=True)
    return 0 if n_ok or not jobs else 1


if __name__ == "__main__":
    sys.exit(main())
