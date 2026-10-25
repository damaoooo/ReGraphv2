"""Lift ONE binary to LLVM IR with ida2llvm.py; print one JSON line (driver.py contract).

Run by /scratch/zhoul0e/iops/projects/ida-batch/driver.py with cwd = a private
node-local temp dir. Output mirrors task1_lift.py: <LIFT_OUT_ROOT>/<rel dir>/<name>.ll plus the lifter's <name>.ll.lift.json,
which Task 3 needs next to the module (segment filter, bound to the .ll by checksum).
Input may be a binary or <name>.i64. ida2llvm verifies with llvm-as: run inside the LLVM 18 SIF.
Env: LIFT_IN_ROOT, LIFT_OUT_ROOT, optional LIFT_TARGET (passed as ida2llvm --target; unset = its default).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

LIFTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ida2llvm.py")


def main():
    src = sys.argv[1]
    in_root = os.environ["LIFT_IN_ROOT"]
    out_root = os.environ["LIFT_OUT_ROOT"]
    rel = os.path.relpath(src, in_root)
    if rel.endswith(".i64"):
        rel = rel[: -len(".i64")]  # x.i64 -> x.ll, as task1_lift.py names it
    dst = os.path.join(out_root, rel + ".ll")
    dst_state = dst + ".lift.json"
    rec = {"input": src, "output": dst}
    if os.path.exists(dst) and os.path.exists(dst_state):  # the state file is published last
        rec.update(ok=True, skipped="exists")
        print(json.dumps(rec))
        return 0

    t0 = time.time()
    local = os.path.join(os.getcwd(), os.path.basename(src))
    shutil.copyfile(src, local)  # IDA writes its database next to the input: keep it on /tmp
    local_ll = os.path.join(os.getcwd(), os.path.basename(dst))
    local_state = local_ll + ".lift.json"
    cmd = [sys.executable, LIFTER, "-f", local, "-o", local_ll]
    if os.environ.get("LIFT_TARGET"):
        cmd += ["--target", os.environ["LIFT_TARGET"]]  # unset: ida2llvm's default (source)
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"lift stats: (.*)", p.stderr)
    if m:  # "status=ok defined=12 failed=0 approx=0 ... unsupported=none no_decompile=3"
        rec["stats"] = {k: int(v) if v.isdigit() else v for k, v in (kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)}
    rec["secs"] = round(time.time() - t0, 2)
    if p.returncode != 0 or not os.path.exists(local_ll) or not os.path.exists(local_state):
        rec.update(ok=False, error=f"exit {p.returncode}", log=p.stderr[-2000:])
        print(json.dumps(rec))
        return 1

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for local_file, final in ((local_ll, dst), (local_state, dst_state)):  # state last: it marks completion
        shutil.copyfile(local_file, final + ".part")
        os.replace(final + ".part", final)
    rec["ok"] = True
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
