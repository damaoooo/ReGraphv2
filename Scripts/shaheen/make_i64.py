"""Build the IDA database (.i64) for ONE binary with idalib; print one JSON line (driver.py contract).

Equivalent of task1_lift.py step 1 (`idat -A -B`), without idat. Run by
/scratch/zhoul0e/iops/projects/ida-batch/driver.py with cwd = a private node-local
temp dir. Output mirrors the input tree: <I64_OUT_ROOT>/<rel path>.i64
Env: I64_IN_ROOT, I64_OUT_ROOT.
"""
import idapro  # must be the first import

import json
import os
import shutil
import sys
import time


def main():
    src = sys.argv[1]
    rel = os.path.relpath(src, os.environ["I64_IN_ROOT"])
    dst = os.path.join(os.environ["I64_OUT_ROOT"], rel + ".i64")
    rec = {"input": src, "output": dst}
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        rec.update(ok=True, skipped="exists")
        print(json.dumps(rec))
        return 0

    t0 = time.time()
    local = os.path.join(os.getcwd(), os.path.basename(src))
    shutil.copyfile(src, local)  # IDA writes .id0/.id1/... next to the input: keep them on /tmp
    rc = idapro.open_database(local, True)
    if rc != 0:
        rec.update(ok=False, error=f"open_database rc={rc}")
        print(json.dumps(rec))
        return 1
    import ida_auto
    import idautils

    ida_auto.auto_wait()
    rec["n_funcs"] = sum(1 for _ in idautils.Functions())
    idapro.close_database(True)  # packs the database into <local>.i64
    local_i64 = local + ".i64"
    rec["secs"] = round(time.time() - t0, 2)
    if not os.path.exists(local_i64) or os.path.getsize(local_i64) == 0:
        rec.update(ok=False, error="no .i64 written")
        print(json.dumps(rec))
        return 1

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(local_i64, dst + ".part")
    os.replace(dst + ".part", dst)  # never leave a truncated .i64 that resume would skip
    rec.update(ok=True, size=os.path.getsize(dst))
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
