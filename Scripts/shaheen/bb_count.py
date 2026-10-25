"""Basic-block count of every function in ONE .i64, as the Dataset-1 benchmark selects functions.

Same count and opcode hash (Scripts/ida_flowchart.py) as Cisco-Talos binary_function_similarity IDA_scripts/IDA_flowchart (Marcelli et al., USENIX
Security '22): len(list(idaapi.FlowChart(func))) for each idautils.Functions() entry; that benchmark keeps
functions with at least five. The lift state X.ll.lift.json maps each IDA address to the LLVM name our
samples carry (function_map's OriginalFunctionName). Publishes <BB_OUT_ROOT>/<rel>.parquet
(ea, ida_name, llvm_name, bb_num, hashopcodes); prints one JSON line (driver.py contract).
Env: BB_IN_ROOT (the .i64 tree), BB_OUT_ROOT, BB_LIFT_ROOT (the .ll tree holding X.ll.lift.json).
"""
import idapro  # must be the first import

import json
import os
import shutil
import sys
import time


def main():
    src = sys.argv[1]
    rel = os.path.relpath(src, os.environ["BB_IN_ROOT"])[: -len(".i64")]
    dst = os.path.join(os.environ["BB_OUT_ROOT"], rel + ".parquet")
    rec = {"input": src, "output": dst}
    if os.path.exists(dst):
        rec.update(ok=True, skipped="exists")
        print(json.dumps(rec))
        return 0

    t0 = time.time()
    with open(os.path.join(os.environ["BB_LIFT_ROOT"], rel + ".ll.lift.json")) as f:
        llvm_name = {int(r["ea"], 16): r.get("llvm_name") for r in json.load(f)["functions"]}
    local = os.path.join(os.getcwd(), os.path.basename(src))
    shutil.copyfile(src, local)  # opening unpacks the database next to it: keep that on /tmp
    if idapro.open_database(local, False) != 0:  # already analyzed; no auto-analysis
        rec.update(ok=False, error="open_database failed")
        print(json.dumps(rec))
        return 1
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from ida_flowchart import function_rows

    rows = [(r["fva"], r["name"], llvm_name.get(r["fva"]), r["bb_num"], r["hashopcodes"]) for r in function_rows()]
    idapro.close_database(False)

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"ea": [r[0] for r in rows], "ida_name": [r[1] for r in rows],
                      "llvm_name": [r[2] for r in rows], "bb_num": [r[3] for r in rows], "hashopcodes": [r[4] for r in rows]})
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(table, dst + ".part")
    os.replace(dst + ".part", dst)
    rec.update(ok=True, functions=len(rows), ge5=sum(r[3] >= 5 for r in rows),
               unmapped=sum(r[2] is None for r in rows), secs=round(time.time() - t0, 1))
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
