"""Per-function IDA facts of ONE .i64 for the lift-coverage denominators: segment, thunk flag, basic blocks.

For every idautils.Functions() entry: ea, name, segment (ida_segment.get_segm_name), thunk (func.flags & FUNC_THUNK),
bb_num (len(FlowChart(func)), the Dataset-1 Step 1 count), size. The database is opened without re-analysis and not
saved. Publishes <FF_OUT_ROOT>/<rel>.parquet and prints one JSON line (driver.py contract).
Env: FF_IN_ROOT (the .i64 tree), FF_OUT_ROOT.
"""
import idapro  # must be the first import

import json
import os
import shutil
import sys
import time


def main():
    src = sys.argv[1]
    rel = os.path.relpath(src, os.environ["FF_IN_ROOT"])[: -len(".i64")]
    dst = os.path.join(os.environ["FF_OUT_ROOT"], rel + ".parquet")
    rec = {"input": src, "output": dst}
    if os.path.exists(dst):
        rec.update(ok=True, skipped="exists")
        print(json.dumps(rec))
        return 0
    t0 = time.time()
    local = os.path.join(os.getcwd(), os.path.basename(src))
    shutil.copyfile(src, local)  # opening unpacks the database next to it: keep that on /tmp
    if idapro.open_database(local, False) != 0:
        rec.update(ok=False, error="open_database failed")
        print(json.dumps(rec))
        return 1
    import ida_funcs
    import ida_gdl
    import ida_segment
    import idautils

    rows = []
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        seg = ida_segment.getseg(ea)
        rows.append((ea, ida_funcs.get_func_name(ea), ida_segment.get_segm_name(seg) if seg else None,
                     bool(f.flags & ida_funcs.FUNC_THUNK), len(list(ida_gdl.FlowChart(f))), f.end_ea - f.start_ea))
    idapro.close_database(False)

    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["ea", "name", "segment", "thunk", "bb_num", "size"]
    table = pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(table, dst + ".part")
    os.replace(dst + ".part", dst)
    rec.update(ok=True, functions=len(rows), thunks=sum(r[3] for r in rows), secs=round(time.time() - t0, 1))
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
