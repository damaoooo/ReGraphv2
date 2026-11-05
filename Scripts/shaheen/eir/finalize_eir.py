"""Coreutils-EIR-fused: coreutils_final_set (build_split.py, as the O0 level) -> test_final_set aligned to
Coreutils-O0-fused/test_final_set, plus o0_row_map.parquet, dropped.parquet and finalize_stats.json.

Every EIR sample was written by eir_b.py for one O0 test_final_set row under that row's own file name (<sha1>.ll) and
function_name, so the EIR set is the O0 set minus the rows without EIR, in the same order. binary_name and fva are
copied from the O0 row. o0_row_map: eir_row -> o0_row. dropped: every O0 row with no EIR row, with the reason
  mips64 | ppc | not_in_lift_list | revng_failed | <eir_b.py row status: addr_unmatched, llvm_dis_failed,
  purify_failed, graph_failed, sample_failed, processing_error> | wash_dropped
"""
import glob, json, os, shutil, sys
import datasets
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

O = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/Coreutils-EIR-fused"     # usage: finalize_eir.py [FUSED] [B status dir]
O0 = f"{DATA}/Coreutils-O0-fused/test_final_set/train_dataset_pool"
W = f"{DATA}/S11A"
src, dst = f"{O}/coreutils_final_set", f"{O}/test_final_set"

idx = pd.read_parquet(f"{DATA}/S11/o0_index/coreutils_test.parquet")      # o0_row, binary, function_name, fname, fva, arch
o0 = datasets.load_from_disk(O0)
o0_bin, o0_name, o0_fva = (o0.data.column(c).to_pylist() for c in ("binary_name", "function_name", "fva"))
assert len(idx) == len(o0) and (idx.o0_row.to_numpy() == np.arange(len(o0))).all()
key = {(b, f): r for r, b, f in zip(idx.o0_row, idx.binary, idx.fname)}

pool = datasets.load_from_disk(f"{src}/train_dataset_pool")
fp = pool.data.column("file_path").to_pylist()
binary = [p.split("/functions/", 1)[1].rsplit("_functions/", 1)[0] for p in fp]
rows = np.array([key[(b, os.path.basename(p))] for b, p in zip(binary, fp)])
names = pool.data.column("function_name").to_pylist()
assert all(names[k] == o0_name[r] for k, r in enumerate(rows)), "function_name differs from the O0 row"
assert all(binary[k] == o0_bin[r] for k, r in enumerate(rows))
pool = pool.add_column("binary_name", binary).add_column("fva", [o0_fva[r] for r in rows])
assert pool.column_names == o0.column_names, (pool.column_names, o0.column_names)

# reasons for the O0 rows without EIR
lift = {l.strip() for f in ("coreutils_lift_bins.txt", "coreutils_bindiff_extra_bins.txt") for l in open(f"{W}/pkg/pkgA/{f}") if l.strip()}
failed = set(open(f"{W}/work/revng_failed.txt").read().split())
bstat = {}
for f in glob.glob(f"{sys.argv[2] if len(sys.argv) > 2 else W + '/out_b/status'}/*.rows.jsonl"):
    for l in open(f):
        r = json.loads(l); bstat[r["o0_row"]] = r["status"]
have = set(rows.tolist())
drop = []
for r, b, n, a in zip(idx.o0_row, idx.binary, idx.function_name, idx.arch):
    if r in have:
        continue
    if a == "mips64": why = "mips64"
    elif a in ("ppc32", "ppc64"): why = "ppc"
    elif b in failed: why = "revng_failed"
    elif b not in lift: why = "not_in_lift_list"
    elif bstat.get(r, "ok") != "ok": why = bstat[r]
    else: why = "wash_dropped"
    drop.append((int(r), b, n, why))
dropped = pd.DataFrame(drop, columns=["o0_row", "binary_name", "function_name", "reason"])

# untruncated lengths from the fused parquet (before wash)
full = {}
for f in glob.glob(f"{O}/parquet/coreutils/*.parquet"):
    t = pq.read_table(f, columns=["file_path", "input_ids"])
    full.update(zip(t.column("file_path").to_pylist(), pc.list_value_length(t.column("input_ids")).to_pylist()))
untrunc = np.array([full[p] for p in fp])
stored = np.array(pc.list_value_length(pool.data.column("input_ids")).to_pylist())
flat = pc.list_flatten(pool.data.column("input_ids"))
unk = int(pc.sum(pc.equal(flat, 1)).as_py() or 0)

shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
pool.save_to_disk(f"{dst}/train_dataset_pool", num_proc=32)
for x in ("train_task_dataset", "train_positive_map.pkl"):
    (shutil.copytree if os.path.isdir(f"{src}/{x}") else shutil.copy2)(f"{src}/{x}", f"{dst}/{x}")
pd.DataFrame({"eir_row": np.arange(len(rows)), "o0_row": rows}).to_parquet(f"{O}/o0_row_map.parquet", index=False)
dropped.to_parquet(f"{O}/dropped.parquet", index=False)
stats = {
    "rows": len(pool), "builds": len(set(binary)), "columns": pool.column_names, "o0_rows": len(o0),
    "o0_rows_in_order": bool((np.diff(rows) > 0).all()), "dropped": len(dropped),
    "dropped_by_reason": dropped.reason.value_counts().to_dict(), "rows_plus_dropped": len(pool) + len(dropped),
    "max_stored_len": int(stored.max()), "truncated": int((untrunc > stored).sum()),
    "truncated_frac": round(float((untrunc > stored).mean()), 4), "untrunc_median": int(np.median(untrunc)),
    "untrunc_p95": int(np.percentile(untrunc, 95)), "unk_tokens": unk, "tokens": int(len(flat)),
    "empty_cfg": int(sum(len(x) == 0 for x in pool.data.column("cfg_graph").to_pylist())),
    "empty_ddg": int(sum(len(x) == 0 for x in pool.data.column("ddg_graph").to_pylist())),
    "anchors_with_positives": len(datasets.load_from_disk(f"{src}/train_task_dataset")),
}
json.dump(stats, open(f"{O}/finalize_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
