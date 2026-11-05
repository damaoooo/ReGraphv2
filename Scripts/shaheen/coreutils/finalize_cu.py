"""copy of $REGRAPH_DATA/Coreutils-Oc-fused/finalize.py, one level per run (usage: finalize_cu.py OPT).
coreutils_final_set -> test_final_set with binary_name and fva added, plus the NOTES statistics.

binary_name = coreutils/<binary> from file_path (.../functions/coreutils/<binary>_functions/<sha1>.ll);
fva = the IDA ea of the function from the lift state (llvm_name -> ea), as in coreutils-Oc-rell.
Untruncated lengths come from the fused parquet (input_ids before wash), to report the truncation rate.
"""
import glob
import json
import os
import shutil
import sys

import datasets
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

OPT = sys.argv[1]
O = f"{DATA}/Coreutils-{OPT}-fused"
LIFT = f"{DATA}/Coreutils-ll"
RELL = f"{DATA}/coreutils-{OPT}-rell"
src = f"{O}/coreutils_final_set"
dst = f"{O}/test_final_set"

pool = datasets.load_from_disk(f"{src}/train_dataset_pool")
fp = pool.data.column("file_path").to_pylist()
binary = [p.split("/functions/", 1)[1].rsplit("_functions/", 1)[0] for p in fp]
names = pool.data.column("function_name").to_pylist()
ea = {}
for b in sorted(set(binary)):
    for r in json.load(open(f"{LIFT}/{b}.ll.lift.json"))["functions"]:
        if r.get("status") == "lifted" and r.get("llvm_name"):
            ea.setdefault((b, r["llvm_name"]), []).append(r["ea"])
fva = [ea[k][0] if len(ea.get(k, [])) == 1 else None for k in zip(binary, names)]
pool = pool.add_column("binary_name", binary).add_column("fva", fva)

# untruncated lengths: the fused parquet before wash, keyed by file_path
full = {}
for f in glob.glob(f"{O}/parquet/coreutils/*.parquet"):
    t = pq.read_table(f, columns=["file_path", "input_ids"])
    full.update(zip(t.column("file_path").to_pylist(), pc.list_value_length(t.column("input_ids")).to_pylist()))
untrunc = np.array([full[p] for p in fp])
stored = np.array(pc.list_value_length(pool.data.column("input_ids")).to_pylist())
flat = pc.list_flatten(pool.data.column("input_ids"))
unk_tokens = int(pc.sum(pc.equal(flat, 1)).as_py() or 0)  # <unk> = id 1 in llvm_ir_bpe.json

# rows of coreutils-Oc-rell (all lifted functions incl. __halt) that this set covers
rell = set()
for f in glob.glob(f"{RELL}/data/*.parquet"):
    t = pq.read_table(f, columns=["binary_name", "function_name"])
    rell.update(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))
keys = set(zip(binary, names))

shutil.rmtree(dst, ignore_errors=True)
os.makedirs(dst)
pool.save_to_disk(f"{dst}/train_dataset_pool", num_proc=32)
for x in ("train_task_dataset", "train_positive_map.pkl"):
    (shutil.copytree if os.path.isdir(f"{src}/{x}") else shutil.copy2)(f"{src}/{x}", f"{dst}/{x}")
stats = {
    "rows": len(pool), "builds": len(set(binary)), "columns": pool.column_names,
    "matched_to_rell": len(keys & rell), "rell_rows": len(rell), "rell_not_here": len(rell - keys),
    "rell_not_here_halt": sum(1 for b, n in rell - keys if n == "__halt"),
    "here_not_in_rell": len(keys - rell), "duplicate_keys": len(binary) - len(keys),
    "fva_missing": sum(v is None for v in fva),
    "max_stored_len": int(stored.max()), "truncated": int((untrunc > stored).sum()),
    "truncated_frac": round(float((untrunc > stored).mean()), 4), "untrunc_over_2048": int((untrunc > 2048).sum()),
    "untrunc_median": int(np.median(untrunc)), "untrunc_p95": int(np.percentile(untrunc, 95)),
    "bos_first": float(np.mean([ids[0] == 2 for ids in pool.data.column("input_ids").slice(0, 5000).to_pylist()])),
    "eos_last": float(np.mean([ids[-1] == 3 for ids in pool.data.column("input_ids").slice(0, 5000).to_pylist()])),
    "unk_tokens": unk_tokens, "tokens": int(len(flat)), "unk_frac": float(unk_tokens / max(len(flat), 1)),
    "anchors_with_positives": len(datasets.load_from_disk(f"{src}/train_task_dataset")),
}
json.dump(stats, open(f"{O}/finalize_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
