"""Extra gate numbers for the noisa set (9950X 2026-11-05 11:44), after gate_bf.py: residual code_?(arm|thumb|aarch64|
mips|x86) matches (normalized text and decoded untruncated input_ids, from the eir_bf.py row statuses), the share of rows
whose whole function body fits in the first 2,048 tokens (body_tokens incl. bos/eos <= 2,048), on all rows and on the
ablation-pool rows, and a per-row comparison of the CFG / DDG edge counts with the delivered bodyfirst fused (same
file name). -> <fused>/gate_noisa.json"""
import glob, json, sys
import numpy as np, pandas as pd
import pyarrow.compute as pc, pyarrow.parquet as pq
import os
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

E, OLD, ST = sys.argv[1], sys.argv[2], sys.argv[3]
rows = pd.DataFrame([json.loads(l) for f in glob.glob(f"{ST}/*.rows.jsonl") for l in open(f)])
ok = rows[rows.status == "ok"]
idx = pd.read_parquet(f"{DATA}/S11/o0_index/coreutils_test.parquet").set_index("o0_row")
pool = pd.read_csv(f"{DATA}/S11A/pkg/pkgA/coreutils_pool_functions_nonmips64.tsv", sep="\t")
pkey = set(zip(pool.bin, pool.fva.map(lambda x: int(x, 16))))
inpool = np.array([(idx.at[r, "binary"], int(idx.at[r, "fva"])) in pkey for r in ok.o0_row])
bt = ok.body_tokens.to_numpy(float)
def fits(sel):
    v = bt[sel]; return {"rows": int(sel.sum()), "body_fits_2048": float((v <= 2048).mean()), "body_tokens_median": float(np.median(v))}
out = {"rows_ok": len(ok), "residual_text": int(ok.residual_text.sum()), "residual_ids": int(ok.residual_ids.sum()),
       "rows_with_residual": int(((ok.residual_text > 0) | (ok.residual_ids > 0)).sum()),
       "all_rows": fits(np.ones(len(ok), bool)), "ablation_pool_rows": fits(inpool)}
eq = n = 0
for f in glob.glob(f"{E}/parquet/coreutils/*.parquet"):
    a = pq.read_table(f, columns=["file_path", "cfg_graph", "ddg_graph"])
    b = pq.read_table(f.replace(E, OLD), columns=["file_path", "cfg_graph", "ddg_graph"])
    da = {p.rsplit("/", 1)[1]: (c, d) for p, c, d in zip(a.column(0).to_pylist(), pc.list_value_length(a.column(1)).to_pylist(), pc.list_value_length(a.column(2)).to_pylist())}
    db = {p.rsplit("/", 1)[1]: (c, d) for p, c, d in zip(b.column(0).to_pylist(), pc.list_value_length(b.column(1)).to_pylist(), pc.list_value_length(b.column(2)).to_pylist())}
    n += len(da); eq += sum(da[k] == db.get(k) for k in da)
out["graph_edge_counts_equal_to_bodyfirst"] = {"rows": n, "equal": eq}
json.dump(out, open(f"{E}/gate_noisa.json", "w"), indent=1); print(json.dumps(out, indent=1))
