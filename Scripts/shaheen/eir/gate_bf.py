"""Gate numbers (9950X 2026-11-05) on the SAME rows: Coreutils-EIR-bodyfirst-fused/test_final_set vs the O0 rows it
maps to (o0_row_map) in Coreutils-O0-fused/test_final_set: share of rows with an empty CFG / DDG after the 2,048 cut,
<unk> share (id 1), untruncated token count (EIR: fused parquet before wash; O0: coreutils-O0-rell token_len, the same
tokenization) median and share > 2,048. Also restricted to the 37,762 ablation-pool rows. -> <fused>/gate.json"""
import glob, json, sys
import datasets, numpy as np, pandas as pd
import pyarrow.compute as pc, pyarrow.parquet as pq
import os
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

E = sys.argv[1]
o0 = datasets.load_from_disk(f"{DATA}/Coreutils-O0-fused/test_final_set/train_dataset_pool").data
eir = datasets.load_from_disk(f"{E}/test_final_set/train_dataset_pool").data
mp = pd.read_parquet(f"{E}/o0_row_map.parquet")
er, orow = mp.eir_row.to_numpy(), mp.o0_row.to_numpy()

def side(t, rows):
    t = t.table.take(rows)
    cfg = pc.list_value_length(t.column("cfg_graph")).to_numpy(); ddg = pc.list_value_length(t.column("ddg_graph")).to_numpy()
    flat = pc.list_flatten(t.column("input_ids"))
    return {"rows": len(rows), "cfg_empty": float((cfg == 0).mean()), "ddg_empty": float((ddg == 0).mean()),
            "unk_share": float(pc.sum(pc.equal(flat, 1)).as_py() / len(flat))}

full = {}
for f in glob.glob(f"{E}/parquet/coreutils/*.parquet"):
    t = pq.read_table(f, columns=["file_path", "input_ids"])
    full.update(zip(t.column(0).to_pylist(), pc.list_value_length(t.column(1)).to_pylist()))
e_untr = np.array([full[p] for p in eir.column("file_path").to_pylist()])
rl = pd.concat([pd.read_parquet(f, columns=["binary_name", "function_name", "token_len"]) for f in glob.glob(f"{DATA}/coreutils-O0-rell/data/*.parquet")])
tl = dict(zip(zip(rl.binary_name, rl.function_name), rl.token_len))
ob, on = o0.column("binary_name").to_pylist(), o0.column("function_name").to_pylist()
o_untr = np.array([tl[(ob[r], on[r])] for r in orow])

idx = pd.read_parquet(f"{DATA}/S11/o0_index/coreutils_test.parquet")
pool = pd.read_csv(f"{DATA}/S11A/pkg/pkgA/coreutils_pool_functions_nonmips64.tsv", sep="\t")
pkey = set(zip(pool.bin, pool.fva.map(lambda x: int(x, 16))))
in_pool = np.array([(b, int(f)) in pkey for b, f in zip(idx.binary.to_numpy()[orow], idx.fva.to_numpy()[orow])])

out = {}
for name, sel in (("all_mapped_rows", np.ones(len(er), bool)), ("ablation_pool_rows", in_pool)):
    a, b = side(eir, er[sel]), side(o0, orow[sel])
    a.update(untrunc_median=float(np.median(e_untr[er[sel]])), untrunc_over_2048=float((e_untr[er[sel]] > 2048).mean()))
    b.update(untrunc_median=float(np.median(o_untr[sel])), untrunc_over_2048=float((o_untr[sel] > 2048).mean()))
    out[name] = {"EIR_bodyfirst": a, "O0": b}
json.dump(out, open(f"{E}/gate.json", "w"), indent=1); print(json.dumps(out, indent=1))
