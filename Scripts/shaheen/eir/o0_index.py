"""Index of the O0 final-set rows that the EIR data must follow: o0_row, binary (dataset path key), function_name,
file name (<sha1>.ll), split, fva (IDA ea from the lift state, llvm_name -> ea; coreutils has fva in the pool).
Writes S11/o0_index/<group>.parquet for dataset1_{train,validation,test} and coreutils_test."""
import json, os, sys
from collections import defaultdict
import datasets, pandas as pd
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
S = f"{DATA}"; OUT = f"{S}/S11/o0_index"; os.makedirs(OUT, exist_ok=True)
SETS = {"dataset1_train": (f"{S}/Dataset-1-O0-fused/train_final_set", f"{S}/Dataset-1-ll"),
        "dataset1_validation": (f"{S}/Dataset-1-O0-fused/validation_final_set", f"{S}/Dataset-1-ll"),
        "dataset1_test": (f"{S}/Dataset-1-O0-fused/test_final_set", f"{S}/Dataset-1-ll"),
        "coreutils_test": (f"{S}/Coreutils-O0-fused/test_final_set", f"{S}/Coreutils-ll")}
for g, (fs, ll) in SETS.items():
    pool = datasets.load_from_disk(f"{fs}/train_dataset_pool")
    fp = pool.data.column("file_path").to_pylist(); names = pool.data.column("function_name").to_pylist()
    rel = [p.split("/functions/", 1)[1] for p in fp]                      # <split>/<proj>/<bin>_functions/<sha1>.ll or coreutils/<bin>_functions/...
    binary = [r.rsplit("_functions/", 1)[0] for r in rel]
    fname = [r.rsplit("/", 1)[1] for r in rel]
    ea = {}
    for b in sorted(set(binary)):
        for r in json.load(open(f"{ll}/{b}.ll.lift.json"))["functions"]:
            if r.get("status") == "lifted" and r.get("llvm_name"):
                ea.setdefault((b, r["llvm_name"]), []).append(int(r["ea"], 16))
    fva = [ea[k][0] if len(ea.get(k, [])) == 1 else None for k in zip(binary, names)]
    df = pd.DataFrame({"o0_row": range(len(fp)), "binary": binary, "function_name": names, "fname": fname, "fva": fva,
                       "arch": [b.rsplit("/", 1)[-1].split("-")[0] for b in binary]})
    df.to_parquet(f"{OUT}/{g}.parquet")
    print(g, len(df), "binaries", df.binary.nunique(), "fva missing", df.fva.isna().sum(), "mips64 rows", (df.arch == "mips64").sum(), flush=True)
