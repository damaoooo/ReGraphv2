"""EIR irnorm summary: mean ± std (n, median) of the normalized token edit distance for O0, Oc and EIR on the SAME
pairs (make_pairs_eir.py subset); XA, XC from v2, XO, XM from v1, NEG from both. No AUC (request S11 3.5).
usage: summarize_eir.py <dir with v1/ v2/ (pairs) and dist_v1/ dist_v2/> <out>
"""
import json, sys
import numpy as np
import pandas as pd

d, out = sys.argv[1], sys.argv[2]
LEVELS, FORMS = ["O0", "Oc", "EIRmain", "EIR"], [("normkd_edit", "normkd (declare lines kept; the rule asked for)"), ("norm_edit", "norm (agreed rules)")]
COLS = [("XA", "v2"), ("XO", "v1"), ("XC", "v2"), ("XM", "v1"), ("NEG", "v1"), ("NEG", "v2")]
table, md = {}, ["# Normalized-IR edit distance on the EIR pairs, coreutils", "",
                 "Same pairs for all three levels: distance subsets of the irsim (v1) / v2 runs restricted to pairs whose two functions "
                 "are non-mips64 and both have an EIR text. Cells: mean ± std of the normalized token Levenshtein (0 = identical); "
                 "no AUC. XA, XC: v2 pairs; XO, XM: irsim (v1) pairs.", ""]
LABEL = {"EIRmain": "EIR (text_main, main)", "EIR": "EIR (text_b, comparison)"}
data = {}
for r in ("v1", "v2"):
    p = pd.read_parquet(f"{d}/{r}/pairs_hash.parquet", columns=["pair", "kind"])
    for L in LEVELS:
        data[r, L] = p.merge(pd.read_parquet(f"{d}/dist_{r}/dist_norm_{L}.parquet"), on="pair", how="left")
for col, title in FORMS:
    md += [f"## {title}", "", "| level | " + " | ".join(f"{k} ({r})" if k == "NEG" else k for k, r in COLS) + " | missing |",
           "|---" * (len(COLS) + 2) + "|"]
    for L in LEVELS:
        cells, miss = [], 0
        for k, r in COLS:
            x = data[r, L]; v = x.loc[x.kind == k, col].to_numpy(float); n_all = len(v); v = v[~np.isnan(v)]
            miss += n_all - len(v)
            e = {"n": int(len(v)), "missing": int(n_all - len(v)), "mean": float(v.mean()) if len(v) else None,
                 "std": float(v.std()) if len(v) else None, "median": float(np.median(v)) if len(v) else None}
            table.setdefault(col, {}).setdefault(L, {})[f"{k}_{r}"] = e
            cells.append(f"{e['mean']:.3f} ± {e['std']:.3f}" if len(v) else "-")
        md.append(f"| {LABEL.get(L, L)} | " + " | ".join(cells) + f" | {miss} |")
    md.append("")
md += ["n per column: " + ", ".join(f"{k} ({r}) {table['normkd_edit']['EIR'][f'{k}_{r}']['n'] + table['normkd_edit']['EIR'][f'{k}_{r}']['missing']}" for k, r in COLS), ""]
import os; os.makedirs(out, exist_ok=True)
open(f"{out}/summary_eir.md", "w").write("\n".join(md)); json.dump(table, open(f"{out}/table_eir.json", "w"), indent=1)
print("\n".join(md))
