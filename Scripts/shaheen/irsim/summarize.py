"""Merge pairs_hash.parquet with dist_<L>.parquet into pairs.parquet, and write summary.md.

Tables, overall / per project / per compiler, for every level present:
  identity rate per kind (raw, canon; all pairs and the subset with different hashopcodes; pair-weighted and
  group-weighted), over all sampled pairs (>= 1M per kind);
  distances on the distance subset (first N pairs per kind): mean, median, AUC vs NEG (P(d_pos < d_neg), ties 1/2).
Compiler of a pair: the compiler of both builds if they agree, else "mixed" (XC is always mixed).
Project of a pair: the anchor's project (NEG pairs are grouped by their anchor too).
"""
import argparse
import json

import numpy as np
import pandas as pd

import os
KINDS = os.environ.get("IRSIM_KINDS", "XA,XO,XC,XM,XV").split(",")
DIST = [("raw_edit", "token edit (raw)"), ("raw_jac", "token 1-Jaccard (raw)"), ("raw_lenr", "|log len ratio| (raw)"),
        ("canon_edit", "token edit (canon)"), ("canon_jac", "token 1-Jaccard (canon)"),
        ("canon_lenr", "|log len ratio| (canon)"), ("ops_edit", "opcode edit"), ("ops_jac", "opcode 1-Jaccard"),
        ("ops_lenr", "opcode |log len ratio|"), ("cfg_lenr", "CFG nodes |log ratio|")]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.sort(np.asarray(neg, float))
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    lt = np.searchsorted(neg, pos, side="right")
    le = np.searchsorted(neg, pos, side="left")
    return float(((len(neg) - lt) + 0.5 * (lt - le)).mean() / len(neg))


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--levels", default="O0,Oc,O3,O1,O2,Os,Oz")
    args = ap.parse_args()
    D = args.pairs_dir
    uni = pd.read_parquet(f"{D}/universe.parquet", columns=["fid", "module", "function_name", "project", "lib", "arch", "comp", "ver", "opt", "group"])
    pairs = pd.read_parquet(f"{D}/pairs_hash.parquet")
    levels = [L for L in args.levels.split(",") if f"{L}_ident_raw" in pairs]
    for side in "ij":
        u = uni.iloc[pairs[side].to_numpy()]
        pairs[f"module_{side}"] = u.module.to_numpy()
        pairs[f"function_{side}"] = u.function_name.to_numpy()
        pairs[f"comp_{side}"] = u.comp.to_numpy()
        pairs[f"group_{side}"] = u.group.to_numpy()
    pairs["project"] = uni.project.to_numpy()[pairs.i]
    pairs["arch"] = uni.arch.to_numpy()[pairs.i]  # the anchor's architecture (XA pairs differ in arch)
    pairs["compiler"] = np.where(pairs.comp_i == pairs.comp_j, pairs.comp_i, "mixed")
    for L in levels:
        pairs[f"{L}_cfg_lenr"] = np.abs(np.log(pairs[f"{L}_nblk_i"].astype(float).clip(lower=1) /
                                               pairs[f"{L}_nblk_j"].astype(float).clip(lower=1)))
        try:
            d = pd.read_parquet(f"{D}/dist_{L}.parquet").drop(columns=["ntok_i", "ntok_j"])
        except FileNotFoundError:
            continue
        d = d.rename(columns={c: f"{L}_{c}" for c in d.columns if c != "pair"})
        pairs = pairs.merge(d, on="pair", how="left")
    pairs.to_parquet(f"{D}/pairs.parquet")

    counts = json.load(open(f"{D}/counts.json"))
    out = ["# IR similarity of builds of the same function (no selection, no deduplication, untruncated)", ""]
    out += ["Pairs per kind (identity tables use all, distance tables the first N of each kind): " +
            ", ".join(f"{k} {v['pairs']:,}" for k, v in counts["sampling"].items()),
            f"Distance subset: {int(pairs.dist.sum()):,} pairs. Universe: {counts['universe']:,} functions.", ""]

    def views(df):
        yield "overall", df
        for p, g in df.groupby("project"):
            yield f"project {p}", g
        for c, g in df.groupby("compiler"):
            yield f"compiler {c}", g
        if os.environ.get("IRSIM_ARCH_VIEW"):
            for a, g in df.groupby("arch"):
                yield f"arch {a} (anchor)", g

    out += ["## 1. Identity rate (byte-identical normalized IR), all sampled pairs", "",
            "Cells: raw / canon. *diff-bin* = only pairs whose two builds have different hashopcodes "
            "(the binary code differs); *grp-w* = each positive group weighted equally (raw).", ""]
    for name, g in views(pairs):
        out += [f"### {name}", "", "| level | " + " | ".join(KINDS + ["NEG"]) + " |", "|---|" + "---|" * (len(KINDS) + 1)]
        for L in levels:
            for label, sub in [("all", g), ("diff-bin", g[g.same_hashopcodes == False])]:  # noqa: E712
                cells = []
                for k in KINDS + ["NEG"]:
                    s = sub[sub.kind == k]
                    if not len(s):
                        cells.append("–")
                        continue
                    cells.append(f"{s[f'{L}_ident_raw'].mean():.3f} / {s[f'{L}_ident_canon'].mean():.3f} (n={len(s):,})")
                out.append(f"| {L} {label} | " + " | ".join(cells) + " |")
            cells = []
            for k in KINDS + ["NEG"]:
                s = g[g.kind == k]
                if not len(s):
                    cells.append("–")
                    continue
                w = 1.0 / s.groupby("group_i")["pair"].transform("size")
                cells.append(fmt(float((s[f"{L}_ident_raw"].astype(float) * w).sum() / w.sum())))
            out.append(f"| {L} grp-w | " + " | ".join(cells) + " |")
        out.append("")

    dist = pairs[pairs.dist]
    out += ["## 2. Distances on the distance subset", "",
            "Cells: mean / median [AUC vs NEG]. AUC = P(d_pos < d_neg), ties 1/2, against the NEG pairs of the same view.", ""]
    for name, g in views(dist):
        out += [f"### {name}", ""]
        for col, title in DIST:
            if not any(f"{L}_{col}" in g for L in levels):
                continue
            out += [f"**{title}**", "", "| level | " + " | ".join(KINDS) + " | NEG |", "|---|" + "---|" * (len(KINDS) + 1)]
            for L in levels:
                c = f"{L}_{col}"
                if c not in g:
                    continue
                neg = g.loc[g.kind == "NEG", c].astype(float)
                cells = []
                for k in KINDS:
                    v = g.loc[g.kind == k, c].astype(float)
                    cells.append(f"{fmt(v.mean())} / {fmt(v.median())} [{fmt(auc(v, neg))}]" if len(v) else "–")
                out.append(f"| {L} | " + " | ".join(cells) + f" | {fmt(neg.mean())} / {fmt(neg.median())} |")
            out.append("")
        out.append(f"pairs: " + ", ".join(f"{k} {int((g.kind == k).sum()):,}" for k in KINDS + ["NEG"]))
        out.append("")
    open(f"{D}/summary.md", "w").write("\n".join(out))
    print(f"wrote {D}/pairs.parquet ({len(pairs):,} rows, {len(pairs.columns)} columns) and summary.md")


if __name__ == "__main__":
    main()
