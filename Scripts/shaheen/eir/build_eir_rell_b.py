"""coreutils-EIR-rell, rule set (b) (9950X 2026-11-04 21:36): one row per EIR function matched to an IDA function,
from eir_text_b.py. Format of coreutils-Oc-rell with two text forms:
  binary_name, function_name (the lift's LLVM name, as in the O0 / Oc rell), fva (IDA ea, hex),
  text_main, token_len_main (main version: function body + external declarations, rules M3 / M4),
  text_b, token_len_b, text_b_bodyfirst, token_len_b_bodyfirst (comparison; ReGraphv2 BPE, bos/eos included, untruncated),
  ida_name (extra: the IDA name).
Also <out>/o0_row_map.parquet (o0_row of Coreutils-O0-fused/test_final_set -> rell_row, matched by (binary, fva)) and
<out>/dropped.parquet (O0 rows with no rell row: o0_row, binary_name, function_name, reason in mips64 | ppc |
revng_failed | not_in_lift_list | addr_unmatched | llvm_dis | purify | normalize).
usage: build_eir_rell_b.py OUT WORKERS TEXT_DIR STATUS_DIR
"""
import glob, json, multiprocessing, os, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

OUT, NW, TEXT, STATUS = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
W = f"{DATA}/S11A"
TOKENIZER = f"{REPO}/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
_TOK = None


def init():
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(TOKENIZER)


def tok_lens(texts, unk_count=False):
    """Token counts (bos/eos included) in small batches, so huge functions do not hold all encodings at once."""
    out, unk = [], 0
    for k in range(0, len(texts), 16):
        for e in _TOK.encode_batch(texts[k:k + 16], add_special_tokens=True):
            out.append(len(e.ids))
            if unk_count:
                unk += e.ids.count(1)
    return out, unk


def one(rel):
    t = pq.read_table(f"{TEXT}/{rel}.parquet").to_pydict()
    lm, u1 = tok_lens(t["text_main"], True)
    lb, u2 = tok_lens(t["text_b"], True)
    lf, _ = tok_lens(t["text_b_bodyfirst"])
    unk = u1 + u2
    n = len(t["function_name"])
    return rel, pa.table({"binary_name": [rel] * n, "function_name": t["function_name"], "fva": t["fva"],
                          "text_main": pa.array(t["text_main"], pa.large_string()), "token_len_main": pa.array(lm, pa.int64()),
                          "text_b": pa.array(t["text_b"], pa.large_string()), "token_len_b": pa.array(lb, pa.int64()),
                          "text_b_bodyfirst": pa.array(t["text_b_bodyfirst"], pa.large_string()),
                          "token_len_b_bodyfirst": pa.array(lf, pa.int64()), "ida_name": t["ida_name"]}), unk


def lens(L):
    L = np.sort(np.asarray(L))
    return {"mean": round(float(L.mean())), "median": int(np.median(L)), "p95": int(np.percentile(L, 95)), "max": int(L[-1]),
            "le_2048": round(float((L <= 2048).mean()), 4)}


def main():
    os.makedirs(f"{OUT}/data", exist_ok=True)
    rels = sorted(p[len(TEXT) + 1:-len(".parquet")] for p in glob.glob(f"{TEXT}/coreutils/*.parquet"))
    t0, unk, k, step = time.time(), 0, 0, -(-len(rels) // 16)
    bins, fvas, names, L = [], [], [], {"token_len_main": [], "token_len_b": [], "token_len_b_bodyfirst": []}
    group, nulls = [], 0
    with multiprocessing.get_context("spawn").Pool(NW, initializer=init) as pool:
        for n, (rel, tab, u) in enumerate(pool.imap(one, rels, chunksize=4)):   # whole binaries per part, in binary order
            unk += u; group.append(tab)
            bins += tab.column("binary_name").to_pylist(); fvas += tab.column("fva").to_pylist()
            names += tab.column("function_name").to_pylist()
            nulls += tab.column("text_main").null_count + tab.column("text_b").null_count
            for c in L: L[c] += tab.column(c).to_pylist()
            if len(group) == step or n == len(rels) - 1:
                pq.write_table(pa.concat_tables(group), f"{OUT}/data/part-{k:05d}.parquet", compression="zstd")
                k += 1; group = []
    rell_row = {(b, int(f, 16)): r for r, (b, f) in enumerate(zip(bins, fvas))}
    # O0 rows -> rell rows, and the reasons for the rest
    idx = pd.read_parquet(f"{DATA}/S11/o0_index/coreutils_test.parquet")
    lift = {l.strip() for f in ("coreutils_lift_bins.txt", "coreutils_bindiff_extra_bins.txt") for l in open(f"{W}/pkg/pkgA/{f}") if l.strip()}
    failed_bins = set(open(f"{W}/work/revng_failed.txt").read().split())
    ffail = {}
    for f in glob.glob(f"{STATUS}/*.jsonl"):
        for l in open(f):
            s = json.loads(l)
            for nm, ea, step_ in s["failed"]:
                ffail[(s["binary"], int(ea, 16))] = step_
    mp, drop, name_diff = [], [], 0
    for r, b, n, fva, a in zip(idx.o0_row, idx.binary, idx.function_name, idx.fva, idx.arch):
        rr = rell_row.get((b, int(fva)))
        if rr is not None:
            mp.append((int(r), rr)); name_diff += names[rr] != n; continue
        if a == "mips64": why = "mips64"
        elif a in ("ppc32", "ppc64"): why = "ppc"
        elif b in failed_bins: why = "revng_failed"
        elif b not in lift: why = "not_in_lift_list"
        else: why = ffail.get((b, int(fva)), "addr_unmatched")
        drop.append((int(r), b, n, why))
    pd.DataFrame(mp, columns=["o0_row", "rell_row"]).to_parquet(f"{OUT}/o0_row_map.parquet", index=False)
    d = pd.DataFrame(drop, columns=["o0_row", "binary_name", "function_name", "reason"])
    d.to_parquet(f"{OUT}/dropped.parquet", index=False)
    stats = {"rows": len(bins), "builds": len(rels), "parts": k, "seconds": round(time.time() - t0),
             "dup_keys": len(bins) - len(rell_row), "null_text": nulls, "unk_tokens_text_main_and_b": unk,
             "token_len_main": lens(L["token_len_main"]), "token_len_b": lens(L["token_len_b"]),
             "token_len_b_bodyfirst": lens(L["token_len_b_bodyfirst"]),
             "o0_rows": len(idx), "o0_rows_mapped": len(mp), "o0_mapped_name_differs": int(name_diff),
             "o0_dropped": len(d), "o0_dropped_by_reason": d.reason.value_counts().to_dict(),
             "text_failed_functions": len(ffail)}
    json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
