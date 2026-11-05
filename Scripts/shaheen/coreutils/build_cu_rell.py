"""coreutils IR text dataset (copy of $REGRAPH_DATA/cu_oc_rell/build_cu_oc_text.py, paths as arguments): one row per lifted function of every coreutils build.

Texts: <TEXT>/coreutils/<binary>.parquet, written by irsim_text.py for the coreutils IR
similarity run (every function outside EXCLUDED_SEGMENTS, purify + normalize_file, untruncated, no selection).
fva: the IDA start address of the function, from the lift state <binary>.ll.lift.json (llvm_name -> ea).
token_len: len(tok(text)["input_ids"]) with the ReGraphv2 BPE tokenizer (Tokenizer/output_tokenizer/llvm_ir_bpe.json),
bos/eos included, untruncated; equal to the input_ids length ReGraphv2 stores (checked on Dataset-1 Oc).
usage: build_cu_rell.py OUT WORKERS TEXT_DIR
Writes <out>/data/part-NNNNN.parquet (binary_name, function_name, fva, text, token_len) and <out>/stats.json.
"""
import glob
import json
import multiprocessing
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

TEXT = sys.argv[3]  # : $REGRAPH_DATA/irsim/coreutils/text/<OPT>
LIFT = f"{DATA}/Coreutils-ll"
TOKENIZER = f"{REPO}/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
_TOK = None


def init():
    global _TOK
    from tokenizers import Tokenizer

    _TOK = Tokenizer.from_file(TOKENIZER)


def one(rel):
    t = pq.read_table(f"{TEXT}/{rel}.parquet", columns=["function_name", "text"]).to_pydict()
    with open(f"{LIFT}/{rel}.ll.lift.json") as f:
        ea = {}
        for r in json.load(f)["functions"]:
            if r.get("status") == "lifted" and r.get("llvm_name"):
                ea.setdefault(r["llvm_name"], []).append(r["ea"])
    names, texts = t["function_name"], t["text"]
    lens = [len(e) for e in _TOK.encode_batch([x or "" for x in texts], add_special_tokens=True)]
    fva = [ea[n][0] if len(ea.get(n, [])) == 1 else None for n in names]
    stats = {"rows": len(names), "fva_missing": sum(v is None for v in fva),
             "fva_ambiguous": sum(len(ea.get(n, [])) > 1 for n in names),
             "dup_names": len(names) - len(set(names)), "null_text": sum(x is None for x in texts)}
    return rel, pa.table({"binary_name": [rel] * len(names), "function_name": names, "fva": fva,
                          "text": pa.array(texts, pa.large_string()), "token_len": pa.array(lens, pa.int64())}), stats


def main():
    out = sys.argv[1]
    os.makedirs(f"{out}/data", exist_ok=True)
    rels = sorted(p[len(TEXT) + 1:-len(".parquet")] for p in glob.glob(f"{TEXT}/coreutils/*.parquet"))
    t0, tables, agg = time.time(), [], {}
    with multiprocessing.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 192, initializer=init) as pool:
        for rel, tab, st in pool.imap(one, rels, chunksize=4):  # imap keeps the sorted binary order
            tables.append(tab)
            for k, v in st.items():
                agg[k] = agg.get(k, 0) + v
    table = pa.concat_tables(tables)
    n_parts = 16
    step = -(-len(rels) // n_parts)
    k = 0
    for i in range(n_parts):  # whole binaries per part, in binary order
        group = tables[i * step:(i + 1) * step]
        if group:
            pq.write_table(pa.concat_tables(group), f"{out}/data/part-{k:05d}.parquet", compression="zstd")
            k += 1
    L = sorted(table.column("token_len").to_pylist())
    agg.update(builds=len(rels), parts=k, seconds=round(time.time() - t0),
               token_len={"mean": round(sum(L) / len(L)), "median": L[len(L) // 2], "p95": L[int(len(L) * .95)],
                          "max": L[-1], "over_2048": round(sum(x > 2048 for x in L) / len(L), 4)})
    json.dump(agg, open(f"{out}/stats.json", "w"), indent=1)
    print(json.dumps(agg))


if __name__ == "__main__":
    main()
