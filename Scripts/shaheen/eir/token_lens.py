"""token_lens.parquet for the noisa set (9950X 2026-11-05 11:46): one row per dataset row, in row order:
row, o0_row, token_len (untruncated ReLL BPE tokens of the whole text incl. bos/eos = input_ids length in the fused parquet
before wash), body_token_len (tokens of the leading function body up to its `}` incl. bos/eos, eir_bf.py body_tokens of
the same normalized text the model input comes from)."""
import glob, json, sys
import datasets, pandas as pd, pyarrow.compute as pc, pyarrow.parquet as pq
E, ST = sys.argv[1], sys.argv[2]
mp = pd.read_parquet(f"{E}/o0_row_map.parquet").sort_values("eir_row")
fp = datasets.load_from_disk(f"{E}/test_final_set/train_dataset_pool").data.column("file_path").to_pylist()
full = {}
for f in glob.glob(f"{E}/parquet/coreutils/*.parquet"):
    t = pq.read_table(f, columns=["file_path", "input_ids"])
    full.update(zip(t.column(0).to_pylist(), pc.list_value_length(t.column(1)).to_pylist()))
body = {}
for f in glob.glob(f"{ST}/*.rows.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["status"] == "ok": body[r["o0_row"]] = r["body_tokens"]
out = pd.DataFrame({"row": mp.eir_row.to_numpy(), "o0_row": mp.o0_row.to_numpy(),
                    "token_len": [full[fp[i]] for i in mp.eir_row], "body_token_len": [body[o] for o in mp.o0_row]})
assert len(out) == len(fp) and (out.row.to_numpy() == range(len(fp))).all() and (out.body_token_len <= out.token_len).all()
out.to_parquet(f"{E}/token_lens.parquet", index=False)
print(len(out), out.token_len.median(), out.body_token_len.median(), (out.body_token_len <= 2048).mean())
