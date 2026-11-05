"""S11 part B, bodyfirst fused (9950X 2026-11-05) for one lift shard (coreutils): fused samples for the O0 test rows
from the EIR with the lifted function first. Same matching, file names and row statuses as eir_b.py; changes:
  input IR = strip_metadata(agreed(llvm-dis)) (transforms 1, 2, B3 = the text_b input, valid IR);
  after process_one_file (purify -> DDG pass -> CFG pass: the last time opt writes the module) the define block is moved
  to the front of the purified IR and of the DDG-instrumented IR (eir_transform.body_first), and only then
  FileProcessor.process_single_file normalizes, tokenizes and maps the CFG / DDG spans. So the final token sequence
  starts with the function body and the graph positions refer to that order (DDG pairs instrumented / purified
  instructions in order, both files are moved the same way; CFG finds blocks by label).
  -> <fused>/parquet/coreutils/<bin>.parquet, <fused>/functions/coreutils/<bin>_functions/function_map.csv,
     <status>/<tag>.{binaries,rows}.jsonl
With --noisa (9950X 2026-11-05 11:44): after the move, eir_transform.noisa (N1 strings `0x..:Code_<isa>` -> PC,
  N2 labels `bb.0x.._Code_<isa>` -> bb.pc<k>, N3 other Code_<isa> -> PC) is applied with ONE label map to the purified IR,
  the instrumented IR and both dot files (the CFG dot names blocks by label), so the graphs are unchanged. Per row the
  status then also records: residual matches of code_?(arm|thumb|aarch64|mips|x86) (case-insensitive) in the normalized
  text and in the decoded input_ids, and body_tokens (tokens of the define block, from the encoding of the final text).
usage: eir_bf.py <lift shard .tar> <tag> <fused out> <status dir> <ida_names.json> <o0_index.parquet> <llvm-dis>
                 [--workers N] [--tmp DIR] [--noisa]
"""
import argparse, csv, io, json, os, shutil, subprocess, sys, tarfile, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories
sys.path[:0] = [str(Path(__file__).resolve().parent), f"{REPO}", f"{REPO}/Scripts"]
os.environ["USE_TORCH"] = "0"
import eir_transform as T

TOKENIZER = f"{REPO}/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
_proc = None


def init():
    global _proc
    from DataProcess.file_processor import FileProcessor
    from Tokenizer.ir_tokenizer import load_tokenizer
    _proc = FileProcessor(load_tokenizer(TOKENIZER), cleanup_temp_files=False)


def move_body_first(path):
    with open(path) as fh:
        text = fh.read()
    with open(path, "w") as fh:
        fh.write(T.body_first(text))


def one(job):
    from GraphBuilder.graph_generator import process_one_file
    import pyarrow as pa, pyarrow.parquet as pq
    key, data, names, rows, dis, tmp = job["key"], job["isolate"], job["names"], job["rows"], job["dis"], job["tmp"]
    b = key.rsplit("/", 1)[1]; t0 = time.time()
    w = Path(tempfile.mkdtemp(prefix="eirbf_", dir=tmp)); fdir = w / f"{b}_functions"; fdir.mkdir()
    st = {"binary": key, "ok": False}; row_status = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            members = {m.name: m for m in t.getmembers() if m.isfile() and "/function/" in "/" + m.name}
            ida = {int(k): v for k, v in names.items()}
            off = T.load_offset(T.local_addresses(members), ida)
            thumb = T.thumb_addresses(f"{REPO}/Binaries/Coreutils/{key}") if b.startswith("arm32-") else set()
            chosen, dups = T.choose_members(list(members), ida, off, thumb)
            mem = {ea + off: members[n] for n, ea in chosen.items()}
            st.update(revng_functions=len(members), ida_functions=len(ida), load_offset=hex(off), isa_duplicate_addresses=dups)
            samples, fmap = [], []
            for r0 in sorted(rows, key=lambda r: r["fva"]):
                ea = int(r0["fva"])
                m = mem.get(ea + off)
                if m is None or ea not in ida:
                    row_status.append((r0["o0_row"], "addr_unmatched")); continue
                bc = w / "f.bc"; bc.write_bytes(t.extractfile(m).read())
                r = subprocess.run([dis, str(bc), "-o", "-"], capture_output=True, text=True, timeout=3600)
                if r.returncode:
                    row_status.append((r0["o0_row"], "llvm_dis_failed")); continue
                fname = r0["fname"]
                ll = fdir / fname; ll.write_text(T.strip_metadata(T.agreed(r.stdout)))
                art = process_one_file(str(ll))
                if art is None:
                    row_status.append((r0["o0_row"], "graph_failed")); ll.unlink(); continue
                extra = {}
                try:
                    move_body_first(art[1]); move_body_first(art[2])      # purified IR, DDG-instrumented IR
                    if job["noisa"]:
                        labels = {}
                        for x in art[1:5]:                                 # purified, instrumented, CFG dot, DDG dot
                            with open(x) as fh: tx = fh.read()
                            tx, labels = T.noisa(tx, labels)
                            with open(x, "w") as fh: fh.write(tx)
                        from Tokenizer.normalizer import normalize_file
                        norm = normalize_file(art[1])
                        j = norm.find("\n}\n")
                        extra = {"residual_text": len(T.RESIDUAL.findall(norm)),
                                 "body_tokens": len(_proc.tokenizer(norm[:j + 2])["input_ids"]) if j > 0 else None}
                    res = _proc.process_single_file(*art)
                    if job["noisa"] and res.success:
                        extra["residual_ids"] = len(T.RESIDUAL.findall(_proc.tokenizer.decode(res.input_ids)))
                finally:
                    for x in art[1:]:
                        if os.path.exists(x): os.remove(x)
                if res.success:
                    s = res.to_dict(); s["file_path"] = f"{job['fused']}/functions/coreutils/{b}_functions/{fname}"
                    samples.append(s); fmap.append((r0["function_name"], fname)); row_status.append((r0["o0_row"], "ok", extra))
                else:
                    row_status.append((r0["o0_row"], "sample_failed"))
                if ll.exists(): ll.unlink()
        schema = pa.schema([("file_path", pa.string()), ("ddg_graph", pa.list_(pa.list_(pa.int32()))),
                            ("cfg_graph", pa.list_(pa.list_(pa.float32()))), ("input_ids", pa.list_(pa.int32()))])
        samples.sort(key=lambda s: s["file_path"])
        pdir = Path(job["fused"]) / "parquet" / "coreutils"; pdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(samples, schema=schema), str(pdir / f"{b}.parquet"))
        mdir = Path(job["fused"]) / "functions" / "coreutils" / f"{b}_functions"; mdir.mkdir(parents=True, exist_ok=True)
        with open(mdir / "function_map.csv", "w", newline="") as f:
            f.write("OriginalFunctionName,HashedFileName\n")
            csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(fmap)
        st.update(ok=True, o0_rows=len(rows), samples=len(samples))
    except Exception as e:  # noqa: BLE001
        st["error"] = repr(e)[-500:]
        row_status = [(r0["o0_row"], "processing_error") for r0 in rows]
    finally:
        st["secs"] = round(time.time() - t0, 1); shutil.rmtree(w, ignore_errors=True)
    return st, [{"o0_row": x[0], "binary": key, "status": x[1], **(x[2] if len(x) > 2 else {})} for x in row_status]


def main():
    ap = argparse.ArgumentParser()
    for x in ("shard", "tag", "fused", "status", "names", "o0_index", "dis"):
        ap.add_argument(x)
    ap.add_argument("--workers", type=int, default=96); ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--noisa", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    names = json.load(open(a.names))
    idx = pd.read_parquet(a.o0_index)
    by = {k: g[["o0_row", "function_name", "fname", "fva"]].to_dict("records") for k, g in idx.groupby("binary")}
    stage = Path(tempfile.mkdtemp(prefix=f"bf_{a.tag}_", dir=a.tmp)); Path(a.status).mkdir(parents=True, exist_ok=True)
    jobs = []
    with tarfile.open(a.shard) as t:
        for m in t.getmembers():
            if m.name.endswith(".isolate.tar"):
                key = "coreutils/" + Path(m.name).name[:-len(".isolate.tar")].split("__", 1)[1]
                jobs.append({"key": key, "isolate": t.extractfile(m).read(), "names": names.get(key, {}), "rows": by.get(key, []),
                             "dis": a.dis, "tmp": str(stage), "fused": a.fused, "noisa": a.noisa})
    stats, rows = [], []
    with ProcessPoolExecutor(a.workers, initializer=init) as ex:
        for s, r in ex.map(one, jobs):
            stats.append(s); rows += r
            print(json.dumps({k: s.get(k) for k in ("binary", "ok", "o0_rows", "samples", "secs", "error")}), flush=True)
    with open(f"{a.status}/{a.tag}.binaries.jsonl", "w") as f:
        for s in sorted(stats, key=lambda s: s["binary"]): f.write(json.dumps(s) + "\n")
    with open(f"{a.status}/{a.tag}.rows.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: r["o0_row"]): f.write(json.dumps(r) + "\n")
    shutil.rmtree(stage, ignore_errors=True)
    print(f"{a.tag}: {sum(s['ok'] for s in stats)}/{len(stats)} binaries, rows ok {sum(r['status'] == 'ok' for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
