"""Task 3 + GraphBuilder + DataProcess for one shard of .bc files, fused on node-local storage.

Same per-function semantics as the repo pipeline, which it calls rather than reimplements:
  split_functions.extract_functions  (EXCLUDED_SEGMENTS filter via X.ll.lift.json, split_funcs;
                                      with --select-root, only the functions <root>/<rel>.ll.select.json lists)
  Utils.utils.should_skip_this_file   (graph_generator's FileFilter pre-filter)
  graph_generator.process_one_file    (purify, DDG, CFG with the LLVMTools passes)
  DataProcess FileProcessor           (normalize, tokenize, DDG/CFG token spans)
What differs is only where files live: function files and graph artifacts exist on $TMPDIR for one
module at a time and never reach Lustre. Per module this publishes
  <out>/parquet/<split>/<proj>/<binary>.parquet          rows in DataProcess's raw schema
  <out>/functions/<split>/<proj>/<binary>_functions/function_map.csv
and each row's file_path names <binary>_functions/<sha1>.ll in that functions tree, so
Pretrain.split_train_validation finds function_map.csv beside it as for a Task 3 output.
Resume: modules recorded in <out>/manifests/fused_<tag>.jsonl are skipped.

    python Scripts/shaheen/fused_task3.py --bc-root <out>/bc --list shard_000.txt --out <out> --workers 384
"""
import os

os.environ.setdefault("USE_TORCH", "0")  # tokenizer-only workers; as the DataProcess CLIs do

import argparse
import csv
import json
import multiprocessing
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "Scripts")]

FUNCTION_CHUNK = 64
_processor = None


def init_worker(tokenizer_path):
    global _processor
    os.environ["USE_TORCH"] = "0"
    from DataProcess.file_processor import FileProcessor
    from Tokenizer.ir_tokenizer import load_tokenizer

    _processor = FileProcessor(load_tokenizer(tokenizer_path), cleanup_temp_files=False)


def extract_module(bc_path, workdir, binary_rel, select_root=None):
    """Split one module into function files under workdir; return (function_map rows, kept files)."""
    from split_functions import extract_functions, lift_state_path, load_selection
    from Utils.utils import should_skip_this_file

    bc = Path(bc_path)
    lift_state = lift_state_path(bc)
    if not lift_state.exists():
        raise ValueError(f"{lift_state} is missing; Task 3 needs the lift state beside the module")
    function_dir = Path(workdir) / (binary_rel + "_functions")
    selection = load_selection(Path(select_root) / (binary_rel + ".ll.select.json")) if select_root else None
    extract_functions(bc.resolve(), function_dir, lift_state.resolve(), selection)
    with open(function_dir / "function_map.csv", newline="") as f:
        rows = list(csv.reader(f))
    kept = []
    for _, filename in rows[1:]:
        path = function_dir / filename
        with open(path, encoding="utf-8") as f:  # undecodable IR is an error, as in FileFilter
            if not should_skip_this_file(f.read()):
                kept.append(str(path))
    return rows, kept


def process_functions(paths, workdir, virtual_dir):
    """Graphs + samples for function files; artifacts are deleted as soon as a sample is built."""
    from GraphBuilder.graph_generator import process_one_file

    samples, graph_failed, sample_failed = [], 0, 0
    for path in paths:
        artifacts = process_one_file(path)
        if artifacts is None:
            graph_failed += 1
            continue
        try:
            result = _processor.process_single_file(*artifacts)
        finally:
            for artifact in artifacts[1:]:
                if os.path.exists(artifact):
                    os.remove(artifact)
        if result.success:
            sample = result.to_dict()
            rel = os.path.relpath(path, workdir)  # <split>/<proj>/<binary>_functions/<sha1>.ll
            sample["file_path"] = os.path.join(virtual_dir, rel)
            samples.append(sample)
        else:
            sample_failed += 1
        os.remove(path)
    return samples, graph_failed, sample_failed


def write_parquet(samples, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("file_path", pa.string()),
        ("ddg_graph", pa.list_(pa.list_(pa.int32()))),
        ("cfg_graph", pa.list_(pa.list_(pa.float32()))),
        ("input_ids", pa.list_(pa.int32())),
    ])
    samples.sort(key=lambda s: s["file_path"])  # DataProcess orders rows by function path
    table = pa.Table.from_pylist(samples, schema=schema)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path + ".part")
    os.replace(path + ".part", path)


def publish_function_map(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", newline="") as f:  # same quoting as split_functions writes it
        f.write("OriginalFunctionName,HashedFileName\n")
        csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(rows[1:])
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-root", required=True)
    ap.add_argument("--list", required=True, help=".ll paths relative to the lift root (a shard list)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-open-modules", type=int, default=32,
                    help="modules extracted on $TMPDIR at once; bounds local disk use")
    ap.add_argument("--tag", default="")
    ap.add_argument("--select-root", help="select_functions.py output; extract only the selected functions")
    ap.add_argument("--tokenizer-path", default=str(REPO / "Tokenizer" / "output_tokenizer" / "llvm_ir_bpe.json"))
    args = ap.parse_args()

    from GraphBuilder.graph_generator import check_toolchain

    check_toolchain()  # fail before any output if opt or a pass plugin is broken
    out = Path(args.out).resolve()
    manifest = out / "manifests" / f"fused_{args.tag}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # torn last line of a killed run
            if rec.get("ok"):
                done.add(rec["module"])

    with open(args.list) as f:
        rels = [l.strip()[: -len(".ll")] for l in f if l.strip()]
    bc_root = Path(args.bc_root).resolve()
    todo = [r for r in rels if r not in done and (bc_root / (r + ".bc")).exists()]
    missing_bc = [r for r in rels if r not in done and not (bc_root / (r + ".bc")).exists()]
    todo.sort(key=lambda r: (bc_root / (r + ".bc")).stat().st_size, reverse=True)  # big modules first
    print(f"fused task3 {args.tag}: {len(rels)} modules, {len(done)} done, {len(todo)} to run, "
          f"{len(missing_bc)} without .bc (task2 failures), {args.workers} workers", flush=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="fused-", dir=os.environ.get("TMPDIR")))
    virtual_dir = str(out / "functions")
    pending_modules = deque(todo)
    opened = 0
    state = {}  # module -> dict(workdir, rows, chunks_left, samples, counters)
    futures = {}
    t0, finished = time.time(), 0
    ctx = multiprocessing.get_context("spawn")
    with open(manifest, "a") as log, ProcessPoolExecutor(
        args.workers, mp_context=ctx, initializer=init_worker, initargs=(args.tokenizer_path,)
    ) as pool:

        def fill():
            nonlocal opened
            while pending_modules and len(state) < args.max_open_modules:
                rel = pending_modules.popleft()
                opened += 1
                workdir = tmp_root / f"m{opened}"  # holds <rel>_functions/ and <rel>_graphs/
                state[rel] = {"workdir": workdir, "samples": [], "graph_failed": 0, "sample_failed": 0,
                              "t": time.time()}
                futures[pool.submit(extract_module, str(bc_root / (rel + ".bc")), str(workdir), rel,
                                     args.select_root)] = ("x", rel)

        def finish(rel, ok, **extra):
            nonlocal finished
            st = state.pop(rel)
            rec = {"module": rel, "ok": ok, "secs": round(time.time() - st["t"], 1), **extra}
            if ok:
                write_parquet(st["samples"], str(out / "parquet" / (rel + ".parquet")))
                publish_function_map(st["rows"], str(out / "functions" / (rel + "_functions") / "function_map.csv"))
                rec.update(samples=len(st["samples"]), functions=len(st["rows"]) - 1,
                           graph_failed=st["graph_failed"], sample_failed=st["sample_failed"])
            shutil.rmtree(st["workdir"], ignore_errors=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            finished += 1
            if finished % 50 == 0 or not (pending_modules or state):
                print(f"  {finished}/{len(todo)} modules {time.time() - t0:.0f}s", flush=True)

        fill()
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in ready:
                kind, rel = futures.pop(fut)
                st = state[rel]
                try:
                    result = fut.result()
                except Exception as error:  # a module-level failure: record it, keep the shard going
                    if kind == "x" or "error" not in st:
                        st["error"] = f"{kind}: {type(error).__name__}: {error}"[:2000]
                    st["chunks_left"] = st.get("chunks_left", 1) - 1
                    if kind == "x" or st["chunks_left"] == 0:
                        finish(rel, False, error=st["error"])
                    continue
                if kind == "x":
                    rows, kept = result
                    st["rows"] = rows
                    chunks = [kept[i:i + FUNCTION_CHUNK] for i in range(0, len(kept), FUNCTION_CHUNK)]
                    st["chunks_left"] = len(chunks)
                    if not chunks:
                        finish(rel, True)
                    for chunk in chunks:
                        futures[pool.submit(process_functions, chunk, str(st["workdir"]), virtual_dir)] = ("f", rel)
                else:
                    samples, graph_failed, sample_failed = result
                    st["samples"] += samples
                    st["graph_failed"] += graph_failed
                    st["sample_failed"] += sample_failed
                    st["chunks_left"] -= 1
                    if st["chunks_left"] == 0:
                        if "error" in st:
                            finish(rel, False, error=st["error"])
                        else:
                            finish(rel, True)
            fill()
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"fused task3 {args.tag}: done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
