"""BPE corpus for one opt level: the normalized IR text DataProcess tokenizes, one row per function.

split_functions (EXCLUDED_SEGMENTS filter via X.ll.lift.json) extracts the module's functions, and
Tokenizer.corpus_builder.function_corpus_text turns each into the text DataProcess.FileProcessor tokenizes.
Function files exist on $TMPDIR one module at a time (Shaheen's file-count quota rules out the on-disk
<binary>_functions/ trees corpus_builder walks); per module this publishes <out>/<rel>.parquet with
columns function_name, text.

    python Scripts/shaheen/corpus_text.py --bc-root <level>/bc --list train_list.txt --out <corpus>/<OPT> --workers 384
"""
import os

os.environ.setdefault("USE_TORCH", "0")

import argparse
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
sys.path[:0] = [str(REPO), str(REPO / "Scripts"), str(Path(__file__).resolve().parent)]

FUNCTION_CHUNK = 128


def extract_module(bc_path, workdir, binary_rel):
    """Split one module into <workdir>/<rel>_functions/; return [(function name, file path)]."""
    import csv
    from split_functions import extract_functions, lift_state_path

    bc = Path(bc_path)
    lift_state = lift_state_path(bc)
    if not lift_state.exists():
        raise ValueError(f"{lift_state} is missing; Task 3 needs the lift state beside the module")
    function_dir = Path(workdir) / (binary_rel + "_functions")
    extract_functions(bc.resolve(), function_dir, lift_state.resolve())
    with open(function_dir / "function_map.csv", newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [(name, str(function_dir / filename)) for name, filename in rows]


def normalize_functions(functions):
    """(function_name, text) for each (name, file) that is a sample; the count of the others."""
    from Tokenizer.corpus_builder import function_corpus_text

    rows, dropped = [], 0
    for name, path in functions:
        text = function_corpus_text(path, os.path.dirname(path) + "_graphs")
        if text is None:
            dropped += 1  # pre-filtered like the dataset, or rejected by the purify pass
        else:
            rows.append((name, text))
        os.remove(path)
    return rows, dropped


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"function_name": [r[0] for r in rows], "text": [r[1] for r in rows]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path + ".part", compression="zstd")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-root", required=True)
    ap.add_argument("--list", required=True, help=".ll paths relative to the lift root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-open-modules", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    manifest = out / f"_corpus_{args.tag}.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("ok"):
                done.add(rec["module"])
    with open(args.list) as f:
        rels = [l.strip()[: -len(".ll")] for l in f if l.strip()]
    bc_root = Path(args.bc_root).resolve()
    todo = sorted((r for r in rels if r not in done), key=lambda r: (bc_root / (r + ".bc")).stat().st_size, reverse=True)
    print(f"corpus {args.tag}: {len(rels)} modules, {len(done)} done, {len(todo)} to run", flush=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="corpus-", dir=os.environ.get("TMPDIR")))
    pending, state, futures, opened, finished, t0 = deque(todo), {}, {}, 0, 0, time.time()
    with open(manifest, "a") as log, ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:

        def fill():
            nonlocal opened
            while pending and len(state) < args.max_open_modules:
                rel = pending.popleft()
                opened += 1
                workdir = tmp_root / f"m{opened}"
                state[rel] = {"workdir": workdir, "rows": [], "dropped": 0, "t": time.time()}
                futures[pool.submit(extract_module, str(bc_root / (rel + ".bc")), str(workdir), rel)] = ("x", rel)

        def finish(rel, ok, **extra):
            nonlocal finished
            st = state.pop(rel)
            rec = {"module": rel, "ok": ok, "secs": round(time.time() - st["t"], 1), **extra}
            if ok:
                write_parquet(st["rows"], str(out / (rel + ".parquet")))
                rec.update(functions=len(st["rows"]), dropped=st["dropped"])
            shutil.rmtree(st["workdir"], ignore_errors=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            finished += 1
            if finished % 100 == 0 or not (pending or state):
                print(f"  {finished}/{len(todo)} modules {time.time() - t0:.0f}s", flush=True)

        fill()
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in ready:
                kind, rel = futures.pop(fut)
                st = state[rel]
                try:
                    result = fut.result()
                except Exception as error:
                    st.setdefault("error", f"{kind}: {type(error).__name__}: {error}"[:2000])
                    st["chunks_left"] = st.get("chunks_left", 1) - 1
                    if kind == "x" or st["chunks_left"] == 0:
                        finish(rel, False, error=st["error"])
                    continue
                if kind == "x":
                    chunks = [result[i:i + FUNCTION_CHUNK] for i in range(0, len(result), FUNCTION_CHUNK)]
                    st["chunks_left"] = len(chunks)
                    if not chunks:
                        finish(rel, True)
                    for chunk in chunks:
                        futures[pool.submit(normalize_functions, chunk)] = ("f", rel)
                else:
                    rows, dropped = result
                    st["rows"] += rows
                    st["dropped"] += dropped
                    st["chunks_left"] -= 1
                    if st["chunks_left"] == 0:
                        finish(rel, "error" not in st, **({"error": st["error"]} if "error" in st else {}))
            fill()
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"corpus {args.tag}: done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
