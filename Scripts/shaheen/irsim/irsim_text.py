"""IR-similarity input for one opt level: every lifted function of every module, no selection, untruncated.

Same text as ReGraphv2 tokenizes (graph_generator._opt_initial_purify, then Tokenizer normalize_file), but no
function is dropped: should_skip_this_file (GraphBuilder's pre-filter) is only recorded as a flag, and a
function whose purify pass fails is kept with text = null. Functions in EXCLUDED_SEGMENTS (PLT stubs,
.init/.fini) are not samples of any ReGraphv2 set and are left out, as in split_functions.

Per module <out>/<rel>.parquet, one row per function:
  function_name, text (normalized IR), skip (should_skip_this_file), purify_ok, n_blocks,
  ntok (whitespace tokens), h_raw / h_canon (blake2b-128 of the text / of the canonicalized text),
  h_ops (opcode sequence), nops (opcode count)
Canonicalization renames every local name (%.N, %.a.b, %varN, %N, block labels "@N" / %"@N", any other %name)
to %n0, %n1, ... in order of first appearance; see canon().

    python irsim_text.py --bc-root <level>/bc --list shard_000.txt --out <irsim>/text/<OPT> --workers 384 --tag 000
"""
import os

os.environ.setdefault("USE_TORCH", "0")

import argparse
import hashlib
import json
import multiprocessing
import re
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
REPO = os.environ.get("REGRAPH_REPO", str(Path(__file__).resolve().parents[3]))  # see Scripts/shaheen/cluster_env.sh
DATA = os.environ.get("REGRAPH_DATA", "")  # root of the data and stage work directories

REPO = Path(f"{REPO}")
sys.path[:0] = [str(REPO), str(REPO / "Scripts")]

FUNCTION_CHUNK = 64

# A local name: %"..." (quoted, e.g. block labels %"@4"), or %name / %N; a block label definition at the start of a
# line: "@4": or name: . A label definition and its uses are the same name (the key drops the leading %).
LOCAL = re.compile(r'%"[^"]*"|%[-a-zA-Z$._0-9]+')
LABEL_DEF = re.compile(r'^("[^"]*"|[-a-zA-Z$._0-9]+):', re.M)
OPS = set("ret br switch indirectbr invoke resume unreachable fneg add fadd sub fsub mul fmul udiv sdiv fdiv urem srem frem "
          "shl lshr ashr and or xor extractelement insertelement shufflevector extractvalue insertvalue alloca load store "
          "fence cmpxchg atomicrmw getelementptr trunc zext sext fptrunc fpext fptoui fptosi uitofp sitofp ptrtoint "
          "inttoptr bitcast addrspacecast icmp fcmp phi select freeze call va_arg landingpad".split())


def canon(text):
    """Rename local names to %n0, %n1, ... in order of first appearance (label definitions included)."""
    seen = {}

    def name(key):
        if key not in seen:
            seen[key] = f"n{len(seen)}"
        return seen[key]

    out = []
    for line in text.split("\n"):
        m = LABEL_DEF.match(line)
        if m:  # "@4": -> n7:  (the use %"@4" maps to %n7)
            line = name(m.group(1)) + line[m.end(1):]
            rest_start = len(name(m.group(1)))
            line = line[:rest_start] + LOCAL.sub(lambda x: "%" + name(x.group(0)[1:]), line[rest_start:])
        else:
            line = LOCAL.sub(lambda x: "%" + name(x.group(0)[1:]), line)
        out.append(line)
    return "\n".join(out)


def blocks(text):
    """Basic blocks of the (single) function: label definitions, plus an unlabeled entry block."""
    body = text.split("{", 1)[1] if "{" in text else text
    labels = LABEL_DEF.findall(body)
    first = next((l for l in body.split("\n") if l.strip()), "")
    return len(labels) + (0 if LABEL_DEF.match(first) else 1)


def h(s):
    return hashlib.blake2b(s.encode(), digest_size=16).digest()


def describe(name, text, skip, purify_ok):
    row = {"function_name": name, "text": text, "skip": skip, "purify_ok": purify_ok}
    if text is None:
        row.update(n_blocks=None, ntok=None, h_raw=None, h_canon=None, h_ops=None, nops=None)
        return row
    words = text.split()
    ops = [w for w in words if w in OPS]
    row.update(n_blocks=blocks(text), ntok=len(words), h_raw=h(text), h_canon=h(canon(text)),
               h_ops=h(" ".join(ops)), nops=len(ops))
    return row


def extract_module(bc_path, workdir, binary_rel):
    import csv
    from split_functions import extract_functions, lift_state_path

    bc = Path(bc_path)
    function_dir = Path(workdir) / (binary_rel + "_functions")
    extract_functions(bc.resolve(), function_dir, lift_state_path(bc).resolve())
    with open(function_dir / "function_map.csv", newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [(name, str(function_dir / filename)) for name, filename in rows]


def normalize_functions(functions):
    from GraphBuilder.graph_generator import _opt_initial_purify
    from Tokenizer.normalizer import normalize_file
    from Utils.utils import should_skip_this_file

    rows = []
    for name, path in functions:
        with open(path, encoding="utf-8") as f:
            skip = bool(should_skip_this_file(f.read()))
        workdir = os.path.dirname(path) + "_graphs"
        os.makedirs(workdir, exist_ok=True)
        purified = _opt_initial_purify(path, workdir)
        text = None
        if purified is not None:
            try:
                text = normalize_file(purified)
            finally:
                os.remove(purified)
        rows.append(describe(name, text, skip, purified is not None))
        os.remove(path)
    return rows


SCHEMA = None


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("function_name", pa.string()), ("text", pa.large_string()), ("skip", pa.bool_()),
                        ("purify_ok", pa.bool_()), ("n_blocks", pa.int32()), ("ntok", pa.int64()),
                        ("h_raw", pa.binary(16)), ("h_canon", pa.binary(16)), ("h_ops", pa.binary(16)),
                        ("nops", pa.int64())])
    rows.sort(key=lambda r: r["function_name"])
    table = pa.Table.from_pylist(rows, schema=schema)
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
    manifest = out / f"_irsim_{args.tag}.jsonl"
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
    no_bc = [r for r in rels if r not in done and not (bc_root / (r + ".bc")).exists()]  # reopt failed at this level
    todo = sorted((r for r in rels if r not in done and r not in no_bc),
                  key=lambda r: (bc_root / (r + ".bc")).stat().st_size, reverse=True)
    print(f"irsim {args.tag}: {len(rels)} modules, {len(done)} done, {len(todo)} to run, {len(no_bc)} without .bc",
          flush=True)
    with open(manifest, "a") as log:
        for rel in no_bc:
            log.write(json.dumps({"module": rel, "ok": False, "error": "no .bc at this level"}) + "\n")

    tmp_root = Path(tempfile.mkdtemp(prefix="irsim-", dir=os.environ.get("TMPDIR")))
    pending, state, futures, opened, finished, t0 = deque(todo), {}, {}, 0, 0, time.time()
    with open(manifest, "a") as log, ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:

        def fill():
            nonlocal opened
            while pending and len(state) < args.max_open_modules:
                rel = pending.popleft()
                opened += 1
                workdir = tmp_root / f"m{opened}"
                state[rel] = {"workdir": workdir, "rows": [], "t": time.time()}
                futures[pool.submit(extract_module, str(bc_root / (rel + ".bc")), str(workdir), rel)] = ("x", rel)

        def finish(rel, ok, **extra):
            nonlocal finished
            st = state.pop(rel)
            rec = {"module": rel, "ok": ok, "secs": round(time.time() - st["t"], 1), **extra}
            if ok:
                rows = st["rows"]
                write_parquet(rows, str(out / (rel + ".parquet")))
                rec.update(functions=len(rows), purify_failed=sum(not r["purify_ok"] for r in rows),
                           skip=sum(r["skip"] for r in rows))
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
                    st["rows"] += result
                    st["chunks_left"] -= 1
                    if st["chunks_left"] == 0:
                        finish(rel, "error" not in st, **({"error": st["error"]} if "error" in st else {}))
            fill()
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"irsim {args.tag}: done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
