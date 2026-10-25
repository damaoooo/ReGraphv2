"""ReLL (Qwen text) final_set for one split, aligned row by row with the ReGraphv2 final_set.

Each pool row of <level>/<split>_final_set/train_dataset_pool gets the normalized IR text of the same
function (the corpus_text.py output: purify + normalize_file, the text ReGraphv2 tokenizes) and its
token_len under the Qwen tokenizer (no truncation; the trainer truncates). Token ids and graphs are
dropped. Row order is kept, so train_task_dataset and train_positive_map.pkl are copied unchanged.
Columns as in the earlier qwen-text final sets: binary_name, file_path, text, token_len, original_idx,
function_name, opt_level, origin_binary_name.

    python Scripts/shaheen/build_rell_text.py --level-dir /scratch/zhoul0e/Dataset-1-Oc-fused --split test \
        --text-roots /scratch/zhoul0e/bpe-corpus/Oc /scratch/zhoul0e/rell-text/Oc \
        --qwen-tokenizer /scratch/zhoul0e/iops/hf/Qwen3-Embedding-0.6B-tokenizer --out /scratch/zhoul0e/Dataset-1-Oc-rell
"""
import argparse
import os
import shutil
import sys

import datasets
import pyarrow.parquet as pq

COLUMNS = ["binary_name", "file_path", "text", "token_len", "original_idx", "function_name", "opt_level", "origin_binary_name"]


def module_of(file_path):
    """.../functions/<split>/<proj>/<binary>_functions/<sha1>.ll -> <split>/<proj>/<binary>"""
    return file_path.split("/functions/", 1)[1].rsplit("_functions/", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--text-roots", nargs="+", required=True, help="corpus_text.py outputs: <root>/<module>.parquet")
    ap.add_argument("--qwen-tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    # map() would otherwise write its cache-*.arrow files into the source final_set
    datasets.disable_caching()
    src = os.path.join(args.level_dir, f"{args.split}_final_set")
    pool = datasets.load_from_disk(os.path.join(src, "train_dataset_pool"))
    pool = pool.remove_columns([c for c in ("input_ids", "cfg_graph", "ddg_graph") if c in pool.column_names])
    needed = {}  # module -> function names in the pool; the corpus text holds every function of a module
    for path, name in zip(pool["file_path"], pool["function_name"]):
        needed.setdefault(module_of(path), set()).add(name)
    modules = sorted(needed)
    print(f"{args.split}: {len(pool):,} rows from {len(modules)} binaries", flush=True)

    text = {}
    for m in modules:
        path = next((os.path.join(r, m + ".parquet") for r in args.text_roots if os.path.exists(os.path.join(r, m + ".parquet"))), None)
        if path is None:
            sys.exit(f"no corpus text for {m}")
        t = pq.read_table(path, columns=["function_name", "text"])
        names = t.column(0).to_pylist()
        rows = [i for i, f in enumerate(names) if f in needed[m]]
        text[m] = dict(zip((names[i] for i in rows), t.column(1).take(rows).to_pylist()))

    def add_text(batch):
        mods = [module_of(p) for p in batch["file_path"]]
        out = [text[m].get(f) for m, f in zip(mods, batch["function_name"])]
        missing = [(m, f) for m, f, x in zip(mods, batch["function_name"], out) if x is None]
        if missing:
            raise ValueError(f"no text for {len(missing)} functions, e.g. {missing[0]}")
        return {"binary_name": mods, "text": out}

    pool = pool.map(add_text, batched=True, batch_size=2000)  # single process: the lookup table is in memory
    del text

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.qwen_tokenizer)

    def add_len(batch):
        return {"token_len": [len(x) for x in tok(batch["text"], add_special_tokens=False)["input_ids"]]}

    pool = pool.map(add_len, batched=True, batch_size=1000, num_proc=args.workers)
    pool = pool.select_columns(COLUMNS)

    local = os.path.join(args.local, f"rell_{args.split}_final_set")
    shutil.rmtree(local, ignore_errors=True)
    os.makedirs(local)
    pool.save_to_disk(os.path.join(local, "train_dataset_pool"), num_proc=min(args.workers, 32))
    shutil.copytree(os.path.join(src, "train_task_dataset"), os.path.join(local, "train_task_dataset"))
    shutil.copy2(os.path.join(src, "train_positive_map.pkl"), local)
    target = os.path.join(args.out, f"{args.split}_final_set")
    shutil.rmtree(target + ".part", ignore_errors=True)
    shutil.copytree(local, target + ".part")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(target + ".part", target)
    shutil.rmtree(local, ignore_errors=True)
    L = sorted(pool["token_len"])
    print(f"{args.split}: published {target}; token_len mean {sum(L) / len(L):.0f} median {L[len(L) // 2]} "
          f"p95 {L[int(len(L) * .95)]} >2048 {sum(x > 2048 for x in L) / len(L):.1%}", flush=True)


if __name__ == "__main__":
    main()
