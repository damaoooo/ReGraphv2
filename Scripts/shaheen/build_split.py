"""Steps 5-6 for one split of one opt level, staged on node-local storage:
fused parquet -> raw HF dataset (DataProcess schema, rows in function-path order)
[-> only the rows --keep lists, i.e. <out>/selection/<split>.txt of select_level.py]
-> python -m DataProcess.wash -> python -m Pretrain.split_train_validation --naming dataset1 --train-ratio 1.0
-> <out>/<split>_final_set. HF reads/writes are slow on Lustre, so every intermediate stays on --local.

    python Scripts/shaheen/build_split.py --out <out> --split train --local $TMPDIR/final --workers 192
"""
import os

os.environ.setdefault("USE_TORCH", "0")

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--naming", default="dataset1")
    ap.add_argument("--keep", help="file of the file_paths to keep (select_level.py output)")
    args = ap.parse_args()

    import datasets
    from DataProcess.builder import get_dataset_features

    out, local = Path(args.out).resolve(), Path(args.local).resolve()
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    split_root = out / "parquet" / args.split
    import pyarrow.parquet as pq

    # a binary with no selected function has an empty parquet, which datasets cannot read
    files = [f for f in glob.glob(str(split_root / "**" / "*.parquet"), recursive=True)
             if pq.read_metadata(f).num_rows]
    # Each file holds one binary's rows sorted by path; ordering files by their <binary>_functions/
    # directory makes the concatenation sorted by function path, as DataProcess orders its rows.
    files.sort(key=lambda f: os.path.relpath(f, split_root)[: -len(".parquet")] + "_functions/")
    print(f"{args.split}: {len(files)} parquet files", flush=True)

    raw = datasets.Dataset.from_parquet(files, features=get_dataset_features(), cache_dir=str(local / "cache"))
    if args.keep:
        keep = set(Path(args.keep).read_text().split())
        rows = [i for i, path in enumerate(raw["file_path"]) if path in keep]
        if len(rows) != len(keep):
            sys.exit(f"{args.keep} lists {len(keep)} samples, {len(rows)} of them are in the parquet")
        raw = raw.select(rows)
    raw.save_to_disk(str(local / "raw"), num_proc=min(args.workers, 64))
    print(f"{args.split}: raw dataset {len(raw)} samples", flush=True)
    del raw
    shutil.rmtree(local / "cache", ignore_errors=True)

    py = sys.executable
    run([py, "-m", "DataProcess.wash", "--dataset-path", local / "raw", "--output-path", local / "wash",
         "--num-proc", args.workers])
    shutil.rmtree(local / "raw")
    final_local = local / f"{args.split}_final_set"
    run([py, "-m", "Pretrain.split_train_validation", local / "wash", "--naming", args.naming,
         "--train-ratio", "1.0", "--output-dir", final_local])

    target = out / f"{args.split}_final_set"
    shutil.rmtree(str(target) + ".part", ignore_errors=True)
    shutil.copytree(final_local, str(target) + ".part")
    shutil.rmtree(target, ignore_errors=True)
    os.replace(str(target) + ".part", target)
    shutil.rmtree(local, ignore_errors=True)
    print(f"{args.split}: published {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
