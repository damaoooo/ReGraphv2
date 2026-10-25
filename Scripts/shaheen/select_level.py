"""DataProcess.select_samples over the fused parquet of ONE opt level (all three splits at once).

Reads <out>/parquet/<split>/**.parquet (file_path, untruncated input_ids), runs IR dedup, Step 5 and
Step 6, and writes the kept file paths to <out>/selection/<split>.txt, which build_split.py --keep reads.

    python Scripts/shaheen/select_level.py --out /scratch/zhoul0e/Dataset-1-Oc-sel --naming dataset1-xm
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SPLITS = ("train", "validation", "test")


def digests(path):
    import pyarrow.parquet as pq
    from DataProcess.select_samples import ir_digest

    table = pq.read_table(path, columns=["file_path", "input_ids"])
    return table.column("file_path").to_pylist(), [ir_digest(ids) for ids in table.column("input_ids").to_pylist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--naming", default="dataset1-xm")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    import pandas as pd
    from DataProcess.select_samples import select_samples

    out = Path(args.out).resolve()
    frames = {}
    with ProcessPoolExecutor(min(args.workers, 128)) as pool:
        for split in SPLITS:
            files = sorted(glob.glob(str(out / "parquet" / split / "**" / "*.parquet"), recursive=True))
            paths, hashes = [], []
            for p, h in pool.map(digests, files, chunksize=4):
                paths += p
                hashes += h
            frames[split] = pd.DataFrame({"file_path": paths, "ir_digest": hashes})
            print(f"{split}: {len(files)} binaries, {len(paths):,} samples", flush=True)
    masks = select_samples(frames, args.naming)

    (out / "selection").mkdir(exist_ok=True)
    summary = {}
    for split, frame in frames.items():
        kept = frame.loc[masks[split], "file_path"]
        path = out / "selection" / f"{split}.txt"
        path.with_suffix(".part").write_text("".join(p + "\n" for p in kept))
        os.replace(path.with_suffix(".part"), path)
        summary[split] = {"samples": len(frame), "kept": len(kept)}
    (out / "selection" / "summary.json").write_text(json.dumps({"naming": args.naming, **summary}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
