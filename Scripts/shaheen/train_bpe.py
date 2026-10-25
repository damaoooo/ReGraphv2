"""Train the LLVM IR BPE tokenizer on the parquet corpus written by corpus_text.py (Shaheen run).

Tokenizer, trainer and special tokens come from Tokenizer/bpe_tokenizer.py; this script only streams the
parquet corpus (instead of one in-memory list) and samples it reproducibly. On the Dataset-1 train corpus
use --sample-fraction 0.25: tokenizers counts pairs in 32 bits and "%." would overflow (see bpe_tokenizer.py).

    python Scripts/shaheen/train_bpe.py --corpus /scratch/zhoul0e/bpe-corpus --levels O0 O1 O2 O3 Os Oz Oc \
        --out <dir> --sample-fraction 0.25 --seed 0
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Tokenizer.bpe_tokenizer import (  # noqa: E402
    DEFAULT_MIN_FREQUENCY, DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS, add_post_processor, create_bpe_tokenizer,
    create_bpe_trainer,
)


def iter_text(files, fraction=1.0, seed=0, batch_rows=10000):
    """Texts of the corpus; with fraction < 1 a reproducible uniform sample (per-file RNG seeded by path)."""
    for path in files:
        rng = random.Random(seed ^ zlib.crc32(path.encode()))
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
            texts = batch.column(0).to_pylist()
            yield texts if fraction >= 1.0 else [t for t in texts if rng.random() < fraction]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    ap.add_argument("--name", default="llvm_ir_bpe")
    ap.add_argument("--sample-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for lvl in args.levels for f in glob.glob(f"{args.corpus}/{lvl}/**/*.parquet", recursive=True))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{len(files)} parquet files, {total:,} functions, levels {args.levels}", flush=True)

    tokenizer = create_bpe_tokenizer()
    trainer = create_bpe_trainer(args.vocab_size, args.min_frequency, show_progress=False)
    t0 = time.time()
    tokenizer.train_from_iterator(iter_text(files, args.sample_fraction, args.seed), trainer=trainer,
                                  length=int(total * args.sample_fraction))
    print(f"trained in {time.time() - t0:.0f}s, vocab {tokenizer.get_vocab_size()}", flush=True)
    add_post_processor(tokenizer)

    os.makedirs(args.out, exist_ok=True)
    tok_path = os.path.join(args.out, f"{args.name}.json")
    vocab_path = os.path.join(args.out, f"{args.name}_vocab.json")
    tokenizer.save(tok_path)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, f"{args.name}_config.json"), "w") as f:
        json.dump({
            "model_type": "BPE", "vocab_size": args.vocab_size, "min_frequency": args.min_frequency,
            "special_tokens": SPECIAL_TOKENS, "pre_tokenizer": "Whitespace",
            "tokenizer_path": os.path.join(os.path.basename(args.out), f"{args.name}.json"),
            "vocab_path": os.path.join(os.path.basename(args.out), f"{args.name}_vocab.json"),
            "corpus": {"source": "Dataset-1 train split (clamav, curl, openssl, unrar), purified + normalize_file text",
                       "levels": args.levels, "functions": total, "function_markers": False,
                       "sample_fraction": args.sample_fraction, "sample_seed": args.seed,
                       "sample_reason": "tokenizers BpeTrainer pair counts are int32; full-corpus '%.' count ~3.7e9 overflows"},
        }, f, indent=2)
    print(f"saved {tok_path}", flush=True)


if __name__ == "__main__":
    main()
