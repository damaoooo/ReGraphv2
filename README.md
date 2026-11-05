# ReLL

ReLL is a binary function similarity model that works on LLVM IR. Binaries are lifted to LLVM IR with an
IDA Pro / Hex-Rays based lifter and re-optimized with a fixed, conservative pass sequence (`-Oc`). Each
function is then encoded from its IR text with a ModernBERT encoder plus control-flow (CFG) and
data-dependency (DDG) graph branches, trained contrastively.

The lift + `-Oc` step is model-agnostic. The same IR can be fed to other text models, or recompiled to an
x86 object (`.re`) and given to any binary-only similarity tool.

## Branches

| branch | for |
|---|---|
| `modernBERT-gnn` (this one) | a single workstation: lifting, dataset building, training, evaluation |
| `modernBERT-cluster` | large-scale data generation on a Slurm cluster (sharded IDA / LLVM stages) |

## Layout

| path | contents |
|---|---|
| `Scripts/ida2llvm.py` | IDA 9 (idalib) microcode → LLVM IR lifter |
| `Scripts/task1_lift.py` … `task4_recompile.py`, `Scripts/pipeline.py` | data pipeline: lift → `-Oc` re-optimization → per-function extraction → `.re` recompilation |
| `LLVMTools/` | LLVM 18 plugins and tools: CFG / DDG exporters, metadata stripper, function splitter |
| `GraphBuilder/` | CFG / DDG graph construction aligned to tokens |
| `DataProcess/` | building the HuggingFace train / test datasets |
| `Tokenizer/` | IR normalizer and BPE tokenizer; the released tokenizer is in `Tokenizer/output_tokenizer/` |
| `Scripts/tokenizer/` | building a BPE corpus and retraining the tokenizer |
| `Model/`, `Pretrain/` | ReLL model and training |
| `evaluation.py`, `inference.py` | retrieval evaluation (Recall@K, MRR) and embedding inference |
| `Scripts/scoring/` | the frozen candidate-list sampler and the scorer shared by all compared methods |
| `Scripts/train/` | training and evaluation wrappers |
| `tests/` | unit and regression tests |

Large local data (datasets, runs, scores) belongs in `workspace/`, which git ignores.

## Setup

Linux x86_64, Python 3.13, managed with [uv](https://github.com/astral-sh/uv) (`pyproject.toml` / `uv.lock`):

```bash
uv sync --locked                       # core
uv sync --locked --extra pipeline      # + data generation
uv sync --locked --extra fa2           # + FlashAttention (needs a CUDA toolkit matching the Torch build)
cmake -S LLVMTools -B LLVMTools/build -DCMAKE_BUILD_TYPE=Release && cmake --build LLVMTools/build -j
```

These are installed separately: LLVM 18.1 (`opt`, `clang`, `llvm-extract`), Graphviz, and IDA Pro 9 with
Hex-Rays and idalib (licensed; needed only for lifting).

## Usage

```bash
# data: lift, re-optimize with -Oc, extract functions, recompile (see Scripts/README.md)
python Scripts/pipeline.py --help

# train + test on the Oc dataset
bash Scripts/train/train_test_fused_opt.sh Oc

# score any method's embeddings on a published candidate list
python Scripts/scoring/score_candidates.py <lists>/<subset> <embeddings.npy> <out_root> <label>
```

More detail: `Scripts/README.md`, `DataProcess/README.md`, `docs/TRAINING_STEPS_GUIDE.md`.

## Tests

```bash
uv run python -m pytest tests -q
```
