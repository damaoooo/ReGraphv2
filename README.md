# ReGraphv2

ReGraphv2 is a research prototype for binary function similarity detection
(BFSD).  It normalizes binary functions through LLVM IR before learning function
embeddings, with the goal of reducing architecture and compiler-optimization
differences without relying on a larger general-purpose model.

The main pipeline is:

```text
binary
  -> IDA/ida2llvm lifting
  -> LLVM IR re-optimization / canonicalization
  -> function-level IR extraction
  -> token, CFG, and DDG construction
  -> RoFormer + graph-branch contrastive embedding
  -> retrieval evaluation
```

The artifact-evaluation release focuses on the `Oc` normalization setting used
for the main paper results.  Code is kept in GitHub; datasets and model weights
are packaged separately because they are large.

## What This Repository Contains

- `Scripts/`: lifting, re-optimization, function extraction, dataset generation,
  training/evaluation wrappers, latency benchmarking, and Dataset-Vulnerability
  evaluation scripts.
- `DataProcess/`: conversion from extracted LLVM IR functions into HuggingFace
  final-set datasets.
- `GraphBuilder/`: CFG and DDG graph construction over normalized LLVM IR token
  spans.
- `Tokenizer/`: LLVM IR tokenizer and normalizer.
- `Model/`: RoFormer backbone and CFG/DDG GATv2 graph branches.
- `Pretrain/`: MoCo-style contrastive training code and configuration.
- `evaluation.py`: Dataset-1 retrieval evaluation.
- `inference.py` and `api_server.py`: single-binary inference and API service.
- `ARTIFACT_EVALUATION.md`: detailed reproduction instructions.

## Method Summary

ReGraphv2 first lifts binaries from different architectures into LLVM IR.  It
then applies conservative re-optimization passes (`Oc`) that are intended to
canonicalize low-level IR without introducing aggressive semantic collapse.  The
model consumes normalized LLVM IR tokens and augments the text representation
with CFG and DDG graph branches.  Training uses a RoFormer encoder with MoCo
contrastive learning and an auxiliary masked-language-model objective.

The `Oc` pass list is defined in:

```text
Scripts/task2_reoptimize.py
Scripts/ray_opt_ablation/ray_fused_pipeline.py
```

It uses memory/SSA cleanup, CFG cleanup, scalar canonicalization, and dead-code
elimination.  It intentionally avoids loop optimizations and interprocedural
optimizations.

## Environment

Use a Python environment with CUDA PyTorch and the dependencies in
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

`pygraphviz` may require the system Graphviz development package.  Full lifting
requires IDA Pro; preprocessed final-set data is provided in the external
artifact package for normal evaluation.

## External Artifact Package

The compact artifact package contains the data and weights needed for the main
Dataset-1 evaluation:

```text
IR/Dataset-1-Oc-fused/train_final_set
IR/Dataset-1-Oc-fused/validation_final_set
IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
runs/dataset1_oc_fused/model_cfg_ddg
selected result markdown snapshots
```

After downloading the package, unpack it from the repository root:

```bash
tar -I zstd -xf regraphv2_artifact_core.tar.zst -C /path/to/regraphv2
```

If your package was created with gzip fallback, use `tar -zxf` instead.

## Quick Reproduction

After unpacking the external artifact package, run:

```bash
mkdir -p runs/artifact_eval

python evaluation.py \
  IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup/train_positive_map.pkl \
  --dataset-path IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup/train_dataset_pool \
  --model-path runs/dataset1_oc_fused/model_cfg_ddg \
  --max-length 2048 \
  --batch-size 16 \
  --gpu-batch-size 256 \
  --eval-samples 0 \
  --pool-samples 0 \
  --embeddings-path runs/artifact_eval/oc_len128_hashdedup_embeddings.pth \
  --markdown-output runs/artifact_eval/oc_len128_hashdedup_results.md \
  --cfg \
  --ddg \
  --bf16
```

Expected main-result scale at Pool Size 10,000:

```text
Recall@1 ~= 0.648
MRR@P    ~= 0.758
```

See `ARTIFACT_EVALUATION.md` for the full set of reproduction commands,
including training, graph ablation, optimization ablation, Dataset-Vulnerability,
and latency benchmarking.

## Training From Packaged Data

The packaged train and validation final sets are enough to retrain the main
CFG+DDG model:

```bash
PYTHON_CMD="python" bash Scripts/train_test_fused_opt.sh Oc \
  --output-root runs/dataset1_oc_retrain \
  --resume
```

For paper-matching evaluation, evaluate the retrained model on
`IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup`.

## Full Preprocessing

Full preprocessing is expensive and requires IDA Pro.  The large-scale fused
Dataset-1 pipeline is:

```bash
python Scripts/ray_opt_ablation/ray_fused_pipeline.py \
  --repo-root /path/to/regraphv2 \
  --dataset-path IR/Dataset-1-new/Dataset-1 \
  --output-path IR/Dataset-1-Oc-fused \
  --opt-level Oc \
  --resume \
  --task3-csv-filter-dir IR/csv_list \
  --task3-prefilter-uninformative \
  --task3-prefilter-max-stub-tokens 128 \
  --task3-dedup-input-ids \
  --final-uninformative-filter \
  --final-uninformative-filter-splits train,validation,test \
  --final-uninformative-max-stub-tokens 128
```

For smaller binary-in use cases, `Scripts/pipeline.py` exposes the four
traditional stages: lifting, re-optimization, function extraction, and optional
recompilation.

## Additional Experiments

- Graph ablation: `Scripts/train_test_oc_graph_ablation.sh`
- Optimization ablation: `Scripts/train_test_fused_opt.sh`
- Dataset-Vulnerability: `Scripts/evaluate_dataset_vulnerability_regraph.py`
- Latency benchmark: `Scripts/benchmark_pipeline_latency.py`

Latency code is included in the release, but local timing results are not part
of the compact data package.  See `Scripts/PIPELINE_LATENCY_BENCHMARK.md`.

## Notes

- The GitHub release branch intentionally excludes datasets, model checkpoints,
  and run caches.
- `Oc2` and later exploratory settings are not part of this release workflow.
- `runs/dataset1_oc_fused/model_cfg_ddg` does not need a standalone
  `config.json`; `evaluation.py` falls back to the repository's
  `PretrainConfig` before loading `pytorch_model.bin`.
