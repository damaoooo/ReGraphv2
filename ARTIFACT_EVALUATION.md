# ReGraphv2 Artifact Evaluation Guide

This branch is the artifact-evaluation release for ReGraphv2.  The GitHub
branch contains code, scripts, and documentation only.  Dataset and model files
are packaged separately and should be distributed through external storage.

## Scope

The default artifact reproduces the main ReGraphv2 retrieval result on Dataset-1
with the paper's `Oc` semantic normalization setting:

- Dataset: `IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup`
- Model: `runs/dataset1_oc_fused/model_cfg_ddg`
- Metric: Recall@1 and MRR@P at retrieval pool size 10,000
- Expected result: Recall@1 about `0.648`, MRR@P about `0.758`

The release also includes scripts for full preprocessing, graph ablation,
optimization ablation, Dataset-Vulnerability, and latency benchmarking.  Those
larger experiments require substantially more compute and/or IDA Pro.

## External Artifact Package

The core package produced by `Scripts/pack_artifact_release.sh` has this layout:

```text
IR/Dataset-1-Oc-fused/train_final_set
IR/Dataset-1-Oc-fused/validation_final_set
IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
runs/dataset1_oc_fused/model_cfg_ddg
runs/dataset1_oc_fused/oc_test_results_len128_hashdedup.md
runs/dataset1_oc_fused/common_oc_csv_results.md
runs/dataset1_oc_graph_ablation/seed_eval_10_summary.md
runs/dataset_vulnerability_regraph/results_bitnorm_fusion_max.md
runs/dataset_vulnerability_regraph/vuln_search_big_table.md
```

The Dataset-1 `Oc` final sets are already filtered for short/uninformative
functions and exact hash collisions.  The main test split contains:

```text
pool examples: 211,018
anchors:       164,969
```

Unpack the package from the repository root:

```bash
tar -I zstd -xf regraphv2_artifact_core.tar.zst -C /path/to/regraphv2
```

If the package was created with gzip fallback, replace `-I zstd` with `-z`.

## Environment

Use a Python environment with CUDA PyTorch, torch-geometric, transformers,
datasets, pygraphviz, llvmlite, bitsandbytes, and xformers.  The local
development environment used `conda`, but the scripts do not require a fixed
environment name.

```bash
cd /path/to/regraphv2
python -m pip install -r requirements.txt
```

`pygraphviz` may require the system Graphviz development package.  Full lifting
requires IDA Pro; the default local path in the scripts is
`/home/damaoooo/ida-pro-9.3/idat`.

## Main Evaluation

After unpacking the core artifact package, run:

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

The generated Markdown should report Pool Size `10,000` with Recall@1 close to
`0.648` and MRR@P close to `0.758`.

If embeddings already exist, the same command reuses the cache.  Delete
`runs/artifact_eval/oc_len128_hashdedup_embeddings.pth` to force regeneration.

## Training From Packaged Data

The packaged train and validation final sets are sufficient to retrain the main
model:

```bash
PYTHON_CMD="python" bash Scripts/train_test_fused_opt.sh Oc \
  --output-root runs/dataset1_oc_retrain \
  --resume
```

That wrapper uses `IR/Dataset-1-Oc-fused/train_final_set` and
`IR/Dataset-1-Oc-fused/validation_final_set`.  For paper-matching test
evaluation, run the explicit `evaluation.py` command above on
`test_final_set_len128_hashdedup`.

## Full Dataset Generation

Use this only when rebuilding Dataset-1 from lifted IR.  It is not the default
AE path because it is expensive.

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

The `Oc` pass list is defined in both:

- `Scripts/task2_reoptimize.py`
- `Scripts/ray_opt_ablation/ray_fused_pipeline.py`

The selected pass categories are memory/SSA cleanup, CFG cleanup, scalar
canonicalization, and dead-code elimination.  Loop optimizations and
interprocedural optimizations are intentionally excluded.

## Graph Ablation

To retrain and evaluate IR-only, IR+CFG, and IR+DDG ablations:

```bash
bash Scripts/train_test_oc_graph_ablation.sh all \
  --dataset-root IR/Dataset-1-Oc-fused \
  --test-set IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup \
  --output-root runs/dataset1_oc_graph_ablation \
  --resume
```

The checked local result snapshot is packaged as:

```text
runs/dataset1_oc_graph_ablation/seed_eval_10_summary.md
```

## Optimization Ablation

The optimization-level ablation requires separate generated final sets and
trained models for `O0/O1/O2/O3/Og/Os/Oc`.  The wrapper is:

```bash
bash Scripts/train_test_fused_opt.sh O0 --resume
bash Scripts/train_test_fused_opt.sh O1 --resume
bash Scripts/train_test_fused_opt.sh O2 --resume
bash Scripts/train_test_fused_opt.sh O3 --resume
bash Scripts/train_test_fused_opt.sh Og --resume
bash Scripts/train_test_fused_opt.sh Os --resume
bash Scripts/train_test_fused_opt.sh Oc --resume
```

This experiment is not included in the compact data package because the
optimization-specific final sets are large.

## Dataset-Vulnerability

Dataset-Vulnerability is evaluated by a separate binary-in ranking script.  It
uses the official candidate pool under `binary_function_similarity` and the
ReGraphv2 `Oc` model.

```bash
python Scripts/evaluate_dataset_vulnerability_regraph.py \
  --bfs-root /path/to/binary_function_similarity \
  --run-dir runs/dataset_vulnerability_regraph \
  --model-run runs/dataset1_oc_fused \
  --workers 8

python Scripts/evaluate_dataset_vulnerability_regraph.py \
  --bfs-root /path/to/binary_function_similarity \
  --run-dir runs/dataset_vulnerability_regraph \
  --model-run runs/dataset1_oc_fused \
  --skip-pipeline \
  --bitnorm i32

python Scripts/evaluate_dataset_vulnerability_regraph.py \
  --bfs-root /path/to/binary_function_similarity \
  --run-dir runs/dataset_vulnerability_regraph \
  --model-run runs/dataset1_oc_fused \
  --skip-pipeline \
  --bitnorm i64

python Scripts/evaluate_dataset_vulnerability_bitnorm_fusion.py \
  --run-dir runs/dataset_vulnerability_regraph \
  --mode max
```

The compact artifact package includes result snapshots:

```text
runs/dataset_vulnerability_regraph/results_bitnorm_fusion_max.md
runs/dataset_vulnerability_regraph/vuln_search_big_table.md
```

## Latency Benchmark

Latency benchmarking code is included for AE, but the compact package does not
ship local timing results.  See:

```text
Scripts/PIPELINE_LATENCY_BENCHMARK.md
Scripts/benchmark_pipeline_latency.py
Scripts/ida_latency_worker.py
```

The benchmark separates IDA database loading from steady-state function lifting.
For the paper-style 100-function table, sample many functions from one binary or
a small number of binaries.

## Notes

- GitHub release branch contains code/docs only.
- Dataset/model artifacts are intentionally distributed separately.
- `Oc2` and later exploratory settings are not part of this release workflow.
- The model directory does not need `config.json`; `evaluation.py` falls back to
  the repository's `PretrainConfig` and loads `pytorch_model.bin`.
