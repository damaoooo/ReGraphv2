# ReLL Artifact Evaluation Guide

This branch accompanies *Semantic Normalization for Binary Function Similarity
Detection: Do We Always Need Large Models?*  It contains code, scripts, and
documentation.  Dataset and model files are packaged separately because they
are too large for Git.

## Scope

The default artifact reproduces the main ReLL retrieval result on Dataset-1
with the paper's `Oc` semantic normalization setting:

- Dataset: `IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup`
- Model: `runs/dataset1_oc_fused/model_cfg_ddg`
- Metric: Recall@1 and MRR@P at retrieval pool size 10,000
- Expected result: Recall@1 about `0.648`, MRR@P about `0.758`

The release also includes scripts for full preprocessing, graph ablation,
optimization ablation, Dataset-Vulnerability, and latency benchmarking.  Those
larger experiments require substantially more compute and a configured lifting
backend.  Ghidra is provided for open-source lifting; the historical IDA backend
and IDA latency benchmark remain available.

## External Artifact Packages

The data and weights are provided as separate archives. Anonymous download
links will be listed in the main README after upload.

| Archive | Size | SHA-256 |
| --- | ---: | --- |
| `rell_sec27_data_2026-08-24.tar.zst` | 3.5 GB | `249f38e2f34a9a5b0ecb24ed6e5f1217d672a6bd2661a4206cfe962ef4a860dc` |
| `rell_sec27_weights_2026-08-24.tar.zst` | 1.3 GB | `7f87553a4ca55db9b16b90bd54a70c6fc9f49df2df929dff432933f4995ffdd5` |

The data archive contains:

```text
IR/Dataset-1-Oc-fused/train_final_set
IR/Dataset-1-Oc-fused/validation_final_set
IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
IR/Dataset-1-Oc-qwen-text-fused/train_final_set
IR/Dataset-1-Oc-qwen-text-fused/validation_final_set
IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup
IR/Dataset-1-ASM-Qwen-text/train_final_set
IR/Dataset-1-ASM-Qwen-text/validation_final_set
IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text
```

The weights archive contains:

```text
runs/dataset1_oc_fused/model_cfg_ddg
case_studies/qwen3_embedding/weights/qwen3_0p6b_ir_lora
case_studies/qwen3_embedding/weights/qwen3_0p6b_asm_lora
```

Intermediate checkpoints, optimizer states, merged copies of the Qwen base
model, and cached evaluation embeddings are not included. They are not needed
for the main reproduction path. Frozen result tables remain in the source
repository next to the scripts that consume them.

The Dataset-1 `Oc` final sets are already filtered for short/uninformative
functions and exact input duplicates.  The test data flow is:

```text
264,548 functions listed by the Dataset-1 test metadata
261,962 functions matched to available lifted inputs
212,402 encoded records after extraction and exact input deduplication
211,018 records after removing 1,384 functions with at most 128 tokens
164,969 query anchors with at least one valid positive match
```

The final main test split therefore contains:

```text
pool examples: 211,018
anchors:       164,969
```

Download both archives, their manifests, and `SHA256SUMS`. Verify the downloads
on Linux with:

```bash
sha256sum -c SHA256SUMS
```

On macOS, use `shasum -a 256 -c SHA256SUMS`. Then unpack both archives from the
repository root:

```bash
cd /path/to/rell
tar -I zstd -xf /path/to/rell_sec27_data_2026-08-24.tar.zst -C .
tar -I zstd -xf /path/to/rell_sec27_weights_2026-08-24.tar.zst -C .
```

The two downloads occupy about 4.8 GB and expand to about 30 GB. At least 40 GB
of available disk space is recommended while downloading and extracting them.

## Environment

Use a Python environment with CUDA PyTorch, torch-geometric, transformers,
datasets, pygraphviz, llvmlite, bitsandbytes, and xformers.  The local
development environment used `conda`, but the scripts do not require a fixed
environment name.

```bash
cd /path/to/rell
python -m pip install -r requirements.txt
```

`pygraphviz` may require the system Graphviz development package.  Open-source
lifting uses Ghidra through `Scripts/pcode2llvm.py`; set `GHIDRA_HOME` or
`GHIDRA_ANALYZE_HEADLESS`, or pass the path through the command line.  IDA Pro
is still supported as the historical backend; its default local path in the
scripts is `/path/to/ida-pro/idat`.

## Main Evaluation

After unpacking both artifact packages, run:

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
  --repo-root /path/to/rell \
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

The checked result snapshot is included in the source repository as:

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
ReLL `Oc` model.

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

The source repository includes the frozen result snapshots:

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

## Qwen3 Case Study

The model-scale case study compares Qwen3-Embedding on normalized IR and raw
assembly, with and without task-specific fine-tuning. Its code, configuration,
frozen result tables, and plotting script are documented in:

```text
case_studies/qwen3_embedding/README.md
```

The five paper settings are 0.6B on IR without fine-tuning, 0.6B on IR with
fine-tuning, 0.6B on assembly without fine-tuning, 0.6B on assembly with
fine-tuning, and 4B on assembly without fine-tuning. To regenerate the two
curves from the frozen tables, run:

```bash
python case_studies/qwen3_embedding/scripts/plot_case_study.py
```

The Qwen base models are downloaded from Hugging Face. The external weights
archive provides the two fine-tuned 0.6B LoRA adapters, and the external data
archive provides the processed IR and assembly final sets. The base 0.6B and 4B
models are not duplicated in the artifact.

## Rebuilding a Core External Package

Run the package builder with a Python environment that includes `datasets`:

```bash
bash Scripts/pack_artifact_release.sh \
  --output-dir /path/to/artifact_output \
  --python /path/to/python
```

This maintainer utility builds the core ReLL data/model bundle. Reviewers do not
need to run it. Before compression, it rewrites author-specific paths in saved
Arrow datasets and text metadata, scans the staged package for identity
markers, and records the archive's SHA-256 digest in the manifest.

## Notes

- The source release contains code and documentation only.
- Dataset/model artifacts are intentionally distributed separately.
- `Oc2` and later exploratory settings are not part of this release workflow.
- The model directory does not need `config.json`; `evaluation.py` falls back to
  the repository's `PretrainConfig` and loads `pytorch_model.bin`.
