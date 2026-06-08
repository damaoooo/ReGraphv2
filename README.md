# ReGraphv2

> **Semantic normalization for binary function similarity detection.**
> Lift binaries to LLVM IR, canonicalize away architecture/compiler noise, then
> learn retrieval-ready function embeddings with IR tokens + CFG/DDG graphs.

---

## ✨ Why ReGraphv2?

Binary Function Similarity Detection (BFSD) is hard because the same source
function can look very different after:

- different ISAs: x86, ARM, MIPS, 32/64-bit;
- different compilers and compiler versions;
- different optimization levels: `O0/O1/O2/O3/Og/Os`;
- binary-level artifacts introduced by lifting and analysis tools.

ReGraphv2 attacks the problem before the model sees the function:

```text
Binary function
   ↓
IDA / ida2llvm lifting
   ↓
LLVM IR semantic normalization (`Oc`)
   ↓
Function-level IR + CFG + DDG
   ↓
RoFormer + graph branches + MoCo contrastive learning
   ↓
Function embedding for 1-to-N retrieval
```

The central idea is simple: **do not ask a larger model to memorize every
binary surface form; first normalize the semantics into a cleaner IR space.**

---

## 🚀 What You Can Do With This Release

This `release` branch is prepared for artifact evaluation.  It contains code,
scripts, and documentation.  Large datasets and model weights are distributed as
a separate artifact package.

| Goal | Entry Point | Notes |
| --- | --- | --- |
| Reproduce the main Dataset-1 retrieval result | `evaluation.py` | Uses packaged `Oc` final set + trained model |
| Retrain ReGraphv2 on packaged data | `Scripts/train_test_fused_opt.sh Oc` | Uses train/validation final sets |
| Rebuild Dataset-1 final sets | `Scripts/ray_opt_ablation/ray_fused_pipeline.py` | Expensive; requires prepared IR and compute |
| Run graph ablations | `Scripts/train_test_oc_graph_ablation.sh` | IR-only, IR+CFG, IR+DDG, IR+CFG+DDG |
| Evaluate Dataset-Vulnerability | `Scripts/evaluate_dataset_vulnerability_regraph.py` | Binary-in ranking workflow |
| Benchmark preprocessing latency | `Scripts/benchmark_pipeline_latency.py` | Requires IDA Pro |

For the full artifact-evaluation checklist, see:

📄 **[`ARTIFACT_EVALUATION.md`](ARTIFACT_EVALUATION.md)**

---

## 🧠 Method at a Glance

### 1. Lift Once, Compare in IR

ReGraphv2 uses IDA/ida2llvm to lift binary functions into LLVM IR.  This gives a
shared intermediate representation across architectures.

### 2. Canonicalize With `Oc`

Instead of blindly applying `-O3`, ReGraphv2 uses a conservative
canonicalization profile called `Oc`.

`Oc` keeps passes that are useful for semantic cleanup:

- memory/SSA cleanup;
- CFG simplification;
- scalar canonicalization;
- dead-code elimination.

It avoids aggressive loop and interprocedural optimizations that can collapse
distinct functions or over-specialize the IR.

The pass list lives in:

```text
Scripts/task2_reoptimize.py
Scripts/ray_opt_ablation/ray_fused_pipeline.py
```

### 3. Encode Text + Graphs

The model combines:

- LLVM IR tokens through a RoFormer encoder;
- CFG span graphs through a GATv2 branch;
- DDG span graphs through a GATv2 branch;
- MoCo-style contrastive training for retrieval embeddings.

---

## 📁 Repository Map

```text
Scripts/              pipeline, training wrappers, AE scripts, benchmarks
DataProcess/          final_set construction from function-level LLVM IR
GraphBuilder/         CFG/DDG graph extraction over token spans
Tokenizer/            LLVM IR tokenizer and normalizer
Model/                RoFormer + CFG/DDG graph branches
Pretrain/             MoCo training loop and config
evaluation.py         Dataset-1 retrieval evaluation
inference.py          single-binary inference pipeline
api_server.py         FastAPI embedding service
ARTIFACT_EVALUATION.md detailed reproduction guide
```

---

## 📦 External Artifact Package

The compact artifact package contains the minimal data and model files needed
for the main reproduction path:

```text
IR/Dataset-1-Oc-fused/train_final_set
IR/Dataset-1-Oc-fused/validation_final_set
IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
runs/dataset1_oc_fused/model_cfg_ddg
selected result markdown snapshots
```

Unpack it from the repository root:

```bash
tar -I zstd -xf regraphv2_artifact_core.tar.zst -C /path/to/regraphv2
```

If the archive was created with gzip fallback:

```bash
tar -zxf regraphv2_artifact_core.tar.gz -C /path/to/regraphv2
```

The packaged main test split contains:

```text
pool examples: 211,018
anchors:       164,969
```

---

## ⚙️ Environment

Create or activate a Python environment with CUDA PyTorch, then install the
project dependencies:

```bash
python -m pip install -r requirements.txt
```

Notes:

- `pygraphviz` may require the system Graphviz development package.
- Full lifting requires IDA Pro.
- The default local IDA path in scripts is `/home/damaoooo/ida-pro-9.3/idat`;
  adjust it for your machine if you run lifting.
- The quick reproduction path does **not** require re-lifting binaries.

---

## ⚡ Quick Reproduction

After unpacking the artifact package:

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

Expected scale at retrieval pool size 10,000:

| Metric | Expected |
| --- | ---: |
| Recall@1 | ~0.648 |
| MRR@P | ~0.758 |

The output Markdown will be written to:

```text
runs/artifact_eval/oc_len128_hashdedup_results.md
```

---

## 🔁 Retraining From Packaged Data

The package includes train and validation final sets, so you can retrain the
main CFG+DDG model without rebuilding the dataset:

```bash
PYTHON_CMD="python" bash Scripts/train_test_fused_opt.sh Oc \
  --output-root runs/dataset1_oc_retrain \
  --resume
```

For paper-matching evaluation, test the retrained model on:

```text
IR/Dataset-1-Oc-fused/test_final_set_len128_hashdedup
```

---

## 🏗️ Full Preprocessing Pipeline

Full preprocessing is intended for advanced reproduction.  It is expensive and
requires IDA Pro plus substantial CPU/storage.

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

For smaller binary-in workflows, use:

```bash
python Scripts/pipeline.py --help
```

That pipeline exposes the traditional stages: lifting, re-optimization,
function extraction, and optional recompilation.

---

## 🧪 Additional Experiments

| Experiment | Script |
| --- | --- |
| CFG/DDG graph ablation | `Scripts/train_test_oc_graph_ablation.sh` |
| Optimization-level ablation | `Scripts/train_test_fused_opt.sh` |
| Dataset-Vulnerability ranking | `Scripts/evaluate_dataset_vulnerability_regraph.py` |
| Bit-width fusion for vulnerability results | `Scripts/evaluate_dataset_vulnerability_bitnorm_fusion.py` |
| Pipeline latency benchmark | `Scripts/benchmark_pipeline_latency.py` |

Latency code is included, but local timing results are not shipped in the
compact package.  See:

📄 **[`Scripts/PIPELINE_LATENCY_BENCHMARK.md`](Scripts/PIPELINE_LATENCY_BENCHMARK.md)**

---

## 🧾 Artifact Packaging

To rebuild the compact artifact package locally:

```bash
bash Scripts/pack_artifact_release.sh \
  --output-dir /path/to/artifact_output
```

The generated archive is meant for external storage such as Google Drive, not
for committing to GitHub.

---

## 📌 Release Notes

- This branch intentionally excludes datasets, model checkpoints, and run
  caches from Git.
- `Oc2` and later exploratory settings are not part of the release workflow.
- `runs/dataset1_oc_fused/model_cfg_ddg` does not need a standalone
  `config.json`; `evaluation.py` falls back to the repository's
  `PretrainConfig` before loading `pytorch_model.bin`.

---

## Citation

If you use this artifact, cite the corresponding ReGraphv2 paper version.  The
release branch is designed to make the main claims auditable through code,
packaged final sets, model weights, and deterministic evaluation commands.
