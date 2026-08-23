# ReLL

> Artifact for *Semantic Normalization for Binary Function Similarity
> Detection: Do We Always Need Large Models?*
>
> Lift binaries to LLVM IR, canonicalize away architecture/compiler noise, then
> learn retrieval-ready function embeddings with IR tokens + CFG/DDG graphs.

---

## ✨ Why ReLL?

Binary Function Similarity Detection (BFSD) is hard because the same source
function can look very different after:

- different ISAs: x86, ARM, MIPS, 32/64-bit;
- different compilers and compiler versions;
- different optimization levels: `O0/O1/O2/O3/Og/Os`;
- binary-level artifacts introduced by lifting and analysis tools.

ReLL attacks the problem before the model sees the function:

```text
Binary function
   ↓
Ghidra High P-Code / pcode2llvm lifting
   or IDA / ida2llvm lifting
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

This artifact release is prepared for evaluation. It contains code,
scripts, and documentation.  Large datasets and model weights are distributed as
a separate artifact package.

| Goal | Entry Point | Notes |
| --- | --- | --- |
| Reproduce the main Dataset-1 retrieval result | `evaluation.py` | Uses packaged `Oc` final set + trained model |
| Retrain ReLL on packaged data | `Scripts/train_test_fused_opt.sh Oc` | Uses train/validation final sets |
| Rebuild Dataset-1 final sets | `Scripts/ray_opt_ablation/ray_fused_pipeline.py` | Expensive; requires prepared IR and compute |
| Lift new binaries without IDA | `Scripts/pcode2llvm.py` or `Scripts/pipeline.py task1 --backend ghidra` | Uses open-source Ghidra High P-Code |
| Run graph ablations | `Scripts/train_test_oc_graph_ablation.sh` | IR-only, IR+CFG, IR+DDG, IR+CFG+DDG |
| Evaluate Dataset-Vulnerability | `Scripts/evaluate_dataset_vulnerability_regraph.py` | Binary-in ranking workflow |
| Benchmark preprocessing latency | `Scripts/benchmark_pipeline_latency.py` | Requires IDA Pro |
| Reproduce the Qwen3 case study | `case_studies/qwen3_embedding/README.md` | Five IR/assembly and fine-tuning settings |

For the full artifact-evaluation checklist, see:

📄 **[`ARTIFACT_EVALUATION.md`](ARTIFACT_EVALUATION.md)**

---

## 🧠 Method at a Glance

### 1. Lift Once, Compare in IR

ReLL lifts binary functions into LLVM IR, giving the model a shared
intermediate representation across architectures.

For artifact evaluation, the release includes an open-source Ghidra path:

```text
Binary
  -> Ghidra decompiler High P-Code
  -> Scripts/pcode2llvm.py
  -> LLVM IR
```

The older IDA/ida2llvm path is still supported for continuity with the original
experiments, but it is no longer the only way to reproduce the lifting stage.

### 2. Canonicalize With `Oc`

Instead of blindly applying `-O3`, ReLL uses a conservative
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
Scripts/pcode2llvm.py Ghidra High P-Code to LLVM IR lifter
Scripts/ghidra/       Ghidra headless export scripts
DataProcess/          final_set construction from function-level LLVM IR
GraphBuilder/         CFG/DDG graph extraction over token spans
Tokenizer/            LLVM IR tokenizer and normalizer
Model/                RoFormer + CFG/DDG graph branches
Pretrain/             MoCo training loop and config
case_studies/         Qwen3 case-study code and frozen result tables
evaluation.py         Dataset-1 retrieval evaluation
inference.py          single-binary inference pipeline
api_server.py         FastAPI embedding service
ARTIFACT_EVALUATION.md detailed reproduction guide
```

---

## 📦 External Artifact Packages

The prepared datasets and trained weights are distributed as two separate
archives because they are too large for Git. Download both archives, their
manifests, and `SHA256SUMS` from the
[anonymous OSF artifact page](https://osf.io/e6a27/files/dropbox?view_only=cef76fc48c0246ca8db864e24258075e).

| Archive | Contents | Size | SHA-256 |
| --- | --- | ---: | --- |
| `rell_sec27_data_2026-08-24.tar.zst` | Prepared ReLL and Qwen evaluation/training data | 3.5 GB | `249f38e2f34a9a5b0ecb24ed6e5f1217d672a6bd2661a4206cfe962ef4a860dc` |
| `rell_sec27_weights_2026-08-24.tar.zst` | ReLL model and Qwen IR/assembly LoRA adapters | 1.3 GB | `7f87553a4ca55db9b16b90bd54a70c6fc9f49df2df929dff432933f4995ffdd5` |

Download both archives, their manifests, and `SHA256SUMS`. On Linux, verify
the downloads with:

```bash
sha256sum -c SHA256SUMS
```

On macOS, use:

```bash
shasum -a 256 -c SHA256SUMS
```

Then unpack both archives into the root of this repository:

```bash
cd /path/to/rell
tar -I zstd -xf /path/to/rell_sec27_data_2026-08-24.tar.zst -C .
tar -I zstd -xf /path/to/rell_sec27_weights_2026-08-24.tar.zst -C .
```

The downloads require about 4.8 GB. The extracted files require about 30 GB;
we recommend at least 40 GB of available disk space while downloading and
unpacking them.

The data archive provides the three Dataset-1 `Oc` final sets used by ReLL and
the prepared normalized-IR and assembly final sets used by the Qwen case study.
The weights archive provides:

```text
runs/dataset1_oc_fused/model_cfg_ddg
case_studies/qwen3_embedding/weights/qwen3_0p6b_ir_lora
case_studies/qwen3_embedding/weights/qwen3_0p6b_asm_lora
```

The ReLL quick reproduction below does not require a Qwen model. The two Qwen
adapters use `Qwen/Qwen3-Embedding-0.6B` as their base model; the base model is
downloaded separately from Hugging Face and is not duplicated in this release.

### Dataset flow

The Dataset-1 test metadata lists 264,548 functions.  The pipeline matches
261,962 of them to the available lifted inputs.  Function extraction and exact
input deduplication produce 212,402 encoded records.  We then remove 1,384
records with at most 128 tokens because they contain too little information for
meaningful retrieval, leaving the final pool used in the paper:

```text
pool examples: 211,018
anchors:       164,969
```

The 164,969 anchors are the pool entries that retain at least one valid
cross-setting positive match.  Evaluation samples candidate pools from the
211,018 entries and reports retrieval metrics over these anchors.  The
filtering summary and retained indices are included with the packaged test
split.

---

## ⚙️ Environment

Create or activate a Python environment with CUDA PyTorch, then install the
project dependencies:

```bash
python -m pip install -r requirements.txt
```

Notes:

- `pygraphviz` may require the system Graphviz development package.
- Open-source lifting uses Ghidra.  Set `GHIDRA_HOME` or
  `GHIDRA_ANALYZE_HEADLESS`, or pass `--ghidra-home` /
  `--analyze-headless` directly.
- IDA Pro is still supported as the historical backend.  The default local IDA
  path in scripts is `/path/to/ida-pro/idat`; adjust it for your
  machine only if you run the IDA backend.
- The quick reproduction path does **not** require re-lifting binaries.

---

## ⚡ Quick Reproduction

After unpacking both artifact packages:

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
requires substantial CPU/storage.  If you start from already lifted IR, use the
Ray fused pipeline below.  If you start from binaries and want an open-source
lifter, use the Ghidra backend in the next section.

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

For smaller binary-in workflows, use:

```bash
python Scripts/pipeline.py --help
```

That pipeline exposes the traditional stages: lifting, re-optimization,
function extraction, and optional recompilation.

---

## 🧩 Open-Source Lifting With Ghidra

Reviewers do not need IDA Pro to inspect or rerun the lifting stage.  This
release includes a Ghidra High P-Code lifter:

```text
Scripts/pcode2llvm.py
Scripts/ghidra/PcodeDump.java
```

`PcodeDump.java` is executed by Ghidra headless mode and exports each function's
High P-Code, CFG, types, and varnodes as JSONL.  `pcode2llvm.py` then emits LLVM
IR with real branches, phi nodes, integer/float operations, load/store, direct
calls, and indirect calls.  The default mode is strict: if any function cannot
be lifted, the command exits non-zero instead of silently emitting a partial
module.

Single-binary lift:

```bash
export GHIDRA_HOME=/path/to/ghidra

python Scripts/pcode2llvm.py \
  -f /path/to/binary \
  -o /tmp/binary.ll \
  --max-cpu 4 \
  --analysis-timeout 300 \
  --decompile-timeout 60
```

Equivalent Task 1 pipeline entry:

```bash
python Scripts/pipeline.py task1 \
  --backend ghidra \
  --input-path /path/to/binaries \
  --output /tmp/regraph_lifted \
  --workers 16 \
  --ghidra-max-cpu 1 \
  --resume
```

By default the generated LLVM module uses the host target triple, matching the
legacy `ida2llvm.py` behavior used by the preprocessing scripts.  To preserve
the architecture triple detected by Ghidra, pass:

```bash
--ghidra-target ghidra
```

Use `--allow-partial` only for debugging unsupported binaries; artifact
reproduction should keep the strict default.

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
external archives.  See:

📄 **[`Scripts/PIPELINE_LATENCY_BENCHMARK.md`](Scripts/PIPELINE_LATENCY_BENCHMARK.md)**

---

## 🧾 Artifact Packaging

The following maintainer utility builds an anonymized core ReLL bundle:

```bash
bash Scripts/pack_artifact_release.sh \
  --output-dir /path/to/artifact_output \
  --python /path/to/python
```

The packer rewrites author-specific paths in both metadata and Arrow datasets,
checks the staged files for identity markers, and records the archive's SHA-256
digest in its manifest. It is not needed for evaluation. The released data and
weights archives above additionally contain the prepared Qwen case-study data
and adapters.

---

## 📌 Release Notes

- This branch intentionally excludes datasets, model checkpoints, and run
  caches from Git.
- Open-source Ghidra/P-Code lifting is included to address reproducibility
  concerns around IDA as commercial software.
- `Oc2` and later exploratory settings are not part of the release workflow.
- `runs/dataset1_oc_fused/model_cfg_ddg` does not need a standalone
  `config.json`; `evaluation.py` falls back to the repository's
  `PretrainConfig` before loading `pytorch_model.bin`.

---

## Citation

If you use this artifact, cite the corresponding ReLL paper version.  The
release branch is designed to make the main claims auditable through code,
packaged final sets, model weights, and deterministic evaluation commands.
