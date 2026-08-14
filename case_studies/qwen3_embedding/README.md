# Qwen3-Embedding Case Study

This directory contains the code and frozen result tables for the paper's
model-scale case study. It asks how much a suitable input representation helps
an LLM-based encoder, rather than treating this model as part of ReLL.

## Compared Settings

| Setting | Representation | Fine-tuned |
| --- | --- | --- |
| Qwen3-Embedding-0.6B | raw assembly | no |
| Qwen3-Embedding-0.6B | raw assembly | yes |
| Qwen3-Embedding-4B | raw assembly | no |
| Qwen3-Embedding-0.6B | normalized LLVM IR | no |
| Qwen3-Embedding-0.6B | normalized LLVM IR | yes |

All settings use the same retrieval protocol. Each prepared final set contains
`train_dataset_pool` and `train_positive_map.pkl`. Large datasets, embedding
caches, and fine-tuned weights are distributed separately from this repository.

## Environment

Install the case-study dependencies from the repository root:

```bash
python -m pip install -r case_studies/qwen3_embedding/requirements.txt
```

The training scripts download `Qwen/Qwen3-Embedding-0.6B` from Hugging Face by
default. Evaluation uses a Text Embeddings Inference (TEI) endpoint. The
provided deployment files under `tei_deployment/` are an example; set
`TEI_ENDPOINT` and `TOKENIZER_NAME` to match the model being served.

## Normalized-IR Runs

Fine-tune the 0.6B model on normalized IR by pointing `OC_ROOT` to the prepared
train and validation final sets:

```bash
OC_ROOT=/path/to/oc_qwen_final_sets \
  bash case_studies/qwen3_embedding/scripts/run_oc_qwen_train.sh
```

After serving either the official model or the fine-tuned model with TEI,
evaluate it on the normalized-IR test set:

```bash
REGRAPH_ROOT=/path/to/rell \
OC_FINAL_SET=/path/to/oc_test_final_set \
TEI_ENDPOINT=http://127.0.0.1:8080 \
TOKENIZER_NAME=Qwen/Qwen3-Embedding-0.6B \
RUN_ID=qwen3_0p6b_ir \
  bash case_studies/qwen3_embedding/scripts/run_oc_qwen_eval.sh
```

Use a distinct `RUN_ID` and cache path for the official and fine-tuned models.

## Assembly Runs

The assembly pipeline builds the text dataset, fine-tunes the 0.6B model, and
evaluates the selected TEI-served checkpoint:

```bash
ASM_FINAL_SET=/path/to/asm_final_set \
OUTPUT_ROOT=/path/to/asm_qwen_workspace \
  bash case_studies/qwen3_embedding/scripts/run_asm_qwen_pipeline.sh
```

For evaluation only, set the prepared dataset and endpoint explicitly:

```bash
DATASET_FINAL_SET=/path/to/asm_test_final_set \
TEI_ENDPOINT=http://127.0.0.1:18081 \
TOKENIZER_NAME=Qwen/Qwen3-Embedding-0.6B \
RUN_ID=qwen3_0p6b_asm \
  bash case_studies/qwen3_embedding/scripts/run_asm_qwen_eval.sh
```

The same evaluator covers the off-the-shelf 4B model. Serve
`Qwen/Qwen3-Embedding-4B`, then set `TOKENIZER_NAME`, `MODEL_ID`, `SUMMARY_NAME`,
`RUN_ID`, and `CACHE_PATH` to 4B-specific values before running the command.

## Frozen Results and Figures

The `results/` directory contains the five Markdown result tables used by the
paper. Regenerate the Recall@1 and MRR curves without model inference:

```bash
python case_studies/qwen3_embedding/scripts/plot_case_study.py
```

By default, the two PDFs are written to
`case_studies/qwen3_embedding/figures/`. Use `--results-dir` and `--output-dir`
to select other locations.
