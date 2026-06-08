# Pipeline Latency Benchmark

This benchmark measures lifting and reoptimization latency for a sampled set of
binary functions. It is intended for artifact evaluation and for reproducing the
preprocessing-latency part of the pipeline table.

The lifting stage reports two timing views:

- `Wall s/100`: end-to-end subprocess wall time, including Python process
  startup, IDA database open, and IDA auto-analysis.
- `Steady-state s/100`: lifting latency after excluding IDA database open and
  auto-analysis. This is the fair per-function pipeline cost when a binary has
  already been loaded by IDA.

For the paper latency table, prefer sampling many functions from one binary, or
from a small number of binaries, and report the steady-state lifting latency.
Randomly sampling one function from many binaries mostly measures IDA startup
and database loading overhead.

## Single-Binary Benchmark

Use this mode for a stable 100-function latency measurement:

```bash
python Scripts/benchmark_pipeline_latency.py \
  --dataset-path Binaries/Dataset-1/validation \
  --functions 100 \
  --functions-per-binary 100 \
  --lift-chunk-size 4 \
  --workers 1 32 \
  --opt-level Oc \
  --work-dir runs/pipeline_latency_validation_100 \
  --rebuild-manifest
```

`--lift-chunk-size` is important for parallel single-binary benchmarking. With
the default chunk size `0`, one binary becomes one lift task, so `--workers 32`
cannot speed up a single-binary sample. A chunk size of `4` turns 100 functions
into roughly 25 lift tasks while still sampling from one binary.

When chunking is enabled, the script copies the `.i64` database once per lift
task before timing. This avoids concurrent IDA workers opening the same database
file. The copy time is recorded in `summary.json` but excluded from the latency
table.

If the script cannot infer the bitness from the file path, add `--arch m32` or
`--arch m64`.

To force a specific binary:

```bash
python Scripts/benchmark_pipeline_latency.py \
  --dataset-path Binaries/Dataset-1/validation \
  --binary path/relative/to/dataset/sample.i64 \
  --functions 100 \
  --functions-per-binary 100 \
  --lift-chunk-size 4 \
  --workers 1 32 \
  --opt-level Oc \
  --work-dir runs/pipeline_latency_specific_binary \
  --rebuild-manifest
```

## Conda / IDA Python

The main process and the IDA worker can use the same active Python by default.
If needed, override the worker command explicitly:

```bash
REGRAPH_IDA_PYTHON_CMD="python" \
python Scripts/benchmark_pipeline_latency.py \
  --dataset-path Binaries/Dataset-1/validation \
  --functions 100 \
  --functions-per-binary 100 \
  --lift-chunk-size 4 \
  --workers 1 32 \
  --opt-level Oc \
  --work-dir runs/pipeline_latency_validation_100 \
  --rebuild-manifest
```

## Outputs

The script writes:

- `summary.json`: full machine-readable timings and failure examples.
- `pipeline_latency.md`: compact Markdown table for reporting.
- `manifest.json`: sampled binaries and function addresses for reproducibility.
