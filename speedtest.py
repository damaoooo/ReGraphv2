from __future__ import annotations

import gc
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import typer
from datasets import load_from_disk

from Model.graph_utils import build_batched_span_graph_tensors
from Pretrain.pretrain_config import PretrainConfig
from Pretrain.pretrain_model import MoCoPretrainModel
from Tokenizer.ir_tokenizer import load_tokenizer
from Utils.utils import DEFAULT_TOKENIZER_PATH


STATE_DICT_BIN = "pytorch_model.bin"
STATE_DICT_SAFETENSORS = "model.safetensors"
DEFAULT_DATASET2_SEQ_LENGTH = 598
DEFAULT_DATASET2_LENGTH_STATS = {
    "num_rows": 1306388,
    "mean": 597.7015457888468,
    "median": 309.0,
    "p75": 882.0,
    "p90": 2017.0,
    "p95": 2048.0,
    "min": 31,
    "max": 2048,
    "resolved_dataset_path": (
        "/path/to/rell/IR/Dataset-2/db2_final_set/train_dataset_pool"
    ),
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter("CUDA was requested, but no GPU is available.")
    return resolved


def _load_state_dict(model_path: Path) -> tuple[Dict[str, torch.Tensor], Path]:
    bin_path = model_path / STATE_DICT_BIN
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu"), bin_path

    safetensors_path = model_path / STATE_DICT_SAFETENSORS
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise RuntimeError(
                "Found model.safetensors but safetensors is not installed."
            ) from exc
        return load_safetensors_file(safetensors_path, device="cpu"), safetensors_path

    raise FileNotFoundError(
        f"Checkpoint file not found under {model_path}: "
        f"{STATE_DICT_BIN} or {STATE_DICT_SAFETENSORS}"
    )


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> PretrainConfig:
    vocab_size, hidden_size = state_dict[
        "encoder_q.roformer.roformer.embeddings.word_embeddings.weight"
    ].shape
    max_seq_length, head_dim = state_dict[
        "encoder_q.roformer.roformer.encoder.embed_positions.weight"
    ].shape
    intermediate_size = state_dict[
        "encoder_q.roformer.roformer.encoder.layer.0.intermediate.dense.weight"
    ].shape[0]
    embedding_size = state_dict["encoder_q.linear.2.weight"].shape[0]

    layer_pattern = re.compile(r"encoder_q\.roformer\.roformer\.encoder\.layer\.(\d+)\.")
    cfg_conv_pattern = re.compile(r"encoder_q\.roformer\.cfg_branch\.convs\.(\d+)\.")
    ddg_conv_pattern = re.compile(r"encoder_q\.roformer\.ddg_branch\.convs\.(\d+)\.")

    num_hidden_layers = len(
        {
            int(match.group(1))
            for key in state_dict
            for match in [layer_pattern.match(key)]
            if match is not None
        }
    )
    cfg_graph_layers = len(
        {
            int(match.group(1))
            for key in state_dict
            for match in [cfg_conv_pattern.match(key)]
            if match is not None
        }
    )
    ddg_graph_layers = len(
        {
            int(match.group(1))
            for key in state_dict
            for match in [ddg_conv_pattern.match(key)]
            if match is not None
        }
    )

    use_cfg = any(key.startswith("encoder_q.roformer.cfg_branch.") for key in state_dict)
    use_ddg = any(key.startswith("encoder_q.roformer.ddg_branch.") for key in state_dict)
    graph_layers = max(cfg_graph_layers, ddg_graph_layers, 0)

    if use_ddg:
        graph_attention_heads = int(state_dict["encoder_q.roformer.ddg_branch.convs.0.att"].shape[1])
    elif use_cfg:
        graph_attention_heads = int(state_dict["encoder_q.roformer.cfg_branch.convs.0.att"].shape[1])
    else:
        graph_attention_heads = 1

    num_attention_heads = hidden_size // head_dim

    return PretrainConfig(
        max_seq_length=max_seq_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        embedding_size=embedding_size,
        use_cfg=use_cfg,
        use_ddg=use_ddg,
        graph_layers=graph_layers,
        graph_attention_heads=graph_attention_heads,
    )


def _resolve_tokenizer_path(model_path: Path, override: Optional[str]) -> Optional[str]:
    if override:
        return override

    model_tokenizer = model_path / "tokenizer.json"
    if model_tokenizer.exists():
        return str(model_tokenizer)

    if os.path.exists(DEFAULT_TOKENIZER_PATH):
        return DEFAULT_TOKENIZER_PATH

    return None


def _resolve_dataset_pool_path(dataset_path: Path) -> Path:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    train_pool = dataset_path / "train_dataset_pool"
    if train_pool.exists():
        return train_pool

    dataset_info = dataset_path / "dataset_info.json"
    state_file = dataset_path / "state.json"
    if dataset_info.exists() and state_file.exists():
        return dataset_path

    raise FileNotFoundError(
        f"Could not find a HuggingFace dataset under {dataset_path}. "
        "Pass either a dataset directory or a root folder that contains train_dataset_pool."
    )


def _compute_input_length_stats(dataset_path: Path) -> Dict[str, float]:
    resolved_dataset_path = _resolve_dataset_pool_path(dataset_path)
    dataset = load_from_disk(str(resolved_dataset_path))

    if "input_ids" not in dataset.features:
        raise ValueError(
            f"Dataset {resolved_dataset_path} does not contain an 'input_ids' column."
        )

    arrow_column = dataset.data.column("input_ids")
    length_chunks = [
        np.asarray(chunk.value_lengths(), dtype=np.int32)
        for chunk in arrow_column.chunks
    ]
    lengths = np.concatenate(length_chunks) if length_chunks else np.empty((0,), dtype=np.int32)
    if lengths.size == 0:
        raise ValueError(f"Dataset {resolved_dataset_path} is empty.")

    return {
        "num_rows": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "resolved_dataset_path": str(resolved_dataset_path),
    }


def _load_encoder_q(
    model_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, PretrainConfig, Path]:
    state_dict, checkpoint_path = _load_state_dict(model_path)
    config = _infer_config_from_state_dict(state_dict)

    moco_model = MoCoPretrainModel(config)
    missing_keys, unexpected_keys = moco_model.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "State dict mismatch while loading checkpoint. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    encoder_q = moco_model.encoder_q
    encoder_q.to(device)
    encoder_q.eval()

    del moco_model
    del state_dict
    gc.collect()

    return encoder_q, config, checkpoint_path


def _build_cfg_graph(seq_length: int, edge_count: int) -> list[list[float]]:
    if seq_length <= 1 or edge_count <= 0:
        return []

    block_count = min(seq_length, edge_count + 1)
    boundaries = np.linspace(0, seq_length, num=block_count + 1, dtype=int)
    edges: list[list[float]] = []
    for idx in range(block_count - 1):
        src_start = int(boundaries[idx])
        src_end = max(src_start, int(boundaries[idx + 1]) - 1)
        dst_start = int(boundaries[idx + 1])
        dst_end = max(dst_start, int(boundaries[idx + 2]) - 1)
        edges.append([src_start, src_end, dst_start, dst_end, 1.0 + (idx % 3) * 0.1])
    return edges


def _build_ddg_graph(seq_length: int, edge_count: int, sample_offset: int) -> list[list[int]]:
    if seq_length <= 1 or edge_count <= 0:
        return []

    window = max(1, seq_length - 1)
    max_jump = max(1, min(8, seq_length - 1))
    edges: list[list[int]] = []
    for idx in range(edge_count):
        src = (idx + sample_offset) % window
        jump = 1 + (idx // window) % max_jump
        dst = min(src + jump, seq_length - 1)
        edges.append([src, src, dst, dst])
    return edges


def _build_synthetic_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    use_cfg: bool,
    use_ddg: bool,
    cfg_edges_per_sample: int,
    ddg_edges_per_sample: int,
    bos_token_id: Optional[int],
    eos_token_id: Optional[int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]], Dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    low = 1 if vocab_size > 1 else 0
    input_ids = torch.randint(
        low=low,
        high=vocab_size,
        size=(batch_size, seq_length),
        generator=generator,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

    if bos_token_id is not None:
        input_ids[:, 0] = bos_token_id
    if eos_token_id is not None and seq_length > 1:
        input_ids[:, -1] = eos_token_id

    seq_lengths = [seq_length] * batch_size

    graph_inputs: Dict[str, Optional[torch.Tensor]] = {
        "ddg_node_spans": None,
        "ddg_node_batch": None,
        "ddg_edge_index": None,
        "cfg_node_spans": None,
        "cfg_node_batch": None,
        "cfg_edge_index": None,
        "cfg_edge_attr": None,
    }
    stats: Dict[str, Any] = {
        "cfg_enabled": use_cfg,
        "ddg_enabled": use_ddg,
        "cfg_edges_per_sample_requested": cfg_edges_per_sample,
        "ddg_edges_per_sample_requested": ddg_edges_per_sample,
    }

    if use_ddg:
        ddg_graphs = [
            _build_ddg_graph(seq_length=seq_length, edge_count=ddg_edges_per_sample, sample_offset=sample_idx)
            for sample_idx in range(batch_size)
        ]
        ddg_tensors = build_batched_span_graph_tensors(
            ddg_graphs,
            seq_lengths,
            include_edge_attr=False,
        )
        graph_inputs["ddg_node_spans"] = ddg_tensors["node_spans"]
        graph_inputs["ddg_node_batch"] = ddg_tensors["node_batch"]
        graph_inputs["ddg_edge_index"] = ddg_tensors["edge_index"]
        stats["ddg_nodes_total"] = int(ddg_tensors["node_spans"].shape[0])
        stats["ddg_edges_total"] = int(ddg_tensors["edge_index"].shape[1])
        stats["ddg_nodes_per_sample"] = stats["ddg_nodes_total"] / batch_size
        stats["ddg_edges_per_sample_actual"] = stats["ddg_edges_total"] / batch_size
    else:
        stats["ddg_nodes_total"] = 0
        stats["ddg_edges_total"] = 0
        stats["ddg_nodes_per_sample"] = 0.0
        stats["ddg_edges_per_sample_actual"] = 0.0

    if use_cfg:
        cfg_graph = _build_cfg_graph(seq_length=seq_length, edge_count=cfg_edges_per_sample)
        cfg_graphs = [cfg_graph for _ in range(batch_size)]
        cfg_tensors = build_batched_span_graph_tensors(
            cfg_graphs,
            seq_lengths,
            include_edge_attr=True,
        )
        graph_inputs["cfg_node_spans"] = cfg_tensors["node_spans"]
        graph_inputs["cfg_node_batch"] = cfg_tensors["node_batch"]
        graph_inputs["cfg_edge_index"] = cfg_tensors["edge_index"]
        graph_inputs["cfg_edge_attr"] = cfg_tensors["edge_attr"]
        stats["cfg_nodes_total"] = int(cfg_tensors["node_spans"].shape[0])
        stats["cfg_edges_total"] = int(cfg_tensors["edge_index"].shape[1])
        stats["cfg_nodes_per_sample"] = stats["cfg_nodes_total"] / batch_size
        stats["cfg_edges_per_sample_actual"] = stats["cfg_edges_total"] / batch_size
    else:
        stats["cfg_nodes_total"] = 0
        stats["cfg_edges_total"] = 0
        stats["cfg_nodes_per_sample"] = 0.0
        stats["cfg_edges_per_sample_actual"] = 0.0

    return input_ids, attention_mask, graph_inputs, stats


def _move_graph_inputs_to_device(
    graph_inputs: Dict[str, Optional[torch.Tensor]],
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    return {
        key: value.to(device) if value is not None else None
        for key, value in graph_inputs.items()
    }


def _benchmark_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    graph_inputs: Dict[str, Optional[torch.Tensor]],
    steps: int,
    warmup_steps: int,
    use_bf16: bool,
) -> tuple[list[float], Dict[str, Any]]:
    device = input_ids.device
    latencies_s: list[float] = []
    last_embedding_shape = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
    else:
        memory_before = None

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode():
        with autocast_context:
            for _ in range(warmup_steps):
                outputs = model.encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    **graph_inputs,
                )
                last_embedding_shape = tuple(outputs["embedding"].shape)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(steps):
                    start_event.record()
                    outputs = model.encode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        **graph_inputs,
                    )
                    end_event.record()
                    torch.cuda.synchronize(device)
                    latencies_s.append(start_event.elapsed_time(end_event) / 1000.0)
                    last_embedding_shape = tuple(outputs["embedding"].shape)
            else:
                for _ in range(steps):
                    start_time = time.perf_counter()
                    outputs = model.encode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        **graph_inputs,
                    )
                    latencies_s.append(time.perf_counter() - start_time)
                    last_embedding_shape = tuple(outputs["embedding"].shape)

    extra_metrics = {
        "embedding_shape": last_embedding_shape,
        "cuda_memory_allocated_before_bytes": memory_before,
    }
    if device.type == "cuda":
        extra_metrics["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        extra_metrics["cuda_memory_allocated_after_bytes"] = torch.cuda.memory_allocated(device)

    return latencies_s, extra_metrics


def _summarize_metrics(
    latencies_s: list[float],
    batch_size: int,
    seq_length: int,
) -> Dict[str, float]:
    latency_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    mean_latency_s = float(np.mean(latencies_s))
    return {
        "mean_latency_ms": float(latency_ms.mean()),
        "median_latency_ms": float(np.median(latency_ms)),
        "p95_latency_ms": float(np.percentile(latency_ms, 95)),
        "min_latency_ms": float(latency_ms.min()),
        "max_latency_ms": float(latency_ms.max()),
        "std_latency_ms": float(latency_ms.std()),
        "samples_per_second": float(batch_size / mean_latency_s),
        "tokens_per_second": float((batch_size * seq_length) / mean_latency_s),
    }


def main(
    model_path: str = typer.Option(
        "/path/to/rell/db1_model_cfg_ddg",
        help="Checkpoint directory used for inference benchmarking.",
    ),
    tokenizer_path: Optional[str] = typer.Option(
        None,
        help="Optional tokenizer.json override. Defaults to model_path/tokenizer.json if present.",
    ),
    device: str = typer.Option(
        "auto",
        help="Benchmark device: auto / cpu / cuda / cuda:0 ...",
    ),
    batch_size: int = typer.Option(8, min=1, help="Synthetic batch size for the forward pass."),
    dataset_path: Optional[str] = typer.Option(
        None,
        help=(
            "Optional dataset path used to derive a realistic average sequence length. "
            "You can pass either a dataset directory or a root folder containing train_dataset_pool."
        ),
    ),
    seq_length: Optional[int] = typer.Option(
        None,
        help=(
            "Synthetic sequence length. If omitted, defaults to the rounded Dataset-2 "
            "mean input length (598). If --dataset-path is set, that dataset mean is used instead."
        ),
    ),
    warmup_steps: int = typer.Option(10, min=0, help="Warmup iterations excluded from timing."),
    steps: int = typer.Option(50, min=1, help="Measured forward iterations."),
    cfg_edges_per_sample: int = typer.Option(
        16,
        min=0,
        help="Synthetic CFG edge count per sample when the checkpoint has a CFG branch.",
    ),
    ddg_edges_per_sample: int = typer.Option(
        64,
        min=0,
        help="Synthetic DDG edge count per sample when the checkpoint has a DDG branch.",
    ),
    seed: int = typer.Option(42, help="Random seed for synthetic inputs."),
    output_json: Optional[str] = typer.Option(
        None,
        help="Optional path to dump metrics as JSON.",
    ),
):
    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.exists():
        raise typer.BadParameter(f"Model path does not exist: {model_dir}")

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    load_start = time.perf_counter()
    model, config, checkpoint_file = _load_encoder_q(model_dir, resolved_device)
    model_load_seconds = time.perf_counter() - load_start

    dataset_length_stats = None
    if dataset_path is not None:
        dataset_length_stats = _compute_input_length_stats(Path(dataset_path).expanduser().resolve())

    resolved_seq_length = seq_length
    if resolved_seq_length is None:
        if dataset_length_stats is not None:
            resolved_seq_length = int(round(dataset_length_stats["mean"]))
        else:
            resolved_seq_length = DEFAULT_DATASET2_SEQ_LENGTH
            dataset_length_stats = dict(DEFAULT_DATASET2_LENGTH_STATS)

    if resolved_seq_length > int(config.max_position_embeddings):
        raise typer.BadParameter(
            f"Requested seq_length={resolved_seq_length}, but checkpoint supports at most "
            f"{config.max_position_embeddings} tokens."
        )
    if resolved_seq_length < 1:
        raise typer.BadParameter(f"Resolved seq_length must be positive, got {resolved_seq_length}.")

    resolved_tokenizer_path = _resolve_tokenizer_path(model_dir, tokenizer_path)
    tokenizer = load_tokenizer(resolved_tokenizer_path) if resolved_tokenizer_path else None

    bos_token_id = tokenizer.bos_token_id if tokenizer is not None else None
    eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None

    input_ids_cpu, attention_mask_cpu, graph_inputs_cpu, graph_stats = _build_synthetic_batch(
        batch_size=batch_size,
        seq_length=resolved_seq_length,
        vocab_size=int(config.vocab_size),
        use_cfg=bool(config.use_cfg),
        use_ddg=bool(config.use_ddg),
        cfg_edges_per_sample=cfg_edges_per_sample,
        ddg_edges_per_sample=ddg_edges_per_sample,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        seed=seed,
    )

    input_ids = input_ids_cpu.to(resolved_device)
    attention_mask = attention_mask_cpu.to(resolved_device)
    graph_inputs = _move_graph_inputs_to_device(graph_inputs_cpu, resolved_device)

    use_bf16 = resolved_device.type == "cuda"
    latencies_s, runtime_details = _benchmark_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        graph_inputs=graph_inputs,
        steps=steps,
        warmup_steps=warmup_steps,
        use_bf16=use_bf16,
    )
    summary = _summarize_metrics(latencies_s=latencies_s, batch_size=batch_size, seq_length=resolved_seq_length)

    result = {
        "model_path": str(model_dir),
        "checkpoint_file": str(checkpoint_file),
        "tokenizer_path": resolved_tokenizer_path,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()) if dataset_path else None,
        "dataset_length_stats": dataset_length_stats,
        "default_seq_length_source": (
            "Dataset-2 train_dataset_pool mean input_ids length"
            if dataset_path is None and seq_length is None
            else None
        ),
        "device": str(resolved_device),
        "use_bf16_autocast": use_bf16,
        "model_load_seconds": model_load_seconds,
        "batch_size": batch_size,
        "seq_length": resolved_seq_length,
        "warmup_steps": warmup_steps,
        "steps": steps,
        "seed": seed,
        "inferred_config": {
            "vocab_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "intermediate_size": int(config.intermediate_size),
            "max_position_embeddings": int(config.max_position_embeddings),
            "embedding_size": int(config.embedding_size),
            "use_cfg": bool(config.use_cfg),
            "use_ddg": bool(config.use_ddg),
            "graph_layers": int(config.graph_layers),
            "graph_attention_heads": int(config.graph_attention_heads),
        },
        "graph_stats": graph_stats,
        "runtime_details": runtime_details,
        "summary": summary,
        "note": (
            "This benchmark measures the current encoder_q forward path only. "
            "Model loading and synthetic input construction are excluded from the timed loop."
        ),
    }

    print("")
    print("=== ReLL Inference Speed Test ===")
    print(f"checkpoint: {checkpoint_file}")
    print(f"device: {resolved_device} | bf16_autocast: {use_bf16}")
    print(
        f"model: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}, cfg={config.use_cfg}, ddg={config.use_ddg}"
    )
    print(
        f"input: batch_size={batch_size}, seq_length={resolved_seq_length}, "
        f"warmup={warmup_steps}, measured_steps={steps}"
    )
    if dataset_length_stats is not None:
        print(
            "dataset seq stats: "
            f"mean={dataset_length_stats['mean']:.2f}, median={dataset_length_stats['median']:.2f}, "
            f"p90={dataset_length_stats['p90']:.2f}, p95={dataset_length_stats['p95']:.2f}"
        )
    print(
        f"graph: cfg_edges/sample={graph_stats['cfg_edges_per_sample_actual']:.1f}, "
        f"ddg_edges/sample={graph_stats['ddg_edges_per_sample_actual']:.1f}"
    )
    print(
        f"graph: cfg_nodes/sample={graph_stats['cfg_nodes_per_sample']:.1f}, "
        f"ddg_nodes/sample={graph_stats['ddg_nodes_per_sample']:.1f}"
    )
    print(f"load_time: {model_load_seconds:.3f} s")
    print("")
    print(f"avg latency: {summary['mean_latency_ms']:.3f} ms/batch")
    print(f"median latency: {summary['median_latency_ms']:.3f} ms/batch")
    print(f"p95 latency: {summary['p95_latency_ms']:.3f} ms/batch")
    print(f"min/max latency: {summary['min_latency_ms']:.3f} / {summary['max_latency_ms']:.3f} ms")
    print(f"std latency: {summary['std_latency_ms']:.3f} ms")
    print(f"throughput: {summary['samples_per_second']:.3f} samples/s")
    print(f"throughput: {summary['tokens_per_second']:.3f} tokens/s")

    if resolved_device.type == "cuda":
        before_mb = runtime_details["cuda_memory_allocated_before_bytes"] / (1024 ** 2)
        after_mb = runtime_details["cuda_memory_allocated_after_bytes"] / (1024 ** 2)
        peak_mb = runtime_details["cuda_peak_memory_allocated_bytes"] / (1024 ** 2)
        print(f"cuda memory: before={before_mb:.1f} MB, after={after_mb:.1f} MB, peak={peak_mb:.1f} MB")

    print(f"embedding shape: {runtime_details['embedding_shape']}")
    print("")
    print(result["note"])

    if output_json:
        output_path = Path(output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)
        print(f"json saved to: {output_path}")


if __name__ == "__main__":
    typer.run(main)
