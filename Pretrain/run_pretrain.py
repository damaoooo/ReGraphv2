import traceback
import os
import copy
import ast
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from .pretrain_model import MoCoPretrainModel, ReFormerPretrainModel
import pickle
import torch
import typer
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.trainer import unwrap_model
from Model.graph_utils import build_batched_span_graph_tensors
from Pretrain.pretrain_dataset import MoCoDataCollator, load_dataset, compute_group_ids
from Tokenizer.ir_tokenizer import load_tokenizer
from .pretrain_config import PretrainConfig, DEFAULT_CONFIG
from typing import Dict, List, Any, Callable, Optional, Tuple

try:
    import bitsandbytes as bnb  # type: ignore
except Exception:
    bnb = None

app = typer.Typer(add_completion=False, no_args_is_help=False)

LEGACY_GRAPH_MODE_TO_FLAGS: Dict[str, Tuple[bool, bool]] = {
    "both": (True, True),
    "no_cfg": (False, True),
    "no_ddg": (True, False),
    "none": (False, False),
}

REMOVED_GRAPH_MODES = {"cfg_as_ddg", "cfg_as_ddg_no_ddg", "both_cfg_as_ddg"}


def _split_override_item(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise typer.BadParameter(f"Invalid --set '{item}', expected key=value")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise typer.BadParameter(f"Invalid --set '{item}', key cannot be empty")
    return key, raw_value


def _coerce_override_value(raw_value: str, current_value: Any) -> Any:
    lowered = raw_value.lower()

    if lowered in {"none", "null"}:
        return None

    if isinstance(current_value, bool):
        bool_map = {
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "on": True,
            "off": False,
        }
        if lowered not in bool_map:
            raise typer.BadParameter(
                f"Cannot parse '{raw_value}' as bool. Use true/false/1/0/yes/no/on/off"
            )
        return bool_map[lowered]

    if isinstance(current_value, int) and not isinstance(current_value, bool):
        try:
            return int(raw_value)
        except ValueError as exc:
            raise typer.BadParameter(f"Cannot parse '{raw_value}' as int") from exc

    if isinstance(current_value, float):
        try:
            return float(raw_value)
        except ValueError as exc:
            raise typer.BadParameter(f"Cannot parse '{raw_value}' as float") from exc

    if isinstance(current_value, (list, tuple, dict, set)):
        try:
            return ast.literal_eval(raw_value)
        except (ValueError, SyntaxError) as exc:
            raise typer.BadParameter(
                f"Cannot parse '{raw_value}' as {type(current_value).__name__}; "
                "please use Python literal syntax"
            ) from exc

    if isinstance(current_value, str):
        return raw_value

    # Fallback for unknown attribute types.
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        return raw_value


def _apply_cli_overrides(config: PretrainConfig, overrides: Optional[List[str]]) -> PretrainConfig:
    if not overrides:
        return config

    for item in overrides:
        key, raw_value = _split_override_item(item)

        if not hasattr(config, key):
            raise typer.BadParameter(f"Unknown config key: '{key}'")

        current_value = getattr(config, key)
        new_value = _coerce_override_value(raw_value, current_value)
        setattr(config, key, new_value)
        print(f"[config override] {key}: {current_value!r} -> {new_value!r}")

    return config


def _build_config(overrides: Optional[List[str]] = None) -> PretrainConfig:
    config = copy.deepcopy(DEFAULT_CONFIG)
    return _apply_cli_overrides(config, overrides)


@dataclass(frozen=True)
class DatasetBundlePaths:
    role: str
    dataset_dir: str
    pool_path: str
    idx_path: str
    map_path: str
    prefix: str
    explicit: bool = False


def _has_path_value(value: Optional[str]) -> bool:
    return value is not None and str(value).strip() != ""


def _normalize_path(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(value))))


def _same_path(left: str, right: str) -> bool:
    return _normalize_path(left) == _normalize_path(right)


def _infer_dataset_dir_from_paths(*paths: Optional[str]) -> Optional[str]:
    for path in paths:
        if _has_path_value(path):
            return str(Path(_normalize_path(path)).parent)
    return None


def _bundle_from_dir(
    dataset_dir: str,
    prefix: str,
    role: str,
    explicit: bool = False,
    pool_path: Optional[str] = None,
    idx_path: Optional[str] = None,
    map_path: Optional[str] = None,
) -> DatasetBundlePaths:
    base_dir = _normalize_path(dataset_dir)
    return DatasetBundlePaths(
        role=role,
        dataset_dir=base_dir,
        pool_path=_normalize_path(pool_path) if _has_path_value(pool_path) else os.path.join(base_dir, f"{prefix}_dataset_pool"),
        idx_path=_normalize_path(idx_path) if _has_path_value(idx_path) else os.path.join(base_dir, f"{prefix}_task_dataset"),
        map_path=_normalize_path(map_path) if _has_path_value(map_path) else os.path.join(base_dir, f"{prefix}_positive_map.pkl"),
        prefix=prefix,
        explicit=explicit,
    )


def _missing_bundle_paths(bundle: DatasetBundlePaths) -> List[str]:
    missing = []
    if not os.path.isdir(bundle.pool_path):
        missing.append(f"pool={bundle.pool_path}")
    if not os.path.isdir(bundle.idx_path):
        missing.append(f"task={bundle.idx_path}")
    if not os.path.isfile(bundle.map_path):
        missing.append(f"map={bundle.map_path}")
    return missing


def _resolve_train_bundle(config: PretrainConfig) -> DatasetBundlePaths:
    dataset_dir = (
        config.train_dataset_dir
        or _infer_dataset_dir_from_paths(
            config.train_dataset_pool_path,
            config.train_dataset_idx_path,
            config.train_dataset_map_path,
        )
    )
    if not _has_path_value(dataset_dir):
        raise typer.BadParameter(
            "Training dataset is not configured. Set train_dataset_dir, or set all "
            "train_dataset_pool_path/train_dataset_idx_path/train_dataset_map_path."
        )

    bundle = _bundle_from_dir(
        dataset_dir=dataset_dir,
        prefix="train",
        role="train",
        explicit=any(
            _has_path_value(path)
            for path in (
                config.train_dataset_pool_path,
                config.train_dataset_idx_path,
                config.train_dataset_map_path,
            )
        ),
        pool_path=config.train_dataset_pool_path,
        idx_path=config.train_dataset_idx_path,
        map_path=config.train_dataset_map_path,
    )
    missing = _missing_bundle_paths(bundle)
    if missing:
        raise FileNotFoundError(
            "Training dataset bundle is incomplete. Missing: " + ", ".join(missing)
        )
    return bundle


def _validation_prefixes_for_dir(dataset_dir: str, train_dataset_dir: str) -> List[str]:
    if _same_path(dataset_dir, train_dataset_dir):
        return ["validation"]
    return ["train", "validation"]


def _resolve_validation_candidates(
    config: PretrainConfig,
    train_bundle: DatasetBundlePaths,
) -> List[DatasetBundlePaths]:
    explicit_paths = any(
        _has_path_value(path)
        for path in (
            config.validation_dataset_pool_path,
            config.validation_dataset_idx_path,
            config.validation_dataset_map_path,
        )
    )

    if explicit_paths:
        dataset_dir = (
            config.validation_dataset_dir
            or _infer_dataset_dir_from_paths(
                config.validation_dataset_pool_path,
                config.validation_dataset_idx_path,
                config.validation_dataset_map_path,
            )
        )
        if not _has_path_value(dataset_dir):
            raise typer.BadParameter(
                "Validation dataset paths are partially configured, but no base directory can be inferred."
            )
        return [
            _bundle_from_dir(
                dataset_dir=dataset_dir,
                prefix="validation",
                role="validation",
                explicit=True,
                pool_path=config.validation_dataset_pool_path,
                idx_path=config.validation_dataset_idx_path,
                map_path=config.validation_dataset_map_path,
            )
        ]

    candidate_dirs: List[Tuple[str, bool]] = []
    if _has_path_value(config.validation_dataset_dir):
        candidate_dirs.append((_normalize_path(config.validation_dataset_dir), True))
    else:
        train_dir = Path(train_bundle.dataset_dir)
        sibling_validation_dir = train_dir.parent / "validation_final_set"
        if train_dir.name == "train_final_set" and sibling_validation_dir.exists():
            candidate_dirs.append((str(sibling_validation_dir), False))
        candidate_dirs.append((train_bundle.dataset_dir, False))

    candidates: List[DatasetBundlePaths] = []
    seen = set()
    for dataset_dir, explicit in candidate_dirs:
        for prefix in _validation_prefixes_for_dir(dataset_dir, train_bundle.dataset_dir):
            key = (_normalize_path(dataset_dir), prefix)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                _bundle_from_dir(
                    dataset_dir=dataset_dir,
                    prefix=prefix,
                    role="validation",
                    explicit=explicit,
                )
            )
    return candidates


def _load_dataset_bundle(
    bundle: DatasetBundlePaths,
    allow_empty: bool = False,
) -> Tuple[Any, Any, Dict[int, List[int]]]:
    dataset_pool = load_dataset(bundle.pool_path)
    dataset_idx = load_dataset(bundle.idx_path)
    with open(bundle.map_path, "rb") as f:
        positive_map = pickle.load(f)

    if not allow_empty:
        if len(dataset_idx) == 0:
            raise ValueError(f"{bundle.role} task dataset is empty: {bundle.idx_path}")
        if len(positive_map) == 0:
            raise ValueError(f"{bundle.role} positive map is empty: {bundle.map_path}")

    return dataset_pool, dataset_idx, positive_map


def _apply_bundle_to_config(config: PretrainConfig, bundle: DatasetBundlePaths) -> None:
    if bundle.role == "train":
        config.train_dataset_dir = bundle.dataset_dir
        config.train_dataset_pool_path = bundle.pool_path
        config.train_dataset_idx_path = bundle.idx_path
        config.train_dataset_map_path = bundle.map_path
    elif bundle.role == "validation":
        config.validation_dataset_dir = bundle.dataset_dir
        config.validation_dataset_pool_path = bundle.pool_path
        config.validation_dataset_idx_path = bundle.idx_path
        config.validation_dataset_map_path = bundle.map_path


def _load_validation_bundle(
    config: PretrainConfig,
    train_bundle: DatasetBundlePaths,
) -> Tuple[Optional[DatasetBundlePaths], Optional[Any], Optional[Any], Optional[Dict[int, List[int]]]]:
    candidates = _resolve_validation_candidates(config, train_bundle)
    explicit_validation = any(candidate.explicit for candidate in candidates)
    last_error: Optional[Exception] = None

    for candidate in candidates:
        missing = _missing_bundle_paths(candidate)
        if missing:
            last_error = FileNotFoundError(", ".join(missing))
            if candidate.explicit:
                continue
            continue

        try:
            dataset_pool, dataset_idx, positive_map = _load_dataset_bundle(candidate)
        except ValueError as exc:
            last_error = exc
            if candidate.explicit:
                raise
            print(f"Skipping empty validation candidate: {candidate.dataset_dir} ({candidate.prefix})")
            continue

        return candidate, dataset_pool, dataset_idx, positive_map

    if explicit_validation and last_error is not None:
        raise FileNotFoundError(f"Validation dataset bundle could not be resolved: {last_error}")

    print("No non-empty validation dataset bundle found; training will run without validation.")
    return None, None, None, None


def _add_eval_strategy_arg(train_args_kwargs: Dict[str, Any], eval_strategy: str) -> None:
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_params:
        train_args_kwargs["eval_strategy"] = eval_strategy
    else:
        train_args_kwargs["evaluation_strategy"] = eval_strategy


def _resolve_graph_flags_from_legacy_mode(mode: str) -> Tuple[bool, bool]:
    lowered = mode.strip().lower()
    if lowered in LEGACY_GRAPH_MODE_TO_FLAGS:
        return LEGACY_GRAPH_MODE_TO_FLAGS[lowered]
    if lowered in REMOVED_GRAPH_MODES:
        raise typer.BadParameter(
            f"graph_mode={mode!r} is no longer supported. "
            "Use --cfg/--no-cfg and --ddg/--no-ddg explicitly."
        )
    supported = "/".join(sorted(LEGACY_GRAPH_MODE_TO_FLAGS.keys()))
    raise typer.BadParameter(f"Unknown graph_mode={mode!r}. Supported legacy modes: {supported}")


def _run_debug_across_supported_modes(base_config: PretrainConfig, runner: Callable[[PretrainConfig], None]):
    for mode in ("both", "no_cfg", "no_ddg", "none"):
        use_cfg, use_ddg = LEGACY_GRAPH_MODE_TO_FLAGS[mode]
        mode_config = copy.deepcopy(base_config)
        mode_config.use_cfg = use_cfg
        mode_config.use_ddg = use_ddg
        print(f"\n=== debug graph_mode={mode} (cfg={use_cfg}, ddg={use_ddg}) ===")
        runner(mode_config)


def _clone_model_config(config: PretrainConfig, vocab_size: int) -> PretrainConfig:
    model_config = copy.deepcopy(config)
    model_config.vocab_size = vocab_size
    return model_config


def _graph_tag(config: PretrainConfig) -> str:
    if config.use_cfg and config.use_ddg:
        return "cfg_ddg"
    if config.use_cfg:
        return "cfg"
    if config.use_ddg:
        return "ddg"
    return "plain"


def _append_graph_suffix(path: str, graph_tag: str) -> str:
    if path.endswith(f"_{graph_tag}"):
        return path
    return f"{path}_{graph_tag}"


def _resolve_debug_vocab_size(config: PretrainConfig) -> int:
    try:
        tokenizer = load_tokenizer(config.tokenizer_path)
        return len(tokenizer.get_vocab())
    except Exception as exc:
        fallback = int(getattr(config, "vocab_size", 0) or 0)
        if fallback <= 0:
            fallback = 32000
        print(f"Tokenizer load failed in debug mode ({exc}), fallback vocab_size={fallback}")
        return fallback


def _build_dummy_view(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    model_config: PretrainConfig,
    device: torch.device,
    use_labels: bool,
    ddg_edges_per_sample: int = 64,
) -> Dict[str, Any]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    seq_lengths = [seq_len] * batch_size
    ddg_graphs = []
    cfg_graphs = []
    max_edges = min(ddg_edges_per_sample, max(seq_len - 1, 0))

    for _ in range(batch_size):
        if model_config.use_ddg:
            ddg_graphs.append([[idx, idx, idx + 1, idx + 1] for idx in range(max_edges)])
        if model_config.use_cfg:
            split = max(seq_len // 2, 1)
            cfg_graphs.append([[0, split - 1, split, seq_len - 1, 1.0]] if seq_len > 1 else [])

    ddg_tensors = (
        build_batched_span_graph_tensors(ddg_graphs, seq_lengths, include_edge_attr=False)
        if model_config.use_ddg
        else {"node_spans": None, "node_batch": None, "edge_index": None}
    )
    cfg_tensors = (
        build_batched_span_graph_tensors(cfg_graphs, seq_lengths, include_edge_attr=True)
        if model_config.use_cfg
        else {"node_spans": None, "node_batch": None, "edge_index": None, "edge_attr": None}
    )

    view = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ddg_node_spans": ddg_tensors["node_spans"].to(device) if ddg_tensors["node_spans"] is not None else None,
        "ddg_node_batch": ddg_tensors["node_batch"].to(device) if ddg_tensors["node_batch"] is not None else None,
        "ddg_edge_index": ddg_tensors["edge_index"].to(device) if ddg_tensors["edge_index"] is not None else None,
        "cfg_node_spans": cfg_tensors["node_spans"].to(device) if cfg_tensors["node_spans"] is not None else None,
        "cfg_node_batch": cfg_tensors["node_batch"].to(device) if cfg_tensors["node_batch"] is not None else None,
        "cfg_edge_index": cfg_tensors["edge_index"].to(device) if cfg_tensors["edge_index"] is not None else None,
        "cfg_edge_attr": cfg_tensors["edge_attr"].to(device) if cfg_tensors["edge_attr"] is not None else None,
        "group_ids": torch.arange(batch_size, device=device, dtype=torch.long),
    }

    if use_labels:
        view["labels"] = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    return view


def _debug_with_dummy_batch(config: PretrainConfig, use_gpu: bool):
    mode_name = "GPU" if use_gpu else "CPU"
    print(f"--- Running in debug mode ({mode_name}, synthetic batch) ---")

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run GPU debug.")

    device = torch.device("cuda" if use_gpu else "cpu")
    seq_len = min(config.max_seq_length, 512)
    batch_size = 1 if use_gpu else 2

    vocab_size = _resolve_debug_vocab_size(config)
    model_config = _clone_model_config(config, vocab_size)
    model = MoCoPretrainModel(config=model_config).to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    view1 = _build_dummy_view(batch_size, seq_len, vocab_size, model_config, device, use_labels=True)
    view2 = _build_dummy_view(batch_size, seq_len, vocab_size, model_config, device, use_labels=False)

    print(f"use_cfg={config.use_cfg}, use_ddg={config.use_ddg}, seq_len={seq_len}, batch_size={batch_size}, device={device}")

    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory before forward: {before:.2f} MB")

    try:
        if use_gpu and config.bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(view1, view2)
        elif use_gpu and config.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(view1, view2)
        else:
            outputs = model(view1, view2)

        print("Forward pass successful")
        print(f"Loss: {outputs['loss'].item():.6f}")

        outputs["loss"].backward()
        print("Backward pass successful")

        if use_gpu and torch.cuda.is_available():
            after = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU memory after backward: {after:.2f} MB")
            print(f"GPU memory delta: {after - before:.2f} MB")
    finally:
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")

def debug_cpu(config: PretrainConfig = DEFAULT_CONFIG):
    _debug_with_dummy_batch(config, use_gpu=False)


def debug_gpu(config: PretrainConfig = DEFAULT_CONFIG):
    _debug_with_dummy_batch(config, use_gpu=True)


def profile_extreme_seq_len_memory(
    config: PretrainConfig = DEFAULT_CONFIG,
    batch_sizes: List[int] = None,
    seq_len: int = None,
    ddg_edges_per_sample: int = 64,
    run_backward: bool = True,
    mem_probe: bool = False,
):
    print("--- Profiling GPU memory at max seq len ---")

    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot profile GPU memory.")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if seq_len is None:
        seq_len = config.max_seq_length

    tokenizer = load_tokenizer(config.tokenizer_path)

    model_config = _clone_model_config(config, len(tokenizer.get_vocab()))
    if run_backward:
        model = MoCoPretrainModel(config=model_config)
    else:
        model = ReFormerPretrainModel(config=model_config)
    model = model.to(device)
    if config.gradient_checkpointing and run_backward:
        model.gradient_checkpointing_enable()
    if run_backward:
        model.train()
    else:
        model.eval()

    if config.bf16:
        autocast_dtype = torch.bfloat16
    elif config.fp16:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    scaler = None
    optimizer = None
    if run_backward:
        if config.fp16:
            scaler = torch.cuda.amp.GradScaler()

        def build_optimizer():
            optim_name = (config.optim or "").lower()
            if optim_name in {"paged_adamw_8bit", "adamw_8bit"}:
                if bnb is None:
                    print("bitsandbytes not available; falling back to torch AdamW")
                    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
                return bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
            return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        optimizer = build_optimizer()

    def build_dummy_view(batch_size: int, use_labels: bool) -> Dict[str, Any]:
        vocab_size = model_config.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        labels = None
        if use_labels:
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        seq_lengths = [seq_len] * batch_size
        ddg_graphs = []
        cfg_graphs = []
        max_edges = min(ddg_edges_per_sample, max(seq_len - 1, 0))
        for _ in range(batch_size):
            if model_config.use_ddg:
                ddg_graphs.append([[idx, idx, idx + 1, idx + 1] for idx in range(max_edges)])
            if model_config.use_cfg:
                split = max(seq_len // 2, 1)
                cfg_graphs.append([[0, split - 1, split, seq_len - 1, 1.0]] if seq_len > 1 else [])

        ddg_tensors = (
            build_batched_span_graph_tensors(ddg_graphs, seq_lengths, include_edge_attr=False)
            if model_config.use_ddg
            else {"node_spans": None, "node_batch": None, "edge_index": None}
        )
        cfg_tensors = (
            build_batched_span_graph_tensors(cfg_graphs, seq_lengths, include_edge_attr=True)
            if model_config.use_cfg
            else {"node_spans": None, "node_batch": None, "edge_index": None, "edge_attr": None}
        )

        view = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ddg_node_spans": ddg_tensors["node_spans"].to(device) if ddg_tensors["node_spans"] is not None else None,
            "ddg_node_batch": ddg_tensors["node_batch"].to(device) if ddg_tensors["node_batch"] is not None else None,
            "ddg_edge_index": ddg_tensors["edge_index"].to(device) if ddg_tensors["edge_index"] is not None else None,
            "cfg_node_spans": cfg_tensors["node_spans"].to(device) if cfg_tensors["node_spans"] is not None else None,
            "cfg_node_batch": cfg_tensors["node_batch"].to(device) if cfg_tensors["node_batch"] is not None else None,
            "cfg_edge_index": cfg_tensors["edge_index"].to(device) if cfg_tensors["edge_index"] is not None else None,
            "cfg_edge_attr": cfg_tensors["edge_attr"].to(device) if cfg_tensors["edge_attr"] is not None else None,
        }
        if run_backward:
            group_ids = torch.arange(batch_size, device=device, dtype=torch.long)
            view["group_ids"] = group_ids
        if labels is not None:
            view["labels"] = labels
        return view

    for batch_size in batch_sizes:
        print(f"\n[Batch Size = {batch_size}] seq_len={seq_len}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            if run_backward:
                view1 = build_dummy_view(batch_size, use_labels=True)
                view2 = build_dummy_view(batch_size, use_labels=False)
            else:
                view1 = build_dummy_view(batch_size, use_labels=False)
                view2 = None

            if mem_probe:
                mem_before_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_before_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Memory before forward: {mem_before_alloc:.2f} MB allocated, {mem_before_reserved:.2f} MB reserved")

            if run_backward:
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        outputs = model(view1, view2)
                else:
                    outputs = model(view1, view2)
            else:
                with torch.inference_mode():
                    if autocast_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            outputs = model(**view1)
                    else:
                        outputs = model(**view1)

            if mem_probe:
                mem_after_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_after_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"Memory after forward: {mem_after_alloc:.2f} MB allocated, {mem_after_reserved:.2f} MB reserved")

            if run_backward:
                if scaler is not None:
                    scaler.scale(outputs["loss"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs["loss"].backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if mem_probe:
                    mem_after_backward_alloc = torch.cuda.memory_allocated() / 1024**2
                    mem_after_backward_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(
                        "Memory after backward: "
                        f"{mem_after_backward_alloc:.2f} MB allocated, "
                        f"{mem_after_backward_reserved:.2f} MB reserved"
                    )

            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            max_reserved = torch.cuda.max_memory_reserved() / 1024**2
            print(f"Peak allocated: {max_allocated:.2f} MB")
            print(f"Peak reserved:  {max_reserved:.2f} MB")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("OOM at this batch size.")
            else:
                print(f"Runtime error: {e}")
            torch.cuda.empty_cache()



def main(config: PretrainConfig = DEFAULT_CONFIG):
    tokenizer = load_tokenizer(config.tokenizer_path)

    train_bundle = _resolve_train_bundle(config)
    _apply_bundle_to_config(config, train_bundle)
    train_dataset_pool, train_dataset_idx, train_positive_map = _load_dataset_bundle(train_bundle)
    print(
        "Train dataset bundle: "
        f"dir={train_bundle.dataset_dir}, prefix={train_bundle.prefix}, "
        f"anchors={len(train_dataset_idx):,}"
    )

    validation_dataset_idx = None
    validation_collator = None
    if config.do_eval:
        (
            validation_bundle,
            validation_dataset_pool,
            validation_dataset_idx,
            validation_positive_map,
        ) = _load_validation_bundle(config, train_bundle)
        if validation_bundle is not None:
            _apply_bundle_to_config(config, validation_bundle)
            validation_group_id_mapping = compute_group_ids(validation_positive_map)
            validation_collator = MoCoDataCollator(
                tokenizer=tokenizer,
                dataset_pool=validation_dataset_pool,
                map_file=validation_positive_map,
                group_id_mapping=validation_group_id_mapping,
                config=config,
            )
            print(
                "Validation dataset bundle: "
                f"dir={validation_bundle.dataset_dir}, prefix={validation_bundle.prefix}, "
                f"anchors={len(validation_dataset_idx):,}"
            )
    else:
        print("Validation disabled by config.do_eval=False.")

    group_id_mapping = compute_group_ids(train_positive_map)

    # dataset = dataset.remove_columns({
    #     "cfg_graph": "cfg_adj_list",
    #     "ddg_graph": "ddg_adj_list",
    # })
    train_collator = MoCoDataCollator(
        tokenizer=tokenizer,
        dataset_pool=train_dataset_pool,
        map_file=train_positive_map,
        group_id_mapping=group_id_mapping,
        config=config
    )

    graph_tag = _graph_tag(config)
    output_dir = _append_graph_suffix(config.output_dir, graph_tag)
    logging_dir = _append_graph_suffix(config.logging_dir, graph_tag)
    final_model_dir = _append_graph_suffix(config.final_model_dir, graph_tag)

    bad_batch_log_path = None
    if config.skip_bad_batch:
        if config.bad_batch_log_path:
            bad_batch_log_path = config.bad_batch_log_path
        else:
            os.makedirs(output_dir, exist_ok=True)
            bad_batch_log_path = os.path.join(output_dir, "bad_batches.log")

        class SafeDataCollator:
            def __init__(self, collator, log_path):
                self.collator = collator
                self.log_path = log_path

            def __call__(self, examples):
                try:
                    return self.collator(examples)
                except Exception as exc:
                    if isinstance(examples, dict):
                        anchor_indices = examples.get("anchor_idx", [])
                    else:
                        anchor_indices = [ex.get("anchor_idx") for ex in examples]
                    msg = f"[BAD BATCH] anchor_idx={anchor_indices} error={exc}"
                    print(msg)
                    traceback.print_exc()
                    if self.log_path:
                        with open(self.log_path, "a") as f:
                            f.write(msg + "\n")
                            traceback.print_exc(file=f)
                    return None

        train_collator = SafeDataCollator(train_collator, bad_batch_log_path)

    model_config = _clone_model_config(config, len(tokenizer.get_vocab()))
    model = MoCoPretrainModel(config=model_config)

    train_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,  # -1 表示按 num_train_epochs 训练，默认 1 个 epoch
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "remove_unused_columns": config.remove_unused_columns,
        "dataloader_num_workers": config.dataloader_num_workers,
        "torch_compile": config.torch_compile,
        "logging_dir": logging_dir,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "optim": config.optim,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        # "save_safetensors": False,
        "logging_strategy": config.logging_strategy,
        "logging_steps": config.logging_steps,
        "report_to": config.report_to,
    }
    if validation_dataset_idx is not None:
        eval_strategy = str(config.eval_strategy)
        _add_eval_strategy_arg(train_args_kwargs, eval_strategy)
        train_args_kwargs["do_eval"] = True
        train_args_kwargs["per_device_eval_batch_size"] = (
            config.per_device_eval_batch_size or config.per_device_train_batch_size
        )
        if eval_strategy.lower() == "steps":
            train_args_kwargs["eval_steps"] = config.eval_steps or config.save_steps
    else:
        _add_eval_strategy_arg(train_args_kwargs, "no")
        train_args_kwargs["do_eval"] = False

    train_args = TrainingArguments(**train_args_kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    class SafeTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.bad_batch_log_path = kwargs.pop("bad_batch_log_path", None)
            self.eval_data_collator = kwargs.pop("eval_data_collator", None)
            super().__init__(*args, **kwargs)
            self.skipped_steps = 0
            self.can_return_loss = True
            self._latest_train_loss: Optional[float] = None
            self._latest_train_loss_step: Optional[int] = None
            self._latest_train_metric_name: Optional[str] = None
            self._latest_validation_loss: Optional[float] = None
            self._latest_validation_loss_step: Optional[int] = None
            self._latest_validation_metric_name: Optional[str] = None
            self.best_train_loss: Optional[float] = None
            self.best_train_loss_step: Optional[int] = None
            self.best_validation_loss: Optional[float] = None
            self.best_validation_loss_step: Optional[int] = None

        @staticmethod
        def _as_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _checkpoint_dir_for_current_step(self, trial=None) -> str:
            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            return os.path.join(run_dir, checkpoint_folder)

        def _write_named_checkpoint_metadata(
            self,
            checkpoint_dir: str,
            *,
            alias: str,
            source_checkpoint: str,
            metric_name: Optional[str] = None,
            metric_value: Optional[float] = None,
        ) -> None:
            metadata = {
                "alias": alias,
                "source_checkpoint": os.path.basename(source_checkpoint),
                "global_step": int(self.state.global_step),
            }
            if metric_name is not None:
                metadata["metric_name"] = metric_name
            if metric_value is not None:
                metadata["metric_value"] = metric_value

            with open(os.path.join(checkpoint_dir, "named_checkpoint_info.json"), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")

        def _replace_named_checkpoint(
            self,
            source_checkpoint: str,
            alias: str,
            *,
            metric_name: Optional[str] = None,
            metric_value: Optional[float] = None,
        ) -> Optional[str]:
            if not self.args.should_save or not os.path.isdir(source_checkpoint):
                return None

            run_dir = os.path.dirname(source_checkpoint)
            target_dir = os.path.join(run_dir, alias)
            if os.path.abspath(source_checkpoint) == os.path.abspath(target_dir):
                return target_dir

            tmp_dir = os.path.join(run_dir, f".{alias}.tmp-{os.getpid()}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

            try:
                shutil.copytree(source_checkpoint, tmp_dir, symlinks=True)
                self._write_named_checkpoint_metadata(
                    tmp_dir,
                    alias=alias,
                    source_checkpoint=source_checkpoint,
                    metric_name=metric_name,
                    metric_value=metric_value,
                )
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                os.replace(tmp_dir, target_dir)
                return target_dir
            finally:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        def _maybe_update_best_train_checkpoint(self, source_checkpoint: str) -> None:
            loss = self._latest_train_loss
            step = self._latest_train_loss_step
            if loss is None or step != self.state.global_step:
                return

            if self.best_train_loss is None or loss < self.best_train_loss:
                self.best_train_loss = loss
                self.best_train_loss_step = int(self.state.global_step)
                self._replace_named_checkpoint(
                    source_checkpoint,
                    "checkpoint-best-train-loss",
                    metric_name=self._latest_train_metric_name or "loss",
                    metric_value=loss,
                )

        def _maybe_update_best_validation_checkpoint(self, source_checkpoint: str) -> None:
            loss = self._latest_validation_loss
            step = self._latest_validation_loss_step
            if loss is None or step != self.state.global_step:
                return

            if self.best_validation_loss is None or loss < self.best_validation_loss:
                self.best_validation_loss = loss
                self.best_validation_loss_step = int(self.state.global_step)
                self._replace_named_checkpoint(
                    source_checkpoint,
                    "checkpoint-best-validation-loss",
                    metric_name=self._latest_validation_metric_name or "eval_loss",
                    metric_value=loss,
                )

        def _sync_best_checkpoints(self, source_checkpoint: str) -> None:
            self._maybe_update_best_train_checkpoint(source_checkpoint)
            self._maybe_update_best_validation_checkpoint(source_checkpoint)

        def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
            train_loss = self._as_float(logs.get("loss"))
            if train_loss is not None:
                self._latest_train_loss = train_loss
                self._latest_train_loss_step = int(self.state.global_step)
                self._latest_train_metric_name = "loss"
            return super().log(logs, *args, **kwargs)

        def evaluate(self, *args, **kwargs):
            metrics = super().evaluate(*args, **kwargs)
            loss_key = None
            for key in ("eval_loss", "validation_loss"):
                if key in metrics:
                    loss_key = key
                    break
            if loss_key is None:
                loss_key = next((key for key in metrics if key.endswith("_loss")), None)

            if loss_key is not None:
                validation_loss = self._as_float(metrics.get(loss_key))
                if validation_loss is not None:
                    self._latest_validation_loss = validation_loss
                    self._latest_validation_loss_step = int(self.state.global_step)
                    self._latest_validation_metric_name = loss_key
            return metrics

        def _save_checkpoint(self, *args, **kwargs):
            trial = kwargs.get("trial")
            if trial is None and len(args) >= 2:
                trial = args[1]

            result = super()._save_checkpoint(*args, **kwargs)
            checkpoint_dir = self._checkpoint_dir_for_current_step(trial=trial)
            self._sync_best_checkpoints(checkpoint_dir)
            return result

        def note_final_train_metrics(self, metrics: Dict[str, float]) -> None:
            if self.best_train_loss is not None:
                return
            train_loss = self._as_float(metrics.get("train_loss"))
            if train_loss is not None:
                self._latest_train_loss = train_loss
                self._latest_train_loss_step = int(self.state.global_step)
                self._latest_train_metric_name = "train_loss"

        def save_final_checkpoint(self) -> Optional[str]:
            if self.state.global_step <= 0:
                return None

            checkpoint_dir = self._checkpoint_dir_for_current_step()
            if not os.path.isdir(checkpoint_dir):
                self._save_checkpoint(self.model, trial=None)
                checkpoint_dir = self._checkpoint_dir_for_current_step()
            else:
                self._sync_best_checkpoints(checkpoint_dir)

            self._replace_named_checkpoint(checkpoint_dir, "checkpoint-last")
            return checkpoint_dir

        def update_best_validation_from_checkpoint(self, source_checkpoint: Optional[str]) -> None:
            if source_checkpoint is not None:
                self._maybe_update_best_validation_checkpoint(source_checkpoint)

        def get_eval_dataloader(self, eval_dataset=None):
            if self.eval_data_collator is None:
                return super().get_eval_dataloader(eval_dataset)

            original_collator = self.data_collator
            self.data_collator = self.eval_data_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = original_collator

        def _init_training_state(
            self,
            max_steps,
            num_update_steps_per_epoch,
            num_train_epochs,
            resume_from_checkpoint,
            trial,
        ):
            result = super()._init_training_state(
                max_steps,
                num_update_steps_per_epoch,
                num_train_epochs,
                resume_from_checkpoint,
                trial,
            )

            old_steps = {
                "logging_steps": self.state.logging_steps,
                "eval_steps": self.state.eval_steps,
                "save_steps": self.state.save_steps,
            }
            self.state.compute_steps(self.args, max_steps)
            new_steps = {
                "logging_steps": self.state.logging_steps,
                "eval_steps": self.state.eval_steps,
                "save_steps": self.state.save_steps,
            }

            if resume_from_checkpoint is not None and old_steps != new_steps:
                print(
                    "Checkpoint trainer_state step intervals were overridden by current args: "
                    f"{old_steps} -> {new_steps}"
                )

            return result

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            """Save with torch bin format to avoid safetensors shared-tensor errors."""
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            model_to_save = unwrap_model(self.model)

            if hasattr(model_to_save, "save_pretrained"):
                if state_dict is None:
                    state_dict = model_to_save.state_dict()

                save_kwargs = {"state_dict": state_dict}
                sig = inspect.signature(model_to_save.save_pretrained)
                if "safe_serialization" in sig.parameters:
                    save_kwargs["safe_serialization"] = False

                model_to_save.save_pretrained(output_dir, **save_kwargs)
            else:
                if state_dict is None:
                    state_dict = model_to_save.state_dict()
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        def training_step(self, model, inputs, num_items_in_batch=None):
            if inputs is None:
                self.skipped_steps += 1
                msg = f"[SKIP STEP] step={self.state.global_step} reason=collator_returned_none"
                print(msg)
                if self.bad_batch_log_path:
                    with open(self.bad_batch_log_path, "a") as f:
                        f.write(msg + "\n")
                return torch.zeros((), device=self.args.device)
            # Do not swallow model/config/runtime errors here. If forward/backward
            # fails, raise immediately so training cannot falsely appear successful.
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    trainer = SafeTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset_idx,
        eval_dataset=validation_dataset_idx,
        data_collator=train_collator,
        eval_data_collator=validation_collator,
        bad_batch_log_path=bad_batch_log_path,
    )

    # 根据配置决定是否从检查点恢复
    last_checkpoint = None
    if config.resume_from_checkpoint:
        if config.resume_checkpoint_path:
            last_checkpoint = config.resume_checkpoint_path
        elif os.path.isdir(train_args.output_dir):
            last_checkpoint = get_last_checkpoint(train_args.output_dir)

    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch...")
        train_result = trainer.train()

    if trainer.skipped_steps:
        print(f"Skipped steps due to bad batches: {trainer.skipped_steps}")
        if train_args.max_steps > 0 and trainer.skipped_steps >= train_args.max_steps:
            raise RuntimeError(
                "All training steps were skipped due to bad batches. "
                "Please check bad_batches.log and dataset/collator integrity."
            )

    print("Training finished.")

    # --- 训练完成后 ---

    # 记录训练过程中的一些指标，并确保最后一步也有可恢复的 checkpoint。
    metrics = train_result.metrics
    trainer.note_final_train_metrics(metrics)
    final_checkpoint_dir = trainer.save_final_checkpoint()

    # 保存最终的模型、分词器和配置
    # 这会创建一个干净的、可以被 from_pretrained 加载的最终模型文件夹
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    if validation_dataset_idx is not None:
        validation_metrics = trainer.evaluate(metric_key_prefix="validation")
        trainer.update_best_validation_from_checkpoint(final_checkpoint_dir)
        trainer.log_metrics("validation", validation_metrics)
        trainer.save_metrics("validation", validation_metrics)
    trainer.save_state() # 保存Trainer的状态，包括随机种子等


@app.callback(invoke_without_command=True)
def cli(ctx: typer.Context):
    """Pretrain CLI. Default behavior is `debug cpu`."""
    if ctx.invoked_subcommand is None:
        debug_cpu(DEFAULT_CONFIG)


@app.command("debug")
def debug_command(
    device: str = typer.Argument("cpu", help="Run on cpu or gpu."),
    graph_mode: Optional[str] = typer.Argument(
        None,
        help="Legacy positional graph mode (all/both/no_cfg/no_ddg/none).",
    ),
    use_cfg: bool = typer.Option(True, "--cfg/--no-cfg", help="Enable CFG graph branch."),
    use_ddg: bool = typer.Option(True, "--ddg/--no-ddg", help="Enable DDG graph branch."),
    override: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Override config item, repeatable. Example: --set num_train_epochs=2 --set bf16=false",
    ),
):
    device = device.lower()

    if device not in {"cpu", "gpu"}:
        raise typer.BadParameter("device must be one of: cpu, gpu")

    runner: Callable[[PretrainConfig], None] = debug_cpu if device == "cpu" else debug_gpu
    config = _build_config(overrides=override)

    if graph_mode is not None:
        mode = graph_mode.lower()
        if mode == "all":
            _run_debug_across_supported_modes(config, runner)
            return
        use_cfg, use_ddg = _resolve_graph_flags_from_legacy_mode(mode)

    config.use_cfg = use_cfg
    config.use_ddg = use_ddg
    runner(config)


@app.command("profile-gpu-mem")
def profile_gpu_mem_command(
    forward_only: bool = typer.Option(False, "--forward-only", help="Only run forward pass."),
    mem_probe: bool = typer.Option(False, "--mem-probe", help="Print allocated/reserved memory info."),
    override: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Override config item, repeatable. Example: --set max_seq_length=2048",
    ),
):
    config = _build_config(overrides=override)
    profile_extreme_seq_len_memory(
        config=config,
        run_backward=not forward_only,
        mem_probe=mem_probe,
    )


@app.command("train")
def train_command(
    use_cfg: bool = typer.Option(True, "--cfg/--no-cfg", help="Enable CFG graph branch."),
    use_ddg: bool = typer.Option(True, "--ddg/--no-ddg", help="Enable DDG graph branch."),
    dataset_dir: Optional[str] = typer.Option(
        None,
        "--dataset-dir",
        "-d",
        help="Directory containing train_dataset_pool/train_task_dataset/train_positive_map.pkl.",
    ),
    validation_dataset_dir: Optional[str] = typer.Option(
        None,
        "--validation-dataset-dir",
        "--val-dataset-dir",
        help=(
            "Validation final_set directory. If omitted, the trainer tries a sibling "
            "validation_final_set, then validation_* files under --dataset-dir."
        ),
    ),
    validation: bool = typer.Option(
        True,
        "--validation/--no-validation",
        help="Enable validation when a validation dataset bundle is available.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume training from checkpoint. Default is false (start from scratch).",
    ),
    override: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Override config item, repeatable. Example: --set num_train_epochs=2 --set learning_rate=1e-4",
    ),
):
    config = _build_config(overrides=override)

    config.use_cfg = use_cfg
    config.use_ddg = use_ddg
    if dataset_dir is not None:
        config.train_dataset_dir = dataset_dir
        config.train_dataset_pool_path = None
        config.train_dataset_idx_path = None
        config.train_dataset_map_path = None
    if validation_dataset_dir is not None:
        config.validation_dataset_dir = validation_dataset_dir
        config.validation_dataset_pool_path = None
        config.validation_dataset_idx_path = None
        config.validation_dataset_map_path = None
    config.do_eval = validation
    config.resume_from_checkpoint = resume
    if not resume:
        config.resume_checkpoint_path = None
    main(config)


if __name__ == "__main__":
    app()
