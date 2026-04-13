import ast
import copy
import inspect
import os
import pickle
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import typer
from transformers import Trainer, TrainingArguments
from transformers.trainer import unwrap_model
from transformers.trainer_utils import get_last_checkpoint

from Pretrain.pretrain_dataset import MoCoDataCollator, compute_group_ids, load_dataset
from Tokenizer.ir_tokenizer import load_tokenizer
from .pretrain_config import DEFAULT_CONFIG, PretrainConfig
from .pretrain_model import MoCoPretrainModel, ReFormerPretrainModel

app = typer.Typer(add_completion=False, no_args_is_help=False)


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


def _clone_model_config(config: PretrainConfig, vocab_size: int) -> PretrainConfig:
    model_config = copy.deepcopy(config)
    model_config.vocab_size = vocab_size
    return model_config


def _resolve_debug_vocab_size(config: PretrainConfig) -> int:
    try:
        tokenizer = load_tokenizer(config.tokenizer_path)
        return len(tokenizer.get_vocab())
    except Exception as exc:
        fallback = int(getattr(config, "vocab_size", 0) or 0) or 32000
        print(f"Tokenizer load failed in debug mode ({exc}), fallback vocab_size={fallback}")
        return fallback


def _build_dummy_view(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    use_labels: bool,
) -> Dict[str, Any]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    view = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "group_ids": torch.arange(batch_size, device=device, dtype=torch.long),
    }

    if use_labels:
        view["labels"] = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    return view


def _debug_with_dummy_batch(config: PretrainConfig, use_gpu: bool):
    mode_name = "GPU" if use_gpu else "CPU"
    print(f"--- Running in debug mode ({mode_name}, synthetic text batch) ---")

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

    view1 = _build_dummy_view(batch_size, seq_len, vocab_size, device, use_labels=True)
    view2 = _build_dummy_view(batch_size, seq_len, vocab_size, device, use_labels=False)

    print(f"seq_len={seq_len}, batch_size={batch_size}, device={device}")

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
    batch_sizes: Optional[List[int]] = None,
    seq_len: Optional[int] = None,
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
    model = MoCoPretrainModel(config=model_config) if run_backward else ReFormerPretrainModel(config=model_config)
    model = model.to(device)

    if config.gradient_checkpointing and run_backward:
        model.gradient_checkpointing_enable()
    model.train() if run_backward else model.eval()

    if config.bf16:
        autocast_dtype = torch.bfloat16
    elif config.fp16:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate) if run_backward else None

    for batch_size in batch_sizes:
        print(f"\n[Batch Size = {batch_size}] seq_len={seq_len}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            if run_backward:
                view1 = _build_dummy_view(batch_size, seq_len, model_config.vocab_size, device, use_labels=True)
                view2 = _build_dummy_view(batch_size, seq_len, model_config.vocab_size, device, use_labels=False)
            else:
                view1 = _build_dummy_view(batch_size, seq_len, model_config.vocab_size, device, use_labels=False)
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

            if run_backward and optimizer is not None:
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
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                print("OOM at this batch size.")
            else:
                print(f"Runtime error: {exc}")
            torch.cuda.empty_cache()


def main(config: PretrainConfig = DEFAULT_CONFIG):
    tokenizer = load_tokenizer(config.tokenizer_path)
    train_dataset_pool = load_dataset(config.train_dataset_pool_path)
    train_dataset_idx = load_dataset(config.train_dataset_idx_path)

    with open(config.train_dataset_map_path, "rb") as handle:
        train_positive_map = pickle.load(handle)

    group_id_mapping = compute_group_ids(train_positive_map)
    my_collator = MoCoDataCollator(
        tokenizer=tokenizer,
        dataset_pool=train_dataset_pool,
        map_file=train_positive_map,
        group_id_mapping=group_id_mapping,
        config=config,
    )

    bad_batch_log_path = None
    if config.skip_bad_batch:
        if config.bad_batch_log_path:
            bad_batch_log_path = config.bad_batch_log_path
        else:
            os.makedirs(config.output_dir, exist_ok=True)
            bad_batch_log_path = os.path.join(config.output_dir, "bad_batches.log")

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
                        anchor_indices = [example.get("anchor_idx") for example in examples]
                    msg = f"[BAD BATCH] anchor_idx={anchor_indices} error={exc}"
                    print(msg)
                    traceback.print_exc()
                    if self.log_path:
                        with open(self.log_path, "a") as handle:
                            handle.write(msg + "\n")
                            traceback.print_exc(file=handle)
                    return None

        my_collator = SafeDataCollator(my_collator, bad_batch_log_path)

    model_config = _clone_model_config(config, len(tokenizer.get_vocab()))
    model = MoCoPretrainModel(config=model_config)

    train_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        fp16=config.fp16,
        bf16=config.bf16,
        remove_unused_columns=config.remove_unused_columns,
        dataloader_num_workers=config.dataloader_num_workers,
        torch_compile=config.torch_compile,
        logging_dir=config.logging_dir,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        optim=config.optim,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    class SafeTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            self.bad_batch_log_path = kwargs.pop("bad_batch_log_path", None)
            super().__init__(*args, **kwargs)
            self.skipped_steps = 0

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
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
                    with open(self.bad_batch_log_path, "a") as handle:
                        handle.write(msg + "\n")
                return torch.zeros((), device=self.args.device)
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    trainer = SafeTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset_idx,
        data_collator=my_collator,
        bad_batch_log_path=bad_batch_log_path,
    )

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
        if trainer.skipped_steps >= train_args.max_steps:
            raise RuntimeError(
                "All training steps were skipped due to bad batches. "
                "Please check bad_batches.log and dataset/collator integrity."
            )

    print("Training finished.")

    trainer.save_model(config.final_model_dir)
    tokenizer.save_pretrained(config.final_model_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


@app.callback(invoke_without_command=True)
def cli(ctx: typer.Context):
    """Pretrain CLI. Default behavior is `debug cpu`."""
    if ctx.invoked_subcommand is None:
        debug_cpu(DEFAULT_CONFIG)


@app.command("debug")
def debug_command(
    device: str = typer.Argument("cpu", help="Run on cpu or gpu."),
    override: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Override config item, repeatable. Example: --set max_steps=2000 --set bf16=false",
    ),
):
    device = device.lower()
    if device not in {"cpu", "gpu"}:
        raise typer.BadParameter("device must be one of: cpu, gpu")

    runner: Callable[[PretrainConfig], None] = debug_cpu if device == "cpu" else debug_gpu
    config = _build_config(overrides=override)
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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume training from checkpoint. Default is false (start from scratch).",
    ),
    override: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Override config item, repeatable. Example: --set max_steps=5000 --set learning_rate=1e-4",
    ),
):
    config = _build_config(overrides=override)
    config.resume_from_checkpoint = resume
    if not resume:
        config.resume_checkpoint_path = None
    main(config)


if __name__ == "__main__":
    app()
