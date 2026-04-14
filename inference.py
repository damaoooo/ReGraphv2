import json
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import typer

from Tokenizer.ir_tokenizer import load_tokenizer
from Utils.utils import DEFAULT_TOKENIZER_PATH
from evaluation import FunctionDataCollator, get_model


app = typer.Typer(add_completion=False)


def _trim_output(output: str, limit: int = 4000) -> str:
    if not output:
        return ""
    if len(output) <= limit:
        return output
    return output[:limit] + "\n...[truncated]..."


@dataclass
class FunctionAsmRecord:
    function_name: str
    function_path: str
    asm_text: str


@dataclass
class InferencePipelineConfig:
    repo_root: str = os.getcwd()
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    python_path: str = sys.executable
    workers: int = os.cpu_count() or 1
    resume: bool = True
    start_from_step2: bool = False
    conda_env: Optional[str] = None
    ida_path: Optional[str] = None
    save_database: bool = False
    model_path: Optional[str] = None
    device: Optional[str] = None
    max_length: int = 4096
    embedding_size: int = 768
    inference_batch_size: int = 8

    @property
    def bin2asm_script_path(self) -> str:
        return os.path.join(self.repo_root, "Scripts", "bin2asm.py")


class ReGraphInferencePipeline:
    def __init__(self, config: Optional[InferencePipelineConfig] = None):
        self.config = config or InferencePipelineConfig()
        self._tokenizer = None
        self._collator = None
        self._model = None
        self._device = None

    def _resolve_device(self) -> torch.device:
        if self._device is None:
            if self.config.device:
                self._device = torch.device(self.config.device)
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _use_bf16(self) -> bool:
        return self._resolve_device().type == "cuda"

    def _run_command(self, command_line: List[str]) -> None:
        result = subprocess.run(
            command_line,
            cwd=self.config.repo_root,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return

        stderr = _trim_output(result.stderr)
        stdout = _trim_output(result.stdout)
        raise RuntimeError(
            "Command failed:\n"
            f"cmd: {' '.join(command_line)}\n"
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.config.tokenizer_path)
        return self._tokenizer

    def get_collator(self) -> FunctionDataCollator:
        if self._collator is None:
            self._collator = FunctionDataCollator(
                self.get_tokenizer(),
                max_length=self.config.max_length,
            )
        return self._collator

    def load_model(self):
        if self._model is None:
            if not self.config.model_path:
                raise ValueError("model_path is required for embedding inference.")

            model = get_model(
                self.config.model_path,
                max_seq_length=self.config.max_length,
                embedding_size=self.config.embedding_size,
                tokenizer_path=self.config.tokenizer_path,
            )
            model.to(self._resolve_device())
            model.eval()
            self._model = model

        return self._model

    def run_inference(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        model = self.load_model()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_bf16()
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_context:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        return outputs

    def run_attention_probe(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        model = self.load_model()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_bf16()
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_context:
                return model.roformer.compute_last_layer_attention_weights(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

    def _iter_asm_files(self, input_path: str) -> Iterable[str]:
        normalized_path = os.path.abspath(input_path)

        if os.path.isfile(normalized_path):
            if not normalized_path.endswith(".asm"):
                raise ValueError(f"Unsupported file type: {normalized_path}")
            yield normalized_path
            return

        if not os.path.isdir(normalized_path):
            raise FileNotFoundError(f"Input path not found: {normalized_path}")

        asm_paths: List[str] = []
        for root, dirs, files in os.walk(normalized_path):
            dirs[:] = [
                directory
                for directory in dirs
                if not directory.startswith(".") and directory != "__pycache__"
            ]
            for file_name in files:
                if file_name.endswith(".asm"):
                    asm_paths.append(os.path.join(root, file_name))

        for asm_path in sorted(asm_paths):
            yield asm_path

    def _load_asm_record(self, asm_path: str) -> FunctionAsmRecord:
        with open(asm_path, "r", encoding="utf-8", errors="ignore") as asm_file:
            asm_text = asm_file.read()

        return FunctionAsmRecord(
            function_name=os.path.splitext(os.path.basename(asm_path))[0],
            function_path=asm_path,
            asm_text=asm_text,
        )

    def collect_function_asms(self, input_path: str) -> List[FunctionAsmRecord]:
        payload: List[FunctionAsmRecord] = []
        for asm_path in self._iter_asm_files(input_path):
            try:
                payload.append(self._load_asm_record(asm_path))
            except Exception as exc:
                typer.echo(f"Skipping {asm_path}: {exc}", err=True)
        return payload

    def export_function_json(self, input_path: str, output_json_path: str) -> List[FunctionAsmRecord]:
        payload = self.collect_function_asms(input_path)
        output_dir = os.path.dirname(os.path.abspath(output_json_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in payload], handle, ensure_ascii=True)
        return payload

    def _record_to_model_sample(self, payload: FunctionAsmRecord) -> Dict[str, Any]:
        return {"text": payload.asm_text}

    def _prepare_inference_batch(self, payloads: Sequence[FunctionAsmRecord]) -> Dict[str, Any]:
        collator_inputs = [self._record_to_model_sample(payload) for payload in payloads]
        return self.get_collator()(collator_inputs)

    def _embed_batch(
        self,
        payloads: Sequence[FunctionAsmRecord],
        verbose: bool = False,
    ):
        if not payloads:
            return ([], []) if verbose else []

        batch = self._prepare_inference_batch(payloads)
        device = self._resolve_device()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = self.run_inference(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs["embedding"]
        embeddings_list: List[List[float]] = embeddings.detach().cpu().float().tolist()

        if not verbose:
            return embeddings_list

        last_attention = self.run_attention_probe(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).detach().cpu().float()
        cpu_input_ids = input_ids.detach().cpu()
        cpu_attention_mask = attention_mask.detach().cpu()
        tokenizer = self.get_tokenizer()

        verbose_data: List[Dict[str, Any]] = []
        for idx, payload in enumerate(payloads):
            valid_len = int(cpu_attention_mask[idx].sum().item())
            token_ids = cpu_input_ids[idx, :valid_len].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            verbose_data.append(
                {
                    "file_path": payload.function_path,
                    "asm": payload.asm_text,
                    "tokens": tokens,
                    "attention_weights": last_attention[idx, :valid_len, :valid_len].tolist(),
                }
            )

        return embeddings_list, verbose_data

    def _iter_batches(self, payloads: Sequence[FunctionAsmRecord]) -> Iterable[Sequence[FunctionAsmRecord]]:
        batch_size = max(1, self.config.inference_batch_size)
        for start_idx in range(0, len(payloads), batch_size):
            yield payloads[start_idx:start_idx + batch_size]

    def _store_embedding(
        self,
        embeddings_by_name: Dict[str, Any],
        name_counts: Dict[str, int],
        payload: FunctionAsmRecord,
        embedding: Any,
    ) -> None:
        base_name = payload.function_name or os.path.splitext(os.path.basename(payload.function_path))[0]
        duplicate_count = name_counts.get(base_name, 0)
        name_counts[base_name] = duplicate_count + 1
        final_name = base_name if duplicate_count == 0 else f"{base_name}#{duplicate_count + 1}"
        embeddings_by_name[final_name] = embedding

    def embed_function_asms(
        self,
        payloads: Sequence[FunctionAsmRecord],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        embeddings_by_name: Dict[str, Any] = {}
        name_counts: Dict[str, int] = {}

        for batch in self._iter_batches(payloads):
            try:
                if verbose:
                    batch_embeddings, batch_verbose = self._embed_batch(batch, verbose=True)
                    for payload, embedding, vdata in zip(batch, batch_embeddings, batch_verbose):
                        self._store_embedding(
                            embeddings_by_name,
                            name_counts,
                            payload,
                            {"embedding": embedding, **vdata},
                        )
                else:
                    batch_embeddings = self._embed_batch(batch)
                    for payload, embedding in zip(batch, batch_embeddings):
                        self._store_embedding(embeddings_by_name, name_counts, payload, embedding)
            except Exception as batch_exc:
                if len(batch) == 1:
                    typer.echo(f"Skipping {batch[0].function_path}: {batch_exc}", err=True)
                    continue

                for payload in batch:
                    try:
                        if verbose:
                            emb_list, vdata_list = self._embed_batch([payload], verbose=True)
                            self._store_embedding(
                                embeddings_by_name,
                                name_counts,
                                payload,
                                {"embedding": emb_list[0], **vdata_list[0]},
                            )
                        else:
                            embedding = self._embed_batch([payload])[0]
                            self._store_embedding(embeddings_by_name, name_counts, payload, embedding)
                    except Exception as item_exc:
                        typer.echo(f"Skipping {payload.function_path}: {item_exc}", err=True)

        return embeddings_by_name

    def embed_asm_dir(self, input_path: str, verbose: bool = False) -> Dict[str, Any]:
        payloads = self.collect_function_asms(input_path)
        return self.embed_function_asms(payloads, verbose=verbose)

    def run_bin2asm(self, executable_dir: str, asm_output_dir: str) -> str:
        command_line = [
            self.config.python_path,
            self.config.bin2asm_script_path,
            "--input-path",
            executable_dir,
            "--output",
            asm_output_dir,
            "--workers",
            str(self.config.workers),
        ]

        if self.config.resume:
            command_line.append("--resume")
        if self.config.start_from_step2:
            command_line.append("--start-from-step2")
        if self.config.conda_env is not None:
            command_line.extend(["--conda-env", self.config.conda_env])
        if self.config.ida_path:
            command_line.extend(["--ida-path", self.config.ida_path])
        if self.config.save_database:
            command_line.append("--save-database")

        self._run_command(command_line)
        return os.path.join(asm_output_dir, os.path.basename(executable_dir.rstrip("/")))

    def embed_executable_dir(
        self,
        executable_dir: str,
        asm_output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if asm_output_dir is None:
            with tempfile.TemporaryDirectory(prefix="regraph_asm_") as temp_dir:
                asm_dir = self.run_bin2asm(executable_dir, temp_dir)
                return self.embed_asm_dir(asm_dir, verbose=verbose)

        asm_dir = self.run_bin2asm(executable_dir, asm_output_dir)
        return self.embed_asm_dir(asm_dir, verbose=verbose)

    def export_embeddings_json(
        self,
        input_path: str,
        output_json_path: str,
        input_mode: str = "binary",
        asm_output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if input_mode == "binary":
            payload = self.embed_executable_dir(
                input_path,
                asm_output_dir=asm_output_dir,
                verbose=verbose,
            )
        elif input_mode == "asm":
            payload = self.embed_asm_dir(input_path, verbose=verbose)
        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        output_dir = os.path.dirname(os.path.abspath(output_json_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
        return payload


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Directory containing binaries or ASM files."),
    output_json_path: str = typer.Argument(..., help="Path to the exported embeddings JSON file."),
    model_path: str = typer.Option(..., help="Model checkpoint directory."),
    input_mode: str = typer.Option("binary", help="Input mode: binary or asm."),
    asm_output_dir: Optional[str] = typer.Option(None, help="ASM export root when input-mode=binary."),
    tokenizer_path: str = typer.Option(DEFAULT_TOKENIZER_PATH, help="Tokenizer path."),
    workers: int = typer.Option(os.cpu_count() or 1, help="Worker count for ASM export."),
    python_path: str = typer.Option(sys.executable, help="Python executable used for subprocess stages."),
    repo_root: str = typer.Option(os.getcwd(), help="Repository root."),
    resume: bool = typer.Option(True, help="Resume ASM export when possible."),
    start_from_step2: bool = typer.Option(False, help="Skip .i64 generation and reuse existing .i64 files."),
    conda_env: Optional[str] = typer.Option(None, help="Conda env passed through to bin2asm."),
    ida_path: Optional[str] = typer.Option(None, help="IDA installation path passed through to bin2asm."),
    save_database: bool = typer.Option(False, help="Pass --save-database through to bin2asm."),
    max_length: int = typer.Option(4096, help="Maximum sequence length."),
    embedding_size: int = typer.Option(768, help="Embedding size."),
    inference_batch_size: int = typer.Option(8, help="Embedding inference batch size."),
    device: Optional[str] = typer.Option(None, help="Torch device, e.g. cpu or cuda."),
    verbose: bool = typer.Option(False, help="Include ASM text, tokens and attention weights."),
):
    normalized_mode = input_mode.strip().lower()
    if normalized_mode not in {"binary", "asm"}:
        raise typer.BadParameter("input_mode must be one of: binary, asm")

    pipeline = ReGraphInferencePipeline(
        InferencePipelineConfig(
            repo_root=repo_root,
            tokenizer_path=tokenizer_path,
            python_path=python_path,
            workers=workers,
            resume=resume,
            start_from_step2=start_from_step2,
            conda_env=conda_env,
            ida_path=ida_path,
            save_database=save_database,
            model_path=model_path,
            device=device,
            max_length=max_length,
            embedding_size=embedding_size,
            inference_batch_size=inference_batch_size,
        )
    )
    payload = pipeline.export_embeddings_json(
        input_path=input_path,
        output_json_path=output_json_path,
        input_mode=normalized_mode,
        asm_output_dir=asm_output_dir,
        verbose=verbose,
    )
    print(f"Exported {len(payload)} embeddings to {output_json_path}")


if __name__ == "__main__":
    app()
