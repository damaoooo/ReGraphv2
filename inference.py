import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import typer

from GraphBuilder.cfg_graph_builder import CFGGraphBuilder
from GraphBuilder.ddg_graph_builder import DataDependencyGraphBuilder
from Tokenizer.ir_tokenizer import load_tokenizer
from Tokenizer.normalizer import normalize_file
from Utils.utils import DEFAULT_TOKENIZER_PATH

from evaluation import FunctionDataCollator, get_model


app = typer.Typer(add_completion=False)


LLVM_DEFINE_NAME_PATTERN = re.compile(
    r'^\s*define\b.*?@(?:"([^"]+)"|([^(\s]+))\s*\(',
    re.MULTILINE,
)

GRAPH_MODE_METADATA_FILES = ("regraph_config.json", "config.json")

def infer_graph_flags_from_model_path(model_path: Optional[str]) -> tuple[Optional[bool], Optional[bool]]:
    if not model_path:
        return None, None

    normalized_model_path = os.path.abspath(model_path)
    for metadata_file in GRAPH_MODE_METADATA_FILES:
        metadata_path = os.path.join(normalized_model_path, metadata_file)
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r", encoding="utf-8") as metadata_fp:
            metadata = json.load(metadata_fp)

        if "use_cfg" in metadata or "use_ddg" in metadata:
            return metadata.get("use_cfg"), metadata.get("use_ddg")

    return None, None


def resolve_graph_flags(
    use_cfg: Optional[bool],
    use_ddg: Optional[bool],
    model_path: Optional[str],
) -> tuple[bool, bool]:
    inferred_use_cfg, inferred_use_ddg = infer_graph_flags_from_model_path(model_path)
    resolved_use_cfg = use_cfg if use_cfg is not None else inferred_use_cfg
    resolved_use_ddg = use_ddg if use_ddg is not None else inferred_use_ddg
    return bool(True if resolved_use_cfg is None else resolved_use_cfg), bool(True if resolved_use_ddg is None else resolved_use_ddg)


@dataclass
class FunctionGraphRecord:
    input_path: str
    purify_path: str
    instrumented_path: str
    cfg_dot: str
    ddg_dot: str


@dataclass
class FunctionGraphJson:
    function_name: str
    function_path: str
    normalized_ir: str
    cfg_graph: List[Any]
    ddg_graph: List[Any]


@dataclass
class InferencePipelineConfig:
    repo_root: str = "/path/to/rell"
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    python_path: str = "python"
    workers: int = os.cpu_count() or 1
    cleanup_graph_temp_files: bool = True
    use_graph_cache: bool = False
    resume: bool = True
    task1_start_from_step2: bool = False
    model_path: Optional[str] = None
    use_cfg: Optional[bool] = None
    use_ddg: Optional[bool] = None
    device: Optional[str] = None
    max_length: int = 4096
    embedding_size: int = 768
    inference_batch_size: int = 8

    @property
    def scripts_pipeline_path(self) -> str:
        return os.path.join(self.repo_root, "Scripts", "pipeline.py")


def build_edges_from_dot_files(
    llvm_ir_path: str,
    cfg_dot_path: Optional[str] = None,
    ddg_dot_path: Optional[str] = None,
    instrumented_ir_path: Optional[str] = None,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
) -> Dict[str, Any]:
    if not os.path.exists(llvm_ir_path):
        raise FileNotFoundError(f"LLVM IR file not found: {llvm_ir_path}")

    if cfg_dot_path is None and ddg_dot_path is None:
        raise ValueError("At least one of cfg_dot_path or ddg_dot_path must be provided.")

    tokenizer = load_tokenizer(tokenizer_path)
    normalized_ir = normalize_file(llvm_ir_path)

    cfg_graph = []
    if cfg_dot_path is not None:
        if not os.path.exists(cfg_dot_path):
            raise FileNotFoundError(f"CFG dot file not found: {cfg_dot_path}")
        cfg_builder = CFGGraphBuilder(tokenizer, llvm_ir_path, cfg_dot_path)
        cfg_graph = cfg_builder.build_cfg_edges()

    ddg_graph = []
    if ddg_dot_path is not None:
        if not os.path.exists(ddg_dot_path):
            raise FileNotFoundError(f"DDG dot file not found: {ddg_dot_path}")
        if instrumented_ir_path is None:
            raise ValueError("instrumented_ir_path is required when ddg_dot_path is provided.")
        if not os.path.exists(instrumented_ir_path):
            raise FileNotFoundError(f"Instrumented LLVM IR file not found: {instrumented_ir_path}")

        ddg_builder = DataDependencyGraphBuilder(tokenizer)
        ddg_graph = ddg_builder.generate_ddg_matrix(
            ddg_dot_path,
            instrumented_ir_path,
            llvm_ir_path,
        )

    return {
        "normalized_ir": normalized_ir,
        "cfg_graph": cfg_graph,
        "ddg_graph": ddg_graph,
    }


class ReGraphInferencePipeline:
    def __init__(self, config: Optional[InferencePipelineConfig] = None):
        self.config = config or InferencePipelineConfig()
        self._tokenizer = None
        self._collator = None
        self._model = None
        self._device = None
        self._resolved_graph_flags = None
        self._function_map_cache: Dict[str, Dict[str, str]] = {}

    def _resolve_device(self) -> torch.device:
        if self._device is None:
            requested_device = self.config.device
            if requested_device:
                self._device = torch.device(requested_device)
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _use_bf16(self) -> bool:
        return self._resolve_device().type == "cuda"

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.config.tokenizer_path)
        return self._tokenizer

    def get_graph_flags(self) -> tuple[bool, bool]:
        if self._resolved_graph_flags is None:
            self._resolved_graph_flags = resolve_graph_flags(
                use_cfg=self.config.use_cfg,
                use_ddg=self.config.use_ddg,
                model_path=self.config.model_path,
            )
        return self._resolved_graph_flags

    def get_collator(self) -> FunctionDataCollator:
        if self._collator is None:
            use_cfg, use_ddg = self.get_graph_flags()
            self._collator = FunctionDataCollator(
                self.get_tokenizer(),
                max_length=self.config.max_length,
                use_cfg=use_cfg,
                use_ddg=use_ddg,
            )
        return self._collator

    def _run_command(self, command_line: List[str], cwd: Optional[str] = None) -> None:
        result = subprocess.run(command_line, cwd=cwd, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {' '.join(command_line)}"
            )

    def _load_function_map(self, function_dir: str) -> Dict[str, str]:
        if function_dir in self._function_map_cache:
            return self._function_map_cache[function_dir]

        function_map_path = os.path.join(function_dir, "function_map.csv")
        mapping: Dict[str, str] = {}
        if os.path.exists(function_map_path):
            # function_map.csv 记录了哈希 ll 文件名到原始函数名的映射，线上接口优先用它。
            with open(function_map_path, "r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    hashed_file_name = row.get("HashedFileName")
                    original_name = row.get("OriginalFunctionName")
                    if hashed_file_name and original_name:
                        mapping[hashed_file_name.strip().strip('"')] = original_name.strip().strip('"')

        self._function_map_cache[function_dir] = mapping
        return mapping

    def _extract_function_name(
        self,
        llvm_ir_path: Optional[str],
        normalized_ir: str,
        fallback_path: str,
    ) -> str:
        function_dir = os.path.dirname(fallback_path)
        hashed_file_name = os.path.basename(fallback_path)
        mapped_name = self._load_function_map(function_dir).get(hashed_file_name)
        if mapped_name:
            return mapped_name

        # CSV 缺失时再退回到 IR 文本解析，尽量保证接口仍然可用。
        source_candidates = [normalized_ir]

        if llvm_ir_path and os.path.exists(llvm_ir_path):
            with open(llvm_ir_path, "r", encoding="utf-8", errors="ignore") as ir_file:
                source_candidates.insert(0, ir_file.read())

        for source in source_candidates:
            match = LLVM_DEFINE_NAME_PATTERN.search(source)
            if match:
                return match.group(1) or match.group(2)

        return os.path.splitext(os.path.basename(fallback_path))[0]

    def _function_to_model_sample(self, payload: FunctionGraphJson) -> Dict[str, Any]:
        return {
            "text": payload.normalized_ir,
            "cfg_graph": payload.cfg_graph,
            "ddg_graph": payload.ddg_graph,
        }

    def _prepare_inference_batch(self, payloads: Sequence[FunctionGraphJson]) -> Dict[str, Any]:
        collator_inputs = [self._function_to_model_sample(payload) for payload in payloads]
        return self.get_collator()(collator_inputs)

    def load_model(self):
        if self._model is None:
            if not self.config.model_path:
                raise ValueError("model_path is required for embedding inference.")

            use_cfg, use_ddg = self.get_graph_flags()
            model = get_model(
                self.config.model_path,
                max_seq_length=self.config.max_length,
                embedding_size=self.config.embedding_size,
                use_cfg=use_cfg,
                use_ddg=use_ddg,
                tokenizer_path=self.config.tokenizer_path,
            )
            model.to(self._resolve_device())
            model.eval()
            self._model = model

        return self._model

    def run_inference(self, input_ids, attention_mask, graph_inputs):
        model = self.load_model()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_bf16()
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_context:
                outputs = model.encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    **graph_inputs,
                )
        return outputs

    def run_attention_probe(self, input_ids, attention_mask):
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

    def _embed_batch(
        self,
        payloads: Sequence[FunctionGraphJson],
        verbose: bool = False,
    ):
        """Return embeddings list, or (embeddings, verbose_data) when verbose=True.

        verbose_data is a list of dicts with keys:
        ir, tokens, cfg_graph, cfg_model_input, ddg, ddg_model_input, attention_weights.
        其中 ddg/cfg_graph 始终返回原始图数据，便于排查图分支关闭时
        “图存在但未参与计算”的场景；*_model_input 表示实际送进模型的图输入。
        attention_weights 只返回最后一层，并在 head 维度做平均聚合，
        最终形状为 [seq, seq]，避免全层/全 head attention 把 RAM 撑爆。
        Each value is a plain Python list (or None when the graph type is absent).
        """
        if not payloads:
            return ([], []) if verbose else []

        batch = self._prepare_inference_batch(payloads)
        device = self._resolve_device()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        graph_tensor_keys = (
            "ddg_node_spans",
            "ddg_node_batch",
            "ddg_edge_index",
            "cfg_node_spans",
            "cfg_node_batch",
            "cfg_edge_index",
            "cfg_edge_attr",
        )
        graph_inputs = {
            key: batch[key].to(device) if batch[key] is not None else None
            for key in graph_tensor_keys
        }

        outputs = self.run_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_inputs=graph_inputs,
        )
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
        cpu_graph_inputs = {
            key: value.detach().cpu().tolist() if value is not None else None
            for key, value in graph_inputs.items()
        }
        tokenizer = self.get_tokenizer()

        def extract_graph_input(prefix: str, sample_idx: int) -> Optional[Dict[str, Any]]:
            node_spans = cpu_graph_inputs[f"{prefix}_node_spans"]
            node_batch = cpu_graph_inputs[f"{prefix}_node_batch"]
            edge_index = cpu_graph_inputs[f"{prefix}_edge_index"]
            edge_attr = cpu_graph_inputs.get(f"{prefix}_edge_attr")

            if node_spans is None or node_batch is None or edge_index is None:
                return None

            selected_nodes = [idx for idx, batch_idx in enumerate(node_batch) if batch_idx == sample_idx]
            if not selected_nodes:
                return None

            node_id_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}
            local_edges = []
            local_edge_attrs = [] if edge_attr is not None else None

            if len(edge_index) == 2:
                edge_sources, edge_targets = edge_index
                for edge_pos, (src_idx, dst_idx) in enumerate(zip(edge_sources, edge_targets)):
                    if src_idx not in node_id_map or dst_idx not in node_id_map:
                        continue
                    local_edges.append([node_id_map[src_idx], node_id_map[dst_idx]])
                    if local_edge_attrs is not None:
                        local_edge_attrs.append(edge_attr[edge_pos])

            return {
                "node_spans": [node_spans[node_idx] for node_idx in selected_nodes],
                "edge_index": local_edges,
                "edge_attr": local_edge_attrs,
            }

        verbose_data: List[Dict[str, Any]] = []
        for i in range(len(payloads)):
            valid_len = int(cpu_attention_mask[i].sum().item())
            token_ids = cpu_input_ids[i, :valid_len].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            verbose_data.append({
                "use_cfg": self.get_graph_flags()[0],
                "use_ddg": self.get_graph_flags()[1],
                "ir": payloads[i].normalized_ir,
                "tokens": tokens,
                "cfg_graph": payloads[i].cfg_graph,
                "cfg_model_input": extract_graph_input("cfg", i),
                "ddg": payloads[i].ddg_graph,
                "ddg_model_input": extract_graph_input("ddg", i),
                "attention_weights": last_attention[i, :valid_len, :valid_len].tolist(),
            })

        return embeddings_list, verbose_data

    def _iter_batches(self, payloads: Sequence[FunctionGraphJson]) -> Iterable[Sequence[FunctionGraphJson]]:
        batch_size = max(1, self.config.inference_batch_size)
        for start_idx in range(0, len(payloads), batch_size):
            yield payloads[start_idx:start_idx + batch_size]

    def _store_embedding(
        self,
        embeddings_by_name: Dict[str, List[float]],
        name_counts: Dict[str, int],
        payload: FunctionGraphJson,
        embedding: List[float],
    ) -> None:
        base_name = payload.function_name or os.path.splitext(os.path.basename(payload.function_path))[0]
        duplicate_count = name_counts.get(base_name, 0)
        name_counts[base_name] = duplicate_count + 1
        final_name = base_name if duplicate_count == 0 else f"{base_name}#{duplicate_count + 1}"
        embeddings_by_name[final_name] = embedding

    def run_lift_pipeline(self, executable_dir: str, lifted_output_dir: str) -> str:
        command_line = [
            self.config.python_path,
            self.config.scripts_pipeline_path,
            "pipeline",
            "--input-path",
            executable_dir,
            "--output",
            lifted_output_dir,
            "--workers",
            str(self.config.workers),
        ]
        if self.config.resume:
            command_line.append("--resume")
        if self.config.task1_start_from_step2:
            command_line.append("--task1-start-from-step2")

        self._run_command(command_line, cwd=self.config.repo_root)
        return os.path.join(lifted_output_dir, os.path.basename(executable_dir.rstrip("/")))

    def run_graph_generation(self, lifted_dataset_dir: str) -> str:
        command_line = [
            self.config.python_path,
            "-m",
            "GraphBuilder.graph_generator",
            lifted_dataset_dir,
            "--jobs",
            str(self.config.workers),
        ]
        if self.config.cleanup_graph_temp_files:
            command_line.append("--cleanup")
        else:
            command_line.append("--no-cleanup")
        if self.config.use_graph_cache:
            command_line.append("--cache")
        else:
            command_line.append("--no-cache")

        self._run_command(command_line, cwd=self.config.repo_root)
        return os.path.join(lifted_dataset_dir, "results.db")

    def iter_graph_records(self, graph_db_path: str) -> Iterable[FunctionGraphRecord]:
        if not os.path.exists(graph_db_path):
            raise FileNotFoundError(f"Graph database not found: {graph_db_path}")

        with sqlite3.connect(graph_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT input_path, purify_path, instrumented_path, cfg_dot, ddg_dot
                FROM results
                ORDER BY input_path
                """
            )
            for row in cursor.fetchall():
                yield FunctionGraphRecord(*row)

    def build_function_json(self, record: FunctionGraphRecord) -> FunctionGraphJson:
        graph_payload = build_edges_from_dot_files(
            llvm_ir_path=record.purify_path,
            cfg_dot_path=record.cfg_dot,
            ddg_dot_path=record.ddg_dot,
            instrumented_ir_path=record.instrumented_path,
            tokenizer_path=self.config.tokenizer_path,
        )
        return FunctionGraphJson(
            function_name=self._extract_function_name(
                llvm_ir_path=record.purify_path,
                normalized_ir=graph_payload["normalized_ir"],
                fallback_path=record.input_path,
            ),
            function_path=record.input_path,
            normalized_ir=graph_payload["normalized_ir"],
            cfg_graph=graph_payload["cfg_graph"],
            ddg_graph=graph_payload["ddg_graph"],
        )

    def collect_function_graphs(self, graph_db_path: str) -> List[FunctionGraphJson]:
        payload = []
        for record in self.iter_graph_records(graph_db_path):
            try:
                payload.append(self.build_function_json(record))
            except Exception as exc:
                typer.echo(f"Skipping {record.input_path}: {exc}", err=True)
        return payload

    def export_graph_json(self, graph_db_path: str, output_json_path: str) -> List[FunctionGraphJson]:
        payload = self.collect_function_graphs(graph_db_path)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in payload], f, ensure_ascii=True)
        return payload

    def embed_function_graphs(
        self,
        payloads: Sequence[FunctionGraphJson],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Embed all payloads.

        Returns Dict[name, List[float]] when verbose=False, or
        Dict[name, {"embedding": List[float], "use_cfg": ..., "use_ddg": ..., "ir": ..., "tokens": ...,
        "cfg_graph": ..., "cfg_model_input": ..., "ddg": ..., "ddg_model_input": ...,
        "attention_weights": ...}] when verbose=True.
        """
        embeddings_by_name: Dict[str, Any] = {}
        name_counts: Dict[str, int] = {}

        for batch in self._iter_batches(payloads):
            try:
                if verbose:
                    batch_embeddings, batch_verbose = self._embed_batch(batch, verbose=True)
                    for payload, embedding, vdata in zip(batch, batch_embeddings, batch_verbose):
                        self._store_embedding(
                            embeddings_by_name, name_counts, payload,
                            {"embedding": embedding, **vdata},
                        )
                else:
                    batch_embeddings = self._embed_batch(batch)
                    for payload, embedding in zip(batch, batch_embeddings):
                        self._store_embedding(embeddings_by_name, name_counts, payload, embedding)
            except Exception as batch_exc:
                if len(batch) == 1:
                    typer.echo(
                        f"Skipping {batch[0].function_path}: {batch_exc}",
                        err=True,
                    )
                    continue

                for payload in batch:
                    try:
                        if verbose:
                            emb_list, vdata_list = self._embed_batch([payload], verbose=True)
                            self._store_embedding(
                                embeddings_by_name, name_counts, payload,
                                {"embedding": emb_list[0], **vdata_list[0]},
                            )
                        else:
                            embedding = self._embed_batch([payload])[0]
                            self._store_embedding(embeddings_by_name, name_counts, payload, embedding)
                    except Exception as item_exc:
                        typer.echo(f"Skipping {payload.function_path}: {item_exc}", err=True)

        return embeddings_by_name

    def embed_graph_db(self, graph_db_path: str, verbose: bool = False) -> Dict[str, Any]:
        payloads = self.collect_function_graphs(graph_db_path)
        return self.embed_function_graphs(payloads, verbose=verbose)

    def embed_executable_dir(
        self,
        executable_dir: str,
        lifted_output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if lifted_output_dir is None:
            lifted_output_dir = executable_dir

        lifted_dataset_dir = self.run_lift_pipeline(executable_dir, lifted_output_dir)
        graph_db_path = self.run_graph_generation(lifted_dataset_dir)
        return self.embed_graph_db(graph_db_path, verbose=verbose)

    def export_embeddings_json(
        self,
        executable_dir: str,
        output_json_path: str,
        lifted_output_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        payload = self.embed_executable_dir(executable_dir, lifted_output_dir=lifted_output_dir)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
        return payload

    def run(self, executable_dir: str, lifted_output_dir: str, output_json_path: str) -> List[FunctionGraphJson]:
        lifted_dataset_dir = self.run_lift_pipeline(executable_dir, lifted_output_dir)
        graph_db_path = self.run_graph_generation(lifted_dataset_dir)
        return self.export_graph_json(graph_db_path, output_json_path)


@app.command()
def main(
    executable_dir: str = typer.Argument(..., help="Directory containing executables."),
    output_json_path: str = typer.Argument(..., help="Path to the exported JSON file."),
    lifted_output_dir: Optional[str] = typer.Option(None, help="Directory for lifted IR outputs."),
    tokenizer_path: str = typer.Option(DEFAULT_TOKENIZER_PATH, help="Tokenizer path."),
    workers: int = typer.Option(os.cpu_count() or 1, help="Worker count for lifting and graph generation."),
    python_path: str = typer.Option(sys.executable, help="Python executable used for subprocess stages."),
    repo_root: str = typer.Option(os.getcwd(), help="Repository root."),
    resume: bool = typer.Option(True, help="Resume lifting pipeline when possible."),
    task1_start_from_step2: bool = typer.Option(False, help="Skip IDA database generation and start task1 from step2."),
    cleanup_graph_temp_files: bool = typer.Option(True, help="Clean graph generation temp files."),
    use_graph_cache: bool = typer.Option(False, help="Reuse graph file cache if present."),
):
    if lifted_output_dir is None:
        lifted_output_dir = executable_dir

    pipeline = ReGraphInferencePipeline(
        InferencePipelineConfig(
            repo_root=repo_root,
            tokenizer_path=tokenizer_path,
            python_path=python_path,
            workers=workers,
            cleanup_graph_temp_files=cleanup_graph_temp_files,
            use_graph_cache=use_graph_cache,
            resume=resume,
            task1_start_from_step2=task1_start_from_step2,
        )
    )
    payload = pipeline.run(executable_dir, lifted_output_dir, output_json_path)
    print(f"Exported {len(payload)} function graph records to {output_json_path}")


if __name__ == "__main__":
    app()
