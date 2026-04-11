import os
import shutil
import sys
import tempfile
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from Utils.utils import DEFAULT_TOKENIZER_PATH
from inference import (
    GRAPH_MODES,
    InferencePipelineConfig,
    ReGraphInferencePipeline,
    resolve_graph_mode,
)


def _build_pipeline_config(model_path: str, graph_mode: Optional[str]) -> InferencePipelineConfig:
    normalized_model_path = os.path.abspath(model_path)
    if not model_path:
        raise RuntimeError("REGRAPH_MODEL_PATH is required before starting the API.")

    return InferencePipelineConfig(
        repo_root=os.getcwd(),
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        python_path=sys.executable,
        model_path=normalized_model_path,
        graph_mode=graph_mode,
    )


@lru_cache(maxsize=8)
def get_pipeline(model_path: str, graph_mode: str) -> ReGraphInferencePipeline:
    return ReGraphInferencePipeline(_build_pipeline_config(model_path, graph_mode))


def _resolve_request_graph_mode(
    model_path: str,
    graph_mode: Optional[str],
    mode: Optional[str],
) -> str:
    if graph_mode and mode and graph_mode.strip().lower() != mode.strip().lower():
        raise HTTPException(
            status_code=400,
            detail="mode and graph_mode must match when both are provided.",
        )

    requested_graph_mode = graph_mode or mode or os.environ.get("REGRAPH_GRAPH_MODE")
    try:
        return resolve_graph_mode(requested_graph_mode, model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_health_payload() -> Dict[str, object]:
    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    configured_graph_mode = os.environ.get("REGRAPH_GRAPH_MODE")
    resolved_graph_mode = None
    resolution_error = None

    if model_path:
        try:
            resolved_graph_mode = resolve_graph_mode(configured_graph_mode, model_path)
        except ValueError as exc:
            resolution_error = str(exc)

    return {
        "status": "ok",
        "model_configured": bool(model_path),
        "model_path": model_path,
        "graph_mode_env": configured_graph_mode,
        "resolved_graph_mode": resolved_graph_mode,
        "supported_graph_modes": list(GRAPH_MODES),
        "graph_mode_resolution_error": resolution_error,
    }


app = FastAPI(title="ReGraph Inference API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, object]:
    return _build_health_payload()


@app.post("/embed")
async def embed_binary(
    binary: UploadFile = File(...),
    verbose: bool = Form(False),
    graph_mode: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Embed all functions in a binary.

    When verbose=True each function value is a dict with keys:
    embedding, graph_mode, ir, tokens, cfg_graph, cfg_u, cfg_v,
    ddg, ddg_model_input, attention_weights.
    Otherwise it is a plain list of floats.
    """
    if not binary.filename:
        raise HTTPException(status_code=400, detail="binary filename is required")

    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    if not model_path:
        raise HTTPException(status_code=500, detail="REGRAPH_MODEL_PATH is required before starting the API.")

    try:
        resolved_graph_mode = _resolve_request_graph_mode(model_path, graph_mode, mode)
        pipeline = get_pipeline(model_path, resolved_graph_mode)
    except HTTPException:
        raise

    safe_name = os.path.basename(binary.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="invalid binary filename")

    try:
        with tempfile.TemporaryDirectory(prefix="regraph_api_") as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            lifted_dir = os.path.join(temp_dir, "lifted")
            os.makedirs(input_dir, exist_ok=True)

            input_path = os.path.join(input_dir, safe_name)
            with open(input_path, "wb") as output_file:
                shutil.copyfileobj(binary.file, output_file)

            return pipeline.embed_executable_dir(input_dir, lifted_output_dir=lifted_dir, verbose=verbose)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to embed binary: {exc}") from exc
    finally:
        await binary.close()
        

if __name__ == "__main__":
    import uvicorn
    os.environ["REGRAPH_MODEL_PATH"] = "db1_model_ablation30000_no_ddg"
    uvicorn.run(app, host="0.0.0.0", port=8000)
