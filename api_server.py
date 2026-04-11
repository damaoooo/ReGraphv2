import os
import shutil
import sys
import tempfile
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from Utils.utils import DEFAULT_TOKENIZER_PATH
from inference import (
    InferencePipelineConfig,
    ReGraphInferencePipeline,
    resolve_graph_flags,
)


def _build_pipeline_config(model_path: str, use_cfg: Optional[bool], use_ddg: Optional[bool]) -> InferencePipelineConfig:
    normalized_model_path = os.path.abspath(model_path)
    if not model_path:
        raise RuntimeError("REGRAPH_MODEL_PATH is required before starting the API.")

    return InferencePipelineConfig(
        repo_root=os.getcwd(),
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        python_path=sys.executable,
        model_path=normalized_model_path,
        use_cfg=use_cfg,
        use_ddg=use_ddg,
    )


@lru_cache(maxsize=8)
def get_pipeline(model_path: str, use_cfg: bool, use_ddg: bool) -> ReGraphInferencePipeline:
    return ReGraphInferencePipeline(_build_pipeline_config(model_path, use_cfg, use_ddg))


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"invalid boolean value: {value}")


def _resolve_request_graph_flags(
    model_path: str,
    use_cfg: Optional[str],
    use_ddg: Optional[str],
) -> tuple[bool, bool]:
    try:
        requested_use_cfg = _parse_optional_bool(use_cfg) if use_cfg is not None else _parse_optional_bool(os.environ.get("REGRAPH_USE_CFG"))
        requested_use_ddg = _parse_optional_bool(use_ddg) if use_ddg is not None else _parse_optional_bool(os.environ.get("REGRAPH_USE_DDG"))
        return resolve_graph_flags(requested_use_cfg, requested_use_ddg, model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_health_payload() -> Dict[str, object]:
    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    configured_use_cfg = os.environ.get("REGRAPH_USE_CFG")
    configured_use_ddg = os.environ.get("REGRAPH_USE_DDG")
    resolved_use_cfg = None
    resolved_use_ddg = None
    resolution_error = None

    if model_path:
        try:
            resolved_use_cfg, resolved_use_ddg = resolve_graph_flags(
                _parse_optional_bool(configured_use_cfg) if configured_use_cfg is not None else None,
                _parse_optional_bool(configured_use_ddg) if configured_use_ddg is not None else None,
                model_path,
            )
        except ValueError as exc:
            resolution_error = str(exc)

    return {
        "status": "ok",
        "model_configured": bool(model_path),
        "model_path": model_path,
        "use_cfg_env": configured_use_cfg,
        "use_ddg_env": configured_use_ddg,
        "resolved_use_cfg": resolved_use_cfg,
        "resolved_use_ddg": resolved_use_ddg,
        "graph_flag_resolution_error": resolution_error,
    }


app = FastAPI(title="ReGraph Inference API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, object]:
    return _build_health_payload()


@app.post("/embed")
async def embed_binary(
    binary: UploadFile = File(...),
    verbose: bool = Form(False),
    use_cfg: Optional[str] = Form(None),
    use_ddg: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Embed all functions in a binary.

    When verbose=True each function value is a dict with keys:
    embedding, use_cfg, use_ddg, ir, tokens, cfg_graph, cfg_model_input,
    ddg, ddg_model_input, attention_weights.
    Otherwise it is a plain list of floats.
    """
    if not binary.filename:
        raise HTTPException(status_code=400, detail="binary filename is required")

    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    if not model_path:
        raise HTTPException(status_code=500, detail="REGRAPH_MODEL_PATH is required before starting the API.")

    try:
        resolved_use_cfg, resolved_use_ddg = _resolve_request_graph_flags(model_path, use_cfg, use_ddg)
        pipeline = get_pipeline(model_path, resolved_use_cfg, resolved_use_ddg)
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
    os.environ["REGRAPH_MODEL_PATH"] = "db1_model"
    uvicorn.run(app, host="0.0.0.0", port=8000)
