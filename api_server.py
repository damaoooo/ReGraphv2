import os
import shutil
import sys
import tempfile
from functools import lru_cache
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from Utils.utils import DEFAULT_TOKENIZER_PATH
from inference import InferencePipelineConfig, ReGraphInferencePipeline


def _build_pipeline_config(model_path: str) -> InferencePipelineConfig:
    normalized_model_path = os.path.abspath(model_path)
    if not model_path:
        raise RuntimeError("REGRAPH_MODEL_PATH is required before starting the API.")

    return InferencePipelineConfig(
        repo_root=os.getcwd(),
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        python_path=sys.executable,
        model_path=normalized_model_path,
    )


@lru_cache(maxsize=8)
def get_pipeline(model_path: str) -> ReGraphInferencePipeline:
    return ReGraphInferencePipeline(_build_pipeline_config(model_path))


def _build_health_payload() -> Dict[str, object]:
    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    return {
        "status": "ok",
        "model_configured": bool(model_path),
        "model_path": model_path,
        "tokenizer_path": DEFAULT_TOKENIZER_PATH,
        "pipeline_mode": "binary_to_asm_to_text_embedding",
    }


app = FastAPI(title="ReGraph ASM Inference API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, object]:
    return _build_health_payload()


@app.post("/embed")
async def embed_binary(
    binary: UploadFile = File(...),
    verbose: bool = Form(False),
) -> Dict[str, Any]:
    """Embed all functions in one uploaded binary via `bin2asm -> text encoder`."""
    if not binary.filename:
        raise HTTPException(status_code=400, detail="binary filename is required")

    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    if not model_path:
        raise HTTPException(status_code=500, detail="REGRAPH_MODEL_PATH is required before starting the API.")

    safe_name = os.path.basename(binary.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="invalid binary filename")

    try:
        pipeline = get_pipeline(model_path)
        with tempfile.TemporaryDirectory(prefix="regraph_api_") as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            asm_dir = os.path.join(temp_dir, "asm")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(asm_dir, exist_ok=True)

            input_path = os.path.join(input_dir, safe_name)
            with open(input_path, "wb") as output_file:
                shutil.copyfileobj(binary.file, output_file)

            return pipeline.embed_executable_dir(
                input_dir,
                asm_output_dir=asm_dir,
                verbose=verbose,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to embed binary: {exc}") from exc
    finally:
        await binary.close()


if __name__ == "__main__":
    import uvicorn

    os.environ["REGRAPH_MODEL_PATH"] = "db1_model_asm"
    uvicorn.run(app, host="0.0.0.0", port=8000)
