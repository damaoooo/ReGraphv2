import os
import shutil
import sys
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from Utils.utils import DEFAULT_TOKENIZER_PATH
from inference import InferencePipelineConfig, ReGraphInferencePipeline


def _build_pipeline_config_from_env() -> InferencePipelineConfig:
    model_path = os.environ.get("REGRAPH_MODEL_PATH")
    if not model_path:
        raise RuntimeError("REGRAPH_MODEL_PATH is required before starting the API.")

    return InferencePipelineConfig(
        repo_root=os.getcwd(),
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        python_path=sys.executable,
        model_path=model_path,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> ReGraphInferencePipeline:
    return ReGraphInferencePipeline(_build_pipeline_config_from_env())


app = FastAPI(title="ReGraph Inference API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "model_configured": bool(os.environ.get("REGRAPH_MODEL_PATH")),
    }


@app.post("/embed")
async def embed_binary(
    binary: UploadFile = File(...),
    verbose: bool = Form(False),
) -> Dict[str, Any]:
    """Embed all functions in a binary.

    When verbose=True each function value is a dict with keys:
    embedding, ir, tokens, cfg_u, cfg_v, ddg, attention_weights.
    Otherwise it is a plain list of floats.
    """
    if not binary.filename:
        raise HTTPException(status_code=400, detail="binary filename is required")

    try:
        pipeline = get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
    os.environ["REGRAPH_MODEL_PATH"] = "/home/damaoooo/Downloads/regraphv2/db1_model_ablation30k_both"
    uvicorn.run(app, host="0.0.0.0", port=8000)
