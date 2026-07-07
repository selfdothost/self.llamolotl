"""
Pipeline and LoRA management endpoints.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import aiofiles
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..state import (
    BAKE_SCRIPT,
    CONVERT_HF_TO_GGUF,
    CONVERT_LORA_TO_GGUF,
    LLAMA_QUANTIZE,
    LOGS_DIR,
    MERGE_SCRIPT,
    MODELS_DIR,
    OUTPUTS_DIR,
    PYTHON,
    BakeRequest,
    ConvertLoraToGgufRequest,
    ConvertToGgufRequest,
    HfModelPullRequest,
    MergeLoraRequest,
    PipelineTask,
    PipelineTaskStatus,
    PipelineTaskType,
    QuantizeRequest,
    _active_downloads,
    _load_models_meta,
    _pipeline_processes,
    _pipeline_tasks,
    _record_lora_meta,
    _record_model_meta,
    _save_pipeline_tasks,
    _start_pipeline_task,
    _validate_path,
)

log = logging.getLogger(__name__)

router = APIRouter()


# ─── Pipeline Endpoints ───────────────────────────────────────────────

@router.post("/api/pipeline/pull-hf-model")
async def pull_hf_model(req: HfModelPullRequest):
    """Download a full HuggingFace model (safetensors, config, tokenizer) for
    use in the convert-to-gguf pipeline.

    Streams progress as NDJSON. The model is saved to OUTPUTS_DIR so it can
    be passed directly to /api/pipeline/convert-to-gguf.
    """
    from huggingface_hub import HfApi, snapshot_download

    repo_id = req.repo_id.strip()
    output_name = req.output_name or repo_id.split("/")[-1]
    dest_dir = OUTPUTS_DIR / output_name

    async def generate():
        try:
            api = HfApi()

            # Validate repo exists
            try:
                repo_info = api.model_info(repo_id, files_metadata=True)
            except Exception as e:
                yield json.dumps({"error": f"Failed to access repo '{repo_id}': {e}"}) + "\n"
                return

            if dest_dir.exists():
                # Check if it already has model files
                if any(dest_dir.glob("*.safetensors")) and (dest_dir / "config.json").exists():
                    yield json.dumps({"error": f"Model already exists at '{output_name}'. Delete it first or use a different output_name."}) + "\n"
                    return

            # Calculate total download size — must match the allow/ignore
            # patterns used by snapshot_download below so the progress bar
            # is accurate (previously we counted .bin files that were never
            # downloaded, doubling the reported total).
            _allow_exts = (".safetensors", ".json", ".txt", ".model", ".py")
            _ignore_exts = (".gguf", ".bin", ".msgpack", ".h5", ".ot", ".md")
            _ignore_names = {".gitattributes"}

            total_size = 0
            download_files = []
            for s in repo_info.siblings:
                fname = s.rfilename
                if fname in _ignore_names:
                    continue
                if any(fname.endswith(ext) for ext in _ignore_exts):
                    continue
                if not any(fname.endswith(ext) for ext in _allow_exts):
                    continue
                size = getattr(s, 'size', None) or 0
                if size == 0 and hasattr(s, 'lfs') and s.lfs:
                    size = s.lfs.get('size', 0) if isinstance(s.lfs, dict) else getattr(s.lfs, 'size', 0)
                total_size += size
                download_files.append(fname)

            # Check that repo has safetensors
            has_safetensors = any(f.endswith(".safetensors") for f in download_files)
            has_config = "config.json" in download_files
            if not has_safetensors:
                yield json.dumps({"error": f"No .safetensors files found in '{repo_id}'. This endpoint is for HF model repos, not GGUF repos."}) + "\n"
                return
            if not has_config:
                yield json.dumps({"error": f"No config.json found in '{repo_id}'. Not a valid model repo."}) + "\n"
                return

            yield json.dumps({
                "status": "downloading",
                "repo_id": repo_id,
                "total": total_size,
                "completed": 0,
                "dest": output_name,
            }) + "\n"

            # Download in background thread
            cancel_event = threading.Event()
            download_key = f"hf-model:{repo_id}"
            _active_downloads[download_key] = cancel_event

            download_error = []
            download_done = threading.Event()
            result_path = [None]

            def download_task():
                try:
                    # Only download safetensors (skip .bin duplicates) + essential config/tokenizer files
                    allow_patterns = [
                        "*.safetensors",
                        "*.json",
                        "*.txt",          # e.g. merges.txt for tokenizer
                        "*.model",        # sentencepiece .model
                        "*.py",           # modeling code if needed
                    ]
                    ignore_patterns = [
                        "*.gguf",
                        "*.bin",          # skip pytorch .bin if safetensors exist
                        "*.msgpack",
                        "*.h5",
                        "*.ot",
                        "*.md",
                        ".gitattributes",
                    ]

                    path = snapshot_download(
                        repo_id=repo_id,
                        revision=req.revision,
                        local_dir=str(dest_dir),
                        local_dir_use_symlinks=False,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                    )
                    result_path[0] = path
                    download_done.set()
                except Exception as e:
                    download_error.append(str(e))
                    download_done.set()

            thread = threading.Thread(target=download_task, daemon=True)
            thread.start()

            # Poll progress by measuring downloaded bytes
            while not download_done.is_set():
                if cancel_event.is_set():
                    yield json.dumps({"error": "Download cancelled"}) + "\n"
                    _active_downloads.pop(download_key, None)
                    # Clean up partial download
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir, ignore_errors=True)
                    return

                current_size = 0
                if dest_dir.exists():
                    for f in dest_dir.rglob("*"):
                        if f.is_file():
                            try:
                                current_size += f.stat().st_size
                            except OSError:
                                pass

                yield json.dumps({
                    "status": "downloading",
                    "repo_id": repo_id,
                    "completed": current_size,
                    "total": total_size,
                }) + "\n"

                await asyncio.sleep(2)

            _active_downloads.pop(download_key, None)

            if download_error:
                yield json.dumps({"error": download_error[0]}) + "\n"
                return

            # Verify download
            if not dest_dir.exists() or not any(dest_dir.glob("*.safetensors")):
                yield json.dumps({"error": "Download completed but no safetensors files found"}) + "\n"
                return

            final_size = sum(
                f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()
            )

            yield json.dumps({
                "status": "downloading",
                "repo_id": repo_id,
                "completed": final_size,
                "total": final_size,
            }) + "\n"

            yield json.dumps({
                "status": "success",
                "dest": output_name,
                "path": str(dest_dir),
                "size": final_size,
                "next_step": f"POST /api/pipeline/convert-to-gguf with model_path=\"{output_name}\"",
            }) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/api/pipeline/merge-lora", status_code=201)
def merge_lora(req: MergeLoraRequest) -> PipelineTask:
    """Merge a LoRA/QLoRA adapter into the base model.

    Uses PEFT to load the adapter and merge it into the base model.
    Output is saved to <output_dir>/merged/ as a full HF-format model.
    Base model is read from adapter_config.json in the output directory.
    """
    output_dir = _validate_path(req.model_output, OUTPUTS_DIR)
    if not output_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training output '{req.model_output}' not found in {OUTPUTS_DIR}",
        )

    # Check for adapter weights (indicates LoRA was used)
    has_adapter = (output_dir / "adapter_model.safetensors").exists() or (
        output_dir / "adapter_model.bin"
    ).exists()
    if not has_adapter:
        raise HTTPException(
            status_code=400,
            detail="No adapter weights found — this may be a full fine-tune (no merge needed)",
        )

    # Verify adapter_config.json exists (needed by merge_lora.py to find base model)
    if not (output_dir / "adapter_config.json").exists():
        raise HTTPException(
            status_code=400,
            detail="adapter_config.json not found — cannot determine base model for merge",
        )

    merged_path = output_dir / "merged"

    # Check if already merged
    if merged_path.exists() and (merged_path / "model.safetensors").exists():
        raise HTTPException(
            status_code=409,
            detail=f"Merged model already exists at {merged_path}",
        )

    cmd = [
        str(PYTHON), str(MERGE_SCRIPT),
        "--adapter_path", str(output_dir),
        "--output_dir", str(output_dir),
    ]

    return _start_pipeline_task(
        task_type=PipelineTaskType.MERGE_LORA,
        cmd=cmd,
        input_path=str(output_dir),
        output_path=str(merged_path),
    )


@router.post("/api/pipeline/convert-to-gguf", status_code=201)
def convert_to_gguf(req: ConvertToGgufRequest) -> PipelineTask:
    """Convert a HuggingFace-format model (safetensors) to GGUF format.

    The resulting GGUF file is placed in MODELS_DIR ready for llama-server.
    """
    # Resolve model path — validate under OUTPUTS_DIR or WORKSPACE
    model_path = Path(req.model_path)
    if model_path.is_absolute():
        # Absolute paths must be under OUTPUTS_DIR or WORKSPACE parent
        resolved = model_path.resolve()
        if not (str(resolved).startswith(str(OUTPUTS_DIR.resolve())) or
                str(resolved).startswith(str(OUTPUTS_DIR.parent.resolve()))):
            raise HTTPException(status_code=400, detail="Absolute path must be under workspace")
    else:
        model_path = _validate_path(req.model_path, OUTPUTS_DIR)

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model path not found: {model_path}",
        )

    # Validate it looks like a HF model directory
    has_safetensors = any(model_path.glob("*.safetensors"))
    has_config = (model_path / "config.json").exists()
    if not has_safetensors:
        raise HTTPException(
            status_code=400,
            detail="No .safetensors files found in model directory",
        )
    if not has_config:
        raise HTTPException(
            status_code=400,
            detail="No config.json found in model directory",
        )

    # Validate outtype
    valid_outtypes = ["f32", "f16", "bf16", "q8_0", "auto"]
    if req.outtype not in valid_outtypes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outtype '{req.outtype}'. Must be one of: {valid_outtypes}",
        )

    # Determine output filename
    if req.model_name:
        out_name = f"{req.model_name}-{req.outtype}.gguf"
    else:
        out_name = f"{model_path.name}-{req.outtype}.gguf"
    outfile = MODELS_DIR / out_name

    if outfile.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Output file already exists: {out_name}",
        )

    cmd = [
        str(PYTHON), str(CONVERT_HF_TO_GGUF),
        str(model_path),
        "--outfile", str(outfile),
        "--outtype", req.outtype,
    ]

    # Set NO_LOCAL_GGUF so convert script uses the installed gguf package
    env = {"NO_LOCAL_GGUF": "1"}

    return _start_pipeline_task(
        task_type=PipelineTaskType.CONVERT_TO_GGUF,
        cmd=cmd,
        input_path=str(model_path),
        output_path=str(outfile),
        env=env,
    )


@router.post("/api/pipeline/quantize", status_code=201)
def quantize_model(req: QuantizeRequest) -> PipelineTask:
    """Quantize a GGUF model to a smaller format.

    Takes an existing GGUF file in MODELS_DIR and produces a quantized version.
    """
    model_path = _validate_path(req.model_file, MODELS_DIR)
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GGUF model not found: {req.model_file}",
        )

    if not req.model_file.endswith(".gguf"):
        raise HTTPException(
            status_code=400,
            detail="Input file must be a .gguf file",
        )

    # Validate quantization type
    valid_quants = [
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
        "F16", "F32", "BF16",
    ]
    quant_upper = req.quant_type.upper()
    if quant_upper not in valid_quants:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quant_type '{req.quant_type}'. Common types: Q4_K_M, Q5_K_M, Q6_K, Q8_0",
        )

    # Build output filename: replace type suffix or append
    stem = req.model_file.rsplit(".gguf", 1)[0]
    out_name = f"{stem}-{quant_upper}.gguf"
    outfile = MODELS_DIR / out_name

    if outfile.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Output file already exists: {out_name}",
        )

    cmd = [
        str(LLAMA_QUANTIZE),
        str(model_path),
        str(outfile),
        quant_upper,
    ]

    return _start_pipeline_task(
        task_type=PipelineTaskType.QUANTIZE,
        cmd=cmd,
        input_path=str(model_path),
        output_path=str(outfile),
    )


@router.post("/api/pipeline/convert-lora-to-gguf", status_code=201)
def convert_lora_to_gguf(req: ConvertLoraToGgufRequest) -> PipelineTask:
    """Convert a LoRA adapter to GGUF format for dynamic loading with llama-server.

    The resulting LoRA GGUF can be applied at inference time via /api/system/apply-loras
    without merging into the base model.
    """
    output_dir = _validate_path(req.model_output, OUTPUTS_DIR)
    if not output_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training output '{req.model_output}' not found",
        )

    # Verify adapter files exist
    has_adapter = (output_dir / "adapter_model.safetensors").exists() or (
        output_dir / "adapter_model.bin"
    ).exists()
    if not has_adapter:
        raise HTTPException(
            status_code=400,
            detail="No adapter weights found in output directory",
        )

    valid_outtypes = ["f32", "f16", "bf16", "q8_0", "auto"]
    if req.outtype not in valid_outtypes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outtype '{req.outtype}'. Must be one of: {valid_outtypes}",
        )

    out_name = req.output_name or f"{req.model_output}-lora-{req.outtype}.gguf"
    if not out_name.endswith(".gguf"):
        out_name += ".gguf"
    outfile = MODELS_DIR / out_name

    if outfile.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Output file already exists: {out_name}",
        )

    cmd = [
        str(PYTHON), str(CONVERT_LORA_TO_GGUF),
        str(output_dir),
        "--outfile", str(outfile),
        "--outtype", req.outtype,
    ]

    if req.base_model:
        cmd.extend(["--base-model-id", req.base_model])

    env = {"NO_LOCAL_GGUF": "1"}

    # Determine base model from adapter_config.json for metadata
    base_model_id = req.base_model
    if not base_model_id:
        adapter_cfg_path = output_dir / "adapter_config.json"
        if adapter_cfg_path.exists():
            try:
                adapter_cfg = json.loads(adapter_cfg_path.read_text())
                base_model_id = adapter_cfg.get("base_model_name_or_path")
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to read adapter_config.json for LoRA meta: %s", e)

    # Record in models_meta.json so the LoRA is discoverable
    _record_lora_meta(out_name, base_model_id, req.model_output)

    return _start_pipeline_task(
        task_type=PipelineTaskType.CONVERT_LORA_TO_GGUF,
        cmd=cmd,
        input_path=str(output_dir),
        output_path=str(outfile),
        env=env,
    )


@router.get("/api/loras/available")
def list_available_loras():
    """List LoRA GGUF files available for dynamic loading.

    Returns LoRAs from models_meta.json (source_type == lora_gguf) that
    still exist on disk, enriched with base_model info.
    """
    if not MODELS_DIR.exists():
        return []

    meta = _load_models_meta()
    loras = []

    # Gather known LoRA GGUFs from metadata
    known_lora_files = set()
    for filename, entry in meta.items():
        if entry.get("source_type") != "lora_gguf":
            continue
        fpath = MODELS_DIR / filename
        if not fpath.exists():
            continue
        known_lora_files.add(filename)
        loras.append({
            "file": filename,
            "base_model": entry.get("base_model"),
            "training_output": entry.get("training_output"),
            "size": fpath.stat().st_size,
            "modified": fpath.stat().st_mtime,
        })

    # Also scan for *-lora-*.gguf files not yet in metadata
    # (e.g. converted before this code was added)
    for f in MODELS_DIR.glob("*-lora-*.gguf"):
        if f.name in known_lora_files:
            continue
        # Try to find base_model from training output adapter_config
        base_model = None
        # Infer training output name from filename pattern: <output>-lora-<outtype>.gguf
        stem = f.stem  # e.g. "qlora-out-lora-f16"
        parts = stem.rsplit("-lora-", 1)
        if len(parts) == 2:
            training_output_name = parts[0]
            adapter_cfg = OUTPUTS_DIR / training_output_name / "adapter_config.json"
            if adapter_cfg.exists():
                try:
                    cfg = json.loads(adapter_cfg.read_text())
                    base_model = cfg.get("base_model_name_or_path")
                except (json.JSONDecodeError, OSError) as e:
                    log.debug("Failed to read adapter_config for %s: %s", f.name, e)
            # Record it for future lookups
            _record_lora_meta(f.name, base_model, training_output_name)

        loras.append({
            "file": f.name,
            "base_model": base_model,
            "training_output": parts[0] if len(parts) == 2 else None,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })

    # Also scan training outputs for unconverted adapters and auto-convert them
    known_training_outputs = {l.get("training_output") for l in loras if l.get("training_output")}
    if OUTPUTS_DIR.exists():
        for output_dir in OUTPUTS_DIR.rglob("adapter_model.safetensors"):
            parent = output_dir.parent
            try:
                rel = str(parent.relative_to(OUTPUTS_DIR))
            except ValueError:
                continue
            if rel in known_training_outputs:
                continue
            # This adapter hasn't been converted yet — read its base_model
            base_model = None
            adapter_cfg = parent / "adapter_config.json"
            if adapter_cfg.exists():
                try:
                    cfg = json.loads(adapter_cfg.read_text())
                    base_model = cfg.get("base_model_name_or_path")
                except (json.JSONDecodeError, OSError) as e:
                    log.debug("Failed to read adapter_config for %s: %s", rel, e)
            out_name = rel.replace("/", "-").replace("\\", "-") + "-lora-f16.gguf"
            outfile = MODELS_DIR / out_name
            if not outfile.exists():
                # Trigger background conversion
                cmd = [
                    str(PYTHON), str(CONVERT_LORA_TO_GGUF),
                    str(parent),
                    "--outfile", str(outfile),
                    "--outtype", "f16",
                ]
                if base_model:
                    cmd.extend(["--base-model-id", base_model])
                _record_lora_meta(out_name, base_model, rel)
                log.info("Auto-converting unconverted LoRA: %s", out_name)
                _start_pipeline_task(
                    task_type=PipelineTaskType.CONVERT_LORA_TO_GGUF,
                    cmd=cmd,
                    input_path=str(parent),
                    output_path=str(outfile),
                    env={"NO_LOCAL_GGUF": "1"},
                )
            # Include it in the response (converting or ready)
            loras.append({
                "file": out_name,
                "base_model": base_model,
                "training_output": rel,
                "size": 0 if not outfile.exists() else outfile.stat().st_size,
                "modified": parent.stat().st_mtime,
                "converting": not outfile.exists(),
            })

    return sorted(loras, key=lambda l: l["modified"], reverse=True)


@router.post("/api/pipeline/bake", status_code=201)
def bake_model(req: BakeRequest) -> PipelineTask:
    """Bake multiple LoRA adapters into a single GGUF model.

    This is the full pipeline: merge LoRAs with weights -> convert to GGUF -> quantize.
    Use this when users are happy with their tested LoRA combinations and want to
    produce a final production model.
    """
    if not req.adapters:
        raise HTTPException(status_code=400, detail="At least one adapter is required")

    # Validate all adapter paths exist
    for adapter in req.adapters:
        adapter_path = Path(adapter.path)
        if not adapter_path.is_absolute():
            adapter_path = OUTPUTS_DIR / adapter.path
        if not adapter_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Adapter not found: {adapter.path}",
            )
        has_adapter = (adapter_path / "adapter_model.safetensors").exists() or (
            adapter_path / "adapter_model.bin"
        ).exists()
        if not has_adapter:
            raise HTTPException(
                status_code=400,
                detail=f"No adapter weights in: {adapter.path}",
            )

    # Check output doesn't already exist
    if req.quant_type:
        final_name = f"{req.output_name}-{req.quant_type}.gguf"
    else:
        final_name = f"{req.output_name}-{req.outtype}.gguf"
    final_path = MODELS_DIR / final_name

    if final_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Output file already exists: {final_name}",
        )

    # Build bake config JSON
    bake_config = {
        "base_model": req.base_model,
        "adapters": [
            {
                "path": str(OUTPUTS_DIR / a.path) if not Path(a.path).is_absolute() else a.path,
                "weight": a.weight,
            }
            for a in req.adapters
        ],
        "output_name": req.output_name,
        "outtype": req.outtype,
        "models_dir": str(MODELS_DIR),
        "convert_script": str(CONVERT_HF_TO_GGUF),
        "quantize_bin": str(LLAMA_QUANTIZE),
    }
    if req.quant_type:
        bake_config["quant_type"] = req.quant_type

    # Record bake lineage in models_meta.json
    _record_model_meta(
        model_filename=final_name,
        hf_repo=req.base_model,
        hf_filename=None,
        source_type="baked",
        quant=req.quant_type or req.outtype,
        bake_info={
            "base_model": req.base_model,
            "adapters": [{"path": a.path, "weight": a.weight} for a in req.adapters],
            "outtype": req.outtype,
            "quant_type": req.quant_type,
            "baked_at": datetime.now().isoformat(),
        },
    )

    # Write config to temp file
    config_file = LOGS_DIR / f"bake-config-{uuid.uuid4().hex[:8]}.json"
    config_file.write_text(json.dumps(bake_config, indent=2))

    cmd = [
        str(PYTHON), str(BAKE_SCRIPT),
        "--config", str(config_file),
    ]

    return _start_pipeline_task(
        task_type=PipelineTaskType.BAKE,
        cmd=cmd,
        input_path=json.dumps([a.path for a in req.adapters]),
        output_path=str(final_path),
    )


@router.get("/api/pipeline/tasks")
def list_pipeline_tasks() -> List[PipelineTask]:
    """List all pipeline tasks."""
    return sorted(_pipeline_tasks.values(), key=lambda t: t.created_at, reverse=True)


@router.get("/api/pipeline/tasks/{task_id}")
def get_pipeline_task(task_id: str) -> PipelineTask:
    """Get pipeline task details."""
    task = _pipeline_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Pipeline task not found")
    return task


@router.get("/api/pipeline/tasks/{task_id}/logs")
async def get_pipeline_task_logs(
    task_id: str, tail: int = Query(100), stream: bool = Query(False)
):
    """Get pipeline task logs."""
    task = _pipeline_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Pipeline task not found")

    log_path = Path(task.log_file)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    if not stream:
        try:
            lines = log_path.read_text().splitlines()
            return {"lines": lines[-tail:]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate():
        async with aiofiles.open(log_path, "r") as f:
            await f.seek(0, 2)
            idle_count = 0
            while True:
                line = await f.readline()
                if line:
                    idle_count = 0
                    yield line
                else:
                    current_task = _pipeline_tasks.get(task_id)
                    if current_task and current_task.status in (
                        PipelineTaskStatus.COMPLETED, PipelineTaskStatus.FAILED
                    ):
                        idle_count += 1
                        if idle_count > 5:
                            return
                    await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/plain")


@router.delete("/api/pipeline/tasks/{task_id}")
def cancel_pipeline_task(task_id: str):
    """Cancel a running pipeline task."""
    task = _pipeline_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Pipeline task not found")
    if task.status != PipelineTaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Task is not running")

    proc = _pipeline_processes.get(task_id)
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        del _pipeline_processes[task_id]

    task.status = PipelineTaskStatus.FAILED
    task.error_message = "Cancelled by user"
    task.finished_at = datetime.now()
    _save_pipeline_tasks()

    # Clean up partial output files
    output_path = Path(task.output_path)
    if output_path.exists() and output_path.is_file():
        output_path.unlink()

    return {"status": "cancelled", "task_id": task_id}
