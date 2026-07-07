"""
Jobs, Configs, and Outputs endpoints.
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..state import (
    CONFIGS_DIR,
    DATASETS_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    TOKENIZED_DATASETS,
    WORKSPACE,
    ConfigCreate,
    DatasetUploadRequest,
    Job,
    JobCreate,
    JobStatus,
    _extract_output_dir,
    _jobs,
    _processes,
    _refresh_metrics,
    _save_jobs,
    _try_start_next_pending,
    _validate_path,
)

router = APIRouter()

log = logging.getLogger(__name__)


# ─── Helper ──────────────────────────────────────────────────────────────

def _build_override_args(overrides: Optional[Dict]) -> List[str]:
    """Convert override dict to CLI args."""
    if not overrides:
        return []
    return [
        f"--{k.replace('_', '-')}={v}" for k, v in overrides.items()
    ]


# ─── Jobs Endpoints ────────────────────────────────────────────────────

@router.post("/api/jobs", status_code=201)
def create_job(req: JobCreate) -> Job:
    """Queue a new training job. It will start automatically when no other job is running."""
    # Validate config input
    if not req.config_path and not req.config_inline:
        raise HTTPException(
            status_code=400,
            detail="Either config_path or config_inline must be provided",
        )
    if req.config_path and req.config_inline:
        raise HTTPException(
            status_code=400,
            detail="Only one of config_path or config_inline should be provided",
        )

    job_id = str(uuid.uuid4())[:8]
    overrides = req.overrides or {}

    # If a base_model override is provided, add it to overrides
    if req.base_model:
        overrides["base_model"] = req.base_model

    # Resolve source config path
    if req.config_inline:
        # Validate YAML before writing to disk
        try:
            yaml.safe_load(req.config_inline)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
        config_path = CONFIGS_DIR / f"job-{job_id}.yaml"
        config_path.write_text(req.config_inline)
    else:
        config_path = _validate_path(req.config_path, CONFIGS_DIR, suffix=".yaml")
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Config file not found")

    # Write a job-specific config with all overrides and dataset_prepared_path baked in
    try:
        config_data = yaml.safe_load(config_path.read_text())
        for k, v in overrides.items():
            config_data[k] = v

        # Scope output_dir under a subfolder named after the base model
        base_model_str = config_data.get("base_model", "")
        if base_model_str:
            model_slug = re.sub(r"[^\w\-.]", "_", Path(base_model_str).name)
            raw_output = config_data.get("output_dir", f"lora-{job_id}")
            output_leaf = Path(raw_output).name
            config_data["output_dir"] = str(OUTPUTS_DIR / model_slug / output_leaf)

        config_data["dataset_prepared_path"] = str(TOKENIZED_DATASETS / job_id)
        job_config_path = CONFIGS_DIR / f"job-{job_id}.yaml"
        job_config_path.write_text(yaml.dump(config_data, default_flow_style=False))
        config_path = job_config_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to prepare config: {e}")

    # Extract output dir
    output_dir = _extract_output_dir(config_path, None)

    log_file = LOGS_DIR / f"{job_id}.log"

    # Create job record as PENDING
    job = Job(
        job_id=job_id,
        status=JobStatus.PENDING,
        config_path=str(config_path),
        output_dir=output_dir,
        created_at=datetime.now(),
        log_file=str(log_file),
    )

    _jobs[job_id] = job
    _save_jobs()

    # Job remains pending until approved by an admin
    return _jobs[job_id]


@router.get("/api/jobs")
def list_jobs() -> List[Job]:
    """List all jobs."""
    for job in _jobs.values():
        if job.status == JobStatus.RUNNING:
            _refresh_metrics(job)
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    """Get job details."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.RUNNING:
        _refresh_metrics(job)
    return job


@router.get("/api/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str, tail: int = Query(100), stream: bool = Query(False)
):
    """Get job logs."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = Path(job.log_file)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    if not stream:
        # Return last N lines
        try:
            lines = log_path.read_text().splitlines()
            return {"lines": lines[-tail:]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Stream response — terminates when job reaches terminal state and no new lines
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
                    # Check if job is done and we've drained the log
                    current_job = _jobs.get(job_id)
                    if current_job and current_job.status in (
                        JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
                    ):
                        idle_count += 1
                        if idle_count > 5:  # ~1.5s of no new lines after completion
                            return
                    await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/plain")


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a running job."""
    import subprocess

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.RUNNING, JobStatus.PENDING):
        raise HTTPException(status_code=400, detail="Job is not running or pending")

    proc = _processes.get(job_id)
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        del _processes[job_id]

    job.status = JobStatus.CANCELLED
    job.finished_at = datetime.now()
    _save_jobs()

    return {"status": "cancelled", "job_id": job_id}


@router.post("/api/jobs/{job_id}/approve")
def approve_job(job_id: str):
    """Approve a pending job so it can be started."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Only pending jobs can be approved")
    if job.approved:
        raise HTTPException(status_code=400, detail="Job is already approved")

    job.approved = True
    _save_jobs()

    # Try to start immediately if nothing else is running
    _try_start_next_pending()

    return _jobs[job_id]


# ─── Config Endpoints ──────────────────────────────────────────────────

@router.post("/api/configs", status_code=201)
def create_config(req: ConfigCreate):
    """Save a new config file."""
    # Validate name
    if not req.name or "/" in req.name or "\\" in req.name:
        raise HTTPException(status_code=400, detail="Invalid config name")

    # Validate YAML before writing
    try:
        yaml.safe_load(req.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    config_path = CONFIGS_DIR / f"{req.name}.yaml"
    if config_path.exists():
        raise HTTPException(status_code=409, detail="Config already exists")

    config_path.write_text(req.content)

    return {"name": req.name, "created": True}


@router.post("/api/datasets", status_code=201)
def upload_dataset(req: DatasetUploadRequest):
    """Save a local training dataset (JSONL) pushed by the API.

    Returns the absolute path the trainer should reference in its dataset
    config. Used for datasets with no HuggingFace path (e.g. curator output);
    train.py loads such local files via load_dataset("json").
    """
    if not req.content:
        raise HTTPException(status_code=400, detail="Empty dataset content")

    # Sanitize the display name into a filesystem-safe stem, then make it unique
    # so re-uploads of the same dataset never collide.
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", (req.name or "dataset")).strip("-.") or "dataset"
    dataset_path = DATASETS_DIR / f"{stem}-{uuid.uuid4().hex[:12]}.jsonl"

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(req.content)

    rows = sum(1 for line in req.content.splitlines() if line.strip())
    log.info("Uploaded dataset %s (%d rows, %d bytes)", dataset_path, rows, len(req.content))

    return {"path": str(dataset_path), "rows": rows}


@router.get("/api/configs")
def list_configs():
    """List all saved configs."""
    configs = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        configs.append(
            {
                "name": path.stem,
                "path": str(path),
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
        )
    return configs


@router.get("/api/configs/{config_name}")
def get_config(config_name: str):
    """Get config content."""
    config_path = _validate_path(config_name, CONFIGS_DIR, suffix=".yaml")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    return {"name": config_name, "content": config_path.read_text()}


@router.delete("/api/configs/{config_name}")
def delete_config(config_name: str):
    """Delete a config."""
    config_path = _validate_path(config_name, CONFIGS_DIR, suffix=".yaml")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")

    # Safety check: reject if any RUNNING job uses this config
    for job in _jobs.values():
        if job.status == JobStatus.RUNNING and job.config_path == str(
            config_path
        ):
            raise HTTPException(
                status_code=409,
                detail="Cannot delete config: job is using it",
            )

    config_path.unlink()
    return {"deleted": True}


# ─── Outputs Endpoint ──────────────────────────────────────────────────

@router.get("/api/outputs")
def list_outputs():
    """List completed training output directories."""
    outputs = []
    if not OUTPUTS_DIR.exists():
        return outputs

    for output_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not output_dir.is_dir():
            continue

        has_model = (output_dir / "model.safetensors").exists() or (
            output_dir / "adapter_model.safetensors"
        ).exists()
        has_config = (output_dir / "config.json").exists()
        # Only count top-level files to avoid expensive recursive walk on large checkpoints
        size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())

        outputs.append(
            {
                "name": output_dir.name,
                "path": str(output_dir),
                "has_model": has_model,
                "has_config": has_config,
                "size_bytes": size,
                "modified": output_dir.stat().st_mtime,
            }
        )

    return sorted(outputs, key=lambda o: o["modified"], reverse=True)
