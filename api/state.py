"""
Shared state, constants, enums, models, and helper functions for the Training API.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────

WORKSPACE = Path("/workspace/training")
CONFIGS_DIR = WORKSPACE / "configs"
OUTPUTS_DIR = WORKSPACE / "outputs"
LOGS_DIR = WORKSPACE / "logs"
# Local training datasets uploaded by the API (e.g. curated JSONL that has no
# HuggingFace path). train.py loads these as local json instead of from HF.
DATASETS_DIR = Path(os.environ.get("UPLOADED_DATASETS", str(WORKSPACE / "datasets")))
JOBS_STATE_FILE = WORKSPACE / "jobs.json"
PIPELINE_STATE_FILE = WORKSPACE / "pipeline_tasks.json"
VENV_BIN = Path("/opt/venv/bin")
ACCELERATE = VENV_BIN / "accelerate"
PYTHON = VENV_BIN / "python"
TRAIN_SCRIPT = Path("/workspace/training/api/train.py")
MERGE_SCRIPT = Path("/workspace/training/api/merge_lora.py")
MODELS_DIR = Path(os.environ.get("LLAMA_ARG_MODELS_DIR", "/models"))
TOKENIZED_DATASETS = Path(os.environ.get("TOKENIZED_DATASETS", "/workspace/cache/tokenized-datasets"))
LLAMA_QUANTIZE = Path("/app/llama-quantize")
CONVERT_HF_TO_GGUF = Path("/app/convert/convert_hf_to_gguf.py")
CONVERT_LORA_TO_GGUF = Path("/app/convert/convert_lora_to_gguf.py")
BAKE_SCRIPT = Path("/workspace/training/api/bake_model.py")
LLAMA_SERVER_ARGS_FILE = Path("/app/llama-server.args")
MODELS_META_FILE = MODELS_DIR / "models_meta.json"

API_VERSION = "1.0.0"

# Chat template paths
CHAT_TEMPLATE_OVERRIDE = Path("/app/chat-template-override.jinja")
CHAT_TEMPLATES_DIR = Path("/app/chat-templates")

# HF cache paths
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/workspace/hf-hub"))

# Curator classifier repos
CURATOR_CLASSIFIER_REPOS = [
    "nvidia/quality-classifier-deberta",
    "nvidia/domain-classifier",
    "nvidia/multilingual-domain-classifier",
    "nvidia/content-type-classifier-deberta",
    "HuggingFaceFW/fineweb-edu-classifier",
    "nvidia/nemocurator-fineweb-mixtral-edu-classifier",
    "nvidia/nemocurator-fineweb-nemotron-4-edu-classifier",
    "nvidia/prompt-task-and-complexity-classifier",
    "microsoft/deberta-v3-base",
]

# FastText model paths
CURATOR_FASTTEXT_DIR = HF_CACHE_DIR / "fasttext"

CURATOR_FASTTEXT_MODELS = [
    {
        "name": "lid.176.bin",
        "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "description": "FastText language identification model (full, ~125 MB)",
        "curator_path": "/workspace/curator/cache/hf-hub/fasttext/lid.176.bin",
    },
    {
        "name": "lid.176.ftz",
        "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        "description": "FastText language identification model (compressed, ~917 KB)",
        "curator_path": "/workspace/curator/cache/hf-hub/fasttext/lid.176.ftz",
    },
]


# ─── Enums and Models ───────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    config_path: Optional[str] = None
    config_inline: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None
    base_model: Optional[str] = None

    class Config:
        example = {
            "config_path": "my-config",
            "base_model": "NousResearch/Meta-Llama-3-8B-Instruct",
            "overrides": {"lora_r": 64, "num_epochs": 2},
        }


class Job(BaseModel):
    job_id: str
    status: JobStatus
    config_path: str
    output_dir: str
    pid: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_file: str
    metrics: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    approved: bool = False


class ConfigCreate(BaseModel):
    name: str
    content: str


class DatasetUploadRequest(BaseModel):
    """A local training dataset pushed by the API (JSONL text).

    Used for datasets that have no HuggingFace path — e.g. curator output. The
    content is the raw JSONL; the trainer loads it via load_dataset("json").
    """

    name: str
    content: str


class HealthResponse(BaseModel):
    status: str  # "ok", "degraded", "unhealthy"
    api_healthy: bool
    inference_healthy: bool
    running_jobs: int
    jobs_total: int
    api_version: str
    loaded_model: Optional[str] = None
    active_loras: Optional[List[Dict[str, Any]]] = None
    gpu_available: Optional[bool] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    disk_models_free_gb: Optional[float] = None
    disk_workspace_free_gb: Optional[float] = None


class ModelPullRequest(BaseModel):
    name: str  # HuggingFace repo ID, e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF"
    filename: Optional[str] = None  # Specific GGUF file in the repo

class ModelDeleteRequest(BaseModel):
    name: str  # Filename in models dir to delete


# ─── Pipeline Models ───────────────────────────────────────────────────

class HfModelPullRequest(BaseModel):
    """Download a full HuggingFace model (safetensors) for conversion."""
    repo_id: str  # HuggingFace repo ID, e.g. "NousResearch/Llama-3.2-1B"
    revision: Optional[str] = None  # Branch, tag, or commit hash
    output_name: Optional[str] = None  # Directory name in OUTPUTS_DIR; defaults to repo name


class PipelineTaskType(str, Enum):
    PULL_HF_MODEL = "pull_hf_model"
    MERGE_LORA = "merge_lora"
    CONVERT_TO_GGUF = "convert_to_gguf"
    CONVERT_LORA_TO_GGUF = "convert_lora_to_gguf"
    QUANTIZE = "quantize"
    BAKE = "bake"


class PipelineTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Pipeline task types that require GPU and should be blocked during training
_GPU_PIPELINE_TYPES = {
    PipelineTaskType.MERGE_LORA,
    PipelineTaskType.CONVERT_TO_GGUF,
    PipelineTaskType.CONVERT_LORA_TO_GGUF,
    PipelineTaskType.QUANTIZE,
    PipelineTaskType.BAKE,
}


class PipelineTask(BaseModel):
    task_id: str
    task_type: PipelineTaskType
    status: PipelineTaskStatus
    pid: Optional[int] = None
    created_at: datetime
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_file: str
    error_message: Optional[str] = None
    input_path: str
    output_path: str
    queued_cmd: Optional[List[str]] = None
    queued_env: Optional[Dict[str, str]] = None


class MergeLoraRequest(BaseModel):
    """Merge a LoRA/QLoRA adapter into its base model."""
    model_output: str  # Name of training output dir in OUTPUTS_DIR (e.g. "qlora-out")


class ConvertToGgufRequest(BaseModel):
    """Convert a HuggingFace-format model to GGUF."""
    model_path: str  # Path relative to OUTPUTS_DIR (e.g. "qlora-out/merged") or absolute
    outtype: str = "f16"  # f32, f16, bf16, q8_0, auto
    model_name: Optional[str] = None  # Optional name for the output file


class QuantizeRequest(BaseModel):
    """Quantize a GGUF model file."""
    model_file: str  # GGUF filename in MODELS_DIR
    quant_type: str = "Q4_K_M"  # Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, etc.


class ConvertLoraToGgufRequest(BaseModel):
    """Convert a LoRA adapter to GGUF format for dynamic loading."""
    model_output: str  # Training output dir name in OUTPUTS_DIR
    base_model: Optional[str] = None  # HF model ID; if None, read from adapter config
    outtype: str = "f32"  # f32, f16, bf16, q8_0
    output_name: Optional[str] = None  # Output filename; defaults to <model_output>-lora.gguf


class ApplyLorasRequest(BaseModel):
    """Configure llama-server to load LoRA adapters at inference time."""
    loras: List[Dict[str, Any]]  # [{"file": "coding-lora.gguf", "scale": 0.7}, ...]
    # If empty list, removes all LoRAs and restarts with base model only


class LoraAdapter(BaseModel):
    path: str  # Path relative to OUTPUTS_DIR
    weight: float = 1.0


class BakeRequest(BaseModel):
    """Bake multiple LoRA adapters into a single GGUF model."""
    base_model: str  # HF model ID or path in OUTPUTS_DIR
    adapters: List[LoraAdapter]
    output_name: str  # Name for the final GGUF file
    outtype: str = "f16"  # f32, f16, bf16, q8_0
    quant_type: Optional[str] = None  # Q4_K_M, Q8_0, etc. If None, skip quantization


class ModelRegisterRequest(BaseModel):
    name: str  # Relative path to first shard or file within MODELS_DIR


class ChatTemplateUploadRequest(BaseModel):
    content: str  # Jinja2 template text


class HfCacheEnsureRequest(BaseModel):
    repos: List[str] = CURATOR_CLASSIFIER_REPOS


# ─── State ──────────────────────────────────────────────────────────────

_state_lock = threading.Lock()  # Guards _jobs, _pipeline_tasks, and their process dicts
_jobs: Dict[str, Job] = {}
_processes: Dict[str, subprocess.Popen] = {}
_active_downloads: Dict[str, threading.Event] = {}  # name -> cancel event
_pipeline_tasks: Dict[str, PipelineTask] = {}
_pipeline_processes: Dict[str, subprocess.Popen] = {}


# ─── Path Safety ────────────────────────────────────────────────────────

def _validate_path(user_input: str, root: Path, suffix: str = "") -> Path:
    """Validate user-supplied path is under root dir. Raises HTTPException on traversal.

    Args:
        user_input: User-supplied filename or relative path
        root: Expected parent directory
        suffix: Optional suffix to append (e.g. ".yaml")
    Returns:
        Resolved safe Path
    """
    from fastapi import HTTPException

    candidate = (root / f"{user_input}{suffix}").resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid path: must be under {root}",
        )
    return candidate


# ─── Initialization ─────────────────────────────────────────────────────

def _ensure_dirs():
    """Create all required directories."""
    for d in [CONFIGS_DIR, OUTPUTS_DIR, LOGS_DIR, TOKENIZED_DATASETS, DATASETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─── Model Metadata ─────────────────────────────────────────────────────

# Common GGUF quant suffixes, ordered longest-first so greedy match works.
_QUANT_TYPES = [
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
    "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K_L",
    "Q6_K", "Q8_0", "Q8_1",
    "F16", "F32", "BF16",
]
_QUANT_PATTERN = re.compile(
    r"[-_.](" + "|".join(re.escape(q) for q in sorted(_QUANT_TYPES, key=len, reverse=True)) + r")(?:[-_.]|\.gguf$)",
    re.IGNORECASE,
)

_SPLIT_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def _parse_quant_from_filename(filename: str) -> Optional[str]:
    """Extract the quantization type from a GGUF filename, e.g. 'Q4_K_M' from 'Model-Q4_K_M.gguf'."""
    m = _QUANT_PATTERN.search(filename)
    return m.group(1).upper() if m else None


def _load_models_meta() -> Dict[str, Any]:
    """Load model metadata from JSON sidecar file."""
    if MODELS_META_FILE.exists():
        try:
            return json.loads(MODELS_META_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load models metadata: %s", e)
            return {}
    return {}


def _save_models_meta(meta: Dict[str, Any]):
    """Persist model metadata (atomic write)."""
    tmp = MODELS_META_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    tmp.replace(MODELS_META_FILE)


def _record_model_meta(
    model_filename: str,
    hf_repo: str,
    hf_filename: Optional[str],
    source_type: str,
    quant: Optional[str] = None,
    bake_info: Optional[Dict[str, Any]] = None,
):
    """Record metadata for a newly pulled or baked model."""
    if quant is None and hf_filename:
        quant = _parse_quant_from_filename(hf_filename)
    if quant is None:
        quant = _parse_quant_from_filename(model_filename)

    meta = _load_models_meta()
    entry = {
        "hf_repo": hf_repo,
        "hf_filename": hf_filename,
        "quant": quant,
        "source_type": source_type,
        "trainable": source_type not in ("gguf", "baked"),
        "pulled_at": datetime.now().isoformat(),
    }
    if bake_info:
        entry["bake_info"] = bake_info
    meta[model_filename] = entry
    _save_models_meta(meta)


def _remove_model_meta(model_filename: str):
    """Remove metadata entry for a deleted model."""
    meta = _load_models_meta()
    if model_filename in meta:
        del meta[model_filename]
        _save_models_meta(meta)


def _record_lora_meta(
    lora_filename: str,
    base_model: Optional[str],
    training_output: str,
):
    """Record metadata for a LoRA GGUF file."""
    meta = _load_models_meta()
    meta[lora_filename] = {
        "source_type": "lora_gguf",
        "base_model": base_model,
        "training_output": training_output,
        "created_at": datetime.now().isoformat(),
    }
    _save_models_meta(meta)


def _save_pipeline_tasks():
    """Persist pipeline task state to JSON file. Thread-safe."""
    with _state_lock:
        tmp = PIPELINE_STATE_FILE.with_suffix(".tmp")
        data = {tid: t.model_dump(mode="json") for tid, t in _pipeline_tasks.items()}
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(PIPELINE_STATE_FILE)


def _load_pipeline_tasks():
    """Load pipeline task state from JSON file."""
    global _pipeline_tasks
    if PIPELINE_STATE_FILE.exists():
        data = json.loads(PIPELINE_STATE_FILE.read_text())
        for task_id, task_data in data.items():
            try:
                task_data["created_at"] = datetime.fromisoformat(task_data["created_at"])
                if task_data.get("finished_at"):
                    task_data["finished_at"] = datetime.fromisoformat(task_data["finished_at"])
                task = PipelineTask(**task_data)
                if task.status == PipelineTaskStatus.RUNNING:
                    task.status = PipelineTaskStatus.FAILED
                    task.error_message = "Process lost on restart"
                    task.finished_at = datetime.now()
                _pipeline_tasks[task_id] = task
            except Exception as e:
                log.warning("Failed to load pipeline task %s: %s", task_id, e)


def _load_jobs():
    """Load job state from JSON file."""
    global _jobs
    if JOBS_STATE_FILE.exists():
        data = json.loads(JOBS_STATE_FILE.read_text())
        for job_id, job_data in data.items():
            try:
                job_data["created_at"] = datetime.fromisoformat(
                    job_data["created_at"]
                )
                if job_data.get("started_at"):
                    job_data["started_at"] = datetime.fromisoformat(
                        job_data["started_at"]
                    )
                if job_data.get("finished_at"):
                    job_data["finished_at"] = datetime.fromisoformat(
                        job_data["finished_at"]
                    )
                job = Job(**job_data)
                # Mark any RUNNING jobs as FAILED (process lost on restart)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error_message = "Process lost on restart"
                    job.finished_at = datetime.now()
                _jobs[job_id] = job
            except Exception as e:
                log.warning("Failed to load job %s: %s", job_id, e)


def _save_jobs():
    """Persist job state to JSON file (atomic write). Thread-safe."""
    with _state_lock:
        tmp = JOBS_STATE_FILE.with_suffix(".tmp")
        data = {jid: j.model_dump(mode="json") for jid, j in _jobs.items()}
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(JOBS_STATE_FILE)


def _refresh_metrics(job: Job):
    """Read latest metrics from trainer_state.json."""
    try:
        output_path = Path(job.output_dir)
        checkpoints = sorted(output_path.glob("checkpoint-*/trainer_state.json"))
        if checkpoints:
            data = json.loads(checkpoints[-1].read_text())
            job.metrics = data.get("log_history", [])
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to refresh metrics for job %s: %s", job.job_id, e)


def _extract_output_dir(config_path: Path, overrides: Optional[Dict]) -> str:
    """Extract and resolve output_dir from config YAML."""
    try:
        config_data = yaml.safe_load(config_path.read_text())
        output_dir = config_data.get("output_dir")
        # Override takes precedence
        if overrides and "output_dir" in overrides:
            output_dir = overrides["output_dir"]
    except (yaml.YAMLError, OSError) as e:
        log.warning("Failed to read output_dir from config %s: %s", config_path, e)
        output_dir = None

    if not output_dir:
        output_dir = "outputs"

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = WORKSPACE / output_path

    return str(output_path)


def _auto_convert_lora(job: Job):
    """Auto-convert a completed training job's LoRA adapter to GGUF format."""
    output_dir = Path(job.output_dir)
    has_adapter = (
        (output_dir / "adapter_model.safetensors").exists()
        or (output_dir / "adapter_model.bin").exists()
    )
    if not has_adapter:
        return  # Full fine-tune or no adapter — nothing to convert

    # Determine the model_output name relative to OUTPUTS_DIR
    try:
        rel_path = output_dir.relative_to(OUTPUTS_DIR)
    except ValueError:
        log.warning("Output dir %s is not under %s, skipping auto-convert", output_dir, OUTPUTS_DIR)
        return

    # Read base model from adapter_config.json
    base_model = None
    adapter_cfg_path = output_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        try:
            cfg = json.loads(adapter_cfg_path.read_text())
            base_model = cfg.get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to read adapter_config.json at %s: %s", adapter_cfg_path, e)

    # Build a safe output filename from the relative path
    out_name = str(rel_path).replace("/", "-").replace("\\", "-") + "-lora-f16.gguf"
    outfile = MODELS_DIR / out_name

    if outfile.exists():
        log.debug("LoRA GGUF already exists: %s, skipping auto-convert", out_name)
        return

    cmd = [
        str(PYTHON), str(CONVERT_LORA_TO_GGUF),
        str(output_dir),
        "--outfile", str(outfile),
        "--outtype", "f16",
    ]
    if base_model:
        cmd.extend(["--base-model-id", base_model])

    env = {"NO_LOCAL_GGUF": "1"}

    # Record metadata now so the LoRA is discoverable even while converting
    _record_lora_meta(out_name, base_model, str(rel_path))

    log.info("Auto-converting LoRA to GGUF: %s", out_name)
    _start_pipeline_task(
        task_type=PipelineTaskType.CONVERT_LORA_TO_GGUF,
        cmd=cmd,
        input_path=str(output_dir),
        output_path=str(outfile),
        env=env,
    )


def _is_training_running() -> bool:
    """Check if any training job is currently RUNNING."""
    return any(j.status == JobStatus.RUNNING for j in _jobs.values())


def _start_job(job: Job) -> None:
    """Actually launch a pending job's training process."""
    config_path = Path(job.config_path)
    if not config_path.exists():
        job.status = JobStatus.FAILED
        job.error_message = f"Config file not found: {config_path}"
        job.finished_at = datetime.now()
        _save_jobs()
        return

    # Build command
    cmd = [
        str(ACCELERATE),
        "launch",
        str(TRAIN_SCRIPT),
        str(config_path),
    ]

    # Open log file
    log_path = Path(job.log_file)
    try:
        log_fh = open(log_path, "w")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to open log: {e}"
        job.finished_at = datetime.now()
        _save_jobs()
        return

    # Launch process
    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(WORKSPACE),
        )
    except Exception as e:
        log_fh.close()
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to start training: {e}"
        job.finished_at = datetime.now()
        _save_jobs()
        return

    # Popen duplicates the fd internally, safe to close our handle
    log_fh.close()

    job.status = JobStatus.RUNNING
    job.pid = proc.pid
    job.started_at = datetime.now()
    _processes[job.job_id] = proc
    _save_jobs()


def _try_start_next_pending() -> None:
    """If no job is running, start the next approved pending job (FIFO)."""
    for job in _jobs.values():
        if job.status == JobStatus.RUNNING:
            return  # Something is already running

    # Find oldest approved pending job
    pending = [j for j in _jobs.values() if j.status == JobStatus.PENDING and j.approved]
    if not pending:
        return
    pending.sort(key=lambda j: j.created_at)
    _start_job(pending[0])


def _is_gpu_pipeline_running() -> bool:
    """Check if any GPU pipeline task is currently RUNNING."""
    for task in _pipeline_tasks.values():
        if task.status == PipelineTaskStatus.RUNNING and task.task_type in _GPU_PIPELINE_TYPES:
            return True
    return False


def _try_start_queued_pipeline_tasks() -> None:
    """Start next queued GPU pipeline task if no training or GPU pipeline task is running.

    Only starts ONE task at a time to prevent GPU memory contention.
    """
    if _is_training_running():
        return
    if _is_gpu_pipeline_running():
        return
    for task_id, task in _pipeline_tasks.items():
        if task.status == PipelineTaskStatus.QUEUED:
            log.info("Auto-starting queued pipeline task %s", task_id)
            _launch_pipeline_process(task)
            return  # One at a time


def _start_pipeline_task(
    task_type: PipelineTaskType,
    cmd: List[str],
    input_path: str,
    output_path: str,
    env: Optional[Dict[str, str]] = None,
) -> PipelineTask:
    """Launch a pipeline task, or queue it if training is running."""
    task_id = str(uuid.uuid4())[:8]
    log_file = LOGS_DIR / f"pipeline-{task_id}.log"

    # Queue GPU-intensive tasks while training is running
    gpu_task = task_type in _GPU_PIPELINE_TYPES
    if gpu_task and _is_training_running():
        task = PipelineTask(
            task_id=task_id,
            task_type=task_type,
            status=PipelineTaskStatus.QUEUED,
            created_at=datetime.now(),
            log_file=str(log_file),
            input_path=input_path,
            output_path=output_path,
        )
        task.queued_cmd = cmd
        task.queued_env = env
        _pipeline_tasks[task_id] = task
        _save_pipeline_tasks()
        log.info("Pipeline task %s queued (training is running)", task_id)
        return task

    task = PipelineTask(
        task_id=task_id,
        task_type=task_type,
        status=PipelineTaskStatus.RUNNING,
        created_at=datetime.now(),
        log_file=str(log_file),
        input_path=input_path,
        output_path=output_path,
    )
    task.queued_cmd = cmd
    task.queued_env = env

    _pipeline_tasks[task_id] = task
    _launch_pipeline_process(task)
    return task


def _launch_pipeline_process(task: PipelineTask) -> None:
    """Actually launch the subprocess for a pipeline task."""
    cmd = task.queued_cmd
    env = task.queued_env
    if not cmd:
        task.status = PipelineTaskStatus.FAILED
        task.error_message = "No command stored for queued task"
        task.finished_at = datetime.now()
        _save_pipeline_tasks()
        return

    run_env = os.environ.copy()
    run_env["PATH"] = f"{VENV_BIN}:/app:{run_env.get('PATH', '')}"
    run_env["PYTHONUNBUFFERED"] = "1"
    if env:
        run_env.update(env)

    try:
        log_fh = open(task.log_file, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=run_env,
            cwd=str(WORKSPACE),
        )
    except Exception as e:
        if 'log_fh' in locals():
            log_fh.close()
        task.status = PipelineTaskStatus.FAILED
        task.error_message = f"Failed to start: {e}"
        task.finished_at = datetime.now()
        _save_pipeline_tasks()
        return

    # Popen duplicates the fd internally, safe to close our handle
    log_fh.close()

    task.status = PipelineTaskStatus.RUNNING
    task.pid = proc.pid
    _pipeline_processes[task.task_id] = proc
    _save_pipeline_tasks()


def _restart_llama_server() -> dict:
    """Restart llama-server via supervisord to rescan models directory."""
    try:
        result = subprocess.run(
            ["supervisorctl", "restart", "llama-server"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return {"status": "restarted"}
        else:
            return {"status": "error", "message": result.stderr.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── Health Check Helpers ────────────────────────────────────────────────

def _check_inference_health() -> tuple:
    """Probe llama-server health. Returns (healthy: bool, model: str|None)."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8080/health", timeout=3) as resp:
            data = json.loads(resp.read())
            healthy = data.get("status") == "ok"
            model = data.get("model_path") or data.get("model")
            return healthy, model
    except Exception:
        return False, None


def _check_gpu() -> tuple:
    """Check system-wide GPU availability and memory via nvidia-smi.

    Returns (available, used_gb, total_gb).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            used_mb, total_mb = [float(x.strip()) for x in line.split(",")]
            return True, round(used_mb / 1024, 2), round(total_mb / 1024, 2)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    # Fallback to torch if nvidia-smi not available
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return True, None, round(total, 2)
    except ImportError:
        pass
    return False, None, None


def _check_disk(path: str) -> float:
    """Return free disk space in GB for given path."""
    try:
        stat = os.statvfs(path)
        return round((stat.f_bavail * stat.f_frsize) / (1024**3), 2)
    except OSError:
        return -1.0


def _get_active_loras_from_server() -> list:
    """Query llama-server for active LoRA adapters."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8080/lora-adapters", timeout=2) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


async def _poll_jobs():
    """Background task: poll job status every 5 seconds and start pending jobs."""
    while True:
        await asyncio.sleep(5)
        any_finished = False
        for job_id, job in list(_jobs.items()):
            if job.status != JobStatus.RUNNING:
                continue

            proc = _processes.get(job_id)
            if not proc:
                continue

            rc = proc.poll()
            if rc is not None:
                # Process finished
                job.exit_code = rc
                job.finished_at = datetime.now()
                job.status = (
                    JobStatus.COMPLETED if rc == 0 else JobStatus.FAILED
                )
                if rc != 0:
                    job.error_message = f"Process exited with code {rc}"
                else:
                    # Auto-convert LoRA adapter to GGUF if applicable
                    _auto_convert_lora(job)
                del _processes[job_id]
                any_finished = True

            # Refresh metrics
            _refresh_metrics(job)
            _save_jobs()

        # If a job just finished, try to start the next pending one
        # and auto-start any queued pipeline tasks
        if any_finished:
            _try_start_next_pending()
            _try_start_queued_pipeline_tasks()

        # Poll pipeline tasks
        for task_id, task in list(_pipeline_tasks.items()):
            if task.status != PipelineTaskStatus.RUNNING:
                continue
            proc = _pipeline_processes.get(task_id)
            if not proc:
                continue
            rc = proc.poll()
            if rc is not None:
                task.exit_code = rc
                task.finished_at = datetime.now()
                task.status = (
                    PipelineTaskStatus.COMPLETED if rc == 0
                    else PipelineTaskStatus.FAILED
                )
                if rc != 0:
                    # Read last lines of log for error context
                    try:
                        log_lines = Path(task.log_file).read_text().splitlines()
                        tail = "\n".join(log_lines[-10:])
                        task.error_message = f"Process exited with code {rc}. Tail:\n{tail}"
                    except OSError as e:
                        log.warning("Failed to read log tail for task %s: %s", task_id, e)
                        task.error_message = f"Process exited with code {rc}"
                del _pipeline_processes[task_id]
                _save_pipeline_tasks()
                # Start next queued GPU task now that this one finished
                _try_start_queued_pipeline_tasks()
