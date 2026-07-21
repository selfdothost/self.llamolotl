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
import time
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

# Max age (seconds) the job-poll heartbeat may reach before it's considered
# stale by _check_api_liveness(). The loop ticks every 5s; this gives a
# couple of missed ticks of slack before treating the process as wedged.
_LIVENESS_MAX_HEARTBEAT_AGE = 30.0

# Monotonic timestamp of the last _poll_jobs() loop iteration. Seeded at
# import time so the process isn't falsely reported unhealthy before the
# background task's first tick.
_last_poll_heartbeat: float = time.monotonic()

# ─── llama-server Watchdog (self.llamolotl#24) ────────────────────────
#
# llama-server runs as its own supervisord-managed program (supervisord.conf)
# — training-api does not spawn it and is not its parent, so unlike
# _jobs/_pipeline_tasks above it has no PID this process can waitpid() on
# directly. What training-api *can* do is watch llama-server's own /health
# signal (_check_inference_health()) and, if it stays unhealthy for an
# unreasonable amount of time with no forward progress, trigger the existing
# _restart_llama_server() recovery primitive — the same
# `supervisorctl restart llama-server` an operator runs by hand today.
#
# How long a sustained inference-unhealthy signal may persist before it's
# treated as wedged rather than "still loading." llama-server's own /health
# returns HTTP 503 "Loading model" for every route (including /health)
# during a load — see middleware_server_state in
# self.llama/tools/server/server-http.cpp — so _check_inference_health()
# cannot currently distinguish "still loading a big model" from "actually
# crashed/hung"; both collapse to healthy=False. Deliberately generous
# because a cold-cache load of a large GGUF (this yard has served
# Qwen2.5-Coder-32B off a single shared 4090) can legitimately take minutes.
# Configurable since the right value depends on model size/storage; matches
# this file's existing os.environ.get() convention (MODELS_DIR,
# TOKENIZED_DATASETS above) rather than llama.cpp's LLAMA_ARG_* convention,
# since this is a training-api-only knob, not a llama-server CLI flag.
_LLAMA_WATCHDOG_TIMEOUT_SECONDS = float(
    os.environ.get("LLAMOLOTL_WATCHDOG_TIMEOUT_SECONDS", "900")
)

# Minimum spacing between two watchdog-triggered recoveries. The unhealthy
# streak is reset to "just started" every time a recovery fires (see
# _check_llama_server_watchdog), which already gives a freshly-restarted
# instance a full new timeout window before firing again — this cooldown is
# a second, independent guard against re-firing too fast if that reset ever
# races with a very short configured timeout.
_LLAMA_WATCHDOG_COOLDOWN_SECONDS = float(
    os.environ.get("LLAMOLOTL_WATCHDOG_COOLDOWN_SECONDS", "120")
)

# Monotonic timestamp when the current unhealthy streak began, or None if
# the most recent check was healthy. This is the "forward progress" signal:
# any healthy observation clears it, restarting the clock from scratch.
_llama_unhealthy_since: Optional[float] = None

# Monotonic timestamp of the watchdog's last recovery action, or None.
_llama_watchdog_last_fired: Optional[float] = None

# Per-model-name monotonic timestamp of when a model was first observed with
# status "loading" via the router's GET /models. This deployment runs
# llama-server in ROUTER MODE (LLAMA_ARG_MODELS_DIR/LLAMA_ARG_MODELS_MAX —
# see the self.ai repo's manifests/llamolotl/10-deployment.yaml, which is
# the actual source of runtime config: this repo's own supervisord.conf /
# entrypoint.sh never set those flags directly, they're injected by the k8s
# Deployment), and in router mode /health does NOT reflect a wedged
# per-model load the way _LLAMA_WATCHDOG_TIMEOUT_SECONDS's "unhealthy
# streak" logic originally assumed — see
# _check_llama_server_model_load_staleness()'s docstring for why. A name
# leaving "loading" (loaded, failed, unloaded, or absent from the next
# response) clears its entry, same forward-progress model as
# _llama_unhealthy_since above.
_llama_model_loading_since: Dict[str, float] = {}


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
    size_bytes: Optional[int] = None,
):
    """Record metadata for a newly pulled or baked model.

    size_bytes is recorded when the caller already has the completed
    file's on-disk size in hand (both GGUF-pull success paths in
    routers/models.py do). It's used by integrity.py's periodic sweep
    (self.llamolotl#23) as a stronger corruption signal than existence
    alone -- a file whose size has since drifted from what was recorded
    at pull time is flagged even if its GGUF header still looks valid.
    Bake/LoRA-convert callers record meta before their async pipeline task
    finishes, so they have no size yet; those entries simply have no
    size_bytes field, and the sweep's size-mismatch check is skipped for
    them (see integrity.sweep_once).
    """
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
    if size_bytes is not None:
        entry["size_bytes"] = size_bytes
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

def _check_api_liveness(max_age: float = _LIVENESS_MAX_HEARTBEAT_AGE) -> tuple:
    """Check whether the training-api process itself is alive and unwedged.

    Deliberately independent of llama-server's status — that's what
    /health/ready checks. cavekit-inference.md R5 / cavekit-platform.md R4
    both call for liveness (API process alive) and readiness (can it serve
    inference) to be distinguished, not conflated. llama-server also already
    has its own supervisord `autorestart=true` (supervisord.conf); tying this
    endpoint's liveness to llama-server's state would make a k8s
    livenessProbe kill and restart this whole pod for a sibling-process
    crash that supervisord already self-heals, on top of what /health/ready
    already reports for traffic-routing purposes.

    Instead, this checks that the background job-poll loop (_poll_jobs, the
    asyncio task started in main.py's lifespan) is still ticking. If the
    event loop is wedged, or the poll task has died on an unhandled
    exception, the heartbeat goes stale and this correctly reports
    unhealthy — the actual condition a k8s livenessProbe exists to catch.

    Returns (alive: bool, detail: str).
    """
    age = time.monotonic() - _last_poll_heartbeat
    if age > max_age:
        return False, (
            f"job-poll loop heartbeat is {age:.1f}s stale (max {max_age:.0f}s) "
            "— event loop may be wedged or the poll task has died"
        )
    return True, f"job-poll loop heartbeat {age:.1f}s ago"


def _check_inference_health() -> tuple:
    """Probe llama-server health. Returns (healthy: bool, model: str|None).

    CAVEAT under router mode (this deployment — see
    _llama_model_loading_since's comment above): /health's `is_ready` flips
    true immediately once the router's own HTTP listener starts
    (tools/server/server.cpp, is_router_server branch), *before* any
    per-model child has loaded anything — unlike single-model mode, where
    is_ready only flips true after the model itself finishes loading. So
    this function reports healthy=True as long as the router process
    itself is up and serving, even while a specific model is hard-wedged in
    `status: loading` with the GPU idle. It still correctly catches "the
    router process itself is down/unresponsive" — just not "one model is
    stuck loading while the router is otherwise fine". See
    _check_llama_server_model_load_staleness() for the complementary,
    router-mode-aware signal that _check_llama_server_watchdog() also
    consults for that case.
    """
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
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        log.debug("nvidia-smi GPU check failed, falling back to torch: %s", e)
    # Fallback to torch if nvidia-smi not available
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return True, None, round(total, 2)
    except ImportError as e:
        log.debug("torch not available for GPU check fallback: %s", e)
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


def _probe_llama_server_raw_status() -> str:
    """Best-effort human-readable reason for the current inference-health
    signal, for watchdog *logging only* — never used for control flow (that
    is _check_inference_health()'s job).

    llama-server returns a generic {"status": "ok"} on success, but while a
    load is in flight it returns HTTP 503 with
    {"error": {"message": "Loading model", ...}} for *every* route,
    including /health (see middleware_server_state in
    self.llama/tools/server/server-http.cpp) — that error message is the
    only place "loading" is actually visible over HTTP, so this exists to
    surface it in watchdog log lines instead of a bare "unhealthy".

    That 503-during-load behavior is gated on `is_ready`, which in ROUTER
    MODE (this deployment) flips true immediately at router boot — before
    any per-model load — so in practice this function will almost always
    return "ready" here even during a wedged per-model load; the 503 path
    is really only reachable during the router's own brief startup window.
    Kept as-is for diagnostic value on that narrower case and because it's
    still accurate for single-model mode; _check_llama_server_model_load_staleness()
    is the router-mode-aware signal for the per-model case.
    """
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8080/health", timeout=3):
            return "ready"
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg = body.get("error", {}).get("message")
            if msg:
                return msg.lower()
        except Exception as parse_err:
            log.debug("llama-server watchdog: could not parse 503 body: %s", parse_err)
        return f"http {e.code}"
    except Exception as e:
        return f"unreachable ({e.__class__.__name__})"


def _get_llama_server_pid() -> Optional[int]:
    """Ask supervisord which pid it currently believes is the llama-server
    instance. Read-only, safe to call at any time. This is the only source
    of truth for "which process is llama-server" available to training-api
    — supervisord spawned it (supervisord.conf), not this process, so
    training-api has no PID of its own to track the way it does for
    Job/PipelineTask subprocesses (_processes / _pipeline_processes above).
    """
    try:
        result = subprocess.run(
            ["supervisorctl", "pid", "llama-server"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        log.debug("llama-server watchdog: could not query supervisorctl pid: %s", e)
    return None


def _looks_like_llama_server_process(proc) -> bool:
    """Identity check shared by _verify_llama_server_identity() (pid known
    in advance, from supervisorctl) and _find_orphaned_llama_server_workers()
    below (pid discovered via a system-wide scan): true if `proc`'s cmdline
    or reported process name contains "llama-server".

    This is a narrow cmdline/name substring match, never a broader
    heuristic — but note it cannot by itself distinguish the router from
    one of its own per-model workers, or from a leftover instance of a
    *prior* router: the router spawns each per-model worker by re-invoking
    the very same llama-server binary with different args (see
    get_server_exec_path() and the subprocess_create_ex() call in
    self.llama/tools/server/server-models.cpp — router and workers are
    literally the same executable). Callers combine this with pid/ppid/
    create_time context to tell those apart; this function only answers
    "is this a llama-server-family process at all".
    """
    try:
        cmdline = " ".join(proc.cmdline()).lower()
        name = proc.name().lower()
        return "llama-server" in cmdline or "llama-server" in name
    except Exception:
        return False


def _verify_llama_server_identity(pid: int) -> bool:
    """Confirm `pid` actually looks like a llama-server process before the
    watchdog logs or acts on it — guards against a stale/reused pid (e.g.
    supervisorctl reporting a pid that has since been recycled by an
    unrelated process). This check gates *logging/diagnostics only* for
    the pid supervisorctl reports as the router itself: the watchdog's
    recovery action for that pid is always the supervisord-mediated
    _restart_llama_server(), never a direct signal to it, so a false
    negative here just means quieter logs, not a missed or wrong kill.
    (Contrast _terminate_orphaned_llama_server_workers() below, which
    *does* signal pids directly — but only ones discovered independently
    via _find_orphaned_llama_server_workers(), never this supervisorctl-
    reported pid.)
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return True  # a verified zombie is still "the" instance, just dead
        return _looks_like_llama_server_process(proc)
    except Exception as e:
        log.debug("llama-server watchdog: could not verify pid %d: %s", pid, e)
        return False


def _find_llama_server_zombie_children(pid: int) -> List[int]:
    """Best-effort enumeration of zombie descendants of a *verified*
    llama-server pid, for diagnostic logging only.

    training-api is a *sibling* of llama-server — both run as separate
    programs under the same supervisord (supervisord.conf) — not its
    parent. It did not fork llama-server or any of llama-server's own
    children, so per POSIX semantics it cannot waitpid()/reap them
    regardless of what's found here: only the true parent (llama-server
    itself, or the container's PID-1 supervisord once a child is
    reparented to it) can actually collect their exit status. This exists
    so an operator sees *why* GPU memory may still look held immediately
    after the watchdog's restart, rather than that being a silent mystery.
    """
    try:
        import psutil
        parent = psutil.Process(pid)
        zombies = []
        for child in parent.children(recursive=True):
            try:
                if child.status() == psutil.STATUS_ZOMBIE:
                    zombies.append(child.pid)
            except Exception:
                continue  # child exited/changed state mid-scan; not our concern
        return zombies
    except Exception as e:
        log.debug("llama-server watchdog: could not enumerate children of pid %d: %s", pid, e)
        return []


def _find_orphaned_llama_server_workers(router_pid: int) -> List[dict]:
    """System-wide scan for llama-server-looking processes that are genuine
    OS-level orphans left behind by a *prior* router instance — as opposed
    to _find_llama_server_zombie_children() above, which only ever looks at
    real, currently-attached descendants of `router_pid` and is diagnostic
    only.

    Why this is a distinct case (self.llamolotl#24 follow-up — the
    original pass of this fix only checked _find_llama_server_zombie_children()
    and missed this entirely): the router
    (self.llama/tools/server/server-models.cpp) spawns each per-model
    worker via subprocess_create_ex(), which on POSIX resolves to
    posix_spawn()/posix_spawnp() (self.llama/vendor/sheredom/subprocess.h)
    — an ordinary fork+exec, with no setsid/session isolation and no
    PR_SET_PDEATHSIG. The router reaps its own children itself: each
    load() spawns a dedicated per-instance thread that blocks reading the
    child's stdout to EOF and then calls subprocess_join()/
    subprocess_destroy() — there is no SIGCHLD handler and nothing kills
    children automatically when the router process exits. That reaping
    only works while the router process (and its threads) are alive. On a
    *clean* SIGTERM, the router's own shutdown path (is_router_server
    branch, tools/server/server.cpp) does call models.unload_all() before
    exiting, which gracefully stops every running child first — so a
    supervisorctl restart under normal conditions generally cleans up
    after itself. But it is a real race, not a guarantee: server-models.cpp's
    own DEFAULT_STOP_TIMEOUT (10s per child, before force-killing it) is
    exactly as long as supervisord's default stopwaitsecs (10s) before
    supervisord escalates to SIGKILL — and an uncleanly-killed router
    (SIGKILL, OOM, or a *previous* firing of this very watchdog racing that
    same window) skips unload_all() entirely. Either way, any child still
    mid-load or mid-serve at that moment is reparented to PID 1
    (supervisord itself — see entrypoint.sh, "supervisord runs as PID 1")
    and keeps running, holding GPU VRAM, invisible to
    _find_llama_server_zombie_children() (which only walks descendants of
    the *current* router pid — a process no longer under that pid at all
    isn't a descendant of it). This is the literal meaning of "orphaned"
    in the original incident report this issue was filed from.

    A process qualifies as an orphan here only if ALL of:
      - it looks like a llama-server process by cmdline/name
        (_looks_like_llama_server_process() — the same narrow substring
        check used everywhere else in this module, never a broader scan)
      - it is not `router_pid` itself
      - its ppid is NOT `router_pid` — a normal, currently-attached child
        of the live router is left strictly alone here, full stop,
        regardless of anything else this scan notices about it
      - its start time (create_time) predates the *current* router
        instance's own start time — this is what distinguishes a genuine
        leftover from a prior router instance from a brand-new legitimate
        child the *current* router just spawned. Every real child of the
        current router was necessarily created after the current router
        started, so requiring create_time < router's create_time is a
        second, independent guard against ever flagging one (on top of
        the ppid check above), even in a scan-timing edge case.

    Returns a list of dicts (pid, ppid, create_time, cmdline) for
    identity-verified orphans, oldest first — read-only, does not signal
    anything. Best-effort: any psutil failure during the scan degrades to
    an empty list rather than propagating; this must never break the
    watchdog tick.
    """
    try:
        import psutil
    except Exception as e:
        log.debug("llama-server watchdog: psutil unavailable for orphan scan: %s", e)
        return []

    try:
        router_create_time = psutil.Process(router_pid).create_time()
    except Exception as e:
        log.debug(
            "llama-server watchdog: could not read router pid %d's create_time "
            "for orphan scan: %s", router_pid, e,
        )
        return []

    orphans: List[dict] = []
    try:
        candidates = list(psutil.process_iter())
    except Exception as e:
        log.debug("llama-server watchdog: orphan scan process_iter failed: %s", e)
        return []

    for proc in candidates:
        try:
            pid = proc.pid
            if pid == router_pid:
                continue
            ppid = proc.ppid()
            if ppid == router_pid:
                continue  # a normal, currently-attached child — never our concern here
            if not _looks_like_llama_server_process(proc):
                continue
            create_time = proc.create_time()
            if create_time >= router_create_time:
                continue  # as new (or newer) than the current router; not a leftover
            orphans.append({
                "pid": pid,
                "ppid": ppid,
                "create_time": create_time,
                "cmdline": " ".join(proc.cmdline()),
            })
        except Exception:
            continue  # process exited/changed state mid-scan, or a bad mock; skip it

    orphans.sort(key=lambda o: o["create_time"])
    return orphans


def _terminate_orphaned_llama_server_workers(orphans: List[dict]) -> List[int]:
    """Best-effort SIGTERM to identity-verified orphaned llama-server
    workers found by _find_orphaned_llama_server_workers().

    This is deliberately the ONE place in this module that signals a pid
    training-api did not itself spawn and is not the parent of.
    _restart_llama_server() (supervisorctl restart) remains the only
    recovery action against the router pid itself, precisely because
    training-api is not its parent and has no standing to reap it even if
    it could signal it usefully. Orphans are different: once a process is
    reparented to PID 1, restarting the router does *nothing* to it — it
    is no longer the router's child, so the router's own graceful
    unload_all()/terminate() shutdown path never touches it either. Restart
    alone would leave it running and holding GPU VRAM indefinitely, which
    is exactly the "orphaned llama-server process...left running on an
    unrelated model" symptom from the original incident. Explicit
    termination here is what actually closes that gap.

    Signaling a non-child pid directly is safe and POSIX-legal: signal
    delivery is gated on effective uid matching (both training-api and
    llama-server run as root under supervisord — supervisord.conf), not on
    process-tree relationship. Only *reaping* (collecting exit status via
    wait()/waitpid()) requires being the parent — and that is PID 1's job
    once the process is orphaned (see entrypoint.sh: "supervisord runs as
    PID 1"), not training-api's, so nothing here needs to (or tries to)
    reap anything.

    SIGTERM only, never SIGKILL: a llama-server child process (spawned by
    the router in non-router "child" mode) registers the same SIGTERM/
    SIGINT handler as the router itself does — see the
    `if (!is_run_by_cli)` sigaction block in tools/server/server.cpp, which
    applies to both — whose handler calls ctx_server.terminate(), the same
    graceful unblock-and-clean-up path the *original* router would have
    driven via unload_all() had it still been alive to do so. No SIGKILL
    escalation is attempted: this function only ever runs against pids
    that already passed independent identity re-verification, and a
    process that doesn't respond to SIGTERM is left for the operator or
    the next watchdog tick rather than force-killed sight unseen.

    Returns the pids actually signalled. Never raises: a failure against
    one pid (already exited, permission error, a bad mock, pid reuse
    between scan and here) is logged and skipped, never fatal to the rest.
    """
    try:
        import psutil
    except Exception as e:
        log.debug("llama-server watchdog: psutil unavailable for orphan termination: %s", e)
        return []

    terminated: List[int] = []
    for orphan in orphans:
        pid = orphan.get("pid")
        try:
            proc = psutil.Process(pid)
            # Re-verify identity at the moment of action, not just at scan
            # time — pids can be recycled by the OS between the scan and
            # here, and this is the one place in this module that actually
            # signals a pid it didn't verify via supervisorctl.
            if not _looks_like_llama_server_process(proc):
                log.warning(
                    "llama-server watchdog: orphan pid %d no longer verifies "
                    "as llama-server at termination time (likely pid reuse) "
                    "— skipping, not signaling.", pid,
                )
                continue
            proc.terminate()
            terminated.append(pid)
            log.error(
                "llama-server watchdog: sent SIGTERM to orphaned llama-server "
                "worker pid %d (ppid=%s, cmdline=%r) — reparented away from "
                "a prior router instance (not a child of the current router), "
                "so the restart below cannot clean it up on its own.",
                pid, orphan.get("ppid"), orphan.get("cmdline"),
            )
        except (Exception) as e:
            log.debug(
                "llama-server watchdog: could not signal orphan pid %d: %s",
                pid, e,
            )
    return terminated


def _probe_llama_server_models_status() -> Optional[list]:
    """GET the router's /models listing
    (server_models_routes::get_router_models in
    self.llama/tools/server/server-models.cpp) and return its `data` array
    of per-model status objects. Never used for anything but the router-
    mode staleness check below — a None return just means "no info right
    now" (unreachable, malformed response, or classic single-model mode
    where this route's response shape differs), not a health signal either
    way.
    """
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8080/models", timeout=3) as resp:
            data = json.loads(resp.read())
            models = data.get("data")
            return models if isinstance(models, list) else None
    except Exception as e:
        log.debug("llama-server watchdog: could not query /models for per-model status: %s", e)
        return None


def _check_llama_server_model_load_staleness() -> Optional[str]:
    """Router-mode-specific wedge signal, complementing (not replacing) the
    /health-based unhealthy streak in _check_llama_server_watchdog().

    Why this exists: this deployment runs llama-server in ROUTER MODE — see
    _llama_model_loading_since's module-level comment for how that's
    confirmed (the self.ai repo's manifests/llamolotl/10-deployment.yaml
    sets LLAMA_ARG_MODELS_DIR/LLAMA_ARG_MODELS_MAX; this repo's own files
    never set them directly). In router mode, /health's readiness flips
    true immediately once the router's own HTTP listener starts
    (tools/server/server.cpp, is_router_server branch) — *before* any
    per-model child has been asked to load anything. So
    _check_inference_health() reports healthy=True almost the entire time
    the router process itself is up, even while one specific model is
    hard-wedged in `status: loading` with the GPU sitting idle: exactly
    issue #24's "router spins waiting until model fully loaded" scenario
    (server_models::ensure_model_ready()'s unbounded condvar `wait()` on
    the model's status, self.llama/tools/server/server-models.cpp — there
    is no load timeout inside the router itself). Without this function,
    the /health-based streak would essentially never fire for that
    scenario, because /health never reports unhealthy for it.

    Tracks, per model name (in the module-level _llama_model_loading_since
    dict), the monotonic time it was first observed with status "loading";
    a name leaving that set (loaded, failed, unloaded, or simply absent
    from the next response) clears its entry — the same forward-progress
    model _llama_unhealthy_since already uses for the router-level streak.

    Returns the name of a model that has been stuck in "loading" for
    longer than _LLAMA_WATCHDOG_TIMEOUT_SECONDS, or None. Never raises:
    _probe_llama_server_models_status() already degrades to None on any
    failure, and this function is otherwise pure dict bookkeeping.
    """
    models = _probe_llama_server_models_status()
    now = time.monotonic()
    if models is None:
        return None

    seen = set()
    stale_name = None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("id")
        status = (entry.get("status") or {}).get("value")
        if not name or status != "loading":
            continue
        seen.add(name)
        since = _llama_model_loading_since.setdefault(name, now)
        if stale_name is None and (now - since) > _LLAMA_WATCHDOG_TIMEOUT_SECONDS:
            stale_name = name

    # Clear tracking for any model no longer reported as "loading" — same
    # forward-progress semantics as the health-based streak above.
    for name in list(_llama_model_loading_since):
        if name not in seen:
            del _llama_model_loading_since[name]

    return stale_name


def _check_llama_server_watchdog() -> None:
    """Detect a wedged llama-server and recover it. See self.llamolotl#24.

    Two independent wedge signals feed this, because this deployment runs
    llama-server in ROUTER MODE (LLAMA_ARG_MODELS_DIR/LLAMA_ARG_MODELS_MAX,
    set by the self.ai repo's manifests/llamolotl/10-deployment.yaml — not
    by anything in this repo directly) and router mode's /health does not
    behave the way a first pass at this fix assumed:

    1. _check_inference_health()'s unhealthy streak: any healthy
       observation clears it and restarts the clock. This still correctly
       catches "the router process itself is down/unresponsive" — but in
       router mode it will almost never fire for "one model is wedged in
       status: loading" specifically, because router mode's /health flips
       ready=true at router boot, before any per-model load even starts
       (see _check_inference_health()'s docstring).
    2. _check_llama_server_model_load_staleness(): the router-mode-aware
       complement, tracking per-model "loading" status directly via
       GET /models. This is what actually catches issue #24's "router
       spins waiting until model fully loaded" scenario — a stale model
       name here already means the timeout has been exceeded, no separate
       streak needed.

    Either signal past _LLAMA_WATCHDOG_TIMEOUT_SECONDS is treated as
    wedged; deliberately generous because a cold-cache load of a large
    GGUF can legitimately take minutes (see the constant's own comment).

    Recovery has two parts once a wedge fires:
      - _restart_llama_server() (supervisorctl restart) — the same action
        an operator takes by hand today. This remains the ONLY action ever
        taken against the router pid itself: llama-server is supervised by
        supervisord, not spawned by this process, so training-api is not
        its parent and has no standing to signal it directly.
      - _terminate_orphaned_llama_server_workers() for anything
        _find_orphaned_llama_server_workers() finds — pids that look like
        llama-server workers but are no longer children of the current
        router (reparented to PID 1 from a *prior* router instance, per
        that function's docstring). Restarting the router does nothing for
        these: they are not its children, so its own graceful shutdown
        path never touches them. This is the one place in this module that
        signals a pid training-api did not itself spawn — see that
        function's docstring for why that's still safe.
    """
    global _llama_unhealthy_since, _llama_watchdog_last_fired

    healthy, _ = _check_inference_health()
    stale_model = _check_llama_server_model_load_staleness()
    now = time.monotonic()

    if healthy and stale_model is None:
        if _llama_unhealthy_since is not None:
            log.info(
                "llama-server watchdog: inference healthy again after a %.1fs unhealthy streak",
                now - _llama_unhealthy_since,
            )
        _llama_unhealthy_since = None
        return

    if not healthy:
        if _llama_unhealthy_since is None:
            # Streak just started — could easily be a normal load in progress.
            _llama_unhealthy_since = now
        unhealthy_duration = now - _llama_unhealthy_since
    else:
        # /health itself is fine, but the router-mode-aware per-model check
        # already found a model stuck in "loading" past the timeout — that
        # IS the wedge condition on its own, independent of any streak.
        unhealthy_duration = _LLAMA_WATCHDOG_TIMEOUT_SECONDS

    if stale_model is None and unhealthy_duration < _LLAMA_WATCHDOG_TIMEOUT_SECONDS:
        return  # still within the generous grace period, and no stale model either

    if (
        _llama_watchdog_last_fired is not None
        and now - _llama_watchdog_last_fired < _LLAMA_WATCHDOG_COOLDOWN_SECONDS
    ):
        return  # already fired recently; let that recovery play out

    # ---- Wedge declared: this is a self-healing path, log it loudly ----
    raw_status = _probe_llama_server_raw_status()
    if not healthy:
        log.error(
            "llama-server watchdog: FIRING — inference has been unhealthy for "
            "%.1fs (timeout %.0fs, raw status: %r, stale model: %r) with no "
            "forward progress. Treating as WEDGED and forcing recovery via "
            "supervisord. This is an automatic self-healing action — if this "
            "fires repeatedly, investigate GPU/model-load issues "
            "(self.llamolotl#24) instead of relying on it.",
            unhealthy_duration, _LLAMA_WATCHDOG_TIMEOUT_SECONDS, raw_status, stale_model,
        )
    else:
        log.error(
            "llama-server watchdog: FIRING — router-level /health reports ok, "
            "but model %r has been stuck in status: loading past the %.0fs "
            "timeout with no forward progress (raw /health status: %r). This "
            "is the router-mode case /health alone cannot see — see "
            "_check_llama_server_model_load_staleness()'s docstring. Treating "
            "as WEDGED and forcing recovery via supervisord. This is an "
            "automatic self-healing action — if this fires repeatedly, "
            "investigate GPU/model-load issues (self.llamolotl#24) instead of "
            "relying on it.",
            stale_model, _LLAMA_WATCHDOG_TIMEOUT_SECONDS, raw_status,
        )

    pid = _get_llama_server_pid()
    if pid is not None and _verify_llama_server_identity(pid):
        zombies = _find_llama_server_zombie_children(pid)
        if zombies:
            log.error(
                "llama-server watchdog: pid %d has %d zombie child process(es) "
                "(%s) — training-api is not their parent and cannot reap them "
                "directly; restarting the supervisord-managed llama-server slot, "
                "which should clear them once pid %d exits and they're reaped by "
                "the container's init.",
                pid, len(zombies), zombies, pid,
            )

        orphans = _find_orphaned_llama_server_workers(pid)
        if orphans:
            terminated = _terminate_orphaned_llama_server_workers(orphans)
            log.error(
                "llama-server watchdog: found %d orphaned llama-server worker(s) "
                "reparented away from a prior router instance (pids %s) — sent "
                "SIGTERM to %d of them (%s). These are not children of the "
                "current router pid %d, so the restart below cannot clean them "
                "up on its own; this is a separate, explicit recovery step.",
                len(orphans), [o["pid"] for o in orphans], len(terminated), terminated, pid,
            )
    elif pid is not None:
        log.warning(
            "llama-server watchdog: supervisorctl reports pid %d for llama-server "
            "but it does not look like a live llama-server process — proceeding "
            "with restart anyway since supervisord's own bookkeeping drives it.",
            pid,
        )
    else:
        log.warning(
            "llama-server watchdog: could not determine llama-server's current "
            "pid via supervisorctl — proceeding with restart anyway. Orphan "
            "detection is skipped this cycle since it needs the current "
            "router's pid/create_time as its reference point."
        )

    result = _restart_llama_server()
    _llama_watchdog_last_fired = now
    # Give the freshly-restarted instance a clean slate / full new timeout
    # window rather than counting its own (expected) loading time against
    # the streak that just triggered this recovery.
    _llama_unhealthy_since = None
    _llama_model_loading_since.clear()
    log.error("llama-server watchdog: recovery action result: %s", result)


async def _poll_jobs():
    """Background task: poll job status every 5 seconds and start pending jobs."""
    global _last_poll_heartbeat
    while True:
        await asyncio.sleep(5)
        _last_poll_heartbeat = time.monotonic()

        try:
            _check_llama_server_watchdog()
        except Exception as e:
            # A bug in the watchdog itself must never take down the poll
            # loop that also drives job/pipeline progress and the liveness
            # heartbeat — log and keep going, same defensive posture as the
            # rest of this loop's per-item try/except-free reads below rely
            # on already-guarded helpers (_check_inference_health() etc).
            log.error("llama-server watchdog: unhandled error in check: %s", e, exc_info=True)

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
