"""
System management and health check endpoints.
"""

import json
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from ..state import (
    API_VERSION,
    CHAT_TEMPLATE_OVERRIDE,
    LLAMA_SERVER_ARGS_FILE,
    MODELS_DIR,
    WORKSPACE,
    ApplyLorasRequest,
    ChatTemplateUploadRequest,
    HealthResponse,
    JobStatus,
    _check_disk,
    _check_gpu,
    _check_inference_health,
    _get_active_loras_from_server,
    _jobs,
    _restart_llama_server,
)

log = logging.getLogger(__name__)

router = APIRouter()


# ─── llama-server Management ──────────────────────────────────────────

@router.post("/api/system/restart-llama-server")
def restart_llama_server():
    """Restart llama-server to rescan models directory."""
    return _restart_llama_server()


# ─── Chat Template Endpoints ─────────────────────────────────────────

@router.post("/api/system/chat-template")
async def upload_chat_template(req: ChatTemplateUploadRequest):
    """Upload a custom Jinja2 chat template file.

    The template is stored and used via --chat-template-file on next server start.
    """
    CHAT_TEMPLATE_OVERRIDE.write_text(req.content)
    log.info("Custom chat template uploaded (%d bytes)", len(req.content))

    # Restart to apply
    restart_result = _restart_llama_server()
    return {"status": "applied", "restart": restart_result}


@router.delete("/api/system/chat-template")
def clear_chat_template():
    """Clear the custom chat template override, reverting to GGUF auto-detection."""
    if CHAT_TEMPLATE_OVERRIDE.exists():
        CHAT_TEMPLATE_OVERRIDE.unlink()
        log.info("Custom chat template cleared, reverting to GGUF auto-detection")
        restart_result = _restart_llama_server()
        return {"status": "cleared", "restart": restart_result}
    return {"status": "no_override", "message": "No custom template was set"}


@router.get("/api/system/chat-template")
def get_chat_template():
    """Return the current chat template override status."""
    if CHAT_TEMPLATE_OVERRIDE.exists():
        content = CHAT_TEMPLATE_OVERRIDE.read_text()
        return {"override": True, "content": content}
    return {"override": False, "message": "Using GGUF-embedded template (auto-detected)"}


@router.get("/api/system/chat-templates/builtin")
def list_builtin_templates():
    """List available built-in chat template names."""
    # llama.cpp built-in templates (52 named templates)
    builtin = [
        "chatml", "llama2", "llama3", "llama4", "mistral-v1", "mistral-v3",
        "mistral-v3-tekken", "mistral-v7", "phi3", "phi4", "falcon3",
        "zephyr", "monarch", "gemma", "gemma2", "orion", "openchat",
        "vicuna", "vicuna-orca", "deepseek", "deepseek2", "deepseek3",
        "command-r", "yarnchat", "solar", "surfer", "granite",
        "gigachat", "megrez", "exaone3", "bailing", "glm3", "glm4",
        "minicpm", "rwkv-world", "rwkv", "chatglm3", "chatglm4",
        "qwen", "qwen3", "smolvlm", "kimi-k2", "grok-2",
    ]
    return {"templates": builtin}


# ─── LoRA Management ─────────────────────────────────────────────────

@router.post("/api/system/apply-loras")
def apply_loras(req: ApplyLorasRequest):
    """Set LoRA adapter scales at runtime via llama-server's native API.

    If all requested adapters are already preloaded, adjusts scales without restart.
    If new adapters need loading, updates the preload args and restarts.
    Pass an empty loras list to set all scales to 0.
    """
    import urllib.request

    # Validate all requested LoRA files exist
    for lora in req.loras:
        lora_file = lora.get("file", "")
        if not lora_file:
            raise HTTPException(status_code=400, detail="Each lora must have a 'file' field")
        lora_path = MODELS_DIR / lora_file
        if not lora_path.exists():
            raise HTTPException(status_code=404, detail=f"LoRA file not found: {lora_file}")

    # Check which adapters are currently loaded in llama-server
    loaded_adapters = _get_active_loras_from_server()
    loaded_paths = {a.get("path", ""): a.get("id") for a in loaded_adapters}

    # Check if all requested adapters are already loaded
    all_loaded = True
    for lora in req.loras:
        lora_path = str(MODELS_DIR / lora["file"])
        if lora_path not in loaded_paths:
            all_loaded = False
            break

    if all_loaded and loaded_adapters:
        # Hot-swap: set scales via native API (no restart needed)
        scale_updates = []
        for lora in req.loras:
            lora_path = str(MODELS_DIR / lora["file"])
            adapter_id = loaded_paths.get(lora_path)
            if adapter_id is not None:
                scale_updates.append({"id": adapter_id, "scale": lora.get("scale", 1.0)})

        try:
            payload = json.dumps(scale_updates).encode()
            api_req = urllib.request.Request(
                "http://localhost:8080/lora-adapters",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(api_req, timeout=5) as resp:
                resp.read()
            return {
                "status": "applied",
                "method": "hot-swap",
                "loras": req.loras,
                "restart": False,
            }
        except Exception as e:
            log.warning("Hot-swap failed, falling back to restart: %s", e)

    # Cold path: update args file with --lora-init-without-apply and restart
    args_parts = []
    for lora in req.loras:
        lora_path = MODELS_DIR / lora["file"]
        scale = lora.get("scale")
        if scale is not None:
            args_parts.extend(["--lora-scaled", f"{lora_path}:{float(scale)}"])
        else:
            args_parts.extend(["--lora", str(lora_path)])

    LLAMA_SERVER_ARGS_FILE.write_text(" ".join(args_parts))
    restart_result = _restart_llama_server()

    return {
        "status": "applied",
        "method": "restart",
        "loras": req.loras,
        "args": " ".join(args_parts) or "(none)",
        "restart": restart_result,
    }


@router.get("/api/system/active-loras")
def get_active_loras():
    """Return currently active LoRA adapters from llama-server's native API."""
    # Try native API first (live state)
    adapters = _get_active_loras_from_server()
    if adapters:
        loras = []
        for a in adapters:
            loras.append({
                "id": a.get("id"),
                "file": Path(a.get("path", "")).name,
                "scale": a.get("scale", 0.0),
            })
        return {"loras": loras, "source": "llama-server"}

    # Fallback to args file if server is not responding
    if not LLAMA_SERVER_ARGS_FILE.exists():
        return {"loras": [], "source": "args-file"}

    raw = LLAMA_SERVER_ARGS_FILE.read_text().strip()
    if not raw:
        return {"loras": [], "source": "args-file"}

    loras = []
    parts = raw.split()
    i = 0
    while i < len(parts):
        if parts[i] == "--lora-scaled" and i + 1 < len(parts):
            val = parts[i + 1]
            if ":" in val:
                fname, scale_str = val.rsplit(":", 1)
                loras.append({"file": Path(fname).name, "scale": float(scale_str)})
            else:
                loras.append({"file": Path(val).name, "scale": 1.0})
            i += 2
        elif parts[i] == "--lora" and i + 1 < len(parts):
            loras.append({"file": Path(parts[i + 1]).name, "scale": 1.0})
            i += 2
        else:
            i += 1

    return {"loras": loras, "source": "args-file"}


# ─── Health Endpoints ─────────────────────────────────────────────────

@router.get("/health")
def health() -> HealthResponse:
    """Composite health check: API + inference server + GPU + disk."""
    running_count = sum(
        1 for j in _jobs.values() if j.status == JobStatus.RUNNING
    )

    inference_healthy, loaded_model = _check_inference_health()
    gpu_available, gpu_used, gpu_total = _check_gpu()
    active_loras = _get_active_loras_from_server() if inference_healthy else None

    models_free = _check_disk(str(MODELS_DIR))
    workspace_free = _check_disk(str(WORKSPACE))

    if inference_healthy:
        status = "ok"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        api_healthy=True,
        inference_healthy=inference_healthy,
        running_jobs=running_count,
        jobs_total=len(_jobs),
        api_version=API_VERSION,
        loaded_model=loaded_model,
        active_loras=active_loras,
        gpu_available=gpu_available,
        gpu_memory_used_gb=gpu_used,
        gpu_memory_total_gb=gpu_total,
        disk_models_free_gb=models_free,
        disk_workspace_free_gb=workspace_free,
    )


@router.get("/health/live")
def health_liveness():
    """Liveness probe — is the API process alive?"""
    return {"status": "alive"}


@router.get("/health/ready")
def health_readiness():
    """Readiness probe — can the system serve inference?"""
    inference_healthy, _ = _check_inference_health()
    if inference_healthy:
        return {"status": "ready"}
    return {"status": "not_ready", "reason": "inference server not responding"}
