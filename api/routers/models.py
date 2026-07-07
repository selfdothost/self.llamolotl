"""
Models, GGUF management, HF cache, and FastText endpoints.
"""

import asyncio
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..state import (
    CONVERT_HF_TO_GGUF,
    CURATOR_CLASSIFIER_REPOS,
    CURATOR_FASTTEXT_DIR,
    CURATOR_FASTTEXT_MODELS,
    HF_CACHE_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PYTHON,
    HfCacheEnsureRequest,
    ModelDeleteRequest,
    ModelPullRequest,
    ModelRegisterRequest,
    _SPLIT_SHARD_RE,
    _active_downloads,
    _load_models_meta,
    _record_model_meta,
    _remove_model_meta,
    _restart_llama_server,
)

log = logging.getLogger(__name__)

router = APIRouter()


# ─── Models Endpoint ───────────────────────────────────────────────────

@router.get("/api/models")
def list_models():
    """List completed model outputs."""
    models = []
    if not OUTPUTS_DIR.exists():
        return models

    for model_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        has_model = (model_dir / "model.safetensors").exists() or (
            model_dir / "adapter_model.safetensors"
        ).exists()
        has_config = (model_dir / "config.json").exists()

        # Calculate size
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

        models.append(
            {
                "name": model_dir.name,
                "path": str(model_dir),
                "has_model": has_model,
                "has_config": has_config,
                "size_bytes": size,
                "modified": model_dir.stat().st_mtime,
            }
        )

    return sorted(models, key=lambda m: m["modified"], reverse=True)


# ─── GGUF Model Management ────────────────────────────────────────────

@router.get("/api/models/available")
def list_available_models():
    """List GGUF model files in the models directory (recursive).
    Split GGUF shards are grouped: only the first shard is shown with the
    combined size of all parts."""
    if not MODELS_DIR.exists():
        return []

    # Clean up any dangling symlinks first
    for f in MODELS_DIR.iterdir():
        if f.is_symlink() and not f.resolve().exists():
            f.unlink()

    # Collect all GGUF files, skipping .cache, .downloading, and symlinks
    all_files = []
    for f in sorted(MODELS_DIR.rglob("*.gguf")):
        if f.name.endswith(".downloading"):
            continue
        if f.is_symlink():
            continue
        try:
            f.relative_to(MODELS_DIR / ".cache")
            continue
        except ValueError:
            pass
        all_files.append(f)

    # Group split shards by their base name
    # e.g. "Model-00001-of-00004.gguf" -> base "Model"
    grouped: Dict[str, List[Path]] = {}
    standalone: List[Path] = []

    for f in all_files:
        m = _SPLIT_SHARD_RE.search(f.name)
        if m:
            base = f.name[:m.start()]
            key = str(f.parent / base)
            grouped.setdefault(key, []).append(f)
        else:
            standalone.append(f)

    meta = _load_models_meta()
    models = []

    def _enrich(entry: dict) -> dict:
        """Add metadata fields (hf_repo, quant, trainable, etc.) to a model entry."""
        # Look up by filename (basename) since metadata keys are basenames
        basename = Path(entry["name"]).name
        m = meta.get(basename, {})
        entry["hf_repo"] = m.get("hf_repo")
        entry["quant"] = m.get("quant")
        entry["source_type"] = m.get("source_type")
        entry["pulled_at"] = m.get("pulled_at")
        if m.get("bake_info"):
            entry["bake_info"] = m["bake_info"]
        # Derive trainable: safetensors_converted models are trainable if original files still exist
        trainable = m.get("trainable", False)
        if trainable and m.get("source_type") == "safetensors_converted" and m.get("hf_repo"):
            safetensors_dir = OUTPUTS_DIR / m["hf_repo"].split("/")[-1]
            if not safetensors_dir.exists() or not any(safetensors_dir.glob("*.safetensors")):
                trainable = False
        entry["trainable"] = trainable
        return entry

    # Standalone (non-split) models
    for f in standalone:
        # Skip symlinks (they point to split shards and are handled below)
        if f.is_symlink():
            continue
        rel_path = str(f.relative_to(MODELS_DIR))
        # A top-level file is always registered (visible to llama-server)
        is_top_level = f.parent == MODELS_DIR
        models.append(_enrich({
            "name": rel_path,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
            "registered": is_top_level,
        }))

    # Split models — show first shard with combined size
    for key, shards in grouped.items():
        shards.sort(key=lambda p: p.name)
        first_shard = shards[0]
        total_size = sum(s.stat().st_size for s in shards)
        latest_modified = max(s.stat().st_mtime for s in shards)
        rel_path = str(first_shard.relative_to(MODELS_DIR))
        # Registered if first shard (or a symlink to it) exists at top level
        top_level_path = MODELS_DIR / first_shard.name
        registered = (
            (first_shard.parent == MODELS_DIR) or
            (top_level_path.is_symlink() or top_level_path.exists())
        )
        models.append(_enrich({
            "name": rel_path,
            "size": total_size,
            "modified": latest_modified,
            "shards": len(shards),
            "registered": registered,
        }))

    return models


@router.post("/api/models/register")
def register_model(req: ModelRegisterRequest):
    """Register a model in a subdirectory by symlinking its first shard
    to the top level of MODELS_DIR so llama-server can discover it."""
    model_path = (MODELS_DIR / req.name).resolve()
    # Path traversal protection
    if not str(model_path).startswith(str(MODELS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid model path")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    # Already at top level — nothing to do
    if model_path.parent == MODELS_DIR.resolve():
        return {"registered": True, "name": req.name, "action": "already_top_level"}

    symlink_path = MODELS_DIR / model_path.name
    if symlink_path.exists() or symlink_path.is_symlink():
        return {"registered": True, "name": req.name, "action": "already_registered"}

    symlink_path.symlink_to(model_path)

    # Restart llama-server so it discovers the newly registered model
    _restart_llama_server()

    return {"registered": True, "name": model_path.name, "action": "symlinked"}


class ModelInspectRequest(BaseModel):
    name: str


@router.post("/api/models/inspect")
async def inspect_model(req: ModelInspectRequest):
    """Query HuggingFace for model file sizes before downloading."""
    from huggingface_hub import HfApi

    repo_id = req.name.strip()

    try:
        api = HfApi()
        repo_info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to access repo '{repo_id}': {e}")

    def get_size(s):
        size = getattr(s, 'size', None) or 0
        if size == 0 and hasattr(s, 'lfs') and s.lfs:
            size = s.lfs.get('size', 0) if isinstance(s.lfs, dict) else getattr(s.lfs, 'size', 0)
        return size

    all_siblings = repo_info.siblings or []
    gguf_siblings = [s for s in all_siblings if s.rfilename.endswith('.gguf')]

    if gguf_siblings:
        files = [{"name": s.rfilename, "size": get_size(s)} for s in gguf_siblings]
        return {"type": "gguf", "files": files, "total_size": sum(f["size"] for f in files)}

    # No GGUFs — check for safetensors (which will be downloaded and converted)
    _allow_exts = (".safetensors", ".json", ".txt", ".model", ".py")
    _ignore_exts = (".gguf", ".bin", ".msgpack", ".h5", ".ot", ".md")
    _ignore_names = {".gitattributes"}

    st_files = []
    for s in all_siblings:
        fname = s.rfilename
        if fname in _ignore_names:
            continue
        if any(fname.endswith(ext) for ext in _ignore_exts):
            continue
        if not any(fname.endswith(ext) for ext in _allow_exts):
            continue
        st_files.append({"name": fname, "size": get_size(s)})

    if not st_files:
        raise HTTPException(status_code=400, detail=f"No downloadable files found in '{repo_id}'")

    # Show only weight shards in the file list; configs are noise to the user.
    # total_size still reflects the full download (weights + configs).
    model_files = [f for f in st_files if f["name"].endswith(".safetensors")]
    display_files = model_files if model_files else st_files
    return {"type": "safetensors", "files": display_files, "total_size": sum(f["size"] for f in st_files)}


@router.post("/api/models/pull")
async def pull_model(req: ModelPullRequest):
    """Pull a GGUF model from HuggingFace. Streams progress as NDJSON."""
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = req.name.strip()
    filename = req.filename

    async def generate():
        try:
            api = HfApi()

            # List repo files to find GGUFs
            try:
                all_files = api.list_repo_files(repo_id)
            except Exception as e:
                yield json.dumps({"error": f"Failed to access repo '{repo_id}': {e}"}) + "\n"
                return

            gguf_files = [f for f in all_files if f.endswith(".gguf")]

            if not gguf_files:
                # No GGUFs — check if this is a safetensors model repo.
                # If so, download it as an HF model (safetensors + config) and
                # then convert to GGUF automatically.
                has_safetensors = any(f.endswith(".safetensors") for f in all_files)
                has_config = "config.json" in all_files
                if not has_safetensors or not has_config:
                    yield json.dumps({"error": f"No GGUF or safetensors files found in '{repo_id}'"}) + "\n"
                    return

                # Delegate to the HF-model pull + auto-convert pipeline
                output_name = repo_id.split("/")[-1]
                dest_dir = OUTPUTS_DIR / output_name
                digest_name = f"{output_name} (safetensors \u2192 GGUF)"

                from huggingface_hub import snapshot_download

                # Calculate total download size using same filters as download
                _allow_exts = (".safetensors", ".json", ".txt", ".model", ".py")
                _ignore_exts = (".gguf", ".bin", ".msgpack", ".h5", ".ot", ".md")
                _ignore_names = {".gitattributes"}
                total_size = 0
                try:
                    repo_info = api.model_info(repo_id, files_metadata=True)
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
                except (AttributeError, TypeError) as e:
                    log.debug("Could not compute total size for HF download: %s", e)
                    total_size = 0

                if dest_dir.exists() and any(dest_dir.glob("*.safetensors")):
                    yield json.dumps({
                        "status": "downloading",
                        "digest": digest_name,
                        "completed": total_size,
                        "total": total_size,
                    }) + "\n"
                else:
                    download_error = []
                    download_done = threading.Event()

                    def _hf_download():
                        try:
                            snapshot_download(
                                repo_id=repo_id,
                                local_dir=str(dest_dir),
                                local_dir_use_symlinks=False,
                                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.py"],
                                ignore_patterns=["*.gguf", "*.bin", "*.msgpack", "*.h5", "*.ot", "*.md", ".gitattributes"],
                            )
                        except Exception as e:
                            download_error.append(str(e))
                        finally:
                            download_done.set()

                    thread = threading.Thread(target=_hf_download, daemon=True)
                    thread.start()

                    while not download_done.is_set():
                        current_size = 0
                        if dest_dir.exists():
                            for f in dest_dir.rglob("*"):
                                if f.is_file():
                                    try:
                                        current_size += f.stat().st_size
                                    except OSError:
                                        pass
                        if total_size > 0:
                            yield json.dumps({
                                "status": "downloading",
                                "digest": digest_name,
                                "completed": min(current_size, total_size),
                                "total": total_size,
                            }) + "\n"
                        else:
                            yield json.dumps({
                                "status": "downloading",
                                "digest": digest_name,
                                "completed": current_size,
                                "total": 0,
                            }) + "\n"
                        await asyncio.sleep(2)

                    if download_error:
                        yield json.dumps({"error": f"Download failed: {download_error[0]}"}) + "\n"
                        return

                # Auto-convert to GGUF (q8_0)
                outtype = "q8_0"
                out_name = f"{output_name}-{outtype}.gguf"
                outfile = MODELS_DIR / out_name

                if outfile.exists():
                    _record_model_meta(out_name, repo_id, None, "safetensors_converted", quant=outtype)
                    yield json.dumps({"status": "success"}) + "\n"
                    return

                # Stream conversion progress with digest so UI shows progress bar
                convert_digest = f"Converting \u2192 {out_name}"

                yield json.dumps({
                    "status": "converting",
                    "digest": convert_digest,
                    "completed": 0,
                    "total": 0,
                }) + "\n"

                convert_cmd = [
                    str(PYTHON), str(CONVERT_HF_TO_GGUF),
                    str(dest_dir),
                    "--outfile", str(outfile),
                    "--outtype", outtype,
                ]
                env = os.environ.copy()
                env["NO_LOCAL_GGUF"] = "1"

                convert_proc = await asyncio.create_subprocess_exec(
                    *convert_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
                async for raw_line in convert_proc.stdout:
                    yield json.dumps({
                        "status": "converting",
                        "digest": convert_digest,
                        "completed": 0,
                        "total": 0,
                        "log": raw_line.decode(errors="replace").rstrip(),
                    }) + "\n"
                await convert_proc.wait()

                if convert_proc.returncode != 0:
                    yield json.dumps({"error": f"GGUF conversion failed (exit code {convert_proc.returncode})"}) + "\n"
                    return

                # Restart llama-server to pick up the new model
                _restart_llama_server()

                _record_model_meta(out_name, repo_id, None, "safetensors_converted", quant=outtype)
                yield json.dumps({"status": "success"}) + "\n"
                return

            # Group GGUFs: detect split shard sets vs standalone files
            # Split shards look like: Model-00001-of-00004.gguf, Model-00002-of-00004.gguf
            shard_groups: Dict[str, List[str]] = {}  # base -> [files]
            standalone_gguf: List[str] = []

            for gf in gguf_files:
                m = _SPLIT_SHARD_RE.search(gf)
                if m:
                    base = gf[:gf.rfind("-", 0, m.start()) + 1] if "-" in gf[:m.start()] else gf[:m.start()]
                    # Use directory + base as key to group shards
                    parent = str(Path(gf).parent)
                    key = f"{parent}/{Path(gf).name[:m.start()]}"
                    shard_groups.setdefault(key, []).append(gf)
                else:
                    standalone_gguf.append(gf)

            # Build selectable options: standalone files + shard groups (by first shard)
            selectable = list(standalone_gguf)
            shard_first_to_group: Dict[str, List[str]] = {}
            for key, shards in shard_groups.items():
                shards.sort()
                first = shards[0]
                selectable.append(first)
                shard_first_to_group[first] = shards

            # If no filename specified and multiple options, return list for selection
            if not filename:
                if len(selectable) == 1:
                    selected = selectable[0]
                else:
                    yield json.dumps({"status": "select_file", "files": selectable}) + "\n"
                    return
            else:
                if filename not in gguf_files and filename not in selectable:
                    yield json.dumps({"error": f"File '{filename}' not found in repo. Available: {selectable}"}) + "\n"
                    return
                selected = filename

            # Determine all files to download (single file or all shards in group)
            files_to_download = shard_first_to_group.get(selected, [selected])
            is_split = len(files_to_download) > 1

            # Check if already exists
            dest_path = MODELS_DIR / Path(files_to_download[0]).name
            if dest_path.exists():
                yield json.dumps({"error": f"Model '{dest_path.name}' already exists"}) + "\n"
                return

            # Get total file size for progress tracking
            try:
                file_info = api.model_info(repo_id, files_metadata=True)
                total_size = 0
                sibling_sizes = {}
                for s in file_info.siblings:
                    size = getattr(s, 'size', None) or 0
                    if size == 0 and hasattr(s, 'lfs') and s.lfs:
                        size = s.lfs.get('size', 0) if isinstance(s.lfs, dict) else getattr(s.lfs, 'size', 0)
                    sibling_sizes[s.rfilename] = size
                for dl_file in files_to_download:
                    total_size += sibling_sizes.get(dl_file, 0)
            except (AttributeError, TypeError) as e:
                log.debug("Could not compute total size for GGUF download: %s", e)
                total_size = 0

            # Set up cancel event
            cancel_event = threading.Event()
            download_key = f"{repo_id}/{selected}"
            _active_downloads[download_key] = cancel_event

            download_error = []
            download_done = threading.Event()

            def download_task():
                try:
                    for dl_file in files_to_download:
                        if cancel_event.is_set():
                            return
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=dl_file,
                            local_dir=str(MODELS_DIR),
                            local_dir_use_symlinks=False,
                        )
                    download_done.set()
                except Exception as e:
                    download_error.append(str(e))
                    download_done.set()

            thread = threading.Thread(target=download_task, daemon=True)
            thread.start()

            status_label = Path(selected).name
            if is_split:
                status_label = f"{Path(selected).name} ({len(files_to_download)} shards)"
            yield json.dumps({"status": f"downloading {status_label}"}) + "\n"

            # Poll file size for progress
            while not download_done.is_set():
                if cancel_event.is_set():
                    # Clean up partial downloads
                    for dl_file in files_to_download:
                        for cleanup in [MODELS_DIR / Path(dl_file).name, MODELS_DIR / dl_file]:
                            if cleanup.exists():
                                cleanup.unlink()
                    yield json.dumps({"error": "Download cancelled"}) + "\n"
                    _active_downloads.pop(download_key, None)
                    return

                # Sum up downloaded bytes across all files
                current_size = 0
                seen_inodes = set()
                for dl_file in files_to_download:
                    dl_name = Path(dl_file).name
                    # Check completed file at top-level or in repo subdir
                    for check_path in [MODELS_DIR / dl_name, MODELS_DIR / dl_file]:
                        if check_path.exists():
                            try:
                                st = check_path.stat()
                                if st.st_ino not in seen_inodes:
                                    seen_inodes.add(st.st_ino)
                                    current_size += st.st_size
                            except OSError:
                                pass
                            break

                    # Check for .incomplete temp file for THIS specific file
                    if current_size == 0:
                        for incomplete_path in [
                            MODELS_DIR / f"{dl_name}.incomplete",
                            MODELS_DIR / dl_file / ".incomplete",
                            MODELS_DIR / f"{dl_file}.incomplete",
                        ]:
                            if incomplete_path.exists():
                                try:
                                    current_size += incomplete_path.stat().st_size
                                except OSError:
                                    pass
                                break

                # Last resort: check HF cache for this specific repo's incomplete files
                if current_size == 0:
                    # Respect HF_HOME env var (set in docker-compose as /workspace/hf-hub)
                    hf_home = os.environ.get("HF_HOME")
                    if hf_home:
                        hf_cache = Path(hf_home) / "hub"
                    else:
                        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
                    # HF cache uses a sanitized repo name
                    cache_repo_dir = hf_cache / ("models--" + repo_id.replace("/", "--"))
                    if cache_repo_dir.exists():
                        for incomplete in cache_repo_dir.rglob("*.incomplete"):
                            try:
                                current_size += incomplete.stat().st_size
                            except OSError:
                                pass

                # Always send progress updates so the UI shows something
                if total_size > 0:
                    yield json.dumps({
                        "status": "downloading",
                        "digest": Path(selected).name,
                        "completed": min(current_size, total_size),
                        "total": total_size,
                    }) + "\n"
                else:
                    # No total_size known — still report bytes downloaded
                    yield json.dumps({
                        "status": "downloading",
                        "digest": Path(selected).name,
                        "completed": current_size,
                        "total": 0,
                    }) + "\n"

                await asyncio.sleep(1)

            _active_downloads.pop(download_key, None)

            if download_error:
                yield json.dumps({"error": download_error[0]}) + "\n"
                return

            # Post-download: ensure files are accessible to llama-server
            # hf_hub_download may place files in subdirs matching repo structure
            for dl_file in files_to_download:
                final_name = Path(dl_file).name
                top_level = MODELS_DIR / final_name
                repo_subpath = MODELS_DIR / dl_file

                if not top_level.exists() and repo_subpath.exists() and repo_subpath != top_level:
                    if not is_split:
                        # Single file: move to top level
                        shutil.move(str(repo_subpath), str(top_level))
                    # For split shards: leave in subdir, we'll symlink the first shard below

            # For split models: symlink first shard to top level so llama-server discovers it
            if is_split:
                first_shard_name = Path(files_to_download[0]).name
                symlink_path = MODELS_DIR / first_shard_name
                actual_path = MODELS_DIR / files_to_download[0]

                if not symlink_path.exists() and actual_path.exists():
                    symlink_path.symlink_to(actual_path)
                    yield json.dumps({"status": f"registered split model ({len(files_to_download)} shards)"}) + "\n"

            # Clean up empty parent dirs from hf_hub_download
            if not is_split:
                for dl_file in files_to_download:
                    repo_subpath = MODELS_DIR / dl_file
                    parent = repo_subpath.parent
                    while parent != MODELS_DIR:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent

            # Verify success
            first_file = Path(files_to_download[0]).name
            check_path = MODELS_DIR / first_file
            if check_path.exists() or check_path.is_symlink():
                if is_split:
                    combined_size = sum(
                        (MODELS_DIR / f).stat().st_size
                        for f in files_to_download
                        if (MODELS_DIR / f).exists()
                    )
                    # Fall back to checking top-level if subdir paths don't exist
                    if combined_size == 0:
                        combined_size = check_path.stat().st_size
                    yield json.dumps({
                        "status": "downloading",
                        "digest": first_file,
                        "completed": combined_size,
                        "total": combined_size,
                    }) + "\n"
                else:
                    yield json.dumps({
                        "status": "downloading",
                        "digest": check_path.name,
                        "completed": check_path.stat().st_size,
                        "total": check_path.stat().st_size,
                    }) + "\n"
                # Restart llama-server so it discovers the new model
                _restart_llama_server()
                _record_model_meta(first_file, repo_id, selected, "gguf")
                yield json.dumps({"status": "success"}) + "\n"
            else:
                yield json.dumps({"error": "Download completed but file not found in models directory"}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/api/models/pull/cancel")
def cancel_pull(req: ModelPullRequest):
    """Cancel an active model download."""
    download_key = f"{req.name.strip()}/{req.filename or ''}"
    # Try exact match first, then prefix match
    event = _active_downloads.get(download_key)
    if not event:
        for key, evt in _active_downloads.items():
            if key.startswith(req.name.strip()):
                event = evt
                download_key = key
                break
    if not event:
        raise HTTPException(status_code=404, detail="No active download found")
    event.set()
    return {"status": "cancelling", "name": download_key}


@router.post("/api/models/delete")
def delete_gguf_model(req: ModelDeleteRequest):
    """Delete a GGUF model file from the models directory.
    If the file is part of a split GGUF set, all shards and symlinks are deleted."""
    raw_path = MODELS_DIR / req.name

    # Handle symlinks (resolve=False so we can check the link itself)
    if raw_path.is_symlink():
        # It's a symlink — delete it and the target shards if they exist
        target = raw_path.resolve()
        raw_path.unlink()
        deleted = [req.name]
        # If target was a split shard, also delete sibling shards
        if target.exists():
            m = _SPLIT_SHARD_RE.search(target.name)
            if m:
                base = target.name[:m.start()]
                for shard in target.parent.glob(f"{base}-*-of-*.gguf"):
                    if _SPLIT_SHARD_RE.search(shard.name):
                        shard.unlink()
                        deleted.append(str(shard))
        _remove_model_meta(Path(req.name).name)
        return {"deleted": True, "name": req.name, "files_removed": deleted}

    model_path = raw_path.resolve()
    # Path traversal protection
    if not str(model_path).startswith(str(MODELS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid model path")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    deleted = []
    m = _SPLIT_SHARD_RE.search(model_path.name)
    if m:
        # Delete all shards with the same base name in the same directory
        base = model_path.name[:m.start()]
        for shard in model_path.parent.glob(f"{base}-*-of-*.gguf"):
            if _SPLIT_SHARD_RE.search(shard.name):
                shard.unlink()
                deleted.append(shard.name)
        # Also remove any top-level symlinks pointing to deleted shards
        for link in MODELS_DIR.iterdir():
            if link.is_symlink() and link.name.endswith(".gguf"):
                try:
                    link_target = link.resolve()
                    if not link_target.exists():
                        link.unlink()
                        deleted.append(f"symlink:{link.name}")
                except OSError as e:
                    log.debug("Failed to clean up symlink %s: %s", link.name, e)
    else:
        model_path.unlink()
        deleted.append(req.name)

    # Restart llama-server so it rescans the models directory
    _restart_llama_server()

    _remove_model_meta(Path(req.name).name)
    return {"deleted": True, "name": req.name, "files_removed": deleted}


# ─── HF Cache Endpoints ────────────────────────────────────────────────

def _is_repo_cached(repo_id: str, cache_dir: Path) -> bool:
    """Return True if repo_id already has a complete snapshot in the HF cache."""
    from huggingface_hub import scan_cache_dir
    try:
        cache_info = scan_cache_dir(cache_dir=str(cache_dir))
        for repo in cache_info.repos:
            if repo.repo_id == repo_id and repo.nb_snapshots > 0:
                return True
    except (OSError, ValueError) as e:
        log.debug("Failed to scan HF cache for %s: %s", repo_id, e)
    return False


@router.post("/api/models/hf-cache/ensure")
async def ensure_hf_cache(req: HfCacheEnsureRequest):
    """Download HuggingFace model repos into the shared HF cache so the
    curator container can load them with local_files_only=True.

    Streams NDJSON progress. Skips repos that are already cached.
    """
    from huggingface_hub import snapshot_download

    async def generate():
        for repo_id in req.repos:
            if _is_repo_cached(repo_id, HF_CACHE_DIR):
                yield json.dumps({"repo_id": repo_id, "status": "cached"}) + "\n"
                continue

            yield json.dumps({"repo_id": repo_id, "status": "downloading"}) + "\n"

            error_holder: list[str] = []
            done_event = threading.Event()

            def download_task():
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=str(HF_CACHE_DIR),
                        local_files_only=False,
                    )
                except Exception as e:
                    error_holder.append(str(e))
                finally:
                    done_event.set()

            t = threading.Thread(target=download_task, daemon=True)
            t.start()

            while not done_event.wait(timeout=2):
                yield json.dumps({"repo_id": repo_id, "status": "downloading"}) + "\n"
                await asyncio.sleep(0)

            if error_holder:
                yield json.dumps({"repo_id": repo_id, "status": "error", "error": error_holder[0]}) + "\n"
            else:
                yield json.dumps({"repo_id": repo_id, "status": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/api/models/hf-cache/status")
async def hf_cache_status():
    """Return cached/missing status for all curator classifier repos."""
    result = []
    for repo_id in CURATOR_CLASSIFIER_REPOS:
        result.append({
            "repo_id": repo_id,
            "cached": _is_repo_cached(repo_id, HF_CACHE_DIR),
        })
    return {"repos": result}


# ─── FastText Model Endpoints ────────────────────────────────────────

@router.get("/api/models/fasttext/status")
async def fasttext_model_status():
    """Return cached/missing status for the known FastText models."""
    result = []
    for model in CURATOR_FASTTEXT_MODELS:
        dest = CURATOR_FASTTEXT_DIR / model["name"]
        result.append({
            "name": model["name"],
            "description": model["description"],
            "cached": dest.exists(),
            "curator_path": model["curator_path"],
        })
    return {"models": result}


@router.post("/api/models/fasttext/ensure")
async def ensure_fasttext_models():
    """Download any missing FastText models into the shared HF cache volume.

    Streams NDJSON progress. Skips models that are already downloaded.
    """
    import requests as _requests

    async def generate():
        CURATOR_FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
        for model in CURATOR_FASTTEXT_MODELS:
            dest = CURATOR_FASTTEXT_DIR / model["name"]
            if dest.exists():
                yield json.dumps({"name": model["name"], "status": "cached"}) + "\n"
                continue

            yield json.dumps({"name": model["name"], "status": "downloading"}) + "\n"
            await asyncio.sleep(0)

            error_holder: list[str] = []
            done_event = threading.Event()

            def download_task(url=model["url"], dest_path=dest):
                try:
                    response = _requests.get(url, stream=True, timeout=300)
                    response.raise_for_status()
                    tmp = dest_path.with_suffix(dest_path.suffix + ".downloading")
                    with open(tmp, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    tmp.rename(dest_path)
                except Exception as e:
                    error_holder.append(str(e))
                finally:
                    done_event.set()

            t = threading.Thread(target=download_task, daemon=True)
            t.start()

            while not done_event.wait(timeout=2):
                yield json.dumps({"name": model["name"], "status": "downloading"}) + "\n"
                await asyncio.sleep(0)

            if error_holder:
                yield json.dumps({"name": model["name"], "status": "error", "error": error_holder[0]}) + "\n"
            else:
                yield json.dumps({"name": model["name"], "status": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
