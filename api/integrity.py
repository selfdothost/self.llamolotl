"""Periodic /models integrity sweep (self.llamolotl#23 / self.ai#38).

self.ai's API server has no filesystem access to the /models PVC --
self.llamolotl owns that volume directly. self.ai!126 landed the only fix
possible from that side: an hourly HTTP-only sweep cross-referencing
`GET {base_url}/v1/models` (what llama-server currently reports as known)
against this service's `GET /api/models/available` (what's on disk, by
name and size). Its own commit message named that as a stopgap and said
the real fix belongs here.

This module is that real fix. It walks the actual /models directory and
models_meta.json on disk -- no HTTP round-trip, and no dependence on
llama-server already having surfaced a model through /v1/models -- so it
additionally catches:

  - Orphaned files: present on disk with no models_meta.json entry at
    all. self.ai's HTTP-only view has no way to see these; it only knows
    about models llama-server already reports.
  - Real byte-level corruption: a GGUF magic-number + version header
    check, not just "is the file smaller than some size floor". No
    checksum/hash is recorded anywhere in this codebase for GGUFs
    (models_meta.json has hf_repo/hf_filename/quant/pulled_at, never a
    digest), so a full digest comparison isn't available -- the header
    check is the cheap, real signal available without inventing a new
    hashing scheme wholesale.
  - Size mismatch against a recorded expected size, when one exists (see
    state._record_model_meta's size_bytes field, added alongside this
    sweep -- most existing meta entries predate it and simply won't have
    the field, so this check is opportunistic, not universal).
  - Missing files: a meta entry with no backing file on disk -- the same
    class self.ai's sweep caught, done here without the HTTP round-trip.

This supersedes self.llamolotl's half of issue #23. self.ai!126 is left
untouched (cross-repo, out of scope here) but is now the strictly
smaller/coarser of the two sweeps; whether it should eventually be
retired in favor of this one is a follow-up observation for self.ai's own
side, not something decided or implemented in this change.
"""

import asyncio
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from . import state

log = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────

GGUF_MAGIC = b"GGUF"
_GGUF_SUPPORTED_VERSIONS = (1, 2, 3)  # mirrors self.llama/gguf-py/gguf/gguf_reader.py

SWEEP_ENABLED = os.environ.get("ENABLE_MODEL_INTEGRITY_SWEEP", "true").lower() == "true"
# Hourly default -- matches self.ai#38's HTTP-only sweep cadence.
SWEEP_INTERVAL_SECONDS = int(os.environ.get("MODEL_INTEGRITY_SWEEP_INTERVAL", str(60 * 60)))

# Off by default: same caution as self.ai's side, which declined to
# implement autopull at all ("shouldn't happen silently without explicit
# opt-in"). This is narrower and safer than self.ai could ever attempt --
# self.llamolotl owns the pull mechanism directly, so it can distinguish a
# real re-pull (models_meta.json shows this exact file pulled successfully
# before) from a first-time pull (no meta entry -- nothing to autopull
# from) -- but it's still opt-in, not on by default.
AUTOPULL_ENABLED = os.environ.get("ENABLE_MODEL_INTEGRITY_AUTOPULL", "false").lower() == "true"

# ─── In-memory findings (surfaced via GET /api/integrity) ──────────────

_warnings: List[Dict[str, Any]] = []
_last_sweep_at: Optional[str] = None
_autopull_in_progress: Set[str] = set()


# ─── Corruption check ───────────────────────────────────────────────────

def _check_gguf_header(path: Path) -> Optional[str]:
    """Read just the 8-byte GGUF header (4-byte magic + 4-byte version).
    Returns None if it looks valid, or a short description of what's wrong.

    This is deliberately cheap (one 8-byte read, not a full parse) so it's
    safe to run against every file on every sweep cycle regardless of
    model size. It catches a truncated/zero-byte file, a non-GGUF file
    dropped in /models, or a GGUF version this build's llama.cpp can't
    read -- it will NOT catch corruption confined to tensor data past the
    header, since there's no recorded reference digest to check tensor
    bytes against (see module docstring).
    """
    try:
        with open(path, "rb") as f:
            header = f.read(8)
    except OSError as e:
        return f"could not read file: {e}"
    if len(header) < 8:
        return "file too short for a GGUF header (truncated)"
    magic = header[:4]
    if magic != GGUF_MAGIC:
        return f"bad GGUF magic {magic!r} (expected {GGUF_MAGIC!r})"
    version = int.from_bytes(header[4:8], "little")
    if version not in _GGUF_SUPPORTED_VERSIONS:
        return f"unrecognized GGUF version {version}"
    return None


# ─── Disk walk ──────────────────────────────────────────────────────────

def _scan_disk() -> Dict[str, List[Path]]:
    """Real (non-symlink) *.gguf files under MODELS_DIR, grouped by their
    models_meta.json identity.

    Split shard sets are grouped under the first shard's basename, which
    is how _record_model_meta keys a split pull (routers/models.py
    pull_model records `first_file = Path(files_to_download[0]).name`).
    Standalone files are grouped under their own basename. Excludes
    .cache/, in-progress *.downloading files, and symlinks -- a symlink
    only ever points at a real file already covered by this walk (e.g. a
    registered split model's top-level symlink to its first shard in a
    subdirectory), so following it would double-count that file under two
    different "on disk" identities.
    """
    groups: Dict[str, List[Path]] = {}
    if not state.MODELS_DIR.exists():
        return groups

    all_files = []
    for f in sorted(state.MODELS_DIR.rglob("*.gguf")):
        if f.is_symlink():
            continue
        if f.name.endswith(".downloading"):
            continue
        try:
            f.relative_to(state.MODELS_DIR / ".cache")
            continue
        except ValueError:
            pass
        all_files.append(f)

    shard_sets: Dict[str, List[Path]] = {}
    for f in all_files:
        m = state._SPLIT_SHARD_RE.search(f.name)
        if m:
            base = f.name[: m.start()]
            key = str(f.parent / base)
            shard_sets.setdefault(key, []).append(f)
        else:
            groups[f.name] = [f]

    for shards in shard_sets.values():
        shards.sort(key=lambda p: p.name)
        groups[shards[0].name] = shards

    return groups


# ─── Autopull (narrow, opt-in re-pull of a previously-successful pull) ──

def _autopullable(entry: Dict[str, Any]) -> bool:
    """Narrow, safe re-pull scope: a single-file (non-split-shard) direct
    GGUF pull that models_meta.json shows completed successfully before
    (source_type == "gguf", with hf_repo + hf_filename recorded).

    Deliberately excludes:
      - safetensors_converted / baked / lora_gguf sources -- re-deriving
        those means re-running a conversion or training pipeline, not a
        single hf_hub_download call. Out of scope for a *safe* autopull;
        a wrong or half-finished re-conversion is worse than no autopull.
      - split-shard pulls -- models_meta.json only records the first
        shard's repo-relative path (used to parse the quant string out of
        the filename), not the full shard list. A correct re-pull would
        need to re-list the repo and re-derive the shard group; deferred
        rather than risking a partial or mismatched re-pull.
      - a preset-only / never-successfully-pulled model -- by
        construction this function is only ever consulted for entries
        that already exist in models_meta.json (see sweep_once: the
        "orphaned" branch, for files with no meta entry, never calls
        this). A model with nothing recorded has nothing to autopull
        from, same ceiling self.ai!126 hit from its side.
    """
    if not AUTOPULL_ENABLED:
        return False
    if entry.get("source_type") != "gguf":
        return False
    hf_repo = entry.get("hf_repo")
    hf_filename = entry.get("hf_filename")
    if not hf_repo or not hf_filename:
        return False
    if state._SPLIT_SHARD_RE.search(Path(hf_filename).name):
        return False
    return True


def _maybe_autopull(filename: str, entry: Dict[str, Any], issue: Dict[str, Any]) -> None:
    """Kick off a background re-pull if this issue is in the narrow
    autopullable scope. Annotates `issue` in place with what happened so
    GET /api/integrity reflects it immediately, without waiting on the
    re-pull (which can take a while) to finish."""
    if not _autopullable(entry):
        return
    if filename in _autopull_in_progress:
        issue["autopull"] = "already_in_progress"
        return
    download_key = f"{entry['hf_repo']}/{entry['hf_filename']}"
    if download_key in state._active_downloads:
        # A manual /api/models/pull for this exact file is already
        # running -- don't start a second, colliding download.
        issue["autopull"] = "skipped_manual_pull_in_progress"
        return

    issue["autopull"] = "started"
    _autopull_in_progress.add(filename)
    thread = threading.Thread(target=_run_autopull, args=(filename, dict(entry)), daemon=True)
    thread.start()


def _run_autopull(filename: str, entry: Dict[str, Any]) -> None:
    """Re-download a previously-successful pull's file from HuggingFace.

    Runs in a background thread (like routers/models.py's own pull/HF
    cache download tasks) so a slow re-download can't stall the async
    sweep loop or block request handling on the shared event loop.
    """
    from huggingface_hub import hf_hub_download

    hf_repo = entry["hf_repo"]
    hf_filename = entry["hf_filename"]
    dest_path = state.MODELS_DIR / filename

    log.warning(
        "Model integrity autopull: re-pulling '%s' from %s (file=%s)",
        filename, hf_repo, hf_filename,
    )
    try:
        # Clear out whatever's there (missing is a no-op; corrupt/wrong-size
        # needs removing first so hf_hub_download doesn't mistake a bad
        # local file for an already-complete one).
        if dest_path.exists() or dest_path.is_symlink():
            dest_path.unlink()

        hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            local_dir=str(state.MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        # hf_hub_download preserves any repo-relative subdirectory in
        # hf_filename; move it to the top level to match where the
        # original successful pull (and models_meta.json's key) put it.
        final_path = state.MODELS_DIR / hf_filename
        top_level = state.MODELS_DIR / filename
        if final_path != top_level and final_path.exists():
            top_level.parent.mkdir(parents=True, exist_ok=True)
            final_path.replace(top_level)
            # Clean up now-empty parent dirs left behind by hf_hub_download.
            parent = final_path.parent
            while parent != state.MODELS_DIR:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

        size_bytes = top_level.stat().st_size if top_level.exists() else None
        state._record_model_meta(
            filename, hf_repo, hf_filename, entry.get("source_type", "gguf"),
            quant=entry.get("quant"), size_bytes=size_bytes,
        )
        state._restart_llama_server()
        log.warning("Model integrity autopull: '%s' re-pulled successfully (%s bytes)", filename, size_bytes)
    except Exception as e:
        log.error("Model integrity autopull failed for '%s': %s", filename, e)
    finally:
        _autopull_in_progress.discard(filename)


# ─── Sweep ──────────────────────────────────────────────────────────────

def sweep_once() -> List[Dict[str, Any]]:
    """Run one integrity sweep. Cross-references the real /models
    directory against models_meta.json:

      - missing:        meta entry, no backing file on disk.
      - orphaned:        file on disk, no meta entry at all.
      - corrupt_header:  file on disk fails the GGUF magic/version check.
      - size_mismatch:   file on disk has a meta-recorded expected size
                          that doesn't match its current size.

    Findings are stashed on module state for GET /api/integrity and also
    returned directly, so a forced rescan (POST /api/integrity/sweep) can
    hand results back immediately instead of waiting on the next
    periodic cycle to populate them.
    """
    global _warnings, _last_sweep_at

    meta = state._load_models_meta()
    on_disk = _scan_disk()

    findings: List[Dict[str, Any]] = []

    # Missing: meta says a model exists, nothing backs it on disk.
    for filename, entry in meta.items():
        if filename in on_disk:
            continue
        issue = {
            "filename": filename,
            "issue": "missing",
            "detail": (
                f"'{filename}' is recorded in models_meta.json but no "
                f"backing file exists under {state.MODELS_DIR}"
            ),
            "hf_repo": entry.get("hf_repo"),
            "source_type": entry.get("source_type"),
        }
        findings.append(issue)
        _maybe_autopull(filename, entry, issue)

    # Orphaned + corruption + size mismatch: walk what's actually on disk.
    for filename, shards in on_disk.items():
        entry = meta.get(filename)
        total_size = sum(s.stat().st_size for s in shards if s.exists())

        if entry is None:
            findings.append({
                "filename": filename,
                "issue": "orphaned",
                "detail": (
                    f"'{filename}' exists under {state.MODELS_DIR} but has "
                    "no models_meta.json entry"
                ),
                "size_bytes": total_size,
            })
            # No autopull for orphans: there's no recorded hf_repo/
            # hf_filename to re-pull from. An orphan already has a file
            # (possibly perfectly fine) -- it's the bookkeeping that's
            # missing, not the model.
            continue

        bad_shard_err = None
        for shard in shards:
            err = _check_gguf_header(shard)
            if err:
                bad_shard_err = f"'{shard.name}' failed GGUF header validation: {err}"
                break

        if bad_shard_err:
            issue = {
                "filename": filename,
                "issue": "corrupt_header",
                "detail": bad_shard_err,
                "size_bytes": total_size,
                "hf_repo": entry.get("hf_repo"),
                "source_type": entry.get("source_type"),
            }
            findings.append(issue)
            _maybe_autopull(filename, entry, issue)
            continue

        expected_size = entry.get("size_bytes")
        if expected_size is not None and expected_size != total_size:
            issue = {
                "filename": filename,
                "issue": "size_mismatch",
                "detail": (
                    f"'{filename}' is {total_size} bytes on disk but "
                    f"models_meta.json recorded {expected_size} bytes at pull time"
                ),
                "size_bytes": total_size,
                "expected_size_bytes": expected_size,
                "hf_repo": entry.get("hf_repo"),
                "source_type": entry.get("source_type"),
            }
            findings.append(issue)
            _maybe_autopull(filename, entry, issue)

    for issue in findings:
        log.warning(
            "Model integrity sweep: %s [%s] %s",
            issue["filename"], issue["issue"], issue["detail"],
        )

    _warnings = findings
    _last_sweep_at = datetime.now().isoformat()
    return findings


async def run_periodic_sweep() -> None:
    """Background task: mirrors state._poll_jobs()'s while-True/try-except/
    sleep shape, but on its own much longer interval (hourly default) so
    walking a potentially large /models tree and hashing GGUF headers
    doesn't add per-tick overhead to the 5s job-poll loop. sweep_once()
    does blocking file I/O, so it's run via asyncio.to_thread rather than
    directly on the event loop.
    """
    if not SWEEP_ENABLED:
        log.info("Model integrity sweep disabled (ENABLE_MODEL_INTEGRITY_SWEEP=false).")
        return

    while True:
        try:
            await asyncio.to_thread(sweep_once)
        except Exception as e:
            log.error("Model integrity sweep cycle failed: %s", e, exc_info=True)
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


def get_last_sweep() -> Dict[str, Any]:
    """Cached findings from the most recent sweep cycle, for GET
    /api/integrity. Empty until the first sweep completes, or permanently
    empty if the sweep is disabled."""
    return {
        "warnings": list(_warnings),
        "last_sweep_at": _last_sweep_at,
        "sweep_enabled": SWEEP_ENABLED,
        "autopull_enabled": AUTOPULL_ENABLED,
    }
