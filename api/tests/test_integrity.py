"""
Tests for the periodic /models integrity sweep (self.llamolotl#23).

Covers: missing-file detection, corrupted-header detection, orphaned-file
detection, a clean sweep producing no warnings, size-mismatch detection,
the narrow autopull scope decision, and an actual re-pull for a
previously-successful, now-missing model.
"""

import struct
import threading
from unittest.mock import MagicMock, patch

import pytest

import api.integrity as integrity


# ─── Helpers ─────────────────────────────────────────────────────────────

def _valid_gguf_bytes(version: int = 3) -> bytes:
    """Minimal bytes that pass the header check: magic + version + padding
    (real files have kv/tensor counts and more after this, but the sweep's
    header check only reads the first 8 bytes)."""
    return b"GGUF" + struct.pack("<I", version) + b"\x00" * 64


def _write_gguf(path, version: int = 3, content: bytes = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else _valid_gguf_bytes(version))


class _SyncThread:
    """Stand-in for threading.Thread that runs its target synchronously on
    .start(), so autopull tests don't need to coordinate with a real
    background thread."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


# ─── Header check ────────────────────────────────────────────────────────

class TestGgufHeaderCheck:
    def test_valid_header_passes(self, tmp_path):
        f = tmp_path / "model.gguf"
        _write_gguf(f)
        assert integrity._check_gguf_header(f) is None

    def test_bad_magic_fails(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"NOTG" + struct.pack("<I", 3) + b"\x00" * 32)
        err = integrity._check_gguf_header(f)
        assert err is not None
        assert "magic" in err.lower()

    def test_truncated_file_fails(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GG")
        err = integrity._check_gguf_header(f)
        assert err is not None
        assert "short" in err.lower() or "truncated" in err.lower()

    def test_unrecognized_version_fails(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF" + struct.pack("<I", 99) + b"\x00" * 32)
        err = integrity._check_gguf_header(f)
        assert err is not None
        assert "version" in err.lower()


# ─── Sweep: missing / orphaned / corrupt / clean / size mismatch ────────

class TestSweepOnce:
    def test_clean_sweep_produces_no_warnings(self, client, patched_state):
        state = patched_state
        model_path = state.MODELS_DIR / "good-model-Q4_K_M.gguf"
        _write_gguf(model_path)
        state._record_model_meta(
            "good-model-Q4_K_M.gguf", "org/good-model", "good-model-Q4_K_M.gguf",
            "gguf", size_bytes=model_path.stat().st_size,
        )

        findings = integrity.sweep_once()

        assert findings == []
        assert integrity.get_last_sweep()["warnings"] == []
        assert integrity.get_last_sweep()["last_sweep_at"] is not None

    def test_missing_file_detected(self, client, patched_state):
        state = patched_state
        # Meta entry recorded, but no backing file was ever written.
        state._record_model_meta(
            "ghost-model-Q4_K_M.gguf", "org/ghost-model", "ghost-model-Q4_K_M.gguf", "gguf",
        )

        findings = integrity.sweep_once()

        assert len(findings) == 1
        assert findings[0]["filename"] == "ghost-model-Q4_K_M.gguf"
        assert findings[0]["issue"] == "missing"

    def test_orphaned_file_detected(self, client, patched_state):
        state = patched_state
        # A GGUF sitting in /models with no models_meta.json entry at all --
        # this is the class self.ai's HTTP-only sweep structurally cannot
        # see (it only knows about models llama-server already reports).
        model_path = state.MODELS_DIR / "manually-dropped.gguf"
        _write_gguf(model_path)

        findings = integrity.sweep_once()

        assert len(findings) == 1
        assert findings[0]["filename"] == "manually-dropped.gguf"
        assert findings[0]["issue"] == "orphaned"
        assert findings[0]["size_bytes"] == model_path.stat().st_size

    def test_corrupted_file_detected(self, client, patched_state):
        state = patched_state
        model_path = state.MODELS_DIR / "corrupt-model-Q4_K_M.gguf"
        # Bad header: wrong magic bytes.
        _write_gguf(model_path, content=b"BADX" + struct.pack("<I", 3) + b"\x00" * 32)
        state._record_model_meta(
            "corrupt-model-Q4_K_M.gguf", "org/corrupt-model",
            "corrupt-model-Q4_K_M.gguf", "gguf",
        )

        findings = integrity.sweep_once()

        assert len(findings) == 1
        assert findings[0]["filename"] == "corrupt-model-Q4_K_M.gguf"
        assert findings[0]["issue"] == "corrupt_header"
        assert "magic" in findings[0]["detail"].lower()

    def test_size_mismatch_detected(self, client, patched_state):
        state = patched_state
        model_path = state.MODELS_DIR / "resized-model-Q4_K_M.gguf"
        _write_gguf(model_path)
        actual_size = model_path.stat().st_size
        # Record an expected size that doesn't match what's on disk now.
        state._record_model_meta(
            "resized-model-Q4_K_M.gguf", "org/resized-model",
            "resized-model-Q4_K_M.gguf", "gguf", size_bytes=actual_size + 12345,
        )

        findings = integrity.sweep_once()

        assert len(findings) == 1
        assert findings[0]["issue"] == "size_mismatch"
        assert findings[0]["size_bytes"] == actual_size
        assert findings[0]["expected_size_bytes"] == actual_size + 12345

    def test_meta_entries_without_size_bytes_skip_size_check(self, client, patched_state):
        """Most existing meta entries predate size_bytes (bake/lora-convert
        paths still don't record it -- see state._record_model_meta's
        docstring). A meta entry with no size_bytes field must not be
        flagged just because it lacks the field."""
        state = patched_state
        model_path = state.MODELS_DIR / "baked-model-Q4_K_M.gguf"
        _write_gguf(model_path)
        state._record_model_meta(
            "baked-model-Q4_K_M.gguf", "org/base-model", None, "baked",
        )

        findings = integrity.sweep_once()

        assert findings == []

    def test_split_shards_grouped_and_checked_individually(self, client, patched_state):
        """A split GGUF set is one logical model for missing/orphan
        purposes (keyed by the first shard), but every shard still gets
        its own header check."""
        state = patched_state
        shard1 = state.MODELS_DIR / "Big-Model-00001-of-00002.gguf"
        shard2 = state.MODELS_DIR / "Big-Model-00002-of-00002.gguf"
        _write_gguf(shard1)
        # Second shard is corrupt.
        _write_gguf(shard2, content=b"NOPE" + struct.pack("<I", 3))
        state._record_model_meta(
            "Big-Model-00001-of-00002.gguf", "org/big-model",
            "Big-Model-00001-of-00002.gguf", "gguf",
        )

        findings = integrity.sweep_once()

        assert len(findings) == 1
        assert findings[0]["filename"] == "Big-Model-00001-of-00002.gguf"
        assert findings[0]["issue"] == "corrupt_header"
        assert "00002-of-00002" in findings[0]["detail"]

    def test_cache_dir_and_downloading_files_excluded_from_scan(self, client, patched_state):
        state = patched_state
        cache_dir = state.MODELS_DIR / ".cache" / "leftover.gguf"
        _write_gguf(cache_dir)
        partial = state.MODELS_DIR / "in-progress.gguf.downloading"
        _write_gguf(partial)

        findings = integrity.sweep_once()

        assert findings == []


# ─── GET / POST endpoints ─────────────────────────────────────────────

class TestIntegrityEndpoints:
    def test_get_integrity_returns_cached_findings(self, client, patched_state):
        state = patched_state
        state._record_model_meta("ghost.gguf", "org/ghost", "ghost.gguf", "gguf")
        integrity.sweep_once()

        resp = client.get("/api/integrity")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["warnings"]) == 1
        assert data["warnings"][0]["filename"] == "ghost.gguf"
        assert data["last_sweep_at"] is not None
        assert "sweep_enabled" in data
        assert "autopull_enabled" in data

    def test_get_integrity_empty_before_first_sweep(self, client, patched_state):
        resp = client.get("/api/integrity")
        assert resp.status_code == 200
        assert resp.json()["warnings"] == []
        assert resp.json()["last_sweep_at"] is None

    def test_post_sweep_triggers_immediate_scan(self, client, patched_state):
        state = patched_state
        model_path = state.MODELS_DIR / "on-disk-only.gguf"
        _write_gguf(model_path)

        resp = client.post("/api/integrity/sweep")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["warnings"]) == 1
        assert data["warnings"][0]["issue"] == "orphaned"


# ─── Autopull scope decision ────────────────────────────────────────────

class TestAutopullScope:
    def test_not_autopullable_when_disabled(self, patched_state):
        entry = {"source_type": "gguf", "hf_repo": "org/m", "hf_filename": "m-Q4_K_M.gguf"}
        with patch.object(integrity, "AUTOPULL_ENABLED", False):
            assert integrity._autopullable(entry) is False

    def test_not_autopullable_for_baked_source(self, patched_state):
        entry = {"source_type": "baked", "hf_repo": "org/base", "hf_filename": None}
        with patch.object(integrity, "AUTOPULL_ENABLED", True):
            assert integrity._autopullable(entry) is False

    def test_not_autopullable_without_hf_filename(self, patched_state):
        entry = {"source_type": "gguf", "hf_repo": "org/m", "hf_filename": None}
        with patch.object(integrity, "AUTOPULL_ENABLED", True):
            assert integrity._autopullable(entry) is False

    def test_not_autopullable_for_split_shard(self, patched_state):
        entry = {
            "source_type": "gguf", "hf_repo": "org/big",
            "hf_filename": "Big-Model-00001-of-00004.gguf",
        }
        with patch.object(integrity, "AUTOPULL_ENABLED", True):
            assert integrity._autopullable(entry) is False

    def test_autopullable_for_previously_successful_single_file_gguf_pull(self, patched_state):
        entry = {"source_type": "gguf", "hf_repo": "org/m", "hf_filename": "m-Q4_K_M.gguf"}
        with patch.object(integrity, "AUTOPULL_ENABLED", True):
            assert integrity._autopullable(entry) is True

    def test_orphans_never_trigger_autopull(self, client, patched_state):
        """An orphaned file has no meta entry, so there is nothing to
        autopull from -- sweep_once must never call _maybe_autopull for
        the orphaned branch."""
        state = patched_state
        model_path = state.MODELS_DIR / "orphan.gguf"
        _write_gguf(model_path)

        with patch.object(integrity, "AUTOPULL_ENABLED", True), \
             patch.object(integrity, "_maybe_autopull") as mock_autopull:
            integrity.sweep_once()

        mock_autopull.assert_not_called()


# ─── Autopull execution (narrow re-pull case) ───────────────────────────

class TestAutopullExecution:
    def test_repull_triggers_for_previously_successful_now_missing_model(
        self, client, patched_state, monkeypatch
    ):
        """The narrow, safe autopull case: models_meta.json shows this
        exact file was pulled successfully before (hf_repo + hf_filename
        recorded, source_type == 'gguf', not a split shard), and the file
        is now missing. sweep_once should kick off a re-pull that
        recreates the file and refreshes models_meta.json."""
        state = patched_state
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        state._record_model_meta(
            "restored-model-Q4_K_M.gguf", "org/restored-model",
            "restored-model-Q4_K_M.gguf", "gguf", size_bytes=999,
        )
        assert not (state.MODELS_DIR / "restored-model-Q4_K_M.gguf").exists()

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks):
            # Simulate huggingface_hub writing the file into local_dir.
            dest = state.MODELS_DIR / filename
            _write_gguf(dest)
            return str(dest)

        with patch.object(integrity, "AUTOPULL_ENABLED", True), \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download), \
             patch.object(state, "_restart_llama_server", return_value={"status": "restarted"}):
            findings = integrity.sweep_once()

        assert len(findings) == 1
        assert findings[0]["issue"] == "missing"
        assert findings[0]["autopull"] == "started"

        restored = state.MODELS_DIR / "restored-model-Q4_K_M.gguf"
        assert restored.exists()
        meta = state._load_models_meta()
        assert meta["restored-model-Q4_K_M.gguf"]["size_bytes"] == restored.stat().st_size
        assert "restored-model-Q4_K_M.gguf" not in integrity._autopull_in_progress

    def test_repull_triggers_for_corrupt_header(self, client, patched_state, monkeypatch):
        """The other half of the narrow autopull case: a previously-
        successful pull whose file is now corrupt (bad header), not just
        absent."""
        state = patched_state
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        bad_path = state.MODELS_DIR / "flaky-model-Q4_K_M.gguf"
        _write_gguf(bad_path, content=b"NOPE" + struct.pack("<I", 3) + b"\x00" * 32)
        state._record_model_meta(
            "flaky-model-Q4_K_M.gguf", "org/flaky-model",
            "flaky-model-Q4_K_M.gguf", "gguf",
        )

        def fake_hf_hub_download(repo_id, filename, local_dir, local_dir_use_symlinks):
            dest = state.MODELS_DIR / filename
            _write_gguf(dest)
            return str(dest)

        with patch.object(integrity, "AUTOPULL_ENABLED", True), \
             patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download), \
             patch.object(state, "_restart_llama_server", return_value={"status": "restarted"}):
            findings = integrity.sweep_once()

        assert findings[0]["issue"] == "corrupt_header"
        assert findings[0]["autopull"] == "started"
        assert integrity._check_gguf_header(bad_path) is None  # file was replaced, now valid

    def test_repull_skipped_when_manual_pull_already_in_progress(
        self, client, patched_state, monkeypatch
    ):
        state = patched_state
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        state._record_model_meta(
            "busy-model-Q4_K_M.gguf", "org/busy-model",
            "busy-model-Q4_K_M.gguf", "gguf",
        )
        state._active_downloads["org/busy-model/busy-model-Q4_K_M.gguf"] = threading.Event()

        with patch.object(integrity, "AUTOPULL_ENABLED", True), \
             patch("huggingface_hub.hf_hub_download") as mock_download:
            findings = integrity.sweep_once()

        assert findings[0]["autopull"] == "skipped_manual_pull_in_progress"
        mock_download.assert_not_called()

        state._active_downloads.clear()

    def test_autopull_failure_is_logged_and_does_not_raise(self, client, patched_state, monkeypatch):
        state = patched_state
        monkeypatch.setattr(threading, "Thread", _SyncThread)
        state._record_model_meta(
            "unreachable-model-Q4_K_M.gguf", "org/unreachable-model",
            "unreachable-model-Q4_K_M.gguf", "gguf",
        )

        with patch.object(integrity, "AUTOPULL_ENABLED", True), \
             patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("network down")):
            findings = integrity.sweep_once()  # must not raise

        assert findings[0]["autopull"] == "started"
        assert "unreachable-model-Q4_K_M.gguf" not in integrity._autopull_in_progress
        assert not (state.MODELS_DIR / "unreachable-model-Q4_K_M.gguf").exists()
