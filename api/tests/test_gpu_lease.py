"""Unit tests for the GPU-lease-client capability (cavekit gpu-lease-client
R1/R2).

Tier-0 primitives (patching subprocess / urllib / the router probe, the same
style as test_watchdog.py): the live VRAM-state probe, the router unload
helper, the LRU eviction selector, and the wire-format Pydantic models.

Tier-1/2 wiring: the release orchestrator `_handle_vram_release` (T-007,
single-flight / timeout-bounded / live-verified) as a direct unit test, and
the two endpoints `GET /api/system/vram-state` (T-003) and
`POST /api/system/vram-release` (T-008) via the FastAPI `TestClient` `client`
fixture from conftest.py.
"""

from unittest.mock import MagicMock, patch

import pytest

import api.state as state


@pytest.fixture(autouse=True)
def _reset_lru():
    """Keep the module-level LRU maps isolated between tests."""
    state._llama_model_loaded_since.clear()
    yield
    state._llama_model_loaded_since.clear()


def _smi(used_mib, total_mib):
    """Build a mock nvidia-smi CompletedProcess."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = f"{used_mib}, {total_mib}\n"
    return m


# ─── T-001: _probe_vram_state ────────────────────────────────────────────

class TestProbeVramState:
    def test_models_resident_reports_our_held_not_the_whole_card(self):
        """(a) models resident → held is OUR summed model footprint, and the
        whole-card figure is reported separately as device_used_bytes.

        The two numbers are deliberately different here (3000 MiB vs 4096 MiB):
        core SUMS held across consumers, so reporting the card in that field
        double-counted every sibling (self.ai#74).

        The footprint is supplied the way the router really supplies it — as a
        published `memory[]` payload on the model entry (self.llamolotl!41 for
        #36) — not by stubbing an aggregate. This test used to patch
        `_lookup_model_size_bytes`, which `_probe_vram_state()` stopped
        consulting when the figure moved to the published measurement. The
        patch then aimed at a function no longer on the path, held correctly
        came back None ("no measured footprint, report unknown rather than
        understate it"), and the assertion failed. Two devices with all three
        keys populated, so this exercises the real cross-device sum in
        `_router_resident_footprints()` instead of trading one stub for
        another."""
        mib = 1024 * 1024
        entry = {
            "id": "m1",
            "status": {"value": "loaded"},
            "memory": [
                {"model": 1500 * mib, "context": 400 * mib, "compute": 100 * mib},
                {"model": 800 * mib, "context": 150 * mib, "compute": 50 * mib},
            ],
        }
        with patch("api.state.subprocess.run", return_value=_smi(4096, 24576)), \
             patch("api.state._probe_llama_server_models_status", return_value=[entry]):
            s = state._probe_vram_state()
        assert s["held_vram_bytes"] == 3000 * 1024 * 1024
        assert s["device_used_bytes"] == 4096 * 1024 * 1024
        assert s["held_vram_bytes"] != s["device_used_bytes"]
        assert s["device_total_bytes"] == 24576 * 1024 * 1024
        assert s["total_capacity_bytes"] == 24576 * 1024 * 1024
        assert s["gpu_reachable"] is True
        assert s["router_reachable"] is True
        assert s["resident_model_count"] == 1
        assert s["status"] == "ok"

    def test_no_models_resident_reports_zero_not_null(self):
        """(b) router answered and nothing is resident → held is a real 0, NOT
        null (R1-AC3). The card may still show residual use (contexts, other
        processes) — that is device_used_bytes' job, not ours."""
        with patch("api.state.subprocess.run", return_value=_smi(12, 24576)), \
             patch("api.state._probe_llama_server_models_status", return_value=[]):
            s = state._probe_vram_state()
        assert s["held_vram_bytes"] == 0
        assert s["held_vram_bytes"] is not None
        # Residual card use is still reported — as the card, not as our held.
        assert s["device_used_bytes"] == 12 * 1024 * 1024
        assert s["router_reachable"] is True
        assert s["resident_model_count"] == 0
        assert s["status"] == "ok"

    def test_nvidia_smi_failure_is_unreachable_not_zero(self):
        """(c) nvidia-smi raises → gpu_reachable=False, held None (never 0) (R1-AC4)."""
        with patch("api.state.subprocess.run", side_effect=OSError("no nvidia-smi")), \
             patch("api.state._probe_llama_server_models_status", return_value=[]):
            s = state._probe_vram_state()
        assert s["gpu_reachable"] is False
        assert s["held_vram_bytes"] is None  # NOT coerced to 0
        assert s["total_capacity_bytes"] is None
        assert s["status"] == "unreachable"

    def test_router_unreachable_surfaced_distinctly(self):
        """(d) router probe returns None → router_reachable=False, distinct from empty set."""
        with patch("api.state.subprocess.run", return_value=_smi(4096, 24576)), \
             patch("api.state._probe_llama_server_models_status", return_value=None):
            s = state._probe_vram_state()
        assert s["router_reachable"] is False
        assert s["resident_model_count"] is None
        # GPU still reachable → the CARD figures are still real, status ok...
        assert s["gpu_reachable"] is True
        assert s["status"] == "ok"
        assert s["device_used_bytes"] == 4096 * 1024 * 1024
        # ...but OUR held is unknown: the router is the only thing that can tell
        # us what we are holding. Null, never a false 0 (self.ai#74 / R1-AC4).
        assert s["held_vram_bytes"] is None


# ─── T-002 / T-005: wire-format models ───────────────────────────────────

class TestWireModels:
    def test_vram_state_response_byte_fields_are_int(self):
        r = state.VramStateResponse(
            held_vram_bytes=4096 * 1024 * 1024,
            total_capacity_bytes=24576 * 1024 * 1024,
            gpu_reachable=True,
            router_reachable=True,
            resident_model_count=1,
            status="ok",
        )
        d = r.model_dump()
        assert isinstance(d["held_vram_bytes"], int)
        assert isinstance(d["total_capacity_bytes"], int)
        assert set(d) == {
            "held_vram_bytes", "total_capacity_bytes", "device_used_bytes",
            "device_total_bytes", "gpu_reachable", "router_reachable",
            "resident_model_count", "status", "loaded_model",
        }
        assert d["loaded_model"] is None  # defaults to null when unset

    def test_release_request_amount_only_no_mechanism(self):
        req = state.VramReleaseRequest(target_bytes=8_000_000_000, timeout_seconds=5.0)
        d = req.model_dump()
        assert isinstance(d["target_bytes"], int)
        assert isinstance(d["timeout_seconds"], float)
        assert "mechanism" not in d  # deliberate: amount-based only

    def test_release_response_fields(self):
        resp = state.VramReleaseResponse(
            freed_bytes=8_000_000_000, status="released", evicted_models=["m1"]
        )
        d = resp.model_dump()
        assert isinstance(d["freed_bytes"], int)
        assert d["status"] == "released"
        assert d["evicted_models"] == ["m1"]


# ─── T-004: _unload_model_via_router ─────────────────────────────────────

class TestUnloadViaRouter:
    def test_posts_model_field_to_correct_url(self):
        captured = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"{\"success\": true}"

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = req.data
            return _Resp()

        with patch("urllib.request.urlopen", _fake_urlopen):
            ok = state._unload_model_via_router("my-model")
        assert ok is True
        assert captured["url"] == "http://localhost:8080/models/unload"
        assert captured["method"] == "POST"
        # Body field name is "model" (confirmed against server-models.cpp:2024).
        assert b'"model"' in captured["body"]
        assert b"my-model" in captured["body"]

    def test_http_error_returns_false_without_raising(self):
        with patch("urllib.request.urlopen", side_effect=OSError("router down")):
            ok = state._unload_model_via_router("my-model")
        assert ok is False


# ─── T-006: _select_models_to_evict ──────────────────────────────────────

class TestSelectModelsToEvict:
    def test_only_stable_loaded_selected(self):
        """(a) mix of loading + loaded → only loaded selected (R2-AC3)."""
        models = [
            {"id": "loading-one", "status": {"value": "loading"}},
            {"id": "loaded-one", "status": {"value": "loaded"}},
            {"id": "sleeping-one", "status": {"value": "sleeping"}},
        ]
        with patch("api.state._probe_llama_server_models_status", return_value=models), \
             patch("api.state._lookup_model_size_bytes", return_value=None):
            sel = state._select_models_to_evict(target_bytes=1)
        names = [m["name"] for m in sel]
        assert names == ["loaded-one"]

    def test_lru_oldest_first(self, monkeypatch):
        """(b) LRU order respected — oldest first-seen-loaded evicted first."""
        # Seed the LRU map so "old" was seen before "new".
        state._llama_model_loaded_since["old"] = 100.0
        state._llama_model_loaded_since["new"] = 200.0
        models = [
            {"id": "new", "status": {"value": "loaded"}},
            {"id": "old", "status": {"value": "loaded"}},
        ]
        with patch("api.state._probe_llama_server_models_status", return_value=models), \
             patch("api.state._lookup_model_size_bytes", return_value=None):
            sel = state._select_models_to_evict(target_bytes=10**12)
        assert [m["name"] for m in sel] == ["old", "new"]

    def test_accumulation_stops_once_target_met(self):
        """(c) accumulation stops once target_bytes met."""
        state._llama_model_loaded_since["a"] = 100.0
        state._llama_model_loaded_since["b"] = 200.0
        state._llama_model_loaded_since["c"] = 300.0
        models = [
            {"id": "a", "status": {"value": "loaded"}},
            {"id": "b", "status": {"value": "loaded"}},
            {"id": "c", "status": {"value": "loaded"}},
        ]
        sizes = {"a": 60, "b": 60, "c": 60}
        with patch("api.state._probe_llama_server_models_status", return_value=models), \
             patch("api.state._lookup_model_size_bytes", side_effect=lambda n: sizes[n]):
            sel = state._select_models_to_evict(target_bytes=100)
        # a(60) + b(60) = 120 >= 100 → stop before c.
        assert [m["name"] for m in sel] == ["a", "b"]

    def test_all_loading_returns_empty(self):
        """(d) all-loading set → empty selection."""
        models = [
            {"id": "x", "status": {"value": "loading"}},
            {"id": "y", "status": {"value": "loading"}},
        ]
        with patch("api.state._probe_llama_server_models_status", return_value=models):
            sel = state._select_models_to_evict(target_bytes=10**12)
        assert sel == []

    def test_router_unreachable_returns_empty(self):
        with patch("api.state._probe_llama_server_models_status", return_value=None):
            sel = state._select_models_to_evict(target_bytes=10**12)
        assert sel == []


# ─── T-007: _handle_vram_release orchestration ───────────────────────────

def _vram(held, total=24576 * 1024 * 1024, gpu_reachable=True):
    """Build a _probe_vram_state()-shaped dict with a given held figure."""
    return {
        "held_vram_bytes": held,
        "total_capacity_bytes": total if gpu_reachable else None,
        # self.ai#74: the card reading, separate from this service's held.
        "device_used_bytes": total if gpu_reachable else None,
        "device_total_bytes": total if gpu_reachable else None,
        "gpu_reachable": gpu_reachable,
        "router_reachable": True,
        "resident_model_count": 0,
        "status": "ok" if gpu_reachable else "unreachable",
    }


def _models_status(unloaded=(), loaded=()):
    """Build a _probe_llama_server_models_status()-shaped list where the
    given model ids report `unloaded`/`loaded`, for the settle-wait helper
    (_wait_for_models_unloaded)."""
    return [{"id": name, "status": {"value": "unloaded"}} for name in unloaded] + [
        {"id": name, "status": {"value": "loaded"}} for name in loaded
    ]


@pytest.fixture(autouse=True)
def _release_lock_unheld():
    """Ensure the single-flight release lock is not left held between tests."""
    yield
    if state._vram_release_lock.locked():
        state._vram_release_lock.release()


class TestHandleVramRelease:
    def test_freed_comes_from_live_verification_not_unload_return(self):
        """AC4: an unload that 'succeeds' but frees nothing → freed_bytes=0.

        The unload helper returns True (claims success) and the model IS
        recorded as evicted, but the post-eviction live probe shows held VRAM
        unchanged — so the confirmed freed figure is the measured delta (0),
        never the assumption the unload worked.
        """
        selected = [{"name": "m1", "size_bytes": 8_000_000_000}]
        with patch("api.state._select_models_to_evict", return_value=selected), \
             patch("api.state._unload_model_via_router", return_value=True) as unload, \
             patch("api.state._probe_llama_server_models_status",
                   return_value=_models_status(unloaded=["m1"])), \
             patch("api.state._probe_vram_state",
                   side_effect=[_vram(1000), _vram(1000)]):  # held unchanged
            resp = state._handle_vram_release(target_bytes=500, timeout_seconds=30.0)
        assert unload.called  # the unload WAS issued and returned True…
        assert resp.evicted_models == ["m1"]
        assert resp.freed_bytes == 0  # …but the live delta confirms nothing freed
        assert resp.status == "partial"

    def test_timeout_bounded_stops_partway(self):
        """AC5: never block past the timeout — stop issuing unloads once the
        wall-clock deadline passes, returning with whatever was confirmed.

        Drives a deterministic monotonic clock: deadline is t0+timeout=10; the
        first model's pre-unload check reads 5 (proceed), the second reads 15
        (>= deadline → break). So only the first of three selected models is
        unloaded, and the call returns promptly rather than blocking.
        """
        selected = [
            {"name": "m0", "size_bytes": 1},
            {"name": "m1", "size_bytes": 1},
            {"name": "m2", "size_bytes": 1},
        ]
        clock = MagicMock(side_effect=[0.0, 5.0, 15.0, 15.0, 15.0])
        with patch("api.state._select_models_to_evict", return_value=selected), \
             patch("api.state._unload_model_via_router", return_value=True) as unload, \
             patch("api.state._probe_llama_server_models_status",
                   return_value=_models_status(unloaded=["m0"])), \
             patch("api.state._probe_vram_state",
                   side_effect=[_vram(1000), _vram(600)]), \
             patch("api.state.time.monotonic", clock):
            resp = state._handle_vram_release(target_bytes=900, timeout_seconds=10.0)
        assert unload.call_count == 1  # stopped after the deadline, before m1/m2
        assert resp.evicted_models == ["m0"]
        assert resp.freed_bytes == 400  # confirmed live delta 1000→600
        assert resp.status == "partial"

    def test_single_flight_second_call_is_busy(self):
        """AC6: at most one release at a time — a second, concurrent request
        gets an immediate 'busy', never a raced second eviction pass."""
        # Simulate a release already in flight by holding the lock.
        acquired = state._vram_release_lock.acquire(blocking=False)
        assert acquired
        try:
            with patch("api.state._select_models_to_evict") as sel, \
                 patch("api.state._unload_model_via_router") as unload, \
                 patch("api.state._probe_vram_state") as probe:
                resp = state._handle_vram_release(target_bytes=1, timeout_seconds=5.0)
            assert resp.status == "busy"
            assert resp.freed_bytes == 0
            assert resp.evicted_models == []
            # The busy pass did NO work — it never touched the selection/evict path.
            sel.assert_not_called()
            unload.assert_not_called()
            probe.assert_not_called()
        finally:
            state._vram_release_lock.release()

    def test_over_capacity_target_evicts_everything_and_reports_real_freed(self):
        """AC7: a target exceeding everything resident evicts all evictable
        models and reports the maximum actually freed — not capped to the
        (impossible) request, not denied."""
        selected = [
            {"name": "m1", "size_bytes": 1_000_000_000},
            {"name": "m2", "size_bytes": 1_000_000_000},
        ]
        with patch("api.state._select_models_to_evict", return_value=selected), \
             patch("api.state._unload_model_via_router", return_value=True) as unload, \
             patch("api.state._probe_llama_server_models_status",
                   return_value=_models_status(unloaded=["m1", "m2"])), \
             patch("api.state._probe_vram_state",
                   side_effect=[_vram(2000), _vram(100)]):  # freed 1900 live
            resp = state._handle_vram_release(target_bytes=10**12, timeout_seconds=30.0)
        assert unload.call_count == 2  # everything evictable was unloaded
        assert resp.evicted_models == ["m1", "m2"]
        assert resp.freed_bytes == 1900  # the real max freeable, not the 10**12 ask
        assert resp.status == "partial"  # honest under-target answer, not a denial

    def test_target_met_reports_released(self):
        """Sanity: freed >= target → status 'released'."""
        selected = [{"name": "m1", "size_bytes": 8_000_000_000}]
        with patch("api.state._select_models_to_evict", return_value=selected), \
             patch("api.state._unload_model_via_router", return_value=True), \
             patch("api.state._probe_llama_server_models_status",
                   return_value=_models_status(unloaded=["m1"])), \
             patch("api.state._probe_vram_state",
                   side_effect=[_vram(9_000_000_000), _vram(500_000_000)]):
            resp = state._handle_vram_release(target_bytes=8_000_000_000, timeout_seconds=30.0)
        assert resp.freed_bytes == 8_500_000_000
        assert resp.status == "released"

    def test_settle_wait_polls_until_router_confirms_unloaded(self):
        """Regression test for the live-validation finding (2026-07-24):
        POST /models/unload returns success before the child process has
        actually released its CUDA context, so an immediate post-eviction
        read can wildly undercount freed_bytes (observed live: 626MB
        reported instead of ~14.8GB actually freed).

        Drives the router status probe to report the evicted model still
        `loaded` on its first two checks and only `unloaded` on the third —
        proving _wait_for_models_unloaded() actually polls more than once
        (settles) before the live-verification read happens, rather than
        measuring immediately after the unload HTTP call returns."""
        selected = [{"name": "m1", "size_bytes": 8_000_000_000}]
        status_probe = MagicMock(
            side_effect=[
                _models_status(loaded=["m1"]),   # still tearing down
                _models_status(loaded=["m1"]),   # still tearing down
                _models_status(unloaded=["m1"]), # settled
            ]
        )
        with patch("api.state._select_models_to_evict", return_value=selected), \
             patch("api.state._unload_model_via_router", return_value=True), \
             patch("api.state._probe_llama_server_models_status", status_probe), \
             patch("api.state.time.sleep") as sleep_mock, \
             patch("api.state._probe_vram_state",
                   side_effect=[_vram(15_500_000_000), _vram(4_000_000)]):
            resp = state._handle_vram_release(target_bytes=8_000_000_000, timeout_seconds=30.0)
        assert status_probe.call_count == 3  # polled until settled, not just once
        assert sleep_mock.called  # actually waited between polls
        # The correct, fully-settled delta -- not the undercounted figure a
        # premature read would have produced.
        assert resp.freed_bytes == 15_496_000_000
        assert resp.status == "released"

    def test_gpu_unreachable_cannot_confirm_freed(self):
        """AC4 corollary: if the live probe can't read held VRAM (GPU
        unreachable), freed_bytes is 0 — an unverifiable release is never
        reported as a confirmed freed amount."""
        selected = [{"name": "m1", "size_bytes": 8_000_000_000}]
        with patch("api.state._select_models_to_evict", return_value=selected), \
             patch("api.state._unload_model_via_router", return_value=True), \
             patch("api.state._probe_llama_server_models_status",
                   return_value=_models_status(unloaded=["m1"])), \
             patch("api.state._probe_vram_state",
                   side_effect=[_vram(None, gpu_reachable=False),
                                _vram(None, gpu_reachable=False)]):
            resp = state._handle_vram_release(target_bytes=500, timeout_seconds=30.0)
        assert resp.freed_bytes == 0


# ─── T-003: GET /api/system/vram-state endpoint ──────────────────────────

class TestVramStateEndpoint:
    def test_authenticated_call_returns_byte_fields(self, client):
        """R1-AC1: an authenticated (system:read) call returns held + total
        capacity byte fields, plus the currently-loaded model (self.ai!225).

        `loaded_model` comes from `_primary_loaded_model()`, NOT from
        `_check_inference_health()`. The endpoint moved off the health probe
        because this fork's /health returns a bare {"status": "ok"} with no
        model field in either mode, so the name was structurally always null.
        This test kept patching the health probe afterwards, which the endpoint
        no longer calls, so the real `_primary_loaded_model()` ran, found no
        router, and returned None."""
        with patch("api.routers.system._probe_vram_state",
                   return_value=_vram(4096 * 1024 * 1024)), \
             patch("api.routers.system._primary_loaded_model",
                   return_value="Qwen2.5-Coder-32B"):
            resp = client.get("/api/system/vram-state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["held_vram_bytes"] == 4096 * 1024 * 1024
        assert data["total_capacity_bytes"] == 24576 * 1024 * 1024
        assert data["status"] == "ok"
        assert data["gpu_reachable"] is True
        # The resident model is surfaced from _primary_loaded_model (self.ai!225).
        assert data["loaded_model"] == "Qwen2.5-Coder-32B"

    def test_unreachable_gpu_distinguishable_not_zero(self, client):
        """R1-AC4: an unreachable GPU returns 200 with a distinguishable
        status and null held — NEVER held_vram_bytes=0. loaded_model is null
        when nothing is resident.

        Patches `_primary_loaded_model` for the same reason as the test above.
        This one was not failing, but only by luck: it patched the health probe
        the endpoint no longer calls, and the unpatched `_primary_loaded_model`
        happened to return None here anyway. Right answer, wrong reason — it
        would have kept passing however that function behaved."""
        with patch("api.routers.system._probe_vram_state",
                   return_value=_vram(None, gpu_reachable=False)), \
             patch("api.routers.system._primary_loaded_model", return_value=None):
            resp = client.get("/api/system/vram-state")
        assert resp.status_code == 200  # not an error that would hide the signal
        data = resp.json()
        assert data["status"] == "unreachable"
        assert data["gpu_reachable"] is False
        assert data["held_vram_bytes"] is None  # null, distinct from a real 0
        assert data["held_vram_bytes"] != 0
        assert data["loaded_model"] is None

    def test_requires_authentication(self, client):
        """R1-AC1: the endpoint is ticket-scoped — a request with no ticket is
        rejected by require_scope('system:read')."""
        resp = client.get("/api/system/vram-state",
                          headers={state_ticket_header(): ""})
        assert resp.status_code == 401


# ─── T-008: POST /api/system/vram-release endpoint ───────────────────────

class TestVramReleaseEndpoint:
    def test_authenticated_call_returns_freed_bytes(self, client):
        """R2-AC1: an authenticated (system:write) call with a valid body
        returns the confirmed freed-bytes response."""
        handler_result = state.VramReleaseResponse(
            freed_bytes=8_000_000_000, status="released", evicted_models=["m1"]
        )
        with patch("api.routers.system._handle_vram_release",
                   return_value=handler_result) as handler:
            resp = client.post("/api/system/vram-release",
                               json={"target_bytes": 8_000_000_000, "timeout_seconds": 5.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["freed_bytes"] == 8_000_000_000
        assert data["status"] == "released"
        assert data["evicted_models"] == ["m1"]
        # The body was parsed and passed through to the handler.
        handler.assert_called_once_with(8_000_000_000, 5.0)

    def test_busy_maps_to_409(self, client):
        """R2-AC6: a 'busy' handler result (a release already in flight) is
        surfaced as HTTP 409, not a misleading 200-with-zero."""
        busy = state.VramReleaseResponse(freed_bytes=0, status="busy", evicted_models=[])
        with patch("api.routers.system._handle_vram_release", return_value=busy):
            resp = client.post("/api/system/vram-release",
                               json={"target_bytes": 1, "timeout_seconds": 5.0})
        assert resp.status_code == 409

    def test_requires_authentication(self, client):
        """R2-AC1: the endpoint is ticket-scoped — no ticket → rejected by
        require_scope('system:write')."""
        resp = client.post("/api/system/vram-release",
                           json={"target_bytes": 1, "timeout_seconds": 5.0},
                           headers={state_ticket_header(): ""})
        assert resp.status_code == 401

    def test_missing_body_fields_is_validation_error(self, client):
        """A malformed release-request (missing required fields) is a 422."""
        resp = client.post("/api/system/vram-release", json={})
        assert resp.status_code == 422


def state_ticket_header():
    """The auth ticket header name, read from the auth module so the test
    tracks the source of truth rather than hardcoding it."""
    import api.auth as auth_module
    return auth_module.TICKET_HEADER
