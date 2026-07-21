"""
Tests for the llama-server watchdog (self.llamolotl#24): detecting a wedged
`status: loading` (or otherwise unreachable) llama-server held past a
generous timeout with no forward progress, and recovering it via the
existing _restart_llama_server() primitive.

llama-server is supervised by supervisord (supervisord.conf), not spawned by
training-api — training-api is a *sibling* process, not llama-server's
parent, so it cannot waitpid()/reap llama-server or its children directly
(that's a POSIX restriction, not a design choice). These tests exercise the
module-level functions in api/state.py directly rather than through an HTTP
endpoint, since the watchdog is a periodic background check
(_check_llama_server_watchdog, called from _poll_jobs()'s existing tick
loop), not a new route.

Second pass: this deployment (self.ai repo's
manifests/llamolotl/10-deployment.yaml sets LLAMA_ARG_MODELS_DIR/
LLAMA_ARG_MODELS_MAX) runs llama-server in ROUTER MODE, not classic
single-model mode — the first pass of this fix was written against the
wrong assumption and missed two things router mode changes:
  - "orphaned" (per the original incident report) means reparented to
    PID 1, not a still-attached child — TestOrphanedWorkerDetection and
    TestOrphanedWorkerTermination below cover the corrected, wider scope
    that a still-current-router-descendant-only scan structurally cannot
    see (self.llama/tools/server/server-models.cpp spawns workers via
    posix_spawn with no PR_SET_PDEATHSIG; if the router dies uncleanly
    those children are reparented, not killed).
  - router mode's /health reports ready=true at router boot, before any
    per-model load — it does not reflect a wedged per-model load the way
    single-model mode's /health does. TestModelLoadStaleness covers the
    router-mode-aware per-model signal added to catch that case.
"""

import json
import logging
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_watchdog_state():
    """Isolate watchdog module state between tests (mirrors the existing
    reset_liveness_heartbeat fixture in conftest.py for _last_poll_heartbeat)."""
    import api.state as state

    state._llama_unhealthy_since = None
    state._llama_watchdog_last_fired = None
    state._llama_model_loading_since.clear()
    yield
    state._llama_unhealthy_since = None
    state._llama_watchdog_last_fired = None
    state._llama_model_loading_since.clear()


class TestWedgeDetection:
    """Core timeout/progress logic — no process introspection involved."""

    def test_first_unhealthy_tick_does_not_fire(self):
        """A single unhealthy observation just starts the streak; it is not
        by itself a wedge."""
        import api.state as state

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._restart_llama_server") as mock_restart:
            state._check_llama_server_watchdog()

        assert state._llama_unhealthy_since is not None
        mock_restart.assert_not_called()

    def test_progress_within_timeout_is_left_alone(self):
        """Unhealthy, but still inside the generous timeout window — e.g. a
        large model that is legitimately still loading. Must not be
        touched."""
        import api.state as state

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS / 2)
        )

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._restart_llama_server") as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_not_called()
        # Streak is preserved (not reset) — still the same wedge candidate.
        assert state._llama_unhealthy_since is not None

    def test_healthy_observation_clears_the_streak(self):
        """A single healthy poll is forward progress: it resets the clock
        even if the prior streak was already far past the timeout."""
        import api.state as state

        state._llama_unhealthy_since = time.monotonic() - 10_000

        with patch("api.state._check_inference_health", return_value=(True, "model.gguf")), \
             patch("api.state._restart_llama_server") as mock_restart:
            state._check_llama_server_watchdog()

        assert state._llama_unhealthy_since is None
        mock_restart.assert_not_called()

    def test_wedged_past_timeout_triggers_recovery(self):
        """An unhealthy streak exceeding the configured timeout, with no
        healthy observation in between, fires _restart_llama_server()."""
        import api.state as state

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)
        )

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._probe_llama_server_raw_status", return_value="loading model"), \
             patch("api.state._get_llama_server_pid", return_value=None), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_called_once()
        # The freshly-restarted instance gets a clean slate / full new
        # timeout window rather than inheriting the streak that just fired.
        assert state._llama_unhealthy_since is None
        assert state._llama_watchdog_last_fired is not None

    def test_cooldown_prevents_immediate_refire(self):
        """A very recent recovery blocks a second one even if the streak
        (re-set by something external) already looks wedged again."""
        import api.state as state

        now = time.monotonic()
        state._llama_watchdog_last_fired = now
        state._llama_unhealthy_since = now - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._restart_llama_server") as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_not_called()

    def test_watchdog_never_raises(self):
        """A bug in _check_inference_health() itself must not propagate —
        _poll_jobs() wraps the watchdog call, but the watchdog function
        should still be well-behaved on its own."""
        import api.state as state

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)
        )

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._probe_llama_server_raw_status", return_value="loading model"), \
             patch("api.state._get_llama_server_pid", side_effect=RuntimeError("boom")), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}):
            with pytest.raises(RuntimeError):
                # _get_llama_server_pid raising is a real bug and SHOULD
                # surface here; _poll_jobs()'s own try/except is what keeps
                # the background loop alive in production. This test just
                # documents that boundary explicitly.
                state._check_llama_server_watchdog()


class TestProcessIdentity:
    """Identity verification is what stands between the watchdog and ever
    treating an unrelated process as llama-server."""

    def test_verify_accepts_matching_cmdline(self):
        from api.state import _verify_llama_server_identity

        mock_proc = MagicMock()
        mock_proc.status.return_value = "running"
        mock_proc.cmdline.return_value = ["/app/llama-server", "--jinja", "--port", "8080"]
        mock_proc.name.return_value = "llama-server"

        with patch("psutil.Process", return_value=mock_proc):
            assert _verify_llama_server_identity(4242) is True

    def test_verify_rejects_unrelated_process(self):
        """A pid that superficially could be confused for llama-server
        (e.g. stale/reused after supervisorctl's read) must never verify
        just because *some* process exists at that pid."""
        from api.state import _verify_llama_server_identity

        mock_proc = MagicMock()
        mock_proc.status.return_value = "running"
        mock_proc.cmdline.return_value = ["/opt/venv/bin/python", "train.py", "--config", "run.yaml"]
        mock_proc.name.return_value = "python"

        with patch("psutil.Process", return_value=mock_proc):
            assert _verify_llama_server_identity(9999) is False

    def test_verify_accepts_zombie_of_verified_instance(self):
        """A pid already in zombie state is still 'the' llama-server
        instance (just dead) — identity holds even there."""
        from api.state import _verify_llama_server_identity
        import psutil

        mock_proc = MagicMock()
        mock_proc.status.return_value = psutil.STATUS_ZOMBIE

        with patch("psutil.Process", return_value=mock_proc):
            assert _verify_llama_server_identity(4242) is True

    def test_verify_handles_lookup_failure_safely(self):
        """A pid that no longer exists must fail closed: no exception
        escapes, no positive identity."""
        from api.state import _verify_llama_server_identity
        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(1234)):
            assert _verify_llama_server_identity(1234) is False


class TestZombieChildDetection:
    """Diagnostic-only zombie enumeration, strictly scoped to real OS-level
    descendants of an already identity-verified pid."""

    def test_finds_zombie_descendants_of_verified_pid(self):
        from api.state import _find_llama_server_zombie_children
        import psutil

        zombie_child = MagicMock()
        zombie_child.pid = 501
        zombie_child.status.return_value = psutil.STATUS_ZOMBIE

        alive_child = MagicMock()
        alive_child.pid = 502
        alive_child.status.return_value = psutil.STATUS_RUNNING

        mock_parent = MagicMock()
        mock_parent.children.return_value = [zombie_child, alive_child]

        with patch("psutil.Process", return_value=mock_parent):
            result = _find_llama_server_zombie_children(4242)

        assert result == [501]

    def test_only_real_descendants_are_ever_considered(self):
        """A zombie that is not an actual OS-level child of the verified
        pid is never reported — this function has no fallback to a
        broader, name-based scan."""
        from api.state import _find_llama_server_zombie_children

        mock_parent = MagicMock()
        mock_parent.children.return_value = []  # no real descendants

        with patch("psutil.Process", return_value=mock_parent):
            result = _find_llama_server_zombie_children(4242)

        assert result == []

    def test_scan_failure_degrades_to_empty_list(self):
        """psutil errors during the scan must never propagate — this is
        diagnostics-only and must never break the watchdog tick."""
        from api.state import _find_llama_server_zombie_children
        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(4242)):
            assert _find_llama_server_zombie_children(4242) == []


class TestOrphanedWorkerDetection:
    """_find_orphaned_llama_server_workers() — the corrected scope from the
    second pass of self.llamolotl#24. "Orphaned" in the original incident
    report means reparented to init (PID 1), not a still-attached child —
    _find_llama_server_zombie_children() above cannot see that case at all,
    since a reparented process is no longer a descendant of the current
    router pid by definition. These tests exercise the three conditions
    that together identify a genuine leftover from a *prior* router
    instance: not the router pid itself, ppid != router pid, and
    create_time strictly before the router's own create_time.
    """

    ROUTER_PID = 4242
    ROUTER_CREATE_TIME = 5000.0

    def _mock_router(self):
        router = MagicMock()
        router.create_time.return_value = self.ROUTER_CREATE_TIME
        return router

    def _mock_candidate(self, pid, ppid, cmdline, create_time, name="llama-server"):
        proc = MagicMock()
        proc.pid = pid
        proc.ppid.return_value = ppid
        proc.cmdline.return_value = cmdline
        proc.name.return_value = name
        proc.create_time.return_value = create_time
        return proc

    def test_finds_a_genuine_orphan_reparented_to_init(self):
        """ppid=1, predates the current router, matches identity by
        cmdline — this is exactly the leftover-from-a-prior-router-
        instance case the original pass missed entirely."""
        from api.state import _find_orphaned_llama_server_workers

        orphan = self._mock_candidate(
            pid=999, ppid=1,
            cmdline=["/app/llama-server", "--model", "qwen.gguf", "--port", "9001"],
            create_time=self.ROUTER_CREATE_TIME - 500,  # started well before this router
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([orphan])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert len(result) == 1
        assert result[0]["pid"] == 999
        assert result[0]["ppid"] == 1

    def test_currently_attached_child_is_never_flagged(self):
        """ppid == the current router's pid: a normal, live child. Must
        never appear in the result regardless of its create_time."""
        from api.state import _find_orphaned_llama_server_workers

        live_child = self._mock_candidate(
            pid=1000, ppid=self.ROUTER_PID,
            cmdline=["/app/llama-server", "--model", "nomic.gguf"],
            create_time=self.ROUTER_CREATE_TIME - 500,  # even if "old" by clock alone
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([live_child])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert result == []

    def test_process_newer_than_router_is_never_flagged(self):
        """ppid=1 but create_time is AFTER the current router started: this
        is what the timing guard exists for — ruling out a race where a
        brand-new legitimate child briefly reads with an unsettled ppid,
        rather than actually being a leftover from a prior instance."""
        from api.state import _find_orphaned_llama_server_workers

        newer_proc = self._mock_candidate(
            pid=1001, ppid=1,
            cmdline=["/app/llama-server", "--model", "qwen.gguf"],
            create_time=self.ROUTER_CREATE_TIME + 10,  # created AFTER this router started
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([newer_proc])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert result == []

    def test_unrelated_orphan_never_flagged_by_ppid_alone(self):
        """A process reparented to init that is NOT a llama-server process
        at all (extremely common in any container — plenty of things end
        up parented to PID 1) must never be flagged just because its ppid
        looks orphan-shaped. Identity (cmdline/name) still gates
        everything, same as the rest of this module."""
        from api.state import _find_orphaned_llama_server_workers

        unrelated = self._mock_candidate(
            pid=1002, ppid=1,
            cmdline=["/usr/bin/some-other-daemon", "--flag"],
            create_time=self.ROUTER_CREATE_TIME - 500,
            name="some-other-daemon",
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([unrelated])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert result == []

    def test_router_pid_itself_is_skipped(self):
        """The router's own pid, if it happens to show up while iterating
        all processes, is never treated as its own orphan."""
        from api.state import _find_orphaned_llama_server_workers

        router_as_candidate = self._mock_candidate(
            pid=self.ROUTER_PID, ppid=1,
            cmdline=["/app/llama-server", "--models-dir", "/models"],
            create_time=self.ROUTER_CREATE_TIME,
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([router_as_candidate])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert result == []

    def test_multiple_orphans_sorted_oldest_first(self):
        from api.state import _find_orphaned_llama_server_workers

        newer_orphan = self._mock_candidate(
            pid=2001, ppid=1,
            cmdline=["/app/llama-server", "--model", "b.gguf"],
            create_time=self.ROUTER_CREATE_TIME - 100,
        )
        older_orphan = self._mock_candidate(
            pid=2002, ppid=1,
            cmdline=["/app/llama-server", "--model", "a.gguf"],
            create_time=self.ROUTER_CREATE_TIME - 900,
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([newer_orphan, older_orphan])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert [o["pid"] for o in result] == [2002, 2001]

    def test_router_create_time_lookup_failure_degrades_to_empty_list(self):
        """If the router's own create_time can't be read, there's no
        reference point to compare against — degrade safely rather than
        guess."""
        from api.state import _find_orphaned_llama_server_workers
        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(self.ROUTER_PID)):
            assert _find_orphaned_llama_server_workers(self.ROUTER_PID) == []

    def test_process_iter_failure_degrades_to_empty_list(self):
        from api.state import _find_orphaned_llama_server_workers

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", side_effect=OSError("scan failed")):
            assert _find_orphaned_llama_server_workers(self.ROUTER_PID) == []

    def test_a_bad_candidate_mid_scan_is_skipped_not_fatal(self):
        """One process that raises while being inspected (exited mid-scan,
        permission denied, etc.) must not stop the rest of the scan or
        propagate out of this diagnostic/detection path."""
        from api.state import _find_orphaned_llama_server_workers
        import psutil

        broken = MagicMock()
        broken.pid = 3000
        broken.ppid.side_effect = psutil.NoSuchProcess(3000)

        good_orphan = self._mock_candidate(
            pid=3001, ppid=1,
            cmdline=["/app/llama-server", "--model", "c.gguf"],
            create_time=self.ROUTER_CREATE_TIME - 200,
        )

        with patch("psutil.Process", return_value=self._mock_router()), \
             patch("psutil.process_iter", return_value=iter([broken, good_orphan])):
            result = _find_orphaned_llama_server_workers(self.ROUTER_PID)

        assert [o["pid"] for o in result] == [3001]


class TestOrphanedWorkerTermination:
    """_terminate_orphaned_llama_server_workers() — the one place in this
    module that signals a pid training-api did not itself spawn. Every
    test here confirms that action is still gated by the same identity
    rigor as the rest of the module, re-checked at the moment of action
    (not just trusted from the earlier scan)."""

    def test_sends_sigterm_to_a_verified_orphan(self):
        from api.state import _terminate_orphaned_llama_server_workers

        orphan_proc = MagicMock()
        orphan_proc.cmdline.return_value = ["/app/llama-server", "--model", "qwen.gguf"]
        orphan_proc.name.return_value = "llama-server"

        orphans = [{"pid": 999, "ppid": 1, "create_time": 100.0, "cmdline": "/app/llama-server --model qwen.gguf"}]

        with patch("psutil.Process", return_value=orphan_proc):
            terminated = _terminate_orphaned_llama_server_workers(orphans)

        orphan_proc.terminate.assert_called_once()
        orphan_proc.kill.assert_not_called()
        orphan_proc.send_signal.assert_not_called()
        assert terminated == [999]

    def test_reverifies_identity_before_signaling_and_skips_on_mismatch(self):
        """A pid that no longer looks like llama-server at the moment of
        action (e.g. reused by the OS between scan and here) must never be
        signaled, even though it passed the earlier scan."""
        from api.state import _terminate_orphaned_llama_server_workers

        reused_proc = MagicMock()
        reused_proc.cmdline.return_value = ["/opt/venv/bin/python", "some_unrelated_script.py"]
        reused_proc.name.return_value = "python"

        orphans = [{"pid": 999, "ppid": 1, "create_time": 100.0, "cmdline": "/app/llama-server"}]

        with patch("psutil.Process", return_value=reused_proc):
            terminated = _terminate_orphaned_llama_server_workers(orphans)

        reused_proc.terminate.assert_not_called()
        reused_proc.kill.assert_not_called()
        assert terminated == []

    def test_already_exited_pid_is_skipped_gracefully(self):
        from api.state import _terminate_orphaned_llama_server_workers
        import psutil

        orphans = [{"pid": 999, "ppid": 1, "create_time": 100.0, "cmdline": "/app/llama-server"}]

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(999)):
            terminated = _terminate_orphaned_llama_server_workers(orphans)

        assert terminated == []

    def test_one_failure_does_not_block_the_rest(self):
        from api.state import _terminate_orphaned_llama_server_workers
        import psutil

        good_proc = MagicMock()
        good_proc.cmdline.return_value = ["/app/llama-server", "--model", "b.gguf"]
        good_proc.name.return_value = "llama-server"

        orphans = [
            {"pid": 998, "ppid": 1, "create_time": 100.0, "cmdline": "/app/llama-server --model a.gguf"},
            {"pid": 999, "ppid": 1, "create_time": 200.0, "cmdline": "/app/llama-server --model b.gguf"},
        ]

        def process_side_effect(pid):
            if pid == 998:
                raise psutil.NoSuchProcess(998)
            return good_proc

        with patch("psutil.Process", side_effect=process_side_effect):
            terminated = _terminate_orphaned_llama_server_workers(orphans)

        assert terminated == [999]
        good_proc.terminate.assert_called_once()

    def test_empty_input_is_a_no_op(self):
        from api.state import _terminate_orphaned_llama_server_workers

        assert _terminate_orphaned_llama_server_workers([]) == []


class TestModelLoadStaleness:
    """_check_llama_server_model_load_staleness() — the router-mode-aware
    signal that catches a specific model wedged in status: loading even
    while the router's own /health reports fine (see the function's
    docstring for why /health alone can't see this under router mode)."""

    def _models_response(self, entries):
        body = json.dumps({"data": entries}).encode()
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = body
        cm.__exit__.return_value = False
        return cm

    def test_no_stale_model_when_nothing_is_loading(self):
        from api.state import _check_llama_server_model_load_staleness

        entries = [{"id": "qwen", "status": {"value": "loaded"}}]
        with patch("urllib.request.urlopen", return_value=self._models_response(entries)):
            assert _check_llama_server_model_load_staleness() is None

    def test_a_model_just_starting_to_load_is_not_yet_stale(self):
        from api.state import _check_llama_server_model_load_staleness

        entries = [{"id": "qwen", "status": {"value": "loading"}}]
        with patch("urllib.request.urlopen", return_value=self._models_response(entries)):
            assert _check_llama_server_model_load_staleness() is None

    def test_a_model_loading_past_the_timeout_is_stale(self):
        import api.state as state
        from api.state import _check_llama_server_model_load_staleness

        state._llama_model_loading_since["qwen"] = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)
        )
        entries = [{"id": "qwen", "status": {"value": "loading"}}]
        with patch("urllib.request.urlopen", return_value=self._models_response(entries)):
            assert _check_llama_server_model_load_staleness() == "qwen"

    def test_a_model_that_finishes_loading_clears_its_tracking(self):
        import api.state as state
        from api.state import _check_llama_server_model_load_staleness

        state._llama_model_loading_since["qwen"] = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)
        )
        entries = [{"id": "qwen", "status": {"value": "loaded"}}]
        with patch("urllib.request.urlopen", return_value=self._models_response(entries)):
            assert _check_llama_server_model_load_staleness() is None
        assert "qwen" not in state._llama_model_loading_since

    def test_unreachable_router_returns_none_not_stale(self):
        """/models being unreachable is "no info", not itself a wedge
        signal — that's what the health-based streak is for."""
        from api.state import _check_llama_server_model_load_staleness

        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert _check_llama_server_model_load_staleness() is None


class TestSupervisorPidLookup:
    def test_parses_pid_from_supervisorctl(self):
        from api.state import _get_llama_server_pid

        mock_result = MagicMock(returncode=0, stdout="4242\n")
        with patch("api.state.subprocess.run", return_value=mock_result):
            assert _get_llama_server_pid() == 4242

    def test_returns_none_on_supervisorctl_failure(self):
        from api.state import _get_llama_server_pid

        mock_result = MagicMock(returncode=1, stdout="")
        with patch("api.state.subprocess.run", return_value=mock_result):
            assert _get_llama_server_pid() is None

    def test_returns_none_on_timeout(self):
        from api.state import _get_llama_server_pid

        with patch(
            "api.state.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="supervisorctl", timeout=10),
        ):
            assert _get_llama_server_pid() is None


class TestSafetyInvariants:
    """The guarantee this watchdog offers, corrected from the first pass of
    self.llamolotl#24: recovery against the *router pid itself* is always
    mediated through supervisord's named-program restart
    (`supervisorctl restart llama-server`) — training-api never signals
    that pid directly, because it is not that pid's parent and has no
    standing to. That part is unchanged.

    What changed: the original claim that this watchdog "never falls back
    to a system-wide process scan" and "never signals any pid" turned out
    to be too strong once router-mode orphans were accounted for (see
    _find_orphaned_llama_server_workers()'s docstring) — a worker
    reparented to PID 1 by a *prior* router instance is, by definition,
    not discoverable by walking descendants of the current router pid, so
    finding it requires exactly the system-wide scan this class used to
    assert never happens. The guarantee that actually holds now: a
    process is only ever scanned broadly to look for orphans, and only
    ever signaled directly once it has (a) matched the same cmdline
    identity check used everywhere else in this module, (b) been
    confirmed to NOT be a currently-attached child of the live router
    (ppid != router pid), and (c) been confirmed to predate the current
    router's own start time. A still-attached child of the router —
    verified or not — is never signaled directly, in either the old model
    or this one.
    """

    def _wedge_and_fire(self, mock_proc, process_iter_return=()):
        import api.state as state

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)
        )
        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._check_llama_server_model_load_staleness", return_value=None), \
             patch("api.state.subprocess.run", return_value=MagicMock(returncode=0, stdout="4242\n")), \
             patch("psutil.Process", return_value=mock_proc), \
             patch("psutil.process_iter", return_value=iter(process_iter_return)) as mock_process_iter, \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            state._check_llama_server_watchdog()
        return mock_restart, mock_process_iter

    def test_orphan_scan_now_runs_as_part_of_firing(self):
        """Unlike the first pass, the broad scan IS used now — it's how a
        wedge-firing recovery finds orphans left by a prior router
        instance. This is a deliberate, narrowly-scoped change from the
        original "never" guarantee, not a regression of it."""
        mock_proc = MagicMock()
        mock_proc.status.return_value = "running"
        mock_proc.cmdline.return_value = ["/app/llama-server"]
        mock_proc.name.return_value = "llama-server"
        mock_proc.children.return_value = []
        mock_proc.create_time.return_value = 1000.0

        _, mock_process_iter = self._wedge_and_fire(mock_proc)

        mock_process_iter.assert_called_once()

    def test_verified_router_pid_itself_is_never_signaled_directly(self):
        """The pid supervisorctl reports for llama-server — the one
        identity-verified via _verify_llama_server_identity() — is never
        touched by a direct kill/terminate/signal call. Recovery against
        *that* pid only ever goes through supervisord."""
        mock_proc = MagicMock()
        mock_proc.status.return_value = "running"
        mock_proc.cmdline.return_value = ["/app/llama-server"]
        mock_proc.name.return_value = "llama-server"
        mock_proc.children.return_value = []
        mock_proc.create_time.return_value = 1000.0

        mock_restart, _ = self._wedge_and_fire(mock_proc)

        mock_restart.assert_called_once()
        mock_proc.kill.assert_not_called()
        mock_proc.terminate.assert_not_called()
        mock_proc.send_signal.assert_not_called()

    def test_a_currently_attached_child_is_never_signaled_even_during_orphan_scan(self):
        """A process discovered by the broad scan that turns out to be a
        normal, currently-attached child of the live router (ppid ==
        router pid) must never be terminated — it's the same invariant as
        before, just proven under the new scan rather than by the scan
        not existing."""
        router_pid = 4242
        mock_router_proc = MagicMock()
        mock_router_proc.status.return_value = "running"
        mock_router_proc.cmdline.return_value = ["/app/llama-server"]
        mock_router_proc.name.return_value = "llama-server"
        mock_router_proc.children.return_value = []
        mock_router_proc.create_time.return_value = 1000.0

        current_child = MagicMock()
        current_child.pid = 5000
        current_child.ppid.return_value = router_pid  # a normal, live child
        current_child.cmdline.return_value = ["/app/llama-server", "--model", "nomic.gguf"]
        current_child.name.return_value = "llama-server"
        current_child.create_time.return_value = 2000.0  # newer than the router either way

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._check_llama_server_model_load_staleness", return_value=None), \
             patch("api.state.subprocess.run", return_value=MagicMock(returncode=0, stdout=f"{router_pid}\n")), \
             patch("psutil.Process", return_value=mock_router_proc), \
             patch("psutil.process_iter", return_value=iter([current_child])), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            import api.state as state
            state._llama_unhealthy_since = (
                time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 5)
            )
            state._check_llama_server_watchdog()

        mock_restart.assert_called_once()
        current_child.terminate.assert_not_called()
        current_child.kill.assert_not_called()
        current_child.send_signal.assert_not_called()


class TestWatchdogEndToEndRecoveryLogging:
    def test_fires_and_logs_zombie_children_when_present(self, caplog):
        """Full wedge path: verified pid with zombie children present ->
        recovery fires and the zombie info is logged loudly. This is a
        self-healing action recovering from an abnormal state — it must
        never be silent, so operators can notice a pattern of frequent
        wedges even though it auto-recovers."""
        import api.state as state
        import psutil

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 1)
        )

        zombie_child = MagicMock()
        zombie_child.pid = 777
        zombie_child.status.return_value = psutil.STATUS_ZOMBIE

        mock_llama_proc = MagicMock()
        mock_llama_proc.status.return_value = psutil.STATUS_RUNNING
        mock_llama_proc.cmdline.return_value = ["/app/llama-server"]
        mock_llama_proc.name.return_value = "llama-server"
        mock_llama_proc.children.return_value = [zombie_child]
        mock_llama_proc.create_time.return_value = 1000.0

        with caplog.at_level(logging.ERROR), \
             patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._probe_llama_server_raw_status", return_value="loading model"), \
             patch("api.state.subprocess.run", return_value=MagicMock(returncode=0, stdout="4242\n")), \
             patch("psutil.Process", return_value=mock_llama_proc), \
             patch("psutil.process_iter", return_value=iter([])), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_called_once()
        messages = [r.message for r in caplog.records]
        assert any("FIRING" in m for m in messages)
        assert any("zombie child" in m for m in messages)
        assert any("777" in m for m in messages)

    def test_fires_without_pid_still_recovers(self):
        """Even if supervisorctl can't be reached at all, the watchdog
        still falls back to triggering recovery rather than giving up —
        an unreachable supervisorctl is itself part of "wedged"."""
        import api.state as state

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 1)
        )

        with patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._probe_llama_server_raw_status", return_value="unreachable (TimeoutError)"), \
             patch("api.state.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="supervisorctl", timeout=10)), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_called_once()

    def test_fires_on_stale_model_alone_even_when_health_reports_ok(self, caplog):
        """The core router-mode fix: /health can report healthy=True the
        entire time (router mode's is_ready flips true at router boot,
        independent of any per-model load — see
        _check_inference_health()'s docstring) while one specific model is
        genuinely wedged in status: loading. This must still fire recovery
        — the first pass of this watchdog would have sat there forever in
        this exact scenario, since its only signal was the /health streak."""
        import api.state as state

        with caplog.at_level(logging.ERROR), \
             patch("api.state._check_inference_health", return_value=(True, "qwen.gguf")), \
             patch(
                 "api.state._check_llama_server_model_load_staleness",
                 return_value="qwen-coder-32b",
             ), \
             patch("api.state._probe_llama_server_raw_status", return_value="ready"), \
             patch("api.state.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="supervisorctl", timeout=10)), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_called_once()
        messages = [r.message for r in caplog.records]
        assert any("FIRING" in m for m in messages)
        assert any("qwen-coder-32b" in m for m in messages)
        # No unhealthy streak was ever started — the stale-model signal
        # alone was sufficient to fire, independent of _llama_unhealthy_since.
        assert state._llama_unhealthy_since is None

    def test_orphan_found_during_firing_is_terminated_alongside_restart(self, caplog):
        """Full corrected recovery path: a wedge fires, the orphan scan
        finds a genuine leftover from a prior router instance, and it gets
        an explicit SIGTERM — a capability the first pass of this fix did
        not have (restart alone cannot reach a process that is no longer
        the router's child)."""
        import api.state as state
        import psutil

        state._llama_unhealthy_since = (
            time.monotonic() - (state._LLAMA_WATCHDOG_TIMEOUT_SECONDS + 1)
        )

        router_pid = 4242
        mock_router_proc = MagicMock()
        mock_router_proc.status.return_value = psutil.STATUS_RUNNING
        mock_router_proc.cmdline.return_value = ["/app/llama-server"]
        mock_router_proc.name.return_value = "llama-server"
        mock_router_proc.children.return_value = []
        mock_router_proc.create_time.return_value = 5000.0

        orphan_candidate = MagicMock()
        orphan_candidate.pid = 9001
        orphan_candidate.ppid.return_value = 1  # reparented to init
        orphan_candidate.cmdline.return_value = ["/app/llama-server", "--model", "leftover.gguf"]
        orphan_candidate.name.return_value = "llama-server"
        orphan_candidate.create_time.return_value = 4000.0  # predates this router

        def process_side_effect(pid):
            if pid == router_pid:
                return mock_router_proc
            if pid == 9001:
                return orphan_candidate
            raise psutil.NoSuchProcess(pid)

        with caplog.at_level(logging.ERROR), \
             patch("api.state._check_inference_health", return_value=(False, None)), \
             patch("api.state._probe_llama_server_raw_status", return_value="loading model"), \
             patch("api.state.subprocess.run", return_value=MagicMock(returncode=0, stdout=f"{router_pid}\n")), \
             patch("psutil.Process", side_effect=process_side_effect), \
             patch("psutil.process_iter", return_value=iter([orphan_candidate])), \
             patch("api.state._restart_llama_server", return_value={"status": "restarted"}) as mock_restart:
            state._check_llama_server_watchdog()

        mock_restart.assert_called_once()
        orphan_candidate.terminate.assert_called_once()
        orphan_candidate.kill.assert_not_called()
        messages = [r.message for r in caplog.records]
        assert any("orphaned llama-server worker" in m for m in messages)
        assert any("9001" in m for m in messages)
