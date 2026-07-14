"""
Test fixtures for the Training API.

Tests run inside the Docker container where all dependencies are available.
Run: docker exec self-llamolotl python -m pytest /workspace/training/api/tests/ -v
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import jwt
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_liveness_heartbeat():
    """Keep the /health/live job-poll heartbeat from leaking between tests.

    Tests that exercise liveness (test_system.py::TestLivenessReadiness)
    mutate api.state._last_poll_heartbeat directly to simulate a fresh or
    stale background poll loop. Reset it to "just ticked" before and after
    each test so an earlier test's stale value can't bleed into an unrelated
    test that happens to hit /health/live.
    """
    import api.state as state

    state._last_poll_heartbeat = time.monotonic()
    yield
    state._last_poll_heartbeat = time.monotonic()

# Test-only HMAC secret for the service-ticket auth layer (self.llamolotl#12).
# Never used outside pytest — real deployments get SERVICE_AUTH_SECRET from
# the selfai-service-auth ExternalSecret.
TEST_SERVICE_AUTH_SECRET = "pytest-only-service-auth-secret"
TEST_SERVICE_AUTH_AUDIENCE = "self.llamolotl"

# Every scope this API currently gates, so the default `client` fixture can
# hit any endpoint without individual tests needing to know about scopes —
# scope enforcement itself is covered separately in test_auth.py.
ALL_SCOPES = (
    "models:read models:pull models:delete models:write "
    "system:read system:write system:restart "
    "jobs:read jobs:write jobs:create "
    "pipeline:read pipeline:write"
)


def mint_test_ticket(
    scope=ALL_SCOPES,
    audience=TEST_SERVICE_AUTH_AUDIENCE,
    secret=TEST_SERVICE_AUTH_SECRET,
    ttl_seconds=120,
    **extra_claims,
):
    """Mint a service ticket signed with the pytest test secret. Mirrors
    self.ai's minting side (api/selfai_ui/utils/service_auth.py) closely
    enough to exercise the same validation path as production."""
    now = int(time.time())
    payload = {
        "iss": "self.ai",
        "aud": audience,
        "scope": scope,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory structure."""
    configs = tmp_path / "configs"
    outputs = tmp_path / "outputs"
    logs = tmp_path / "logs"
    configs.mkdir()
    outputs.mkdir()
    logs.mkdir()

    jobs_file = tmp_path / "jobs.json"
    pipeline_file = tmp_path / "pipeline_tasks.json"

    return {
        "workspace": tmp_path,
        "configs": configs,
        "outputs": outputs,
        "logs": logs,
        "jobs_file": jobs_file,
        "pipeline_file": pipeline_file,
    }


# Attributes to patch in state.py and all modules that import them
_PATH_ATTRS = [
    ("WORKSPACE", "workspace"),
    ("CONFIGS_DIR", "configs"),
    ("OUTPUTS_DIR", "outputs"),
    ("LOGS_DIR", "logs"),
    ("JOBS_STATE_FILE", "jobs_file"),
    ("PIPELINE_STATE_FILE", "pipeline_file"),
]

# All modules that import path constants from state
_MODULES_TO_PATCH = [
    "api.state",
    "api.routers.jobs",
    "api.routers.models",
    "api.routers.pipeline",
    "api.routers.system",
]


@pytest.fixture
def patched_state(temp_workspace):
    """Patch state module and all routers to use temp workspace."""
    import api.state as state
    import api.routers.jobs as jobs_mod
    import api.routers.models as models_mod
    import api.routers.pipeline as pipeline_mod
    import api.routers.system as system_mod

    modules = [state, jobs_mod, models_mod, pipeline_mod, system_mod]

    # Save originals
    originals = {}
    for mod in modules:
        for attr_name, _ in _PATH_ATTRS:
            if hasattr(mod, attr_name):
                originals[(mod, attr_name)] = getattr(mod, attr_name)

    # Patch all modules
    for mod in modules:
        for attr_name, ws_key in _PATH_ATTRS:
            if hasattr(mod, attr_name):
                setattr(mod, attr_name, temp_workspace[ws_key])

    # Clear in-memory state
    state._jobs.clear()
    state._processes.clear()
    state._pipeline_tasks.clear()
    state._pipeline_processes.clear()

    yield state

    # Restore
    for (mod, attr_name), val in originals.items():
        setattr(mod, attr_name, val)
    state._jobs.clear()
    state._processes.clear()
    state._pipeline_tasks.clear()
    state._pipeline_processes.clear()


@pytest.fixture
def client(patched_state):
    """Provide a FastAPI TestClient with patched state and a valid,
    all-scopes service ticket attached by default.

    These tests exercise business logic, not the auth layer itself — scope/
    expiry/audience enforcement is covered in test_auth.py. Patching
    api.auth.SERVICE_AUTH_SECRET here (rather than leaving it unset) also
    means a missing-secret misconfiguration can't accidentally make these
    tests pass by having every route 503 in a way that looks like success.
    """
    import api.auth as auth_module

    original_secret = auth_module.SERVICE_AUTH_SECRET
    auth_module.SERVICE_AUTH_SECRET = TEST_SERVICE_AUTH_SECRET

    from api.main import app
    c = TestClient(app)
    c.headers.update({auth_module.TICKET_HEADER: mint_test_ticket()})

    yield c

    auth_module.SERVICE_AUTH_SECRET = original_secret
