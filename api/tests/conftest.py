"""
Test fixtures for the Training API.

Tests run inside the Docker container where all dependencies are available.
Run: docker exec self-llamolotl python -m pytest /workspace/training/api/tests/ -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


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
    """Provide a FastAPI TestClient with patched state."""
    from api.main import app
    return TestClient(app)
