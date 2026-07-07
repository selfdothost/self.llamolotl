"""
T-062: Tests for pipeline task lifecycle.
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


def _mock_popen():
    """Return context manager that mocks Popen for pipeline tasks."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    return patch("api.state.subprocess.Popen", return_value=mock_proc)


class TestPipelineTaskCreation:
    """Test pipeline task creation and tracking."""

    def test_task_tracked_in_state(self, patched_state):
        """Pipeline tasks are tracked in _pipeline_tasks dict."""
        from api.state import PipelineTaskType, PipelineTaskStatus

        popen_patch = _mock_popen()
        with popen_patch:
            task = patched_state._start_pipeline_task(
                task_type=PipelineTaskType.QUANTIZE,
                cmd=["echo", "test"],
                input_path="/models/test.gguf",
                output_path="/models/test-q4.gguf",
            )

        assert task.task_id in patched_state._pipeline_tasks
        assert task.status == PipelineTaskStatus.RUNNING
        assert task.task_type == PipelineTaskType.QUANTIZE

    def test_task_persisted_to_file(self, patched_state):
        """Pipeline tasks are saved to pipeline_tasks.json."""
        from api.state import PipelineTaskType

        popen_patch = _mock_popen()
        with popen_patch:
            patched_state._start_pipeline_task(
                task_type=PipelineTaskType.QUANTIZE,
                cmd=["echo", "test"],
                input_path="/models/test.gguf",
                output_path="/models/test-q4.gguf",
            )

        assert patched_state.PIPELINE_STATE_FILE.exists()
        data = json.loads(patched_state.PIPELINE_STATE_FILE.read_text())
        assert len(data) == 1


class TestPipelineTaskQueries:
    """Test pipeline task query endpoints."""

    def test_list_tasks(self, client, patched_state):
        """GET /api/pipeline/tasks returns task list."""
        from api.state import PipelineTask, PipelineTaskType, PipelineTaskStatus

        task = PipelineTask(
            task_id="test-123",
            task_type=PipelineTaskType.QUANTIZE,
            status=PipelineTaskStatus.COMPLETED,
            created_at=datetime.now(),
            log_file="/tmp/test.log",
            input_path="/models/test.gguf",
            output_path="/models/test-q4.gguf",
        )
        patched_state._pipeline_tasks["test-123"] = task

        resp = client.get("/api/pipeline/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert len(tasks) >= 1
        assert any(t["task_id"] == "test-123" for t in tasks)

    def test_get_task_by_id(self, client, patched_state):
        """GET /api/pipeline/tasks/{id} returns specific task."""
        from api.state import PipelineTask, PipelineTaskType, PipelineTaskStatus

        task = PipelineTask(
            task_id="fetch-456",
            task_type=PipelineTaskType.CONVERT_TO_GGUF,
            status=PipelineTaskStatus.RUNNING,
            created_at=datetime.now(),
            log_file="/tmp/test.log",
            input_path="/workspace/model",
            output_path="/models/output.gguf",
        )
        patched_state._pipeline_tasks["fetch-456"] = task

        resp = client.get("/api/pipeline/tasks/fetch-456")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "fetch-456"
        assert resp.json()["status"] == "running"

    def test_get_nonexistent_task(self, client):
        """GET /api/pipeline/tasks/{id} returns 404 for missing task."""
        resp = client.get("/api/pipeline/tasks/does-not-exist")
        assert resp.status_code == 404


class TestGPUMutualExclusion:
    """Test pipeline/training mutual exclusion."""

    def test_task_queued_when_training_running(self, patched_state):
        """GPU pipeline tasks are queued when training is running."""
        from api.state import (
            PipelineTaskType, PipelineTaskStatus,
            Job, JobStatus,
        )

        running_job = Job(
            job_id="train-1",
            config_path="test.yaml",
            output_dir="/workspace/training/outputs/test",
            status=JobStatus.RUNNING,
            approved=True,
            created_at=datetime.now(),
            log_file="/workspace/training/logs/test.log",
        )
        patched_state._jobs["train-1"] = running_job

        task = patched_state._start_pipeline_task(
            task_type=PipelineTaskType.QUANTIZE,
            cmd=["echo", "test"],
            input_path="/models/test.gguf",
            output_path="/models/test-q4.gguf",
        )

        assert task.status == PipelineTaskStatus.QUEUED

    def test_task_runs_when_no_training(self, patched_state):
        """GPU pipeline tasks run immediately when no training is active."""
        from api.state import PipelineTaskType, PipelineTaskStatus

        popen_patch = _mock_popen()
        with popen_patch:
            task = patched_state._start_pipeline_task(
                task_type=PipelineTaskType.QUANTIZE,
                cmd=["echo", "test"],
                input_path="/models/test.gguf",
                output_path="/models/test-q4.gguf",
            )

        assert task.status == PipelineTaskStatus.RUNNING

    def test_pull_not_blocked_by_training(self, patched_state):
        """Non-GPU tasks (PULL_HF_MODEL) are NOT blocked by training."""
        from api.state import (
            PipelineTaskType, PipelineTaskStatus,
            Job, JobStatus,
        )

        running_job = Job(
            job_id="train-1",
            config_path="test.yaml",
            output_dir="/workspace/training/outputs/test",
            status=JobStatus.RUNNING,
            approved=True,
            created_at=datetime.now(),
            log_file="/workspace/training/logs/test.log",
        )
        patched_state._jobs["train-1"] = running_job

        popen_patch = _mock_popen()
        with popen_patch:
            task = patched_state._start_pipeline_task(
                task_type=PipelineTaskType.PULL_HF_MODEL,
                cmd=["echo", "test"],
                input_path="test/model",
                output_path="/models/test",
            )

        assert task.status == PipelineTaskStatus.RUNNING


class TestPipelineTaskCancellation:
    """Test pipeline task cancellation (DELETE only works on RUNNING tasks)."""

    def test_cancel_running_task(self, client, patched_state):
        """DELETE /api/pipeline/tasks/{id} cancels a running task."""
        from api.state import PipelineTask, PipelineTaskType, PipelineTaskStatus

        task = PipelineTask(
            task_id="cancel-test",
            task_type=PipelineTaskType.QUANTIZE,
            status=PipelineTaskStatus.RUNNING,
            created_at=datetime.now(),
            log_file="/tmp/test.log",
            input_path="/models/test.gguf",
            output_path="/models/test-q4.gguf",
            pid=99999,
        )
        patched_state._pipeline_tasks["cancel-test"] = task

        mock_proc = MagicMock()
        patched_state._pipeline_processes["cancel-test"] = mock_proc

        resp = client.delete("/api/pipeline/tasks/cancel-test")
        assert resp.status_code == 200

    def test_cannot_cancel_completed_task(self, client, patched_state):
        """DELETE on completed task returns 400."""
        from api.state import PipelineTask, PipelineTaskType, PipelineTaskStatus

        task = PipelineTask(
            task_id="done-task",
            task_type=PipelineTaskType.QUANTIZE,
            status=PipelineTaskStatus.COMPLETED,
            created_at=datetime.now(),
            log_file="/tmp/test.log",
            input_path="/models/test.gguf",
            output_path="/models/test-q4.gguf",
        )
        patched_state._pipeline_tasks["done-task"] = task

        resp = client.delete("/api/pipeline/tasks/done-task")
        assert resp.status_code == 400


class TestQueuedTaskPersistence:
    """Test that queued task commands survive serialization."""

    def test_queued_cmd_serialized(self, patched_state):
        """Queued task command is stored in the model and persists to JSON."""
        from api.state import (
            PipelineTaskType, PipelineTaskStatus,
            Job, JobStatus,
        )

        # Running training so task gets queued
        running_job = Job(
            job_id="train-1",
            config_path="test.yaml",
            output_dir="/workspace/training/outputs/test",
            status=JobStatus.RUNNING,
            approved=True,
            created_at=datetime.now(),
            log_file="/workspace/training/logs/test.log",
        )
        patched_state._jobs["train-1"] = running_job

        task = patched_state._start_pipeline_task(
            task_type=PipelineTaskType.QUANTIZE,
            cmd=["echo", "test-command"],
            input_path="/models/test.gguf",
            output_path="/models/test-q4.gguf",
        )

        assert task.status == PipelineTaskStatus.QUEUED
        assert task.queued_cmd == ["echo", "test-command"]

        # Verify it survives JSON roundtrip
        data = json.loads(patched_state.PIPELINE_STATE_FILE.read_text())
        task_data = list(data.values())[0]
        assert task_data["queued_cmd"] == ["echo", "test-command"]
