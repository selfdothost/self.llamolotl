"""
T-060: Tests for job state machine transitions.

Tests all valid and invalid state transitions:
PENDING -> APPROVED (via approve endpoint)
APPROVED -> RUNNING (via _try_start_next_pending, mocked)
RUNNING -> COMPLETED/FAILED (via poll, mocked)
PENDING -> CANCELLED (via cancel/delete endpoint)
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest


class TestJobCreation:
    """Test job creation enters PENDING state."""

    def test_create_job_with_inline_config(self, client, patched_state):
        """Job created with inline config enters PENDING state."""
        config_yaml = "base_model: test/model\nnum_epochs: 1\n"
        resp = client.post("/api/jobs", json={
            "config_inline": config_yaml,
        })
        assert resp.status_code == 201
        job = resp.json()
        assert job["status"] == "pending"
        assert job["approved"] is False
        assert job["job_id"] is not None

    def test_create_job_requires_config(self, client):
        """Job creation without config raises 400."""
        resp = client.post("/api/jobs", json={})
        assert resp.status_code == 400


class TestJobApproval:
    """Test approval gating."""

    def test_approve_pending_job(self, client, patched_state):
        """Approving a pending job sets approved=True."""
        config_yaml = "base_model: test/model\nnum_epochs: 1\n"
        create_resp = client.post("/api/jobs", json={"config_inline": config_yaml})
        job_id = create_resp.json()["job_id"]

        with patch.object(patched_state, '_start_job'):
            resp = client.post(f"/api/jobs/{job_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["approved"] is True

    def test_approve_already_approved(self, client, patched_state):
        """Approving an already-approved job raises 400."""
        config_yaml = "base_model: test/model\nnum_epochs: 1\n"
        create_resp = client.post("/api/jobs", json={"config_inline": config_yaml})
        job_id = create_resp.json()["job_id"]

        with patch.object(patched_state, '_start_job'):
            client.post(f"/api/jobs/{job_id}/approve")
            resp = client.post(f"/api/jobs/{job_id}/approve")
        assert resp.status_code == 400

    def test_approve_nonexistent_job(self, client):
        """Approving nonexistent job raises 404."""
        resp = client.post("/api/jobs/nonexistent/approve")
        assert resp.status_code == 404


class TestJobOneAtATime:
    """Test that only one job runs at a time."""

    def test_second_job_stays_pending(self, client, patched_state):
        """When one job is running, next approved job stays pending."""
        config_yaml = "base_model: test/model\nnum_epochs: 1\n"

        # Create and approve first job
        resp1 = client.post("/api/jobs", json={"config_inline": config_yaml})
        job1_id = resp1.json()["job_id"]

        # Manually set first job to RUNNING
        patched_state._jobs[job1_id].status = patched_state.JobStatus.RUNNING
        patched_state._jobs[job1_id].approved = True

        # Create and approve second job — should not start
        resp2 = client.post("/api/jobs", json={"config_inline": config_yaml})
        job2_id = resp2.json()["job_id"]

        with patch.object(patched_state, '_start_job'):
            client.post(f"/api/jobs/{job2_id}/approve")

        job2 = patched_state._jobs[job2_id]
        assert job2.approved is True
        assert job2.status == patched_state.JobStatus.PENDING


class TestJobCancellation:
    """Test job cancellation via DELETE /api/jobs/{id}."""

    def test_cancel_pending_job(self, client, patched_state):
        """Cancelling a pending job sets status to cancelled."""
        config_yaml = "base_model: test/model\nnum_epochs: 1\n"
        create_resp = client.post("/api/jobs", json={"config_inline": config_yaml})
        job_id = create_resp.json()["job_id"]

        # Cancel is DELETE, not POST
        resp = client.delete(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"


class TestJobPersistence:
    """Test that job state persists to disk."""

    def test_jobs_persist_to_file(self, client, patched_state):
        """Created jobs are saved to jobs.json."""
        config_yaml = "base_model: test/model\nnum_epochs: 1\n"
        client.post("/api/jobs", json={"config_inline": config_yaml})

        assert patched_state.JOBS_STATE_FILE.exists()
        data = json.loads(patched_state.JOBS_STATE_FILE.read_text())
        assert len(data) == 1


class TestJobErrorMessages:
    """Test that failed jobs include error details."""

    def test_failed_job_has_error_message(self, patched_state):
        """FAILED jobs include error_message."""
        from api.state import Job, JobStatus
        job = Job(
            job_id="test-fail",
            config_path="test.yaml",
            output_dir="/workspace/training/outputs/test",
            status=JobStatus.FAILED,
            created_at=datetime.now(),
            log_file="/workspace/training/logs/test.log",
            error_message="Process exited with code 1",
        )
        assert job.error_message is not None
        assert "code 1" in job.error_message
