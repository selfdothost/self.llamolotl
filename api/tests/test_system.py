"""
F-017: Tests for health, system, and chat template endpoints.
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest


class TestHealthEndpoint:
    """Test composite health endpoint."""

    def test_health_returns_composite_status(self, client):
        """GET /health returns api_healthy + inference_healthy."""
        with patch("api.routers.system._check_inference_health", return_value=(False, None)), \
             patch("api.routers.system._check_gpu", return_value=(False, None, None)):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_healthy"] is True
        assert "inference_healthy" in data
        assert "status" in data

    def test_health_degraded_when_inference_down(self, client):
        """Health status is 'degraded' when inference server not responding."""
        with patch("api.routers.system._check_inference_health", return_value=(False, None)), \
             patch("api.routers.system._check_gpu", return_value=(False, None, None)):
            resp = client.get("/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["inference_healthy"] is False

    def test_health_ok_when_inference_up(self, client):
        """Health status is 'ok' when inference server is healthy."""
        with patch("api.routers.system._check_inference_health", return_value=(True, "/models/test.gguf")), \
             patch("api.routers.system._get_active_loras_from_server", return_value=[]), \
             patch("api.routers.system._check_gpu", return_value=(True, 4.0, 24.0)):
            resp = client.get("/health")
        assert resp.json()["status"] == "ok"
        assert resp.json()["inference_healthy"] is True

    def test_health_includes_gpu_info(self, client):
        """Health response includes GPU availability and memory."""
        with patch("api.routers.system._check_inference_health", return_value=(False, None)), \
             patch("api.routers.system._check_gpu", return_value=(True, 4.2, 24.0)):
            resp = client.get("/health")
        data = resp.json()
        assert data["gpu_available"] is True
        assert data["gpu_memory_used_gb"] == 4.2
        assert data["gpu_memory_total_gb"] == 24.0

    def test_health_includes_disk_info(self, client):
        """Health response includes disk space."""
        with patch("api.routers.system._check_inference_health", return_value=(False, None)), \
             patch("api.routers.system._check_gpu", return_value=(False, None, None)):
            resp = client.get("/health")
        data = resp.json()
        assert "disk_models_free_gb" in data
        assert "disk_workspace_free_gb" in data


class TestLivenessReadiness:
    """Test liveness and readiness probes."""

    def test_liveness_always_alive(self, client):
        """GET /health/live always returns alive."""
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_when_inference_up(self, client):
        """GET /health/ready returns ready when inference is up."""
        with patch("api.routers.system._check_inference_health", return_value=(True, None)):
            resp = client.get("/health/ready")
        assert resp.json()["status"] == "ready"

    def test_readiness_when_inference_down(self, client):
        """GET /health/ready returns not_ready when inference is down."""
        with patch("api.routers.system._check_inference_health", return_value=(False, None)):
            resp = client.get("/health/ready")
        assert resp.json()["status"] == "not_ready"


class TestChatTemplate:
    """Test chat template upload, get, clear, and built-in list."""

    def test_upload_template(self, client, patched_state):
        """POST /api/system/chat-template stores template."""
        with patch("api.routers.system._restart_llama_server", return_value={"status": "restarted"}), \
             patch("api.routers.system.CHAT_TEMPLATE_OVERRIDE") as mock_path:
            mock_path.write_text = MagicMock()
            resp = client.post("/api/system/chat-template", json={
                "content": "{% for msg in messages %}{{ msg.content }}{% endfor %}",
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "applied"

    def test_get_template_when_not_set(self, client):
        """GET /api/system/chat-template returns status."""
        resp = client.get("/api/system/chat-template")
        assert resp.status_code == 200

    def test_list_builtin_templates(self, client):
        """GET /api/system/chat-templates/builtin returns template list."""
        resp = client.get("/api/system/chat-templates/builtin")
        assert resp.status_code == 200
        templates = resp.json()["templates"]
        assert len(templates) > 30
        assert "chatml" in templates
        assert "llama3" in templates

    def test_upload_requires_content(self, client):
        """POST /api/system/chat-template without content fails validation."""
        resp = client.post("/api/system/chat-template", json={})
        assert resp.status_code == 422


class TestPathTraversal:
    """Test path traversal protection across endpoints."""

    def test_validate_path_blocks_traversal(self):
        """_validate_path raises on directory traversal attempts."""
        from api.state import _validate_path
        from fastapi import HTTPException
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Normal path should work
            result = _validate_path("myconfig", root, suffix=".yaml")
            assert str(result).startswith(str(root))

            # Traversal should raise
            with pytest.raises(HTTPException) as exc_info:
                _validate_path("../../etc/passwd", root, suffix=".yaml")
            assert exc_info.value.status_code == 400

            with pytest.raises(HTTPException) as exc_info:
                _validate_path("../secrets", root)
            assert exc_info.value.status_code == 400
