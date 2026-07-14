"""
self.llamolotl#12: tests for the service-ticket auth layer on the training-
api control port (:8093). Covers the validating side (api/auth.py) plus a
representative sample of the endpoints it now gates in models/system/jobs/
pipeline routers.

Uses a plain, unauthenticated TestClient (not the `client` fixture from
conftest.py, which pre-attaches a valid ticket) so each test controls
exactly what ticket — if any — is sent.
"""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from .conftest import (
    TEST_SERVICE_AUTH_AUDIENCE,
    TEST_SERVICE_AUTH_SECRET,
    mint_test_ticket,
)


@pytest.fixture
def auth_client(patched_state):
    """TestClient with SERVICE_AUTH_SECRET configured but NO default ticket
    attached — each test attaches whatever ticket it wants to exercise."""
    import api.auth as auth_module

    original_secret = auth_module.SERVICE_AUTH_SECRET
    auth_module.SERVICE_AUTH_SECRET = TEST_SERVICE_AUTH_SECRET

    from api.main import app
    c = TestClient(app)

    yield c

    auth_module.SERVICE_AUTH_SECRET = original_secret


class TestMissingOrMalformedTicket:
    def test_no_ticket_header_is_rejected(self, auth_client):
        resp = auth_client.get("/api/models")
        assert resp.status_code == 401
        assert "X-Selfai-Ticket" in resp.json()["detail"]

    def test_malformed_ticket_is_rejected(self, auth_client):
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": "not-a-jwt"}
        )
        assert resp.status_code == 401
        assert "Invalid service ticket" in resp.json()["detail"]

    def test_empty_ticket_header_is_rejected(self, auth_client):
        resp = auth_client.get("/api/models", headers={"X-Selfai-Ticket": ""})
        assert resp.status_code == 401


class TestExpiryAndAudience:
    def test_expired_ticket_is_rejected(self, auth_client):
        now = int(time.time())
        expired = jwt.encode(
            {
                "iss": "self.ai",
                "aud": TEST_SERVICE_AUTH_AUDIENCE,
                "scope": "models:read",
                "iat": now - 600,
                "exp": now - 60,
            },
            TEST_SERVICE_AUTH_SECRET,
            algorithm="HS256",
        )
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": expired}
        )
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_wrong_audience_is_rejected(self, auth_client):
        ticket = mint_test_ticket(
            scope="models:read", audience="self.curator"
        )
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": ticket}
        )
        assert resp.status_code == 401
        assert "audience" in resp.json()["detail"].lower()

    def test_wrong_signing_secret_is_rejected(self, auth_client):
        ticket = mint_test_ticket(secret="not-the-real-secret")
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": ticket}
        )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()


class TestScopeEnforcement:
    def test_correct_scope_is_accepted(self, auth_client):
        ticket = mint_test_ticket(scope="models:read")
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": ticket}
        )
        assert resp.status_code == 200

    def test_missing_scope_is_rejected_with_403(self, auth_client):
        # Valid ticket, correct audience, but wrong/insufficient scope.
        ticket = mint_test_ticket(scope="jobs:read")
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": ticket}
        )
        assert resp.status_code == 403
        assert "models:read" in resp.json()["detail"]

    def test_one_of_several_scopes_is_sufficient(self, auth_client):
        ticket = mint_test_ticket(scope="jobs:read models:read jobs:write")
        resp = auth_client.get(
            "/api/models", headers={"X-Selfai-Ticket": ticket}
        )
        assert resp.status_code == 200

    def test_write_scope_does_not_grant_delete(self, auth_client):
        """A models:write-scoped ticket must not be usable against the
        models:delete endpoint — scopes are per-capability, not hierarchical."""
        ticket = mint_test_ticket(scope="models:write")
        resp = auth_client.post(
            "/api/models/delete",
            json={"name": "does-not-matter.gguf"},
            headers={"X-Selfai-Ticket": ticket},
        )
        assert resp.status_code == 403

    def test_pull_scope_does_not_grant_restart(self, auth_client):
        ticket = mint_test_ticket(scope="models:pull")
        resp = auth_client.post(
            "/api/system/restart-llama-server",
            headers={"X-Selfai-Ticket": ticket},
        )
        assert resp.status_code == 403


class TestNoSecretConfigured:
    def test_missing_server_secret_fails_closed_with_503(self, patched_state):
        """If SERVICE_AUTH_SECRET isn't set on this node at all, every ticket
        check must fail closed (503), never silently accept the request."""
        import api.auth as auth_module

        original_secret = auth_module.SERVICE_AUTH_SECRET
        auth_module.SERVICE_AUTH_SECRET = ""
        try:
            from api.main import app
            c = TestClient(app)
            ticket = mint_test_ticket()
            resp = c.get("/api/models", headers={"X-Selfai-Ticket": ticket})
            assert resp.status_code == 503
        finally:
            auth_module.SERVICE_AUTH_SECRET = original_secret


class TestHealthEndpointsStayOpen:
    """Probes hit these with no ticket; they must never require one."""

    def test_health_requires_no_ticket(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_health_live_requires_no_ticket(self, auth_client):
        resp = auth_client.get("/health/live")
        assert resp.status_code == 200

    def test_health_ready_requires_no_ticket(self, auth_client):
        resp = auth_client.get("/health/ready")
        assert resp.status_code in (200, 200)


class TestGatedAcrossRouters:
    """Spot-check that each router (models/system/jobs/pipeline) actually
    enforces auth end-to-end, not just the one endpoint used above."""

    def test_jobs_create_requires_ticket(self, auth_client):
        resp = auth_client.post(
            "/api/jobs", json={"config_path": "whatever"}
        )
        assert resp.status_code == 401

    def test_jobs_create_with_correct_scope_passes_auth_layer(self, auth_client):
        """Scope check passes; the 400 below comes from business-logic
        validation (config not found), proving auth let the request through."""
        ticket = mint_test_ticket(scope="jobs:create")
        resp = auth_client.post(
            "/api/jobs",
            json={"config_path": "nonexistent-config"},
            headers={"X-Selfai-Ticket": ticket},
        )
        assert resp.status_code != 401
        assert resp.status_code != 403

    def test_pipeline_tasks_requires_ticket(self, auth_client):
        resp = auth_client.get("/api/pipeline/tasks")
        assert resp.status_code == 401

    def test_pipeline_tasks_with_scope_is_accepted(self, auth_client):
        ticket = mint_test_ticket(scope="pipeline:read")
        resp = auth_client.get(
            "/api/pipeline/tasks", headers={"X-Selfai-Ticket": ticket}
        )
        assert resp.status_code == 200

    def test_system_restart_requires_ticket(self, auth_client):
        resp = auth_client.post("/api/system/restart-llama-server")
        assert resp.status_code == 401
