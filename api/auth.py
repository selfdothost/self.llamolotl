"""
Ticket-based auth for the training-api control port (self.llamolotl#12).

self.ai mints short-lived, scoped JWTs before calling into this control
plane (self.ai#25's proposal: self.ai is the yard's internal ticket-granting
service). This module is the validating side: every mutating/control
endpoint on :8093 requires a valid ticket in the `X-Selfai-Ticket` header —
signature, audience, expiry, and scope are all checked.

First cut is deliberately narrow: self.ai <-> self.llamolotl only. Other
backends (self.curator, self.code-eval, self.language-eval,
self.transcribe) are explicit follow-up scope once this pattern is
proven, not handled here (see self.ai#25).

Scope taxonomy (mirror of self.ai's minting side in
api/selfai_ui/utils/service_auth.py — keep both lists in sync):
  models:read     - list/inspect models, GGUF cache/fasttext status, read
                    integrity sweep findings (GET /api/integrity)
  models:pull     - download a model (HF pull, hf-cache ensure, fasttext
                    ensure), force an integrity sweep (POST /api/integrity/
                    sweep — can trigger an autopull re-download)
  models:delete   - delete a GGUF model file
  models:write    - register a model (symlink into the top-level models dir)
  system:read     - read chat-template / active-lora / health-adjacent state
  system:write    - upload/clear chat template, apply LoRAs
  system:restart  - restart llama-server
  jobs:read       - list/get training jobs, configs, outputs, logs
  jobs:create     - create a new training job (incl. heretic runs)
  jobs:write      - approve/cancel a job, create/delete a config, upload a dataset
  pipeline:read   - list/get pipeline tasks, list available LoRAs
  pipeline:write  - bake/merge/convert/quantize pipeline tasks, cancel a task

NetworkPolicy-level pod-to-pod restriction is a complementary defense-in-
depth layer, explicitly out of scope here (see self.llamolotl#12).
"""

import logging
import os
from typing import Callable, List, Optional

import jwt
from fastapi import Header, HTTPException, status

log = logging.getLogger(__name__)

# Shared HMAC secret with self.ai. Empty by default so a misconfigured
# deployment fails closed (every ticket check 503s) instead of silently
# accepting unsigned requests.
SERVICE_AUTH_SECRET = os.environ.get("SERVICE_AUTH_SECRET", "")
# This service's own audience value — tickets minted for any other
# audience are rejected even if otherwise well-formed and correctly signed.
SERVICE_AUTH_AUDIENCE = os.environ.get("SERVICE_AUTH_AUDIENCE", "self.llamolotl")
SERVICE_AUTH_ALGORITHM = "HS256"

TICKET_HEADER = "X-Selfai-Ticket"


class TicketError(HTTPException):
    """A ticket validation failure. Carries a plain-English detail message
    (never the raw exception) so we don't leak signing internals on the wire."""

    def __init__(self, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(status_code=status_code, detail=detail)


def _decode_ticket(token: str) -> dict:
    if not SERVICE_AUTH_SECRET:
        log.error(
            "SERVICE_AUTH_SECRET is not configured — rejecting all service "
            "tickets on the control port until it is set."
        )
        raise TicketError(
            "Service auth is not configured on this node",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        claims = jwt.decode(
            token,
            SERVICE_AUTH_SECRET,
            algorithms=[SERVICE_AUTH_ALGORITHM],
            audience=SERVICE_AUTH_AUDIENCE,
        )
    except jwt.ExpiredSignatureError:
        raise TicketError("Service ticket expired")
    except jwt.InvalidAudienceError:
        raise TicketError("Service ticket audience mismatch")
    except jwt.InvalidTokenError as e:
        log.warning("Rejected malformed service ticket: %s", e)
        raise TicketError("Invalid service ticket")

    return claims


def _scopes_from_claims(claims: dict) -> List[str]:
    scope = claims.get("scope", "")
    if isinstance(scope, str):
        return scope.split()
    if isinstance(scope, (list, tuple)):
        return list(scope)
    return []


def require_scope(required_scope: str) -> Callable[..., dict]:
    """FastAPI dependency factory.

    Usage: `Depends(require_scope("models:pull"))` on a route. Validates the
    `X-Selfai-Ticket` header: present, correctly signed, not expired,
    audience == this service, and `required_scope` is among the ticket's
    granted scopes. Returns the decoded claims (available to the route via
    the dependency's return value, though most routes don't need it).
    """

    def _dependency(
        x_selfai_ticket: Optional[str] = Header(default=None, alias=TICKET_HEADER),
    ) -> dict:
        if not x_selfai_ticket:
            raise TicketError(f"Missing {TICKET_HEADER} header")

        claims = _decode_ticket(x_selfai_ticket)

        granted = _scopes_from_claims(claims)
        if required_scope not in granted:
            raise TicketError(
                f"Ticket does not grant required scope '{required_scope}'",
                status.HTTP_403_FORBIDDEN,
            )

        return claims

    return _dependency
