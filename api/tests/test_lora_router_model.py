"""self.llamolotl#44: the LoRA routes must name a model in router mode.

`GET /lora-adapters` is a router-proxied route: `server_models_routes::proxy_get`
(self.llama/tools/server/server-models.cpp:2423-2432) reads the target from the
`model` QUERY parameter and 400s without it. Every call this API made omitted it
and swallowed the resulting error into an empty list, so:

  - `apply_loras` concluded nothing was preloaded and took the RESTART path on
    every single invocation, while reporting `"method": "restart"` as though it
    had weighed the choice;
  - `GET /api/system/active-loras` fell through to parsing the launch args file
    while labelling the answer `source: "llama-server"`.

Verified live against gemma-4-26B-A4B-it-qat-UD-Q4_K_XL on 2026-08-11:

    GET  /lora-adapters                -> 400 "model name is missing from the request"
    GET  /lora-adapters?model=<name>   -> 200 []

These tests assert on the URL actually handed to urlopen, following
`test_gpu_lease.py::TestUnloadViaRouter`'s convention of capturing the request
rather than trusting the return value -- a function that returns `[]` looks
identical whether it asked correctly or not, which is exactly how this defect
survived.
"""

import json
from unittest.mock import patch

import pytest

import api.state as state


class _Resp:
    """Minimal urlopen context-manager stand-in (test_gpu_lease.py idiom)."""

    def __init__(self, payload: bytes = b"[]"):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def _capturing_urlopen(captured, payload: bytes = b"[]"):
    def _fake(url_or_req, timeout=None):
        captured["url"] = getattr(url_or_req, "full_url", url_or_req)
        captured["method"] = getattr(url_or_req, "get_method", lambda: "GET")()
        captured["body"] = getattr(url_or_req, "data", None)
        return _Resp(payload)

    return _fake


class TestActiveLoraReadNamesTheModel:
    def test_explicit_model_is_sent_as_a_query_parameter(self):
        captured = {}
        with patch("urllib.request.urlopen", _capturing_urlopen(captured)):
            state._get_active_loras_from_server("gemma-4-26B")
        assert captured["url"] == "http://localhost:8080/lora-adapters?model=gemma-4-26B"

    def test_model_is_url_encoded(self):
        """Model ids carry characters that must not leak into the query string."""
        captured = {}
        with patch("urllib.request.urlopen", _capturing_urlopen(captured)):
            state._get_active_loras_from_server("a/b c&d")
        assert captured["url"].endswith("?model=a%2Fb%20c%26d")

    def test_omitted_model_resolves_to_the_primary_loaded_model(self):
        captured = {}
        with patch("api.state._primary_loaded_model", return_value="the-big-one"), \
             patch("urllib.request.urlopen", _capturing_urlopen(captured)):
            state._get_active_loras_from_server()
        assert captured["url"].endswith("?model=the-big-one")

    def test_no_resident_model_returns_empty_without_calling_the_server(self):
        """Nothing resident means no adapters BY DEFINITION -- an honest [],
        as opposed to the swallowed-400 [] this issue is about."""
        captured = {}
        with patch("api.state._primary_loaded_model", return_value=None), \
             patch("urllib.request.urlopen", _capturing_urlopen(captured)):
            assert state._get_active_loras_from_server() == []
        assert "url" not in captured, "should not have contacted llama-server at all"

    def test_the_url_is_never_the_bare_route(self):
        """The regression guard. A bare /lora-adapters is the defect itself."""
        captured = {}
        with patch("api.state._primary_loaded_model", return_value="m"), \
             patch("urllib.request.urlopen", _capturing_urlopen(captured)):
            state._get_active_loras_from_server()
        assert captured["url"] != "http://localhost:8080/lora-adapters"
        assert "model=" in captured["url"]

    def test_a_returned_adapter_list_is_passed_through(self):
        payload = json.dumps([{"id": 0, "path": "/models/x.gguf", "scale": 0.5}]).encode()
        with patch("urllib.request.urlopen", _capturing_urlopen({}, payload)):
            adapters = state._get_active_loras_from_server("m")
        assert adapters == [{"id": 0, "path": "/models/x.gguf", "scale": 0.5}]


class TestFailuresAreLoggedNotErased:
    """The bare `except: return []` is what made the missing parameter
    invisible for as long as it was. [] is still returned -- callers have no
    better option -- but the reason must reach the log."""

    def test_http_error_is_logged_at_warning(self, caplog):
        import urllib.error

        err = urllib.error.HTTPError(
            url="http://localhost:8080/lora-adapters",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=None,
        )
        with caplog.at_level("WARNING"), \
             patch("urllib.request.urlopen", side_effect=err):
            assert state._get_active_loras_from_server("m") == []
        assert any("400" in r.getMessage() for r in caplog.records), caplog.text

    def test_connection_error_is_logged_at_warning(self, caplog):
        with caplog.at_level("WARNING"), \
             patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            assert state._get_active_loras_from_server("m") == []
        assert caplog.records, "a swallowed connection failure must still be logged"


class TestActiveLorasEndpoint:
    def test_model_query_is_forwarded_to_the_server_read(self, client):
        with patch("api.routers.system._get_active_loras_from_server", return_value=[]) as spy:
            client.get("/api/system/active-loras?model=some-model")
        spy.assert_called_once_with("some-model")

    def test_live_answer_names_the_model_it_describes(self, client):
        adapters = [{"id": 0, "path": "/models/a.gguf", "scale": 1.0}]
        with patch("api.routers.system._get_active_loras_from_server", return_value=adapters):
            resp = client.get("/api/system/active-loras?model=some-model")
        data = resp.json()
        assert data["source"] == "llama-server"
        assert data["model"] == "some-model", "a per-model answer must say which model"


class TestApplyLorasCarriesTheModel:
    def test_hot_swap_post_url_carries_the_model(self, client, tmp_path):
        """The POST cannot succeed until the proxy_post patch ships (the child
        requires an array body, the router demands a body object), but it must
        already SEND the model in the query string -- that is the parameter the
        patched proxy_post reads, and sending it now means no second edit here."""
        lora = tmp_path / "character.gguf"
        lora.write_bytes(b"stub")
        resident = [{"id": 0, "path": str(tmp_path / "character.gguf"), "scale": 0.1}]
        captured = {}

        with patch("api.routers.system.MODELS_DIR", tmp_path), \
             patch("api.routers.system._primary_loaded_model", return_value="big-chat-model"), \
             patch("api.routers.system._get_active_loras_from_server", return_value=resident), \
             patch("urllib.request.urlopen", _capturing_urlopen(captured, b"{}")):
            resp = client.post(
                "/api/system/apply-loras",
                json={"loras": [{"file": "character.gguf", "scale": 0.8}]},
            )

        assert resp.status_code == 200
        assert captured["method"] == "POST"
        assert "model=big-chat-model" in captured["url"]
        assert json.loads(captured["body"]) == [{"id": 0, "scale": 0.8}], (
            "body must stay a bare array -- that is what the CHILD handler requires"
        )

    def test_explicit_model_in_the_request_wins(self, client, tmp_path):
        lora = tmp_path / "character.gguf"
        lora.write_bytes(b"stub")
        resident = [{"id": 0, "path": str(tmp_path / "character.gguf"), "scale": 0.1}]
        captured = {}

        with patch("api.routers.system.MODELS_DIR", tmp_path), \
             patch("api.routers.system._primary_loaded_model", return_value="big-chat-model"), \
             patch("api.routers.system._get_active_loras_from_server", return_value=resident), \
             patch("urllib.request.urlopen", _capturing_urlopen(captured, b"{}")):
            client.post(
                "/api/system/apply-loras",
                json={
                    "loras": [{"file": "character.gguf", "scale": 0.8}],
                    "model": "a-different-model",
                },
            )

        assert "model=a-different-model" in captured["url"]

    def test_adapter_read_is_scoped_to_the_target_model(self, client, tmp_path):
        """The read that decides hot-swap-vs-restart must ask about the SAME
        model the apply targets; asking about no model is what made it always
        answer 'nothing loaded'.

        With no adapters resident the endpoint falls through to the restart
        path, which WRITES the args file -- and LLAMA_SERVER_ARGS_FILE is not
        one of conftest's `_PATH_ATTRS`, so `patched_state` does not redirect
        it and the real /app/llama-server.args is attempted. Redirected here
        rather than added to the shared fixture: this is the only test that
        reaches the write, and widening the fixture would change the
        environment of every test that uses it.
        """
        lora = tmp_path / "character.gguf"
        lora.write_bytes(b"stub")

        with patch("api.routers.system.MODELS_DIR", tmp_path), \
             patch("api.routers.system.LLAMA_SERVER_ARGS_FILE", tmp_path / "llama-server.args"), \
             patch("api.routers.system._primary_loaded_model", return_value="big-chat-model"), \
             patch("api.routers.system._get_active_loras_from_server", return_value=[]) as spy, \
             patch("api.routers.system._restart_llama_server", return_value={"ok": True}):
            client.post(
                "/api/system/apply-loras",
                json={"loras": [{"file": "character.gguf", "scale": 0.8}]},
            )

        spy.assert_called_once_with("big-chat-model")
