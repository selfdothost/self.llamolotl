import threading
import pytest
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_router_props():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.models_vram_aware = False
    server.models_vram_margin_mb = 256
    server.models_vram_overhead_pct = 15
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["role"] == "router"
    assert res.body["max_instances"] == 2
    assert res.body["models_autoload"] is False
    assert res.body["models_vram_aware"] is False
    assert res.body["models_vram_margin_mb"] == 256
    assert res.body["models_vram_overhead_pct"] == 15
    assert res.body["build_info"].startswith("b")


@pytest.mark.parametrize(
    "model,success",
    [
        ("ggml-org/tinygemma3-GGUF:Q8_0", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(model: str, success: bool):
    global server
    server.start()
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_ids(is_reload: bool) -> set[str]:
    res = server.make_request("GET", "/models" + ("?reload=1" if is_reload else ""))
    assert res.status_code == 200
    return {item["id"] for item in res.body.get("data", [])}


def _get_model_status(model_id: str) -> str:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id)
        if last_status in desired:
            return last_status
        time.sleep(1)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout)


def test_router_unload_model():
    global server
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K:F32",
        "ggml-org/test-model-stories260K-infill:F32",
    ]

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


# ── VRAM-aware eviction (issue #22) ─────────────────────────────────────
#
# query_free_vram_bytes() in server-models.cpp checks
# LLAMA_TEST_FAKE_FREE_VRAM_MB(_FILE) before shelling out to nvidia-smi, so
# these tests can deterministically force "free VRAM" up or down without
# real GPU memory pressure -- CI may run these router tests on GPU-less
# hosts with no nvidia-smi at all. See ServerProcess.fake_free_vram_mb(_file)
# in tests/utils.py for how that's plumbed through.

VRAM_CANDIDATE_MODELS = [
    "ggml-org/tinygemma3-GGUF:Q8_0",
    "ggml-org/test-model-stories260K:F32",
    "ggml-org/test-model-stories260K-infill:F32",
    "ggml-org/test-model-router-download:F16",
]


def test_router_vram_aware_load_fits_without_eviction():
    """Free VRAM comfortably covers both models -> VRAM-aware eviction must
    not kick in, and models_max (set generously here) doesn't interfere
    either. This is the "everything fits" baseline the other VRAM tests
    contrast with."""
    global server
    server.models_max = 4
    server.fake_free_vram_mb = 999_999  # far more than these tiny test models need
    server.start()

    first, second = VRAM_CANDIDATE_MODELS[:2]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"


def test_router_vram_aware_evicts_one_lru():
    """Free VRAM is too small for a second model -> loading it must evict
    the LRU resident model even though models_max (set high here) hasn't
    been reached. This is issue #22's actual bug: two models both fit under
    the count limit but not both fit in VRAM, and the count-only check
    doesn't catch it."""
    global server
    server.models_max = 4  # high enough that count-based eviction never triggers here
    server.fake_free_vram_mb = 1  # smaller than any test model's estimated footprint
    server.start()

    first, second = VRAM_CANDIDATE_MODELS[:2]

    _load_model_and_wait(first, timeout=120)
    assert _get_model_status(first) == "loaded"

    _load_model_and_wait(second, timeout=120)
    assert _get_model_status(second) == "loaded"
    assert _get_model_status(first) == "unloaded"


def test_router_vram_aware_evicts_multiple_lru():
    """A load that needs more room than any single eviction frees must keep
    evicting LRU models until it fits (or nothing is left to evict), not
    stop after the first one."""
    global server
    server.models_max = 4  # high enough that count-based eviction doesn't interfere
    fake_vram_file = os.path.join(TMP_DIR, "test_router_fake_free_vram_mb.txt")
    with open(fake_vram_file, "w") as f:
        f.write("999999")  # plenty of room while the first three load
    server.fake_free_vram_mb_file = fake_vram_file
    server.start()

    try:
        residents = VRAM_CANDIDATE_MODELS[:3]
        for m in residents:
            _load_model_and_wait(m, timeout=120)
        for m in residents:
            assert _get_model_status(m) == "loaded"

        # starve free VRAM -- the next load must evict all three residents to
        # make room, one at a time, not just the single LRU
        with open(fake_vram_file, "w") as f:
            f.write("1")

        fourth = VRAM_CANDIDATE_MODELS[3]
        _load_model_and_wait(fourth, timeout=120)

        assert _get_model_status(fourth) == "loaded"
        for m in residents:
            assert _get_model_status(m) == "unloaded"
    finally:
        if os.path.exists(fake_vram_file):
            os.remove(fake_vram_file)


# ── VRAM-unfittable terminal load error (issue #27, R6 / AC3, AC4) ──────
#
# The contrasting happy path -- a resident model IS present, so the incoming
# load evicts the LRU and succeeds -- is already covered by
# test_router_vram_aware_evicts_one_lru above (and the multi-eviction case by
# test_router_vram_aware_evicts_multiple_lru). These two tests cover the
# terminal case those don't: nothing is left to evict and the model still
# doesn't fit, which must surface as a specific, structured HTTP 503 (not a
# generic 500 that would slip through into a real OOM), carrying the model
# name and both the estimated-footprint and free-VRAM figures so a proxy
# (self.ai -> self.chat) can forward the reason unchanged.


def _prime_estimable_then_unload(model: str, fake_vram_file: str) -> None:
    """Load `model` once (so its GGUF is cached locally and
    estimate_model_footprint_bytes() returns non-zero), then unload it so NO
    resident LOADED model is left for evict_for_vram() to reclaim. Priming
    while free VRAM is abundant keeps the prime-load itself from tripping the
    VRAM check. After this returns, starve `fake_vram_file` to force the
    unfittable path on the next load."""
    with open(fake_vram_file, "w") as f:
        f.write("999999")  # abundant room for the prime-load
    _load_model_and_wait(model, timeout=120)
    unload_res = server.make_request("POST", "/models/unload", data={"model": model})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model, {"unloaded"})
    with open(fake_vram_file, "w") as f:
        f.write("1")  # starve: smaller than any test model's footprint


def _assert_unfittable_503(body, model: str) -> None:
    """Assert `body` is the structured OpenAI-style error envelope
    format_error_response()/ex_wrapper produce for the T-004 exception:
    {"error": {"code": 503, "message": ..., "type": "unavailable_error"}}."""
    assert isinstance(body, dict), f"expected a JSON error object, got: {body!r}"
    assert "error" in body, f"expected an 'error' envelope, got: {body!r}"
    err = body["error"]
    assert err.get("type") == "unavailable_error", f"unexpected error type: {err!r}"
    msg = err.get("message", "")
    # T-004 format_message embeds all three facts AC3 requires
    assert model in msg, f"error message must name the model, got: {msg!r}"
    assert "does not fit in VRAM" in msg, f"unexpected error wording: {msg!r}"
    assert "estimated" in msg, f"error message must report an estimated footprint, got: {msg!r}"
    assert "MiB" in msg, f"error message must report MiB figures, got: {msg!r}"
    assert "is free" in msg, f"error message must report the free-VRAM figure, got: {msg!r}"


def test_router_vram_aware_unfittable_load_returns_503():
    """POST /models/load for an estimable model that cannot fit in free VRAM
    with NO resident model to evict must return a structured HTTP 503 -- not a
    generic 500 -- naming the model plus its estimated footprint and the free
    VRAM (issue #27, R6/AC3, AC4). The load is invoked synchronously inside
    the ex_wrapper-wrapped handler, so evict_for_vram()'s throw becomes the
    HTTP response rather than a background failure."""
    global server
    server.models_max = 4  # high enough that count-based eviction never interferes
    fake_vram_file = os.path.join(TMP_DIR, "test_router_unfittable_free_vram_mb.txt")
    server.fake_free_vram_mb_file = fake_vram_file
    server.start()

    try:
        model = VRAM_CANDIDATE_MODELS[0]
        _prime_estimable_then_unload(model, fake_vram_file)

        res = server.make_request("POST", "/models/load", data={"model": model})

        # AC3: a specific error, not a generic 500
        assert res.status_code == 503, \
            f"expected 503 for an unfittable load, got {res.status_code}: {res.body}"
        # AC4: structured envelope a proxy can forward, carrying all three facts
        _assert_unfittable_503(res.body, model)

        # the throw happens before any child instance is created, so the model
        # must be left UNLOADED (not stuck LOADING / half-loaded)
        assert _get_model_status(model) == "unloaded"
    finally:
        if os.path.exists(fake_vram_file):
            os.remove(fake_vram_file)


def test_router_vram_aware_unfittable_autoload_chat_returns_503():
    """The same terminal 503 must surface through the autoload path used by
    /v1/chat/completions (router_validate_model -> ensure_model_ready ->
    load), not only the explicit /models/load handler -- both go through the
    single ex_wrapper catch (T-006). A chat request that triggers an
    unfittable autoload must return the structured 503, not a generic 500."""
    global server
    server.models_max = 4
    fake_vram_file = os.path.join(TMP_DIR, "test_router_unfittable_autoload_free_vram_mb.txt")
    server.fake_free_vram_mb_file = fake_vram_file
    server.start()

    try:
        model = VRAM_CANDIDATE_MODELS[0]
        _prime_estimable_then_unload(model, fake_vram_file)

        res = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
            },
        )

        assert res.status_code == 503, \
            f"expected 503 for an unfittable autoload, got {res.status_code}: {res.body}"
        _assert_unfittable_503(res.body, model)

        assert _get_model_status(model) == "unloaded"
    finally:
        if os.path.exists(fake_vram_file):
            os.remove(fake_vram_file)


def test_router_vram_aware_disabled_does_not_evict():
    """--no-models-vram-aware turns the new eviction layer off entirely:
    models_max is still the (only) limit, so a tiny (fake) free-VRAM
    reading must not force an eviction below the count limit."""
    global server
    server.models_max = 4
    server.models_vram_aware = False
    server.fake_free_vram_mb = 1
    server.start()

    first, second = VRAM_CANDIDATE_MODELS[:2]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"


def test_router_models_max_still_evicts_with_vram_awareness_enabled():
    """VRAM-aware eviction is layered ON TOP of the count-based models_max
    limit, not a replacement for it: with free VRAM reported as abundant
    (so the VRAM check alone would never trigger an eviction), models_max
    must still evict the LRU model once the count limit is reached -- the
    pre-existing behavior covered by test_router_models_max_evicts_lru must
    keep working unchanged now that VRAM-awareness is layered on top."""
    global server
    server.models_max = 2
    server.fake_free_vram_mb = 999_999  # VRAM check alone would never evict here
    server.start()

    first, second, third = VRAM_CANDIDATE_MODELS[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    _load_model_and_wait(third, timeout=120)
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_reload_models():
    """POST /models/reload re-reads the INI preset and updates the model list."""
    global server

    preset_path = os.path.join(TMP_DIR, "test_reload.ini")

    # Initial preset: two models
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-a]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "\n"
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
        )

    server.models_preset = preset_path
    server.start()

    ids = _get_model_ids(is_reload=False)
    assert "model-reload-a" in ids
    assert "model-reload-b" in ids

    # Updated preset: remove a, keep b unchanged, add c
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
            "\n"
            "[model-reload-c]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
        )

    try:
        ids = _get_model_ids(is_reload=True)
        assert "model-reload-a" not in ids, "removed model should no longer appear"
        assert "model-reload-b" in ids, "unchanged model should still appear"
        assert "model-reload-c" in ids, "newly added model should appear"
    finally:
        os.remove(preset_path)


def test_router_remote_preset():
    global server
    server.model_hf_repo = "ggml-org/test-preset-ci"
    server.model_hf_file = None
    server.offline = False
    server.start()

    # Should see preset models in GET /models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    ids = {item["id"] for item in res.body.get("data", [])}
    assert "tinygemma3-preset" in ids
    assert "stories260K-test" in ids

    # Should be able to load a preset model
    model_id = "tinygemma3-preset"
    _load_model_and_wait(model_id)


MODEL_DOWNLOAD_ID = "ggml-org/test-model-router-download:F16"
MODEL_DOWNLOAD_TIMEOUT = 30


def _listen_sse(
    server: ServerProcess, collected: list, stop: threading.Event, ready: threading.Event | None = None
):
    """Collect /models/sse events into `collected` until `stop` is set.

    When `ready` is provided, it is set once the streaming response is open,
    i.e. the server has accepted the connection and registered us as a
    subscriber. Callers that trigger one-shot events (e.g. download_finished)
    must wait on `ready` before acting, otherwise the event can be broadcast
    before this client is subscribed and be lost.
    """
    url = f"http://{server.server_host}:{server.server_port}/models/sse"
    try:
        with requests.get(url, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT) as resp:
            if ready is not None:
                ready.set()
            for line_bytes in resp.iter_lines():
                if stop.is_set():
                    break
                line = line_bytes.decode("utf-8")
                if line.startswith("data: "):
                    collected.append(json.loads(line[6:]))
    except Exception:
        pass


def _wait_for_sse_event(collected: list, event_type: str, model: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e.get("event") == event_type and e.get("model") == model for e in collected):
            return True
        time.sleep(0.5)
    return False


def test_router_download_model():
    """Case 1: download a model, verify SSE events and GET /models."""
    global server
    server.start()

    # Ensure the model is not present before we start
    server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")

    sse_events: list = []
    stop = threading.Event()
    sse_ready = threading.Event()
    sse_thread = threading.Thread(
        target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
    )
    sse_thread.start()

    # wait for the SSE client to be subscribed before triggering the download,
    # otherwise the one-shot download_finished event can be broadcast before
    # this client is registered and be lost
    assert sse_ready.wait(10), "SSE client failed to connect"

    # Trigger the download
    res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
    assert res.status_code == 200
    assert res.body.get("success") is True

    # Wait for download_finished SSE event
    finished = _wait_for_sse_event(
        sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
    )
    stop.set()

    assert finished, "Never received download_finished SSE event"
    assert any(
        e.get("event") == "download_progress" and e.get("model") == MODEL_DOWNLOAD_ID
        for e in sse_events
    ), "No download_progress events received"

    # Model should now appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID in ids, f"{MODEL_DOWNLOAD_ID} not found in /models after download"


def test_router_delete_model():
    """Case 2: delete the downloaded model, verify it disappears from GET /models."""
    global server
    server.start()

    # Ensure the model exists (download it if needed)
    if MODEL_DOWNLOAD_ID not in _get_model_ids(is_reload=False):
        sse_events: list = []
        stop = threading.Event()
        sse_ready = threading.Event()
        threading.Thread(
            target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
        ).start()
        # subscribe before triggering the download so the one-shot
        # download_finished event is not lost (see test_router_download_model)
        assert sse_ready.wait(10), "SSE client failed to connect"
        res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
        assert res.status_code == 200
        finished = _wait_for_sse_event(
            sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
        )
        stop.set()
        assert finished, "Model did not finish downloading before delete test"

    # Delete the model
    del_res = server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")
    assert del_res.status_code == 200
    assert del_res.body.get("success") is True

    # Model should no longer appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID not in ids, f"{MODEL_DOWNLOAD_ID} still present after deletion"


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _router_stream_chat_and_collect(server: ServerProcess, model: str, content: str, results: dict) -> None:
    """Background-thread worker: streams a chat completion through the
    router, records the completion id from the first chunk into
    `results["id"]`, counts content chunks into `results["n_chunks"]`, and
    marks `results["ended"] = True` once the stream terminates (naturally,
    via explicit cancel, or via error). Any exception lands in
    `results["error"]` instead of being raised, since this runs off the main
    test thread.
    """
    url = f"http://{server.server_host}:{server.server_port}/v1/chat/completions"
    try:
        resp = requests.post(url, json={
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }, stream=True)
        try:
            if resp.status_code != 200:
                results["error"] = f"status {resp.status_code}: {resp.text}"
                return
            for line_bytes in resp.iter_lines():
                if not line_bytes:
                    continue
                line = line_bytes.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                if "[DONE]" in line:
                    break
                data = json.loads(line[6:])
                if "id" not in results and data.get("id"):
                    results["id"] = data["id"]
                choices = data.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    results["n_chunks"] = results.get("n_chunks", 0) + 1
        finally:
            resp.close()
    except Exception as e:
        results["error"] = str(e)
    finally:
        results["ended"] = True


def test_router_explicit_cancel_targets_correct_worker():
    """self.ai#39 R2: with two different models loaded on two separate
    router-spawned child workers, an explicit
    POST /v1/chat/completions/control {"action": "cancel", "model": A}
    call must stop only model A's in-flight generation, reaching the correct
    child worker through the router's existing generic proxy_post routing
    (routing by the "model" field in the request body -- the same routing
    /v1/chat/completions itself already uses, see server-models.cpp
    router_validate_model). It must not touch model B's concurrently-running
    generation on the other child worker.

    This exercises R2's router-mode acceptance criterion: no router-specific
    forwarding code was written for this feature, because proxy_post's
    existing "model" routing plus the plain (non-streaming) request/response
    shape of a control call already carries it to the right child -- this
    test is what would catch it if that assumption were wrong.
    """
    global server
    server.n_slots = 1
    server.start()

    model_a = "ggml-org/test-model-stories260K:F32"
    model_b = "ggml-org/test-model-stories260K-infill:F32"
    _load_model_and_wait(model_a, timeout=120)
    _load_model_and_wait(model_b, timeout=120)

    results_a: dict = {}
    results_b: dict = {}
    t_a = threading.Thread(
        target=_router_stream_chat_and_collect,
        args=(server, model_a, "Tell a very long story about apples", results_a),
        daemon=True,
    )
    t_b = threading.Thread(
        target=_router_stream_chat_and_collect,
        args=(server, model_b, "Tell a very long story about oranges", results_b),
        daemon=True,
    )
    t_a.start()
    t_b.start()

    assert _wait_until(lambda: "id" in results_a and "id" in results_b, timeout=30.0), \
        f"both concurrent streams should have started and reported an id, got a={results_a}, b={results_b}"
    assert results_a["id"] != results_b["id"]

    # cancel only model A's completion, explicitly, while both are still generating
    cancel_res = server.make_request("POST", "/v1/chat/completions/control", data={
        "id": results_a["id"],
        "action": "cancel",
        "model": model_a,
    })
    assert cancel_res.status_code == 200
    assert cancel_res.body.get("success") is True

    # model A's stream should end promptly
    assert _wait_until(lambda: results_a.get("ended", False), timeout=5.0), \
        "model A's stream did not end promptly after explicit cancel"
    assert "error" not in results_a, f"model A's stream ended with an error: {results_a.get('error')}"
    # ended well short of the full 512 max_tokens requested
    assert results_a.get("n_chunks", 0) < 512

    # model B's generation must be unaffected by A's cancel: it should still
    # be running (not already marked "ended") right after A's cancel landed
    assert results_b.get("ended", False) is False, \
        "model B's stream must not have been cancelled by model A's cancel call"

    # cleanup: cancel B too so this test doesn't leave a background
    # generation running past teardown
    if "id" in results_b:
        server.make_request("POST", "/v1/chat/completions/control", data={
            "id": results_b["id"],
            "action": "cancel",
            "model": model_b,
        })
    t_a.join(timeout=10)
    t_b.join(timeout=10)
