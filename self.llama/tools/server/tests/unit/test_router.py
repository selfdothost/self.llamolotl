import re
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
#
# Those two hooks return a FIXED reading, which is enough for every test that
# only needs free VRAM to be abundant or hopeless. It is not enough for the
# two eviction tests further down: a fixed reading does not go up when a model
# is evicted, so no eviction can ever make room. Those use a third hook,
# LLAMA_TEST_FAKE_TOTAL_VRAM_MB_FILE, which fakes capacity and lets the server
# derive free VRAM from what is resident. See the note above them.

# Every entry must be one of the presets the router has CACHED -- the test
# server runs --offline, so anything it would have to fetch answers 404 to
# /models/load. "ggml-org/test-model-router-download:F16" used to sit at the
# end of this list and is exactly that case; it is the download fixture, it is
# not among the six cached presets, and it 404'd for every test that touched
# it. Removed rather than fixed: nothing here needs a downloadable model.
# Order matters -- test_router_models_max_still_evicts_with_vram_awareness_enabled
# takes [:3], and the eviction tests below take the largest/smallest by
# measured footprint rather than by position.
VRAM_CANDIDATE_MODELS = [
    "ggml-org/tinygemma3-GGUF:Q8_0",
    "ggml-org/test-model-stories260K:F32",
    "ggml-org/test-model-stories260K-infill:F32",
    "ggml-org/stories15M_MOE:F16",
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


# ── Eviction that actually succeeds (self.llamolotl#34) ────────────────
#
# The two tests below need something the fixed-value fakes above cannot give
# them: a free-VRAM reading that RISES when a model is evicted. With a fixed
# fake, evict_for_vram()'s loop re-queries the same number every iteration, so
# no eviction ever makes room -- it runs itself out of candidates and throws
# the #27 unfittable 503 instead of loading. That is what made these two
# unsatisfiable before (self.llamolotl#34), and CI deselected them.
#
# LLAMA_TEST_FAKE_TOTAL_VRAM_MB_FILE fixes that by faking *capacity* instead of
# free space: the server derives free = total - (estimated footprint of the
# models currently resident), so unloading one genuinely gives the next query
# more room, the way a real driver does.
#
# Both tests then have to place `total` inside a window only as wide as the
# model an eviction frees. Two things make that robust rather than a set of
# magic MiB constants:
#
#   * the footprints are MEASURED off the server itself (_prime_and_measure),
#     never hardcoded, so swapping a test model cannot silently invalidate the
#     arithmetic -- it either still works or fails an explicit assert;
#   * --models-vram-overhead-pct is turned up so every footprint is inflated
#     well clear of the 1 MiB truncation in the reported figures. The smallest
#     test models are ~1 MiB on disk, which leaves no room to aim between; at
#     10x they are tens of MiB and the window is wide. The overhead percentage
#     is applied uniformly, so the RELATIVE sizes the tests reason about are
#     unchanged.
#
# --models-vram-margin-mb is set to 0 for the same reason: the margin is a
# constant added to every comparison, and it is exercised by the tests above.
# Zeroing it here keeps the capacity arithmetic exact.

VRAM_TEST_OVERHEAD_PCT = 900  # 10x on-disk size; see the note above


def _write_total_vram_mb(path: str, mib: int) -> None:
    with open(path, "w") as f:
        f.write(str(mib))


def _prime_and_measure(model: str, total_file: str) -> int:
    """Cache `model` locally and return the server's OWN estimated footprint
    for it, in MiB.

    Both steps are load-bearing. The prime-load with abundant capacity is what
    puts the GGUF in LLAMA_CACHE: estimate_model_footprint_bytes() resolves an
    HF-repo preset offline, so a model that has never been fetched estimates as
    0 ("unknown", self.llamolotl#31) and no VRAM arithmetic is possible. Then,
    with nothing resident, a load against a capacity of 0 MiB is guaranteed to
    hit the #27 terminal error -- whose message carries the exact figure the
    server computed. Reading the number back out beats hardcoding it: a
    hardcoded MiB constant goes quietly wrong the first time a test model or
    the overhead percentage changes.

    Leaves `model` unloaded and `total_file` starved; the caller sets the
    capacity it actually wants to test with.
    """
    _write_total_vram_mb(total_file, 999_999)
    _load_model_and_wait(model, timeout=120)
    unload_res = server.make_request("POST", "/models/unload", data={"model": model})
    assert unload_res.status_code == 200
    _wait_for_model_status(model, {"unloaded"})

    _write_total_vram_mb(total_file, 0)
    res = server.make_request("POST", "/models/load", data={"model": model})
    assert res.status_code == 503, \
        f"expected the unfittable 503 to read {model}'s estimate off, got {res.status_code}: {res.body}"
    match = re.search(r"needs an estimated (\d+) MiB", res.body["error"]["message"])
    assert match, f"could not read an estimated footprint out of: {res.body}"
    est_mib = int(match.group(1))
    assert est_mib > 0, f"{model} estimated at 0 MiB -- not cached? (self.llamolotl#31)"
    return est_mib


def test_router_vram_aware_evicts_one_lru():
    """Free VRAM is too small for a second model -> loading it must evict
    the LRU resident model even though models_max (set high here) hasn't
    been reached. This is issue #22's actual bug: two models both fit under
    the count limit but not both fit in VRAM, and the count-only check
    doesn't catch it."""
    global server
    server.models_max = 4  # high enough that count-based eviction never triggers here
    server.models_vram_margin_mb = 0
    server.models_vram_overhead_pct = VRAM_TEST_OVERHEAD_PCT
    total_file = os.path.join(TMP_DIR, "test_router_evicts_one_total_vram_mb.txt")
    _write_total_vram_mb(total_file, 999_999)
    server.fake_total_vram_mb_file = total_file
    server.start()

    try:
        first, second = VRAM_CANDIDATE_MODELS[:2]
        est_first = _prime_and_measure(first, total_file)
        est_second = _prime_and_measure(second, total_file)

        # `first` must be the larger of the two: after it is evicted the whole
        # capacity is free, and `second` has to fit in it.
        assert est_first >= est_second, \
            f"this test needs {first} ({est_first} MiB) to be the larger model, not {second} ({est_second} MiB)"
        # the aiming window is est_second wide; below ~4 MiB the 1 MiB
        # truncation in the reported figures eats it
        assert est_second >= 4, \
            f"{second} estimates at only {est_second} MiB -- raise VRAM_TEST_OVERHEAD_PCT"

        # Capacity that holds `first` alone with room to spare, but leaves less
        # than `second` needs once `first` is resident:
        #   load first  -> free = total          >= est_first          fits
        #   load second -> free = total - first  == est_second // 2    does NOT fit
        #   evict first -> free = total          >= est_second         fits
        total = est_first + est_second // 2
        _write_total_vram_mb(total_file, total)

        _load_model_and_wait(first, timeout=120)
        assert _get_model_status(first) == "loaded"

        _load_model_and_wait(second, timeout=120)
        assert _get_model_status(second) == "loaded"
        assert _get_model_status(first) == "unloaded"
    finally:
        if os.path.exists(total_file):
            os.remove(total_file)


def test_router_vram_aware_evicts_multiple_lru():
    """A load that needs more room than any single eviction frees must keep
    evicting LRU models until it fits (or nothing is left to evict), not
    stop after the first one."""
    global server
    # count-based eviction must never interfere, whatever the candidate list holds
    server.models_max = len(VRAM_CANDIDATE_MODELS) + 2
    server.models_vram_margin_mb = 0
    server.models_vram_overhead_pct = VRAM_TEST_OVERHEAD_PCT
    total_file = os.path.join(TMP_DIR, "test_router_evicts_multiple_total_vram_mb.txt")
    _write_total_vram_mb(total_file, 999_999)
    server.fake_total_vram_mb_file = total_file
    server.start()

    try:
        # The BIG model is the incoming one here and the small ones are the
        # residents -- the reverse of the single-eviction test above, and
        # required rather than cosmetic. Freeing the last resident has to be
        # what finally makes room, so the incoming model must outweigh the
        # residents it displaces; with a small model incoming, the very first
        # eviction would always be enough on its own and nothing would prove
        # the loop iterates.
        #
        # Which model plays which role is decided from MEASURED footprints, not
        # from position in the list: largest is the incoming one, and residents
        # are taken smallest-first for as long as they still total less than it.
        # Hardcoding the split would rot the moment a fixture model changes size.
        est = {m: _prime_and_measure(m, total_file) for m in VRAM_CANDIDATE_MODELS}
        incoming = max(est, key=lambda m: est[m])

        residents: list[str] = []
        for m in sorted((m for m in est if m != incoming), key=lambda m: est[m]):
            if sum(est[r] for r in residents) + est[m] < est[incoming]:
                residents.append(m)

        assert len(residents) >= 2, (
            f"need at least two residents that together weigh less than {incoming} "
            f"({est[incoming]} MiB) to prove more than one eviction happens; got {est}"
        )
        est_mru = est[residents[-1]]  # largest resident -> loaded last -> evicted last
        assert est_mru >= 4, \
            f"{residents[-1]} estimates at only {est_mru} MiB -- raise VRAM_TEST_OVERHEAD_PCT"

        # Capacity that fits the incoming model only once EVERY resident is
        # gone. With residents r1..rN (ascending) and S = their total:
        #   free after 0 evictions = total - S                  < est_incoming
        #   free after k           = total - (r_k+1 + ... + rN) < est_incoming
        #   free after N           = total                     >= est_incoming
        # The last two lines pin `total` to a window one resident wide: at or
        # above est_incoming, and strictly below est_incoming + rN.
        total = est[incoming] + est_mru // 2
        _write_total_vram_mb(total_file, total)

        for m in residents:
            _load_model_and_wait(m, timeout=120)
        for m in residents:
            assert _get_model_status(m) == "loaded"

        _load_model_and_wait(incoming, timeout=120)

        assert _get_model_status(incoming) == "loaded"
        for m in residents:
            assert _get_model_status(m) == "unloaded"
    finally:
        if os.path.exists(total_file):
            os.remove(total_file)


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


# ── Declared footprints for unmeasurable configurations (#39) ──────────
#
# cold_start_estimate_bytes() disqualifies itself when a preset sets n-cpu-moe,
# cpu-moe or override-tensor, because a GGUF's size stops being an upper bound
# on what reaches the card once weights are deliberately kept in system RAM
# (measured 6.0x over for GLM-4.5-Air-q8_0 at n-cpu-moe=40). Refusing to guess
# is right, but it leaves those configurations with NO admission check until the
# router has loaded them once and measured them -- and the child's own --fit
# cannot cover the gap when n-cpu-moe and n-gpu-layers are both pinned, because
# there is no layer budget left for it to reduce. In production that admitted a
# 21.3 GiB model onto a card with 19.6 GiB free and it died on CUDA OOM.
#
# vram-footprint-mib is the seed for that window: a preset-only option (never
# passed to the child) carrying a footprint somebody measured and wrote down.
#
# The first two tests below are a pair and should be read as one: the same
# preset, the same starved card, differing only by the declaration.

SPLIT_PRESET_MODEL = "model-ncmoe-declared"


def _write_split_preset(path: str, declared_mib: int | None,
                        shed_mib_per_layer: int | None = None) -> None:
    """A preset whose n-cpu-moe disqualifies the file-size estimate, optionally
    carrying a declared footprint. stories15M_MOE is used because it is the one
    cached test model with experts for -ncmoe to act on; the repo/file pair is
    spelled exactly as ServerPreset.stories15m_moe() spells it, because the test
    server runs --offline and only resolves what load_all() has already
    cached."""
    lines = [
        f"[{SPLIT_PRESET_MODEL}]\n",
        "hf-repo = ggml-org/stories15M_MOE\n",
        "hf-file = stories15M_MOE-F16.gguf\n",
        "n-cpu-moe = 1\n",
    ]
    if declared_mib is not None:
        lines.append(f"vram-footprint-mib = {declared_mib}\n")
    if shed_mib_per_layer is not None:
        lines.append(f"vram-shed-mib-per-layer = {shed_mib_per_layer}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def test_router_split_model_without_a_declaration_is_admitted_unchecked():
    """Baseline for #39, and the reason it was filed: with n-cpu-moe set and no
    declared footprint, the model is UNKNOWN -- so a starved card does not stop
    it. This is the hole, asserted deliberately so that closing it elsewhere
    cannot silently change this path too."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_split_undeclared.ini")
    _write_split_preset(preset_path, declared_mib=None)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 1  # hopeless for anything with a known size
    server.start()

    try:
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        assert _get_model_status(SPLIT_PRESET_MODEL) == "loaded", \
            "an unsized configuration must be admitted, not refused on a guess"
    finally:
        os.remove(preset_path)


def test_router_declared_footprint_makes_a_split_model_checkable():
    """The fix: the same preset plus vram-footprint-mib is sized, so the same
    starved card refuses it with the structured 503 instead of admitting it into
    an OOM. Nothing is resident, so there is nothing to evict first."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_split_declared.ini")
    _write_split_preset(preset_path, declared_mib=4096)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 1
    server.start()

    try:
        res = server.make_request("POST", "/models/load", data={"model": SPLIT_PRESET_MODEL})
        assert res.status_code == 503, \
            f"expected the unfittable 503 for a declared footprint, got {res.status_code}: {res.body}"
        _assert_unfittable_503(res.body, SPLIT_PRESET_MODEL)
        # refused before any child was spawned
        assert _get_model_status(SPLIT_PRESET_MODEL) == "unloaded"

        # the declared figure is the one being reasoned about, not a file size
        assert "4096 MiB" in res.body["error"]["message"], \
            f"expected the declared 4096 MiB in the refusal, got: {res.body['error']['message']!r}"
    finally:
        os.remove(preset_path)


def test_router_declared_footprint_does_not_reach_the_child():
    """vram-footprint-mib is preset-only. If it ever leaked into the child's
    argv, llama-server would reject the unknown flag and every declared model
    would fail to start -- so this asserts the load works, not just the plumbing."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_split_child_argv.ini")
    _write_split_preset(preset_path, declared_mib=8)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 999_999  # abundant: the declaration must not refuse
    server.start()

    try:
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        assert _get_model_status(SPLIT_PRESET_MODEL) == "loaded"

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        entry = next(m for m in res.body["data"] if m["id"] == SPLIT_PRESET_MODEL)
        args = entry.get("status", {}).get("args", [])
        assert not any("vram-footprint" in str(a) for a in args), \
            f"preset-only option leaked into the child argv: {args!r}"
    finally:
        os.remove(preset_path)


def test_router_measurement_outranks_a_stale_declaration():
    """A declaration is only a seed. Once the router has measured the
    configuration itself, a wrong declaration must not be able to refuse a load
    that fits.

    This also pins the decision to strip vram-footprint-mib out of
    footprint_key(): the declaration is a claim ABOUT a configuration, not part
    of it, so adding one must not invalidate a measurement already taken for the
    same configuration. If the key included it, the reload below would look like
    a brand-new configuration and the stale declaration would win -- which is
    exactly the 503 this test asserts against.

    fake_measured_mib is required, not a convenience: CI runs against a stub
    CUDA driver, children report "memory": [], and every real measurement is
    therefore 0. Without it there is nothing for a measurement to outrank and
    the test cannot pass on any GPU-less runner."""
    global server
    server.models_max = 4
    server.fake_measured_mib = 64  # what the child would have reported on a card
    preset_path = os.path.join(TMP_DIR, "test_router_split_measured_wins.ini")
    _write_split_preset(preset_path, declared_mib=None)
    server.models_preset = preset_path
    # the file hook, not the static one: free VRAM has to change mid-test, and
    # the static value is read from the environment once at spawn
    fake_vram_file = os.path.join(TMP_DIR, "test_router_split_measured_wins_vram_mb.txt")
    with open(fake_vram_file, "w") as f:
        f.write("999999")
    server.fake_free_vram_mb_file = fake_vram_file
    server.start()

    try:
        # load once so the child reports its real footprint (a few MiB), then
        # unload so nothing is resident to evict later
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        unload_res = server.make_request("POST", "/models/unload", data={"model": SPLIT_PRESET_MODEL})
        assert unload_res.status_code == 200
        _wait_for_model_status(SPLIT_PRESET_MODEL, {"unloaded"})

        # now declare an absurd figure and re-read the preset
        _write_split_preset(preset_path, declared_mib=999_999)
        assert SPLIT_PRESET_MODEL in _get_model_ids(is_reload=True)

        # 8 GiB free: far more than the measured footprint, far less than the
        # declaration. Admitting it proves the measurement won.
        with open(fake_vram_file, "w") as f:
            f.write("8192")
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        assert _get_model_status(SPLIT_PRESET_MODEL) == "loaded", \
            "a measured footprint must outrank a declared one"
    finally:
        os.remove(preset_path)
        if os.path.exists(fake_vram_file):
            os.remove(fake_vram_file)


@pytest.mark.parametrize("declared", ["0", "-1", "not-a-number"])
def test_router_unusable_declaration_is_treated_as_absent(declared):
    """A zero, negative or unparseable declaration is no claim at all -- it must
    fall through to unknown and admit the load, never be read as a footprint of
    zero (which would make every model look free) and never abort the router."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_split_bad_decl.ini")
    with open(preset_path, "w") as f:
        f.write(
            f"[{SPLIT_PRESET_MODEL}]\n"
            "hf-repo = ggml-org/stories15M_MOE\n"
            "hf-file = stories15M_MOE-F16.gguf\n"
            "n-cpu-moe = 1\n"
            f"vram-footprint-mib = {declared}\n"
        )
    server.models_preset = preset_path
    server.fake_free_vram_mb = 1
    server.start()

    try:
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        assert _get_model_status(SPLIT_PRESET_MODEL) == "loaded"
    finally:
        os.remove(preset_path)


# ── Publishing the footprint on /v1/models (self.ai#107) ───────────────
#
# self.ai has to size a VRAM lease request BEFORE asking us to load, and it has
# no way to compute a footprint itself. Publishing ours is what lets it ask the
# broker for the right amount instead of guessing or -- as it does today --
# skipping the broker entirely and OOMing around lower-priority holders.
#
# `source` is published alongside `bytes` because the two are not
# interchangeable: "measured" is what a child reported, "estimated" is a file
# size times an overhead percentage. Anything about to reclaim another tenant's
# VRAM on the strength of this number should be able to tell which it got.


def _footprint_of(model_id: str):
    """The published vram_footprint object for `model_id`, or None if the key is
    absent (which is how UNKNOWN is encoded -- deliberately not a null or a 0)."""
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    entry = next(m for m in res.body["data"] if m["id"] == model_id)
    return entry.get("vram_footprint")


def test_router_publishes_a_declared_footprint():
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_publish_declared.ini")
    _write_split_preset(preset_path, declared_mib=4096)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 999_999
    server.start()

    try:
        fp = _footprint_of(SPLIT_PRESET_MODEL)
        assert fp is not None, "a declared footprint must be published"
        assert fp["bytes"] == 4096 * 1024 * 1024
        assert fp["source"] == "declared"
    finally:
        os.remove(preset_path)


def test_router_omits_the_footprint_when_it_is_unknown():
    """The important half. A 0 or a null would read as "this model is free", and
    a consumer sizing a lease off that would ask for nothing and get it. Absence
    of information is encoded as absence of the key.

    n-cpu-moe with no declaration is exactly that case: the file-size estimate
    disqualifies itself (#39) and nothing has been measured."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_publish_unknown.ini")
    _write_split_preset(preset_path, declared_mib=None)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 999_999
    server.start()

    try:
        res = server.make_request("GET", "/models")
        entry = next(m for m in res.body["data"] if m["id"] == SPLIT_PRESET_MODEL)
        assert "vram_footprint" not in entry, \
            f"unknown must be absent, not published as a value: {entry.get('vram_footprint')!r}"
    finally:
        os.remove(preset_path)


def test_router_publishes_an_estimated_footprint_for_a_cached_dense_model():
    """A dense model keeps nothing in system RAM, so the file-size estimate is a
    legitimate upper bound and gets published as `estimated` -- distinguishable
    from a real measurement by the source field alone.

    Note what is NOT asserted here. The estimate resolves an HF-repo preset to
    its cache path offline (#31), so a model that has never been fetched reports
    unknown -- but CI points LLAMA_CACHE at a PERSISTENT /opt/llama-cache that
    ServerPreset.load_all() has already populated, so there is no never-fetched
    state to observe. The unknown case is covered by
    test_router_omits_the_footprint_when_it_is_unknown instead, which reaches it
    through the -ncmoe gate rather than through an empty cache."""
    global server
    server.models_max = 4
    server.fake_free_vram_mb = 999_999
    server.start()

    model = VRAM_CANDIDATE_MODELS[0]
    fp = _footprint_of(model)
    assert fp is not None, "a cached dense model has a file size to estimate from"
    assert fp["source"] == "estimated", \
        f"CI has a stub CUDA driver so nothing is ever measured here; got {fp!r}"
    assert fp["bytes"] > 0

    # and it stays estimated after a load: on a real card the child would report
    # a measurement here and the source would flip (see the next test, which
    # supplies that reading through the fake hook)
    _load_model_and_wait(model, timeout=120)
    assert _footprint_of(model)["source"] == "estimated"


def test_router_publishes_a_measurement_in_preference_to_a_declaration():
    """Precedence has to be visible, not just internal: a model carrying a
    declaration that has since been measured must publish the MEASUREMENT, or a
    consumer reading this would size its lease off a stale operator claim.

    fake_measured_mib is required here for the same reason as in
    test_router_measurement_outranks_a_stale_declaration -- CI's stub CUDA
    driver means children report "memory": [] and nothing is ever measured."""
    global server
    server.models_max = 4
    server.fake_measured_mib = 64
    preset_path = os.path.join(TMP_DIR, "test_router_publish_measured.ini")
    _write_split_preset(preset_path, declared_mib=4096)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 999_999
    server.start()

    try:
        before = _footprint_of(SPLIT_PRESET_MODEL)
        assert before["source"] == "declared" and before["bytes"] == 4096 * 1024 * 1024

        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)

        after = _footprint_of(SPLIT_PRESET_MODEL)
        assert after["source"] == "measured", f"measurement must win over a declaration: {after!r}"
        assert after["bytes"] == 64 * 1024 * 1024
    finally:
        os.remove(preset_path)


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

    Also stamps `results["ended_at"]` (time.monotonic()) at termination.
    `ended` on its own can't tell "cancelled" apart from "finished by
    itself", so the cancel-isolation test compares that stamp against when
    the cancel was sent rather than racing it -- see
    test_router_explicit_cancel_targets_correct_worker.
    """
    url = f"http://{server.server_host}:{server.server_port}/v1/chat/completions"
    try:
        # ignore_eos stops the model halting on an EOS token, but it does NOT
        # make a stream long-lived: the stories260K fixtures run with
        # n_ctx=1024 shared across the router's children, and a stream still
        # terminates on context exhaustion ("truncated = 1" in the server log)
        # after ~128 tokens regardless of max_tokens. So neither this flag nor
        # any max_tokens value can guarantee a concurrent stream is still alive
        # at an arbitrary later moment -- which is why the isolation check is a
        # timestamp comparison, not a liveness race.
        resp = requests.post(url, json={
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "ignore_eos": True,
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
        results["ended_at"] = time.monotonic()
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
    cancel_sent_at = time.monotonic()
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

    # model B's generation must be unaffected by A's cancel. The check is
    # "B did not end BECAUSE of the cancel", not "B is still running": B is a
    # tiny fixture model on a small shared context, so it legitimately
    # finishes on its own (context exhaustion, "truncated = 1") in well under
    # a second -- often before the cancel is even sent. Asserting liveness
    # here made the test fail whenever B simply won that race, which says
    # nothing about isolation.
    #
    # So: if B already ended before the cancel went out, it provably wasn't
    # the cancel that ended it. Otherwise B must survive a short window past
    # the cancel -- if A's cancel bled across to B's child worker, B would
    # terminate within milliseconds of it, the same way A does.
    b_ended_at = results_b.get("ended_at")
    if b_ended_at is None or b_ended_at > cancel_sent_at:
        assert not _wait_until(
            lambda: results_b.get("ended", False), timeout=1.0
        ), (
            "model B's stream ended right after model A's cancel call -- "
            f"cancel bled across child workers (b={results_b})"
        )

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


####################
# issue #29 -- shed expert layers instead of refusing the load
####################


def _child_args(model_id: str) -> list[str]:
    """The argv the router rendered for a model, from /v1/models. This is what
    proves a shed reached the CHILD rather than only the router's bookkeeping."""
    res = server.make_request("GET", "/v1/models")
    for model in res.body["data"]:
        if model["id"] == model_id:
            return (model.get("status") or {}).get("args", []) or []
    raise AssertionError(f"{model_id!r} not in /v1/models")


def _ncmoe_of(model_id: str) -> int:
    args = _child_args(model_id)
    for flag in ("--n-cpu-moe", "-ncmoe"):
        if flag in args:
            return int(args[args.index(flag) + 1])
    return 0


def test_router_sheds_expert_layers_instead_of_refusing():
    """The point of #29: 'slower but serving' beats 'gone'. A card too small for
    the declared footprint used to raise the terminal 503; with a per-layer shed
    cost declared, the router pushes more experts to system RAM and loads."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_shed.ini")
    # The gap must be CLOSABLE, or the "do not extrapolate past meaning" guard
    # correctly refuses instead and this asserts a shed that cannot happen.
    # With the 512 MiB default margin:
    #     budget  = free - 512          = 2048 - 512 = 1536
    #     deficit = declared - budget   = 4096 - 1536 = 2560
    #     layers  = ceil(2560 / 512)    = 5
    #     guard   : 4096 - 5*512 = 1536 > 0  -> shed is sized, load proceeds
    # A 1 MiB card would need 9 layers (4608 MiB) -- more than the model's whole
    # declared footprint -- which is a refusal, not a shed.
    _write_split_preset(preset_path, declared_mib=4096, shed_mib_per_layer=512)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 2048
    server.start()

    try:
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        assert _get_model_status(SPLIT_PRESET_MODEL) == "loaded", \
            "a model with a declared shed cost must degrade, not be refused"
        assert _ncmoe_of(SPLIT_PRESET_MODEL) > 1, \
            "the shed must reach the child's argv, not just the router's preset"
    finally:
        os.remove(preset_path)


def test_router_without_a_shed_cost_still_refuses():
    """The declared shed cost is what makes the degrade path reachable. Without
    it the model is sized but unsheddable, and refusing is still correct --
    guessing a per-layer cost is how #36 got filed."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_no_shed.ini")
    _write_split_preset(preset_path, declared_mib=4096, shed_mib_per_layer=None)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 1
    server.start()

    try:
        res = server.make_request("POST", "/models/load", data={"model": SPLIT_PRESET_MODEL})
        assert res.status_code == 503, f"expected the terminal refusal, got {res.status_code}"
        msg = json.dumps(res.body)
        assert "does not fit in VRAM" in msg, f"unexpected error wording: {msg!r}"
    finally:
        os.remove(preset_path)


def test_router_refuses_when_shedding_cannot_close_the_gap():
    """A per-layer cost so small that closing the gap would take the declared
    footprint to zero means the linear model has been extrapolated past where it
    means anything. Refuse rather than spawn a child on a number we do not
    believe -- an over-optimistic shed is an OOM, which is the outage #29 exists
    to avoid."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_shed_hopeless.ini")
    _write_split_preset(preset_path, declared_mib=4096, shed_mib_per_layer=1)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 1
    server.start()

    try:
        res = server.make_request("POST", "/models/load", data={"model": SPLIT_PRESET_MODEL})
        assert res.status_code == 503, f"expected a refusal, got {res.status_code}"
    finally:
        os.remove(preset_path)


def test_router_shed_declaration_does_not_reach_the_child():
    """vram-shed-mib-per-layer is preset-only, like vram-footprint-mib. Leaking
    it into argv would make llama-server reject an unknown flag and fail every
    load of a model that declares one."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_shed_leak.ini")
    _write_split_preset(preset_path, declared_mib=4096, shed_mib_per_layer=512)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 65536  # roomy: no shed, so argv is the plain render
    server.start()

    try:
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        args = _child_args(SPLIT_PRESET_MODEL)
        assert not any("shed" in a for a in args), f"shed declaration leaked into argv: {args}"
        assert not any("vram-footprint" in a for a in args), f"footprint leaked into argv: {args}"
    finally:
        os.remove(preset_path)


def test_router_does_not_shed_a_model_that_already_fits():
    """The shed is the last thing tried before refusing, never an optimisation.
    A model with room must spawn with exactly the operator's n-cpu-moe."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_shed_unneeded.ini")
    _write_split_preset(preset_path, declared_mib=64, shed_mib_per_layer=512)
    server.models_preset = preset_path
    server.fake_free_vram_mb = 65536
    server.start()

    try:
        _load_model_and_wait(SPLIT_PRESET_MODEL, timeout=120)
        assert _ncmoe_of(SPLIT_PRESET_MODEL) == 1, \
            "a fitting model must keep the declared n-cpu-moe untouched"
    finally:
        os.remove(preset_path)


# ── pin: LRU must not evict the support models (self.ai#128) ────────────
#
# The bug this closes is quiet rather than loud. An embedder and a reranker are
# by nature the LEAST recently used things running: nothing touches them between
# requests, while the chat model is used constantly. Plain LRU therefore picks
# exactly the models that must stay resident, and nobody notices until the next
# embedding request pays a cold load. `pin = 1` in the preset exempts a model
# from being CHOSEN as the victim; it still COUNTS toward models_max, because a
# pin reserves a slot rather than conjuring one.

# Named for what they stand in for, not for what they are: the point of every
# test below is which ROLE gets evicted. All four are cached presets -- the test
# server runs --offline, so anything else answers 404 to /models/load.
PIN_EMBEDDER = "pin-embedder"
PIN_CHAT_A = "pin-chat-a"
PIN_CHAT_B = "pin-chat-b"


def _write_pin_preset(path: str, pin_embedder: bool, also_pin_chat_a: bool = False) -> None:
    """One embedder plus two chat models, with the embedder optionally pinned.

    The `pin_embedder=False` variant is not dead weight -- it is the control that
    proves the eviction being asserted is really LRU order and really changed by
    the pin, rather than something incidental about load sequence.
    """
    lines = [
        f"[{PIN_EMBEDDER}]\n",
        "hf-repo = ggml-org/models\n",
        "hf-file = bert-bge-small/ggml-model-f16.gguf\n",
        "embeddings = 1\n",
    ]
    if pin_embedder:
        lines.append("pin = 1\n")
    lines += [
        f"\n[{PIN_CHAT_A}]\n",
        "hf-repo = ggml-org/test-model-stories260K\n",
    ]
    if also_pin_chat_a:
        lines.append("pin = 1\n")
    lines += [
        f"\n[{PIN_CHAT_B}]\n",
        "hf-repo = ggml-org/test-model-stories260K-infill\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def test_router_unpinned_embedder_is_evicted_first():
    """The control, and the bug being fixed. Loaded first and never touched
    again, the embedder is the LRU entry, so a second chat model evicts it."""
    global server
    server.models_max = 2
    preset_path = os.path.join(TMP_DIR, "test_router_pin_control.ini")
    _write_pin_preset(preset_path, pin_embedder=False)
    server.models_preset = preset_path
    server.start()

    try:
        _load_model_and_wait(PIN_EMBEDDER, timeout=120)
        _load_model_and_wait(PIN_CHAT_A, timeout=120)
        _load_model_and_wait(PIN_CHAT_B, timeout=120)

        assert _get_model_status(PIN_EMBEDDER) == "unloaded", \
            "without a pin the idle embedder is the LRU victim -- if this ever " \
            "stops holding, the pin test below proves nothing"
        assert _get_model_status(PIN_CHAT_B) == "loaded"
    finally:
        os.remove(preset_path)


def test_router_pinned_embedder_survives_a_chat_model_swap():
    """The fix: same load order, same models_max, `pin = 1` on the embedder.
    The swap now costs chat-a -- the thing actually being replaced -- and the
    embedder stays resident."""
    global server
    server.models_max = 2
    preset_path = os.path.join(TMP_DIR, "test_router_pin_survives.ini")
    _write_pin_preset(preset_path, pin_embedder=True)
    server.models_preset = preset_path
    server.start()

    try:
        _load_model_and_wait(PIN_EMBEDDER, timeout=120)
        _load_model_and_wait(PIN_CHAT_A, timeout=120)
        _load_model_and_wait(PIN_CHAT_B, timeout=120)

        assert _get_model_status(PIN_EMBEDDER) == "loaded", \
            "a pinned model must never be chosen as the LRU victim"
        assert _get_model_status(PIN_CHAT_B) == "loaded"
        assert _get_model_status(PIN_CHAT_A) == "unloaded", \
            "the eviction must still happen -- to the unpinned model"
    finally:
        os.remove(preset_path)


def test_router_pin_reserves_a_slot_it_does_not_create_one():
    """A pin is an exemption from eviction, NOT extra capacity. With every slot
    pinned there is no victim, so the load is refused rather than quietly
    exceeding models_max and over-committing the card."""
    global server
    server.models_max = 2
    preset_path = os.path.join(TMP_DIR, "test_router_pin_no_victim.ini")
    _write_pin_preset(preset_path, pin_embedder=True, also_pin_chat_a=True)
    server.models_preset = preset_path
    server.start()

    try:
        _load_model_and_wait(PIN_EMBEDDER, timeout=120)
        _load_model_and_wait(PIN_CHAT_A, timeout=120)

        res = server.make_request("POST", "/models/load", data={"model": PIN_CHAT_B})
        assert res.status_code != 200, \
            f"models_max must hold when nothing is evictable, got {res.status_code}: {res.body}"

        # Both pinned models are untouched -- a refused load must not have
        # evicted anything on its way to failing.
        assert _get_model_status(PIN_EMBEDDER) == "loaded"
        assert _get_model_status(PIN_CHAT_A) == "loaded"
        assert _get_model_status(PIN_CHAT_B) == "unloaded"
    finally:
        os.remove(preset_path)


def test_router_pin_does_not_reach_the_child():
    """pin is preset-only. If it leaked into the child's argv, llama-server
    would reject the unknown flag and every pinned model would fail to start --
    so this asserts the load SUCCEEDS, not merely that the string is absent."""
    global server
    server.models_max = 4
    preset_path = os.path.join(TMP_DIR, "test_router_pin_child_argv.ini")
    _write_pin_preset(preset_path, pin_embedder=True)
    server.models_preset = preset_path
    server.start()

    try:
        _load_model_and_wait(PIN_EMBEDDER, timeout=120)
        assert _get_model_status(PIN_EMBEDDER) == "loaded"
        args = _child_args(PIN_EMBEDDER)
        # Matched exactly, not by substring: argv carries cache paths, and a
        # bare "pin" in a filename would fail this test for no reason.
        assert not any(a == "--pin" or a.startswith("--pin=") for a in args), \
            f"preset-only option leaked into the child argv: {args!r}"
    finally:
        os.remove(preset_path)


# ── pin under VRAM pressure: a preference, not a veto (self.ai#128) ──────
#
# unload_lru() treats a pin as absolute -- there the alternative is only
# exceeding a bookkeeping limit. evict_for_vram() must not, because there the
# alternative is a refused load: measured on the deployed 4090 the two pinned
# support models hold 1308 MiB even at n-gpu-layers=0 (CUDA context plus
# KV/compute buffers), and an absolute pin would make the largest model
# permanently unloadable. So: prefer unpinned, fall back to pinned only when
# nothing else is left.


def _write_pin_solo_preset(path: str) -> None:
    """One PINNED model and one unpinned model -- the fallback fixture. With
    only the pinned one resident, a load that needs the room has no unpinned
    candidate to take.

    Roles are assigned by SIZE, not by what the model is for: the pinned entry
    must be the SMALLER of the two. The capacity this test aims at is
    `incoming + pinned // 2`, so a pinned model larger than the incoming one
    cannot even be loaded into it in the first place. Measured under
    VRAM_TEST_OVERHEAD_PCT, stories260K estimates at 11 MiB and bert-bge-small
    at 254 MiB -- so stories260K is the pinned one here, and the embedder plays
    the incoming model. That is the reverse of the deployment's roles and the
    reverse of the sibling test above; the mechanism under test is the eviction
    choice, which does not care what a model is used for.
    """
    with open(path, "w") as f:
        f.writelines([
            f"[{PIN_CHAT_A}]\n",
            "hf-repo = ggml-org/test-model-stories260K\n",
            "pin = 1\n",
            f"\n[{PIN_EMBEDDER}]\n",
            "hf-repo = ggml-org/models\n",
            "hf-file = bert-bge-small/ggml-model-f16.gguf\n",
            "embeddings = 1\n",
        ])


def test_router_vram_eviction_prefers_the_unpinned_model():
    """With an unpinned model available, VRAM pressure must take THAT one --
    even though the pinned support model is older and is what plain LRU would
    pick. This is the bug: unload_lru() was taught about pins and
    evict_for_vram() was not, so a pinned model still got evicted by the other
    door."""
    global server
    server.models_max = 4  # high: count-based eviction must never interfere here
    server.models_vram_margin_mb = 0
    server.models_vram_overhead_pct = VRAM_TEST_OVERHEAD_PCT
    preset_path = os.path.join(TMP_DIR, "test_router_pin_vram_prefers.ini")
    _write_pin_preset(preset_path, pin_embedder=True)
    server.models_preset = preset_path
    total_file = os.path.join(TMP_DIR, "test_router_pin_vram_prefers_total.txt")
    _write_total_vram_mb(total_file, 999_999)
    server.fake_total_vram_mb_file = total_file
    server.start()

    try:
        est = {m: _prime_and_measure(m, total_file)
               for m in (PIN_EMBEDDER, PIN_CHAT_A, PIN_CHAT_B)}
        # `first` must be the larger chat model: after it alone is evicted,
        # `second` has to fit in what is left.
        first, second = sorted((PIN_CHAT_A, PIN_CHAT_B), key=lambda m: est[m], reverse=True)
        assert est[second] >= 4, \
            f"{second} estimates at only {est[second]} MiB -- raise VRAM_TEST_OVERHEAD_PCT"

        #   load pinned -> free = total                     >= pinned    fits
        #   load first  -> free = total - pinned            >= first     fits
        #   load second -> free = second // 2               <  second    does NOT fit
        #   evict first -> free = first + second // 2       >= second    fits
        total = est[PIN_EMBEDDER] + est[first] + est[second] // 2
        _write_total_vram_mb(total_file, total)

        _load_model_and_wait(PIN_EMBEDDER, timeout=120)
        _load_model_and_wait(first, timeout=120)
        _load_model_and_wait(second, timeout=120)

        assert _get_model_status(PIN_EMBEDDER) == "loaded", \
            "VRAM pressure took the pinned model while an unpinned one was available"
        assert _get_model_status(first) == "unloaded", \
            "the unpinned model is the one that should have paid for the room"
        assert _get_model_status(second) == "loaded"
    finally:
        for p in (preset_path, total_file):
            if os.path.exists(p):
                os.remove(p)


def test_router_vram_eviction_falls_back_to_a_pinned_model():
    """When NOTHING unpinned is left, the pin yields rather than refusing the
    load. Deliberate, and the reason this differs from unload_lru(): an
    absolute pin here turns 'this model is slower to serve' into 'this model
    can never load', which is a worse failure than a support model taking a
    cold start.

    Unlike its sibling above, this one does NOT fail against the unfixed code:
    with a single resident model, plain LRU evicts it too. It is a guard on the
    DECISION, not a reproduction of the bug -- it fails the day someone
    "completes" the pin by making it absolute here."""
    global server
    server.models_max = 4
    server.models_vram_margin_mb = 0
    server.models_vram_overhead_pct = VRAM_TEST_OVERHEAD_PCT
    preset_path = os.path.join(TMP_DIR, "test_router_pin_vram_fallback.ini")
    _write_pin_solo_preset(preset_path)
    server.models_preset = preset_path
    total_file = os.path.join(TMP_DIR, "test_router_pin_vram_fallback_total.txt")
    _write_total_vram_mb(total_file, 999_999)
    server.fake_total_vram_mb_file = total_file
    server.start()

    try:
        # PIN_CHAT_A is the PINNED one here and PIN_EMBEDDER the incoming --
        # roles assigned by size, see _write_pin_solo_preset.
        est_pin = _prime_and_measure(PIN_CHAT_A, total_file)
        est_incoming = _prime_and_measure(PIN_EMBEDDER, total_file)
        assert est_pin >= 2, \
            f"{PIN_CHAT_A} estimates at only {est_pin} MiB -- raise VRAM_TEST_OVERHEAD_PCT; " \
            "below 2 MiB the pin // 2 aiming window collapses"
        assert est_incoming >= est_pin, \
            f"this test needs the incoming model ({est_incoming} MiB) to be no smaller than " \
            f"the pinned one ({est_pin} MiB) -- the capacity below is sized for the incoming " \
            "model, so a larger pinned model could not be loaded into it to begin with"

        #   load pinned   -> free = total                    >= pinned    fits
        #   load incoming -> free = incoming - ceil(pin/2)   <  incoming  does NOT fit
        #   evict pinned  -> free = incoming + pin // 2      >= incoming  fits
        total = est_incoming + est_pin // 2
        _write_total_vram_mb(total_file, total)

        _load_model_and_wait(PIN_CHAT_A, timeout=120)
        assert _get_model_status(PIN_CHAT_A) == "loaded"

        _load_model_and_wait(PIN_EMBEDDER, timeout=120)
        assert _get_model_status(PIN_EMBEDDER) == "loaded", \
            "an absolute pin would have refused this load with the #27 terminal error"
        assert _get_model_status(PIN_CHAT_A) == "unloaded", \
            "the pin must yield when it is the only thing left to evict"
    finally:
        for p in (preset_path, total_file):
            if os.path.exists(p):
                os.remove(p)
