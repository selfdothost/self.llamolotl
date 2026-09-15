# Fair prefill scheduling (self.llamolotl#53, manifestations B/C).
#
# Before the fix, the first prompt slot in iteration order filled the entire
# n_batch and every other pending prompt received zero tokens per iteration -
# a concurrently arriving long session sat completely stalled until the first
# prompt finished (measured on a live child: 118 s at zero progress), and any
# slot decoding alongside it crawled at 1 token per prefill-dominated
# iteration.
#
# test_concurrent_prompts_both_progress: two long prompts submitted together
# on a 2-slot server; while the first slot is still mid-prompt, the second
# slot must have made progress too. On the old scheduler the second slot
# stays at exactly 0 until the first prompt completes.

import json
import threading
import time

import pytest
from utils import *


def _get_slots(server):
    raw = server.make_request("GET", "/slots").body
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # n_batch 2 (one token per pending prompt per iteration) makes a ~300-token
    # prompt take hundreds of scheduler iterations, so the poll loop below has
    # a real window to observe the interleave on this tiny CPU-fast model.
    # prompts stay under the model's 512-token trained ctx per sequence.
    # slots endpoint on, unified pool to mirror the router children.
    server.n_ctx = 4096
    server.n_batch = 2
    server.n_predict = 8
    server.server_slots = True
    server.kv_unified = True
    server.temperature = 0.0
    yield server


LONG_PROMPT_A = ("The lighthouse keeper counted the incoming tide and logged "
    "the height of every wave. ") * 30  # ~300 tokens
LONG_PROMPT_B = ("The harbor master reviewed the manifest of every crate and "
    "noted the weight of each pallet. ") * 30


def test_concurrent_prompts_both_progress():
    server.start()

    results = {}

    def send(key, prompt):
        results[key] = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "cache_prompt": False,
            "n_predict": 8,
        })

    t_a = threading.Thread(target=send, args=("a", LONG_PROMPT_A))
    t_b = threading.Thread(target=send, args=("b", LONG_PROMPT_B))

    t_a.start()
    t_b.start()

    deadline = time.time() + 120
    observed = False

    while time.time() < deadline:
        slots = _get_slots(server)

        # note: next_token serializes as a one-element array
        mid_flight = next((s for s in slots
            if s["is_processing"] and s["next_token"][0]["n_decoded"] == 0
            and s["n_prompt_tokens_processed"] > 4), None)

        if mid_flight is not None:
            others = [s for s in slots if s["id"] != mid_flight["id"]]

            # only judge once the second prompt has actually launched -
            # otherwise this is just thread startup jitter
            launched = [s for s in others if s["is_processing"]]

            if launched:
                # the other pending prompt must have tokens processed too,
                # while the first one is still mid-prefill - on the old
                # exclusive scheduler the first prompt's completion is the
                # only thing that unlocks the second slot
                assert max(s["n_prompt_tokens_processed"] for s in launched) > 0, (
                    "second prompt made no progress while first is mid-prefill at "
                    f"{mid_flight['n_prompt_tokens_processed']} tokens"
                )
                observed = True
                break

        time.sleep(0.05)

    t_a.join(timeout=120)
    t_b.join(timeout=120)

    assert observed, "never observed both slots processing concurrently"
    for key in ("a", "b"):
        assert results[key].status_code == 200, f"request {key} failed"
