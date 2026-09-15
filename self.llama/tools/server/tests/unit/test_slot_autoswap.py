# Slot persistence across child exit (self.llamolotl#54).
#
# The RAM prompt cache dies with the process, so a model swap or restart
# historically cost every parked session a full re-prefill. With
# --slot-save-path set, the child now flushes non-empty slots to disk on
# graceful exit and restores them on boot:
#
#   test_slot_autoswap_roundtrip: process a prompt, SIGTERM the server
#   (graceful), assert the snapshot file exists, boot a fresh server on the
#   same save path, and assert the same prompt wakes from the restored state
#   (prompt_n == 1) with the snapshot consumed.
#
#   test_slot_autoswap_no_path_no_files: without --slot-save-path a graceful
#   exit writes nothing (feature is opt-in and inert by default).

import pytest
from utils import *

SNAPSHOT = "./tmp/autoswap-tinyllama-2.slot1.llama"


def _autoswap_server():
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    return server


def _cleanup_snapshots():
    if os.path.exists(SNAPSHOT):
        os.remove(SNAPSHOT)
    if os.path.exists(SNAPSHOT + ".tmp"):
        os.remove(SNAPSHOT + ".tmp")


def test_slot_autoswap_roundtrip():
    _cleanup_snapshots()

    server = _autoswap_server()
    server.start()

    # establish a session in slot 1
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_tokens_first = res.body["timings"]["prompt_n"]
    assert n_tokens_first > 0

    # graceful exit (SIGTERM) must flush slot 1's state to disk
    server.stop()
    assert os.path.exists(SNAPSHOT), "graceful exit did not flush the slot snapshot"

    # a fresh instance on the same save path restores it at boot
    server2 = _autoswap_server()
    server2.start()

    # the identical prompt should wake from the restored state: only the last
    # token is re-evaluated (same contract as the in-process prompt cache)
    res = server2.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 1

    # restores are one-shot: the snapshot was consumed on boot
    assert not os.path.exists(SNAPSHOT)


def test_slot_autoswap_no_path_no_files():
    _cleanup_snapshots()

    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    server.stop()

    # without --slot-save-path the exit path writes nothing anywhere
    assert not os.path.exists(SNAPSHOT)
