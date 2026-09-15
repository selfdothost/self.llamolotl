import os
import pytest
from filelock import FileLock
from utils import *


@pytest.fixture(scope="session", autouse=True)
def configure_worker_port(request):
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    if worker_id != "master":
        worker_num = int(worker_id[2:])
        os.environ["PORT"] = str(8080 + worker_num * 10)

# Servers started with --slot-save-path flush an "autoswap-<alias>.slot<N>.llama"
# snapshot for every non-empty slot on graceful exit (self.llamolotl#54), and the
# next server with the same alias restores those snapshots at boot. Across tests
# that share the default "./tmp" save dir + "tinyllama-2" alias, a previous
# test's shutdown would otherwise leak its slot state into the next test's
# fresh server (observed as a prompt-cache hit in test_slot_erase). Wipe the
# auto-snapshots before each test so every test starts from clean slots.
@pytest.fixture(autouse=True)
def clean_slot_autoswap_snapshots():
    if os.path.isdir("./tmp"):
        for name in os.listdir("./tmp"):
            if name.startswith("autoswap-"):
                os.remove(os.path.join("./tmp", name))

# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets(configure_worker_port, tmp_path_factory):
    # this will be run once per test session, before any tests

    # serialize model downloads across parallel workers.
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / "load_all.lock")):
        ServerPreset.load_all()
