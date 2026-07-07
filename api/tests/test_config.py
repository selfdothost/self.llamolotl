"""
T-061: Tests for config parsing, field aliasing, and dataset format detection.
"""

import pytest
import yaml


class TestFieldAliasing:
    """Test axolotl-compatible field name mapping."""

    def test_micro_batch_size_alias(self, tmp_path):
        """micro_batch_size maps to per_device_train_batch_size."""
        from api.train import load_config

        config = {"micro_batch_size": 4, "base_model": "test/model"}
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))

        loaded = load_config(str(config_path))
        assert "per_device_train_batch_size" in loaded
        assert loaded["per_device_train_batch_size"] == 4
        assert "micro_batch_size" not in loaded

    def test_num_epochs_alias(self, tmp_path):
        """num_epochs maps to num_train_epochs."""
        from api.train import load_config

        config = {"num_epochs": 3, "base_model": "test/model"}
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))

        loaded = load_config(str(config_path))
        assert "num_train_epochs" in loaded
        assert loaded["num_train_epochs"] == 3

    def test_lr_scheduler_alias(self, tmp_path):
        """lr_scheduler maps to lr_scheduler_type."""
        from api.train import load_config

        config = {"lr_scheduler": "cosine", "base_model": "test/model"}
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))

        loaded = load_config(str(config_path))
        assert loaded["lr_scheduler_type"] == "cosine"

    def test_no_alias_when_target_exists(self, tmp_path):
        """If target field already exists, alias is not applied."""
        from api.train import load_config

        config = {
            "micro_batch_size": 4,
            "per_device_train_batch_size": 8,
            "base_model": "test/model",
        }
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))

        loaded = load_config(str(config_path))
        assert loaded["per_device_train_batch_size"] == 8
        assert loaded["micro_batch_size"] == 4


class TestConfigCRUD:
    """Test config CRUD via API endpoints."""

    def test_create_config(self, client, patched_state):
        """POST /api/configs creates a config file."""
        resp = client.post("/api/configs", json={
            "name": "test-config",
            "content": "base_model: test/model\nnum_epochs: 1\n",
        })
        assert resp.status_code == 201

        config_file = patched_state.CONFIGS_DIR / "test-config.yaml"
        assert config_file.exists()

    def test_list_configs(self, client, patched_state):
        """GET /api/configs lists saved configs."""
        (patched_state.CONFIGS_DIR / "my-config.yaml").write_text("base_model: test\n")

        resp = client.get("/api/configs")
        assert resp.status_code == 200
        configs = resp.json()
        names = [c["name"] for c in configs]
        assert "my-config" in names

    def test_get_config(self, client, patched_state):
        """GET /api/configs/{name} returns config content."""
        (patched_state.CONFIGS_DIR / "read-test.yaml").write_text("base_model: test\n")

        resp = client.get("/api/configs/read-test")
        assert resp.status_code == 200
        assert "base_model" in resp.json()["content"]

    def test_delete_config(self, client, patched_state):
        """DELETE /api/configs/{name} removes the file."""
        config_file = patched_state.CONFIGS_DIR / "del-test.yaml"
        config_file.write_text("base_model: test\n")

        resp = client.delete("/api/configs/del-test")
        assert resp.status_code == 200
        assert not config_file.exists()

    def test_get_nonexistent_config(self, client, patched_state):
        """GET /api/configs/{name} returns 404 for missing config."""
        resp = client.get("/api/configs/does-not-exist")
        assert resp.status_code == 404


class TestDatasetFormatDetection:
    """Test dataset format detection in train.py."""

    def test_format_chat_template_function(self):
        """chat_template format calls apply_chat_template."""
        from api.train import _format_chat_template

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "formatted: " + str(len(messages)) + " messages"

        example = {"messages": [{"role": "user", "content": "hi"}]}
        result = _format_chat_template(example, "messages", FakeTokenizer())
        assert "text" in result
        assert "formatted" in result["text"]

    def test_format_alpaca_function(self):
        """alpaca format creates instruction/response structure."""
        from api.train import _format_alpaca

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return f"msgs={len(messages)}"

        example = {
            "instruction": "Summarize this",
            "input": "Long text here",
            "output": "Summary",
        }
        result = _format_alpaca(example, FakeTokenizer())
        assert "text" in result
