"""Tests for configuration loading and precedence."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent.config import Config, load_config


class ConfigTestCase(unittest.TestCase):
    def setUp(self):
        # Isolate the workspace so tests never read a real ~/.coding-agent.
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, overrides=None, env=None):
        overrides = dict(overrides or {})
        overrides.setdefault("workspace", str(self.workspace))
        with mock.patch.dict(os.environ, env or {}, clear=True):
            return load_config(overrides)

    def test_defaults(self):
        cfg = self._load()
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.max_iterations, 40)
        self.assertFalse(cfg.allow_outside_workdir)
        self.assertEqual(cfg.workspace, str(self.workspace))
        self.assertFalse(cfg.compact)

    def test_cli_overrides_win(self):
        cfg = self._load({"model": "cli-model", "max_iterations": 5})
        self.assertEqual(cfg.model, "cli-model")
        self.assertEqual(cfg.max_iterations, 5)

    def test_env_overrides_defaults(self):
        env = {"LLM_API_KEY": "sk-env", "LLM_MODEL": "env-model", "LLM_BASE_URL": "https://example/v1"}
        cfg = self._load(env=env)
        self.assertEqual(cfg.api_key, "sk-env")
        self.assertEqual(cfg.model, "env-model")
        self.assertEqual(cfg.base_url, "https://example/v1")

    def test_openai_fallback_env(self):
        cfg = self._load(env={"OPENAI_API_KEY": "sk-openai"})
        self.assertEqual(cfg.api_key, "sk-openai")

    def test_cli_overrides_env(self):
        cfg = self._load({"model": "cli-model"}, env={"LLM_MODEL": "env-model"})
        self.assertEqual(cfg.model, "cli-model")

    def test_numeric_env_override(self):
        cfg = self._load(env={"LLM_MAX_ITERATIONS": "12", "LLM_TEMPERATURE": "0.7"})
        self.assertEqual(cfg.max_iterations, 12)
        self.assertEqual(cfg.temperature, 0.7)

    def test_invalid_numeric_env_ignored(self):
        cfg = self._load(env={"LLM_MAX_ITERATIONS": "not-a-number"})
        self.assertEqual(cfg.max_iterations, 40)

    def test_project_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".coding-agent.json").write_text('{"model": "file-model", "max_iterations": 7}')
            with mock.patch("os.getcwd", return_value=tmp):
                cfg = self._load()
            self.assertEqual(cfg.model, "file-model")
            self.assertEqual(cfg.max_iterations, 7)

    def test_workspace_config_file(self):
        (self.workspace / "config.json").write_text(
            '{"model": "ws-model", "compact": true, "temperature": 0.5}'
        )
        cfg = self._load()
        self.assertEqual(cfg.model, "ws-model")
        self.assertTrue(cfg.compact)
        self.assertEqual(cfg.temperature, 0.5)

    def test_env_overrides_workspace_config(self):
        (self.workspace / "config.json").write_text('{"model": "ws-model"}')
        cfg = self._load(env={"LLM_MODEL": "env-model"})
        self.assertEqual(cfg.model, "env-model")

    def test_cli_workspace_resolution(self):
        # --workspace flag wins over env, and workspace config is read from there.
        target = self.workspace / "custom"
        target.mkdir()
        (target / "config.json").write_text('{"model": "custom-model"}')
        cfg = load_config({"workspace": str(target)})
        self.assertEqual(cfg.model, "custom-model")
        self.assertEqual(cfg.workspace, str(target))

    def test_with_overrides(self):
        cfg = Config().with_overrides(model="new", temperature=None)
        self.assertEqual(cfg.model, "new")
        self.assertEqual(cfg.temperature, Config().temperature)


if __name__ == "__main__":
    unittest.main()
