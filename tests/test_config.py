"""Tests for configuration loading and precedence."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent.config import Config, load_config


class ConfigTestCase(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config({})
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.max_iterations, 40)
        self.assertFalse(cfg.allow_outside_workdir)

    def test_cli_overrides_win(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_config({"model": "cli-model", "max_iterations": 5})
        self.assertEqual(cfg.model, "cli-model")
        self.assertEqual(cfg.max_iterations, 5)

    def test_env_overrides_defaults(self):
        env = {"LLM_API_KEY": "sk-env", "LLM_MODEL": "env-model", "LLM_BASE_URL": "https://example/v1"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config({})
        self.assertEqual(cfg.api_key, "sk-env")
        self.assertEqual(cfg.model, "env-model")
        self.assertEqual(cfg.base_url, "https://example/v1")

    def test_openai_fallback_env(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=True):
            cfg = load_config({})
        self.assertEqual(cfg.api_key, "sk-openai")

    def test_cli_overrides_env(self):
        env = {"LLM_MODEL": "env-model"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config({"model": "cli-model"})
        self.assertEqual(cfg.model, "cli-model")

    def test_numeric_env_override(self):
        env = {"LLM_MAX_ITERATIONS": "12", "LLM_TEMPERATURE": "0.7"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config({})
        self.assertEqual(cfg.max_iterations, 12)
        self.assertEqual(cfg.temperature, 0.7)

    def test_invalid_numeric_env_ignored(self):
        env = {"LLM_MAX_ITERATIONS": "not-a-number"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_config({})
        self.assertEqual(cfg.max_iterations, 40)

    def test_project_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".coding-agent.json").write_text(
                '{"model": "file-model", "max_iterations": 7}'
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("os.getcwd", return_value=tmp):
                    cfg = load_config({})
            self.assertEqual(cfg.model, "file-model")
            self.assertEqual(cfg.max_iterations, 7)

    def test_with_overrides(self):
        cfg = Config().with_overrides(model="new", temperature=None)
        self.assertEqual(cfg.model, "new")
        self.assertEqual(cfg.temperature, Config().temperature)


if __name__ == "__main__":
    unittest.main()
