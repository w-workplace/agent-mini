"""Tests for security helpers: redaction, env scrubbing, sandbox argv."""

import os
import unittest
from unittest import mock

from coding_agent.security import (
    build_sandbox_argv,
    command_env,
    detect_sandbox_backend,
    redact,
    redact_obj,
)


class RedactTestCase(unittest.TestCase):
    def test_redact_pem(self):
        pem = "-----BEGIN PRIVATE KEY-----\nsecretbase64\n-----END PRIVATE KEY-----"
        out = redact(pem)
        self.assertIn("[REDACTED PRIVATE KEY]", out)
        self.assertNotIn("secretbase64", out)

    def test_redact_tokens(self):
        self.assertNotIn("sk-abc1234567890", redact("key=sk-abc1234567890"))
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", redact("AKIAIOSFODNN7EXAMPLE"))
        self.assertNotIn("mytoken", redact("Authorization: Bearer mytoken"))

    def test_redact_quoted_assignment(self):
        self.assertIn("api_key=[REDACTED]", redact('api_key = "sk-secret-value"'))
        # A value that is an expression (not a literal) is left alone, so
        # ordinary source code is not mangled.
        self.assertIn('os.environ["X"]', redact('api_key = os.environ["X"]'))

    def test_redact_obj_recursive(self):
        obj = {
            "stdout": "key sk-abc1234567890 here",
            "nested": [{"x": "Authorization: Bearer abc123token"}],
        }
        out = redact_obj(obj)
        self.assertNotIn("sk-abc1234567890", out["stdout"])
        self.assertNotIn("abc123token", out["nested"][0]["x"])


class CommandEnvTestCase(unittest.TestCase):
    def test_scrubs_secrets_and_unknown_vars(self):
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "LLM_API_KEY": "sk-secret",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtn",
            "CUSTOM_VAR": "hello",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            out = command_env("/tmp/work")
        self.assertIn("PATH", out)
        self.assertIn("HOME", out)
        self.assertNotIn("LLM_API_KEY", out)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", out)
        self.assertNotIn("CUSTOM_VAR", out)
        self.assertEqual(out["PWD"], "/tmp/work")

    def test_extra_allow(self):
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin", "MY_VAR": "x"}, clear=True):
            out = command_env("/tmp/work", extra_allow="MY_VAR")
        self.assertEqual(out.get("MY_VAR"), "x")

    def test_sandbox_home_redirect(self):
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/u"}, clear=True):
            out = command_env("/tmp/work", sandbox=True)
        self.assertEqual(out["HOME"], "/tmp/work")
        self.assertEqual(out["TMPDIR"], "/tmp")


class SandboxArgvTestCase(unittest.TestCase):
    def test_bwrap_argv(self):
        argv = build_sandbox_argv("bwrap", "/tmp/work", "echo hi")
        self.assertEqual(argv[0], "bwrap")
        self.assertIn("--unshare-net", argv)
        self.assertIn("--ro-bind", argv)
        self.assertIn("/tmp/work", argv)
        self.assertEqual(argv[-3:], ["/bin/sh", "-c", "echo hi"])

    def test_bwrap_skips_tmpfs_under_tmp(self):
        self.assertNotIn("--tmpfs", build_sandbox_argv("bwrap", "/tmp/work", "echo"))
        self.assertIn("--tmpfs", build_sandbox_argv("bwrap", "/home/u/work", "echo"))

    def test_firejail_argv(self):
        argv = build_sandbox_argv("firejail", "/tmp/work", "echo hi")
        self.assertIn("--net=none", argv)
        self.assertIn("--read-only=/", argv)
        self.assertIn("--read-write=/tmp/work", argv)
        self.assertEqual(argv[-3:], ["/bin/sh", "-c", "echo hi"])

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            build_sandbox_argv("nope", "/tmp", "echo")


class DetectBackendTestCase(unittest.TestCase):
    def test_detection(self):
        with mock.patch("coding_agent.security.shutil.which",
                        side_effect=lambda x: "/usr/bin/bwrap" if x == "bwrap" else None):
            self.assertEqual(detect_sandbox_backend(), "bwrap")
        with mock.patch("coding_agent.security.shutil.which",
                        side_effect=lambda x: "/usr/bin/firejail" if x == "firejail" else None):
            self.assertEqual(detect_sandbox_backend(), "firejail")
        with mock.patch("coding_agent.security.shutil.which", return_value=None):
            self.assertIsNone(detect_sandbox_backend())


if __name__ == "__main__":
    unittest.main()
