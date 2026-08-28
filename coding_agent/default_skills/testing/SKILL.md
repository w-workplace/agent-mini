---
name: testing
description: Write and run tests; treat failing tests as the source of truth.
keywords: test, tests, unit test, coverage, pytest, unittest, assertion, regression
---
Testing conventions:
- Prefer the project's existing test runner and framework (pytest/unittest/etc.).
- For every behavior change, add or update a test; cover both the happy path and
  the failure/edge cases.
- After editing code, run the relevant tests with run_command and fix failures
  before finishing. Do not claim success without running the tests.
- Keep tests fast and deterministic; avoid network/disk dependencies where possible.
