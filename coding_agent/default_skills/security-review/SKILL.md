---
name: security-review
description: Audit code for common vulnerabilities and harden where possible.
keywords: security, vulnerability, auth, secret, sql injection, xss, sandbox, exploit, input validation
---
Security checklist for code you write or review:
- Validate and sanitize all external input (command args, file paths, user data);
  never build shell commands by string concatenation of untrusted values.
- Use parameterized queries; avoid string-built SQL. Escape or encode output to
  prevent injection (SQL/XSS/command/path).
- Never hard-code secrets; read them from environment or a secrets store.
- Restrict file access to expected directories; resolve symlinks and check paths.
- Principle of least privilege: drop permissions, sandbox subprocesses, set
  timeouts and resource limits.
- Report any finding as a blocking issue with a concrete fix.
