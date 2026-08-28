---
name: code-review
description: Review a change for correctness, edge cases, and readability.
keywords: review, code review, refactor, bug, quality, feedback, critique
---
When reviewing a diff or a set of changes:
- Look for correctness bugs, unhandled edge cases, and race conditions first.
- Check error handling: are exceptions caught, are resources closed, are timeouts
  and retries in place?
- Check for security problems (injection, path traversal, secrets, unsafe
  subprocess use) and note them explicitly.
- Comment on readability and naming only where it materially helps.
- Summarize findings as a prioritized list (blocking / non-blocking) and, when
  asked, apply the fixes yourself with edit_file.
