---
name: git-commit
description: Write clear, conventional commit messages and keep git history tidy.
keywords: commit, git, conventional commit, changelog, release, history
---
Follow the Conventional Commits format: `<type>(<scope>): <subject>`, e.g.
`feat(parser): add markdown frontmatter support`, `fix(cli): handle missing api key`,
`docs: document sandbox flags`, `test: cover skill discovery`, `refactor`, `chore`.

Rules:
- One logical change per commit; keep the subject under ~72 chars, imperative mood.
- Do not commit secrets, build artifacts, or generated files.
- If the repository has a CONTRIBUTING guide or commit conventions, follow those
  instead of these defaults.
- When asked to commit, stage only the relevant files and run `git status`/`git diff`
  first to confirm what changed.
