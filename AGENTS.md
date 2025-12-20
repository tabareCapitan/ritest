# AI / Agent Usage in `ritest`

This repository uses AI coding assistants (so far Codex) to:
- Propose refactors or docstrings.
- Suggest tests or example code.
- Speed up writing documentation.
- Help with boilerplate (CI, packaging files, etc.).
- Others.

## Principles

- All AI-generated changes are **reviewed and edited** by a human before commit.
- No secrets, credentials, or private datasets are ever pasted into AI tools.
- AI tools may suggest code, but do not have authority over:
  - Statistical design choices.
  - Core algorithmic decisions.
  - Public API stability.

## Practical guidelines

- Use AI for:
  - Rewriting docstrings, comments, or small helper functions.
  - Generating repetitive boilerplate (e.g. CI YAML, packaging config).
- Do **not** use AI for:
  - Introducing silent behavior changes without tests.
  - Writing code that you don’t understand.
- Always:
  - Run tests after applying AI suggestions.
  - Edit wording so that docstrings and comments read like human-written,
    domain-aware code (no “AI-speak”). Readability is paramount.
