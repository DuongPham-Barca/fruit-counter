# Repository Guidelines

## Project Structure & Module Organization

This checkout has no application source, tests, assets, or dependency manifest yet. The existing `.venv/` is a local Python environment; the other hidden directories contain no project files. Put new application code in a clearly named top-level package or `src/`, tests in `tests/`, and static files in `assets/` if needed. Keep test paths aligned with the code they cover. Do not commit `.venv/` or generated files.

## Build, Test, and Development Commands

No build, run, lint, or test command is defined. For local Python work, `source .venv/bin/activate` activates the existing environment, but no dependencies are declared. When adding the first runnable component, commit its dependency manifest and document exact setup, run, and verification commands in a README. Prefer commands that work from the repository root.

## Coding Style & Naming Conventions

No formatter or linter is configured. For Python code, follow PEP 8: use four-space indentation, `snake_case` for modules and functions, and `PascalCase` for classes. Add a formatter or linter configuration alongside the first substantial code contribution, then apply it consistently. Use descriptive names that reflect the component's purpose.

## Testing Guidelines

There is no test framework or coverage requirement yet. Add focused tests with each behavior change, including failure cases. Name Python test files `test_*.py` under `tests/`; if another language is introduced, document its test conventions and command. Avoid tests that only repeat implementation details.

## Commit & Pull Request Guidelines

The checkout has no usable Git history, so no established commit format can be inferred. Use a short, imperative commit subject that describes the change. In pull requests, explain the purpose, summarize verification performed, link relevant issues, and include screenshots for visible interface changes. Note any new configuration or setup steps.
