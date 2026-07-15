<!-- codex-python-env:start -->
## Python environment

This project uses a local Python environment.

Always run Python commands through:

```bash
./dev.sh python ...
./dev.sh pip ...
./dev.sh pytest ...
```

Do not use bare `python`, `pip`, `pytest`, `mypy`, or `ruff`.

If `.venv` exists, use it. Do not create another virtual environment.
If `.venv` is missing, run `./dev.sh bootstrap` only after confirmation.
<!-- codex-python-env:end -->

<!-- protected-local-docs:start -->
## Protected local working documents

Execution task files, implementation/result records, handoff logs, acceptance reports, and personal operation/usage instructions are durable local artifacts. They are intentionally ignored by Git and must not be treated as disposable files.

- Never delete, rename, truncate, overwrite, stage, force-add, or commit these documents during cleanup, repository organization, cache removal, or test-data removal.
- A file being ignored or untracked does not make it safe to delete.
- If a matching document is already tracked, remove only its Git index entry and preserve the working-tree file.
- Modify or remove one of these documents only when the user explicitly identifies that document or explicitly requests changes to protected local documents.
- Before any cleanup, inventory matching files and preserve them in place.

Protected naming families include:

```text
*_TASK.md
*_PLAN.md
*_RESULT.md
*_REPORT.md
*_HANDOFF.md
*_OPERATIONS.md
*_INSTRUCTIONS.md
*任务*.md
*任务书*.md
*实施说明*.md
*执行说明*.md
*结果*.md
*验收*.md
*报告*.md
*续接*.md
*交接*.md
*操作说明*.md
*使用说明*.md
*使用说明*.txt
```

Public repository documents such as `README.md`, `RELEASE_NOTES_CN.md`, `LICENSE`, dependency files, and source-code documentation remain version-controlled unless the user explicitly says otherwise.
<!-- protected-local-docs:end -->
