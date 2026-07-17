# memlife Release Checklist

Use this before any GitHub tag or PyPI upload. Skipping steps is how
regressions, stale docs, and version mismatches end up in a release.

## 1. Pre-release code state

- [ ] Branch is `main` and working tree is clean.
- [ ] All intended changes are committed.
- [ ] No `git diff` against the last tag that contains uncommitted release
      mechanics (version bumps, CHANGELOG edits must be in a commit).

## 2. Automated checks

Run these in the dev environment first:

```bash
cd /home/ezyrider/PROJECTS/ACTIVE/memlife
python -m ruff check src tests
python -m pytest tests/ -q
```

- [ ] `ruff check src tests` passes with no errors.
- [ ] `pytest tests/` passes: check the final count against the last green run.
      A drop in passed count means something silently stopped running.

## 3. Clean-environment install and smoke test

This catches missing dependencies, stale editable installs, and packaging
metadata problems that the dev venv hides.

```bash
cd /tmp
python -m venv memlife-release-test
source memlife-release-test/bin/activate
pip install /home/ezyrider/PROJECTS/ACTIVE/memlife/dist/memlife-*.whl[ollama]
python - <<'PY'
import memlife
print("version:", memlife.__version__)
from memlife import MemoryStore, MemoryConfig, DummyEmbedder
store = MemoryStore(config=MemoryConfig(db_path="/tmp/memlife_release.db"), embedder=DummyEmbedder())
store.remember("smoke test", "success")
import asyncio
facts = asyncio.run(store.recall_facts("smoke"))
print("facts:", len(facts))
store.close()
PY
```

- [ ] Package installs cleanly from wheel/sdist in a fresh venv.
- [ ] `memlife.__version__` matches the intended release.
- [ ] Basic store/create/recall/close works with no extra setup.

## 4. README verification

- [ ] `README.md` "Current version" badge/line matches `pyproject.toml` and
      `src/memlife/__init__.py`.
- [ ] Every `MemoryConfig` field mentioned in README examples exists in
      `src/memlife/config.py`. Grep for field names and verify.
- [ ] Every code block in README can be executed as written. Run them in a
      fresh Python interpreter, including the "No-LLM Mode" example.
- [ ] MCP tool list in README matches the actual tools exposed by
      `memlife-mcp-server --help`.
- [ ] Resources vs tools are correctly distinguished (e.g.
      `memlife://contradictions` is a resource, not a tool).

## 5. CHANGELOG and versioning

- [ ] New release section exists in `CHANGELOG.md`.
- [ ] Version bumped in `pyproject.toml`.
- [ ] Version bumped in `src/memlife/__init__.py`.
- [ ] README "Current version" updated.
- [ ] CHANGELOG describes user-visible changes in plain language, not only
      developer internals.

## 6. Build artifacts

```bash
rm -rf dist build
python -m build
```

- [ ] `dist/` contains exactly one wheel and one sdist for the new version.
- [ ] No stale artifacts from previous releases remain.

## 7. Tag and GitHub release

- [ ] Commit the version/CHANGELOG/README changes.
- [ ] Create an annotated tag: `git tag -a vX.Y.Z -m "memlife X.Y.Z: <summary>"`.
- [ ] Push the commit: `git push origin main`.
- [ ] Push the tag: `git push origin vX.Y.Z`.
- [ ] **Do not force-push tags.** PyPI artifacts are immutable; a retagged
      commit that does not match the uploaded files creates permanent
      inconsistency.

## 8. PyPI upload

```bash
python -m twine upload dist/memlife-X.Y.Z-*
```

- [ ] Upload succeeds.
- [ ] Verify the new version appears at `https://pypi.org/project/memlife/X.Y.Z/`.
- [ ] Verify `pip install memlife==X.Y.Z` installs the correct version.

## 9. MCP / OpenClaw smoke test (if MCP server changed)

- [ ] `memlife-mcp-server --help` runs and shows the expected flags.
- [ ] `openclaw mcp probe memlife` reports the expected tool/resource counts.
- [ ] A representative tool (`memory_store`, `memory_retrieve`, `memory_reflect`)
      runs end-to-end via OpenClaw without timeout or crash.

## 10. Post-release rollback awareness

- [ ] PyPI files cannot be overwritten. If a release is broken, the fix is a
      new version number, not a re-upload.
- [ ] If a critical bug is found after PyPI upload, immediately bump to the
      next patch version and run this checklist again.

---

## Notes

- This checklist exists because several 0.5.x releases shipped with version
  mismatches, stale README examples, or regressions that full pytest runs did
  not catch. The extra steps here are cheaper than a post-release scramble.
- If an item is not applicable to a particular release (e.g. no MCP changes),
  mark it N/A and note why.
