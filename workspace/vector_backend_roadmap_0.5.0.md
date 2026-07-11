# Vector backend ABC / sqlite-vec retrieval path — 0.5.0 sketch

## Context

memlife 0.4.3 has sqlite-vec working as an **opt-in fallback** behind the
`use_sqlite_vec: bool` flag. The current code in `vec_backend.py` is a set of
free functions called directly from `_embeddings.py` and the recall methods.
This works, but it scatters backend logic across multiple files and makes
adding a third backend (e.g. `BinaryVectorBackend`) awkward.

ROADMAP.md for 0.5.0 already calls for promoting sqlite-vec to a **first-class
pluggable vector backend** via an ABC, replacing the boolean flags with a
single `vector_backend` field on `MemoryConfig`.

## What is already in place

- `src/memlife/vec_backend.py` — free-function adapter around `sqlite_vec`
  with `available()`, `can_load()`, `ensure_schema()`, `store()`, `search()`,
  `delete()`.
- `src/memlife/binary_vectors.py` — binarize / debinarize / hamming helpers.
- `src/memlife/_embeddings.py` — `_serialize_vec`, `_deserialize_vec`,
  `_maybe_store_vec`, `embed_texts`, `backfill_embeddings`, `embedding_health`.
- Recall methods branch on `self.config.use_sqlite_vec`:
  - `_episodes.py`: `recall_episodes_vector` / `_recall_episodes_sqlite_vec`
  - `_facts.py`: `recall_facts` / `_recall_facts_sqlite_vec`
  - `_journal.py`: `recall_journal_vector` / `_recall_journal_sqlite_vec`
- Tests: `tests/test_sqlite_vec*.py` pass (sqlite-vec 0.1.9 installed in venv).
- `pyproject.toml` already has `[project.optional-dependencies] sqlite-vec = ["sqlite-vec"]`.

## Proposed approach for 0.5.0

### 1. Introduce `VectorBackend` ABC

New file: `src/memlife/vectors/backend.py`

```python
from abc import ABC, abstractmethod
from typing import Any

class VectorBackend(ABC):
    """Pluggable vector storage and search."""

    @property
    @abstractmethod
    def dim(self) -> int | None: ...

    @abstractmethod
    def store(self, kind: str, item_id: str, vec: list[float]) -> bool: ...

    @abstractmethod
    def search(self, kind: str, query_vec: list[float], *, limit: int = 20) -> list[tuple[str, float]]: ...

    @abstractmethod
    def delete(self, kind: str, item_id: str) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

class VectorBackendError(Exception): ...
```

### 2. Implement three backends

New package: `src/memlife/vectors/backends/`

- `json.py` — `JsonVectorBackend`
  - Stores vectors in the existing `embedding_json` column.
  - Search is brute-force in Python (current default behavior).
  - `dim` inferred from first stored vector or passed explicitly.

- `binary.py` — `BinaryVectorBackend`
  - Stores binarized vectors in `embedding_json` as `binary:<dim>:<base64>`.
  - Search uses Hamming similarity over decoded vectors.
  - Depends only on `binary_vectors.py`.

- `sqlite_vec.py` — `SqliteVecBackend`
  - Wraps the existing `vec_backend.py` logic.
  - Raises `VectorBackendError` if `sqlite-vec` is not installed or extension
    loading is unavailable.
  - Creates virtual tables per dimension on first use.

### 3. Replace boolean config flags

In `src/memlife/config.py`:

```python
from memlife.vectors.backend import VectorBackend, JsonVectorBackend

@dataclass
class MemoryConfig:
    ...
    vector_backend: VectorBackend | None = None  # default resolved in __post_init__
```

Deprecate (remove in 0.5.0):
- `use_sqlite_vec`
- `use_binary_vectors`

Add env var: `MEMLIFE_VECTOR_BACKEND` with values `json`, `binary`, `sqlite-vec`.

### 4. Centralise vector operations in `MemoryStore`

`MemoryStore.__init__` resolves the backend:

```python
backend = config.vector_backend or JsonVectorBackend()
if isinstance(backend, type):
    backend = backend()  # allow class references
self.vector_backend = backend
backend.attach(self.conn, self.config.embedding_model)
```

Then replace all scattered backend checks:

- `_embeddings.py`: `_maybe_store_vec` → `self.vector_backend.store(...)`
- `_episodes.py`, `_facts.py`, `_journal.py`: branch on backend type or call
  `self.vector_backend.search(...)` uniformly.

### 5. Keep the public API unchanged

`MemoryStore.remember`, `store_fact`, `recall_*`, `backfill_embeddings`, etc.
all keep the same signatures. Only the config mechanism changes.

### 6. Migration / backfill story

- Switching backends does **not** auto-backfill (per ROADMAP decision R6).
- `backfill_embeddings()` becomes the explicit path; it reads from
  `embedding_json` and writes through the active backend.
- Future: add an `embedding_cache` table keyed on `(model_name, text_sha256)`
  so model swaps are cheap. This is a nice-to-have, not a blocker.

### 7. Testing

- Parameterise existing recall tests across `JsonVectorBackend`,
  `BinaryVectorBackend`, and `SqliteVecBackend`.
- Add `tests/test_vectors_backends.py` with backend contract tests:
  - store/search/delete round-trip
  - dimension mismatch raises `VectorBackendError`
  - kind isolation (facts vs episodes vs journal)
  - empty backend returns empty search
- Keep `tests/test_sqlite_vec*.py` as integration tests.

### 8. Docs and packaging

- `docs/vector-backends.md` — comparison table and migration examples.
- Update `README.md` with backend selection example.
- `pyproject.toml` extras: keep `sqlite-vec`, add `all` extra that includes it.

## Suggested first slice (MVP)

To avoid a big-bang refactor, do it in two PRs:

1. **ABC + JSON backend only**
   - Add `vectors/backend.py` and `vectors/backends/json.py`.
   - Add `MemoryConfig.vector_backend` and env var parsing.
   - Route `_maybe_store_vec` and recall paths through the backend instance.
   - Default to `JsonVectorBackend`; existing behavior unchanged.
   - Tests green.

2. **Add sqlite-vec and binary backends**
   - Move existing `vec_backend.py` logic into `vectors/backends/sqlite_vec.py`.
   - Add `vectors/backends/binary.py`.
   - Deprecate/remove `use_sqlite_vec` and `use_binary_vectors`.
   - Parameterise tests.

This keeps each step reviewable and bisectable.

## Open questions for James

1. Do you want the `embedding_cache` table in 0.5.0, or defer to 0.5.1?
2. Should `BinaryVectorBackend` be the new default for fresh installs, or
   stay with JSON for compatibility?
3. Any interest in a `SqliteVecBackend` auto-fallback to JSON when the
   extension is missing, or should it raise a clear error (ROADMAP says
   raise)?
4. Should the ABC live under `memlife.vectors` (new package) or
   `memlife.vector_backends`?
