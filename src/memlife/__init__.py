"""memlife — memory that degrades gracefully.

Four-tier lifecycle memory for AI agents:
  Episode -> Fact -> Journal -> Decay/Prune

Unified scoring: relevance x confidence x recency.
No-LLM mode: store, retrieve, decay, and GC work without any model.
Only reflection needs an LLM.

Quickstart:
    from memlife import MemoryStore, MemoryConfig
    store = MemoryStore(MemoryConfig(db_path="./mem.db"))
    store.remember(task="hello", outcome="success")
    context = await store.retrieve("hello")
"""

from memlife.config import MemoryConfig
from memlife.models import Episode, Fact, JournalEntry, Metrics
from memlife.protocols import ChatCallable, Embedder
from memlife.embedders import DummyEmbedder
from memlife.llm import DummyChat
from memlife.reflection import Reflector, ReflectionResult
from memlife.store import MemoryStore
from memlife.sync_store import SyncMemoryStore
from memlife.vectors import cosine, recency_weight
from memlife.vector_backends import (
    VectorBackend,
    VectorSearchResult,
    JsonVectorBackend,
    BinaryVectorBackend,
    SqliteVecBackend,
    create_vector_backend,
)

__version__ = "0.6.4"

__all__ = [
    "MemoryStore",
    "SyncMemoryStore",
    "MemoryConfig",
    "Reflector",
    "ReflectionResult",
    "Episode",
    "Fact",
    "JournalEntry",
    "Metrics",
    "Embedder",
    "ChatCallable",
    "DummyEmbedder",
    "DummyChat",
    "cosine",
    "recency_weight",
    "retrieve",
    "run_gc",
    "export_jsonl",
    "import_jsonl",
    "vec_backend",
    "binary_vectors",
    "memorias",
    "polyphonic",
    "NamespaceError",
    "validate_namespace",
    "list_namespaces",
    "VectorBackend",
    "VectorSearchResult",
    "JsonVectorBackend",
    "BinaryVectorBackend",
    "SqliteVecBackend",
    "create_vector_backend",
]

# Convenience imports
from memlife.retrieval import retrieve
from memlife.gc import run_gc
from memlife.io import export_jsonl, import_jsonl
from memlife import memorias
from memlife.namespace import NamespaceError, validate_namespace, list_namespaces

