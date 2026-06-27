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
from memlife.models import Episode, Fact, JournalEntry
from memlife.protocols import ChatCallable, Embedder
from memlife.embedders import DummyEmbedder
from memlife.llm import DummyChat
from memlife.reflection import Reflector, ReflectionResult
from memlife.store import MemoryStore
from memlife.sync_store import SyncMemoryStore
from memlife.vectors import cosine, recency_weight

__version__ = "0.1.0b0"

__all__ = [
    "MemoryStore",
    "SyncMemoryStore",
    "MemoryConfig",
    "Reflector",
    "ReflectionResult",
    "Episode",
    "Fact",
    "JournalEntry",
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
]

# Convenience imports
from memlife.retrieval import retrieve
from memlife.gc import run_gc
from memlife.io import export_jsonl, import_jsonl