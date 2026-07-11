"""MCP server for memlife — exposes memory tools to any MCP-compatible agent.

Usage:
    memlife-mcp-server --db ./mem.db --embedder ollama --model mxbai-embed-large:latest

Or programmatically:
    from memlife.mcp_server import run_server
    run_server(db_path="./mem.db", embedder_type="ollama", model="mxbai-embed-large:latest")

Tools exposed:
    memory_store        — Store a fact
    memory_search        — Search facts by query
    memory_search_journal    — Search journal entries
    memory_search_episodes   — Search episodes by keyword or tool name
    memory_revise       — Revise an existing fact
    memory_expire       — Mark a fact as expired
    memory_gc           — Run garbage collection
    memory_reflect      — Run reflection pass (synthesise episodes into journal)

Resources:
    memlife://stats     — Memory statistics
    memlife://health    — Embedding health report
    memlife://contradictions — Detected contradictions
"""

from __future__ import annotations

import argparse
import atexit
import signal
import json
import logging
import os
import sys

from memlife.config import MemoryConfig
from memlife.store import MemoryStore

logger = logging.getLogger(__name__)


def _make_embedder(embedder_type: str, model: str, base_url: str):
    """Create an embedder based on the type string."""
    if embedder_type == "dummy":
        from memlife.embedders import DummyEmbedder
        return DummyEmbedder()
    elif embedder_type == "ollama":
        from memlife.adapters.ollama import OllamaEmbedder
        return OllamaEmbedder(base_url=base_url, model=model)
    elif embedder_type == "openai":
        from memlife.adapters.openai import OpenAIEmbedder
        return OpenAIEmbedder(model=model)
    elif embedder_type == "sentence_transformers":
        from memlife.adapters.sentence_transformers import STEmbedder
        return STEmbedder(model=model)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


def create_server(
    db_path: str = "",
    data_dir: str = "./memlife_data",
    namespace: str = "default",
    embedder_type: str = "dummy",
    embedding_model: str = "dummy",
    base_url: str = "http://localhost:11434",
    chat_model: str = "",  # MF-011: caller must provide
    critic_model: str = "",  # MF-011: caller must provide
):
    """Create and configure a FastMCP server with memlife tools.

    Returns a configured FastMCP instance ready to run.
    """
    from mcp.server.fastmcp import FastMCP

    config = MemoryConfig(
        db_path=db_path,
        data_dir=data_dir,
        namespace=namespace,
        embedding_model=embedding_model if embedder_type != "dummy" else "dummy",
    )
    embedder = _make_embedder(embedder_type, embedding_model, base_url)
    store = MemoryStore(config=config, embedder=embedder)

    # Lazy-init: Reflector and chat adapter created on first reflect() call.
    _reflector = None
    _chat_adapter = None

    async def _get_reflector():
        """Lazily create a persistent Reflector with an Ollama chat adapter.

        The Reflector is created once and reused across calls so that
        _last_contradiction_scan persists between reflection passes (see
        MF-003 in BACKLOG.md). Created lazily so the server doesn't need
        an LLM endpoint just to start up — only when reflection is called.
        """
        nonlocal _reflector, _chat_adapter
        if _reflector is None:
            from memlife.adapters.ollama import OllamaChat
            from memlife.reflection import Reflector

            _chat_adapter = OllamaChat(
                base_url=base_url,
                model=chat_model,
                fallback_models=[critic_model],
            )

            # Reflector calls model_chat.chat(messages, model) — OllamaChat
            # implements that directly. No wrapper needed.
            _reflector = Reflector(
                memory=store,
                model_chat=_chat_adapter,
                critic=True,
                critic_model=critic_model,
            )
            _reflector.model_name = chat_model
        return _reflector

    mcp = FastMCP("memlife")

    # ── Tools ──────────────────────────────────────────────────────

    @mcp.tool()
    async def memory_store(
        content: str,
        source: str = "agent",
        confidence: float = 0.7,
    ) -> str:
        """Store a durable fact, preference, or entity relationship in
        long-term memory. Use for things worth remembering across sessions.

        Args:
            content: The fact to store.
            source: Origin of the fact ('user', 'agent', 'journal').
            confidence: Confidence 0.0-1.0.
        """
        fact_id = await store.store_fact(content, source=source, confidence=confidence)
        return f"Stored fact {fact_id}: {content}"

    @mcp.tool()
    async def memory_search(query: str, limit: int = 5) -> str:
        """Search long-term memory for facts relevant to a query.

        Args:
            query: What to look for.
            limit: Max results.
        """
        facts = await store.recall_facts(query, limit=limit)
        if not facts:
            return f"No facts found for '{query}'."
        lines = [
            f"[{i+1}] (conf={f.confidence:.2f}) {f.content}"
            for i, f in enumerate(facts)
        ]
        return "\n".join(lines)

    @mcp.tool()
    def memory_search_journal(query: str, limit: int = 5) -> str:
        """Search journal entries (observations and hypotheses) by keyword.

        These are private reflections — never quote them verbatim to the user.

        Args:
            query: What to search for.
            limit: Max results.
        """
        entries = store.search_journal(query, limit=limit)
        if not entries:
            return f"No journal entries found for '{query}'."
        lines = [
            f"[{i+1}] ({e.type}, conf={e.confidence:.2f}) {e.content}"
            for i, e in enumerate(entries)
        ]
        return "\n".join(lines)

    @mcp.tool()
    def memory_search_episodes(
        query: str = "",
        tool_name: str = "",
        outcome: str = "",
        limit: int = 10,
    ) -> str:
        """Search past episodes by keyword or by tool name.

        Pass tool_name to search by tool (e.g. 'read_file'), or query
        for text search. Optionally filter by outcome ('success' or 'failed').

        Args:
            query: Text to search for in task/summary.
            tool_name: Find episodes where this tool was used.
            outcome: Filter by outcome: 'success' or 'failed'.
            limit: Max results.
        """
        if tool_name:
            episodes = store.search_episodes_by_tool(
                tool_name, outcome=outcome or None, limit=limit,
            )
        else:
            episodes = store.search_episodes_by_keyword(query, limit=limit)
        if not episodes:
            desc = f"tool='{tool_name}'" if tool_name else f"query='{query}'"
            return f"No episodes found for {desc}."
        lines = [
            f"[{i+1}] ({e.outcome}) {e.task}"
            + (f" -> {e.summary[:120]}" if e.summary else "")
            for i, e in enumerate(episodes)
        ]
        return "\n".join(lines)

    @mcp.tool()
    async def memory_revise(fact_id: str, new_content: str, confidence: float = 0.7) -> str:
        """Revise an existing fact with new content. The old fact is superseded.

        Args:
            fact_id: ID of the fact to revise.
            new_content: The corrected content.
            confidence: Confidence 0.0-1.0.
        """
        new_id = await store.revise_fact(fact_id, new_content, confidence)
        return f"Revised fact {fact_id} -> {new_id}: {new_content}"

    @mcp.tool()
    def memory_expire(fact_id: str) -> str:
        """Mark a fact as expired (superseded). Use when a fact is no longer true.

        Args:
            fact_id: ID of the fact to expire.
        """
        updated = store.expire_fact(fact_id)
        if updated:
            return f"Expired fact {fact_id}"
        return f"Fact {fact_id} not found or already expired"

    @mcp.tool()
    async def memory_retrieve(query: str) -> str:
        """Retrieve and rank memories across all layers (episodes, facts,
        journal) using the unified scoring metric. Returns formatted context
        with labelled sections.

        Args:
            query: What to look for.
        """
        return await store.retrieve(query)

    @mcp.tool()
    def memory_gc() -> str:
        """Run garbage collection on old/superseded memory data. Prunes
        superseded facts (90d), superseded journal (90d), completed runs
        (60d), metrics (30d), reflected queue entries (30d), and old
        episodes (180d). Does NOT run VACUUM — use memory_vacuum separately
        when the store is idle."""
        result = store.run_gc()
        return (
            f"Pruned {result['total_pruned']} rows.\n"
            f"  superseded facts:   {result.get('superseded_facts', 0)}\n"
            f"  superseded journal: {result.get('superseded_journal', 0)}\n"
            f"  agent runs:         {result.get('agent_runs', 0)}\n"
            f"  checkpoints:        {result.get('checkpoints', 0)}\n"
            f"  reflection metrics: {result.get('reflection_metrics', 0)}\n"
            f"  reflected queue:    {result.get('reflected_queue', 0)}\n"
            f"  episodes:           {result.get('episodes', 0)}"
        )

    @mcp.tool()
    def memory_vacuum() -> str:
        """Reclaim disk space by rebuilding the database file (VACUUM).
        This needs an exclusive lock and can stall active operations —
        run when the store is idle, not during active MCP traffic."""
        vacuum = store.run_vacuum()
        return (
            f"DB: {vacuum['db_size_before_mb']}MB -> {vacuum['db_size_after_mb']}MB"
        )

    @mcp.tool()
    async def memory_reflect(max_episodes: int = 50) -> str:
        """Run a reflection pass — synthesise pending episodes into journal
        entries, detect contradictions, and apply confidence decay.

        This is the lifecycle engine. Without reflection, episodes never
        become journal entries, confidence never decays, and contradictions
        go undetected. Run periodically (e.g. daily via cron).

        Args:
            max_episodes: Max episodes to reflect on per pass (default 50).
        """
        try:
            reflector = await _get_reflector()
            result = await reflector.reflect(max_episodes=max_episodes)
        except Exception as e:
            logger.error("Reflection failed: %s", e, exc_info=True)
            return f"Reflection failed: {e}"

        parts = [
            "Reflection complete.\n"
            f"  Episodes reflected: {len(result.episode_ids)}",
            f"  Observations kept:  {len(result.observations)}",
            f"  Hypotheses kept:    {len(result.hypotheses)}",
            f"  Revisions kept:     {len(result.revisions)}",
            f"  Contradictions:     {len(result.contradictions)}",
            f"  Dropped by critic:  {len(result.dropped)}",
        ]
        return "\n".join(parts)

    # ── Resources ──────────────────────────────────────────────────

    @mcp.resource("memlife://stats")
    def stats() -> str:
        """Memory statistics: counts, embedding health, reflection metrics,
        and recall path counters (MV2-006)."""
        health = store.embedding_health()
        summary = store.get_metrics_summary()
        recall = store.recall_stats()
        return json.dumps({
            "embedding_model": health.get("embedding_model", ""),
            "facts": health["facts"],
            "journal": health["journal"],
            "episodes": health["episodes"],
            "reflection_count": summary.get("total_reflections", 0),
            "unresolved_contradictions": summary.get("unresolved_contradictions", 0),
            "recall": recall,
        }, indent=2)

    @mcp.resource("memlife://health")
    def health_resource() -> str:
        """Embedding health report: coverage, staleness, failure count."""
        h = store.embedding_health()
        return json.dumps(h, indent=2)

    @mcp.resource("memlife://contradictions")
    def contradictions() -> str:
        """Detected contradictions in the memory system."""
        items = store.list_contradictions(limit=20)
        return json.dumps(items, indent=2, default=str)

    # Store reference for cleanup
    mcp._memlife_store = store
    mcp._memlife_embedder = embedder
    mcp._memlife_get_reflector = _get_reflector

    return mcp


def main():
    """CLI entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="memlife MCP server — lifecycle memory for AI agents"
    )
    parser.add_argument(
        "--db", default=os.getenv("MEMLIFE_DB_PATH", ""),
        help="Path to the SQLite database. If not set, resolves to data_dir/namespace/memlife.db",
    )
    parser.add_argument(
        "--data-dir", default=os.getenv("MEMLIFE_DATA_DIR", "./memlife_data"),
        help="Root directory for namespace databases (default: ./memlife_data)",
    )
    parser.add_argument(
        "--namespace", default=os.getenv("MEMLIFE_NAMESPACE", "default"),
        help="Namespace for this server instance (default: default)",
    )
    parser.add_argument(
        "--embedder", default=os.getenv("MEMLIFE_EMBEDDER", "dummy"),
        choices=["dummy", "ollama", "openai", "sentence_transformers"],
        help="Embedder type (default: dummy)",
    )
    parser.add_argument(
        "--embedding-model", default=os.getenv("MEMLIFE_EMBEDDING_MODEL", "dummy"),
        help="Embedding model name (default: dummy)",
    )
    parser.add_argument(
        "--ollama-url", default=os.getenv("MEMLIFE_OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--chat-model", default=os.getenv("MEMLIFE_CHAT_MODEL", ""),
        help="Ollama model for reflection synthesis (required for reflection)",
    )
    parser.add_argument(
        "--critic-model", default=os.getenv("MEMLIFE_CRITIC_MODEL", ""),
        help="Ollama model for reflection critic pass (optional)",
    )
    parser.add_argument(
        "--log-level", default=os.getenv("MEMLIFE_LOG_LEVEL", "INFO"),
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, stream=sys.stderr)

    server = create_server(
        db_path=args.db,
        data_dir=args.data_dir,
        namespace=args.namespace,
        embedder_type=args.embedder,
        embedding_model=args.embedding_model,
        base_url=args.ollama_url,
        chat_model=args.chat_model,
        critic_model=args.critic_model,
    )

    resolved_db = server._memlife_store.db_path  # type: ignore[attr-defined]
    logger.info(
        "memlife MCP server starting (db=%s, embedder=%s, model=%s, namespace=%s)",
        resolved_db, args.embedder, args.embedding_model, args.namespace,
    )

    # MF-016: ensure store, embedder, and sessions are cleaned up on exit.
    def _shutdown() -> None:
        try:
            server._memlife_store.close()  # type: ignore[attr-defined]
            logger.info("memlife store closed")
        except Exception as exc:  # pragma: no cover
            logger.warning("error closing memlife store: %s", exc)

    atexit.register(_shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: _shutdown())

    server.run()


if __name__ == "__main__":
    main()