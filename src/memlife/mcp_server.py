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
    memory_store_triple      — Store a subject-predicate-object triple
    memory_search_triples    — Search triples connected to an entity
    memory_entity_neighbors  — Explore graph neighbors of an entity
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
import asyncio
import atexit
import json
import logging
import os
import signal
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
    namespace: str = "_default",
    vector_backend: str = "",
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

    config_kwargs = {
        "db_path": db_path,
        "data_dir": data_dir,
        "namespace": namespace,
        "embedding_model": embedding_model if embedder_type != "dummy" else "dummy",
    }
    if vector_backend:
        config_kwargs["vector_backend"] = vector_backend
    config = MemoryConfig(**config_kwargs)
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
            # Expose the adapter to shutdown_mcp_server once created.
            mcp._memlife_chat_adapter = _chat_adapter

            # Reflector calls model_chat.chat(messages, model) — OllamaChat
            # implements that directly. No wrapper needed.
            _reflector = Reflector(
                memory=store,
                model_chat=_chat_adapter,
                critic=True,
                critic_model=critic_model,
            )
            _reflector.model_name = chat_model
            mcp._memlife_reflector = _reflector
        return _reflector

    mcp = FastMCP("memlife")

    # ── Tools ──────────────────────────────────────────────────────

    # MV2-010: graph/triples tools are registered alongside memory tools.

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
    def memory_store_triple(
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
    ) -> str:
        """Store a subject-predicate-object triple in the knowledge graph.

        Useful for explicit entity relationships (e.g. person-knows-person,
        user-works_at-place). Aliases are resolved to canonical entity names.

        Args:
            subject: The entity the relationship starts from.
            predicate: The relationship type.
            object: The entity the relationship points to.
            confidence: Confidence 0.0-1.0.
        """
        triple_id = store.store_triple(subject, predicate, object, confidence=confidence)
        return f"Stored triple {triple_id}: {subject} {predicate} {object}"

    @mcp.tool()
    def memory_search_triples(
        entity: str,
        predicate: str = "",
        limit: int = 10,
    ) -> str:
        """Search triples connected to an entity.

        Returns triples where the entity appears as subject or object.
        Pass predicate to filter by relationship type.

        Args:
            entity: Entity to search around.
            predicate: Optional relationship filter.
            limit: Max results.
        """
        triples = store.triples_about(
            entity, predicate=predicate or None, limit=limit,
        )
        if not triples:
            return f"No triples found for '{entity}'."
        lines = [
            f"[{i+1}] {t['subject']} {t['predicate']} {t['object']} (conf={t['confidence']:.2f})"
            for i, t in enumerate(triples)
        ]
        return "\n".join(lines)

    @mcp.tool()
    def memory_entity_neighbors(
        entity: str,
        predicate: str = "",
        depth: int = 1,
        limit: int = 10,
    ) -> str:
        """Explore the graph around an entity.

        Returns neighbors reachable within ``depth`` edge-hops. Optional
        predicate filter restricts which edges are followed.

        Args:
            entity: Starting entity.
            predicate: Optional relationship filter.
            depth: How many hops to follow (default 1).
            limit: Max neighbors to return.
        """
        neighbors = store.entity_neighbors(
            entity, predicate=predicate or None, depth=depth, limit=limit,
        )
        if not neighbors:
            return f"No neighbors found for '{entity}'."
        lines = [
            f"[{i+1}] {n['entity']} (depth {n['depth']})"
            for i, n in enumerate(neighbors)
        ]
        return "\n".join(lines)

    @mcp.tool()
    def memory_gc() -> str:
        """Run garbage collection on old/superseded memory data. Prunes
        superseded facts (90d), superseded journal (90d), completed runs
        (60d), metrics (30d), reflected queue entries (30d), old
        episodes (180d), and closed temporal triples with their orphaned
        entities/aliases (90d). Does NOT run VACUUM — use memory_vacuum
        separately when the store is idle."""
        result = store.run_gc()
        return (
            f"Pruned {result['total_pruned']} rows.\n"
            f"  superseded facts:   {result.get('superseded_facts', 0)}\n"
            f"  superseded journal: {result.get('superseded_journal', 0)}\n"
            f"  agent runs:         {result.get('agent_runs', 0)}\n"
            f"  checkpoints:        {result.get('checkpoints', 0)}\n"
            f"  reflection metrics: {result.get('reflection_metrics', 0)}\n"
            f"  reflected queue:    {result.get('reflected_queue', 0)}\n"
            f"  episodes:           {result.get('episodes', 0)}\n"
            f"  closed triples:     {result.get('closed_triples', 0)}\n"
            f"  orphan provenance:  {result.get('orphan_provenance', 0)}\n"
            f"  orphan aliases:     {result.get('orphan_aliases', 0)}\n"
            f"  orphan entities:    {result.get('orphan_entities', 0)}"
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
        metrics = store.metrics()
        return json.dumps({
            "db_path": metrics.db_path,
            "db_size_mb": metrics.db_size_mb,
            "namespace": metrics.namespace,
            "vector_backend": metrics.vector_backend,
            "embedding_model": metrics.embedding_model,
            "counts": {
                "episodes": metrics.episodes,
                "facts": metrics.facts,
                "active_facts": metrics.active_facts,
                "journal_entries": metrics.journal_entries,
                "active_journal": metrics.active_journal,
                "contradictions": metrics.contradictions,
                "unresolved_contradictions": metrics.unresolved_contradictions,
                "user_corrections": metrics.user_corrections,
                "sessions": metrics.sessions,
                "agent_runs": metrics.agent_runs,
                "triples": metrics.triples,
                "entities": metrics.entities,
            },
            "embeddings": {
                "embedded_episodes": metrics.embedded_episodes,
                "embedded_facts": metrics.embedded_facts,
                "embedded_journal": metrics.embedded_journal,
                "pending_embeddings": metrics.pending_embeddings,
                "health": metrics.embedding_health,
            },
            "reflection": {
                "total_reflections": metrics.total_reflections,
                "last_reflection_at": metrics.last_reflection_at,
                "avg_keep_rate": metrics.avg_keep_rate,
                "avg_confidence": metrics.avg_confidence,
                "total_observations_kept": metrics.total_observations_kept,
                "total_hypotheses_kept": metrics.total_hypotheses_kept,
                "total_revisions_kept": metrics.total_revisions_kept,
                "total_contradictions_found": metrics.total_contradictions_found,
                "total_retired": metrics.total_retired,
                "total_merged": metrics.total_merged,
            },
            "recall": metrics.recall,
            "migration": metrics.migration,
        }, indent=2, default=str)

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

    # Store references for cleanup (MF-016: MCP server cleanup).
    mcp._memlife_store = store
    mcp._memlife_embedder = embedder
    mcp._memlife_chat_adapter = _chat_adapter
    mcp._memlife_get_reflector = _get_reflector

    return mcp


def _close_resource(name: str, resource: object | None) -> None:
    """Close a resource that may have a sync or async close method."""
    if resource is None:
        return
    close_fn = getattr(resource, "close", None)
    if close_fn is None:
        return
    try:
        if asyncio.iscoroutinefunction(close_fn):
            try:
                asyncio.run(close_fn())
            except Exception as exc:  # pragma: no cover
                logger.warning("error closing async resource %s: %s", name, exc)
        else:
            close_fn()
    except Exception as exc:  # pragma: no cover
        logger.warning("error closing resource %s: %s", name, exc)


def shutdown_mcp_server(mcp) -> None:
    """Close all resources owned by a memlife MCP server.

    Safe to call multiple times. Closes the store, embedder, chat adapter,
    and reflector (if any were created).
    """
    store = getattr(mcp, "_memlife_store", None)
    embedder = getattr(mcp, "_memlife_embedder", None)
    chat = getattr(mcp, "_memlife_chat_adapter", None)
    reflector = getattr(mcp, "_memlife_reflector", None)

    _close_resource("reflector", reflector)
    _close_resource("chat_adapter", chat)
    _close_resource("embedder", embedder)
    _close_resource("store", store)


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
        "--namespace", default=os.getenv("MEMLIFE_NAMESPACE", "_default"),
        help="Namespace for this server instance (default: _default)",
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
        "--vector-backend",
        default=os.getenv("MEMLIFE_VECTOR_BACKEND", ""),
        choices=["", "json", "sqlite_vec", "binary"],
        help="Vector backend to use: json (default), sqlite_vec, or binary. "
             "If empty, MemoryConfig's default applies.",
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
        vector_backend=args.vector_backend,
        embedder_type=args.embedder,
        embedding_model=args.embedding_model,
        base_url=args.ollama_url,
        chat_model=args.chat_model,
        critic_model=args.critic_model,
    )

    resolved_db = server._memlife_store.db_path  # type: ignore[attr-defined]
    backend_name = server._memlife_store.vector_backend.name  # type: ignore[attr-defined]
    logger.info(
        "memlife MCP server starting (db=%s, embedder=%s, model=%s, namespace=%s, vector_backend=%s)",
        resolved_db, args.embedder, args.embedding_model, args.namespace, backend_name,
    )

    # MF-016: ensure store, embedder, and sessions are cleaned up on exit.
    # Use shutdown_mcp_server so all resources (store, embedder, chat adapter,
    # reflector) are closed, not just the SQLite connection.
    atexit.register(lambda: shutdown_mcp_server(server))

    # SIGTERM: clean up and exit. SIGINT is left to its default handler so
    # server.run() receives KeyboardInterrupt, atexit still runs, and the
    # process terminates cleanly. The old handler returned without exiting,
    # which could leave the server hung on SIGINT/SIGTERM.
    def _handle_signal(signum, _frame) -> None:  # pragma: no cover
        sig_name = signal.Signals(signum).name
        logger.info("received %s, shutting down memlife MCP server", sig_name)
        shutdown_mcp_server(server)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)

    server.run()


if __name__ == "__main__":
    main()