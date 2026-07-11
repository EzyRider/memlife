"""Journal reflection engine — the continuity centrepiece.

Runs (nightly, or on demand) over recent episodes and prior journal entries
and writes:

  * observations — what seems true now
  * hypotheses   — things to test
  * revisions    — updates to prior beliefs

A *revision* closes the loop: it references a prior journal entry (``revises``)
which is then marked superseded, so the old belief stops shaping retrieval and
the new one takes its place.

Quality gating: each entry must cite the episode(s) that ground it (``grounds``),
and a second *critic* model pass scores entries on grounding and
non-obviousness; entries the critic rejects are dropped before storage. This is
the home of the Month-3 reflection-quality metric.

It also performs lightweight memory consolidation: lexical contradiction
detection among facts, and (via the store) confidence decay / retirement of
stale journal entries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from memlife.store import MemoryStore
from memlife.config import MemoryConfig
from memlife.vectors import cosine
from memlife import memorias

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Outcome of one reflection pass."""

    observations: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    revisions: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    # Entries dropped by the critic, kept for transparency/telemetry.
    dropped: list[dict] = field(default_factory=list)
    consolidation: dict = field(default_factory=dict)
    raw: str = ""


class Reflector:
    """Drives the journal reflection process over a MemoryStore."""

    def __init__(
        self,
        memory: MemoryStore,
        model_chat,
        *,
        config=None,
        critic: bool = True,
        critic_model: str = "",  # MF-011: caller must provide
        decay_halflife_days: float = 30.0,
        decay_floor: float = 0.15,
        fact_merge_threshold: float = 0.90,
        fact_conflict_threshold: float = 0.75,
        timeout: float = 120.0,
        total_timeout: float = 300.0,
        contradiction_retirement_cycles: int = 14,
        agent_name: str = "the agent",
        model_name: str = "",
    ):
        """
        Accepts either a ``MemoryConfig`` via ``config`` or explicit keyword
        overrides. If both are provided, keyword values take precedence and a
        ``DeprecationWarning`` is emitted for the duplicated keywords.

        ``config`` pulls its defaults from ``memory.config`` when omitted, so
        callers can construct a Reflector with just the memory and model_chat.
        """
        """
        ``model_chat`` is an awaitable with the signature
            await model_chat(messages: list[dict], model: str) -> str
        returning the raw model text content. We pass a thin adapter from
        OllamaInterface so the journal does not depend on the model module.

        ``critic`` enables the second quality-gating pass.
        ``critic_model`` is the model used for the critic pass (should be cheap).

        ``timeout`` is the per-call budget for each model invocation
        (generation + critic). ``total_timeout`` is the budget for the whole
        reflect() pass; if it fires, partial work is abandoned and a result
        with the gathered episode IDs is returned.

        ``fact_merge_threshold`` / ``fact_conflict_threshold`` define the cosine
        bands used by :meth:`_detect_contradictions`: pairs at/above the merge
        threshold are same-fact duplicates (handled at store time, not flagged);
        pairs in [conflict, merge) are flagged as candidate contradictions; below
        conflict is unrelated. Threaded from Config so tests can override.
        """
        # MF-016: pull defaults from MemoryConfig to reduce parameter duplication.
        if config is None:
            config = getattr(memory, "config", None)
        if isinstance(config, MemoryConfig):
            decay_halflife_days = config.journal_decay_halflife_days if decay_halflife_days == 30.0 else decay_halflife_days
            decay_floor = config.journal_decay_floor if decay_floor == 0.15 else decay_floor
            fact_merge_threshold = config.fact_merge_threshold if fact_merge_threshold == 0.90 else fact_merge_threshold
            fact_conflict_threshold = config.fact_conflict_threshold if fact_conflict_threshold == 0.75 else fact_conflict_threshold
            timeout = config.reflection_timeout if timeout == 120.0 else timeout
            total_timeout = config.reflection_total_timeout if total_timeout == 300.0 else total_timeout
            contradiction_retirement_cycles = config.contradiction_retirement_cycles if contradiction_retirement_cycles == 14 else contradiction_retirement_cycles
        self.memory = memory
        self.model_chat = model_chat
        self.model_name = model_name  # MF-011: caller must provide
        self.critic = critic
        self.critic_model = critic_model  # MF-011: caller must provide
        self.decay_halflife_days = decay_halflife_days
        self.decay_floor = decay_floor
        self.fact_merge_threshold = fact_merge_threshold
        self.fact_conflict_threshold = fact_conflict_threshold
        self.timeout = timeout
        self.total_timeout = total_timeout
        # Timestamp of the last contradiction scan. On the first run, a full
        # all-pairs scan is performed. Subsequent runs only compare new/updated
        # facts (created_at or updated_at >= this timestamp) against the full
        # active set — O(new × total) instead of O(total²).
        #
        # MF-003: Reflector is designed to be created once and reused across
        # reflection passes. Recreating it resets this to 0.0, disabling
        # incremental scanning and forcing a full O(n²) scan every pass.
        # Callers that must recreate the Reflector should save and restore
        # `reflector.last_contradiction_scan` to preserve incremental scanning.
        self._last_contradiction_scan: float = 0.0
        # Reflection pass counter for contradiction retirement. Incremented
        # each time reflect() completes a full pass; used to track how many
        # reflection cycles ago a contradiction was last re-detected.
        self._reflection_cycle: int = 0
        # Retire active contradictions not re-detected in this many reflection
        # passes. 0 disables retirement.
        self.contradiction_retirement_cycles: int = contradiction_retirement_cycles
        # MF-010: agent name for the reflection prompt — was hardcoded "Ingrid".
        self.agent_name: str = agent_name

    @property
    def last_contradiction_scan(self) -> float:
        """Timestamp of the last contradiction scan (MF-003).

        Callers that recreate the Reflector each pass should save this value
        and pass it to the new instance via the constructor or restore it
        after construction to preserve incremental scanning.
        """
        return self._last_contradiction_scan

    @last_contradiction_scan.setter
    def last_contradiction_scan(self, value: float) -> None:
        self._last_contradiction_scan = value

    async def reflect(self, since: float | None = None, max_episodes: int = 50) -> ReflectionResult:
        """Reflect on episodes since ``since`` (epoch seconds).

        If ``since`` is None, reflects on all currently-pending (unreflected)
        queued episodes, falling back to today's episodes.

        A long-term context section is included: older episodes sampled
        across time and older journal entries, so the reflector can see
        patterns beyond the current window.

        The entire pass is bounded by ``self.total_timeout``. If it fires,
        the result contains the episode IDs that were considered but no
        stored entries — callers should treat it as an incomplete pass.
        """
        try:
            return await asyncio.wait_for(
                self._reflect_inner(since, max_episodes), timeout=self.total_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "Reflection exceeded total timeout of %.1fs; abandoning pass.", self.total_timeout
            )
            # Gather episodes without storing anything so the next run can retry.
            episodes = self._gather_episodes(since, max_episodes)
            return ReflectionResult(episode_ids=[e.id for e in episodes])

    async def _reflect_inner(self, since: float | None, max_episodes: int) -> ReflectionResult:
        """Core reflection logic (called inside the total-timeout wrapper)."""
        episodes = self._gather_episodes(since, max_episodes)
        if not episodes:
            logger.info("Reflection: no new episodes to reflect on.")
            return ReflectionResult()

        # Guard: model_name must be set before the model call. DummyChat
        # ignores it, but real adapters need a valid model identifier.
        if not self.model_name:
            raise ValueError(
                "Reflector.model_name is empty. Set it via the constructor "
                "or by assigning reflector.model_name = 'your-model' before "
                "calling reflect()."
            )

        ep_ids = [e.id for e in episodes]
        prior_journal = self.memory.journal_recent(limit=10)

        # Gather historical context: older episodes sampled across time
        # and older journal entries beyond the recent 10. This widens
        # the reflection window so slow patterns become visible.
        import time as _time

        now = _time.time()
        # Use the oldest pending episode's timestamp as the boundary —
        # everything before that is "historical".
        oldest_recent = min(e.created_at for e in episodes) if episodes else now
        historical_eps = self.memory.episodes_sample_historical(
            before=oldest_recent,
            limit=15,
            bins=5,
        )
        # Older journal entries (skip the 10 we already have as "prior").
        historical_journal = self.memory.journal_thematic_summary(limit=15)
        # Deduplicate: don't include journal entries that are in prior_journal.
        prior_ids = {j.id for j in prior_journal}
        historical_journal = [j for j in historical_journal if j.id not in prior_ids]

        prompt = self._build_prompt(
            episodes,
            prior_journal,
            historical_episodes=historical_eps,
            historical_journal=historical_journal,
        )

        try:
            raw = await asyncio.wait_for(
                self.model_chat.chat(prompt, self.model_name), timeout=self.timeout
            )
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Reflection generation timed out after %.1fs: %s", self.timeout, exc
            )
            return ReflectionResult(episode_ids=ep_ids, raw=f"generation timeout: {exc}")
        except Exception as exc:
            logger.warning("Reflection model call failed: %s", exc)
            return ReflectionResult(episode_ids=ep_ids, raw=str(exc))

        parsed = self._parse(raw, ep_ids, historical_eps)
        parsed.episode_ids = ep_ids
        parsed.contradictions = self._detect_contradictions()
        self._reflection_cycle += 1

        # Reinforce only contradictions that were re-detected in this pass
        # before retiring stale ones. This makes contradiction_retirement_cycles
        # meaningful: retire entries not re-detected for N passes.
        detected_pairs = {
            tuple(sorted((c["fact_a"], c["fact_b"])))
            for c in parsed.contradictions
            if isinstance(c.get("fact_a"), str) and isinstance(c.get("fact_b"), str)
        }
        reinforced = self.memory.reinforce_unresolved_contradictions(
            self._reflection_cycle, detected_pairs=detected_pairs
        )
        retired = self.memory.retire_stale_contradictions(
            self._reflection_cycle, self.contradiction_retirement_cycles
        )
        if reinforced:
            logger.debug("Reinforced %d unresolved contradiction(s)", reinforced)
        if retired:
            logger.info("Retired %d stale contradiction(s)", len(retired))

        # Quality gate: drop entries the critic rejects.
        if self.critic and parsed.observations + parsed.hypotheses + parsed.revisions:
            parsed = await self._critique(parsed, episodes)

        await self._store(parsed)

        # Structured MEMORIA extraction: if enabled, scan the raw reflection
        # output for explicit facts/preferences/instructions/timelines/KG
        # triples and persist them. KG triples are anchored to the reflection
        # episode(s) that produced them so provenance is preserved.
        memorias_stored: dict = {}
        if self.memory.config.memorias_extraction:
            memorias_stored = await memorias.persist_extraction(
                self.memory, parsed.raw, source="reflection"
            )
            # Record episode provenance on extracted triples.
            for tid, *_ in memorias_stored.get("kg_triples", []):
                self.memory._add_triple_provenance(
                    tid,
                    [{"kind": "episode", "id": ep_id} for ep_id in ep_ids],
                )

        parsed.consolidation = self.memory.consolidate_journal(
            halflife_days=self.decay_halflife_days,
            floor=self.decay_floor,
        )
        for ep_id in ep_ids:
            self.memory.mark_reflected(ep_id)

        # ── Record continuity metrics ──────────────────────────────────
        all_kept = parsed.observations + parsed.hypotheses + parsed.revisions
        all_proposed = len(all_kept) + len(parsed.dropped)
        confidences = [e.get("confidence", 0.5) for e in all_kept]

        def _proposed(kind: str, kept: list) -> int:
            return len(kept) + sum(1 for d in parsed.dropped if d.get("_drop_kind") == kind)

        self.memory.record_reflection_metrics(
            {
                "episodes_considered": len(ep_ids),
                "observations_proposed": _proposed("observation", parsed.observations),
                "observations_kept": len(parsed.observations),
                "hypotheses_proposed": _proposed("hypothesis", parsed.hypotheses),
                "hypotheses_kept": len(parsed.hypotheses),
                "revisions_proposed": _proposed("revision", parsed.revisions),
                "revisions_kept": len(parsed.revisions),
                "contradictions_found": len(parsed.contradictions),
                "avg_confidence": round(sum(confidences) / max(1, len(confidences)), 3),
                "keep_rate": round(len(all_kept) / max(1, all_proposed), 3),
                "consolidated_retired": parsed.consolidation.get("retired", 0),
                "consolidated_merged": parsed.consolidation.get("merged", 0),
                "total_journal_entries": len(self.memory.journal_recent(limit=1000)),
                "total_facts": len(self.memory._active_facts()),
                "total_episodes": len(self.memory.recent(limit=1000)),
            }
        )

        return parsed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _gather_episodes(self, since: float | None, max_episodes: int):
        if since is not None:
            return self.memory.episodes_since(since, limit=max_episodes)
        pending = self.memory.pending_reflections()
        if pending:
            eps = self.memory.episodes_by_ids(pending[:max_episodes])
            if eps:
                return eps
        # Fallback: last day's worth (86400s).
        import time

        return self.memory.episodes_since(time.time() - 86400, limit=max_episodes)

    def _build_prompt(
        self,
        episodes,
        prior_journal,
        historical_episodes=None,
        historical_journal=None,
    ) -> list[dict]:
        ep_lines = "\n".join(
            f"- [{e.id}] ({e.outcome}) {e.task}" + (f" → {e.summary[:160]}" if e.summary else "")
            for e in episodes
        )
        j_lines = (
            "\n".join(
                f"- [{j.id}] ({j.type}, conf={j.confidence:.2f}) {j.content}" for j in prior_journal
            )
            or "(none yet)"
        )
        ep_id_list = ", ".join(e.id for e in episodes)

        # Historical context: older episodes and journal entries that let
        # the model see patterns beyond the current window.
        hist_lines = ""
        if historical_episodes:
            hist_ep = "\n".join(
                f"- [{e.id}] ({e.outcome}) {e.task}"
                + (f" → {e.summary[:120]}" if e.summary else "")
                for e in historical_episodes
            )
            hist_lines += f"Older episodes (sampled across time):\n{hist_ep}\n\n"
        if historical_journal:
            hist_j = "\n".join(
                f"- [{j.id}] ({j.type}, conf={j.confidence:.2f}) {j.content}"
                for j in historical_journal
            )
            hist_lines += f"Older journal entries:\n{hist_j}\n\n"

        system = (
            f"You are {self.agent_name}'s reflective faculty. Review today's episodes and "
            "your prior journal, then write private entries that update your "
            "model of the user and the work.\n\n"
            "Rules:\n"
            "- Be specific and grounded. Every entry MUST cite the episode id(s) "
            "that support it in `grounds`. If you cannot ground an entry in a "
            "specific episode, do not write it.\n"
            "- Distinguish observations (supported by evidence now) from "
            "hypotheses (plausible but unproven) from revisions (an update to a "
            "prior belief — set `revises` to the id of the prior journal entry "
            "it corrects).\n"
            "- Do not over-interpret. If the evidence only says X, write X, not "
            "what X 'might say about' the user. Avoid psychologising. Avoid "
            "flattery. Prefer fewer, sharper entries to many flowery ones.\n"
            "- Never repeat episodes verbatim — synthesise.\n\n"
            "Respond with ONLY a JSON object of this exact shape:\n"
            "{\n"
            '  "observations": [{"content": "...", "confidence": 0.0-1.0, '
            '"grounds": ["ep_..."]}],\n'
            '  "hypotheses":   [{"content": "...", "confidence": 0.0-1.0, '
            '"grounds": ["ep_..."]}],\n'
            '  "revisions":    [{"content": "...", "confidence": 0.0-1.0, '
            '"revises": "jrn_..."}]\n'
            "}\n"
            "No prose outside the JSON. No markdown fences."
        )
        user = (
            f"Today's episodes (ids: {ep_id_list}):\n{ep_lines}\n\n"
            f"Prior journal:\n{j_lines}\n\n"
            f"{hist_lines}"
            "Write the journal for today. Consider both recent and historical "
            "context. If you notice long-term patterns (recurring themes, "
            "shifts over time, things that have been true for a while and "
            "may be changing now), call them out as hypotheses."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse(
        self, raw: str, ep_ids: list[str], historical_episodes: list | None = None
    ) -> ReflectionResult:
        result = ReflectionResult(raw=raw)
        data = self._extract_json(raw)
        if data is None:
            logger.warning("Reflection: could not parse JSON from model output.")
            return result

        valid_ep = set(ep_ids)
        if historical_episodes:
            valid_ep.update(e.id for e in historical_episodes if getattr(e, "id", None))
        for key in ("observations", "hypotheses", "revisions"):
            entries = data.get(key, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                content = entry.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                conf = self._clamp_conf(entry.get("confidence", 0.5))
                grounds = entry.get("grounds", [])
                grounds = (
                    [g for g in grounds if isinstance(g, str) and g in valid_ep]
                    if isinstance(grounds, list)
                    else []
                )
                revises = entry.get("revises", "")
                if not isinstance(revises, str) or not revises.startswith("jrn_"):
                    revises = ""
                getattr(result, key).append(
                    {
                        "content": content.strip(),
                        "confidence": conf,
                        "grounds": grounds,
                        "revises": revises,
                    }
                )
        return result

    @staticmethod
    def _clamp_conf(value) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError):
            conf = 0.5
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Pull a JSON object out of model output, tolerating fences/noise."""
        if not raw:
            return None
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                return None
            text = text[start:]
        end = text.rfind("}")
        if end == -1:
            return None
        text = text[: end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    async def _critique(self, parsed: ReflectionResult, episodes) -> ReflectionResult:
        """Second model pass: score each entry on grounding, non-obviousness,
        and redundancy against prior journal entries.

        Drops entries the critic marks ``keep: false``. Grounding is also
        checked mechanically: an entry with no cited grounds (for observations)
        is dropped outright, since it can't be traced to evidence.
        """
        ep_text = "\n".join(f"[{e.id}] {e.task} → {e.summary[:120]}" for e in episodes)
        bundle = []
        for kind in ("observations", "hypotheses", "revisions"):
            for i, entry in enumerate(getattr(parsed, kind)):
                bundle.append(
                    {
                        "ref": f"{kind[:-1]}#{i}",
                        "kind": kind[:-1],
                        "content": entry["content"],
                        "grounds": entry.get("grounds", []),
                        "revises": entry.get("revises", ""),
                    }
                )

        if not bundle:
            return parsed

        # Include prior journal entries so the critic can detect redundancy.
        prior = self.memory.journal_recent(limit=30)
        prior_text = (
            "\n".join(f"[{j.id}] ({j.type}) {j.content[:200]}" for j in prior)
            or "(no prior entries)"
        )

        system = (
            "You are a strict journal editor. Judge each candidate entry "
            "by the rules for its type. Respond with ONLY a JSON array: "
            '[{"ref": "observation#0", "keep": true|false, '
            '"grounding": 0.0-1.0, "non_obviousness": 0.0-1.0, '
            '"redundant": true|false, "reason": "..."}]. '
            "No prose, no fences.\n\n"
            "── Per-type rules ──\n"
            "OBSERVATIONS: Must be grounded in cited episodes. "
            "Drop if over-interpreting or psychologising. "
            "redundant=true if this observation restates something already "
            "captured in a prior *observation* — same fact, different words. "
            "Do NOT flag as redundant just because it relates to the same "
            "topic; only if it says the same thing.\n\n"
            "REVISIONS: These correct prior beliefs. They are expected to "
            "relate to prior entries — that is NOT redundancy. "
            "redundant should ALWAYS be false for revisions. "
            "Judge on whether the revision is a genuine correction "
            "(keep=true) or just noise (keep=false). "
            "Revisions do not need grounding in today's episodes.\n\n"
            "HYPOTHESES: Speculative by design. Do NOT require grounding "
            "in cited episodes. Judge on plausibility and whether they "
            "offer a testable, non-obvious prediction. "
            "redundant=true only if the exact same hypothesis was already "
            "stated in a prior journal entry.\n\n"
            "grounding: how well supported by cited episodes (0-1). "
            "non_obviousness: how insightful (0=trivial, 1=genuinely new). "
            "redundant: true if this entry adds nothing new vs prior journal."
        )
        user = (
            f"Prior journal entries:\n{prior_text}\n\n"
            f"Today's episodes:\n{ep_text}\n\n"
            f"Candidates:\n{json.dumps(bundle, indent=2)}"
        )
        try:
            raw = await asyncio.wait_for(
                self.model_chat.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    self.critic_model,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("Reflection critic timed out after %.1fs: %s", self.timeout, exc)
            return parsed
        except Exception as exc:
            logger.warning("Reflection critic failed (%s); keeping all entries.", exc)
            return parsed

        verdicts = self._parse_verdicts(raw)
        reject_refs = {v["ref"] for v in verdicts if v.get("keep") is False}
        reject_reasons = {v["ref"]: v.get("reason", "critic rejected") for v in verdicts}
        redundant_refs = {v["ref"] for v in verdicts if v.get("redundant") is True}

        # Collect critic scores for metrics.
        critic_scores: dict[str, dict] = {}
        for v in verdicts:
            ref = v.get("ref", "")
            critic_scores[ref] = {
                "grounding": self._clamp_conf(v.get("grounding", 0.5)),
                "non_obviousness": self._clamp_conf(v.get("non_obviousness", 0.3)),
                "redundant": bool(v.get("redundant", False)),
            }

        def filter_entries(entries, singular):
            kept, dropped = [], []
            for i, entry in enumerate(entries):
                ref = f"{singular}#{i}"
                grounded = True if singular == "revision" else bool(entry.get("grounds"))
                redundant = ref in redundant_refs
                if grounded and ref not in reject_refs and not redundant:
                    # Attach critic scores to kept entries.
                    if ref in critic_scores:
                        entry["critic_grounding"] = critic_scores[ref]["grounding"]
                        entry["critic_non_obviousness"] = critic_scores[ref]["non_obviousness"]
                    kept.append(entry)
                else:
                    if redundant:
                        entry["drop_reason"] = "redundant — already covered by prior journal entry"
                    else:
                        entry["drop_reason"] = reject_reasons.get(ref) or (
                            "no grounds cited" if not grounded else "critic rejected"
                        )
                    if ref in critic_scores:
                        entry["critic_grounding"] = critic_scores[ref]["grounding"]
                        entry["critic_non_obviousness"] = critic_scores[ref]["non_obviousness"]
                    dropped.append(entry)
            return kept, dropped

        result = ReflectionResult(
            raw=parsed.raw, episode_ids=parsed.episode_ids, contradictions=parsed.contradictions
        )
        all_dropped = []
        for kind in ("observations", "hypotheses", "revisions"):
            singular = kind[:-1]
            kept, dropped = filter_entries(getattr(parsed, kind), singular)
            # Tag dropped entries with their original kind for metrics.
            for d in dropped:
                d["_drop_kind"] = singular
            setattr(result, kind, kept)
            all_dropped.extend(dropped)
        result.dropped = all_dropped
        return result

    @staticmethod
    def _parse_verdicts(raw: str) -> list[dict]:
        if not raw:
            return []
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            arr = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        return [v for v in arr if isinstance(v, dict)]

    def _detect_contradictions(self) -> list[dict]:
        """Flag fact pairs that are semantically close but differ — candidates
        for the agent to resolve.

        Incremental: on the first run (no last scan timestamp), does a full
        all-pairs scan. On subsequent runs, only facts created or updated since
        the last scan are compared against the full active set. This reduces
        the cost from O(n²) to O(new × n) for steady-state operation, where
        ``new`` is typically a handful of recently stored or revised facts.

        Two paths, picked by what's available:

        * **Embeddings present** (≥2 facts carry embeddings): cosine-gated. Flag
          pairs whose similarity lands in the *contradiction band*
          ``[fact_conflict_threshold, fact_merge_threshold)``. Pairs at/above
          the merge threshold are same-fact duplicates — those are merged at
          store time, so they're explicitly excluded here (they aren't
          conflicts). No token pre-narrow: the whole point is to catch
          paraphrased contradictions ("I live in Wellington" vs "I'm based in
          Auckland") that share no significant tokens and that the lexical path
          misses.

        * **No/few embeddings**: fall back to the original lexical heuristic
          (shared significant tokens, common-term filter) so detection still
          works in the optional-embeddings scenario. It degrades, it doesn't
          vanish.

        Output is capped at 20 to avoid flooding the journal with noise. Sync:
        embeddings are already stored, so no embedding calls happen here.
        """
        scan_start = time.time()
        prev_scan = self._last_contradiction_scan

        facts = self.memory._facts_with_embeddings()
        if len(facts) >= 2:
            # On first run (prev_scan == 0), compare all pairs.
            # On subsequent runs, only compare new/updated facts vs all.
            if prev_scan > 0:
                new_facts = self.memory._facts_with_embeddings_since(prev_scan)
                new_ids = {f.id for f in new_facts}
                if not new_ids:
                    # Nothing new — no contradictions to find.
                    self._last_contradiction_scan = scan_start
                    return []
                result = self._detect_contradictions_embedding(facts, new_ids=new_ids)
            else:
                result = self._detect_contradictions_embedding(facts)
        else:
            all_facts = self.memory._active_facts()
            if prev_scan > 0:
                new_facts = self.memory._active_facts_since(prev_scan)
                new_ids = {f.id for f in new_facts}
                if not new_ids:
                    self._last_contradiction_scan = scan_start
                    return []
                result = self._detect_contradictions_lexical(all_facts, new_ids=new_ids)
            else:
                result = self._detect_contradictions_lexical(all_facts)

        self._last_contradiction_scan = scan_start
        return result

    def _detect_contradictions_embedding(
        self,
        facts: list,
        new_ids: set[str] | None = None,
    ) -> list[dict]:
        """Cosine-band contradiction detection over facts that carry embeddings.

        When ``new_ids`` is provided (incremental mode), only pairs where at
        least one fact is in ``new_ids`` are compared. This reduces the scan
        from O(n²) to O(new × n). When ``new_ids`` is None, full all-pairs scan.
        """
        merge = self.fact_merge_threshold
        conflict = self.fact_conflict_threshold
        seen: list[dict] = []
        n = len(facts)
        for i in range(n):
            a = facts[i]
            if a.embedding is None:
                continue
            a_lower = a.content.lower()
            for j in range(i + 1, n):
                b = facts[j]
                # Incremental: skip pairs where neither fact is new.
                if new_ids is not None and a.id not in new_ids and b.id not in new_ids:
                    continue
                if b.embedding is None:
                    continue
                sim = cosine(a.embedding, b.embedding)
                if sim < conflict or sim >= merge:
                    continue
                # Skip exact / containment pairs — those are duplicates, not
                # contradictions (and store-time merge should have collapsed them).
                if a_lower == b.content.lower():
                    continue
                if a_lower in b.content.lower() or b.content.lower() in a_lower:
                    continue
                # shared_terms stays as an informative signal for the agent, even
                # though the gate is cosine, not token overlap (paraphrases may
                # share none — that's fine, the list is just empty then).
                shared = {w.lower() for w in re.findall(r"\w+", a.content) if len(w) > 4} & {
                    w.lower() for w in re.findall(r"\w+", b.content) if len(w) > 4
                }
                seen.append(
                    {
                        "fact_a": a.id,
                        "fact_b": b.id,
                        "content_a": a.content,
                        "content_b": b.content,
                        "similarity": round(sim, 3),
                        "shared_terms": sorted(shared),
                    }
                )
                if len(seen) >= 20:
                    return seen
        return seen

    def _detect_contradictions_lexical(
        self,
        facts: list,
        new_ids: set[str] | None = None,
    ) -> list[dict]:
        """Original token-overlap heuristic — the fallback when embeddings are
        unavailable. Preserves the pre-embedding behaviour exactly.

        When ``new_ids`` is provided (incremental mode), only pairs where at
        least one fact is in ``new_ids`` are compared."""
        if len(facts) < 2:
            return []

        fact_tokens: list[set[str]] = []
        token_freq: dict[str, int] = {}
        for f in facts:
            tokens = {w.lower() for w in re.findall(r"\w+", f.content) if len(w) > 4}
            fact_tokens.append(tokens)
            for t in tokens:
                token_freq[t] = token_freq.get(t, 0) + 1

        # MF-015: was max(2, len(facts) * 0.3) — too aggressive for small
        # fact sets. For 3 facts, threshold was 2, so any term in all 3 was
        # "common" and filtered out, missing real contradictions.
        threshold = max(3, len(facts) * 0.5)
        common = {t for t, c in token_freq.items() if c > threshold}

        seen: list[dict] = []
        for i, a in enumerate(facts):
            a_tokens = fact_tokens[i] - common
            if len(a_tokens) < 2:
                continue
            for j in range(i + 1, len(facts)):
                b = facts[j]
                # Incremental: skip pairs where neither fact is new.
                if new_ids is not None and a.id not in new_ids and b.id not in new_ids:
                    continue
                b_tokens = fact_tokens[j] - common
                shared = a_tokens & b_tokens
                if len(shared) < 2:
                    continue
                if a.content.lower() == b.content.lower():
                    continue
                if a.content.lower() in b.content.lower() or b.content.lower() in a.content.lower():
                    continue
                seen.append(
                    {
                        "fact_a": a.id,
                        "fact_b": b.id,
                        "content_a": a.content,
                        "content_b": b.content,
                        "shared_terms": sorted(shared),
                    }
                )
                if len(seen) >= 20:
                    return seen
        return seen

    async def _store(self, result: ReflectionResult) -> None:
        for obs in result.observations:
            jid = self.memory.add_journal_entry(
                "observation",
                obs["content"],
                obs["confidence"],
                source_episodes=obs.get("grounds") or result.episode_ids,
            )
            await self.memory.embed_journal_entry(jid)
        for hyp in result.hypotheses:
            jid = self.memory.add_journal_entry(
                "hypothesis",
                hyp["content"],
                hyp["confidence"],
                source_episodes=hyp.get("grounds") or result.episode_ids,
            )
            await self.memory.embed_journal_entry(jid)
        # Revisions close the loop: store the new entry, then supersede the
        # prior journal entry it revises.
        for rev in result.revisions:
            new_id = self.memory.add_journal_entry(
                "revision",
                rev["content"],
                rev["confidence"],
                source_episodes=result.episode_ids,
            )
            await self.memory.embed_journal_entry(new_id)
            target = rev.get("revises", "")
            if target:
                self.memory.supersede_journal(target, new_id)
        # Persist contradictions as journal entries so they survive the run and
        # can feed future reflection/revision. They're grounded in the two
        # conflicting facts and tagged type='contradiction', which the recall
        # layer excludes from injected context (see _active_journal_sql) — so
        # storing them records the tension without polluting turns.
        for c in result.contradictions:
            if self.memory.has_active_contradiction(c["fact_a"], c["fact_b"]):
                self.memory.touch_active_contradiction(
                    c["fact_a"], c["fact_b"], self._reflection_cycle
                )
                logger.debug(
                    "Re-detected contradiction '%s' ↔ '%s'; touched",
                    c["content_a"],
                    c["content_b"],
                )
                continue
            content = (
                f"Possible contradiction between facts: "
                f"'{c['content_a']}' ↔ '{c['content_b']}' "
                f"(shared terms: {', '.join(c['shared_terms'])})"
            )
            jid = self.memory.add_journal_entry(
                "contradiction",
                content,
                confidence=0.6,
                source_episodes=[c["fact_a"], c["fact_b"]],
                last_detected=self._reflection_cycle,
            )
            await self.memory.embed_journal_entry(jid)
            logger.info(
                "Stored contradiction %s: '%s' ↔ '%s' (shared: %s)",
                jid,
                c["content_a"],
                c["content_b"],
                ", ".join(c["shared_terms"]),
            )
