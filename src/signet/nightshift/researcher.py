"""Nightshift autonomous research engine.

Runs as a background task during quiet periods. Selects topics,
executes multi-step research using existing knowledge and external
providers, stores findings, and reports to a designated Discord channel.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import structlog

from signet.brain.client import Brain
from signet.config import settings
from signet.knowledge.store import WikiStore
from signet.memory.store import MemoryStore
from signet.models.research import (
    ResearchArtifact,
    ResearchReport,
    ResearchSection,
    ResearchStatus,
)
from signet.nightshift.research_prompts import (
    DEEP_DIVE_PROMPT,
    DEEP_DIVE_SYSTEM,
    RESEARCH_PLAN_PROMPT,
    RESEARCH_PLAN_SYSTEM,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM,
    TOPIC_SELECTION_PROMPT,
    TOPIC_SELECTION_SYSTEM,
)
from signet.nightshift.research_store import ResearchStore
from signet.nightshift.store import DreamStore
from signet.nightshift.wiki_writer import write_artifact_to_wiki

log = structlog.get_logger()


class Researcher:
    """Orchestrates autonomous nightshift research sessions."""

    def __init__(
        self,
        brain: Brain,
        memory: MemoryStore,
        wiki: WikiStore,
        dreams: DreamStore,
        research: ResearchStore,
    ) -> None:
        self._brain = brain
        self._memory = memory
        self._wiki = wiki
        self._dreams = dreams
        self._research = research
        self._interrupted = False

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self) -> None:
        """Signal the researcher to stop after the current step."""
        self._interrupted = True
        log.info("research.interrupted")

    async def run(self) -> ResearchReport:
        """Execute a full research session: pick topic, research, synthesize."""
        self._interrupted = False
        start = time.monotonic()

        # Check daily session limit
        sessions_today = await self._research.count_sessions_today()
        if sessions_today >= settings.nightshift_max_sessions:
            log.info(
                "research.session_limit_reached",
                sessions_today=sessions_today,
                max=settings.nightshift_max_sessions,
            )
            return ResearchReport(status=ResearchStatus.FAILED)

        # Check daily budget
        tokens_today = await self._research.total_tokens_today()
        if tokens_today >= settings.nightshift_daily_token_budget:
            log.info("research.budget_exceeded", tokens_today=tokens_today)
            return ResearchReport(status=ResearchStatus.FAILED)

        # Pick a topic
        topic, angle = await self._pick_topic()
        if not topic:
            log.info("research.no_topic")
            return ResearchReport(status=ResearchStatus.FAILED)

        # Create the artifact
        artifact = ResearchArtifact(
            topic=topic,
            angle=angle,
            status=ResearchStatus.IN_PROGRESS,
            model_used=settings.model_heavy,
        )
        await self._research.save(artifact)

        try:
            # Step 1: Plan
            artifact = await self._plan(artifact)
            await self._research.save(artifact)

            if self._interrupted:
                artifact.status = ResearchStatus.INTERRUPTED
                await self._research.save(artifact)
                return self._make_report(artifact, start)

            # Step 2: Deep dives
            artifact = await self._deep_dives(artifact)
            await self._research.save(artifact)

            if self._interrupted:
                artifact.status = ResearchStatus.INTERRUPTED
                await self._research.save(artifact)
                return self._make_report(artifact, start)

            # Step 3: Synthesis
            artifact = await self._synthesize(artifact)
            artifact.status = ResearchStatus.COMPLETED
            artifact.completed_at = datetime.now(timezone.utc)
            await self._research.save(artifact)

            # Step 4: Write back to wiki and sync to DB
            await self._write_to_wiki(artifact)

            log.info(
                "research.complete",
                topic=topic,
                sections=len(artifact.sections),
                tokens=artifact.token_count,
            )
            return self._make_report(artifact, start)

        except Exception:
            log.exception("research.error", topic=topic)
            artifact.status = ResearchStatus.FAILED
            await self._research.save(artifact)
            return self._make_report(artifact, start)

    async def _pick_topic(self) -> tuple[str, str]:
        """Select the best research topic from available candidates."""
        # Priority 1: explicit queue
        queued = await self._research.next_queued()
        if queued:
            queue_id, topic = queued
            # Use LLM to refine the angle
            angle = await self._refine_angle(topic)
            await self._research.consume_queue_item(queue_id)
            return topic, angle

        # Gather candidates from multiple sources
        candidates = await self._gather_candidates()
        if not candidates:
            return "", ""

        # Fetch recently completed topics so the LLM avoids them
        already_done = []
        try:
            recent_artifacts = await self._research.recent(
                limit=20, status=ResearchStatus.COMPLETED
            )
            already_done = [a.topic for a in recent_artifacts]
        except Exception:
            pass

        # Use LLM to pick the best one
        already_section = ""
        if already_done:
            already_section = (
                "\n\nAlready researched (DO NOT pick these or close variants):\n"
                + "\n".join(f"- {t}" for t in already_done)
            )
        prompt = TOPIC_SELECTION_PROMPT.format(
            candidates="\n".join(f"- {c}" for c in candidates)
        ) + already_section
        raw = await asyncio.to_thread(
            self._brain.quick, prompt, TOPIC_SELECTION_SYSTEM
        )

        try:
            data = _parse_json(raw)
            topic = data.get("topic", "")
            angle = data.get("angle", "")
            log.info("research.topic_selected", topic=topic, angle=angle)
            return topic, angle
        except (json.JSONDecodeError, KeyError):
            log.warning("research.topic_selection_failed", raw=raw[:200])
            # Fallback: use first candidate
            return candidates[0], ""

    async def _refine_angle(self, topic: str) -> str:
        """Given a broad topic, generate a specific research angle."""
        context = await self._gather_context(topic)
        prompt = RESEARCH_PLAN_PROMPT.format(
            topic=topic, angle="(to be determined)", context=context
        )
        raw = await asyncio.to_thread(
            self._brain.quick, prompt, RESEARCH_PLAN_SYSTEM
        )
        try:
            data = _parse_json(raw)
            questions = data.get("sub_questions", [])
            return questions[0] if questions else ""
        except (json.JSONDecodeError, KeyError):
            return ""

    async def _gather_candidates(self) -> list[str]:
        """Collect research topic candidates from various sources."""
        # Fetch recently completed topics to avoid repeats
        recently_researched: set[str] = set()
        try:
            recent_artifacts = await self._research.recent(
                limit=20, status=ResearchStatus.COMPLETED
            )
            for a in recent_artifacts:
                recently_researched.add(a.topic.lower())
        except Exception:
            log.exception("research.recent_lookup_error")

        candidates = []

        # Recent entity facts from dreams (what Pete's been working on)
        try:
            dream_results = await self._dreams.recall(
                "research topics cancer genomics bioinformatics",
                limit=5,
            )
            for r in dream_results:
                if r.dream.entity_name:
                    label = f"{r.dream.entity_name}: {r.dream.content[:100]}"
                    if r.dream.entity_name.lower() not in recently_researched:
                        candidates.append(label)
        except Exception:
            log.exception("research.candidate_dreams_error")

        # Recent wiki topics (existing knowledge to extend)
        try:
            wiki_results = await self._wiki.search(
                "recent developments cancer genomics",
                limit=3,
            )
            for r in wiki_results:
                title = r.article.frontmatter.title
                if title.lower() not in recently_researched:
                    candidates.append(f"Extend knowledge on: {title}")
        except Exception:
            log.exception("research.candidate_wiki_error")

        return candidates

    async def _gather_context(self, topic: str) -> str:
        """Pull existing knowledge relevant to a topic."""
        parts = []

        # Wiki context - pass substantial body content so the researcher
        # can actually use ingested PDFs and articles
        try:
            wiki_results = await self._wiki.search(topic, limit=3)
            for r in wiki_results:
                body = r.article.body
                summary = r.article.frontmatter.summary
                # Use full body up to 8k chars per article
                content = body[:8000] if body else (summary or "")
                parts.append(
                    f"[Wiki: {r.article.frontmatter.title}]\n{content}"
                )
        except Exception:
            pass

        # Dream context
        try:
            dream_results = await self._dreams.recall(topic, limit=3)
            for r in dream_results:
                parts.append(f"[Past insight] {r.dream.content[:200]}")
        except Exception:
            pass

        # Prior research on similar topics
        try:
            research_results = await self._research.recall(topic, limit=2)
            for r in research_results:
                parts.append(
                    f"[Prior research: {r.artifact.topic}] "
                    f"{r.artifact.synthesis[:200]}"
                )
        except Exception:
            pass

        return "\n\n".join(parts) if parts else "No existing context found."

    async def _plan(self, artifact: ResearchArtifact) -> ResearchArtifact:
        """Step 1: Generate a research plan with sub-questions."""
        context = await self._gather_context(artifact.topic)

        prompt = RESEARCH_PLAN_PROMPT.format(
            topic=artifact.topic,
            angle=artifact.angle,
            context=context,
        )

        raw = await asyncio.to_thread(
            self._brain.quick, prompt, RESEARCH_PLAN_SYSTEM
        )

        try:
            data = _parse_json(raw)
            questions = data.get("sub_questions", [])
        except (json.JSONDecodeError, KeyError):
            log.warning("research.plan_parse_failed", raw=raw[:200])
            questions = [artifact.angle or artifact.topic]

        # Cap at configured max
        questions = questions[: settings.nightshift_max_sections]

        artifact.plan = "\n".join(f"- {q}" for q in questions)
        artifact.sections = [
            ResearchSection(question=q, findings="", confidence="pending")
            for q in questions
        ]

        log.info("research.planned", topic=artifact.topic, questions=len(questions))
        return artifact

    async def _deep_dives(self, artifact: ResearchArtifact) -> ResearchArtifact:
        """Step 2: Investigate each sub-question."""
        total_tokens = 0

        for i, section in enumerate(artifact.sections):
            if self._interrupted:
                break

            context = await self._gather_context(
                f"{artifact.topic} {section.question}"
            )

            # Enrich with bioRxiv if available
            biorxiv_context = await self._fetch_biorxiv_context(
                artifact.topic, section.question
            )
            if biorxiv_context:
                context += f"\n\n{biorxiv_context}"

            # Enrich with PubMed if available
            pubmed_context = await self._fetch_pubmed_context(
                artifact.topic, section.question
            )
            if pubmed_context:
                context += f"\n\n{pubmed_context}"

            prompt = DEEP_DIVE_PROMPT.format(
                topic=artifact.topic,
                question=section.question,
                context=context,
            )

            from signet.models.memory import Message, MessageRole

            messages = [Message(role=MessageRole.USER, content=prompt)]
            raw = await asyncio.to_thread(
                self._brain.chat,
                DEEP_DIVE_SYSTEM,
                messages,
                settings.model_heavy,
                4096,
            )

            section.findings = raw
            section.confidence = "medium"  # could parse from response
            total_tokens += len(raw.split()) * 2  # rough estimate

            log.info(
                "research.deep_dive",
                topic=artifact.topic,
                section=i + 1,
                total=len(artifact.sections),
            )

        artifact.token_count += total_tokens
        return artifact

    async def _synthesize(self, artifact: ResearchArtifact) -> ResearchArtifact:
        """Step 3: Combine findings into a cohesive report."""
        section_findings = ""
        for i, section in enumerate(artifact.sections):
            if section.findings:
                section_findings += (
                    f"\n\n### Question {i + 1}: {section.question}\n"
                    f"{section.findings}"
                )

        prompt = SYNTHESIS_PROMPT.format(
            topic=artifact.topic,
            angle=artifact.angle,
            section_findings=section_findings,
        )

        from signet.models.memory import Message, MessageRole

        messages = [Message(role=MessageRole.USER, content=prompt)]
        raw = await asyncio.to_thread(
            self._brain.chat,
            SYNTHESIS_SYSTEM,
            messages,
            settings.model_heavy,
            8192,
        )

        parsed = _parse_synthesis(raw)
        artifact.synthesis = parsed["synthesis"] or raw
        artifact.confidence = parsed["confidence"] or "medium"
        artifact.open_questions = parsed["open_questions"]
        artifact.suggested_next = parsed["suggested_next"]

        return artifact

    async def _write_to_wiki(self, artifact: ResearchArtifact) -> None:
        """Step 4: Persist research as a wiki article and sync to DB.

        This closes the loop: research findings become wiki markdown,
        which gets embedded and available as context for future research.
        """
        try:
            wiki_path = write_artifact_to_wiki(artifact, settings.wikis_path)
            log.info("research.wiki_written", path=str(wiki_path))

            # Sync the wiki to DB so embeddings are immediately available
            result = await self._wiki.sync()
            log.info(
                "research.wiki_synced",
                added=result["added"],
                updated=result["updated"],
            )
        except Exception:
            # Wiki writeback is not critical - log and continue
            log.exception("research.wiki_write_failed", topic=artifact.topic)

    async def _fetch_biorxiv_context(self, topic: str, question: str) -> str:
        """Try to find relevant bioRxiv preprints. Returns formatted context or empty string."""
        try:
            from signet.providers.biorxiv import search_preprints

            query = f"{topic} {question}"[:100]
            preprints = await search_preprints(query, days=30, max_results=3)

            if not preprints:
                return ""

            lines = ["[Recent bioRxiv preprints]"]
            for p in preprints:
                lines.append(f"- {p.title} ({p.authors[:50]})")
                if p.abstract:
                    lines.append(f"  Abstract: {p.abstract[:300]}")
                lines.append(f"  URL: {p.url}")

            return "\n".join(lines)
        except Exception:
            return ""

    async def _fetch_pubmed_context(self, topic: str, question: str) -> str:
        """Try to find relevant PubMed articles. Returns formatted context or empty string."""
        try:
            from signet.providers.pubmed import search_and_fetch

            query = f"{topic} {question}"[:100]
            articles = await search_and_fetch(query, max_results=3)

            if not articles:
                return ""

            lines = ["[Relevant PubMed articles]"]
            for a in articles:
                lines.append(f"- {a.title} ({a.journal}, {a.pub_date})")
                if a.abstract:
                    lines.append(f"  Abstract: {a.abstract[:300]}")
                lines.append(f"  URL: {a.url}")
                if a.has_full_text:
                    lines.append(f"  Full text available (PMC: {a.pmc_id})")

            return "\n".join(lines)
        except Exception:
            return ""

    def _make_report(
        self, artifact: ResearchArtifact, start: float
    ) -> ResearchReport:
        """Build a report from a completed/failed/interrupted artifact."""
        return ResearchReport(
            topic=artifact.topic,
            sections_completed=sum(
                1 for s in artifact.sections if s.findings
            ),
            total_tokens=artifact.token_count,
            duration_seconds=time.monotonic() - start,
            status=artifact.status,
        )


def format_research_for_discord(artifact: ResearchArtifact) -> str:
    """Format a completed research artifact for Discord posting."""
    parts = [
        f"did some digging on **{artifact.topic}** overnight. "
        f"here's what i found:\n",
        artifact.synthesis,
    ]

    if artifact.confidence:
        parts.append(f"\n**confidence:** {artifact.confidence}")

    if artifact.open_questions:
        parts.append("\n**open questions:**")
        for q in artifact.open_questions:
            parts.append(f"- {q}")

    if artifact.suggested_next:
        parts.append("\n**next steps i'd suggest:**")
        for s in artifact.suggested_next:
            parts.append(f"- {s}")

    duration = ""
    if artifact.completed_at and artifact.started_at:
        mins = (artifact.completed_at - artifact.started_at).total_seconds() / 60
        duration = f" | took {mins:.0f} min"

    parts.append(f"\n*{artifact.token_count:,} tokens used{duration}*")

    return "\n".join(parts)


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)


_SYNTH_MARKERS = ("SYNTHESIS", "CONFIDENCE", "OPEN_QUESTIONS", "NEXT_STEPS")


def _parse_synthesis(raw: str) -> dict:
    """Parse the labeled synthesis response into its parts.

    Degrades gracefully if the model drops a marker or the response is
    truncated mid-section. Falls back to JSON parsing if the model
    ignored the marker format and returned JSON anyway.
    """
    text = raw.strip()
    # Strip an outer code fence if the model wrapped the whole thing anyway
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    sections: dict[str, str] = {m: "" for m in _SYNTH_MARKERS}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        marker = None
        for m in _SYNTH_MARKERS:
            if stripped == f"==={m}===" or stripped == f"==={m}":
                marker = m
                break
        if marker is not None:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = marker
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()

    def _bullets(block: str) -> list[str]:
        out: list[str] = []
        for line in block.splitlines():
            s = line.strip()
            if s.startswith(("- ", "* ")):
                out.append(s[2:].strip())
            elif s.startswith("-") or s.startswith("*"):
                out.append(s[1:].strip())
        return [b for b in out if b]

    # Fallback: model ignored markers and returned JSON
    if not sections["SYNTHESIS"] and text.lstrip().startswith("{"):
        try:
            data = _parse_json(raw)
            synth_field = data.get("synthesis", "")
            if isinstance(synth_field, list):
                synth_field = "\n\n".join(str(s) for s in synth_field)
            conf = str(data.get("confidence", "")).strip().lower()
            oq = data.get("open_questions", []) or []
            ns = data.get("suggested_next", []) or data.get("next_steps", []) or []
            return {
                "synthesis": str(synth_field).strip(),
                "confidence": conf,
                "open_questions": [str(x).strip() for x in oq if str(x).strip()],
                "suggested_next": [str(x).strip() for x in ns if str(x).strip()],
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return {
        "synthesis": sections["SYNTHESIS"],
        "confidence": sections["CONFIDENCE"].splitlines()[0].strip().lower()
        if sections["CONFIDENCE"]
        else "",
        "open_questions": _bullets(sections["OPEN_QUESTIONS"]),
        "suggested_next": _bullets(sections["NEXT_STEPS"]),
    }
