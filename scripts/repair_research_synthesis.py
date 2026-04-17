"""One-off repair for nightshift research artifacts whose synthesis field was
stored as a raw ```json ...``` blob because the old synthesis path tried to
wrap a long markdown string inside JSON and the parse failed.

Run dry-run first to see what would change:

    uv run python scripts/repair_research_synthesis.py

Apply fixes:

    uv run python scripts/repair_research_synthesis.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

import asyncpg
from pgvector.asyncpg import register_vector

from signet.config import settings
from signet.memory.embeddings import EmbeddingService


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _looks_broken(synthesis: str | None) -> bool:
    if not synthesis:
        return False
    s = synthesis.lstrip()
    if s.startswith("```json") or s.startswith("```JSON"):
        return True
    # bare JSON with a synthesis key at the top
    if s.startswith("{") and '"synthesis"' in s[:200]:
        return True
    return False


def _try_parse_json(text: str) -> dict | None:
    """Try hard to parse a JSON blob — including truncated ones where the
    trailing ``` or closing brace is missing."""
    candidate = _strip_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # try: chop off trailing garbage progressively and append a closing brace
    for end in range(len(candidate), 0, -1):
        piece = candidate[:end]
        if piece.rstrip().endswith(('"', "]", "}")):
            for suffix in ("", "}", '"}', '"]}'):
                try:
                    return json.loads(piece + suffix)
                except json.JSONDecodeError:
                    continue
    return None


def _extract_synthesis_string(text: str) -> str | None:
    """Fallback: regex-extract the value of the synthesis key even if the
    JSON is broken/truncated."""
    m = re.search(r'"synthesis"\s*:\s*"', text)
    if not m:
        return None
    start = m.end()
    out: list[str] = []
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
            if nxt in mapping:
                out.append(mapping[nxt])
                i += 2
                continue
            if nxt == "u" and i + 5 < len(text):
                try:
                    out.append(chr(int(text[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(nxt)
            i += 2
            continue
        if ch == '"':
            return "".join(out)
        out.append(ch)
        i += 1
    # truncated — return what we have
    return "".join(out) if out else None


def _repair(raw: str) -> dict:
    """Return a dict with repaired synthesis/confidence/open_questions/suggested_next.
    Empty strings / lists if a field couldn't be recovered."""
    result: dict = {
        "synthesis": "",
        "confidence": "",
        "open_questions": [],
        "suggested_next": [],
    }

    parsed = _try_parse_json(raw)
    if parsed and isinstance(parsed, dict):
        if "synthesis" in parsed and isinstance(parsed["synthesis"], str):
            result["synthesis"] = parsed["synthesis"].strip()
        if isinstance(parsed.get("confidence"), str):
            result["confidence"] = parsed["confidence"].strip()
        if isinstance(parsed.get("open_questions"), list):
            result["open_questions"] = [
                str(q).strip() for q in parsed["open_questions"] if str(q).strip()
            ]
        if isinstance(parsed.get("suggested_next"), list):
            result["suggested_next"] = [
                str(q).strip() for q in parsed["suggested_next"] if str(q).strip()
            ]
        return result

    # Couldn't parse — try to recover just the synthesis string
    recovered = _extract_synthesis_string(_strip_fence(raw))
    if recovered:
        result["synthesis"] = recovered.strip()
    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply repairs")
    args = parser.parse_args()

    pool = await asyncpg.create_pool(
        settings.database_url, min_size=1, max_size=2, init=register_vector
    )
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, topic, synthesis, confidence, open_questions, "
                "suggested_next FROM research ORDER BY started_at DESC"
            )

        broken = []
        for row in rows:
            if _looks_broken(row["synthesis"]):
                broken.append(row)

        if not broken:
            print("No broken research artifacts found.")
            return 0

        print(f"Found {len(broken)} broken artifact(s):\n")
        repairs: list[tuple] = []
        for row in broken:
            repaired = _repair(row["synthesis"])
            orig_preview = (row["synthesis"] or "")[:80].replace("\n", " ")
            new_preview = (repaired["synthesis"] or "")[:80].replace("\n", " ")
            print(f"  id={row['id']}")
            print(f"    topic: {row['topic']}")
            print(f"    before: {orig_preview}...")
            print(f"    after : {new_preview}...")
            print(
                f"    confidence={repaired['confidence'] or '(unchanged)'}, "
                f"open_qs={len(repaired['open_questions'])}, "
                f"next={len(repaired['suggested_next'])}"
            )
            print()
            if not repaired["synthesis"]:
                print("    WARNING: could not recover synthesis; skipping.\n")
                continue
            repairs.append((row["id"], repaired))

        if not args.apply:
            print(f"Dry run. {len(repairs)} artifact(s) would be updated.")
            print("Re-run with --apply to write changes.")
            return 0

        print(f"Applying {len(repairs)} repair(s)...")
        embedder = EmbeddingService()
        async with pool.acquire() as conn:
            for artifact_id, repaired in repairs:
                embedding = await embedder.embed(repaired["synthesis"])
                await conn.execute(
                    """
                    UPDATE research
                    SET synthesis = $2,
                        confidence = COALESCE(NULLIF($3, ''), confidence),
                        open_questions = CASE WHEN $4::text[] = '{}'::text[]
                            THEN open_questions ELSE $4 END,
                        suggested_next = CASE WHEN $5::text[] = '{}'::text[]
                            THEN suggested_next ELSE $5 END,
                        embedding = $6
                    WHERE id = $1
                    """,
                    artifact_id,
                    repaired["synthesis"],
                    repaired["confidence"],
                    repaired["open_questions"],
                    repaired["suggested_next"],
                    embedding,
                )
        print("Done.")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
