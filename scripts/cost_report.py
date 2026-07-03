#!/usr/bin/env python3
"""Measure real per-event API costs for a game and emit a markdown table.

Drives the actual engine code paths — the same Gemini narration calls,
compaction call, character discovery, and ElevenLabs Voice Design calls that
live play makes — against a throwaway session and a temporary voice cache, so
every number comes from real API responses (usage metadata and the ElevenLabs
`character-cost` header), never from estimates.

This SPENDS REAL MONEY (typically well under a dollar for a mid-size game;
roughly $0.50 for Anchorhead). The throwaway session and generated voices are
discarded afterwards.

Examples:
    .venv/bin/python scripts/cost_report.py anchorhead
    .venv/bin/python scripts/cost_report.py anchorhead --turns 3 --skip-voices
    .venv/bin/python scripts/cost_report.py anchorhead \
        --character "Michael" --character "real estate agent"
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ENGINE_PATH = BASE_DIR / "ferrytale.py"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "scripts"))


def load_engine_module():
    spec = importlib.util.spec_from_file_location("ferrytale_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load engine module from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine_module()
import elevenlabs_voices as ev  # noqa: E402
import pregenerate_character_voices as pre  # noqa: E402


def status(message: str) -> None:
    print(f"[cost-report] {message}", file=sys.stderr, flush=True)


def silent(_text: str) -> None:
    pass


class Row:
    def __init__(
        self,
        event: str,
        usage: dict | None = None,
        gemini_cost: float = 0.0,
        credits: int = 0,
        elevenlabs_cost: float = 0.0,
        note: str = "",
    ):
        self.event = event
        self.usage = usage
        self.gemini_cost = gemini_cost
        self.credits = credits
        self.elevenlabs_cost = elevenlabs_cost
        self.note = note

    @property
    def total(self) -> float:
        return self.gemini_cost + self.elevenlabs_cost


def usage_cells(usage: dict | None) -> tuple[str, str, str]:
    if not usage:
        return "—", "—", "—"
    out_plus_thoughts = int(usage.get("output", 0)) + int(usage.get("thoughts", 0))
    return (
        f"{int(usage.get('prompt', 0)):,}",
        f"{int(usage.get('cached', 0)):,}",
        f"{out_plus_thoughts:,}",
    )


def gemini_row(event: str, usage: dict, note: str = "") -> Row:
    return Row(event, usage=usage, gemini_cost=engine.usage_cost(usage), note=note)


def markdown_report(
    *,
    game: str,
    title: str,
    rows: list[Row],
    notes: list[str],
) -> str:
    today = datetime.date.today().isoformat()
    lines = [
        f"# Ferrytale cost report — {title} ({game})",
        "",
        f"Measured {today} with `{engine.MODEL}` (thinking level "
        f"`{engine.THINKING_LEVEL}`) and real API responses; ElevenLabs dollar "
        f"figures use ${engine.elevenlabs_voice_design_rate_per_1k_chars():.2f}"
        "/1K credits and stay estimates (plan-dependent).",
        "",
        "| Event | Input tokens | Cached | Output + thoughts | ElevenLabs credits | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        prompt, cached, out = usage_cells(row.usage)
        credits = f"{row.credits:,}" if row.credits else "—"
        event = row.event + (f" — {row.note}" if row.note else "")
        lines.append(
            f"| {event} | {prompt} | {cached} | {out} | {credits} | ${row.total:.4f} |"
        )
    total = sum(row.total for row in rows)
    lines.append(f"| **Total for this report** |  |  |  |  | **${total:.4f}** |")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure real per-event API costs and emit a markdown table."
    )
    parser.add_argument("game", help="catalog/transcript slug, e.g. anchorhead")
    parser.add_argument("--turns", type=int, default=2,
                        help="player turns to measure after the opening (default 2)")
    parser.add_argument("--skip-compaction", action="store_true",
                        help="skip the forced compaction measurement")
    parser.add_argument("--skip-discovery", action="store_true",
                        help="skip the Gemini character-discovery measurement")
    parser.add_argument("--skip-voices", action="store_true",
                        help="skip ElevenLabs character voice creation")
    parser.add_argument("--character", action="append", default=[],
                        help="character name(s) for voice creation; default: first "
                             "two discovered speakers")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the markdown report to this file")
    args = parser.parse_args()

    engine.require_gemini_api_key()
    transcript_text = engine.load_transcript_text(args.game)
    title = engine.game_title_for(args.game)
    rows: list[Row] = []
    notes: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ferrytale-cost-report-") as tmp:
        tmp_path = Path(tmp)
        # Keep the measurement session out of the real sessions/ directory.
        engine.SESSIONS_DIR = tmp_path / "sessions"
        session = engine.Session("cost-report")
        session.append({"type": "meta", "game": args.game})
        game = engine.Game(
            session,
            transcript_text,
            game_name=args.game,
            game_title=title,
        )

        status(f"opening page for {title} (~{game.context_tokens:,} est. tokens)…")
        game.narrator_turn("opening", emit=silent)
        opening = next(e for e in reversed(session.events) if e["type"] == "narrator")
        rows.append(gemini_row(
            "Opening page (new game, cold cache)", opening["usage"]))
        if int(opening["usage"].get("cached", 0)) > 0:
            notes.append(
                "The opening call reported cached tokens — an identical prefix "
                "was requested within the implicit-cache window (e.g. a recent "
                "run of this script). Re-run later for a true cold-cache figure."
            )

        turn_inputs = [
            "Look around slowly and carefully.",
            "Keep going, taking in every detail.",
            "Pause and consider what to do next.",
        ]
        for i in range(max(0, args.turns)):
            text = turn_inputs[i % len(turn_inputs)]
            status(f"player turn {i + 1}…")
            game.record_player_turn(text)
            game.narrator_turn("turn", emit=silent)
            turn = next(e for e in reversed(session.events) if e["type"] == "narrator")
            rows.append(gemini_row(
                f"Player turn {i + 1} (warm cache)", turn["usage"]))

        if not args.skip_compaction:
            status("forcing a compaction…")
            game.compact_at_override = 1
            game.compact_at = 1
            game.maybe_compact(emit=silent)
            compaction = next(
                (e for e in reversed(session.events) if e["type"] == "compaction"),
                None,
            )
            if compaction is not None:
                rows.append(gemini_row(
                    "Compaction (summary of full context)", compaction["usage"]))

    characters = pre.explicit_character_names(args.character)
    if not args.skip_discovery or (not args.skip_voices and not characters):
        status("discovering dialogue characters…")
        discovery_events: list[dict] = []
        discovered = pre.discover_dialogue_characters(
            transcript_text=transcript_text,
            game_title=title,
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            cost_recorder=discovery_events.append,
        )
        if not args.skip_discovery and discovery_events:
            rows.append(gemini_row(
                "Character discovery (pregeneration only)",
                discovery_events[0]["usage"]))
        if not characters:
            characters = discovered[:2]

    if not args.skip_voices:
        if not os.environ.get("ELEVENLABS_API_KEY"):
            notes.append(
                "ELEVENLABS_API_KEY is not set — character voice creation was "
                "skipped."
            )
        elif not characters:
            notes.append("No characters resolved — voice creation was skipped.")
        else:
            with tempfile.TemporaryDirectory(
                prefix="ferrytale-cost-voices-"
            ) as voice_tmp:
                voice_events: list[dict] = []
                cache = ev.CharacterVoiceCache(
                    cache_root=voice_tmp,
                    transcript_filename_stem=args.game,
                    transcript_filename=f"{args.game}.txt",
                    transcript_text=transcript_text,
                    game_title=title,
                    cost_recorder=voice_events.append,
                    log=lambda message: status(message),
                    max_workers=1,
                )
                for i, name in enumerate(characters[:2]):
                    status(f"creating character voice {i + 1}: {name!r}…")
                    before = len(voice_events)
                    metadata = cache.wait_for_ready(name)
                    if metadata is None:
                        notes.append(f"Voice creation failed for {name!r}.")
                        continue
                    ordinal = "First" if i == 0 else "Subsequent"
                    for event in voice_events[before:]:
                        if event.get("service") == "gemini":
                            rows.append(gemini_row(
                                f"{ordinal} character voice — Gemini description",
                                event["usage"], note=name))
                        elif event.get("service") == "elevenlabs":
                            credits = int(event.get("credits", 0) or 0)
                            rate = engine.elevenlabs_voice_design_rate_per_1k_chars()
                            rows.append(Row(
                                f"{ordinal} character voice — ElevenLabs design",
                                credits=credits,
                                elevenlabs_cost=credits * rate / 1_000,
                                note=name,
                            ))
            rows.append(Row(
                "Reusing a cached character voice",
                note="no API calls",
            ))

    notes.append(
        "Costs scale with transcript size; warm-cache rows depend on Gemini "
        "implicit caching (needs a ≥4,096-token prefix, and requests within a "
        "few minutes of each other)."
    )
    notes.append(
        "A turn after a long break (cold cache) re-pays fresh input for the "
        "whole context — it costs about the same as the opening page."
    )

    report = markdown_report(game=args.game, title=title, rows=rows, notes=notes)
    print(report)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
        status(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
