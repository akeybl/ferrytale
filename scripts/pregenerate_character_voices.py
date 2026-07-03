#!/usr/bin/env python3
"""Pre-generate OmniVoice character voices for a transcript-backed game."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from elevenlabs_voices import (  # noqa: E402
    CharacterVoiceCache,
    DEFAULT_DESCRIPTION_MODEL,
    DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
    ELEVENLABS_PREVIEW_TEXT,
    _gemini_usage_to_dict,
    load_env_file,
    transcript_prompt_preamble,
)


def _strip_json_fence(text: str) -> str:
    stripped = str(text).strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip()
    name = name.strip("\"'")
    return name


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw_name in names:
        name = _clean_name(raw_name)
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _slugish(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def load_index(transcripts_dir: Path) -> dict[str, dict[str, Any]]:
    path = transcripts_dir / "index.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain an object")
    return {
        str(slug): value if isinstance(value, dict) else {}
        for slug, value in data.items()
    }


def resolve_transcript(query: str, transcripts_dir: Path) -> tuple[str, Path, str]:
    explicit = Path(query).expanduser()
    if explicit.is_file():
        return explicit.stem, explicit, explicit.stem

    candidates = [
        query,
        query.replace(" ", "-").replace("_", "-"),
        _slugish(query),
    ]
    for candidate in candidates:
        path = transcripts_dir / f"{candidate}.txt"
        if path.exists():
            index = load_index(transcripts_dir)
            title = str(index.get(candidate, {}).get("title") or candidate)
            return candidate, path, title

    index = load_index(transcripts_dir)
    needle = query.casefold()
    slug_needle = _slugish(query)
    matches: list[tuple[str, dict[str, Any]]] = []
    for slug, metadata in index.items():
        haystack = " ".join(
            str(metadata.get(key) or "") for key in ("title", "author", "source")
        )
        if needle in f"{slug} {haystack}".casefold() or slug_needle in slug:
            matches.append((slug, metadata))

    if len(matches) == 1:
        slug, metadata = matches[0]
        path = transcripts_dir / f"{slug}.txt"
        if path.exists():
            return slug, path, str(metadata.get("title") or slug)
    if len(matches) > 1:
        choices = "\n".join(
            f"  {slug}: {metadata.get('title') or slug}"
            for slug, metadata in matches[:20]
        )
        raise RuntimeError(f"{query!r} matches multiple games:\n{choices}")
    raise RuntimeError(f"could not find transcript for {query!r} in {transcripts_dir}")


def parse_character_response(text: str) -> list[str]:
    try:
        data = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini did not return valid JSON: {text[:300]!r}") from exc

    raw_characters: Any
    if isinstance(data, dict):
        raw_characters = data.get("characters", [])
    else:
        raw_characters = data

    names: list[str] = []
    if isinstance(raw_characters, list):
        for item in raw_characters:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.append(str(item.get("name") or ""))
    return _dedupe_names(names)


def discover_dialogue_characters(
    *,
    transcript_text: str,
    game_title: str,
    api_key: str,
    cost_recorder=None,
) -> list[str]:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for character discovery") from exc

    # The shared transcript-first preamble is byte-identical to the one used
    # by the per-character description calls, so this discovery call warms the
    # Gemini implicit cache for the transcript prefix they all reuse.
    prompt = transcript_prompt_preamble(game_title, transcript_text) + """\
Identify every separate speaker who has spoken dialogue in the transcript
above. Return only JSON with this exact shape:
{"characters":[{"name":"..."}]}

Rules:
- Include named characters and unnamed role speakers with actual spoken lines.
- For unnamed speakers, create a short stable descriptive role name, such as
  "hotel manager", "day clerk", "room constable", or "cab driver".
- Add local context only when it is needed to distinguish multiple speakers
  with the same role.
- Merge aliases for the same speaker under one stable name.
- Exclude the narrator, parser, player commands, game UI, room names, and
  purely described characters with no dialogue.
- Order speakers by their first meaningful dialogue appearance.
"""
    model = os.environ.get(
        "IF_ENGINE_CHARACTER_DISCOVERY_MODEL",
        os.environ.get("IF_ENGINE_CHARACTER_VOICE_DESCRIPTION_MODEL", DEFAULT_DESCRIPTION_MODEL),
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    if cost_recorder is not None:
        usage = _gemini_usage_to_dict(getattr(response, "usage_metadata", None))
        if usage is not None:
            cost_recorder(
                {
                    "service": "gemini",
                    "category": "character_discovery",
                    "label": "Gemini dialogue character discovery",
                    "model": model,
                    "usage": usage,
                }
            )
    names = parse_character_response(getattr(response, "text", "") or "")
    if not names:
        raise RuntimeError("Gemini did not identify any dialogue characters")
    return names


def explicit_character_names(values: list[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(part for part in value.split(",") if part.strip())
    return _dedupe_names(names)


def elevenlabs_rate() -> float:
    value = os.environ.get("IF_ENGINE_ELEVENLABS_VOICE_DESIGN_PRICE_PER_1K_CHARS", "")
    if not value:
        return 0.10
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.10


def print_cost_summary(events: list[dict[str, Any]]) -> None:
    gemini_events = [event for event in events if event.get("service") == "gemini"]
    elevenlabs_events = [
        event
        for event in events
        if event.get("service") == "elevenlabs" and event.get("category") == "voice_design"
    ]
    if not gemini_events and not elevenlabs_events:
        return
    print("\nCost events:")
    if gemini_events:
        prompt = output = thoughts = cached = 0
        for event in gemini_events:
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            prompt += int(usage.get("prompt", 0) or 0)
            cached += int(usage.get("cached", 0) or 0)
            output += int(usage.get("output", 0) or 0)
            thoughts += int(usage.get("thoughts", 0) or 0)
        print(
            f"  Gemini (discovery + descriptions): {len(gemini_events)} calls; "
            f"in {prompt:,}, cached {cached:,}, out {output:,}, thoughts {thoughts:,}"
        )
    if elevenlabs_events:
        credits = sum(
            int(event.get("credits", event.get("characters", 0)) or 0)
            for event in elevenlabs_events
        )
        estimate = credits * elevenlabs_rate() / 1000
        print(
            f"  ElevenLabs Voice Design: {len(elevenlabs_events)} calls; "
            f"{credits:,} credits; estimated ${estimate:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-generate cached ElevenLabs-designed OmniVoice character voices."
    )
    parser.add_argument("game", help="transcript slug, title match, or transcript .txt path")
    parser.add_argument(
        "-c",
        "--character",
        action="append",
        default=[],
        help="character name to generate; repeat or comma-separate to skip Gemini discovery",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only process the first N discovered/explicit characters",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
        help="voice cache root; defaults to .cache/elevenlabs-voices",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        help="directory containing transcripts and index.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list target characters and cache status without creating voices",
    )
    args = parser.parse_args()

    load_env_file(BASE_DIR / ".env")
    transcript_stem, transcript_path, game_title = resolve_transcript(
        args.game,
        args.transcripts_dir.expanduser(),
    )
    transcript_text = transcript_path.read_text(encoding="utf-8")

    cost_events: list[dict[str, Any]] = []
    names = explicit_character_names(args.character)
    if not names:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for character discovery")
        print(f"Discovering dialogue characters for {game_title}...")
        names = discover_dialogue_characters(
            transcript_text=transcript_text,
            game_title=game_title,
            api_key=api_key,
            cost_recorder=cost_events.append,
        )

    if args.limit > 0:
        names = names[: args.limit]

    cache = CharacterVoiceCache(
        cache_root=args.cache_dir.expanduser(),
        transcript_filename_stem=transcript_stem,
        transcript_filename=transcript_path.name,
        transcript_text=transcript_text,
        game_title=game_title,
        log=lambda message: print(f"  {message}"),
        cost_recorder=lambda event: cost_events.append(event),
        max_workers=1,
    )

    print(f"Game: {game_title} ({transcript_path.name})")
    print(f"Cache: {args.cache_dir.expanduser() / transcript_stem}")
    print(f"Preview text: {ELEVENLABS_PREVIEW_TEXT!r}")
    print(f"Characters: {len(names)}")

    failures: list[str] = []
    for name in names:
        cached = cache.cached_voice(name)
        if cached is not None:
            print(f"  cached: {name}")
            continue
        if args.dry_run:
            print(f"  would create: {name}")
            continue
        print(f"  creating: {name}")
        metadata = cache.wait_for_ready(name)
        if metadata is None:
            failures.append(name)
            print(f"  failed: {name}")
        else:
            print(f"  ready: {metadata.character_name} -> {metadata.cache_dir}")

    print_cost_summary(cost_events)
    if failures:
        print("\nFailed voices:")
        for name in failures:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
