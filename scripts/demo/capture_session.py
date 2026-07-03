#!/usr/bin/env python3
"""Capture the demo's game session with real Gemini narration.

Runs the real engine (no audio) against the demo game, with the system prompt
configured exactly as OmniVoice voice mode would be — so the narration carries
<voice name="..."> tags that match the pregenerated character voice cache.

The captured session lands in .cache/demo/sessions/<name>.jsonl and is what
scripts/demo/render_demo.py replays with full voice + video. Iterate freely:

    .venv/bin/python scripts/demo/capture_session.py --new           # opening
    .venv/bin/python scripts/demo/capture_session.py --turn "..."    # one turn
    .venv/bin/python scripts/demo/capture_session.py --show          # review
    .venv/bin/python scripts/demo/capture_session.py --undo          # drop last turn
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ENGINE_PATH = BASE_DIR / "ferrytale.py"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEMO_SESSIONS_DIR = BASE_DIR / ".cache" / "demo" / "sessions"
DEFAULT_SESSION = "demo-make-it-good"
DEFAULT_GAME = "make-it-good"


def load_engine():
    spec = importlib.util.spec_from_file_location("ferrytale_engine", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_game(engine, session_name: str, game_slug: str, fresh: bool):
    engine.SESSIONS_DIR = DEMO_SESSIONS_DIR
    DEMO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DEMO_SESSIONS_DIR / f"{session_name}.jsonl"
    if fresh and path.exists():
        path.unlink()
    session = engine.Session(session_name)
    if session.is_new:
        session.append({"type": "meta", "game": game_slug})
    game = engine.Game(
        session,
        engine.load_transcript_text(game_slug),
        game_name=game_slug,
        game_title=engine.game_title_for(game_slug),
    )
    # Match live OmniVoice voice mode so narration carries <voice> tags and
    # the system prompt lists the pregenerated character voices.
    game.refresh_system_prompt(
        voice_prompt_enabled=True,
        tts_engine="omnivoice",
        wake_word_enabled=False,
        car_mode=False,
        wake_threshold=None,
        wake_preprocess=False,
        omnivoice_whisper_tags=False,
        omnivoice_character_voices=True,
        reestimate_context=True,
    )
    return game


def show(engine, game) -> None:
    for i, e in enumerate(game.session.events):
        if e["type"] in ("player", "interruption"):
            print(f"\n[{i}] ❯ {e.get('shown') or e['text']}")
        elif e["type"] == "narrator":
            print(f"\n[{i}] ── narrator ({e.get('label', 'turn')}):")
            print(engine.visible_display_text(e["text"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the demo game session.")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--game", default=DEFAULT_GAME)
    parser.add_argument("--new", action="store_true", help="start a fresh capture")
    parser.add_argument("--turn", action="append", default=[],
                        help="player input to run (repeatable, in order)")
    parser.add_argument("--note", default=None,
                        help="invisible engine reminder recorded before the turns "
                             "(e.g. to ask for tighter narration)")
    parser.add_argument("--show", action="store_true", help="print the session so far")
    parser.add_argument("--undo", action="store_true",
                        help="remove the last player+narrator turn")
    args = parser.parse_args()

    engine = load_engine()

    if args.undo:
        engine.SESSIONS_DIR = DEMO_SESSIONS_DIR
        session = engine.Session(args.session)
        events = session.events
        cut = None
        for i in range(len(events) - 1, -1, -1):
            if events[i]["type"] in ("player", "interruption"):
                cut = i
                break
        if cut is None:
            print("nothing to undo")
            return 1
        session.events = events[:cut]
        engine._rewrite_session_file(session)
        print(f"dropped events from index {cut}")
        return 0

    game = build_game(engine, args.session, args.game, fresh=args.new)

    if game.needs_opening() and (args.new or args.turn):
        print("── generating opening page…", file=sys.stderr)
        game.narrator_turn("opening", emit=lambda _t: None)

    if args.note:
        game.record_engine_reminder(f"[ENGINE NOTE — not a player action.]\n{args.note}")

    for text in args.turn:
        print(f"── turn: {text!r}", file=sys.stderr)
        game.record_player_turn(text)
        game.narrator_turn("turn", emit=lambda _t: None)

    if args.show or args.turn or args.new:
        show(engine, game)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
