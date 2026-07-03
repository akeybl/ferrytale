#!/usr/bin/env python3
"""Design and cache the demo's fake player voice, plus per-line TTS audio.

The demo video needs a "player" speaking commands aloud. This module:

1. designs a natural male voice via ElevenLabs Voice Design (once),
2. persists the chosen preview as a permanent voice (once), and
3. synthesizes each scripted player line with that voice (cached per line).

Everything is cached under .cache/demo/ (gitignored), so re-rendering the
demo never re-bills ElevenLabs unless the cache is deleted or a line changes.

Usage:
    .venv/bin/python scripts/demo/player_voice.py --line "Get out of the car."
    .venv/bin/python scripts/demo/player_voice.py --script demo_script.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from elevenlabs_voices import (  # noqa: E402
    _select_widest_preview,
    _default_preview_audio_scorer,
    load_env_file,
)

DEMO_CACHE_DIR = BASE_DIR / ".cache" / "demo"
PLAYER_VOICE_JSON = DEMO_CACHE_DIR / "player-voice" / "voice.json"
PLAYER_VOICE_PREVIEW = DEMO_CACHE_DIR / "player-voice" / "preview.mp3"
PLAYER_LINES_DIR = DEMO_CACHE_DIR / "player-lines"

PLAYER_VOICE_NAME = "Ferrytale Demo Player"
PLAYER_VOICE_DESCRIPTION = (
    "A relaxed American man in his early thirties speaking casually and "
    "clearly, as if sitting at his desk playing a voice-controlled game. "
    "Medium pitch, natural conversational pacing with small thoughtful "
    "pauses, slightly amused and engaged, recorded on a good desktop "
    "microphone with natural room tone. Perfect audio quality, "
    "studio-quality recording."
)
PLAYER_VOICE_PREVIEW_TEXT = (
    "Alright, let's take a look around. Open the glove compartment, grab my "
    "badge, and then get out of the car and head up the path to the front door."
)
TTS_MODEL_ID = "eleven_multilingual_v2"


def _api_key() -> str:
    load_env_file(BASE_DIR / ".env")
    import os

    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    return key


def _post_json(url: str, payload: dict, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "xi-api-key": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_audio(url: str, payload: dict, api_key: str) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def ensure_player_voice() -> dict:
    """Design (once) and persist (once) the demo player voice; return its
    cached metadata including the permanent ElevenLabs voice_id."""
    if PLAYER_VOICE_JSON.exists():
        return json.loads(PLAYER_VOICE_JSON.read_text(encoding="utf-8"))
    api_key = _api_key()
    print("designing demo player voice via ElevenLabs...", file=sys.stderr)
    design = _post_json(
        "https://api.elevenlabs.io/v1/text-to-voice/design",
        {
            "voice_description": PLAYER_VOICE_DESCRIPTION,
            "text": PLAYER_VOICE_PREVIEW_TEXT,
            "auto_generate_text": False,
        },
        api_key,
    )
    previews = design.get("previews") or []
    if not previews:
        raise RuntimeError("ElevenLabs returned no previews for the player voice")
    selected = _select_widest_preview(previews, _default_preview_audio_scorer)
    print("persisting the selected preview as a permanent voice...", file=sys.stderr)
    created = _post_json(
        "https://api.elevenlabs.io/v1/text-to-voice",
        {
            "voice_name": PLAYER_VOICE_NAME,
            "voice_description": PLAYER_VOICE_DESCRIPTION,
            "generated_voice_id": selected.generated_voice_id,
        },
        api_key,
    )
    voice_id = created.get("voice_id")
    if not voice_id:
        raise RuntimeError(f"voice creation returned no voice_id: {created}")
    PLAYER_VOICE_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    PLAYER_VOICE_PREVIEW.write_bytes(selected.audio)
    metadata = {
        "voice_id": voice_id,
        "generated_voice_id": selected.generated_voice_id,
        "voice_name": PLAYER_VOICE_NAME,
        "voice_description": PLAYER_VOICE_DESCRIPTION,
        "preview_text": PLAYER_VOICE_PREVIEW_TEXT,
        "tts_model_id": TTS_MODEL_ID,
    }
    PLAYER_VOICE_JSON.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"player voice ready: {voice_id}", file=sys.stderr)
    return metadata


def line_audio_path(text: str) -> Path:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    return PLAYER_LINES_DIR / f"{digest}.mp3"


def ensure_line_audio(text: str) -> Path:
    """Synthesize one player line with the cached voice (cached per line)."""
    path = line_audio_path(text)
    if path.exists() and path.stat().st_size > 0:
        return path
    voice = ensure_player_voice()
    api_key = _api_key()
    print(f"synthesizing player line: {text!r}", file=sys.stderr)
    audio = _post_audio(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice['voice_id']}"
        "?output_format=mp3_44100_128",
        {"text": text, "model_id": voice.get("tts_model_id", TTS_MODEL_ID)},
        api_key,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return path


def lines_from_script(script_path: Path) -> list[str]:
    data = json.loads(script_path.read_text(encoding="utf-8"))
    lines = []
    for step in data.get("steps", []):
        spoken = step.get("spoken") or step.get("text")
        if spoken:
            lines.append(str(spoken))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache the demo player voice + lines.")
    parser.add_argument("--line", action="append", default=[], help="player line to TTS")
    parser.add_argument("--script", type=Path, default=None,
                        help="demo script JSON; synthesizes every step's spoken line")
    args = parser.parse_args()

    ensure_player_voice()
    lines = list(args.line)
    if args.script is not None:
        lines.extend(lines_from_script(args.script))
    for text in lines:
        print(ensure_line_audio(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
