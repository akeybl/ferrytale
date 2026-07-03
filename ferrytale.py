#!/usr/bin/env python3
"""Ferrytale — a transcript-grounded Interactive Fiction AI "Interpreter".

Plays classic interactive fiction from full transcript canon via the Gemini API.
It does not run original story files such as .gblorb/.zblorb/.ulx. Run
build_transcripts.py --game <slug> to populate one transcript cache file, or
let ./play download the selected transcript on demand.

Usage:
    python ferrytale.py                    # resume newest session
    python ferrytale.py [session_name]      # new or resumed named session
    python ferrytale.py --new [session]     # start a fresh session
    python ferrytale.py --new --game superluminal-vagrant-twin
    python ferrytale.py --list              # list saved sessions
    python ferrytale.py --list-games        # list catalogued games

Type actions/dialog at any time; pressing Enter during narration interrupts
the stream and sends your text as a clarification or redirection.
/quit leaves the game (progress is always saved).

With voice enabled (the default in the live terminal; disable with
--no-voice), narration is also read aloud with the configured TTS engine and
the microphone can accept spoken input (Silero VAD + WebRTC AEC + whisper.cpp).
By default, hold Shift while speaking to submit voice input. Caps Lock toggles
open mic. When wake word mode is enabled, the default voice input path listens
for the wake word before accepting speech.
"""

import argparse
import asyncio
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from elevenlabs_voices import (
    DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
    SYSTEM_DISPLAY_COLOR,
    build_omnivoice_voice_prompt_block,
    color_markup,
    read_cached_character_voice,
    strip_color_markup,
)

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("google-genai is not installed. Run: .venv/bin/pip install google-genai")

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea
except ImportError:
    Application = None
    Point = None
    Document = None
    KeyBindings = None
    HSplit = None
    Layout = None
    Window = None
    FormattedTextControl = None
    TextArea = None

# ── Configuration ────────────────────────────────────────────────────────────

MODEL = "gemini-flash-latest"          # resolves to gemini-3.5-flash (June 2026)
THINKING_LEVEL = "high"
FAST_SERVICE_TIER = types.ServiceTier.PRIORITY
FAST_SERVICE_TIER_LABEL = FAST_SERVICE_TIER.value
COMPACT_MARGIN_TOKENS = 150_000        # compact this many tokens past system prompt + transcript
COMPACT_RETAIN_TURNS = 4               # keep this many recent player turns verbatim, outside the summary
CHARS_PER_TOKEN = 4.0                  # fallback estimate when usage is missing
OUTPUT_INDENT = "  "
DEFAULT_GAME: str | None = None
WHISPER_PROPER_NOUN_PROMPT_MAX_CHARS = 900
WHISPER_PROPER_NOUN_PROMPT_MAX_TERMS = 80

# Bounded retry with exponential backoff for transient Gemini errors (rate
# limits / 5xx / connection-timeout). Backoff is deterministic — no random,
# time-seeded jitter that would break the deterministic smoke tests.
MAX_GEMINI_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SECONDS = 1.0

# USD per 1M tokens for gemini-flash-latest (gemini-3.5-flash, paid tier,
# per ai.google.dev/gemini-api/docs/pricing as of June 2026). Thinking tokens
# are billed at the output rate; implicit-cache hits at the cached-input rate.
PRICE_IN_PER_M = 1.50
PRICE_IN_CACHED_PER_M = 0.15
PRICE_OUT_PER_M = 9.00
ELEVENLABS_VOICE_DESIGN_PRICE_PER_1K_CHARS_DEFAULT = 0.10
ELEVENLABS_VOICE_DESIGN_PRICE_ENV = "IF_ENGINE_ELEVENLABS_VOICE_DESIGN_PRICE_PER_1K_CHARS"

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "catalog" / "clubfloyd.json"
SESSIONS_DIR = BASE_DIR / "sessions"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
WHISPER_PROMPT_CACHE_DIR = BASE_DIR / ".cache" / "whisper-prompts"

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(BASE_DIR / ".env")
DEFAULT_FAST_MODE = os.environ.get("IF_ENGINE_FAST_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def require_gemini_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Create .env with GEMINI_API_KEY=... or export it.")
    return api_key

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
The user wants to play {title}, but they want it to allow them to do \
anything. Have the story redirect the user to important next steps if they \
wander too much. When the player does something unexpected, prefer a \
"yes, and..." response that makes it matter in-world, unless allowing it \
would prevent future game progression. Use the original transcript as reference. Do not change any \
major plot points, but allow embellishment that doesn't conflict with the \
future story or cut off whole sections of the story. When the player goes \
beyond what the transcript covers — exploring other rooms, examining objects, \
or trying alternate solutions and interactions — also draw on your own \
knowledge of {title}, which was published and widely discussed well before \
your training cutoff: prefer how the real game would actually respond (its \
true geography, items, puzzles, characters, and alternate paths) over free \
invention, so off-path play stays faithful to the original work. Where your \
recollection conflicts with the transcript, the transcript wins for plot, \
canon, and pacing; still never reveal or shortcut future plot points or puzzle \
solutions before their time; and when you are genuinely unsure of a detail, \
fall back to concise invention that matches the established tone. Focus on what is \
important now, and do not repeat story details just because they are timely \
or nearby in the transcript. Keep invented material \
concise: when you embellish beyond the original transcript, add brief, vivid \
touches — a few sentences, not paragraphs — and never pad a scene; the \
player's time is precious; invented narration and dialogue must match the \
tone and style of the transcript writing. Reserve fuller pages for major moments drawn from \
the original story. Characters can \
and should have dialogue that was not in the original transcript, as long as \
they do not reveal major plot points before it is time; very subtle \
foreshadowing is allowed. Anything the user says is a dialog line to a \
character, an action, an internal thought, or some combination of those. Your \
story is told in the second person, and the user must control a character \
inside the game or movie; start with the obvious/correct main character. Your \
output should be concise yet descriptive interactive fiction text reacting \
to what the user has done — usually a paragraph or two, at most a short page \
for major story moments. Markdown italicization and bolding are allowed for \
emphasis. You do not need to repeat the player's action or \
dialogue in narration; when possible, start with the effect, reaction, \
answer, or impact so the turn reaches its point quickly. The play through \
will be long, take \
your time - exploration is allowed and a large part of the fun. In long \
conversations, stop at natural beats so the player has chances to interject \
before the exchange moves on. Before leaving a location or sub-location, give \
the player room to notice, examine, or interact with what is there; before a \
character leaves the scene, leave space for the player to speak to or interact \
with them. These three pacing rules are important and override the transcript's \
turn boundaries: even if the transcript immediately continues a conversation, \
moves away, or has a character leave, pause the assistant message early enough \
for the player to interject, explore, or interact first. Do not ever \
reference cardinal directions (North, South, East, or West), as you can fully \
describe each way and that knowledge would require a compass. Never announce, \
title, or restate a location's name when the player moves; the prose itself \
conveys where they are. Instead, mark every location change with the single \
tag <location/>, inserted at the exact moment the location changes — even \
mid-sentence: entering a different room, crossing to the other side of a \
location, or arriving somewhere new or notable. A good rule of thumb: would \
this location sound different to the character? If so, insert <location/>. \
Never output chapter names, chapter or section titles, or chapter epigraph \
quotes — not in the opening page and not at chapter transitions; the story \
flows continuously without headings. \
Use \
<hidden></hidden> around any text that is hidden from the user but important \
to the future. No thinking/planing should happen outside of <hidden></hidden> \
tags. When the page newly reveals meaningful story-progress information to \
the player, include a separate self-closing <progress/> tag outside any \
hidden span. The progress tag is never displayed; it is for engine tracking \
only. Use it only to mark newly revealed canonical progress: new clues, \
objectives, puzzle state, inventory or access changes, character revelations, \
time/story advancement, or irreversible events. Do not use it for ordinary \
exploration, atmosphere, made-up filler, incidental reactions or dialogue, or \
restating information the player already knew. If no real story progress was \
newly revealed, omit the progress tag. If the user asks to go to a specific place, treat that as valid \
movement: summarize the \
journey there when the path is uneventful, but stop early if the route \
reveals someone, something interesting, or something the user should do, \
experience, or interact with. Do not talk directly to the user outside the \
fiction, and do not end pages with direct prompts or questions such as \
"What would you like to do?" Your first message of a new playthrough — \
before the player has acted — must be the game's opening page, reproduced \
from the transcript: the opening narration, scene-setting, initial location \
description, and any opening dialog, staying faithful to the original text. \
Leave out everything that is not story: title screens and ASCII art, author \
and copyright lines, release/serial/version banners, interpreter or engine \
messages, help/about instructions, content warnings, tutorial text, chapter \
names and epigraph quotes, and "[Press any key]"-style prompts. End that first page where the original \
game first hands control to the player. If the player interrupts the opening \
and asks or implies that they want to start the interactive part, skip ahead, \
fast-forward, or begin later, you may skip the rest of the opening and \
continue from the requested starting point while preserving necessary context.

--- Engine notes (mechanics, not story) ---
- The first user message contains the complete original game transcript
  between BEGIN/END markers. It is private reference material: never echo it,
  quote it wholesale, or mention that a transcript exists.
- A user message starting with [SESSION MEMORY] is a summary that has replaced
  older conversation history. Treat it as established canon for this
  playthrough and continue seamlessly from the situation it describes.
- A user message starting with [INTERRUPTION] means the player interrupted
  your narration; the bracketed note states exactly how much of your previous
  message reached them (it may have cut off mid-stream, or been read aloud
  only partway). What follows is their clarification or redirection. Pick up
  from that exact moment without repeating narration the player already
  received, and treat anything beyond that point as never having reached them.
  If you include hidden text after an interruption, every hidden span must have
  both an opening <hidden> tag and a closing </hidden> tag in the same response.
- Every other user message is the player's dialog line and/or action.
"""

VOICE_SYSTEM_PROMPT_ADDENDUM = """\

--- Voice mode (mechanics, not story) ---
The player is playing by voice: their messages are automatic speech-to-text \
transcriptions, so expect mis-transcriptions — homophones, dropped or merged \
words, misspellings, odd punctuation, and stray filler words. Do your best \
to accommodate them: infer the intended action or dialogue from context, \
favoring phonetically similar readings, and do not correct the player's \
spelling in narration or dialogue. Treat garbled fragments charitably rather \
than literally. Only if an input is truly uninterpretable should the story \
gently reflect the confusion in-world. Never mention transcription, speech \
recognition, or these errors to the player. Your narration is read aloud by \
a voice engine that renders em dashes and ellipses as real spoken pauses: \
use an em dash (—) for emphatic mid-sentence breaks and "..." for trailing \
hesitation, and avoid hyphens or en dashes as punctuation. Write for the \
ear: no abbreviations unless they are meant to be read as one, and spell \
initialisms with periods or spaces so they are read letter by letter ("F.B.I." \
or "F B I", not "FBI"; write "Doctor" not "Dr.", "Street" not "St.", "for example" not "e.g."), \
and spell out numbers exactly as they should be spoken ("eighteen \
ninety-three" not "1893", "four dollars and fifty cents" not "$4.50", \
"zero point zero one two eight" not "0.0128").\
"""

OMNIVOICE_SYSTEM_PROMPT_ADDENDUM = """\

When voice narration is enabled through OmniVoice, bracketed cues [laughter] \
and [sigh] are supported in narration and may be used sparingly when they \
fit character dialogue or the tone of the moment, but only place them inside \
dialogue quotes, not in narration. These are audio cues only; they are read \
aloud but never displayed as literal text.\
"""

OMNIVOICE_WHISPER_SYSTEM_PROMPT_ADDENDUM = """\

OmniVoice whisper tags are enabled. For words that should be spoken quietly \
or secretively, wrap only that visible narration or dialogue in paired \
<whisper>...</whisper> tags. Whisper tags may cover part of a \
sentence or multiple sentences, but whisper delivery applies only until the \
closing </whisper> tag. These tags are for OmniVoice rendering only and \
are never displayed; do not use them for hidden information, which still \
belongs inside <hidden>...</hidden>.\
"""

RUNTIME_CONFIG_PROMPT_TEMPLATE = """\

--- Runtime configuration (mechanics, not story) ---
This section reflects the current launch configuration and replaces any prior \
runtime configuration. Voice mode: {voice_mode}. TTS engine: {tts_engine}. \
Latency tier: {latency_tier}. Wake word: {wake_word}. Car mode: {car_mode}. \
Wake threshold: {wake_threshold}. Wake preprocessing: {wake_preprocess}. \
Whisper tags: {whisper_tags}. Character voices: {character_voices}.\
"""

TRANSCRIPT_PREAMBLE_TEMPLATE = """\
Below is the complete transcript of an original playthrough of {title}. It \
is your canonical reference for the world: plot, characters, locations, \
items, puzzle logic, and tone. Never echo or quote it wholesale, and never \
mention it to the player.

=== BEGIN ORIGINAL TRANSCRIPT ===
"""

TRANSCRIPT_EPILOGUE = """
=== END ORIGINAL TRANSCRIPT ===

Begin the playthrough now: produce the game's opening page from this \
transcript — only the actual story content, per the system instructions — \
and stop where the original game first hands control to the player.\
"""

# A direction word followed by a capitalized word is a proper-noun compound
# ("West London Cricketeers", "East End types"), not compass usage.
CARDINAL_DIRECTION_RE = re.compile(
    r"\b(?:north|south|east|west)\b(?!\s+(?-i:[A-Z]))", re.IGNORECASE
)
LAST_LINE_QUESTION_RE = re.compile(r"\?\s*[\"')\]}]*\s*$")
# Engine preambles prepended to player text (e.g. [INTERRUPTION — …:]) end
# with ":]" at a line break; strip them when replaying saved sessions.
LEADING_ENGINE_NOTE_RE = re.compile(r"^\[.*?:\]\s*\n", re.DOTALL)
SESSION_INSTRUCTION_PREFIXES = (
    "Type at any time",
    "Type at the prompt;",
    "No voice:",
    "Voice input:",
    "Caps Lock",
)


def is_session_instruction_line(line: str) -> bool:
    return line.startswith(SESSION_INSTRUCTION_PREFIXES)

CARDINAL_DIRECTION_REMINDER = """\
[ENGINE REMINDER — the previous visible narration used a cardinal direction.]
Follow the system prompt rule: do not ever reference cardinal directions \
(North, South, East, or West). Describe paths, exits, and relative locations \
without compass terms.\
"""

QUESTION_ENDING_REMINDER = """\
[ENGINE REMINDER — the previous visible narration ended with a question.]
Follow the system prompt rule: do not talk directly to the user or end pages \
with direct prompts/questions such as "What would you like to do?" Keep the \
narration in-world and let the player's next input decide the action.\
"""

EMPTY_REPLY_REMINDER = """\
[ENGINE REMINDER — your reply contained no visible narration, so nothing \
reached the player.]
Respond to the player's last message now with in-world story text, following \
the system prompt rules. Never reply with an empty message or with only \
<hidden> content.\
"""

STORY_PROGRESS_REMINDER = """\
[ENGINE REMINDER — the story has not progressed in a couple of pages.]
You should consider helping the player move closer to story progression \
through narration, an event, a clue, a consequence, an obstacle, or character \
dialogue. Do not force a major reveal, but give the scene a meaningful path \
toward the next important development when it fits.\
"""

COMPACTION_PROMPT = """\
[ENGINE: COMPACTION REQUEST — this is not a player action. Pause the story.]

The conversation history is about to be replaced by your answer — it becomes \
your working memory of everything you are summarizing, so whatever you leave \
out is lost for good. Produce a thorough, structured breakdown organized by \
storyline; for each, cover the who, what, when, where, and why. Prefer \
concrete, specific facts — exact names, objects, and the actual wording of \
anything promised or agreed — over vague summary.

Capture all of the following explicitly; do not abbreviate them away:
- The current scene: exact situation, location, day/time, and weather.
- The player character's inventory, condition, and immediate goals.
- Item and world state — for every notable object, where it is right now and
  how it got there: what is being carried, and what was dropped, left behind,
  hidden, stored, handed off, used up, or broken, and exactly where. Likewise
  every lasting change to the world: doors unlocked, things moved or destroyed,
  mechanisms triggered.
- Every notable character: who they are, where they are, what they know, and
  whether they know the player knows it.
- How characters feel — each character's attitude toward the player and toward
  the other characters: trust or suspicion, warmth, hostility, fear, affection,
  debts, and grudges; how those feelings have shifted over the playthrough and
  why; and how the last encounter with each one ended.
- Every promise, deal, threat, demand, or debt — made by the player or by a
  character — who it binds, the exact terms, any condition or deadline, and
  whether it is still unfulfilled.
- Every location discovered and how they connect (no cardinal directions).
- All secrets, foreshadowing, and plans you previously wrote inside <hidden>
  tags — restate them inside <hidden> tags here so they stay hidden.
- Puzzles or obstacles: solved and unsolved, and what is known about each.
- Unresolved threads and the next important story step to steer toward.

Write a dense reference document. Do not address the player. Do not continue \
the story.\
"""

SUMMARY_WRAPPER = """\
[SESSION MEMORY — earlier conversation was compacted away. This is the \
canonical summary of everything that has happened so far in this playthrough:]

{summary}

[End of session memory. Continue the story seamlessly from the current scene \
when the player next acts.]\
"""

INTERRUPTION_PREAMBLE = (
    "[INTERRUPTION — the player interrupted your narration at the exact "
    "point where your previous message cuts off, to clarify or redirect. "
    "If you include hidden text, every hidden span must have both an opening "
    "<hidden> tag and a closing </hidden> tag in the same response:]\n"
)

VOICE_INTERRUPTION_HEARD = """\
[INTERRUPTION — the player spoke over your narration while it was being read \
aloud. They heard it only up to approximately this point:
{heard}
Do not assume anything after that point reached the player. What follows is \
their spoken clarification or redirection from that exact moment. If you \
include hidden text, every hidden span must have both an opening <hidden> tag \
and a closing </hidden> tag in the same response:]
"""

VOICE_INTERRUPTION_UNHEARD = """\
[INTERRUPTION — the player spoke over your narration just as it began to be \
read aloud; essentially none of it reached them. Do not assume the player \
knows anything from it. What follows is their spoken clarification or \
redirection. If you include hidden text, every hidden span must have both an \
opening <hidden> tag and a closing </hidden> tag in the same response:]
"""


HEARD_TAIL_MAX_WORDS = 12


def voice_interruption_preamble(heard_text: str) -> str:
    if heard_text:
        # The model knows its own narration; the tail is enough to locate the
        # cut-off point without echoing the whole passage back.
        words = heard_text.split()
        if len(words) > HEARD_TAIL_MAX_WORDS:
            heard_text = "... " + " ".join(words[-HEARD_TAIL_MAX_WORDS:])
        return VOICE_INTERRUPTION_HEARD.format(
            heard=json.dumps(heard_text, ensure_ascii=False)
        )
    return VOICE_INTERRUPTION_UNHEARD

# ── Terminal helpers ─────────────────────────────────────────────────────────

IS_TTY = sys.stdout.isatty() and sys.stdin.isatty()


def style(code: str, text: str) -> str:
    # Only emit ANSI when attached to a TTY and color is not opted out, so
    # piped/non-TTY output (e.g. the smoke tests) stays plain text.
    if not IS_TTY or "NO_COLOR" in os.environ:
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    return style("2", text)


def cyan(text: str) -> str:
    return style("36", text)


def notice(text: str) -> None:
    print(dim(text), flush=True)


def emit_notice(text: str, emit=None) -> None:
    if emit is None:
        notice(text)
    else:
        emit(system_markup(text) + "\n")


# ── Transient-error retry ────────────────────────────────────────────────────

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_STATUS_NAMES = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}
_TRANSIENT_ERROR_HINTS = (
    "resourceexhausted",
    "servererror",
    "serviceunavailable",
    "unavailable",
    "deadlineexceeded",
    "internalservererror",
    "timeout",
    "timedout",
    "connection",
    "connecterror",
    "overloaded",
)


def is_transient_gemini_error(exc: BaseException) -> bool:
    """Best-effort classifier for retryable Gemini/transport errors.

    Matches defensively on the HTTP status code, the API status string, the
    exception class name, or message keywords, so it works without importing a
    specific google.genai error type.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _TRANSIENT_STATUS_CODES:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.strip().upper() in _TRANSIENT_STATUS_NAMES:
        return True
    name = type(exc).__name__.lower()
    if any(hint in name for hint in _TRANSIENT_ERROR_HINTS):
        return True
    text = str(exc).lower()
    return any(hint in text for hint in (
        "resource_exhausted",
        "resource exhausted",
        "unavailable",
        "deadline",
        "timed out",
        "timeout",
        "connection",
        "overloaded",
        "try again",
    ))


def should_retry_gemini(exc: BaseException, attempts_done: int, emit=None) -> bool:
    """If ``exc`` is transient and attempts remain, emit a notice, sleep with
    deterministic exponential backoff, and return True; otherwise return False."""
    if attempts_done >= MAX_GEMINI_ATTEMPTS or not is_transient_gemini_error(exc):
        return False
    emit_notice(
        f"[model busy, retrying… ({attempts_done + 1}/{MAX_GEMINI_ATTEMPTS})]",
        emit=emit,
    )
    time.sleep(RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempts_done - 1)))
    return True


def session_header(session_name: str, game_title: str, fast_mode: bool = False) -> str:
    tier = "low latency" if fast_mode else "standard latency"
    return f"{game_title} • {session_name} • {MODEL} ({THINKING_LEVEL} effort, {tier})"


# ── Transcript loading ───────────────────────────────────────────────────────

def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def available_games() -> list[str]:
    games = set(load_game_catalog())
    if TRANSCRIPTS_DIR.exists():
        games.update(p.stem for p in TRANSCRIPTS_DIR.glob("*.txt") if p.is_file())
    return sorted(games)


def load_game_catalog() -> dict:
    if CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def load_local_transcript_index() -> dict:
    path = TRANSCRIPTS_DIR / "index.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def load_game_index() -> dict:
    index = load_game_catalog()
    local_index = load_local_transcript_index()
    if TRANSCRIPTS_DIR.exists():
        for path in TRANSCRIPTS_DIR.glob("*.txt"):
            if path.stem not in index:
                index[path.stem] = local_index.get(path.stem) or {
                    "title": path.stem.replace("-", " ").title(),
                    "author": "",
                    "source": "local user-provided transcript",
                }
    return index


def game_title_for(game: str) -> str:
    info = load_game_index().get(game) or {}
    return info.get("title") or game.replace("-", " ").title()


def load_transcript_text(game: str) -> str:
    path = TRANSCRIPTS_DIR / f"{game}.txt"
    catalog = load_game_catalog()
    if path.exists() and path.stat().st_size > 0:
        return normalize_line_endings(path.read_text(encoding="utf-8"))
    if not path.exists() or path.stat().st_size == 0:
        if game in catalog and os.environ.get("IF_ENGINE_SKIP_TRANSCRIPT_DOWNLOADS") != "1":
            info = catalog[game]
            title = info.get("title") or game.replace("-", " ").title()
            print(f"[transcripts] downloading {title} from ClubFloyd...", file=sys.stderr)
            try:
                from build_transcripts import download_game
                path = download_game(game, catalog=catalog)
            except Exception as exc:
                raise RuntimeError(f"could not download transcript for '{game}': {exc}") from exc
        if not path.exists():
            games = available_games()
            hint = ", ".join(games[:15]) + ("…" if len(games) > 15 else "")
            raise RuntimeError(
                f"no transcript for game '{game}' in {TRANSCRIPTS_DIR}/. "
                f"Run build_transcripts.py --game {game}, add a local "
                f"{TRANSCRIPTS_DIR / (game + '.txt')} file, or pick one of: "
                f"{hint or '(none catalogued yet)'}"
            )
    return normalize_line_endings(path.read_text(encoding="utf-8"))


PROPER_NOUN_TOKEN_RE = (
    r"(?:[A-Z]\.){2,}|[A-Z][A-Za-z]+(?:[-'’][A-Za-z]+)*|[A-Z]{2,}(?:[-'’][A-Z]+)*"
)
PROPER_NOUN_PHRASE_RE = re.compile(
    rf"\b{PROPER_NOUN_TOKEN_RE}(?:[ \t]+(?:&[ \t]+)?{PROPER_NOUN_TOKEN_RE})*"
)
PROPER_NOUN_WORD_RE = re.compile(PROPER_NOUN_TOKEN_RE)
PROPER_NOUN_STOPWORDS = {
    "a", "about", "after", "again", "all", "along", "also", "an", "and",
    "another", "any", "are", "around", "as", "at", "away", "back", "before",
    "behind", "below", "beside", "between", "but", "by", "can", "could",
    "did", "ding", "down", "each", "even", "every", "for", "from", "good", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "inside", "into", "is", "it", "its", "just", "near", "nearby", "no",
    "none", "not", "now", "of", "off", "ok", "okay", "old", "on", "once",
    "one", "onto", "or", "out", "over", "please", "she", "so", "some",
    "something", "still", "such", "suddenly", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "taken", "to", "toward", "under", "up", "upon", "was", "well", "were", "what",
    "whatever", "when", "where", "which", "while", "with", "would", "you",
    "your", "yours",
}
PROPER_NOUN_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}


def _is_sentence_start(text: str, index: int) -> bool:
    i = index - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    while i >= 0 and text[i] in "\"'“”‘’([{":
        i -= 1
        while i >= 0 and text[i].isspace():
            i -= 1
    return i < 0 or text[i] in ".!?\n>"


def _normalize_proper_noun(raw: str) -> str:
    term = re.sub(r"\s+", " ", raw).strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    term = re.sub(r"\s*&\s*", " & ", term)
    if re.search(r"(?i)[’'](?:re|ve|ll|d|m|t)\b", term):
        return ""
    term = re.sub(r"(?i)[’']s\b", "", term)
    words = term.split()
    while words and (words[0] == "&" or words[0].casefold() in PROPER_NOUN_STOPWORDS):
        words.pop(0)
    while words and (words[-1] == "&" or words[-1].casefold() in PROPER_NOUN_STOPWORDS):
        words.pop()
    return " ".join(words)


def _rank_proper_noun_counts(
    counts: Counter[str],
    display: dict[str, str],
    first_seen: dict[str, int],
) -> list[str]:
    scored = [(-count, first_seen[key], display[key]) for key, count in counts.items()]
    scored.sort()
    return [term for _neg_count, _first, term in scored]


def _spacy_proper_nouns(text: str) -> list[str]:
    try:
        import spacy
    except Exception:
        return []
    try:
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
        )
    except Exception:
        return []
    nlp.max_length = max(nlp.max_length, len(text) + 100)
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    labels = {"PERSON", "ORG", "GPE", "LOC", "FAC", "WORK_OF_ART", "PRODUCT", "EVENT"}
    for ent in nlp(text).ents:
        if ent.label_ not in labels or _is_sentence_start(text, ent.start_char):
            continue
        term = _normalize_proper_noun(ent.text)
        if not term or not term[0].isupper():
            continue
        words = PROPER_NOUN_WORD_RE.findall(term)
        if not words:
            continue
        if len(words) == 1:
            word = words[0]
            folded = word.casefold().rstrip(".")
            if (
                folded in PROPER_NOUN_STOPWORDS
                or folded in PROPER_NOUN_MONTHS
                or (word.isupper() and len(word) > 1)
            ):
                continue
        key = term.casefold()
        counts[key] += 1
        display.setdefault(key, term)
        first_seen.setdefault(key, ent.start_char)
    return _rank_proper_noun_counts(counts, display, first_seen)


def _regex_proper_nouns(text: str) -> list[str]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    first_seen: dict[str, int] = {}

    for match in PROPER_NOUN_PHRASE_RE.finditer(text):
        if _is_sentence_start(text, match.start()):
            continue
        term = _normalize_proper_noun(match.group(0))
        if not term:
            continue
        words = PROPER_NOUN_WORD_RE.findall(term)
        if not words:
            continue
        folded_words = [word.casefold().rstrip(".") for word in words]
        if all(word in PROPER_NOUN_STOPWORDS for word in folded_words):
            continue
        if len(words) == 1:
            word = folded_words[0]
            if word in PROPER_NOUN_STOPWORDS or word in PROPER_NOUN_MONTHS:
                continue

        key = term.casefold()
        counts[key] += 1
        display.setdefault(key, term)
        first_seen.setdefault(key, match.start())

    filtered_counts: Counter[str] = Counter()
    filtered_display: dict[str, str] = {}
    filtered_first_seen: dict[str, int] = {}
    for key, count in counts.items():
        term = display[key]
        words = PROPER_NOUN_WORD_RE.findall(term)
        if len(words) == 1 and words[0].isupper() and len(words[0]) > 1:
            continue
        filtered_counts[key] = count
        filtered_display[key] = term
        filtered_first_seen[key] = first_seen[key]
    return _rank_proper_noun_counts(filtered_counts, filtered_display, filtered_first_seen)


def transcript_proper_nouns(text: str) -> list[str]:
    return _spacy_proper_nouns(text) or _regex_proper_nouns(text)


def build_whisper_proper_noun_prompt(
    transcript_text: str,
    game_title: str,
    max_chars: int = WHISPER_PROPER_NOUN_PROMPT_MAX_CHARS,
    max_terms: int = WHISPER_PROPER_NOUN_PROMPT_MAX_TERMS,
) -> str:
    cache_key = hashlib.sha256(
        f"{game_title}\0{max_chars}\0{max_terms}\0{transcript_text}".encode("utf-8")
    ).hexdigest()
    cache_path = WHISPER_PROMPT_CACHE_DIR / f"{cache_key}.txt"
    try:
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
    except OSError:
        pass

    terms = transcript_proper_nouns(game_title + "\n" + transcript_text)
    if not terms:
        return ""
    prefix = "Possible spoken proper nouns in this game: "
    selected: list[str] = []
    current_len = len(prefix)
    for term in terms:
        addition = len(term) + (2 if selected else 0)
        if len(selected) >= max_terms or current_len + addition > max_chars:
            break
        selected.append(term)
        current_len += addition
    prompt = prefix + ", ".join(selected) + "."
    try:
        WHISPER_PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(prompt, encoding="utf-8")
    except OSError:
        pass
    return prompt


# ── Streaming non-display tag filters ────────────────────────────────────────

OPEN_TAG = "<hidden>"
CLOSE_TAG = "</hidden>"
PROGRESS_MARKER_TAGS = ("<progress/>", "<progress />")
PROGRESS_MARKER_TAG_RE = re.compile(r"<\s*progress\s*/\s*>", re.IGNORECASE)
# Permissive partial prefix of a progress marker, anchored at end of string, so
# the streaming strip can hold back a marker split across a chunk boundary using
# the same spacing tolerance as PROGRESS_MARKER_TAG_RE (strip and detect must
# never disagree, e.g. a "< progress />" variant must not leak).
PROGRESS_MARKER_TAG_PARTIAL_RE = re.compile(
    r"<\s*(?:p(?:r(?:o(?:g(?:r(?:e(?:s(?:s\s*(?:/\s*)?)?)?)?)?)?)?)?)?$",
    re.IGNORECASE,
)
WHISPER_TAG_RE = re.compile(r"</?\s*whisper\s*>", re.IGNORECASE)
VOICE_TAG_RE = re.compile(
    r"<\s*/?\s*voice\b[^>]*>",
    re.IGNORECASE,
)
AUDIO_CUE_TAG_RE = re.compile(
    r"\[(?:sigh|laughter|pause\s+\d+(?:\.\d+)?\s*(?:ms|s)?)\]",
    re.IGNORECASE,
)

# ── Display colors ───────────────────────────────────────────────────────────
# System messages and the input chevron render in SYSTEM_DISPLAY_COLOR blue;
# each character's dialogue renders in the color chosen at voice design (see
# elevenlabs_voices.pick_display_color). Narration and player input stay
# uncolored. Colored spans travel as inline ⟦fg=#rrggbb⟧…⟦/fg⟧ markup that the
# terminal UI turns into styled fragments and every plain-text path strips.

QUOTE_CHARS = "\"'“”‘’«»"
VOICE_OPEN_TAG_RE = re.compile(r"<\s*voice\b[^>]*>", re.IGNORECASE)
VOICE_NAME_ATTR_RE = re.compile(
    r"\bname\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.IGNORECASE
)
VOICE_SPAN_RE = re.compile(
    rf"([{QUOTE_CHARS}]?)(<\s*voice\b[^>]*>)(.*?)(<\s*/\s*voice\s*>)([{QUOTE_CHARS}]?)",
    re.IGNORECASE | re.DOTALL,
)


def system_markup(text: str) -> str:
    return color_markup(text, SYSTEM_DISPLAY_COLOR) if text.strip() else text


def elevenlabs_voice_cache_root() -> Path:
    override = os.environ.get("IF_ENGINE_ELEVENLABS_VOICE_CACHE_DIR", "").strip()
    return Path(override).expanduser() if override else DEFAULT_ELEVENLABS_VOICE_CACHE_DIR


def _voice_span_replacement(match: "re.Match[str]", color_resolver) -> str:
    quote_before, open_tag, inner, _close_tag, quote_after = match.groups()
    name_match = VOICE_NAME_ATTR_RE.search(open_tag)
    name = ""
    if name_match is not None:
        name = (name_match.group(1) or name_match.group(2) or "").strip()
    body = f"{quote_before}{inner}{quote_after}"
    color = color_resolver(name) if (color_resolver is not None and name) else None
    return color_markup(body, color)


def colorize_voice_spans(text: str, color_resolver) -> str:
    """Replace completed <voice name="X">…</voice> spans with color markup in
    that character's voice-design color, pulling the quotation marks that
    directly wrap the span into the colored region. Spans whose character has
    no cached color simply lose their tags (the previous behavior)."""
    return VOICE_SPAN_RE.sub(
        lambda match: _voice_span_replacement(match, color_resolver), text
    )


class EmDashSpacingFilter:
    """Normalize model-stream em dashes across chunk boundaries."""

    def __init__(self) -> None:
        self.last_emitted = ""
        self.after_dash = False

    def feed(self, text: str) -> str:
        out: list[str] = []
        for char in text:
            if char == "—":
                if self.last_emitted and not self.last_emitted.isspace():
                    out.append(" ")
                    self.last_emitted = " "
                out.append(char)
                self.last_emitted = char
                self.after_dash = True
                continue

            if self.after_dash:
                if char in " \t":
                    continue
                if not char.isspace():
                    out.append(" ")
                    self.last_emitted = " "
                self.after_dash = False

            out.append(char)
            self.last_emitted = char
        return "".join(out)

    def flush(self) -> str:
        if not self.after_dash:
            return ""
        self.after_dash = False
        if self.last_emitted == "—":
            self.last_emitted = " "
            return " "
        return ""


class HiddenStreamFilter:
    """Incrementally strips <hidden>...</hidden> spans from streamed text,
    holding back partial tags that straddle chunk boundaries."""

    def __init__(self):
        self.hidden = False
        self.pending = ""

    def feed(self, text: str) -> str:
        self.pending += text
        out = []
        while True:
            tag = CLOSE_TAG if self.hidden else OPEN_TAG
            i = self.pending.find(tag)
            if i < 0:
                break
            if not self.hidden:
                out.append(self.pending[:i])
            self.pending = self.pending[i + len(tag):]
            self.hidden = not self.hidden
        # Hold back any trailing partial tag; emit (or drop, if hidden) the rest.
        tag = CLOSE_TAG if self.hidden else OPEN_TAG
        hold = 0
        for k in range(min(len(tag) - 1, len(self.pending)), 0, -1):
            if self.pending.endswith(tag[:k]):
                hold = k
                break
        emit = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[len(self.pending) - hold:] if hold else ""
        if not self.hidden:
            out.append(emit)
        return "".join(out)

    def flush(self) -> str:
        emit = "" if self.hidden else self.pending
        self.pending = ""
        return emit


def strip_hidden(text: str) -> str:
    f = HiddenStreamFilter()
    return f.feed(text) + f.flush()


class ProgressStreamFilter:
    """Incrementally strips <progress/> markers from streamed text."""

    def __init__(self):
        self.pending = ""

    def feed(self, text: str) -> str:
        self.pending += text
        out = []
        while True:
            m = PROGRESS_MARKER_TAG_RE.search(self.pending)
            if not m:
                break
            out.append(self.pending[:m.start()])
            self.pending = self.pending[m.end():]

        # Hold back a marker that straddles the chunk boundary, using the same
        # permissive pattern as detection so strip and detect cannot disagree.
        partial = PROGRESS_MARKER_TAG_PARTIAL_RE.search(self.pending)
        hold = len(self.pending) - partial.start() if partial else 0
        emit = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[len(self.pending) - hold:] if hold else ""
        out.append(emit)
        return "".join(out)

    def flush(self) -> str:
        emit = self.pending
        self.pending = ""
        return emit


def strip_progress(text: str) -> str:
    f = ProgressStreamFilter()
    return f.feed(text) + f.flush()


def strip_whisper_tags(text: str) -> str:
    return WHISPER_TAG_RE.sub("", text)


def strip_voice_tags(text: str) -> str:
    return VOICE_TAG_RE.sub("", text)


def strip_audio_cue_tags(text: str) -> str:
    text = AUDIO_CUE_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return "\n".join(line.strip(" \t") for line in text.split("\n"))


def strip_rendering_tags_for_display(text: str) -> str:
    return strip_audio_cue_tags(strip_voice_tags(strip_whisper_tags(text)))


def strip_rendering_tags_for_stream_display(text: str) -> str:
    text = strip_whisper_tags(text)
    text = strip_voice_tags(text)
    text = AUDIO_CUE_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text


def _possible_rendering_tag_prefix(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if lower.startswith("<") and ">" not in lower:
        body = lower[1:].lstrip()
        if body.startswith("/"):
            body = body[1:].lstrip()
        body = body.strip()
        return (
            not body
            or "whisper".startswith(body)
            or "voice".startswith(body)
            or body.startswith("voice")
        )
    if lower.startswith("[") and "]" not in lower:
        body = lower[1:].strip()
        return (
            not body
            or "sigh".startswith(body)
            or "laughter".startswith(body)
            or "pause".startswith(body)
            or re.fullmatch(r"pause\s+\d*(?:\.\d*)?\s*(?:m?s?)?", body) is not None
        )
    return False


def _rendering_tag_suffix_start(text: str) -> int | None:
    start = max(0, len(text) - 64)
    for i in range(start, len(text)):
        if text[i] in "<[" and _possible_rendering_tag_prefix(text[i:]):
            return i
    return None


class RenderingTagStreamFilter:
    """Incrementally strip display-only rendering tags from streamed text.

    With a `color_resolver`, completed <voice> spans (plus the quotation marks
    directly around them) are converted to that character's color markup
    instead of just losing their tags; an opened-but-unclosed span is held
    back until its closing tag arrives so the whole span can be colored."""

    def __init__(self, color_resolver=None) -> None:
        self.pending = ""
        self.last_emitted = ""
        self.color_resolver = color_resolver

    def _emit(self, text: str) -> str:
        if self.last_emitted and self.last_emitted.isspace():
            text = text.lstrip(" \t")
        if text:
            self.last_emitted = text[-1]
        return text

    def _voice_span_hold_start(self, text: str) -> int | None:
        m = VOICE_OPEN_TAG_RE.search(text)
        if m is None:
            return None
        start = m.start()
        if start > 0 and text[start - 1] in QUOTE_CHARS:
            start -= 1
        return start

    def _colorize_completed_spans(self, text: str) -> str:
        """Colorize voice spans, but hold back one that ends exactly at the
        end of the buffer with no trailing quote yet — the closing quotation
        mark may still be streaming in and belongs inside the color."""
        out: list[str] = []
        pos = 0
        for match in VOICE_SPAN_RE.finditer(text):
            if match.end() == len(text) and not match.group(5):
                break
            out.append(text[pos:match.start()])
            out.append(_voice_span_replacement(match, self.color_resolver))
            pos = match.end()
        return "".join(out) + text[pos:]

    def feed(self, text: str) -> str:
        self.pending += text
        hold = None
        if self.color_resolver is not None:
            self.pending = self._colorize_completed_spans(self.pending)
            hold = self._voice_span_hold_start(self.pending)
        head = self.pending if hold is None else self.pending[:hold]
        tail = "" if hold is None else self.pending[hold:]
        head = strip_rendering_tags_for_stream_display(head)
        start = _rendering_tag_suffix_start(head)
        if start is not None:
            tail = head[start:] + tail
            head = head[:start]
        if (
            self.color_resolver is not None
            and head
            and head[-1] in QUOTE_CHARS
            and (not tail or tail.startswith("<"))
        ):
            # The quote may open a voice span still streaming in; hold it so
            # it can be colored with the span.
            tail = head[-1] + tail
            head = head[:-1]
        self.pending = tail
        return self._emit(head)

    def flush(self) -> str:
        text = self.pending
        if self.color_resolver is not None:
            text = colorize_voice_spans(text, self.color_resolver)
        text = strip_rendering_tags_for_stream_display(text)
        start = _rendering_tag_suffix_start(text)
        if start is not None:
            text = text[:start]
        self.pending = ""
        return self._emit(text)


# One or more <location/> tags (the model's location-change marker), together
# with all whitespace around them, collapse to a single paragraph break.
LOCATION_TAG_RE = re.compile(r"\s*(?:</?location\s*/?>\s*)+")
SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]}”’]*\s+")
SENTENCE_ABBREVIATIONS = {
    "dr.", "e.g.", "etc.", "i.e.", "jr.", "mr.", "mrs.", "ms.",
    "prof.", "sr.", "st.", "vs.",
}
NO_SPACE_BEFORE_RE = re.compile(r"^(?:[.!?,;:…)\]}”’]|[\"'][.!?,;:…)\]}”’\s]|[\"']$)")


def last_sentence_start(text: str) -> int:
    """Offset where the last (possibly incomplete) sentence in `text`
    begins: after the final sentence terminator or newline, skipping
    abbreviations, initials, and lowercase continuations."""
    start = text.rfind("\n") + 1
    for m in SENTENCE_END_RE.finditer(text, start):
        token = text[: m.start() + 1].split()[-1].strip("\"'([{").lower()
        if token in SENTENCE_ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)+", token):
            continue
        if m.end() == len(text):
            continue  # next word unseen — could be a lowercase continuation
        if text[m.end()].islower():
            continue
        start = m.end()
    return start


class LocationTagFilter:
    """Incrementally converts <location/> markers in streamed text into a
    single paragraph break. A tag mid-sentence moves its break back to the
    start of that sentence (so the break never lands inside one); whitespace
    already around the tag is collapsed into the break. The current
    in-progress sentence is held back so a still-streaming tag can relocate
    its break."""

    def __init__(self):
        self.pending = ""
        self.emitted_any = False
        self.last_was_break = False

    def _push(self, out: list, text: str) -> None:
        if not text:
            return
        out.append(text)
        self.emitted_any = True
        if text.strip():
            self.last_was_break = False

    def _maybe_break(self, out: list) -> None:
        if self.emitted_any and not self.last_was_break:
            out.append("\n\n")
            self.last_was_break = True

    def _handle_tag(self, out: list, m: "re.Match[str]") -> None:
        before = self.pending[: m.start()]
        ws_prefix = m.group(0)[: m.group(0).index("<")]
        ss = last_sentence_start(before)
        frag = before[ss:]
        at_sentence_start = (
            not frag.strip()
            or "\n" in ws_prefix
            or re.search(r"[.!?…][\"')\]}”’]*$", before) is not None
        )
        if at_sentence_start:
            self._push(out, before)
            self._maybe_break(out)
            self.pending = self.pending[m.end():]
        else:
            # Mid-sentence tag: the break moves to the sentence start, and
            # the prose joins seamlessly where the tag was.
            self._push(out, before[:ss].rstrip())
            self._maybe_break(out)
            self._push(out, frag.lstrip())
            after = self.pending[m.end():]
            self.pending = after if NO_SPACE_BEFORE_RE.match(after) else " " + after

    def feed(self, text: str) -> str:
        self.pending += text
        out: list[str] = []
        while True:
            m = LOCATION_TAG_RE.search(self.pending)
            if m is None or m.end() == len(self.pending):
                break  # absent, or its trailing whitespace could still grow
            self._handle_tag(out, m)
        # Emit completed sentences; hold the in-progress one (and any
        # trailing whitespace or tag that may still be streaming in).
        m = LOCATION_TAG_RE.search(self.pending)
        limit = m.start() if m is not None else len(self.pending)
        cut = last_sentence_start(self.pending[:limit])
        while cut > 0 and self.pending[cut - 1].isspace():
            cut -= 1
        self._push(out, self.pending[:cut])
        self.pending = self.pending[cut:]
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        while True:
            m = LOCATION_TAG_RE.search(self.pending)
            if m is None:
                break
            if m.end() == len(self.pending):
                # A tag run at the very end of the page marks nothing.
                self._push(out, self.pending[: m.start()])
                self.pending = ""
                break
            self._handle_tag(out, m)
        self._push(out, self.pending)
        self.pending = ""
        return "".join(out)


def collapse_location_tags(text: str) -> str:
    f = LocationTagFilter()
    return f.feed(text) + f.flush()


def strip_display_markup(text: str) -> str:
    return strip_color_markup(text).replace("**", "").replace("*", "")


def visible_markup_text(text: str, color_resolver=None) -> str:
    text = collapse_location_tags(strip_progress(strip_hidden(text)))
    if color_resolver is not None:
        text = colorize_voice_spans(text, color_resolver)
    return strip_rendering_tags_for_display(text)


def visible_display_text(text: str) -> str:
    return strip_display_markup(visible_markup_text(text))


def contains_cardinal_direction(text: str) -> bool:
    return CARDINAL_DIRECTION_RE.search(visible_display_text(text)) is not None


def last_visible_line_is_question(text: str) -> bool:
    visible = visible_display_text(text)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    return bool(lines and LAST_LINE_QUESTION_RE.search(lines[-1]))


def has_story_progress_tag(text: str) -> bool:
    visible_to_engine = strip_hidden(text)
    return PROGRESS_MARKER_TAG_RE.search(visible_to_engine) is not None


def story_progress_missing_streak(events: list[dict]) -> int:
    streak = 0
    for e in reversed(events):
        if e["type"] != "narrator":
            continue
        text = e.get("text", "")
        if not visible_display_text(text).strip():
            continue
        if has_story_progress_tag(text):
            break
        streak += 1
    return streak


# ── Cost accounting ──────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def usage_to_dict(um) -> dict:
    return {
        "prompt": um.prompt_token_count or 0,
        "cached": getattr(um, "cached_content_token_count", 0) or 0,
        "output": um.candidates_token_count or 0,
        "thoughts": getattr(um, "thoughts_token_count", 0) or 0,
    }


def usage_cost(u: dict) -> float:
    fresh_in = max(0, u["prompt"] - u["cached"])
    return (
        fresh_in * PRICE_IN_PER_M
        + u["cached"] * PRICE_IN_CACHED_PER_M
        + (u["output"] + u["thoughts"]) * PRICE_OUT_PER_M
    ) / 1_000_000


def elevenlabs_voice_design_rate_per_1k_chars() -> float:
    value = os.environ.get(ELEVENLABS_VOICE_DESIGN_PRICE_ENV, "").strip()
    if not value:
        return ELEVENLABS_VOICE_DESIGN_PRICE_PER_1K_CHARS_DEFAULT
    try:
        return float(value)
    except ValueError:
        return ELEVENLABS_VOICE_DESIGN_PRICE_PER_1K_CHARS_DEFAULT


def elevenlabs_voice_design_cost(characters: int) -> float:
    return max(0, int(characters)) * elevenlabs_voice_design_rate_per_1k_chars() / 1_000


def fmt_cost_line(label: str, u: dict, cost: float, total: float, estimated: bool) -> str:
    est = " (estimated)" if estimated else ""
    return (
        f"─ {label}: ${cost:.4f}{est} "
        f"[in {u['prompt']:,} | out {u['output']:,} | thoughts {u['thoughts']:,}] "
        f"· total: ${total:.4f}"
    )


def _empty_usage() -> dict[str, int]:
    return {"prompt": 0, "cached": 0, "output": 0, "thoughts": 0}


def _add_usage(total: dict[str, int], usage: dict | None) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt", "cached", "output", "thoughts"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)


def summarize_cost_events(events: list[dict]) -> dict[str, dict]:
    buckets = {
        "gemini_narration": {
            "label": "Gemini narration",
            "cost": 0.0,
            "calls": 0,
            "usage": _empty_usage(),
            "estimated": 0,
        },
        "gemini_compaction": {
            "label": "Gemini compaction",
            "cost": 0.0,
            "calls": 0,
            "usage": _empty_usage(),
            "estimated": 0,
        },
        "gemini_voice_description": {
            "label": "Gemini character voice descriptions",
            "cost": 0.0,
            "calls": 0,
            "usage": _empty_usage(),
            "estimated": 0,
        },
        "gemini_discarded": {
            "label": "Gemini discarded/interrupted attempts",
            "cost": 0.0,
            "calls": 0,
            "usage": _empty_usage(),
            "estimated": 0,
        },
        "elevenlabs_voice_design": {
            "label": "ElevenLabs voice design",
            "cost": 0.0,
            "calls": 0,
            "characters": 0,
            "credits": 0,
            "estimated": 0,
            "rate_per_1k_chars": elevenlabs_voice_design_rate_per_1k_chars(),
            "character_names": set(),
        },
        "other": {
            "label": "Other",
            "cost": 0.0,
            "calls": 0,
            "usage": _empty_usage(),
            "estimated": 0,
        },
    }

    for event in events:
        event_type = event.get("type")
        if event_type == "narrator":
            key = "gemini_narration"
        elif event_type == "compaction":
            key = "gemini_compaction"
        elif event_type == "external_cost":
            service = event.get("service")
            category = event.get("category")
            if service == "gemini" and category == "character_voice_description":
                key = "gemini_voice_description"
            elif service == "gemini" and category in (
                "discarded_narration",
                "cancelled_narration",
            ):
                key = "gemini_discarded"
            elif service == "elevenlabs" and category == "voice_design":
                key = "elevenlabs_voice_design"
            else:
                key = "other"
        elif "cost" in event:
            key = "other"
        else:
            continue

        bucket = buckets[key]
        bucket["cost"] += float(event.get("cost", 0.0) or 0.0)
        bucket["calls"] += 1
        if event.get("estimated"):
            bucket["estimated"] += 1
        if key == "elevenlabs_voice_design":
            bucket["characters"] += int(event.get("characters", 0) or 0)
            bucket["credits"] += int(event.get("credits", 0) or 0)
            character_name = str(event.get("character_name") or "").strip()
            if character_name:
                bucket["character_names"].add(character_name)
        else:
            _add_usage(bucket["usage"], event.get("usage"))
    return buckets


def format_cost_breakdown_section(title: str, events: list[dict]) -> str:
    buckets = summarize_cost_events(events)
    total = sum(bucket["cost"] for bucket in buckets.values())
    lines = [f"{title}: ${total:.4f}"]
    any_bucket = False
    for key in (
        "gemini_narration",
        "gemini_compaction",
        "gemini_voice_description",
        "gemini_discarded",
        "elevenlabs_voice_design",
        "other",
    ):
        bucket = buckets[key]
        if bucket["calls"] <= 0 and bucket["cost"] == 0:
            continue
        any_bucket = True
        estimated = " estimated" if bucket.get("estimated") else ""
        if key == "elevenlabs_voice_design":
            names = bucket["character_names"]
            voice_text = f"; {len(names)} voice{'s' if len(names) != 1 else ''}" if names else ""
            lines.append(
                "  "
                f"{bucket['label']}: ${bucket['cost']:.4f}{estimated} "
                f"({bucket['calls']} call{'s' if bucket['calls'] != 1 else ''}; "
                f"{bucket['credits']:,} credits{voice_text}; "
                f"${bucket['rate_per_1k_chars']:.2f}/1K credits)"
            )
        else:
            usage = bucket["usage"]
            lines.append(
                "  "
                f"{bucket['label']}: ${bucket['cost']:.4f}{estimated} "
                f"({bucket['calls']} call{'s' if bucket['calls'] != 1 else ''}; "
                f"in {usage['prompt']:,}, cached {usage['cached']:,}, "
                f"out {usage['output']:,}, thoughts {usage['thoughts']:,})"
            )
    if not any_bucket:
        lines.append("  no billable events")
    return "\n".join(lines)


def configured_tts_engine_for_prompt() -> str:
    return os.environ.get("IF_ENGINE_TTS_ENGINE", "omnivoice").strip().lower() or "omnivoice"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_is_set(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value != ""


def resolve_voice_runtime_options(args) -> dict:
    car_mode = args.car_mode if args.car_mode is not None else env_bool("IF_ENGINE_CAR_MODE", False)
    if args.wake_word is not None:
        wake_word = args.wake_word
    else:
        wake_word = env_bool("IF_ENGINE_WAKE_WORD", car_mode)

    if args.wake_word_threshold is not None:
        wake_threshold = args.wake_word_threshold
    elif env_is_set("IF_ENGINE_WAKE_WORD_THRESHOLD"):
        wake_threshold = env_float("IF_ENGINE_WAKE_WORD_THRESHOLD", 0.9)
    elif car_mode:
        wake_threshold = env_float("IF_ENGINE_CAR_WAKE_WORD_THRESHOLD", 0.4)
    else:
        wake_threshold = 0.9

    if args.wake_word_preprocess is not None:
        wake_preprocess = args.wake_word_preprocess
    else:
        wake_preprocess = env_bool("IF_ENGINE_WAKE_WORD_PREPROCESS", car_mode)

    whisper_tags = (
        args.whisper_tags
        if args.whisper_tags is not None
        else env_bool("IF_ENGINE_OMNIVOICE_WHISPER_TAGS", False)
    )

    options = {
        "car_mode": car_mode,
        "wake_word_enabled": wake_word,
        "openwakeword_threshold": wake_threshold,
        "wake_word_preprocess": wake_preprocess,
        "omnivoice_whisper_tags": whisper_tags,
    }
    if args.wake_word_model:
        options["openwakeword_models"] = [Path(item).expanduser() for item in args.wake_word_model]
    return options


def apply_voice_runtime_options(config, options: dict) -> None:
    config.car_mode = bool(options["car_mode"])
    config.wake_word_enabled = bool(options["wake_word_enabled"])
    config.openwakeword_threshold = float(options["openwakeword_threshold"])
    config.wake_word_preprocess = bool(options["wake_word_preprocess"])
    config.omnivoice_whisper_tags = bool(options["omnivoice_whisper_tags"])
    if "openwakeword_models" in options:
        config.openwakeword_models = list(options["openwakeword_models"])


# ── Session persistence ──────────────────────────────────────────────────────

class Session:
    """Append-only JSONL event log; doubles as the resume/playback source."""

    def __init__(self, name: str):
        self.name = name
        self.path = SESSIONS_DIR / f"{name}.jsonl"
        self.events: list[dict] = []
        self.lock = threading.RLock()
        self._lock_fd = None
        if self.path.exists():
            # Load tolerantly: a kill -9 / power loss mid-write can leave a
            # truncated final line. Skip any unparseable line (one damaged line
            # must not brick this session, nor crash --list over all sessions).
            damaged = 0
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        self.events.append(json.loads(line))
                    except ValueError:
                        damaged += 1
            if damaged:
                print(
                    f"warning: skipped {damaged} damaged line(s) in "
                    f"{self.path.name} (incomplete write?)",
                    file=sys.stderr,
                )

    def lock_for_play(self) -> None:
        """Best-effort advisory cross-process lock for interactive play, so two
        processes don't open the same session and clobber each other via the
        whole-file rewrite. Strictly best-effort: silently skipped where fcntl
        is unavailable. NOT called from list_sessions / read-only listing."""
        try:
            import fcntl
        except ImportError:
            return
        lock_path = self.path.with_suffix(".lock")
        try:
            SESSIONS_DIR.mkdir(exist_ok=True)
            fd = open(lock_path, "w")
        except OSError:
            return
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.close()
            sys.exit(f"session '{self.name}' is already open in another process.")
        # Held for the process lifetime; the OS releases it on exit.
        self._lock_fd = fd

    @property
    def is_new(self) -> bool:
        return not self.events

    def append(self, event: dict) -> None:
        with self.lock:
            event["ts"] = datetime.now(timezone.utc).isoformat()
            self.events.append(event)
            SESSIONS_DIR.mkdir(exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def total_cost(self, start: int = 0) -> float:
        with self.lock:
            return sum(e.get("cost", 0.0) for e in self.events[start:])

    def cost_events(self, start: int = 0) -> list[dict]:
        with self.lock:
            return [dict(e) for e in self.events[start:] if "cost" in e]

    def last_context_tokens(self) -> int:
        with self.lock:
            for e in reversed(self.events):
                if "context_tokens" in e:
                    return e["context_tokens"]
        return 0

    def has_narration(self) -> bool:
        # A narrator event with no visible content (an opening that was cut
        # off before any text arrived) does not count — otherwise the session
        # would resume with nothing to show and no opening to regenerate.
        return any(
            e["type"] == "narrator" and visible_display_text(e.get("text", "")).strip()
            for e in self.events
        )

    def game(self) -> str:
        for e in self.events:
            if e["type"] == "meta" and e.get("game"):
                return e["game"]
        return DEFAULT_GAME


def compaction_retain_start(events: list[dict], n_turns: int) -> int:
    """Index into ``events`` of the first event to keep verbatim when compacting,
    so the last ``n_turns`` player turns survive outside the summary. A turn
    begins at a player or interruption event. Returns ``len(events)`` (retain
    nothing) when there are not more than ``n_turns`` turns to summarize."""
    if n_turns <= 0:
        return len(events)
    turn_starts = [i for i, e in enumerate(events)
                   if e["type"] in ("player", "interruption")]
    if len(turn_starts) <= n_turns:
        return len(events)
    return turn_starts[-n_turns]


def build_history(session: Session, transcript_message: str) -> list[types.Content]:
    """Rebuild the Gemini contents list: transcript message, last compaction
    summary (if any), then every retained/subsequent turn. The most recent
    turns are kept verbatim outside the summary (see ``retain_start`` on the
    compaction event); older sessions, whose compaction events predate that
    field, fall back to keeping only the turns after the summary."""

    def msg(role: str, text: str) -> types.Content:
        return types.Content(role=role, parts=[types.Part(text=text)])

    history = [msg("user", transcript_message)]

    last_compaction = None
    last_compaction_idx = None
    for i, e in enumerate(session.events):
        if e["type"] == "compaction":
            last_compaction = e
            last_compaction_idx = i
    if last_compaction is not None:
        history.append(msg("user", SUMMARY_WRAPPER.format(summary=last_compaction["summary"])))
        start = last_compaction.get("retain_start", last_compaction_idx + 1)
    else:
        start = 0

    for e in session.events[start:]:
        if e["type"] in ("player", "interruption"):
            history.append(msg("user", e["text"]))
        elif e["type"] == "narrator":
            if e.get("text", "").strip():
                history.append(msg("model", e["text"]))
        elif e["type"] == "engine_reminder":
            history.append(msg("user", e["text"]))
        # compaction (and non-message) events in this range are skipped: the
        # one whose summary we already emitted, plus any retained-range marker.
    return history


class TerminalUI:
    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self):
        self.raw_output_text = ""
        self.submit_callback = None
        self.lock = threading.RLock()
        self.spinner_active = False
        self.spinner_index = 0
        self.spinner_pos = None
        self.spinner_suffix = ""
        self.spinner_task = None
        self.first_spinner_label_pending = True
        self._exiting = False
        self.key_bindings = KeyBindings()
        self.key_bindings.add("c-c")(self._handle_ctrl_c)
        self.key_bindings.add("pageup")(self._page_up)
        self.key_bindings.add("pagedown")(self._page_down)

        # The output pane owns its scroll position: after each accepted input
        # the echo line is pinned to the top of the screen and narration
        # streams in below it, so nothing jumps while you read — and manual
        # scrolling (mouse wheel / PageUp / PageDown) is never overridden.
        # The control reports a virtual cursor at the top visible line so
        # prompt_toolkit's keep-cursor-visible logic never adjusts the view.
        self.display_text = ""
        self.display_fragments = []
        self.scroll_mode = "bottom"   # follow the bottom until the first echo
        self.scroll_pending = None    # one-shot: "anchor" | "bottom"
        self.last_line_count = 0
        self.output_control = FormattedTextControl(
            lambda: self.display_fragments,
            get_cursor_position=lambda: Point(
                x=0, y=self.output_window.vertical_scroll
            ),
        )
        self.output_window = Window(
            self.output_control,
            wrap_lines=False,
            allow_scroll_beyond_bottom=True,
        )
        self.input_prompt = "❯ "
        self.input = TextArea(
            height=1,
            multiline=False,
            prompt=lambda: [(self.SYSTEM_STYLE, self.input_prompt)],
            wrap_lines=False,
            accept_handler=self._accept_input,
        )
        self.app = Application(
            layout=Layout(
                HSplit(
                    [
                        self.output_window,
                        Window(FormattedTextControl(""), height=1),
                        self.input,
                    ]
                ),
                focused_element=self.input,
            ),
            full_screen=True,
            mouse_support=True,
            key_bindings=self.key_bindings,
        )

    def _handle_ctrl_c(self, event) -> None:
        if self.submit_callback is not None:
            self.submit_callback("/exit")
        self.exit()

    def _accept_input(self, buffer) -> bool:
        text = buffer.text
        buffer.set_document(Document("", 0), bypass_readonly=True)
        if self.submit_callback is not None:
            self.submit_callback(text)
        return True

    async def run_async(self):
        return await self.app.run_async()

    def exit(self) -> None:
        self.stop_spinner()
        if self._exiting:
            return
        self._exiting = True
        try:
            self.app.exit()
        except Exception as exc:
            if str(exc) != "Return value already set. Application.exit() failed.":
                raise

    def append_input_echo(self, text: str, append_to_previous: bool = False) -> None:
        """Append a '❯ text' echo, guaranteeing a blank line before it no
        matter how the preceding narration text ended."""
        if append_to_previous and self._append_to_last_input_echo(text):
            return

        with self.lock:
            current = self.raw_output_text
            if self.spinner_active and self.spinner_pos is not None:
                current = self._without_spinner(current)
            current = current.rstrip(" \t")
            if not current or current.endswith("\n\n"):
                prefix = ""
            elif current.endswith("\n"):
                prefix = "\n"
            else:
                prefix = "\n\n"
            # Pin this echo to the top of the screen on the next render.
            self.scroll_pending = "anchor"
        self.append(f"{prefix}❯ {text}\n\n", follow_bottom=False)

    def _append_to_last_input_echo(self, text: str) -> bool:
        with self.lock:
            current = self.raw_output_text
            removed_spinner = False
            if self.spinner_active and self.spinner_pos is not None:
                current = self._without_spinner(current)
                self.spinner_active = False
                self.spinner_pos = None
                self.spinner_suffix = ""
                removed_spinner = True

            lines = current.split("\n")
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("❯ "):
                    lines[i] = lines[i].rstrip() + " " + text
                    self.raw_output_text = "\n".join(lines)
                    self.scroll_pending = "anchor"
                    break
            else:
                if removed_spinner:
                    self.raw_output_text = current
                return False
        self._schedule_render()
        return True

    def set_input_prompt(self, prompt: str) -> None:
        with self.lock:
            if self.input_prompt == prompt:
                return
            self.input_prompt = prompt
        self._schedule_render()

    def set_text(self, text: str) -> None:
        with self.lock:
            self.raw_output_text = text
            self.spinner_active = False
            self.spinner_pos = None
            self.spinner_suffix = ""
            self.scroll_mode = "bottom"
            self.scroll_pending = "bottom"
        self._schedule_render()

    def append(self, text: str, follow_bottom: bool = True) -> None:
        with self.lock:
            if self.spinner_active and self.spinner_pos is not None:
                self.raw_output_text = self._without_spinner(self.raw_output_text)
                self.spinner_active = False
                self.spinner_pos = None
                self.spinner_suffix = ""
            self.raw_output_text += text
            if text and follow_bottom:
                self.scroll_pending = "bottom"
        self._schedule_render()

    def _spinner_text(self, frame: str | None = None) -> str:
        if frame is None:
            frame = self.SPINNER_FRAMES[self.spinner_index]
        return frame + self.spinner_suffix

    def _without_spinner(self, text: str) -> str:
        if self.spinner_pos is None:
            return text
        token_len = len(self._spinner_text())
        return text[: self.spinner_pos] + text[self.spinner_pos + token_len:]

    def start_spinner(self, label: str | None = None, blank_before: bool = False) -> None:
        with self.lock:
            if self.spinner_active:
                return
            if blank_before and self.raw_output_text and not self.raw_output_text.endswith("\n\n"):
                self.raw_output_text += "\n" if self.raw_output_text.endswith("\n") else "\n\n"
            if self.raw_output_text and not self.raw_output_text.endswith("\n"):
                self.raw_output_text += "\n"
            if label:
                self.spinner_suffix = " " + label
            else:
                self.spinner_suffix = " Loading game..." if self.first_spinner_label_pending else ""
                self.first_spinner_label_pending = False
            self.spinner_active = True
            self.spinner_index = 0
            self.spinner_pos = len(self.raw_output_text)
            self.raw_output_text += self._spinner_text()
        self._schedule_render()
        if self.spinner_task is None or self.spinner_task.done():
            self.spinner_task = asyncio.create_task(self._spin())

    def stop_spinner(self) -> None:
        with self.lock:
            if self.spinner_active and self.spinner_pos is not None:
                self.raw_output_text = self._without_spinner(self.raw_output_text)
            self.spinner_active = False
            self.spinner_pos = None
            self.spinner_suffix = ""
        self._schedule_render()

    async def _spin(self) -> None:
        while True:
            await asyncio.sleep(0.12)
            with self.lock:
                if not self.spinner_active or self.spinner_pos is None:
                    return
                self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_FRAMES)
                frame = self.SPINNER_FRAMES[self.spinner_index]
                before = self.raw_output_text[: self.spinner_pos]
                after = self.raw_output_text[self.spinner_pos + len(self._spinner_text()):]
                self.raw_output_text = before + self._spinner_text(frame) + after
            self._schedule_render()

    def _schedule_render(self) -> None:
        loop = getattr(self.app, "loop", None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._render)
        else:
            self._render()

    def _viewport_height(self) -> int:
        info = self.output_window.render_info
        if info is not None:
            return info.window_height
        return max(5, shutil.get_terminal_size((80, 24)).lines - 2)

    @staticmethod
    def _last_echo_line(text: str) -> int | None:
        lines = text.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("❯ "):
                return i
        return None

    def _page_up(self, event) -> None:
        step = max(1, self._viewport_height() - 2)
        w = self.output_window
        w.vertical_scroll = max(0, w.vertical_scroll - step)

    def _page_down(self, event) -> None:
        step = max(1, self._viewport_height() - 2)
        last_line = max(0, self.display_text.count("\n"))
        w = self.output_window
        w.vertical_scroll = min(last_line, w.vertical_scroll + step)

    def _render(self) -> None:
        with self.lock:
            marked_text = self._word_wrap(self.raw_output_text).rstrip("\n")
            text = strip_display_markup(marked_text)
            self.display_text = text
            self.display_fragments = self._format_display_text(marked_text)
            lines = text.count("\n") + 1
            pending = self.scroll_pending
            self.scroll_pending = None
            if pending == "anchor":
                anchor = self._last_echo_line(text)
                if anchor is not None:
                    self.output_window.vertical_scroll = anchor
                self.scroll_mode = "anchor"
            elif self.scroll_mode == "anchor" and (
                pending == "bottom" or lines != self.last_line_count
            ):
                anchor = self._last_echo_line(text)
                if anchor is None:
                    self.scroll_mode = "bottom"
                else:
                    self.output_window.vertical_scroll = max(
                        anchor,
                        lines - self._viewport_height(),
                    )
            if self.scroll_mode == "bottom" and (
                pending == "bottom" or lines != self.last_line_count
            ):
                self.output_window.vertical_scroll = max(
                    0, lines - self._viewport_height()
                )
            self.last_line_count = lines
        self.app.invalidate()

    @staticmethod
    def _visible_len(text: str, parse_markup: bool = True) -> int:
        if parse_markup:
            text = strip_display_markup(text)
        return len(text)

    def _wrap_content(self, content: str, width: int, parse_markup: bool = True) -> list[str]:
        words = re.findall(r"\S+", content)
        if not words:
            return []
        lines: list[str] = []
        current = ""
        current_len = 0
        for word in words:
            word_len = self._visible_len(word, parse_markup=parse_markup)
            if current and current_len + 1 + word_len > width:
                lines.append(current)
                current = word
                current_len = word_len
            else:
                if current:
                    current += " " + word
                    current_len += 1 + word_len
                else:
                    current = word
                    current_len = word_len
        if current:
            lines.append(current)
        return lines

    def _word_wrap(self, text: str) -> str:
        width = max(20, shutil.get_terminal_size((80, 24)).columns - 2)
        wrapped = []
        for index, line in enumerate(text.split("\n")):
            if not line:
                wrapped.append("")
                continue
            is_input = line.startswith("❯ ")
            is_chrome = index == 0 or is_session_instruction_line(line)
            prefix = "" if is_input or is_chrome else OUTPUT_INDENT
            content_width = max(20, width - len(prefix))
            content = line.strip()
            chunks = self._wrap_content(
                content,
                content_width,
                parse_markup=not is_input,
            )
            if chunks:
                wrapped.extend(prefix + chunk for chunk in chunks)
            else:
                wrapped.append(prefix + line.strip())
        return "\n".join(wrapped)

    SYSTEM_STYLE = f"fg:{SYSTEM_DISPLAY_COLOR}"
    DISPLAY_MARKUP_TOKEN_RE = re.compile(r"\*\*|\*|⟦fg=(#[0-9a-fA-F]{6})⟧|⟦/fg⟧")

    @staticmethod
    def _combine_style(base: str, bold: bool, italic: bool, color: str | None = None) -> str:
        styles = base.split() if base else []
        if bold and "bold" not in styles:
            styles.append("bold")
        if italic and "italic" not in styles:
            styles.append("italic")
        if color and not any(s.startswith("fg:") for s in styles):
            styles.append(f"fg:{color}")
        return " ".join(styles)

    @classmethod
    def _append_markdown_fragments(
        cls,
        fragments: list[tuple[str, str]],
        line: str,
        base_style: str = "",
        bold_active: bool = False,
        italic_active: bool = False,
        color_active: str | None = None,
        parse_markup: bool = True,
    ) -> tuple[bool, bool, str | None]:
        if not parse_markup:
            fragments.append((base_style, line))
            return bold_active, italic_active, color_active
        pos = 0
        while pos < len(line):
            token = cls.DISPLAY_MARKUP_TOKEN_RE.match(line, pos)
            if token is not None:
                matched = token.group(0)
                if matched == "**":
                    bold_active = not bold_active
                elif matched == "*":
                    italic_active = not italic_active
                elif matched == "⟦/fg⟧":
                    color_active = None
                else:
                    color_active = token.group(1)
                pos = token.end()
                continue
            nxt = cls.DISPLAY_MARKUP_TOKEN_RE.search(line, pos + 1)
            end = len(line) if nxt is None else nxt.start()
            fragments.append((
                cls._combine_style(base_style, bold_active, italic_active, color_active),
                line[pos:end],
            ))
            pos = end
        return bold_active, italic_active, color_active

    @classmethod
    def _format_display_text(cls, text: str):
        fragments = []
        instruction_active = False
        bold_active = False
        italic_active = False
        color_active: str | None = None
        for index, line in enumerate(text.split("\n")):
            if not line:
                instruction_active = False
                fragments.append(("", "\n"))
                continue
            if index == 0:
                bullet = line.find(" • ")
                if bullet >= 0:
                    fragments.append((f"bold {cls.SYSTEM_STYLE}", line[:bullet]))
                    fragments.append((cls.SYSTEM_STYLE, line[bullet:]))
                else:
                    fragments.append((f"bold {cls.SYSTEM_STYLE}", line))
                bold_active = False
                italic_active = False
                color_active = None
            elif is_session_instruction_line(line):
                instruction_active = True
                fragments.append((f"italic {cls.SYSTEM_STYLE}", line))
                bold_active = False
                italic_active = False
                color_active = None
            elif instruction_active:
                fragments.append((f"italic {cls.SYSTEM_STYLE}", line))
            elif line.startswith("❯ "):
                fragments.append((cls.SYSTEM_STYLE, "❯ "))
                fragments.append(("", line[2:]))
                bold_active = False
                italic_active = False
                color_active = None
            else:
                bold_active, italic_active, color_active = cls._append_markdown_fragments(
                    fragments,
                    line,
                    bold_active=bold_active,
                    italic_active=italic_active,
                    color_active=color_active,
                )
            fragments.append(("", "\n"))
        if fragments:
            fragments.pop()
        return fragments


class VoiceUtterance:
    """A transcribed spoken player input, with how much narration they heard
    and how much was shown on screen when they spoke."""

    def __init__(self, text: str, heard_text: str, displayed_text: str, was_speaking: bool):
        self.text = text
        self.heard_text = heard_text
        self.displayed_text = displayed_text
        self.was_speaking = was_speaking


class NarrationJob:
    def __init__(
        self,
        game,
        label: str,
        rollback_snapshot=None,
        cancel_snapshot=None,
        flush_chunks: bool = True,
        emit=None,
    ):
        self.game = game
        self.label = label
        self.rollback_snapshot = rollback_snapshot
        self.cancel_snapshot = cancel_snapshot or rollback_snapshot
        self.flush_chunks = flush_chunks
        self.emit = emit
        self.interrupt_event = threading.Event()
        self.done = threading.Event()
        self.interrupted = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def interrupt(self) -> None:
        self.interrupt_event.set()

    def wait(self) -> bool:
        self.done.wait()
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.interrupted

    def _run(self) -> None:
        try:
            self.interrupted = self.game.narrator_turn(
                label=self.label,
                interrupt_event=self.interrupt_event,
                flush_chunks=self.flush_chunks,
                emit=self.emit,
            )
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()


# ── The game engine ──────────────────────────────────────────────────────────

class Game:
    def __init__(
        self,
        session: Session,
        transcript_text: str,
        show_costs: bool = False,
        voice=None,
        voice_enabled: bool = False,
        game_name: str | None = DEFAULT_GAME,
        game_title: str = "Interactive Fiction",
        compact_at: int | None = None,
        fast_mode: bool = False,
        voice_options: dict | None = None,
        resumed_session: bool = False,
    ):
        self.session = session
        self.transcript_text = transcript_text
        self.show_costs = show_costs
        self.voice = voice
        self.voice_enabled = voice_enabled
        self.game_name = game_name
        self.game_title = game_title
        self.fast_mode = fast_mode
        self.voice_options = voice_options or {}
        self.run_start_event_index = len(session.events)
        self.resumed_session = resumed_session
        self._exit_cost_summary_printed = False
        self.base_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(title=game_title)
        self.transcript_message = (
            TRANSCRIPT_PREAMBLE_TEMPLATE.format(title=game_title)
            + transcript_text
            + TRANSCRIPT_EPILOGUE
        )
        self.compact_at_override = compact_at
        self.compact_at = compact_at or 0
        self.voice_prompt_enabled = voice is not None
        voice_config = getattr(voice, "config", None)
        self.prompt_tts_engine = (
            getattr(voice_config, "tts_engine", None)
            or (configured_tts_engine_for_prompt() if voice_enabled else "none")
        )
        self.prompt_wake_word_enabled = (
            bool(self.voice_options.get("wake_word_enabled", False)) if voice_enabled else False
        )
        self.prompt_car_mode = bool(self.voice_options.get("car_mode", False)) if voice_enabled else False
        self.prompt_wake_threshold = (
            self.voice_options.get("openwakeword_threshold") if voice_enabled else None
        )
        self.prompt_wake_preprocess = (
            bool(self.voice_options.get("wake_word_preprocess", False)) if voice_enabled else False
        )
        self.prompt_omnivoice_whisper_tags = (
            bool(self.voice_options.get("omnivoice_whisper_tags", False)) if voice_enabled else False
        )
        self.prompt_omnivoice_character_voices = (
            bool(getattr(
                voice_config,
                "omnivoice_character_voices_enabled",
                env_bool("IF_ENGINE_OMNIVOICE_CHARACTER_VOICES", True),
            ))
            if voice_enabled
            else False
        )
        self.system_prompt = ""
        self._voice_prompt_block: str | None = None
        self._character_display_colors: dict[str, str] = {}
        self.refresh_system_prompt()
        self.client = genai.Client(api_key=require_gemini_api_key())
        self.history = build_history(session, self.transcript_message)
        self.context_tokens = self.estimate_context_tokens()

    # ── streaming with interruption ──────────────────────────────────────

    def character_display_color(self, character_name: str) -> str | None:
        """Voice-design display color for a character, from the voice cache.
        Misses are not cached: a voice generated later in the session starts
        coloring as soon as its cache entry exists."""
        key = character_name.casefold()
        color = self._character_display_colors.get(key)
        if color:
            return color
        cached = read_cached_character_voice(
            elevenlabs_voice_cache_root(),
            self.game_name or "default",
            character_name,
        )
        if cached is None or not cached.display_color:
            return None
        self._character_display_colors[key] = cached.display_color
        return cached.display_color

    def record_external_cost_event(self, event: dict) -> None:
        service = str(event.get("service") or "").strip().lower()
        category = str(event.get("category") or "").strip().lower()
        cost_event = {
            "type": "external_cost",
            "service": service,
            "category": category,
            "label": event.get("label") or f"{service} {category}".strip(),
        }
        for key in (
            "character_name",
            "transcript_filename",
            "transcript_filename_stem",
            "model",
            "preview_count",
            "quality",
            "characters",
            "credits",
        ):
            if key in event:
                cost_event[key] = event[key]

        if service == "gemini":
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            usage_dict = {
                "prompt": int(usage.get("prompt", 0) or 0),
                "cached": int(usage.get("cached", 0) or 0),
                "output": int(usage.get("output", 0) or 0),
                "thoughts": int(usage.get("thoughts", 0) or 0),
            }
            cost_event["usage"] = usage_dict
            cost_event["cost"] = usage_cost(usage_dict)
            cost_event["estimated"] = False
        elif service == "elevenlabs" and category == "voice_design":
            characters = int(event.get("characters", 0) or 0)
            credits = int(event.get("credits", characters) or characters)
            rate = elevenlabs_voice_design_rate_per_1k_chars()
            cost_event["characters"] = characters
            cost_event["credits"] = credits
            cost_event["rate_per_1k_chars"] = rate
            # Credits are the actual billing unit (taken from the ElevenLabs
            # `character-cost` response header when available); dollar
            # conversion depends on the current plan/PAYG rate, so the cost
            # stays marked as estimated.
            cost_event["cost"] = credits * rate / 1_000
            cost_event["estimated"] = True
        else:
            raw_cost = event.get("cost", 0.0)
            cost_event["cost"] = float(raw_cost or 0.0)
            cost_event["estimated"] = bool(event.get("estimated", True))

        self.session.append(cost_event)

    def cost_summary_text(self, *, include_context: bool = True) -> str:
        sections = ["[Cost breakdown]"]
        if self.resumed_session:
            sections.append(
                format_cost_breakdown_section(
                    "This run",
                    self.session.cost_events(self.run_start_event_index),
                )
            )
        sections.append(
            format_cost_breakdown_section(
                "Entire session",
                self.session.cost_events(),
            )
        )
        if include_context:
            sections.append(f"Context ≈ {self.context_tokens:,} tokens")
        return "\n".join(sections) + "\n"

    def emit_exit_cost_summary(self, emit=None) -> None:
        if self._exit_cost_summary_printed:
            return
        self._exit_cost_summary_printed = True
        body = self.cost_summary_text(include_context=True)
        if emit is None:
            print("\n" + body, end="", flush=True)
        else:
            emit("\n" + system_markup(body.rstrip("\n")) + "\n")

    def compose_system_prompt(self) -> str:
        voice_mode = "enabled" if self.voice_prompt_enabled else "disabled"
        tts_engine = self.prompt_tts_engine if self.voice_prompt_enabled else "none"
        system_prompt = self.base_system_prompt
        system_prompt += RUNTIME_CONFIG_PROMPT_TEMPLATE.format(
            voice_mode=voice_mode,
            tts_engine=tts_engine,
            latency_tier="low latency" if self.fast_mode else "standard latency",
            wake_word=(
                "enabled" if self.voice_prompt_enabled and self.prompt_wake_word_enabled else "disabled"
            ),
            car_mode="on" if self.voice_prompt_enabled and self.prompt_car_mode else "off",
            wake_threshold=(
                f"{float(self.prompt_wake_threshold):.2f}"
                if self.voice_prompt_enabled and self.prompt_wake_threshold is not None
                else "n/a"
            ),
            wake_preprocess=(
                "on" if self.voice_prompt_enabled and self.prompt_wake_preprocess else "off"
            ),
            whisper_tags=(
                "enabled"
                if self.voice_prompt_enabled and self.prompt_omnivoice_whisper_tags
                else "disabled"
            ),
            character_voices=(
                "enabled"
                if (
                    self.voice_prompt_enabled
                    and self.prompt_tts_engine == "omnivoice"
                    and self.prompt_omnivoice_character_voices
                )
                else "disabled"
            ),
        )
        if self.voice_prompt_enabled:
            system_prompt += VOICE_SYSTEM_PROMPT_ADDENDUM
            if self.prompt_tts_engine == "omnivoice":
                system_prompt += OMNIVOICE_SYSTEM_PROMPT_ADDENDUM
                if self.prompt_omnivoice_character_voices:
                    # Snapshot the cached-voice block for the whole session:
                    # Gemini implicit caching matches byte-identical request
                    # prefixes (system instruction included), so re-reading
                    # the voice cache every turn would bust the cache for the
                    # entire prefix — transcript and all — the moment a new
                    # voice finishes generating mid-session. The narration
                    # model still reuses mid-session voice names from its own
                    # <voice> tags in history; the refreshed list is picked
                    # up on the next voice-config change or session start.
                    if self._voice_prompt_block is None:
                        self._voice_prompt_block = build_omnivoice_voice_prompt_block(
                            self.game_name or "default"
                        )
                    system_prompt += self._voice_prompt_block
                if self.prompt_omnivoice_whisper_tags:
                    system_prompt += OMNIVOICE_WHISPER_SYSTEM_PROMPT_ADDENDUM
        return system_prompt

    def estimate_context_tokens(self) -> int:
        history_text = "".join(p.text or "" for c in self.history for p in c.parts)
        return estimate_tokens(self.system_prompt) + estimate_tokens(history_text)

    def refresh_system_prompt(
        self,
        *,
        voice_prompt_enabled: bool | None = None,
        tts_engine: str | None = None,
        wake_word_enabled: bool | None = None,
        car_mode: bool | None = None,
        wake_threshold: float | None = None,
        wake_preprocess: bool | None = None,
        omnivoice_whisper_tags: bool | None = None,
        omnivoice_character_voices: bool | None = None,
        reestimate_context: bool = False,
    ) -> None:
        # A real voice-config change (voice startup/failure) may legitimately
        # change the system prompt; refresh the cached-voice snapshot then, so
        # newly cached voices land in one batch instead of busting the Gemini
        # implicit cache turn by turn.
        if any(
            arg is not None
            for arg in (
                voice_prompt_enabled,
                tts_engine,
                wake_word_enabled,
                car_mode,
                wake_threshold,
                wake_preprocess,
                omnivoice_whisper_tags,
                omnivoice_character_voices,
            )
        ):
            self._voice_prompt_block = None
        if voice_prompt_enabled is not None:
            self.voice_prompt_enabled = voice_prompt_enabled
        if tts_engine is not None:
            self.prompt_tts_engine = tts_engine.strip().lower() or "none"
        if wake_word_enabled is not None:
            self.prompt_wake_word_enabled = wake_word_enabled
        if car_mode is not None:
            self.prompt_car_mode = car_mode
        if wake_threshold is not None:
            self.prompt_wake_threshold = wake_threshold
        if wake_preprocess is not None:
            self.prompt_wake_preprocess = wake_preprocess
        if omnivoice_whisper_tags is not None:
            self.prompt_omnivoice_whisper_tags = omnivoice_whisper_tags
        if omnivoice_character_voices is not None:
            self.prompt_omnivoice_character_voices = omnivoice_character_voices
        old_prompt = self.system_prompt
        self.system_prompt = self.compose_system_prompt()
        if self.compact_at_override is None:
            # Compact once the context grows a fixed margin past the immutable
            # base (current system prompt + transcript message).
            self.compact_at = (
                estimate_tokens(self.system_prompt)
                + estimate_tokens(self.transcript_message)
                + COMPACT_MARGIN_TOKENS
            )
        if reestimate_context and old_prompt != self.system_prompt:
            self.context_tokens = self.estimate_context_tokens()

    def _gen_config(self) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        )
        if self.fast_mode:
            config.service_tier = FAST_SERVICE_TIER
        return config

    def stream_narration(self, interrupt_event=None, flush_chunks: bool = True, emit=None):
        """Stream one model response, printing visible text as it arrives.

        Returns (raw_text, usage_dict_or_None, interrupted: bool).
        Raw text (including <hidden> spans and <progress/> tags) goes to history;
        the terminal only ever sees filtered text.
        """
        dash = EmDashSpacingFilter()
        filt = HiddenStreamFilter()
        prog = ProgressStreamFilter()
        loc = LocationTagFilter()
        render = RenderingTagStreamFilter(color_resolver=self.character_display_color)
        raw_parts: list[str] = []
        last_usage = None
        interrupted = False
        visible_ends_with_newline = False
        # Retrying is only safe before any chunk has been displayed; once the
        # stream has progressed, a transient mid-stream error must surface as-is
        # (retrying would duplicate already-shown narration).
        progressed = False
        attempts_done = 1

        while True:
            try:
                stream = self.client.models.generate_content_stream(
                    model=MODEL, contents=self.history, config=self._gen_config()
                )
            except Exception as exc:
                if not progressed and should_retry_gemini(exc, attempts_done, emit):
                    attempts_done += 1
                    continue
                raise

            q: queue.Queue = queue.Queue()
            abort = threading.Event()

            def consume(stream=stream, q=q, abort=abort):
                try:
                    for chunk in stream:
                        if abort.is_set():
                            break
                        try:
                            text = chunk.text
                        except Exception:
                            text = None
                        q.put(("chunk", text, getattr(chunk, "usage_metadata", None)))
                    q.put(("done", None, None))
                except Exception as exc:  # surfaced to the main loop
                    q.put(("error", exc, None))
                finally:
                    # Stop server-side generation promptly on done/abort/error;
                    # closing an already-exhausted stream is a harmless no-op.
                    try:
                        stream.close()
                    except Exception:
                        pass

            threading.Thread(target=consume, daemon=True).start()

            retry_stream = False
            try:
                done = False
                while not done:
                    if interrupt_event is not None and interrupt_event.is_set():
                        abort.set()
                        try:
                            stream.close()
                        except Exception:
                            pass
                        interrupted = True
                        break
                    try:
                        kind, payload, um = q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if kind == "error":
                        exc = payload
                        if not progressed and should_retry_gemini(exc, attempts_done, emit):
                            attempts_done += 1
                            retry_stream = True
                            break
                        raise exc
                    if kind == "done":
                        done = True
                        continue
                    progressed = True
                    if um is not None:
                        last_usage = um
                    if payload:
                        payload = dash.feed(payload)
                    if payload:
                        raw_parts.append(payload)
                        visible = loc.feed(prog.feed(filt.feed(payload)))
                        if visible:
                            if self.voice is not None:
                                # Voice mode: the UI shows each sentence when its
                                # audio starts playing, not as the text streams.
                                self.voice.feed_text(visible)
                            else:
                                display_visible = render.feed(visible)
                                if display_visible:
                                    visible_ends_with_newline = display_visible.endswith("\n")
                                    if emit is None:
                                        print(
                                            strip_display_markup(display_visible),
                                            end="",
                                            flush=flush_chunks,
                                        )
                                    else:
                                        emit(display_visible)
                if not retry_stream:
                    dash_tail = dash.flush()
                    if dash_tail:
                        raw_parts.append(dash_tail)
                    hidden_tail = (filt.feed(dash_tail) if dash_tail else "") + filt.flush()
                    tail = loc.feed(prog.feed(hidden_tail) + prog.flush()) + loc.flush()
                    if tail and not interrupted:
                        if self.voice is not None:
                            self.voice.feed_text(tail)
                        else:
                            display_tail = render.feed(tail) + render.flush()
                            if display_tail:
                                visible_ends_with_newline = display_tail.endswith("\n")
                                if emit is None:
                                    print(strip_display_markup(display_tail), end="", flush=True)
                                else:
                                    emit(display_tail)
                    elif not interrupted and self.voice is None:
                        display_tail = render.flush()
                        if display_tail:
                            visible_ends_with_newline = display_tail.endswith("\n")
                            if emit is None:
                                print(strip_display_markup(display_tail), end="", flush=True)
                            else:
                                emit(display_tail)
            except KeyboardInterrupt:
                abort.set()
                try:
                    stream.close()
                except Exception:
                    pass
                interrupted = True
            if retry_stream:
                continue
            break

        if self.voice is None and not visible_ends_with_newline:
            if emit is None:
                print(flush=True)
            else:
                emit("\n")

        usage = usage_to_dict(last_usage) if last_usage is not None else None
        return "".join(raw_parts), usage, interrupted

    # ── one full turn ────────────────────────────────────────────────────

    def needs_opening(self) -> bool:
        """A new playthrough opens with a model-generated first page: the
        game's real opening content from the transcript, minus title screens,
        banners, help text, and other non-story chrome."""
        return not self.session.has_narration()

    def pending_player_turn(self) -> bool:
        """True when the session's last story event is player input with no
        narrator reply (the reply errored out or was repaired away); the
        resume path answers it immediately."""
        for e in reversed(self.session.events):
            if e["type"] in ("player", "interruption"):
                return True
            if e["type"] == "narrator":
                return False
        return False

    def narrator_turn(
        self, label: str = "turn", interrupt_event=None, flush_chunks: bool = True, emit=None
    ) -> bool:
        """Stream a narrator response to the current history, record it,
        report cost, and compact if needed. Returns interrupted flag."""
        self.refresh_system_prompt(reestimate_context=True)
        attempts = 0
        while True:
            if self.voice is not None:
                self.voice.begin_utterance()
            try:
                raw, usage, interrupted = self.stream_narration(
                    interrupt_event=interrupt_event, flush_chunks=flush_chunks, emit=emit
                )
            except BaseException:
                if self.voice is not None:
                    self.voice.cancel_utterance()
                raise
            empty = not interrupted and not visible_display_text(raw).strip()
            if self.voice is not None:
                if interrupted or empty:
                    self.voice.cancel_utterance()
                else:
                    self.voice.finish_utterance()
            if not raw and usage is None and not interrupted:
                raise RuntimeError("model returned an empty response")
            if interrupted and not visible_display_text(raw).strip():
                # Cut off before any visible text arrived (e.g. the player
                # quit during the opening). No narrator event to record —
                # leaving the session as-is means it resumes cleanly
                # (regenerating the opening, or answering a pending player
                # turn) instead of being poisoned by an empty narrator event.
                # The call itself was still billed, so keep its real usage.
                if usage is not None:
                    self.record_external_cost_event(
                        {
                            "service": "gemini",
                            "category": "cancelled_narration",
                            "label": "Gemini narration cancelled before display",
                            "usage": usage,
                        }
                    )
                return interrupted
            if not empty:
                break
            # The model produced no visible narration (empty or hidden-only
            # reply). Drop the attempt, nudge it, and try again once. The
            # dropped call was still billed, so keep its real usage.
            if usage is not None:
                self.record_external_cost_event(
                    {
                        "service": "gemini",
                        "category": "discarded_narration",
                        "label": "Gemini narration attempt without visible text",
                        "usage": usage,
                    }
                )
            attempts += 1
            if attempts >= 2:
                raise RuntimeError("model returned no visible narration twice")
            self.record_engine_reminder(EMPTY_REPLY_REMINDER)

        estimated = usage is None
        if estimated:
            prompt_est = self.estimate_context_tokens()
            usage = {
                "prompt": prompt_est,
                "cached": 0,
                "output": estimate_tokens(raw),
                "thoughts": 0,
            }
        cost = usage_cost(usage)
        self.context_tokens = usage["prompt"] + usage["output"]

        self.history.append(types.Content(role="model", parts=[types.Part(text=raw)]))
        self.session.append(
            {
                "type": "narrator",
                "label": label,
                "text": raw,
                "interrupted": interrupted,
                "usage": usage,
                "cost": cost,
                "estimated": estimated,
                "context_tokens": self.context_tokens,
            }
        )
        reminders = []
        if contains_cardinal_direction(raw):
            reminders.append(CARDINAL_DIRECTION_REMINDER)
        if last_visible_line_is_question(raw):
            reminders.append(QUESTION_ENDING_REMINDER)
        if story_progress_missing_streak(self.session.events) >= 2:
            reminders.append(STORY_PROGRESS_REMINDER)
        if reminders:
            self.record_engine_reminder("\n\n".join(reminders))
        if self.show_costs:
            emit_notice(
                fmt_cost_line(
                    f"{label} cost", usage, cost, self.session.total_cost(), estimated
                ),
                emit=emit,
            )
        try:
            self.maybe_compact(emit=emit)
        except Exception as exc:
            # Compaction is best-effort and separate from this turn, which is
            # already recorded and billed. A failure here must NOT propagate
            # into the turn-level rollback (that would truncate this turn and
            # the player's input). It will be retried next turn while the
            # threshold is still exceeded.
            emit_notice(f"[compaction deferred: {exc}]", emit=emit)
        return interrupted

    def player_turn(
        self,
        text: str,
        kind: str = "player",
        shown: str | None = None,
        append_to_previous_echo: bool = False,
    ) -> bool:
        self.record_player_turn(
            text,
            kind=kind,
            shown=shown,
            append_to_previous_echo=append_to_previous_echo,
        )
        return self.narrator_turn()

    def record_player_turn(
        self,
        text: str,
        kind: str = "player",
        shown: str | None = None,
        append_to_previous_echo: bool = False,
    ) -> None:
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        event = {"type": kind, "text": text}
        if shown is not None and shown != text:
            event["shown"] = shown
        if append_to_previous_echo:
            event["append_to_previous_echo"] = True
        self.session.append(event)

    def coalesced_player_text(self, events_len: int, new_text: str) -> str:
        for e in reversed(self.session.events[events_len:]):
            if e["type"] in ("player", "interruption"):
                shown = e.get("shown")
                if shown is None:
                    shown = LEADING_ENGINE_NOTE_RE.sub("", e["text"], count=1)
                shown = shown.strip()
                if shown:
                    return shown.rstrip() + " " + new_text.lstrip()
                break
        return new_text

    def record_narration_display(self, text: str) -> None:
        """Record how much of the last narration was actually shown/played
        before it was cut short; replay displays this instead of the full
        text. Display-only — the model context keeps the full narration."""
        last_narrator = None
        last_display = None
        for i, e in enumerate(self.session.events):
            if e["type"] == "narrator":
                last_narrator = i
            elif e["type"] == "narration_display":
                last_display = i
        if last_narrator is None:
            return
        if last_display is not None and last_display > last_narrator:
            return
        self.session.append({"type": "narration_display", "text": text})

    def record_engine_reminder(self, text: str) -> None:
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        self.session.append({"type": "engine_reminder", "text": text})
        self.context_tokens += estimate_tokens(text)

    def restore_snapshot(self, history_len: int, events_len: int) -> None:
        self.history = self.history[:history_len]
        with self.session.lock:
            preserved_costs = [
                e for e in self.session.events[events_len:]
                if e.get("type") == "external_cost"
            ]
            self.session.events = self.session.events[:events_len] + preserved_costs
            _rewrite_session_file(self.session)

    def history_len_for_event_prefix(self, events_len: int) -> int:
        events = self.session.events[:events_len]
        last_compaction = None
        last_compaction_idx = None
        for i, e in enumerate(events):
            if e["type"] == "compaction":
                last_compaction = e
                last_compaction_idx = i
        count = 1
        if last_compaction is not None:
            count += 1
            start = last_compaction.get("retain_start", last_compaction_idx + 1)
        else:
            start = 0
        for e in events[start:]:
            if e["type"] in ("player", "interruption", "engine_reminder"):
                count += 1
            elif e["type"] == "narrator" and e.get("text", "").strip():
                count += 1
        return count

    def pending_player_cancel_snapshot(self) -> tuple[int, int] | None:
        for i in range(len(self.session.events) - 1, -1, -1):
            e = self.session.events[i]
            if e["type"] in ("player", "interruption"):
                return self.history_len_for_event_prefix(i), i
            if e["type"] == "narrator":
                return None
        return None

    # ── compaction ───────────────────────────────────────────────────────

    def maybe_compact(self, emit=None) -> None:
        self.refresh_system_prompt(reestimate_context=True)
        if self.context_tokens < self.compact_at:
            return
        trigger_context_tokens = self.context_tokens
        emit_notice(
            f"\n[Context reached {self.context_tokens:,} tokens (≥ {self.compact_at:,}) — "
            "compacting story memory. This cannot be interrupted…]",
            emit=emit,
        )
        # Keep the most recent turns verbatim, outside the summary: they are the
        # trailing messages of self.history (build_history emits them last), so
        # we summarize everything up to the cutoff and replay the rest as-is.
        retain_turns = max(0, int(env_float("IF_ENGINE_COMPACT_RETAIN_TURNS", COMPACT_RETAIN_TURNS)))
        retain_start = compaction_retain_start(self.session.events, retain_turns)
        retained_events = self.session.events[retain_start:]
        retain_msg_count = sum(
            1 for e in retained_events
            if e["type"] in ("player", "interruption", "engine_reminder")
            or (e["type"] == "narrator" and e.get("text", "").strip())
        )
        retain_msg_count = min(retain_msg_count, len(self.history) - 1)
        summarize_history = (
            self.history[:-retain_msg_count] if retain_msg_count else list(self.history)
        )
        retained_text = "".join(
            (p.text or "")
            for c in (self.history[-retain_msg_count:] if retain_msg_count else [])
            for p in c.parts
        )
        contents = summarize_history + [
            types.Content(role="user", parts=[types.Part(text=COMPACTION_PROMPT)])
        ]
        attempts_done = 1
        while True:
            try:
                response = self.client.models.generate_content(
                    model=MODEL, contents=contents, config=self._gen_config()
                )
                break
            except Exception as exc:
                if should_retry_gemini(exc, attempts_done, emit):
                    attempts_done += 1
                    continue
                raise
        summary = response.text or ""
        if not summary.strip():
            raise RuntimeError("compaction produced an empty summary")
        usage = usage_to_dict(response.usage_metadata)
        cost = usage_cost(usage)

        self.session.append(
            {
                "type": "compaction",
                "summary": summary,
                "retain_start": retain_start,
                "usage": usage,
                "cost": cost,
                "trigger_context_tokens": trigger_context_tokens,
                "threshold": self.compact_at,
                "context_tokens": estimate_tokens(self.system_prompt)
                + estimate_tokens(self.transcript_message)
                + estimate_tokens(summary)
                + estimate_tokens(retained_text),
            }
        )
        self.history = build_history(self.session, self.transcript_message)
        self.context_tokens = self.session.last_context_tokens()
        if self.show_costs:
            emit_notice(
                fmt_cost_line("compaction cost", usage, cost, self.session.total_cost(), False),
                emit=emit,
            )
        emit_notice("[Compaction complete — story memory rebuilt.]\n", emit=emit)

    # ── replay / playback ────────────────────────────────────────────────

    def replay_text(self, live_prompt: bool = False) -> str:
        parts = [session_header(self.session.name, self.game_title, self.fast_mode) + "\n"]
        if IS_TTY:
            if live_prompt:
                for line in self.live_prompt_instruction_lines():
                    parts.append(line + "\n")
                parts.append("\n")
            else:
                parts.append("Type at the prompt; /quit to leave.\n\n")
        running_total = 0.0
        narrator_count = 0
        last_context_tokens = None
        last_narrator_part = None
        for e in self.session.events:
            if e["type"] in ("player", "interruption"):
                shown = e.get("shown")
                if shown is None:
                    shown = LEADING_ENGINE_NOTE_RE.sub("", e["text"], count=1)
                if e.get("append_to_previous_echo"):
                    for i in range(len(parts) - 1, -1, -1):
                        if parts[i].startswith("\n❯ ") and parts[i].endswith("\n\n"):
                            parts[i] = parts[i].rstrip() + " " + shown + "\n\n"
                            break
                    else:
                        parts.append(f"\n❯ {shown}\n\n")
                else:
                    parts.append(f"\n❯ {shown}\n\n")
            elif e["type"] == "narration_display":
                # The narration was cut short; show only what was played.
                if last_narrator_part is not None:
                    shown = (e.get("text") or "").rstrip()
                    parts[last_narrator_part] = shown + "\n" if shown else ""
                    last_narrator_part = None
            elif e["type"] == "narrator":
                narrator_count += 1
                label = e.get("label") or ("opening" if narrator_count == 1 else "turn")
                visible = visible_markup_text(
                    e["text"], color_resolver=self.character_display_color
                )
                if not visible.endswith("\n"):
                    visible += "\n"
                last_narrator_part = len(parts)
                parts.append(visible)
                running_total += e.get("cost", 0.0)
                last_context_tokens = e.get("context_tokens")
                if self.show_costs and not e.get("local"):
                    parts.append(
                        system_markup(
                            fmt_cost_line(
                                f"{label} cost",
                                e["usage"],
                                e.get("cost", 0.0),
                                running_total,
                                e.get("estimated", False),
                            )
                        )
                        + "\n"
                    )
            elif e["type"] == "compaction":
                trigger_context_tokens = e.get("trigger_context_tokens", last_context_tokens)
                trigger_text = (
                    f"{trigger_context_tokens:,}"
                    if isinstance(trigger_context_tokens, int)
                    else "unknown"
                )
                threshold = e.get("threshold")
                threshold_text = f"{threshold:,}" if isinstance(threshold, int) else "unknown"
                parts.append(
                    "\n"
                    + system_markup(
                        f"[Context reached {trigger_text} "
                        f"tokens (≥ {threshold_text}) — "
                        "compacting story memory. This cannot be interrupted…]"
                    )
                    + "\n"
                )
                running_total += e["cost"]
                if self.show_costs:
                    parts.append(
                        system_markup(
                            fmt_cost_line(
                                "compaction cost",
                                e["usage"],
                                e["cost"],
                                running_total,
                                False,
                            )
                        )
                        + "\n"
                    )
                parts.append(
                    system_markup("[Compaction complete — story memory rebuilt.]") + "\n\n"
                )
                last_context_tokens = e.get("context_tokens")
        return "".join(parts)

    def live_prompt_instruction_lines(self) -> list[str]:
        if not self.voice_enabled:
            return [
                "No voice: type at any time and press Enter to send; /quit to leave.",
            ]
        lines = [
            "Type at any time and press Enter to send; /quit to leave.",
        ]
        if self.prompt_wake_word_enabled:
            lines.append('Voice input: say "Okay" and then speak, or hold Shift while speaking.')
        else:
            lines.append("Voice input: hold Shift while speaking.")
        lines.append("Caps Lock toggles open mic until you turn it off.")
        return lines

    def replay(self, live_prompt: bool = False) -> None:
        # Plain (non-UI) output: drop the inline color markup, keep the text.
        print(
            strip_color_markup(self.replay_text(live_prompt=live_prompt)),
            end="",
            flush=True,
        )

    # ── main loop ────────────────────────────────────────────────────────

    def play_startup_acknowledgement(self) -> None:
        """Use the same click as accepted voice input when startup sends a
        request to the model before the player has typed or spoken."""
        if self.voice is not None:
            self.voice.play_confirm_cue()

    def run(self) -> None:
        try:
            if IS_TTY and Application is not None:
                asyncio.run(self._run_live_prompt())
            else:
                if IS_TTY and Application is None:
                    notice("[prompt-toolkit is not installed; live input while streaming is "
                           "disabled. Run: .venv/bin/pip install prompt-toolkit]")
                self._run_blocking_prompt()
        finally:
            self.emit_exit_cost_summary()

    async def _run_live_prompt(self) -> None:
        ui = TerminalUI()
        input_queue: asyncio.Queue = asyncio.Queue()
        startup_exit_event = asyncio.Event()
        startup_loading = False
        job = None
        job_task = None
        job_displayed_text = False
        pending_pre_display_cancel_snapshot = None
        input_task = None

        def submit(text: str) -> None:
            input_queue.put_nowait(text)
            if startup_loading and text.strip().lower() in ("/quit", "/exit", "/q"):
                startup_exit_event.set()

        ui.submit_callback = submit

        def mark_displayed(text: str) -> None:
            nonlocal job_displayed_text, pending_pre_display_cancel_snapshot
            shown = strip_display_markup(text).strip()
            if shown and not shown.startswith(("─ ", "[")):
                job_displayed_text = True
                pending_pre_display_cancel_snapshot = None

        def display_narration(text: str) -> None:
            mark_displayed(text)
            ui.append(text)

        def attach_voice_callbacks() -> None:
            if self.voice is None:
                return
            loop = asyncio.get_running_loop()

            def on_voice_transcript(
                text: str, heard_text: str, displayed_text: str, was_speaking: bool
            ) -> None:
                loop.call_soon_threadsafe(
                    input_queue.put_nowait,
                    VoiceUtterance(text, heard_text, displayed_text, was_speaking),
                )

            self.voice.on_transcript = on_voice_transcript
            self.voice.on_notice = lambda message: ui.append(
                system_markup(f"[{message}]") + "\n"
            )
            self.voice.on_vad = lambda active: ui.set_input_prompt("🔴 " if active else "❯ ")
            self.voice.on_display = display_narration

        async def start_voice_if_needed(app_task) -> bool:
            nonlocal startup_loading
            if not self.voice_enabled or self.voice is not None:
                attach_voice_callbacks()
                return True
            voice = None
            done = threading.Event()
            result: dict[str, object] = {}
            startup_loading = True
            try:
                from voice import VoiceConfig, VoiceLoop

                voice_config = VoiceConfig(
                    whisper_prompt=build_whisper_proper_noun_prompt(
                        self.transcript_text, self.game_title
                    ),
                    omnivoice_character_voice_transcript_stem=self.game_name or "default",
                    omnivoice_character_voice_transcript_filename=(
                        f"{self.game_name}.txt" if self.game_name else "default.txt"
                    ),
                    omnivoice_character_voice_transcript_text=self.transcript_text,
                    omnivoice_character_voice_game_title=self.game_title,
                    external_cost_recorder=self.record_external_cost_event,
                )
                apply_voice_runtime_options(voice_config, self.voice_options)
                ui.start_spinner("Loading voice models...", blank_before=True)
                voice = VoiceLoop(config=voice_config)
                voice.on_notice = lambda _message: None
                result["voice"] = voice

                def run_startup() -> None:
                    try:
                        voice.start()
                    except BaseException as exc:
                        result["error"] = exc
                    finally:
                        done.set()

                threading.Thread(
                    target=run_startup,
                    name="if-engine-voice-startup",
                    daemon=True,
                ).start()
                while not done.is_set():
                    if startup_exit_event.is_set() or app_task.done():
                        threading.Thread(
                            target=voice.close,
                            name="if-engine-voice-startup-close",
                            daemon=True,
                        ).start()
                        ui.stop_spinner()
                        if not app_task.done():
                            ui.exit()
                        return False
                    await asyncio.sleep(0.05)
                if "error" in result:
                    raise result["error"]  # type: ignore[misc]
                self.voice = voice
                actual_wake_word_enabled = getattr(voice, "wake_gate", None) is not None
                self.voice_options["wake_word_enabled"] = actual_wake_word_enabled
                self.refresh_system_prompt(
                    voice_prompt_enabled=True,
                    tts_engine=voice.config.tts_engine,
                    wake_word_enabled=actual_wake_word_enabled,
                    car_mode=voice.config.car_mode,
                    wake_threshold=voice.config.openwakeword_threshold,
                    wake_preprocess=voice.config.wake_word_preprocess,
                    omnivoice_whisper_tags=voice.config.omnivoice_whisper_tags,
                    omnivoice_character_voices=(
                        voice.config.omnivoice_character_voices_enabled
                    ),
                    reestimate_context=True,
                )
                attach_voice_callbacks()
                ui.set_text(self.replay_text(live_prompt=True))
            except Exception as exc:
                self.voice = None
                self.voice_enabled = False
                self.refresh_system_prompt(
                    voice_prompt_enabled=False,
                    tts_engine="none",
                    wake_word_enabled=False,
                    car_mode=False,
                    wake_threshold=None,
                    wake_preprocess=False,
                    omnivoice_whisper_tags=False,
                    omnivoice_character_voices=False,
                    reestimate_context=True,
                )
                ui.set_text(self.replay_text(live_prompt=True))
                ui.append(system_markup(f"[voice disabled: {exc}]") + "\n")
            finally:
                startup_loading = False
                ui.stop_spinner()
            return True

        def start_job(label: str, rollback_snapshot=None, cancel_snapshot=None) -> None:
            nonlocal job, job_task, job_displayed_text, pending_pre_display_cancel_snapshot
            job_displayed_text = False
            pending_pre_display_cancel_snapshot = None
            ui.start_spinner()
            job = NarrationJob(
                self,
                label,
                rollback_snapshot=rollback_snapshot,
                cancel_snapshot=cancel_snapshot,
                flush_chunks=False,
                emit=display_narration,
            )
            job.start()
            job_task = asyncio.create_task(asyncio.to_thread(job.wait))

        async def collect_job() -> tuple[bool, bool, tuple[int, int] | None]:
            nonlocal job, job_task, pending_pre_display_cancel_snapshot
            if job is None or job_task is None:
                return False, False, None
            current_job = job
            current_task = job_task
            job = None
            job_task = None
            try:
                interrupted = await current_task
                displayed = job_displayed_text
                # In voice mode the model can finish streaming before the
                # first sentence has played; keep the spinner until that
                # first displayed sentence replaces it (ui.append does this).
                if (
                    self.voice is not None
                    and not interrupted
                    and not displayed
                    and current_job.cancel_snapshot is not None
                ):
                    pending_pre_display_cancel_snapshot = current_job.cancel_snapshot
                if self.voice is None or not self.voice.is_speaking():
                    ui.stop_spinner()
                return interrupted, displayed, current_job.cancel_snapshot
            except Exception as exc:
                displayed = job_displayed_text
                ui.stop_spinner()
                if current_job.rollback_snapshot is not None:
                    self.restore_snapshot(*current_job.rollback_snapshot)
                    ui.append(
                        system_markup(
                            f"[error talking to the model: {exc} — your last input "
                            "was not recorded; please try again]"
                        )
                        + "\n"
                    )
                else:
                    ui.append(system_markup(f"[error talking to the model: {exc}]") + "\n")
                return False, displayed, None

        ui.set_text(self.replay_text(live_prompt=True))

        app_task = asyncio.create_task(ui.run_async())
        if not await start_voice_if_needed(app_task):
            if not app_task.done():
                ui.exit()
            await app_task
            return
        input_task = asyncio.create_task(input_queue.get())

        if self.needs_opening():
            # New playthrough: ask the model for the game's opening page.
            self.play_startup_acknowledgement()
            start_job(
                "opening",
                rollback_snapshot=(len(self.history), len(self.session.events)),
            )
        elif self.pending_player_turn():
            # Resumed with unanswered player input: answer it now.
            self.play_startup_acknowledgement()
            rollback_snapshot = (len(self.history), len(self.session.events))
            start_job(
                "turn",
                rollback_snapshot=rollback_snapshot,
                cancel_snapshot=self.pending_player_cancel_snapshot() or rollback_snapshot,
            )

        try:
            while True:
                wait_for = [app_task, input_task]
                if job_task is not None:
                    wait_for.append(job_task)
                done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

                if app_task in done:
                    if job is not None:
                        job.interrupt()
                        await collect_job()
                    return

                if job_task is not None and job_task in done:
                    await collect_job()
                    continue

                if input_task not in done:
                    continue

                item = input_task.result()
                input_task = asyncio.create_task(input_queue.get())

                if isinstance(item, VoiceUtterance):
                    text = item.text.strip()
                    from_voice = True
                    voice_was_speaking = item.was_speaking
                    voice_heard = item.heard_text
                    narration_truncated = item.was_speaking
                    voice_displayed = item.displayed_text
                else:
                    text = item.strip()
                    from_voice = False
                    voice_was_speaking = False
                    voice_heard = ""
                    narration_truncated = False
                    voice_displayed = ""

                if not text:
                    continue

                if not from_voice and self.voice is not None:
                    # Typed input silences any read-aloud in progress; the
                    # player has the displayed text on screen, so no heard-note.
                    if self.voice.is_speaking():
                        narration_truncated = True
                        voice_displayed = self.voice.displayed_text()
                    self.voice.cancel_utterance()

                interrupted = False
                interrupted_before_display = False
                interrupted_rollback = None
                if job is not None:
                    job.interrupt()
                    interrupted, displayed_before_interrupt, interrupted_rollback = await collect_job()
                    interrupted_before_display = interrupted and not displayed_before_interrupt

                command = text.lower()
                pre_display_cancel_snapshot = (
                    interrupted_rollback
                    if interrupted_before_display
                    else pending_pre_display_cancel_snapshot
                )
                coalesce_cancelled_turn = (
                    pre_display_cancel_snapshot is not None and not command.startswith("/")
                )
                if coalesce_cancelled_turn:
                    model_text = self.coalesced_player_text(pre_display_cancel_snapshot[1], text)
                    self.restore_snapshot(*pre_display_cancel_snapshot)
                    pending_pre_display_cancel_snapshot = None
                    kind = "player"
                    shown = model_text
                else:
                    if narration_truncated:
                        self.record_narration_display(voice_displayed)
                    kind = (
                        "interruption"
                        if interrupted or (from_voice and voice_was_speaking)
                        else "player"
                    )
                    model_text = text
                    if from_voice and voice_was_speaking:
                        model_text = voice_interruption_preamble(voice_heard) + text
                    elif interrupted:
                        model_text = INTERRUPTION_PREAMBLE + text
                    shown = text

                append_to_previous_echo = coalesce_cancelled_turn
                ui.append_input_echo(text, append_to_previous=append_to_previous_echo)

                if command in ("/quit", "/exit", "/q"):
                    self.emit_exit_cost_summary(emit=ui.append)
                    ui.append(
                        system_markup(f"[Session '{self.session.name}' saved. Goodbye.]")
                        + "\n"
                    )
                    ui.exit()
                    if not app_task.done():
                        await app_task
                    return
                if command == "/cost":
                    ui.append(
                        system_markup(
                            self.cost_summary_text(include_context=True).rstrip("\n")
                        )
                        + "\n"
                    )
                    continue

                snapshot = (len(self.history), len(self.session.events))
                if from_voice and self.voice is not None:
                    # Click only now: the transcription is being accepted as
                    # actual player input (not noise, not a dropped command).
                    self.voice.play_confirm_cue()
                self.record_player_turn(
                    model_text,
                    kind=kind,
                    shown=shown,
                )
                start_job("turn", rollback_snapshot=snapshot)
        finally:
            if input_task is not None and not input_task.done():
                input_task.cancel()
            if not app_task.done():
                ui.exit()

    def _run_blocking_prompt(self) -> None:
        self.replay(live_prompt=False)
        if self.needs_opening() or self.pending_player_turn():
            label = "opening" if self.needs_opening() else "turn"
            snapshot = (len(self.history), len(self.session.events))
            try:
                self.play_startup_acknowledgement()
                self.narrator_turn(label=label)
            except Exception as exc:
                self.restore_snapshot(*snapshot)
                notice(f"[error talking to the model: {exc}]")
        interrupted = False

        while True:
            if interrupted:
                prompt = cyan("\n⟪ interrupted — what do you want to say or change? ⟫\n❯ ")
                kind = "interruption"
            else:
                prompt = cyan("\n❯ ")
                kind = "player"
            try:
                text = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.emit_exit_cost_summary()
                notice(f"[Session '{self.session.name}' saved. Goodbye.]")
                return
            if not text:
                interrupted = False
                continue
            if text.lower() in ("/quit", "/exit", "/q"):
                print()
                self.emit_exit_cost_summary()
                notice(f"[Session '{self.session.name}' saved. Goodbye.]")
                return
            if text.lower() == "/cost":
                print(self.cost_summary_text(include_context=True), end="")
                continue
            print()
            model_text = text
            if kind == "interruption":
                model_text = INTERRUPTION_PREAMBLE + text
            snapshot = (len(self.history), len(self.session.events))
            try:
                interrupted = self.player_turn(model_text, kind=kind, shown=text)
            except Exception as exc:
                self.restore_snapshot(*snapshot)
                notice(f"[error talking to the model: {exc} — your last input was "
                       "not recorded; please try again]")
                interrupted = False


def _rewrite_session_file(session: Session) -> None:
    with session.lock:
        SESSIONS_DIR.mkdir(exist_ok=True)
        tmp = session.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in session.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.replace(session.path)


# ── Entry point ──────────────────────────────────────────────────────────────

def list_sessions() -> None:
    if not SESSIONS_DIR.exists():
        print("no sessions yet")
        return
    for path in sorted(SESSIONS_DIR.glob("*.jsonl")):
        try:
            s = Session(path.stem)
            turns = sum(1 for e in s.events if e.get("type") == "narrator")
            print(f"{path.stem:24s} {s.game():28s} {turns:4d} turns   ${s.total_cost():.4f}")
        except Exception as exc:
            print(f"warning: skipping unreadable session {path.name}: {exc}", file=sys.stderr)


def list_games() -> None:
    games = available_games()
    if not games:
        print(f"no games in {CATALOG_PATH} and no local transcripts in {TRANSCRIPTS_DIR}/")
        return
    index = load_game_index()
    for name in games:
        info = index.get(name) or {}
        title = info.get("title") or name.replace("-", " ").title()
        author = info.get("author", "")
        details = title + (f" by {author}" if author else "")
        print(f"{name:44s} {details}")


def show_config() -> None:
    """Print the effective engine configuration and any IF_ENGINE_* overrides.

    Intentionally lightweight: it reports the engine settings and the environment
    overrides currently in effect, without importing the heavy voice stack."""
    fast = os.environ.get("IF_ENGINE_FAST_MODE", "").strip().lower() not in (
        "", "0", "false", "no", "off")
    tts = os.environ.get("IF_ENGINE_TTS_ENGINE", "omnivoice").strip().lower() or "omnivoice"
    print("Ferrytale configuration")
    print(f"  Model:            {MODEL}")
    print(f"  Thinking level:   {THINKING_LEVEL}")
    print(f"  Fast mode:        {'on' if fast else 'off'}  (--fast-mode / IF_ENGINE_FAST_MODE)")
    retain_turns = max(0, int(env_float("IF_ENGINE_COMPACT_RETAIN_TURNS", COMPACT_RETAIN_TURNS)))
    print(f"  Compaction:       {COMPACT_MARGIN_TOKENS:,} tokens past system prompt + transcript"
          f"  (keeps last {retain_turns} turns verbatim / IF_ENGINE_COMPACT_RETAIN_TURNS)")
    print(f"  TTS engine:       {tts}  (./play --kokoro/--omnivoice / IF_ENGINE_TTS_ENGINE)")
    print(f"  Cost per 1M tok:  fresh ${PRICE_IN_PER_M:.2f} / cached "
          f"${PRICE_IN_CACHED_PER_M:.2f} / output+thinking ${PRICE_OUT_PER_M:.2f}")
    overrides = sorted((k, v) for k, v in os.environ.items() if k.startswith("IF_ENGINE_"))
    print()
    if overrides:
        print("  IF_ENGINE_* set in your environment:")
        for key, value in overrides:
            print(f"    {key}={value}")
    else:
        print("  No IF_ENGINE_* overrides set (using defaults).")
    print()
    print("  Configure with environment variables or .env — see .env.example and the README.")


def next_session_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"session-{stamp}"
    name = base
    i = 2
    while (SESSIONS_DIR / f"{name}.jsonl").exists():
        name = f"{base}-{i}"
        i += 1
    return name


def latest_session_name():
    if not SESSIONS_DIR.exists():
        return None
    paths = [p for p in SESSIONS_DIR.glob("*.jsonl") if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime).stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Ferrytale transcript-based Interactive Fiction AI "Interpreter"'
    )
    parser.add_argument("session", nargs="?", default=None,
                        help="session name (default: newest saved session, or generated with --new)")
    parser.add_argument("--new", "--new-session", action="store_true",
                        help="start a fresh session; fail if the named session exists")
    parser.add_argument("--game", default=None,
                        help="game to play, by catalog slug "
                             "(required for new sessions — sessions keep their game)")
    parser.add_argument("--list", action="store_true", help="list saved sessions")
    parser.add_argument("--list-games", action="store_true",
                        help="list catalogued games")
    parser.add_argument("--show-costs", action="store_true",
                        help="show per-turn token/cost status lines")
    parser.add_argument("--compact-at", type=int, default=None,
                        help="override the compaction threshold (testing)")
    parser.add_argument("--fast-mode", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_FAST_MODE,
                        help="use Gemini priority service tier for lower latency "
                             "(off by default; can also set IF_ENGINE_FAST_MODE=1)")
    parser.add_argument("--voice", action=argparse.BooleanOptionalAction, default=True,
                        help="read narration aloud with the configured TTS engine "
                             "and accept spoken input (Silero VAD + WebRTC AEC + "
                             "whisper.cpp); live terminal only")
    parser.add_argument("--wake-word", action=argparse.BooleanOptionalAction, default=None,
                        help="enable or disable the Okay wake word "
                             "(default: disabled, unless car mode is enabled)")
    parser.add_argument("--car-mode", action=argparse.BooleanOptionalAction, default=None,
                        help="enable car/Bluetooth wake-word defaults "
                             "(wake word on, lower threshold, wake preprocessing)")
    parser.add_argument("--wake-word-threshold", type=float, default=None,
                        help="openWakeWord activation threshold "
                             "(default: 0.9, or 0.4 in car mode)")
    parser.add_argument("--wake-word-preprocess", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="enable or disable realtime wake preprocessing "
                             "(high-pass, pre-emphasis, AGC)")
    parser.add_argument("--wake-word-model", action="append", default=[],
                        help="openWakeWord model path/name; repeat for multiple models")
    parser.add_argument("--whisper-tags", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="enable or disable OmniVoice <whisper> style tags "
                             "(default: disabled; env IF_ENGINE_OMNIVOICE_WHISPER_TAGS=1)")
    parser.add_argument("--kokoro", action="store_true",
                        help="use the Kokoro TTS engine for this run "
                             "(same as IF_ENGINE_TTS_ENGINE=kokoro)")
    parser.add_argument("--omnivoice", action="store_true",
                        help="use the OmniVoice TTS engine for this run (the default)")
    parser.add_argument("--show-config", action="store_true",
                        help="print the effective configuration and exit")
    args = parser.parse_args()

    if args.kokoro:
        os.environ["IF_ENGINE_TTS_ENGINE"] = "kokoro"
    elif args.omnivoice:
        os.environ["IF_ENGINE_TTS_ENGINE"] = "omnivoice"

    if args.show_config:
        show_config()
        return

    if args.list:
        list_sessions()
        return
    if args.list_games:
        list_games()
        return

    # Every path beyond here talks to Gemini. Fail fast on a missing key now —
    # before the (possibly networked) transcript download — with the clear
    # message from require_gemini_api_key(). Game.__init__ also checks it.
    require_gemini_api_key()

    if args.game and not args.new:
        sys.exit("--game can only be used with --new; resumed sessions keep their game")

    session_name = args.session or latest_session_name()
    if session_name is None and not args.new:
        sys.exit(
            "no saved sessions yet. Start one with: "
            "python ferrytale.py --new --game <game>  (or run ./play <game>)"
        )
    if args.new:
        session_name = args.session or next_session_name()
        if (SESSIONS_DIR / f"{session_name}.jsonl").exists():
            sys.exit(
                f"session already exists: {session_name}. "
                "Choose another name or omit the name to generate one."
            )

    session = Session(session_name)
    # Interactive play only: guard against two processes opening the same
    # session and clobbering each other via the whole-file rewrite.
    session.lock_for_play()
    session_was_new = session.is_new
    game_name = (args.game or DEFAULT_GAME) if session.is_new else session.game()
    if session.is_new and game_name is None:
        sys.exit("--game is required when starting a new session")

    try:
        transcript_text = load_transcript_text(game_name)
    except RuntimeError as exc:
        sys.exit(str(exc))
    if session.is_new:
        session.append({"type": "meta", "game": game_name})

    voice_options = resolve_voice_runtime_options(args)
    voice = None
    game = None

    def close_voice_best_effort(active_voice, timeout: float = 3.0) -> None:
        if active_voice is None:
            return
        done = threading.Event()

        def close() -> None:
            try:
                active_voice.close()
            except BaseException:
                pass
            finally:
                done.set()

        threading.Thread(
            target=close,
            name="if-engine-voice-final-close",
            daemon=True,
        ).start()
        done.wait(timeout)

    try:
        game = Game(
            session,
            transcript_text,
            show_costs=args.show_costs,
            voice=voice,
            voice_enabled=args.voice and IS_TTY and Application is not None,
            game_name=game_name,
            game_title=game_title_for(game_name),
            compact_at=args.compact_at,
            fast_mode=args.fast_mode,
            voice_options=voice_options,
            resumed_session=not session_was_new,
        )
        game.run()
    except KeyboardInterrupt:
        print()
        notice(f"[Session '{session_name}' saved. Goodbye.]")
    finally:
        if voice is None and game is not None:
            voice = game.voice
        close_voice_best_effort(voice)
    # All session writes are fsync'd as they happen, and CoreAudio/PortAudio
    # teardown can wedge the interpreter at exit — leave decisively so Ctrl-C
    # and /quit always actually end the process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
