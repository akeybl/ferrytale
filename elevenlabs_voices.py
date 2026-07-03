"""ElevenLabs-designed character voice cache for OmniVoice."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ELEVENLABS_VOICE_CACHE_DIR = BASE_DIR / ".cache" / "elevenlabs-voices"
ELEVENLABS_VOICE_DESIGN_URL = "https://api.elevenlabs.io/v1/text-to-voice/design"
ELEVENLABS_PREVIEW_TEXT = (
    "Prosecutors have opened a massive investigation into allegations of "
    "fixing games and illegal betting."
)
ELEVENLABS_VOICE_DESIGN_QUALITY = 1
DEFAULT_DESCRIPTION_MODEL = "gemini-flash-latest"
AUDIO_QUALITY_DESCRIPTION = (
    "Perfect audio quality, studio-quality recording, broadcast quality, "
    "full-band microphone capture with rich low end and open high frequencies."
)
DEGRADED_AUDIO_TERMS_RE = re.compile(
    r"\b(?:phone|telephone|radio|intercom|walkie[- ]?talkie|tape|tinny|"
    r"muffled|narrowband|band[- ]?limited)\b",
    re.IGNORECASE,
)
ACCENT_MARKER_RE = re.compile(
    r"\b(?:accent|dialect|pronunciation|received pronunciation|r\.?p\.?|"
    r"british|american|english|irish|scottish|welsh|australian|canadian|"
    r"french|german|spanish|italian|russian|native|non-native|regional)\b",
    re.IGNORECASE,
)
OMNIVOICE_GENDERS = ("male", "female")
OMNIVOICE_AGES = ("child", "teenager", "young adult", "middle-aged", "elderly")
OMNIVOICE_PITCHES = (
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "very high pitch",
    "high pitch",
)
OMNIVOICE_STYLES = ("whisper",)
OMNIVOICE_ENGLISH_ACCENTS = (
    "american accent",
    "british accent",
    "australian accent",
    "canadian accent",
    "indian accent",
    "chinese accent",
    "korean accent",
    "japanese accent",
    "portuguese accent",
    "russian accent",
)
OMNIVOICE_ACCENT_ALIASES = {
    "received pronunciation": "british accent",
    "r.p.": "british accent",
    "rp": "british accent",
    "english accent": "british accent",
}

# ── Terminal display colors ──────────────────────────────────────────────────
# The engine colors system messages and the input chevron in this blue;
# character voice colors are chosen at voice-design time to stay clearly
# distinct from it and from every other cached voice. Colored spans travel
# through the display pipeline as inline markup that the terminal UI renders
# and every plain-text path strips.

SYSTEM_DISPLAY_COLOR = "#5f87ff"

DISPLAY_COLOR_CLOSE = "⟦/fg⟧"
DISPLAY_COLOR_TOKEN_RE = re.compile(r"⟦fg=#[0-9a-fA-F]{6}⟧|⟦/fg⟧")
HEX_COLOR_RE = re.compile(r"\A#[0-9a-fA-F]{6}\Z")


def color_markup(text: str, color: str | None) -> str:
    """Wrap text in inline display-color markup (no-op without a valid color)."""
    if not text or not color or not HEX_COLOR_RE.match(color):
        return text
    return f"⟦fg={color.lower()}⟧{text}{DISPLAY_COLOR_CLOSE}"


def strip_color_markup(text: str) -> str:
    return DISPLAY_COLOR_TOKEN_RE.sub("", text)


def _hue_to_hex(hue_degrees: float, saturation: float = 0.62, lightness: float = 0.62) -> str:
    import colorsys

    r, g, b = colorsys.hls_to_rgb((hue_degrees % 360.0) / 360.0, lightness, saturation)
    return "#{:02x}{:02x}{:02x}".format(
        round(r * 255), round(g * 255), round(b * 255)
    )


def _hex_to_hue(color: str) -> float | None:
    if not color or not HEX_COLOR_RE.match(color):
        return None
    import colorsys

    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    if saturation == 0:
        return None
    return hue * 360.0


def _hue_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _name_hue(character_name: str) -> float:
    digest = hashlib.sha256(normalize_character_name(character_name).encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "big") % 360


def fallback_display_color(character_name: str) -> str:
    """Deterministic color for cached voices that predate stored colors."""
    hue = _name_hue(character_name)
    system_hue = _hex_to_hue(SYSTEM_DISPLAY_COLOR) or 0.0
    if _hue_distance(hue, system_hue) < 30.0:
        hue = (hue + 60.0) % 360.0
    return _hue_to_hex(hue)


def pick_display_color(existing_colors: list[str], character_name: str) -> str:
    """Choose a hue as far as possible from the system blue and every existing
    voice color. Deterministic: candidates rotate from a name-derived hue."""
    taken = [_hex_to_hue(SYSTEM_DISPLAY_COLOR) or 0.0]
    for color in existing_colors:
        hue = _hex_to_hue(color)
        if hue is not None:
            taken.append(hue)
    base = _name_hue(character_name)
    best_hue = float(base)
    best_score = -1.0
    for step in range(24):
        candidate = (base + step * 15.0) % 360.0
        score = min(_hue_distance(candidate, hue) for hue in taken)
        if score > best_score + 1e-9:
            best_score = score
            best_hue = candidate
    return _hue_to_hex(best_hue)


class CharacterVoiceCancelled(Exception):
    """Raised when waiting for a generated character voice is cancelled."""


@dataclass(frozen=True)
class CachedCharacterVoice:
    character_name: str
    normalized_name: str
    prompt_hint: str
    voice_description: str
    omnivoice_description: str
    generated_voice_id: str
    preview_text: str
    transcript_filename: str
    transcript_filename_stem: str
    created_at: str
    updated_at: str
    cache_dir: Path
    voice_json_path: Path
    preview_path: Path
    display_color: str = ""


@dataclass(frozen=True)
class PreviewAudioScore:
    score: float
    spectral_centroid_hz: float
    rolloff_95_hz: float
    rolloff_99_hz: float
    low_band_db: float
    speech_band_db: float
    high_band_db: float

    def as_metadata(self) -> dict[str, float]:
        return {
            "score": round(self.score, 3),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 1),
            "rolloff_95_hz": round(self.rolloff_95_hz, 1),
            "rolloff_99_hz": round(self.rolloff_99_hz, 1),
            "low_band_db": round(self.low_band_db, 2),
            "speech_band_db": round(self.speech_band_db, 2),
            "high_band_db": round(self.high_band_db, 2),
        }


@dataclass(frozen=True)
class GeneratedVoiceDescriptions:
    elevenlabs_voice_description: str
    omnivoice_description: str
    gemini_usage: dict[str, int] | None = None
    gemini_model: str = ""


@dataclass(frozen=True)
class _PreviewCandidate:
    index: int
    generated_voice_id: str
    audio: bytes
    score: PreviewAudioScore | None


def load_env_file(path: Path = BASE_DIR / ".env") -> None:
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_character_name(name: str) -> str:
    display = re.sub(r"\s+", " ", str(name)).strip()
    slug = re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_")
    if not slug:
        slug = "character_" + hashlib.sha256(display.encode("utf-8")).hexdigest()[:12]
    if len(slug) > 80:
        digest = hashlib.sha256(display.encode("utf-8")).hexdigest()[:12]
        slug = f"{slug[:67].rstrip('_')}_{digest}"
    return slug


def _clean_character_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def _safe_transcript_stem(transcript_filename_stem: str) -> str:
    stem = str(transcript_filename_stem or "default").strip()
    stem = stem.replace("\\", "/")
    stem = "/".join(part for part in stem.split("/") if part not in {"", ".", ".."})
    return stem or "default"


def character_voice_cache_dir(
    cache_root: Path | str,
    transcript_filename_stem: str,
    character_name: str,
) -> Path:
    stem = _safe_transcript_stem(transcript_filename_stem)
    return Path(cache_root).expanduser() / stem / normalize_character_name(character_name)


def _metadata_from_json(path: Path) -> CachedCharacterVoice | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cache_dir = path.parent
    preview_path = cache_dir / "preview.mp3"
    required = {
        "character_name",
        "normalized_character_name",
        "voice_description",
        "generated_voice_id",
        "preview_text",
        "transcript_filename",
        "transcript_filename_stem",
    }
    if not required.issubset(data) or not preview_path.exists():
        return None
    voice_description = str(data["voice_description"])
    omnivoice_description = _sanitize_omnivoice_description(
        str(data.get("omnivoice_description") or ""),
        fallback_text=voice_description,
    )
    display_color = str(data.get("display_color") or "").lower()
    if not HEX_COLOR_RE.match(display_color):
        # Voices cached before colors existed get a stable, name-derived one.
        display_color = fallback_display_color(str(data["character_name"]))
    return CachedCharacterVoice(
        character_name=str(data["character_name"]),
        normalized_name=str(data["normalized_character_name"]),
        prompt_hint=str(data.get("prompt_hint") or ""),
        voice_description=voice_description,
        omnivoice_description=omnivoice_description,
        generated_voice_id=str(data["generated_voice_id"]),
        preview_text=str(data["preview_text"]),
        transcript_filename=str(data["transcript_filename"]),
        transcript_filename_stem=str(data["transcript_filename_stem"]),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
        cache_dir=cache_dir,
        voice_json_path=path,
        preview_path=preview_path,
        display_color=display_color,
    )


def read_cached_character_voice(
    cache_root: Path | str,
    transcript_filename_stem: str,
    character_name: str,
) -> CachedCharacterVoice | None:
    path = character_voice_cache_dir(cache_root, transcript_filename_stem, character_name) / "voice.json"
    return _metadata_from_json(path)


def read_cached_character_voices(
    transcript_filename_stem: str,
    cache_root: Path | str = DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
) -> list[CachedCharacterVoice]:
    stem = _safe_transcript_stem(transcript_filename_stem)
    root = Path(cache_root).expanduser() / stem
    if not root.exists():
        return []
    voices: list[CachedCharacterVoice] = []
    for path in sorted(root.glob("*/voice.json")):
        metadata = _metadata_from_json(path)
        if metadata is not None:
            voices.append(metadata)
    voices.sort(key=lambda item: item.character_name.casefold())
    return voices


def build_omnivoice_voice_prompt_block(
    transcript_filename_stem: str,
    cache_root: Path | str = DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
) -> str:
    cached = read_cached_character_voices(transcript_filename_stem, cache_root=cache_root)
    lines = [
        "",
        "OmniVoice named character voices are enabled. For every span of "
        "spoken dialogue by any speaker, wrap only the spoken dialogue words "
        "in <voice name=\"Speaker Name\">...</voice>. This applies to named "
        "characters and to unnamed role speakers such as bellboys, clerks, "
        "managers, constables, cabbies, servants, guards, witnesses, and "
        "bystanders. Do not leave quoted speech untagged merely because the "
        "speaker lacks a proper name.",
        "",
        "The voice name is the cache key. Use a proper character name when "
        "known. If the speaker has no proper name, invent a short, stable, "
        "descriptive role name with enough context to be reusable and distinct, "
        "such as \"hotel bellboy\", \"hotel manager\", \"day clerk\", "
        "\"room constable\", or \"cab driver\". Add local context only when "
        "needed to distinguish multiple speakers with the same role. Reuse "
        "the same descriptive name for the same recurring role speaker across "
        "later turns.",
        "",
        "For quoted dialogue, keep ordinary quotation marks visible and put "
        "the voice tags inside them: "
        '"<voice name="hotel bellboy">...</voice>". Narration, action text, '
        "dialogue attribution words outside the spoken words, hidden text, "
        "location tags, and progress tags must never be inside a voice tag.",
        "",
        "Use visible prose to establish the speaker the first time a "
        "character's named voice is heard, or whenever context would otherwise "
        "be genuinely unclear. After that voice has been used once, do not "
        "add dialogue attribution words such as 'he said,' 'she replied,' or "
        "'Watson asked' solely to identify the speaker. This means omitting "
        "attribution prose outside the quote; it does not mean removing "
        "ordinary quotation marks around dialogue. Let the established voice "
        "identify the speaker; use a brief action beat only when it adds story "
        "information or prevents real ambiguity.",
        "",
        "Do not narrate how voice-tagged dialogue is spoken merely as "
        "performance direction: avoid phrases like 'he said angrily,' 'she "
        "whispered softly,' or 'Watson replied in a trembling voice' when "
        "punctuation, wording, rhythm, or supported audio tags can carry the "
        "delivery. Keep concrete actions and important sensory facts, but "
        "leave delivery-only tags out of visible narration.",
        "",
        "Do not add prompt attributes or numbered voice IDs. The voice-design "
        "system receives the full original transcript separately and creates "
        "the ElevenLabs and OmniVoice voice prompts itself from the character "
        "name, transcript context, and cached voices.",
        "",
        "Cached OmniVoice character voices for this transcript:",
    ]
    if cached:
        for voice in cached:
            name = html.escape(voice.character_name, quote=True)
            description = html.escape(voice.voice_description, quote=False)
            lines.append(f'<voice name="{name}">{description}</voice>')
    else:
        lines.append("(none yet)")
    return "\n".join(lines) + "\n"


def _completed_future(result: CachedCharacterVoice | None) -> Future:
    future: Future = Future()
    future.set_result(result)
    return future


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        tmp.write(encoded)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _read_cached_description(cache_dir: Path) -> GeneratedVoiceDescriptions | None:
    """Reuse a previously generated Gemini description so a later ElevenLabs
    design retry does not re-bill the description model."""
    try:
        data = json.loads((cache_dir / "description.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    voice_description = str(data.get("elevenlabs_voice_description") or "")
    if len(voice_description) < 20:
        return None
    return GeneratedVoiceDescriptions(
        elevenlabs_voice_description=voice_description,
        omnivoice_description=str(data.get("omnivoice_description") or ""),
    )


def _write_cached_description(cache_dir: Path, descriptions: GeneratedVoiceDescriptions) -> None:
    if len(descriptions.elevenlabs_voice_description) < 20:
        return
    _atomic_write_json(
        cache_dir / "description.json",
        {
            "schema_version": 1,
            "elevenlabs_voice_description": descriptions.elevenlabs_voice_description,
            "omnivoice_description": descriptions.omnivoice_description,
            "created_at": _now_iso(),
        },
    )


def _sanitize_voice_description(text: str) -> str:
    text = re.sub(r"^```(?:\w+)?|```$", "", str(text).strip())
    text = re.sub(r"(?i)^voice_description\s*:\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) < 20:
        text = (text + " Studio quality, natural dialogue delivery.").strip()
    text = DEGRADED_AUDIO_TERMS_RE.sub("clear", text)
    additions: list[str] = []
    if not ACCENT_MARKER_RE.search(text):
        additions.append("Accent: neutral English accent.")
    quality_terms = (
        "perfect audio quality",
        "studio-quality recording",
        "broadcast quality",
        "full-band microphone",
        "rich low end",
        "open high frequencies",
    )
    if not all(term in text.lower() for term in quality_terms):
        additions.append(AUDIO_QUALITY_DESCRIPTION)
    if additions:
        suffix = " " + " ".join(additions)
        if len(text) + len(suffix) > 1000:
            text = text[: max(20, 1000 - len(suffix))].rstrip()
        text = (text + suffix).strip()
    if len(text) > 1000:
        text = text[:1000].rstrip()
    return text


def _strip_json_fence(text: str) -> str:
    stripped = str(text).strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _find_supported_attribute(text: str, values: tuple[str, ...]) -> str:
    lowered = text.lower()
    for value in values:
        if re.search(rf"(?<![a-z]){re.escape(value)}(?![a-z])", lowered):
            return value
    return ""


def _find_omnivoice_accent(text: str) -> str:
    lowered = text.lower()
    for alias, value in OMNIVOICE_ACCENT_ALIASES.items():
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered):
            return value
    return _find_supported_attribute(lowered, OMNIVOICE_ENGLISH_ACCENTS)


def _sanitize_omnivoice_description(text: str, fallback_text: str = "") -> str:
    text = re.sub(r"^```(?:\w+)?|```$", "", str(text).strip())
    text = re.sub(r"(?i)^omnivoice_description\s*:\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    combined = f"{text} {fallback_text}".strip()
    combined = DEGRADED_AUDIO_TERMS_RE.sub("clear", combined)

    gender = _find_supported_attribute(combined, OMNIVOICE_GENDERS)
    age = _find_supported_attribute(combined, OMNIVOICE_AGES)
    pitch = _find_supported_attribute(combined, OMNIVOICE_PITCHES)
    style = _find_supported_attribute(combined, OMNIVOICE_STYLES)
    accent = _find_omnivoice_accent(combined)

    if not pitch:
        pitch = "moderate pitch"
    if not accent:
        accent = "american accent"

    attrs = [item for item in (gender, age, pitch, style, accent) if item]
    return ", ".join(dict.fromkeys(attrs))


def _coerce_generated_voice_descriptions(result: Any) -> GeneratedVoiceDescriptions:
    gemini_usage = None
    gemini_model = ""
    if isinstance(result, GeneratedVoiceDescriptions):
        voice_description = result.elevenlabs_voice_description
        omnivoice_description = result.omnivoice_description
        gemini_usage = result.gemini_usage
        gemini_model = result.gemini_model
    elif isinstance(result, dict):
        voice_description = (
            result.get("elevenlabs_voice_description")
            or result.get("voice_description")
            or ""
        )
        omnivoice_description = result.get("omnivoice_description") or ""
        raw_usage = result.get("gemini_usage") or result.get("_gemini_usage")
        if isinstance(raw_usage, dict):
            gemini_usage = {
                "prompt": int(raw_usage.get("prompt", 0) or 0),
                "cached": int(raw_usage.get("cached", 0) or 0),
                "output": int(raw_usage.get("output", 0) or 0),
                "thoughts": int(raw_usage.get("thoughts", 0) or 0),
            }
        gemini_model = str(result.get("gemini_model") or result.get("_gemini_model") or "")
    else:
        text = str(result or "")
        try:
            data = json.loads(_strip_json_fence(text))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            voice_description = (
                data.get("elevenlabs_voice_description")
                or data.get("voice_description")
                or ""
            )
            omnivoice_description = data.get("omnivoice_description") or ""
            raw_usage = data.get("gemini_usage") or data.get("_gemini_usage")
            if isinstance(raw_usage, dict):
                gemini_usage = {
                    "prompt": int(raw_usage.get("prompt", 0) or 0),
                    "cached": int(raw_usage.get("cached", 0) or 0),
                    "output": int(raw_usage.get("output", 0) or 0),
                    "thoughts": int(raw_usage.get("thoughts", 0) or 0),
                }
            gemini_model = str(data.get("gemini_model") or data.get("_gemini_model") or "")
        else:
            voice_description = text
            omnivoice_description = ""

    voice_description = _sanitize_voice_description(str(voice_description))
    omnivoice_description = _sanitize_omnivoice_description(
        str(omnivoice_description),
        fallback_text=voice_description,
    )
    return GeneratedVoiceDescriptions(
        elevenlabs_voice_description=voice_description,
        omnivoice_description=omnivoice_description,
        gemini_usage=gemini_usage,
        gemini_model=gemini_model,
    )


def _parse_character_cost(value: Any, default: int) -> int:
    """Parse the ElevenLabs `character-cost` response header (billed credits)."""
    if value is None:
        return max(0, int(default))
    try:
        credits = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return max(0, int(default))
    return credits if credits >= 0 else max(0, int(default))


def _gemini_usage_to_dict(usage_metadata: Any) -> dict[str, int] | None:
    if usage_metadata is None:
        return None
    return {
        "prompt": int(getattr(usage_metadata, "prompt_token_count", 0) or 0),
        "cached": int(getattr(usage_metadata, "cached_content_token_count", 0) or 0),
        "output": int(getattr(usage_metadata, "candidates_token_count", 0) or 0),
        "thoughts": int(getattr(usage_metadata, "thoughts_token_count", 0) or 0),
    }


def _band_db(power: Any, freqs: Any, low_hz: float, high_hz: float, total: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    band_power = float(power[mask].sum()) if bool(mask.any()) else 0.0
    return 10.0 * math.log10((band_power + 1e-20) / max(total, 1e-20))


def _default_preview_audio_scorer(audio: bytes) -> PreviewAudioScore | None:
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if decoded.returncode != 0 or not decoded.stdout:
        return None

    samples = np.frombuffer(decoded.stdout, dtype=np.float32)
    samples = samples[np.isfinite(samples)]
    if samples.size < 512:
        return None
    samples = samples - float(np.mean(samples))

    frame_size = 2048
    hop = 1024
    if samples.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: samples.size] = samples
        frames = padded.reshape(1, frame_size)
    else:
        frame_count = (samples.size - frame_size) // hop + 1
        shape = (frame_count, frame_size)
        strides = (samples.strides[0] * hop, samples.strides[0])
        frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)

    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    if frame_rms.size:
        active = frame_rms > max(float(frame_rms.max()) * 0.03, 1e-5)
        if bool(active.any()):
            frames = frames[active]

    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    power = spectrum.mean(axis=0)
    total = float(power.sum())
    if total <= 0:
        return None
    freqs = np.fft.rfftfreq(frame_size, 1.0 / 24000.0)
    cumulative = np.cumsum(power) / total
    rolloff_95 = float(freqs[min(np.searchsorted(cumulative, 0.95), freqs.size - 1)])
    rolloff_99 = float(freqs[min(np.searchsorted(cumulative, 0.99), freqs.size - 1)])
    centroid = float((freqs * power).sum() / total)
    low_band_db = _band_db(power, freqs, 80.0, 300.0, total)
    speech_band_db = _band_db(power, freqs, 300.0, 3400.0, total)
    high_band_db = _band_db(power, freqs, 3400.0, 8000.0, total)

    high_ratio = 10.0 ** (high_band_db / 10.0)
    low_ratio = 10.0 ** (low_band_db / 10.0)
    score = rolloff_95 + (rolloff_99 * 0.25) + (high_ratio * 2500.0) + (low_ratio * 500.0)
    return PreviewAudioScore(
        score=float(score),
        spectral_centroid_hz=centroid,
        rolloff_95_hz=rolloff_95,
        rolloff_99_hz=rolloff_99,
        low_band_db=low_band_db,
        speech_band_db=speech_band_db,
        high_band_db=high_band_db,
    )


def _select_widest_preview(
    previews: list[Any],
    preview_scorer: Callable[[bytes], PreviewAudioScore | None],
) -> _PreviewCandidate:
    candidates: list[_PreviewCandidate] = []
    for index, preview in enumerate(previews):
        if not isinstance(preview, dict):
            continue
        audio_base64 = preview.get("audio_base_64")
        generated_voice_id = preview.get("generated_voice_id")
        if not isinstance(audio_base64, str) or not audio_base64:
            continue
        if not isinstance(generated_voice_id, str) or not generated_voice_id:
            continue
        try:
            audio = base64.b64decode(audio_base64)
        except (TypeError, ValueError):
            continue
        try:
            audio_score = preview_scorer(audio)
        except Exception:
            audio_score = None
        candidates.append(
            _PreviewCandidate(
                index=index,
                generated_voice_id=generated_voice_id,
                audio=audio,
                score=audio_score,
            )
        )
    if not candidates:
        raise RuntimeError("ElevenLabs returned no usable voice previews")
    scored = [candidate for candidate in candidates if candidate.score is not None]
    if not scored:
        return candidates[0]
    return max(scored, key=lambda candidate: (candidate.score.score, -candidate.index))


def transcript_prompt_preamble(game_title: str, transcript_text: str) -> str:
    """Shared, byte-stable prompt prefix for transcript-grounded voice calls.

    Gemini implicit caching matches on exact request prefixes, so every
    voice-related call (speaker discovery, per-character voice description)
    starts with this identical block — the large common content first, per
    Google's caching guidance — with the per-call variable parts (character
    name, cached-voice list, task instructions) after the transcript.
    """
    return f"""\
You are working with the complete original transcript of an interactive
fiction playthrough. Read the transcript, then follow the task instructions
that come after it.

Game title: {game_title}

Full original transcript:
{transcript_text}

=== END OF TRANSCRIPT — TASK INSTRUCTIONS FOLLOW ===

"""


def _default_description_generator(
    *,
    character_name: str,
    prompt_hint: str,
    transcript_text: str,
    game_title: str,
    existing_voices: list[CachedCharacterVoice],
    api_key: str,
) -> GeneratedVoiceDescriptions:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for character voice descriptions") from exc

    existing_block = "\n".join(
        (
            f"- {voice.character_name}: ElevenLabs: {voice.voice_description} "
            f"OmniVoice: {voice.omnivoice_description}"
        )
        for voice in existing_voices
    ) or "(none)"
    prompt = transcript_prompt_preamble(game_title, transcript_text) + f"""\
Create voice prompts for an interactive fiction character from the transcript
above. Do not rely on any voice hint from the story response; infer the voice
entirely from the character name, transcript context, and existing cached
voices.

Return only JSON with exactly these string keys:
{{
  "elevenlabs_voice_description": "...",
  "omnivoice_description": "..."
}}

elevenlabs_voice_description must be 20-1000 characters and should follow
ElevenLabs Voice Design guidance: always include a specific accent or dialect;
if unclear, say neutral English accent. Also include age, vocal weight, tone or
timbre, pacing, emotional delivery, and persona. Include these audio quality
ideas in natural prose: perfect audio quality, studio-quality recording,
broadcast quality, full-band microphone capture, rich low end, and open high
frequencies. Avoid naming actors, real public figures, or copyrighted
performers. Do not include degraded audio-effect terms such as phone, telephone,
radio, intercom, walkie-talkie, tape, tinny, muffled, narrowband, or
band-limited.

Every generated character voice MUST be clearly distinct from every already
cached voice. Prioritize speaker separation over transcript fidelity: if the
transcript implies two characters should sound similar, deliberately choose
different age, pitch, vocal weight, timbre, pacing, accent or dialect, persona,
and emotional delivery so the listener can tell the speakers apart.

omnivoice_description is used as OmniVoice's `instruct` parameter while
generating speech with the cloned preview voice. Follow the OmniVoice voice
design rules: write a comma-separated string of supported speaker attributes;
use at most one attribute from each category; combine attributes across
categories freely; use half-width commas. Supported gender attributes: male,
female. Supported age attributes: child, teenager, young adult, middle-aged,
elderly. Supported pitch attributes: very low pitch, low pitch, moderate pitch,
high pitch, very high pitch. Supported style attribute: whisper, but omit it
unless the character should always whisper because temporary whisper delivery is
handled separately. Supported English accents: american accent, british accent,
australian accent, canadian accent, indian accent, chinese accent, korean
accent, japanese accent, portuguese accent, russian accent. Use only supported
OmniVoice attributes in omnivoice_description, and always include an accent.
Make both descriptions distinct from already cached voices, even when that
breaks strict transcript fidelity.

Already cached voices:
{existing_block}

Character name: {character_name}
"""
    client = genai.Client(api_key=api_key)
    model = os.environ.get("IF_ENGINE_CHARACTER_VOICE_DESCRIPTION_MODEL", DEFAULT_DESCRIPTION_MODEL)
    response = client.models.generate_content(model=model, contents=prompt)
    descriptions = _coerce_generated_voice_descriptions(getattr(response, "text", "") or "")
    if len(descriptions.elevenlabs_voice_description) < 20:
        raise RuntimeError("Gemini returned an unusably short voice description")
    return GeneratedVoiceDescriptions(
        elevenlabs_voice_description=descriptions.elevenlabs_voice_description,
        omnivoice_description=descriptions.omnivoice_description,
        gemini_usage=_gemini_usage_to_dict(getattr(response, "usage_metadata", None)),
        gemini_model=model,
    )


def _default_voice_designer(
    *,
    voice_description: str,
    preview_text: str,
    api_key: str,
    quality: int = ELEVENLABS_VOICE_DESIGN_QUALITY,
) -> dict[str, Any]:
    payload = {
        "voice_description": voice_description,
        "text": preview_text,
        "auto_generate_text": False,
        "quality": quality,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ELEVENLABS_VOICE_DESIGN_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "xi-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
            # ElevenLabs reports the actual billed credits for the request in
            # the documented `character-cost` response header; stash it on the
            # body so cost accounting uses real billing data, not an estimate.
            character_cost = response.headers.get("character-cost")
            if isinstance(body, dict) and character_cost is not None:
                body["_character_cost_header"] = character_cost
            return body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ElevenLabs voice design failed with HTTP {exc.code}: {body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ElevenLabs voice design failed: {exc}") from exc


class CharacterVoiceCache:
    """Threaded cache for ElevenLabs-designed character voice previews."""

    def __init__(
        self,
        *,
        cache_root: Path | str = DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
        transcript_filename_stem: str,
        transcript_filename: str,
        transcript_text: str,
        game_title: str = "",
        notify: Callable[[str], None] | None = None,
        log: Callable[[str], None] | None = None,
        description_generator: Callable[..., Any] | None = None,
        voice_designer: Callable[..., dict[str, Any]] | None = None,
        preview_scorer: Callable[[bytes], PreviewAudioScore | None] | None = None,
        cost_recorder: Callable[[dict[str, Any]], None] | None = None,
        gemini_api_key: str | None = None,
        elevenlabs_api_key: str | None = None,
        max_workers: int = 2,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser()
        self.transcript_filename_stem = transcript_filename_stem or "default"
        self.transcript_filename = transcript_filename or f"{self.transcript_filename_stem}.txt"
        self.transcript_text = transcript_text
        self.game_title = game_title or self.transcript_filename_stem
        self.notify = notify
        self.log = log
        self.description_generator = description_generator or _default_description_generator
        self.voice_designer = voice_designer or _default_voice_designer
        self.preview_scorer = preview_scorer or _default_preview_audio_scorer
        self.cost_recorder = cost_recorder
        self.gemini_api_key = gemini_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="if-engine-elevenlabs-voice",
        )
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._jobs: dict[str, Future] = {}
        self._failed: set[str] = set()

    def cache_dir_for(self, character_name: str) -> Path:
        return character_voice_cache_dir(
            self.cache_root,
            self.transcript_filename_stem,
            character_name,
        )

    def cached_voice(self, character_name: str) -> CachedCharacterVoice | None:
        return read_cached_character_voice(
            self.cache_root,
            self.transcript_filename_stem,
            character_name,
        )

    def cached_voices(self) -> list[CachedCharacterVoice]:
        return read_cached_character_voices(
            self.transcript_filename_stem,
            cache_root=self.cache_root,
        )

    def ensure_started(self, character_name: str, prompt_hint: str | None = None) -> Future:
        clean_name = _clean_character_name(character_name)
        if not clean_name:
            return _completed_future(None)
        normalized = normalize_character_name(clean_name)
        cached = self.cached_voice(clean_name)
        if cached is not None:
            return _completed_future(cached)
        with self._lock:
            if normalized in self._failed:
                return _completed_future(None)
            existing = self._jobs.get(normalized)
            if existing is not None:
                return existing
            future = self._executor.submit(
                self._generate_voice,
                clean_name,
                "",
            )
            self._jobs[normalized] = future
            return future

    def wait_for_ready(
        self,
        character_name: str,
        prompt_hint: str | None = None,
        cancel_event=None,
    ) -> CachedCharacterVoice | None:
        """Wait for cache readiness without ever cancelling the generation job.

        `cancel_event` only aborts this caller's wait, such as an interrupted
        narration turn. The Future remains in `_jobs`, so the same character's
        next line will wait for the still-running job or use its completed
        cache entry.
        """
        future = self.ensure_started(character_name, prompt_hint)
        while not future.done():
            if cancel_event is not None and cancel_event.is_set():
                raise CharacterVoiceCancelled()
            time.sleep(0.05)
        try:
            metadata = future.result()
        except Exception as exc:
            self._record_failure(character_name, exc)
            return None
        if metadata is None:
            self._record_failure(character_name, None)
        return metadata

    def _record_failure(self, character_name: str, exc: BaseException | None) -> None:
        normalized = normalize_character_name(character_name)
        with self._lock:
            self._failed.add(normalized)
        if exc is not None:
            self._emit(f"voice: character voice '{character_name}' unavailable: {exc}")

    def _emit(self, message: str) -> None:
        if self.log is not None:
            self.log(message)
        if self.notify is not None:
            self.notify(message)

    def _record_cost(self, event: dict[str, Any]) -> None:
        if self.cost_recorder is None:
            return
        try:
            self.cost_recorder(dict(event))
        except Exception as exc:
            if self.log is not None:
                self.log(f"voice: cost accounting failed: {exc}")

    def _api_keys(self) -> tuple[str, str]:
        load_env_file()
        gemini_key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        elevenlabs_key = self.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if not elevenlabs_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        return gemini_key, elevenlabs_key

    def _generate_voice(self, character_name: str, prompt_hint: str) -> CachedCharacterVoice | None:
        cached = self.cached_voice(character_name)
        if cached is not None:
            return cached
        with self._generation_lock:
            return self._generate_voice_locked(character_name, prompt_hint)

    def _generate_voice_locked(
        self,
        character_name: str,
        prompt_hint: str,
    ) -> CachedCharacterVoice | None:
        cached = self.cached_voice(character_name)
        if cached is not None:
            return cached
        try:
            gemini_key, elevenlabs_key = self._api_keys()
            cache_dir = self.cache_dir_for(character_name)
            descriptions = _read_cached_description(cache_dir)
            if descriptions is None:
                existing_voices = [
                    voice
                    for voice in self.cached_voices()
                    if voice.character_name.casefold() != character_name.casefold()
                ]
                description_result = self.description_generator(
                    character_name=character_name,
                    prompt_hint="",
                    transcript_text=self.transcript_text,
                    game_title=self.game_title,
                    existing_voices=existing_voices,
                    api_key=gemini_key,
                )
                descriptions = _coerce_generated_voice_descriptions(description_result)
                _write_cached_description(cache_dir, descriptions)
                if descriptions.gemini_usage is not None:
                    self._record_cost(
                        {
                            "service": "gemini",
                            "category": "character_voice_description",
                            "label": "Gemini character voice description",
                            "character_name": character_name,
                            "transcript_filename": self.transcript_filename,
                            "transcript_filename_stem": self.transcript_filename_stem,
                            "model": descriptions.gemini_model or DEFAULT_DESCRIPTION_MODEL,
                            "usage": descriptions.gemini_usage,
                        }
                    )
            response = self.voice_designer(
                voice_description=descriptions.elevenlabs_voice_description,
                preview_text=ELEVENLABS_PREVIEW_TEXT,
                api_key=elevenlabs_key,
                quality=ELEVENLABS_VOICE_DESIGN_QUALITY,
            )
            previews = response.get("previews")
            preview_count = len(previews) if isinstance(previews, list) else 0
            # Bill from the actual response: the text ElevenLabs reports it
            # used, and the billed credits from the `character-cost` response
            # header when present (falling back to the preview text length).
            billed_text = str(response.get("text") or ELEVENLABS_PREVIEW_TEXT)
            billed_credits = _parse_character_cost(
                response.get("_character_cost_header"), default=len(billed_text)
            )
            self._record_cost(
                {
                    "service": "elevenlabs",
                    "category": "voice_design",
                    "label": "ElevenLabs voice design",
                    "character_name": character_name,
                    "transcript_filename": self.transcript_filename,
                    "transcript_filename_stem": self.transcript_filename_stem,
                    "preview_text": billed_text,
                    "characters": len(billed_text),
                    "credits": billed_credits,
                    "preview_count": preview_count,
                    "quality": ELEVENLABS_VOICE_DESIGN_QUALITY,
                }
            )
            if not isinstance(previews, list) or not previews:
                raise RuntimeError("ElevenLabs returned no voice previews")
            selected_preview = _select_widest_preview(previews, self.preview_scorer)
            preview_path = cache_dir / "preview.mp3"
            voice_json_path = cache_dir / "voice.json"
            now = _now_iso()
            display_color = pick_display_color(
                [
                    voice.display_color
                    for voice in self.cached_voices()
                    if voice.character_name.casefold() != character_name.casefold()
                ],
                character_name,
            )
            metadata = {
                "character_name": character_name,
                "normalized_character_name": normalize_character_name(character_name),
                "prompt_hint": "",
                "voice_description": descriptions.elevenlabs_voice_description,
                "omnivoice_description": descriptions.omnivoice_description,
                "generated_voice_id": selected_preview.generated_voice_id,
                "preview_text": billed_text,
                "elevenlabs_billable_characters": billed_credits,
                "display_color": display_color,
                "selected_preview_index": selected_preview.index,
                "preview_count": len(previews),
                "transcript_filename": self.transcript_filename,
                "transcript_filename_stem": self.transcript_filename_stem,
                "created_at": now,
                "updated_at": now,
            }
            if selected_preview.score is not None:
                metadata["preview_audio_score"] = selected_preview.score.as_metadata()
            _atomic_write_bytes(preview_path, selected_preview.audio)
            _atomic_write_json(voice_json_path, metadata)
            if self.log is not None:
                score_text = (
                    f" score={selected_preview.score.score:.1f}"
                    if selected_preview.score is not None
                    else " score=n/a"
                )
                self.log(
                    "voice: cached ElevenLabs character voice "
                    f"'{character_name}' from preview "
                    f"{selected_preview.index + 1}/{len(previews)}{score_text}"
                )
            return _metadata_from_json(voice_json_path)
        except Exception as exc:
            self._record_failure(character_name, exc)
            return None
