#!/usr/bin/env python3
"""Continuous voice I/O for the IF engine, with optional wake-word gating.

Adapted from https://github.com/akeybl/voice-loop/tree/main, trimmed to what
the game needs:

  - Narration out: streamed narration text is split into TTS chunks as it
    arrives, synthesized with OmniVoice by default (or Kokoro when selected),
    and played through one queued CoreAudio output stream per narration.
    OmniVoice gets the first sentence quickly, then paragraph-sized chunks;
    Kokoro keeps sentence-sized chunks.
  - Player in: mic at 16 kHz -> WebRTC AEC (removes the echo of our own
    playback) -> Silero VAD continuous turn detector -> whisper.cpp.
  - Policy: a VAD speech start during playback pauses playback; an empty
    transcript (noise) resumes it; a real transcript stops playback and is
    handed to the game as player input.

Heavy dependencies (torch, omnivoice, kokoro, silero-vad, sounddevice,
aec-audio-processing) are imported at module load; import this module lazily.
whisper.cpp binaries/models can be configured with IF_ENGINE_WHISPER_DIR,
IF_ENGINE_WHISPER_CLI, IF_ENGINE_WHISPER_MODEL, and
IF_ENGINE_WHISPER_VAD_MODEL.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
import ctypes
import ctypes.util
import html

# Model-loading noise that would clutter the game terminal: benign torch
# warnings from Kokoro's architecture, and the HF Hub anonymous-access notice
# (printed from native code, unsuppressible from Python). Models are cached
# locally, so default to offline mode — KokoroSpeaker falls back to a one-time
# online download if the cache is missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", message="dropout option adds dropout")
warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from elevenlabs_voices import (
    CharacterVoiceCache,
    CharacterVoiceCancelled,
    DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
    color_markup,
    strip_color_markup,
)

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad

try:
    from aec_audio_processing import AudioProcessor
except Exception:
    AudioProcessor = None

SAMPLE_RATE_IN = 16_000
SAMPLE_RATE_TTS = 24_000
AEC_FRAME_SIZE = 160
VAD_FRAME_SIZE = 512
OPENWAKEWORD_FRAME_SIZE = 1280

VOICE_DIR = Path(__file__).resolve().parent
DEFAULT_WHISPER_DIR = Path(
    os.environ.get(
        "IF_ENGINE_WHISPER_DIR",
        VOICE_DIR / ".cache" / "whisper.cpp",
    )
)
DEFAULT_WHISPER_MODEL = DEFAULT_WHISPER_DIR / "models" / "ggml-large-v3-turbo-q5_0.bin"
DEFAULT_WAKE_WORD_DIR = VOICE_DIR / "models" / "wake-word"
DEFAULT_OMNIVOICE_CLONE_REFERENCE = VOICE_DIR / "assets" / "voice-clone-reference.wav"
DEFAULT_OMNIVOICE_CLONE_TRANSCRIPT = VOICE_DIR / "assets" / "voice-clone-reference.txt"
DEFAULT_OMNIVOICE_WHISPER_CLONE_REFERENCE = (
    VOICE_DIR / "assets" / "voice-clone-whisper-reference.wav"
)
DEFAULT_OMNIVOICE_WHISPER_CLONE_TRANSCRIPT = (
    VOICE_DIR / "assets" / "voice-clone-whisper-reference.txt"
)
DEFAULT_OPENWAKEWORD_CACHE_DIR = Path(
    os.environ.get("IF_ENGINE_OPENWAKEWORD_CACHE_DIR", VOICE_DIR / ".cache" / "openwakeword")
)
DEFAULT_OPENWAKEWORD_MODEL = DEFAULT_WAKE_WORD_DIR / "okay.onnx"
DEFAULT_OPENWAKEWORD_MODEL_CANDIDATES = (
    DEFAULT_OPENWAKEWORD_MODEL,
)

# Hard caps so a wedged whisper-cli can never freeze the pipeline silently.
TRANSCRIBE_TIMEOUT_SECONDS = 90.0
ALIGN_TIMEOUT_SECONDS = 45.0
# A VAD pause hold that outlives this is presumed leaked and gets released.
PAUSE_HOLD_MAX_SECONDS = 45.0
KEYBOARD_GATE_POLL_SECONDS = 0.03
VAD_GATE_SUPPRESS_SECONDS = 0.45
# A live mic (even silent) delivers input callbacks continuously; this much
# silence from the callback means the input stream died (device unplugged).
INPUT_STREAM_STALL_SECONDS = 4.0
TTS_EDGE_FADE_MS = 8.0
TTS_PEAK_HEADROOM = 0.98

VOICE_LOG_PATH = Path(os.environ.get(
    "IF_ENGINE_VOICE_LOG", Path(__file__).resolve().parent / "voice.log"
))

ANNOTATED_NON_SPEECH_RE = re.compile(
    r"\[[^\]\n]*\]|\([^)\n]*\)|\*{1,2}[^*\n]*\*{1,2}"
)
ONLY_NON_SPEECH_RE = re.compile(
    r"(?i)^(?:[>\s.,;:!?()*_\-\[\]]|blank[_\s-]*audio|no[_\s-]*speech)+$"
)


class CancelledError(Exception):
    pass


@dataclass(frozen=True)
class ModifierState:
    caps_lock: bool = False
    shift: bool = False


@dataclass(frozen=True)
class KeyboardGateDecision:
    manual_active: bool
    wake_only: bool
    blocked: bool
    open_mic: bool


def keyboard_gate_decision(
    state: ModifierState,
    *,
    wake_gate_available: bool,
) -> KeyboardGateDecision:
    manual_active = state.shift
    open_mic = state.caps_lock and not manual_active
    wake_only = not open_mic and not manual_active and wake_gate_available
    blocked = not open_mic and not manual_active and not wake_gate_available
    return KeyboardGateDecision(
        manual_active=manual_active,
        wake_only=wake_only,
        blocked=blocked,
        open_mic=open_mic,
    )


class ModifierKeyReader:
    """Read global modifier state. Currently implemented for macOS through
    CoreGraphics; unavailable platforms report no modifiers and leave the
    microphone gate disabled."""

    ALPHA_SHIFT_MASK = 1 << 16
    SHIFT_MASK = 1 << 17

    def __init__(self) -> None:
        self.available = False
        self._flags_fn = None
        if sys.platform != "darwin":
            return
        path = ctypes.util.find_library("ApplicationServices")
        if not path:
            return
        try:
            lib = ctypes.CDLL(path)
            flags_fn = lib.CGEventSourceFlagsState
            flags_fn.argtypes = [ctypes.c_uint32]
            flags_fn.restype = ctypes.c_uint64
        except Exception:
            return
        self._flags_fn = flags_fn
        self.available = True

    def read(self) -> ModifierState:
        if self._flags_fn is None:
            return ModifierState()
        flags = int(self._flags_fn(0))
        return ModifierState(
            caps_lock=bool(flags & self.ALPHA_SHIFT_MASK),
            shift=bool(flags & self.SHIFT_MASK),
        )


# ── Small audio helpers ──────────────────────────────────────────────────────

def rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block.astype(np.float32)))))


def float_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def float_to_pcm16_array(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def pcm16_bytes_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def mono_float32(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 0:
        return samples.reshape(1)
    samples = np.squeeze(samples)
    if samples.ndim <= 1:
        return samples.reshape(-1).astype(np.float32, copy=False)
    if samples.shape[0] <= 8 and samples.shape[0] < samples.shape[-1]:
        samples = samples.mean(axis=0)
    elif samples.shape[-1] <= 8:
        samples = samples.mean(axis=-1)
    else:
        samples = samples.reshape(-1)
    return samples.astype(np.float32, copy=False).reshape(-1)


def apply_edge_fade(
    samples: np.ndarray,
    sample_rate: int,
    fade_ms: float = TTS_EDGE_FADE_MS,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).copy()
    if samples.size == 0 or fade_ms <= 0:
        return samples
    fade_frames = int(round(sample_rate * fade_ms / 1000))
    fade_frames = min(max(0, fade_frames), samples.size // 2)
    if fade_frames <= 1:
        return samples
    fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_frames, dtype=np.float32))
    samples[:fade_frames] *= fade
    samples[-fade_frames:] *= fade[::-1]
    return samples


def postprocess_tts_audio(
    samples: np.ndarray,
    *,
    volume: float,
    sample_rate: int = SAMPLE_RATE_TTS,
    fade_ms: float = TTS_EDGE_FADE_MS,
    peak_headroom: float = TTS_PEAK_HEADROOM,
    remove_dc: bool = True,
) -> np.ndarray:
    samples = mono_float32(samples)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if remove_dc and samples.size > 1:
        dc_offset = float(np.mean(samples))
        if abs(dc_offset) > 1e-5:
            samples = samples - dc_offset
    samples = samples.astype(np.float32, copy=False) * float(volume)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > peak_headroom > 0:
        samples = samples * (peak_headroom / peak)
    samples = apply_edge_fade(samples, sample_rate, fade_ms)
    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate or samples.size == 0:
        return samples.copy()
    if samples.size == 1:
        return np.repeat(samples, max(1, round(target_rate / source_rate))).astype(np.float32)
    target_count = max(1, int(round(samples.size * target_rate / source_rate)))
    source_x = np.arange(samples.size, dtype=np.float64) / source_rate
    target_x = np.arange(target_count, dtype=np.float64) / target_rate
    return np.interp(target_x, source_x, samples).astype(np.float32)


class StreamingLinearResampler:
    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._buffer = np.zeros(0, dtype=np.float32)
        self._source_offset = 0
        self._next_target_index = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.source_rate == self.target_rate:
            return samples.copy()

        self._buffer = np.concatenate((self._buffer, samples))
        max_source_index = self._source_offset + self._buffer.size - 1
        max_target_index = int(np.floor(max_source_index * self.target_rate / self.source_rate))
        if max_target_index < self._next_target_index:
            return np.zeros(0, dtype=np.float32)

        target_indexes = np.arange(self._next_target_index, max_target_index + 1, dtype=np.float64)
        source_positions = target_indexes * self.source_rate / self.target_rate
        local_positions = source_positions - self._source_offset
        lower = np.floor(local_positions).astype(np.int64)
        upper = np.minimum(lower + 1, self._buffer.size - 1)
        fractions = (local_positions - lower).astype(np.float32)
        output = ((1.0 - fractions) * self._buffer[lower] + fractions * self._buffer[upper]).astype(np.float32)
        self._next_target_index = max_target_index + 1

        next_source_position = self._next_target_index * self.source_rate / self.target_rate
        drop = int(np.floor(next_source_position)) - self._source_offset - 1
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._source_offset += drop
        return output


def callback_time_seconds(time_info, field_name: str) -> float | None:
    try:
        value = getattr(time_info, field_name)
    except Exception:
        try:
            value = time_info[field_name]
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def load_cue_samples(path: Path, volume: float) -> np.ndarray:
    """Load a short 16-bit PCM WAV cue, downmixed to mono at SAMPLE_RATE_TTS."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    data = resample_linear(data, sample_rate, SAMPLE_RATE_TTS)
    return np.clip(data * volume, -1.0, 1.0)


def choose_output_channels(output_device) -> int:
    try:
        info = sd.query_devices(output_device, "output")
        return max(1, min(2, int(info.get("max_output_channels", 1))))
    except Exception:
        return 1


def choose_output_sample_rate(output_device, requested: int | None) -> int:
    if requested:
        return requested
    try:
        info = sd.query_devices(output_device, "output")
        rate = int(info.get("default_samplerate") or SAMPLE_RATE_TTS)
        return rate if rate > 0 else SAMPLE_RATE_TTS
    except Exception:
        return SAMPLE_RATE_TTS


IPHONE_DEVICE_RE = re.compile(r"\b(?:iphone|ipad)\b|continuity", re.IGNORECASE)
SYSTEM_AUDIO_RE = re.compile(
    r"macbook|imac|studio display|built[- ]?in|internal", re.IGNORECASE
)
VIRTUAL_AUDIO_RE = re.compile(
    r"blackhole|loopback|soundflower|aggregate|multi[- ]output|zoom|teams|"
    r"background music|vb[- ]?cable|obs|screen", re.IGNORECASE
)


def _audio_devices(kind: str) -> list[tuple[int, dict]]:
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices: list[tuple[int, dict]] = []
    for idx, info in enumerate(sd.query_devices()):
        try:
            if int(info.get(channel_key, 0)) < 1:
                continue
        except Exception:
            continue
        name = str(info.get("name", ""))
        if IPHONE_DEVICE_RE.search(name):
            continue
        devices.append((idx, info))
    return devices


def _device_name(info: dict) -> str:
    return str(info.get("name", ""))


def _is_system_audio(name: str) -> bool:
    return bool(SYSTEM_AUDIO_RE.search(name))


def _is_virtual_audio(name: str) -> bool:
    return bool(VIRTUAL_AUDIO_RE.search(name))


def _choose_audio_device(requested, kind: str):
    if requested is not None:
        return requested
    try:
        devices = _audio_devices(kind)
        # Prefer external physical devices first. CoreAudio device names do not
        # reliably expose transport, so non-system/non-virtual devices are the
        # best proxy for Bluetooth headphones, speakers, car audio, and mics.
        for idx, info in devices:
            name = _device_name(info)
            if not _is_system_audio(name) and not _is_virtual_audio(name):
                return idx
        for idx, info in devices:
            if _is_system_audio(_device_name(info)):
                return idx
        if devices:
            return devices[0][0]
    except Exception:
        pass
    return None


def choose_input_device(requested):
    return _choose_audio_device(requested, "input")


def choose_output_device(requested):
    return _choose_audio_device(requested, "output")


# ── Text helpers ─────────────────────────────────────────────────────────────

def clean_transcript(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = ANNOTATED_NON_SPEECH_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if ONLY_NON_SPEECH_RE.fullmatch(text):
        return ""
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def capitalize_transcript_start(text: str) -> str:
    match = re.search(r"[A-Za-z]", text)
    if match is None:
        return text
    index = match.start()
    return text[:index] + text[index].upper() + text[index + 1:]


WAKE_TRANSCRIPT_PREFIX_RE = re.compile(
    r"^\W*(?:(?:o[\W_]*k(?:ay)?)|(?:okay))\b[ \t\r\n,.;:!?-]*",
    re.IGNORECASE,
)
WAKE_WORD_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:o[\W_]*k(?:ay)?|okay)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def strip_leading_wake_word(text: str) -> str | None:
    match = WAKE_TRANSCRIPT_PREFIX_RE.match(text)
    if match is None:
        return None
    stripped = text[match.end():].lstrip(" \t\r\n,.;:!?-")
    return clean_transcript(stripped)


def contains_wake_word_text(text: str) -> bool:
    return WAKE_WORD_TEXT_RE.search(text) is not None


ELLIPSIS_TOKEN_RE = re.compile(r"(?:\.\.\.+|…)")
CUT_MARKER_WORD = "cut"
CUT_MARKER_PAD_MS = 35
WHISPER_TAG_RE = re.compile(r"<\s*(/?)\s*whisper\s*>", re.IGNORECASE)
OMNIVOICE_WHISPER_INSTRUCT_SUFFIX = "whisper"
VOICE_TAG_RE = re.compile(
    r"<\s*(/?)\s*voice\b([^>]*)>",
    re.IGNORECASE,
)
VOICE_ATTR_RE = re.compile(
    r"([A-Za-z_:][\w:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'>/]+))"
)
AUDIO_CUE_TAG_RE = re.compile(
    r"\[(?:sigh|laughter|pause\s+\d+(?:\.\d+)?\s*(?:ms|s)?)\]",
    re.IGNORECASE,
)


def strip_audio_cue_tags_for_display(text: str) -> str:
    text = WHISPER_TAG_RE.sub("", text)
    text = VOICE_TAG_RE.sub("", text)
    text = AUDIO_CUE_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return "\n".join(line.strip(" \t") for line in text.split("\n"))


def _kokoro_stress_link(text: str, level: int) -> str:
    if not re.search(r"[A-Za-z0-9]", text) or any(char in text for char in "[]"):
        return text
    leading_len = len(text) - len(text.lstrip())
    trailing_len = len(text) - len(text.rstrip())
    leading = text[:leading_len]
    trailing = text[len(text) - trailing_len:] if trailing_len else ""
    body_end = len(text) - trailing_len if trailing_len else len(text)
    body = text[leading_len:body_end]
    if not body:
        return text
    return f"{leading}[{body}](+{level}){trailing}"


def apply_kokoro_markdown_stress(text: str) -> str:
    parts: list[str] = []
    pos = 0
    bold_active = False
    italic_active = False
    while pos < len(text):
        if text.startswith("**", pos):
            bold_active = not bold_active
            pos += 2
            continue
        if text[pos] == "*":
            italic_active = not italic_active
            pos += 1
            continue
        next_star = text.find("*", pos)
        end = len(text) if next_star < 0 else next_star
        segment = text[pos:end]
        if bold_active:
            parts.append(_kokoro_stress_link(segment, 2))
        elif italic_active:
            parts.append(_kokoro_stress_link(segment, 1))
        else:
            parts.append(segment)
        pos = end
    return "".join(parts)


def normalize_spoken_text(
    text: str,
    *,
    cut_ellipses: bool = False,
    kokoro_markup: bool = False,
) -> str:
    text = WHISPER_TAG_RE.sub("", text)
    text = VOICE_TAG_RE.sub("", text)
    if kokoro_markup:
        text = apply_kokoro_markdown_stress(text)
    if cut_ellipses:
        text = ELLIPSIS_TOKEN_RE.sub(f", {CUT_MARKER_WORD},", text)
    text = re.sub(r"[*_`#|]+", "", text)
    # Dash pause punctuation is displayed, but Kokoro should not vocalize it.
    # The chunker turns these tokens into boundaries with real pause timing.
    text = SPOKEN_PAUSE_TOKEN_RE.sub(" ", text)
    # Kokoro fuses hyphenated words into one awkward phoneme token, which
    # renders with a hitch ("wood-panelled" -> wˌʊdpˈænᵊld); speak them as
    # separate words instead. "--" pause dashes are untouched (no \w-\w).
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)

    # Kokoro mangles all-caps words (spelled out or over-stressed), so shouted
    # text and headers are spoken in regular case. A lone token of up to three
    # letters is kept verbatim — those are deliberate acronyms (FBI) per the
    # voice prompt, plus "I" and friends — but inside a run of all-caps words
    # ("THE BOY WHO LIVED") every token is shouting, not an acronym.
    tokens = list(re.finditer(r"\b[A-Z]+(?:'[A-Z]+)*\b", text))
    runs: list[list["re.Match[str]"]] = []
    for tok in tokens:
        prev = runs[-1][-1] if runs else None
        gap = text[prev.end(): tok.start()] if prev is not None else None
        if gap is not None and not re.search(r"[A-Za-z0-9]", gap):
            runs[-1].append(tok)
        else:
            runs.append([tok])
    pieces = []
    pos = 0
    for run in runs:
        if len(run) == 1 and sum(c.isalpha() for c in run[0].group(0)) <= 3:
            continue
        for i, tok in enumerate(run):
            word = tok.group(0)
            if sum(c.isalpha() for c in word) == 1:
                continue  # dotted initials ("H.P."), "I", "A" stay verbatim
            lowered = word.lower()
            if i == 0:
                prior = text[: tok.start()].rstrip(" \"'“”‘’")
                if not prior or prior.endswith((".", "!", "?", "…", ":", "\n")):
                    lowered = lowered.capitalize()
            pieces.append(text[pos: tok.start()])
            pieces.append(lowered)
            pos = tok.end()
    pieces.append(text[pos:])
    text = "".join(pieces)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


# Kokoro renders dashes poorly (rushed or vocalized), so the chunker treats
# them as boundaries with real silence and the spoken text strips the
# punctuation before synthesis. Ellipses stay inside the sentence and are
# rendered by synthesizing a CUT marker, aligning it, then replacing that
# marker audio with real silence.
# Only " - ", "--", and the em dash count as pause dashes; bare hyphens and
# en dashes (compound words, ranges) are left alone.
PAUSE_BOUNDARY_RE = re.compile(
    r"(?:—|--+|\s+-\s+)"
)
SPOKEN_PAUSE_TOKEN_RE = PAUSE_BOUNDARY_RE


def pause_for_boundary_token(token: str, dash_pause: float, ellipsis_pause: float) -> float:
    return ellipsis_pause if ("…" in token or "..." in token) else dash_pause


def display_suffix_for_whitespace(whitespace: str) -> str:
    if "\n" in whitespace:
        return "\n\n" if whitespace.count("\n") >= 2 else "\n"
    return " "


def normalize_heard_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?\]\)}%])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text


def join_spoken_chunks(parts) -> str:
    return normalize_heard_text(" ".join(part.strip() for part in parts if part and part.strip()))


# ── Word-level alignment of generated TTS audio (whisper.cpp timestamps) ────

@dataclass
class TimedTextWord:
    text: str
    start_ms: int
    end_ms: int


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _parse_int(value, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _token_has_word_char(text: str) -> bool:
    return bool(re.search(r"\w", text, flags=re.UNICODE))


def _append_alignment_token(words: list[TimedTextWord], token_text: str, start_ms: int, end_ms: int) -> None:
    if not token_text:
        return
    stripped = re.sub(r"\s+", " ", token_text).strip()
    if not stripped or (stripped.startswith("[") and stripped.endswith("]")):
        return
    if not _token_has_word_char(stripped):
        if words:
            words[-1].text += stripped
            words[-1].end_ms = max(words[-1].end_ms, end_ms)
        return
    starts_new_word = token_text[:1].isspace() or not words
    if starts_new_word:
        words.append(TimedTextWord(stripped, start_ms, end_ms))
        return
    words[-1].text += stripped
    words[-1].end_ms = max(words[-1].end_ms, end_ms)


def whisper_json_alignment_words(data) -> list[TimedTextWord]:
    words: list[TimedTextWord] = []
    transcription = _as_dict(data).get("transcription")
    if not isinstance(transcription, list):
        return words
    for segment in transcription:
        tokens = _as_dict(segment).get("tokens")
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            record = _as_dict(token)
            text = record.get("text")
            offsets = _as_dict(record.get("offsets"))
            if not isinstance(text, str):
                continue
            start_ms = _parse_int(offsets.get("from"))
            end_ms = _parse_int(offsets.get("to"), start_ms)
            _append_alignment_token(words, text, start_ms, max(start_ms, end_ms))
    return words


def join_timed_words(words: list[TimedTextWord]) -> str:
    text = ""
    for word in words:
        part = word.text.strip()
        if not part:
            continue
        if not text:
            text = part
        elif re.match(r"^[,.;:!?\]\)}%]", part):
            text += part
        elif text.endswith(("(", "[", "{", "$", "#")):
            text += part
        else:
            text += " " + part
    return text


def _is_cut_marker_word(text: str) -> bool:
    return re.sub(r"\W+", "", text).lower() == CUT_MARKER_WORD


def replace_cut_markers_with_silence(
    samples: np.ndarray,
    alignment_words: list[TimedTextWord],
    ellipsis_pause_ms: int,
) -> tuple[np.ndarray, list[TimedTextWord], int]:
    if samples.size == 0 or ellipsis_pause_ms < 0:
        return samples, alignment_words, 0
    marker_indexes = [i for i, word in enumerate(alignment_words) if _is_cut_marker_word(word.text)]
    if not marker_indexes:
        return samples, alignment_words, 0

    pad_samples = int(round(max(0, CUT_MARKER_PAD_MS) / 1000 * SAMPLE_RATE_TTS))
    silence_samples = int(round(max(0, ellipsis_pause_ms) / 1000 * SAMPLE_RATE_TTS))
    edits: list[tuple[int, int]] = []
    for index in marker_indexes:
        word = alignment_words[index]
        start = max(0, int(round(word.start_ms / 1000 * SAMPLE_RATE_TTS)) - pad_samples)
        end = min(samples.size, int(round(word.end_ms / 1000 * SAMPLE_RATE_TTS)) + pad_samples)
        if end > start:
            edits.append((start, end))
    if not edits:
        return samples, alignment_words, 0
    edits.sort()
    merged: list[tuple[int, int]] = []
    for start, end in edits:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    pieces: list[np.ndarray] = []
    pos = 0
    removed_before = [0 for _ in merged]
    inserted_before = [0 for _ in merged]
    removed_total = 0
    inserted_total = 0
    silence = np.zeros(silence_samples, dtype=samples.dtype)
    for edit_index, (start, end) in enumerate(merged):
        pieces.append(samples[pos:start])
        pieces.append(silence)
        pos = end
        removed_before[edit_index] = removed_total
        inserted_before[edit_index] = inserted_total
        removed_total += end - start
        inserted_total += silence_samples
    pieces.append(samples[pos:])
    new_samples = np.concatenate(pieces).astype(samples.dtype, copy=False) if pieces else samples

    def adjust_ms(ms: int) -> int:
        sample = int(round(ms / 1000 * SAMPLE_RATE_TTS))
        shift = 0
        for edit_index, (start, end) in enumerate(merged):
            if sample >= end:
                shift = inserted_before[edit_index] + silence_samples - (removed_before[edit_index] + end - start)
                continue
            if sample >= start:
                return int(round((start + inserted_before[edit_index] - removed_before[edit_index]) / SAMPLE_RATE_TTS * 1000))
            break
        return max(0, int(round((sample + shift) / SAMPLE_RATE_TTS * 1000)))

    kept_words: list[TimedTextWord] = []
    for index, word in enumerate(alignment_words):
        if index in marker_indexes:
            continue
        kept_words.append(TimedTextWord(word.text, adjust_ms(word.start_ms), adjust_ms(word.end_ms)))
    return new_samples, kept_words, len(marker_indexes)


@dataclass
class StyledTextSpan:
    text: str
    instruct_suffix: str = ""  # OmniVoice-only style appended to the base instruct
    voice_name: str | None = None
    voice_prompt: str | None = None


@dataclass
class TextChunk:
    text: str
    pause_after: float
    suffix: str = " "  # whitespace that followed the TTS chunk ("\n\n" for paragraphs)
    instruct_suffix: str = ""  # OmniVoice-only style appended to the base instruct
    speech_text: str = ""  # TTS-only text; display uses text
    style_spans: tuple[StyledTextSpan, ...] = field(default_factory=tuple)
    display_markup_text: str = ""  # display text with per-character color markup


@dataclass
class DisplayEvent:
    start_ms: int
    text: str


@dataclass
class AudioChunk:
    samples: np.ndarray
    pause_after: float
    spoken_text: str = ""
    alignment_words: list[TimedTextWord] = field(default_factory=list)
    display_text: str = ""  # original sentence text, shown when playback starts
    display_events: list[DisplayEvent] = field(default_factory=list)
    owner: object = None    # the _TtsSession this belongs to (None = system cue)
    on_render: Callable[[float, int, int, int], None] | None = field(
        default=None, repr=False, compare=False
    )

    def heard_prefix(self, elapsed_seconds: float, safety_margin_ms: int) -> str:
        """Spoken text heard after elapsed_seconds of this chunk's playback."""
        if self.samples.size == 0:
            return ""
        duration_ms = self.samples.size / SAMPLE_RATE_TTS * 1000
        cutoff_ms = max(0, int(round(elapsed_seconds * 1000)) - max(0, safety_margin_ms))
        if cutoff_ms >= duration_ms - max(0, safety_margin_ms):
            return self.spoken_text.strip()
        if not self.alignment_words:
            return ""
        words = [word for word in self.alignment_words if word.end_ms <= cutoff_ms]
        return normalize_heard_text(join_timed_words(words))


DISPLAY_SENTENCE_RE = re.compile(
    r".+?(?:[.!?…][\"')\]}”’]*)(?:\s+|$)",
    re.DOTALL,
)
DISPLAY_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


DISPLAY_QUOTE_CHARS = "\"'“”‘’«»"


def display_words(text: str) -> list[str]:
    text = strip_audio_cue_tags_for_display(strip_color_markup(text))
    return [
        word.lower()
        for word in DISPLAY_WORD_RE.findall(normalize_spoken_text(text))
    ]


def display_markup_from_spans(
    text: str,
    style_spans: tuple[StyledTextSpan, ...],
    color_for_voice: Callable[[str | None], str | None],
) -> str:
    """Rebuild chunk text with inline color markup around named-voice spans.

    The style spans partition the chunk text exactly; the quotation marks that
    directly wrap a voiced span live in the neighboring unvoiced spans, so
    they are pulled into the colored region — dialogue quotes render in the
    speaking character's color, per the display design."""
    if not style_spans:
        return text
    segments: list[list] = [
        [span.text, color_for_voice(span.voice_name) if span.voice_name else None]
        for span in style_spans
        if span.text
    ]
    if "".join(segment[0] for segment in segments) != text:
        return text  # spans out of sync with the text; don't risk mangling it
    for index, segment in enumerate(segments):
        if not segment[1]:
            continue
        if index > 0 and not segments[index - 1][1]:
            prev = segments[index - 1]
            while prev[0] and prev[0][-1] in DISPLAY_QUOTE_CHARS:
                segment[0] = prev[0][-1] + segment[0]
                prev[0] = prev[0][:-1]
        if index + 1 < len(segments) and not segments[index + 1][1]:
            nxt = segments[index + 1]
            while nxt[0] and nxt[0][0] in DISPLAY_QUOTE_CHARS:
                segment[0] = segment[0] + nxt[0][0]
                nxt[0] = nxt[0][1:]
    return "".join(
        color_markup(segment_text, color) if color else segment_text
        for segment_text, color in segments
        if segment_text
    )


def split_display_sentences(text: str) -> list[str]:
    """Split display text into sentence-ish fragments, preserving whitespace."""
    if not text:
        return []
    chunker_cls = globals().get("StreamingSentenceChunker")
    if chunker_cls is not None:
        try:
            chunker = chunker_cls(0.0, 0.0, pause_boundaries=False)
            chunks = chunker.add(text) + chunker.flush()
            parts = [chunk.text + chunk.suffix for chunk in chunks if chunk.text]
            if parts:
                return parts
        except Exception:
            pass
    parts: list[str] = []
    pos = 0
    for match in DISPLAY_SENTENCE_RE.finditer(text):
        if match.start() > pos:
            prefix = text[pos:match.start()]
            if prefix:
                parts.append(prefix)
        part = match.group(0)
        if part:
            parts.append(part)
        pos = match.end()
    if pos < len(text):
        tail = text[pos:]
        if tail:
            parts.append(tail)
    return parts or [text]


def voice_text_with_dash_pauses(text: str) -> str:
    """Use ellipses around em dashes for TTS without changing displayed text."""
    return re.sub(r"[ \t]*—[ \t]*", " ... — ... ", text)


def display_word_count(text: str) -> int:
    return len(display_words(text))


def _alignment_word_key(word: TimedTextWord) -> str:
    words = display_words(word.text)
    return words[0] if words else ""


def _find_alignment_start(
    alignment_keys: list[str],
    start_index: int,
    targets: list[str],
) -> int | None:
    if not targets:
        return start_index
    probe = targets[: min(3, len(targets))]
    for index in range(start_index, len(alignment_keys)):
        cursor = index
        matched = True
        for target in probe:
            while cursor < len(alignment_keys) and not alignment_keys[cursor]:
                cursor += 1
            if cursor >= len(alignment_keys) or alignment_keys[cursor] != target:
                matched = False
                break
            cursor += 1
        if matched:
            return index
    return None


def build_aligned_display_events(
    display_parts: list[str],
    alignment_words: list[TimedTextWord],
) -> list[DisplayEvent]:
    if not display_parts or not alignment_words:
        return []
    events: list[DisplayEvent] = []
    alignment_keys = [_alignment_word_key(word) for word in alignment_words]
    word_index = 0
    for part in display_parts:
        words = display_words(part)
        if not words:
            start_ms = events[-1].start_ms if events else 0
            events.append(DisplayEvent(start_ms, part))
            continue
        if word_index >= len(alignment_words):
            return []
        start_probe = words[: min(3, len(words))]
        matched_index = _find_alignment_start(alignment_keys, word_index, start_probe)
        if matched_index is None:
            matched_index = word_index
        events.append(DisplayEvent(max(0, alignment_words[matched_index].start_ms), part))
        end_probe = words[-min(3, len(words)):]
        end_index = _find_alignment_start(alignment_keys, matched_index, end_probe)
        if end_index is None:
            word_index = matched_index + len(start_probe)
        else:
            word_index = end_index + len(end_probe)
        word_index = min(len(alignment_words), max(matched_index + 1, word_index))
    return events


class StreamingSentenceChunker:
    CLOSERS = set("\"')]}*") | {"”", "’"}
    ABBREVIATIONS = {
        "dr.", "e.g.", "etc.", "i.e.", "jr.", "mr.", "mrs.", "ms.",
        "prof.", "sr.", "st.", "vs.",
    }

    def __init__(
        self,
        sentence_pause: float,
        paragraph_pause: float,
        dash_pause: float | None = None,
        ellipsis_pause: float | None = None,
        pause_boundaries: bool = True,
    ) -> None:
        self.sentence_pause = sentence_pause
        self.paragraph_pause = paragraph_pause
        self.dash_pause = sentence_pause if dash_pause is None else dash_pause
        self.ellipsis_pause = sentence_pause if ellipsis_pause is None else ellipsis_pause
        self.pause_boundaries = pause_boundaries
        self.buffer = ""

    def add(self, text: str) -> list[TextChunk]:
        self.buffer += text
        return self._extract_chunks(final=False)

    def flush(self) -> list[TextChunk]:
        return self._extract_chunks(final=True)

    def _extract_chunks(self, final: bool) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        while True:
            boundary = self._find_sentence_boundary(final=final)
            if boundary is None:
                break
            text_end, whitespace_end, pause_after = boundary
            chunk_text = self.buffer[:text_end].strip()
            whitespace = self.buffer[text_end:whitespace_end]
            self.buffer = self.buffer[whitespace_end:]
            if chunk_text:
                chunks.append(
                    TextChunk(chunk_text, pause_after, display_suffix_for_whitespace(whitespace))
                )
        if final:
            chunk_text = self.buffer.strip()
            self.buffer = ""
            if chunk_text:
                chunks.append(TextChunk(chunk_text, 0.0))
        return chunks

    def _find_sentence_boundary(self, final: bool) -> tuple[int, int, float] | None:
        if not self.buffer:
            return None
        line_match = re.search(r"\n+", self.buffer)
        line_at = line_match.start() if line_match is not None else -1

        def line_boundary() -> tuple[int, int, float]:
            assert line_match is not None
            whitespace_end = line_match.end()
            while whitespace_end < len(self.buffer) and self.buffer[whitespace_end].isspace():
                whitespace_end += 1
            return line_at, whitespace_end, self.paragraph_pause

        pause_match = PAUSE_BOUNDARY_RE.search(self.buffer) if self.pause_boundaries else None

        def pause_boundary() -> tuple[int, int, float]:
            assert pause_match is not None
            if 0 <= line_at < pause_match.start():
                return line_boundary()
            text_end = pause_match.end()
            whitespace_end = text_end
            while whitespace_end < len(self.buffer) and self.buffer[whitespace_end].isspace():
                whitespace_end += 1
            whitespace = self.buffer[text_end:whitespace_end]
            pause_after = pause_for_boundary_token(
                pause_match.group(0),
                self.dash_pause,
                self.ellipsis_pause,
            )
            if "\n" in whitespace:
                pause_after = max(pause_after, self.paragraph_pause)
            return text_end, whitespace_end, pause_after

        for idx, char in enumerate(self.buffer):
            if pause_match is not None and pause_match.start() <= idx:
                return pause_boundary()
            if char not in ".!?":
                continue
            if (
                char == "."
                and (
                    (idx > 0 and self.buffer[idx - 1] == ".")
                    or (idx + 1 < len(self.buffer) and self.buffer[idx + 1] == ".")
                )
            ):
                continue
            if 0 <= line_at < idx:
                # A line break precedes this terminator — it ends the
                # chunk even without punctuation ("— H.P. Lovecraft\n\n…").
                return line_boundary()
            text_end = idx + 1
            while text_end < len(self.buffer) and self.buffer[text_end] in self.CLOSERS:
                text_end += 1
            whitespace_end = text_end
            while whitespace_end < len(self.buffer) and self.buffer[whitespace_end].isspace():
                whitespace_end += 1
            if self._looks_like_abbreviation(idx, final):
                continue
            if whitespace_end == text_end and text_end < len(self.buffer):
                continue
            whitespace = self.buffer[text_end:whitespace_end]
            if not final and whitespace_end == len(self.buffer) and "\n" not in whitespace:
                # The next word is not visible yet: a closing quote or a
                # lowercase attribution ('" she cried') may still stream in.
                continue
            if (
                whitespace_end < len(self.buffer)
                and self.buffer[whitespace_end].islower()
                and "\n" not in whitespace
            ):
                continue  # lowercase continuation: '"Stop!" she cried.'
            pause_after = self.paragraph_pause if "\n" in whitespace else self.sentence_pause
            return text_end, whitespace_end, pause_after
        if pause_match is not None:
            return pause_boundary()
        if line_at >= 0:
            return line_boundary()
        return None

    def _looks_like_abbreviation(self, period_index: int, final: bool = False) -> bool:
        prefix = self.buffer[: period_index + 1].strip().lower()
        if not prefix:
            return False
        parts = prefix.rsplit(maxsplit=2)
        token = parts[-1].strip("\"'([{")
        if token in self.ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.){2,}", token):
            return True
        # Single letter + period: a name initial ("H. P. Lovecraft"), the
        # start of a dotted abbreviation still streaming in ("e." of "e.g."),
        # or a genuine sentence end ("so did I.").
        if not re.fullmatch(r"[a-z]\.", token):
            return False
        prev = parts[-2].strip("\"'([{") if len(parts) >= 2 else ""
        if re.fullmatch(r"[a-z]\.", prev):
            return True  # mid-run: the P. of "H. P.", second A. of "A. A."
        j = period_index + 1
        while j < len(self.buffer) and (self.buffer[j] in self.CLOSERS or self.buffer[j].isspace()):
            j += 1
        rest = self.buffer[j:]
        if not final and (not rest or (len(rest) == 1 and rest.isupper())):
            return True  # can't tell yet — wait for more text
        if not self.buffer[period_index - 1].isupper():
            return False  # lowercase: dotted runs were caught above
        if re.match(r"[A-Z]\.", rest):
            return True  # next token is another initial ("H." before "P.")
        # "a"/"i" are real words, so "an A." / "was I." usually end sentences;
        # other lone capitals before a capitalized word are middle initials
        # ("John D. Rockefeller").
        return token[0] not in "ai" and bool(rest) and rest[0].isupper()


class _StreamingOmniVoiceCoreChunker(StreamingSentenceChunker):
    """Emit the first sentence promptly, then synthesize paragraph-sized chunks."""

    def __init__(
        self,
        sentence_pause: float,
        paragraph_pause: float,
        dash_pause: float | None = None,
        ellipsis_pause: float | None = None,
    ) -> None:
        super().__init__(
            sentence_pause,
            paragraph_pause,
            dash_pause,
            ellipsis_pause,
            pause_boundaries=False,
        )
        self.first_sentence_emitted = False

    def _extract_chunks(self, final: bool) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        if not self.first_sentence_emitted:
            boundary = self._find_sentence_boundary(final=final)
            if boundary is None:
                if not final:
                    return chunks
                self.first_sentence_emitted = True
            else:
                text_end, whitespace_end, pause_after = boundary
                chunk_text = self.buffer[:text_end].strip()
                whitespace = self.buffer[text_end:whitespace_end]
                self.buffer = self.buffer[whitespace_end:]
                self.first_sentence_emitted = True
                if chunk_text:
                    chunks.append(
                        TextChunk(chunk_text, pause_after, display_suffix_for_whitespace(whitespace))
                    )

        while True:
            line_match = re.search(r"\n+", self.buffer)
            if line_match is None:
                break
            whitespace_end = line_match.end()
            while whitespace_end < len(self.buffer) and self.buffer[whitespace_end].isspace():
                whitespace_end += 1
            whitespace = self.buffer[line_match.start():whitespace_end]
            chunk_text = self.buffer[:line_match.start()].strip()
            self.buffer = self.buffer[whitespace_end:]
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_text,
                        self.paragraph_pause,
                        display_suffix_for_whitespace(whitespace),
                    )
                )

        if final:
            chunk_text = self.buffer.strip()
            self.buffer = ""
            if chunk_text:
                chunks.append(TextChunk(chunk_text, 0.0))
        return chunks


def _possible_omnivoice_tag_prefix(text: str) -> bool:
    if not text.startswith("<"):
        return False
    body = text[1:].lower().lstrip()
    if body.startswith("/"):
        body = body[1:].lstrip()
    return (
        not body
        or "whisper".startswith(body)
        or "voice".startswith(body)
        or body.startswith("voice")
    )


def _parse_voice_tag(candidate: str) -> tuple[bool, str | None, str | None] | None:
    match = VOICE_TAG_RE.fullmatch(candidate)
    if match is None:
        return None
    closing = bool(match.group(1))
    if closing:
        return True, None, None
    attrs_text = match.group(2) or ""
    attrs: dict[str, str] = {}
    for attr_match in VOICE_ATTR_RE.finditer(attrs_text):
        key = attr_match.group(1).casefold()
        value = next(
            item
            for item in attr_match.groups()[1:]
            if item is not None
        )
        attrs[key] = html.unescape(value).strip()
    name = attrs.get("name", "").strip()
    prompt = attrs.get("prompt", "").strip() or None
    return False, name or None, prompt


class _StreamingOmniVoiceTagParser:
    """Strip OmniVoice tags while tracking whisper state and character voice."""

    def __init__(self) -> None:
        self.pending = ""
        self.whisper_active = False
        self.voice_name: str | None = None
        self.voice_prompt: str | None = None

    def feed(self, text: str) -> list[tuple[str, object]]:
        self.pending += text
        return self._extract(final=False)

    def flush(self) -> list[tuple[str, object]]:
        return self._extract(final=True)

    def _extract(self, final: bool) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        while self.pending:
            tag_start = self.pending.find("<")
            if tag_start < 0:
                if self.pending:
                    events.append(("text", self.pending))
                    self.pending = ""
                break

            if tag_start > 0:
                events.append(("text", self.pending[:tag_start]))
                self.pending = self.pending[tag_start:]
                continue

            tag_end = self.pending.find(">")
            if tag_end < 0:
                if final:
                    if not _possible_omnivoice_tag_prefix(self.pending):
                        events.append(("text", self.pending))
                    self.pending = ""
                break

            candidate = self.pending[:tag_end + 1]
            whisper_match = WHISPER_TAG_RE.fullmatch(candidate)
            voice_tag = _parse_voice_tag(candidate)
            if whisper_match is None and voice_tag is None:
                events.append(("text", self.pending[:1]))
                self.pending = self.pending[1:]
                continue

            self.pending = self.pending[tag_end + 1:]
            if whisper_match is not None:
                new_state = not bool(whisper_match.group(1))
                if new_state != self.whisper_active:
                    self.whisper_active = new_state
                    events.append(
                        ("style", (self.whisper_active, self.voice_name, self.voice_prompt))
                    )
                continue

            if voice_tag is not None:
                closing, name, prompt = voice_tag
                new_voice_name = None if closing else name
                new_voice_prompt = None if closing else prompt
                if (
                    new_voice_name != self.voice_name
                    or new_voice_prompt != self.voice_prompt
                ):
                    self.voice_name = new_voice_name
                    self.voice_prompt = new_voice_prompt
                    events.append(
                        ("style", (self.whisper_active, self.voice_name, self.voice_prompt))
                    )
        return events


class StreamingOmniVoiceChunker:
    """OmniVoice chunker with stream-safe style tag support."""

    def __init__(
        self,
        sentence_pause: float,
        paragraph_pause: float,
        dash_pause: float | None = None,
        ellipsis_pause: float | None = None,
        whisper_tags_enabled: bool = False,
        voice_tag_callback: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.core = _StreamingOmniVoiceCoreChunker(
            sentence_pause,
            paragraph_pause,
            dash_pause,
            ellipsis_pause,
        )
        self.parser = _StreamingOmniVoiceTagParser()
        self.whisper_active = False
        self.voice_name: str | None = None
        self.voice_prompt: str | None = None
        self.whisper_tags_enabled = whisper_tags_enabled
        self.voice_tag_callback = voice_tag_callback
        self.pending_spans: deque[StyledTextSpan] = deque()

    def add(self, text: str) -> list[TextChunk]:
        return self._handle_events(self.parser.feed(text), final=False)

    def flush(self) -> list[TextChunk]:
        return self._handle_events(self.parser.flush(), final=True)

    def _handle_events(
        self,
        events: list[tuple[str, object]],
        *,
        final: bool,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for kind, payload in events:
            if kind == "text":
                text = str(payload)
                self._append_span(
                    text,
                    self.whisper_active and self.whisper_tags_enabled,
                    self.voice_name,
                    self.voice_prompt,
                )
                chunks.extend(self._attach_spans(self.core.add(text)))
                continue

            if kind == "style":
                whisper_active, voice_name, voice_prompt = payload  # type: ignore[misc]
                self.whisper_active = bool(whisper_active)
                self.voice_name = str(voice_name) if voice_name is not None else None
                self.voice_prompt = str(voice_prompt) if voice_prompt is not None else None
                if self.voice_name and self.voice_tag_callback is not None:
                    self.voice_tag_callback(self.voice_name, self.voice_prompt)

        if final:
            chunks.extend(self._attach_spans(self.core.flush()))
            self.pending_spans.clear()
        return chunks

    def _append_span(
        self,
        text: str,
        whisper: bool,
        voice_name: str | None,
        voice_prompt: str | None,
    ) -> None:
        if not text:
            return
        styles: list[str] = []
        if whisper:
            styles.append(OMNIVOICE_WHISPER_INSTRUCT_SUFFIX)
        instruct_suffix = ", ".join(styles)
        if (
            self.pending_spans
            and self.pending_spans[-1].instruct_suffix == instruct_suffix
            and self.pending_spans[-1].voice_name == voice_name
            and self.pending_spans[-1].voice_prompt == voice_prompt
        ):
            last = self.pending_spans.pop()
            self.pending_spans.append(
                StyledTextSpan(last.text + text, instruct_suffix, voice_name, voice_prompt)
            )
            return
        self.pending_spans.append(StyledTextSpan(text, instruct_suffix, voice_name, voice_prompt))

    def _drop_pending_prefix(self, count: int) -> None:
        remaining = max(0, count)
        while remaining and self.pending_spans:
            span = self.pending_spans[0]
            if len(span.text) <= remaining:
                remaining -= len(span.text)
                self.pending_spans.popleft()
            else:
                self.pending_spans[0] = StyledTextSpan(
                    span.text[remaining:],
                    span.instruct_suffix,
                    span.voice_name,
                    span.voice_prompt,
                )
                remaining = 0

    def _discard_pending_leading_space(self) -> None:
        while self.pending_spans:
            span = self.pending_spans[0]
            stripped = span.text.lstrip()
            removed = len(span.text) - len(stripped)
            if removed <= 0:
                return
            self._drop_pending_prefix(removed)

    @staticmethod
    def _merge_style_spans(spans: list[StyledTextSpan]) -> tuple[StyledTextSpan, ...]:
        merged: list[StyledTextSpan] = []
        for span in spans:
            if not span.text:
                continue
            if (
                merged
                and merged[-1].instruct_suffix == span.instruct_suffix
                and merged[-1].voice_name == span.voice_name
                and merged[-1].voice_prompt == span.voice_prompt
            ):
                merged[-1] = StyledTextSpan(
                    merged[-1].text + span.text,
                    span.instruct_suffix,
                    span.voice_name,
                    span.voice_prompt,
                )
            else:
                merged.append(span)
        return tuple(merged)

    def _consume_style_spans(self, text: str) -> tuple[StyledTextSpan, ...]:
        self._discard_pending_leading_space()
        remaining = len(text)
        spans: list[StyledTextSpan] = []
        while remaining and self.pending_spans:
            span = self.pending_spans[0]
            take = min(remaining, len(span.text))
            spans.append(
                StyledTextSpan(
                    span.text[:take],
                    span.instruct_suffix,
                    span.voice_name,
                    span.voice_prompt,
                )
            )
            if take == len(span.text):
                self.pending_spans.popleft()
            else:
                self.pending_spans[0] = StyledTextSpan(
                    span.text[take:],
                    span.instruct_suffix,
                    span.voice_name,
                    span.voice_prompt,
                )
            remaining -= take
        if remaining:
            spans.append(StyledTextSpan(text[len(text) - remaining:], ""))
        return self._merge_style_spans(spans)

    def _attach_spans(self, chunks: list[TextChunk]) -> list[TextChunk]:
        styled: list[TextChunk] = []
        for chunk in chunks:
            spans = self._consume_style_spans(chunk.text)
            styled.append(
                TextChunk(
                    chunk.text,
                    chunk.pause_after,
                    chunk.suffix,
                    chunk.instruct_suffix,
                    chunk.speech_text,
                    spans,
                )
            )
        return styled


# ── whisper.cpp ──────────────────────────────────────────────────────────────

def _terminate_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)
    finally:
        # Close the pipes we never drained so their FDs don't leak.
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass


def run_whisper_process(
    cmd: list[str],
    cancel_event: threading.Event | None,
    timeout_seconds: float,
) -> tuple[int, str, str]:
    """Run whisper-cli, honoring cancellation and a hard timeout so a wedged
    process can never silently freeze the voice pipeline."""
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + timeout_seconds
    # Drain the pipes via communicate() in short slices so a chatty process can
    # never deadlock on a full OS pipe buffer, while still honoring cancellation
    # and the hard timeout. communicate() retains partial output across retries.
    while True:
        if cancel_event is not None and cancel_event.is_set():
            _terminate_process(process)
            raise CancelledError()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(process)
            raise RuntimeError(f"whisper-cli timed out after {timeout_seconds:.0f}s")
        try:
            stdout, stderr = process.communicate(timeout=min(0.02, remaining))
        except subprocess.TimeoutExpired:
            continue
        return process.returncode, stdout, stderr


class WhisperCppTranscriber:
    def __init__(
        self,
        whisper_cli: Path,
        whisper_model: Path,
        vad_model: Path,
        language: str,
        threads: int,
        vad_threshold: float,
        vad_min_speech_ms: int,
        vad_min_silence_ms: int,
        initial_prompt: str = "",
    ) -> None:
        self.whisper_cli = Path(whisper_cli)
        self.whisper_model = Path(whisper_model)
        self.vad_model = Path(vad_model)
        self.language = language
        self.threads = threads
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = vad_min_speech_ms
        self.vad_min_silence_ms = vad_min_silence_ms
        self.initial_prompt = initial_prompt

    def check(self) -> None:
        for path in (self.whisper_cli, self.whisper_model, self.vad_model):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; set IF_ENGINE_WHISPER_DIR (or _CLI/_MODEL/_VAD_MODEL) "
                    "to a whisper.cpp checkout with a built whisper-cli and models"
                )

    def transcribe_samples(self, samples: np.ndarray, cancel_event: threading.Event | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="if-engine-voice-") as td:
            tmpdir = Path(td)
            wav_path = tmpdir / "utterance.wav"
            out_base = tmpdir / "transcript"
            json_path = out_base.with_suffix(".json")
            write_wav(wav_path, samples, SAMPLE_RATE_IN)
            cmd = [
                str(self.whisper_cli),
                "-m", str(self.whisper_model),
                "--vad",
                "--vad-model", str(self.vad_model),
                "--vad-threshold", str(self.vad_threshold),
                "--vad-min-speech-duration-ms", str(self.vad_min_speech_ms),
                "--vad-min-silence-duration-ms", str(self.vad_min_silence_ms),
                "-f", str(wav_path),
                "-np", "-nt",
                "-l", self.language,
                "-t", str(self.threads),
                "-oj",
                "-of", str(out_base),
            ]
            prompt = clean_transcript(self.initial_prompt)
            if prompt:
                cmd.extend(["--prompt", prompt[:1000], "--carry-initial-prompt"])
            returncode, stdout, stderr = run_whisper_process(
                cmd, cancel_event, TRANSCRIBE_TIMEOUT_SECONDS
            )
            if returncode != 0:
                raise RuntimeError(
                    f"whisper-cli failed\ncommand: {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )
            if not json_path.exists():
                raise RuntimeError(f"whisper-cli did not create {json_path}")
            data = json.loads(json_path.read_text())
            parts = [item.get("text", "") for item in data.get("transcription", [])]
            return clean_transcript(" ".join(parts))

    def align_tts_samples(
        self,
        samples: np.ndarray,
        prompt_text: str,
        cancel_event: threading.Event | None = None,
    ) -> list[TimedTextWord]:
        """Word timestamps for generated TTS audio, so a barge-in can report
        exactly how much of a sentence the player heard."""
        if samples.size == 0:
            return []
        with tempfile.TemporaryDirectory(prefix="if-engine-align-") as td:
            tmpdir = Path(td)
            wav_path = tmpdir / "tts.wav"
            out_base = tmpdir / "alignment"
            json_path = out_base.with_suffix(".json")
            write_wav(wav_path, samples, SAMPLE_RATE_TTS)
            cmd = [
                str(self.whisper_cli),
                "-m", str(self.whisper_model),
                "-f", str(wav_path),
                "-np",
                "-l", self.language,
                "-t", str(self.threads),
                "-oj", "-ojf",
                "-of", str(out_base),
            ]
            prompt = clean_transcript(prompt_text)
            if prompt:
                cmd.extend(["--prompt", prompt[:1000]])
            returncode, stdout, stderr = run_whisper_process(
                cmd, cancel_event, ALIGN_TIMEOUT_SECONDS
            )
            if returncode != 0:
                raise RuntimeError(
                    f"whisper-cli alignment failed\ncommand: {' '.join(cmd)}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            if not json_path.exists():
                raise RuntimeError(f"whisper-cli did not create {json_path}")
            return whisper_json_alignment_words(json.loads(json_path.read_text()))


# ── Kokoro TTS ───────────────────────────────────────────────────────────────

def default_torch_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class KokoroSpeaker:
    def __init__(self, lang: str, voice: str, speed: float, device: str, volume: float) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("kokoro is not installed") from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
        except Exception:
            # Cache miss (first run on this machine): allow one online fetch.
            import huggingface_hub.constants as hf_constants
            if not hf_constants.HF_HUB_OFFLINE:
                raise
            print("downloading Kokoro model (first run)…", file=sys.stderr)
            hf_constants.HF_HUB_OFFLINE = False
            self.pipeline = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
        self.voice = voice
        self.speed = speed
        self.volume = volume

    def synthesize(self, text: str, cancel_event: threading.Event | None = None) -> np.ndarray:
        chunks: list[np.ndarray] = []
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        for result in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        samples = np.concatenate(chunks).astype(np.float32)
        return postprocess_tts_audio(samples, volume=self.volume)


class OmniVoiceSpeaker:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype_name: str,
        num_step: int,
        instruct: str,
        speed: float | None,
        clone_reference: Path | None,
        clone_transcript: str,
        clone_transcript_path: Path,
        whisper_clone_reference: Path | None,
        whisper_clone_transcript: str,
        whisper_clone_transcript_path: Path,
        character_voice_cache: CharacterVoiceCache | None,
        volume: float,
    ) -> None:
        try:
            from omnivoice import OmniVoice, OmniVoiceGenerationConfig
        except ImportError as exc:
            raise RuntimeError("omnivoice is not installed") from exc
        self._disable_model_progress_bars()
        if device == "auto":
            device = default_torch_device()
        if device == "cpu" and dtype_name == "float16":
            dtype_name = "float32"
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise RuntimeError(f"unknown torch dtype for OmniVoice: {dtype_name}") from exc
        try:
            self.model = OmniVoice.from_pretrained(model_name, device_map=device, dtype=dtype)
        except Exception:
            # Cache miss (first run on this machine): allow one online fetch.
            import huggingface_hub.constants as hf_constants
            if not hf_constants.HF_HUB_OFFLINE:
                raise
            print("downloading OmniVoice model (first run)…", file=sys.stderr)
            hf_constants.HF_HUB_OFFLINE = False
            self._disable_model_progress_bars()
            self.model = OmniVoice.from_pretrained(model_name, device_map=device, dtype=dtype)
        self.config = OmniVoiceGenerationConfig(num_step=num_step)
        self.instruct = instruct
        self.volume = volume
        self.device = device
        self.dtype_name = dtype_name
        self.num_step = num_step
        self.speed = speed
        self.model_name = model_name
        self.character_voice_cache = character_voice_cache
        self.character_voice_clone_prompts: dict[str, tuple[object, Path, str]] = {}
        self.character_voice_clone_prompt_misses: set[str] = set()
        self.clone_reference = None
        self.voice_clone_prompt = None
        if clone_reference is not None:
            self.voice_clone_prompt, self.clone_reference = self._load_voice_clone_prompt(
                clone_reference=clone_reference,
                clone_transcript=clone_transcript,
                clone_transcript_path=clone_transcript_path,
                env_var="IF_ENGINE_OMNIVOICE_CLONE_REFERENCE",
                label="OmniVoice clone reference",
            )
        self.whisper_clone_reference = None
        self.whisper_voice_clone_prompt = None
        if whisper_clone_reference is not None:
            (
                self.whisper_voice_clone_prompt,
                self.whisper_clone_reference,
            ) = self._load_voice_clone_prompt(
                clone_reference=whisper_clone_reference,
                clone_transcript=whisper_clone_transcript,
                clone_transcript_path=whisper_clone_transcript_path,
                env_var="IF_ENGINE_OMNIVOICE_WHISPER_CLONE_REFERENCE",
                label="OmniVoice whisper clone reference",
            )

    @staticmethod
    def _disable_model_progress_bars() -> None:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
        try:
            from transformers.utils import logging as transformers_logging
            transformers_logging.disable_progress_bar()
            transformers_logging.set_verbosity_error()
        except Exception:
            pass

    def _load_voice_clone_prompt(
        self,
        *,
        clone_reference: Path,
        clone_transcript: str,
        clone_transcript_path: Path,
        env_var: str,
        label: str,
    ) -> tuple[object | None, Path | None]:
        ref_path = clone_reference.expanduser()
        explicit_ref = env_var in os.environ
        if not ref_path.exists():
            if explicit_ref:
                raise RuntimeError(f"{label} not found: {ref_path}")
            return None, None
        if not ref_path.is_file():
            raise RuntimeError(f"{label} is not a file: {ref_path}")

        ref_text = clone_transcript.strip()
        transcript_path = clone_transcript_path.expanduser()
        if not ref_text and transcript_path.is_file():
            ref_text = transcript_path.read_text(encoding="utf-8").strip()

        prompt = self.model.create_voice_clone_prompt(
            str(ref_path),
            ref_text=ref_text or None,
        )
        return prompt, ref_path

    def _instruct_with_suffix(self, suffix: str = "", base_instruct: str | None = None) -> str:
        base_items = [item.strip() for item in (base_instruct or self.instruct).split(",") if item.strip()]
        suffix_items = [item.strip() for item in suffix.split(",") if item.strip()]
        existing = {item.lower() for item in base_items}
        for item in suffix_items:
            if item.lower() not in existing:
                base_items.append(item)
                existing.add(item.lower())
        return ", ".join(base_items)

    def ensure_character_voice_started(self, voice_name: str, prompt_hint: str | None = None) -> None:
        if self.character_voice_cache is None:
            return
        self.character_voice_cache.ensure_started(voice_name, prompt_hint)

    def _character_voice_clone_prompt(
        self,
        voice_name: str,
        prompt_hint: str | None,
        cancel_event: threading.Event | None,
    ) -> tuple[object, str] | None:
        if self.character_voice_cache is None:
            return None
        if voice_name in self.character_voice_clone_prompt_misses:
            return None
        try:
            metadata = self.character_voice_cache.wait_for_ready(
                voice_name,
                prompt_hint,
                cancel_event=cancel_event,
            )
        except CharacterVoiceCancelled as exc:
            raise CancelledError() from exc
        if metadata is None:
            self.character_voice_clone_prompt_misses.add(voice_name)
            return None
        key = metadata.normalized_name
        if key in self.character_voice_clone_prompts:
            cached_prompt, _ref_path, cached_instruct = self.character_voice_clone_prompts[key]
            return cached_prompt, cached_instruct
        try:
            prompt, ref_path = self._load_voice_clone_prompt(
                clone_reference=metadata.preview_path,
                clone_transcript=metadata.preview_text,
                clone_transcript_path=Path(),
                env_var="",
                label=f"OmniVoice character voice {metadata.character_name}",
            )
        except Exception as exc:
            self.character_voice_clone_prompt_misses.add(voice_name)
            if self.character_voice_cache is not None:
                self.character_voice_cache._emit(
                    "voice: character voice "
                    f"'{metadata.character_name}' clone prompt unavailable: {exc}"
                )
            return None
        if prompt is None or ref_path is None:
            self.character_voice_clone_prompt_misses.add(voice_name)
            return None
        character_instruct = metadata.omnivoice_description.strip()
        self.character_voice_clone_prompts[key] = (prompt, ref_path, character_instruct)
        return prompt, character_instruct

    def _resolve_style_instruct(
        self,
        suffix: str,
        *,
        voice_name: str | None = None,
        voice_prompt_hint: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, object | None, bool, bool]:
        base_instruct = self.instruct
        styles: list[str] = []
        character_clone_prompt: object | None = None
        whisper = False
        character_voice = bool(voice_name)
        if voice_name:
            character_clone = self._character_voice_clone_prompt(
                voice_name,
                voice_prompt_hint,
                cancel_event,
            )
            if character_clone is not None:
                character_clone_prompt, character_instruct = character_clone
                if character_instruct:
                    base_instruct = character_instruct
        for item in (part.strip() for part in suffix.split(",")):
            if not item:
                continue
            lowered = item.lower()
            styles.append(item)
            if lowered == OMNIVOICE_WHISPER_INSTRUCT_SUFFIX:
                whisper = True
        return (
            self._instruct_with_suffix(", ".join(styles), base_instruct),
            character_clone_prompt,
            character_voice,
            whisper,
        )

    def synthesize(
        self,
        text: str,
        cancel_event: threading.Event | None = None,
        instruct_suffix: str = "",
        voice_name: str | None = None,
        voice_prompt_hint: str | None = None,
    ) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        instruct, character_clone_prompt, character_voice, whisper = self._resolve_style_instruct(
            instruct_suffix,
            voice_name=voice_name,
            voice_prompt_hint=voice_prompt_hint,
            cancel_event=cancel_event,
        )
        with torch.inference_mode():
            if character_clone_prompt is not None:
                audios = self.model.generate(
                    text=text,
                    language="en",
                    voice_clone_prompt=character_clone_prompt,
                    instruct=instruct,
                    speed=self.speed,
                    generation_config=self.config,
                )
            elif whisper and not character_voice and self.whisper_voice_clone_prompt is not None:
                audios = self.model.generate(
                    text=text,
                    language="en",
                    voice_clone_prompt=self.whisper_voice_clone_prompt,
                    speed=self.speed,
                    generation_config=self.config,
                )
            elif self.voice_clone_prompt is not None and not instruct_suffix and not voice_name:
                kwargs = {
                    "text": text,
                    "language": "en",
                    "voice_clone_prompt": self.voice_clone_prompt,
                    "speed": self.speed,
                    "generation_config": self.config,
                }
                audios = self.model.generate(**kwargs)
            else:
                audios = self.model.generate(
                    text=text,
                    language="en",
                    instruct=instruct,
                    speed=self.speed,
                    generation_config=self.config,
                )
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        if not audios:
            return np.zeros(0, dtype=np.float32)
        samples = mono_float32(audios[0])
        sample_rate = int(getattr(self.model, "sampling_rate", None) or SAMPLE_RATE_TTS)
        if sample_rate != SAMPLE_RATE_TTS and samples.size:
            samples = resample_linear(samples, sample_rate, SAMPLE_RATE_TTS)
        return postprocess_tts_audio(samples, volume=self.volume)


# ── AEC reference + WebRTC audio processing ──────────────────────────────────

class PlaybackReference:
    """Recently played output audio (at 16 kHz), timestamped with DAC time, so
    the AEC can line up the far-end signal with what the mic actually heard."""

    def __init__(self, max_seconds: float = 4.0) -> None:
        self._lock = threading.Lock()
        self._frames: deque[np.ndarray] = deque()
        self._pending = np.zeros(0, dtype=np.float32)
        self._timed_segments: deque[tuple[float, np.ndarray]] = deque()
        self._max_frames = max(1, int((max_seconds * SAMPLE_RATE_IN) / AEC_FRAME_SIZE))
        self._max_seconds = max_seconds

    def push_samples(self, samples: np.ndarray, start_time: float | None = None) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return
        if start_time is not None:
            with self._lock:
                self._timed_segments.append((float(start_time), samples.copy()))
                cutoff = float(start_time) - self._max_seconds
                while self._timed_segments:
                    segment_start, segment_samples = self._timed_segments[0]
                    segment_end = segment_start + segment_samples.size / SAMPLE_RATE_IN
                    if segment_end >= cutoff:
                        break
                    self._timed_segments.popleft()
            return
        with self._lock:
            pending = np.concatenate((self._pending, samples))
            while pending.size >= AEC_FRAME_SIZE:
                self._frames.append(pending[:AEC_FRAME_SIZE].copy())
                pending = pending[AEC_FRAME_SIZE:]
                while len(self._frames) > self._max_frames:
                    self._frames.popleft()
            self._pending = pending.copy()

    def flush(self) -> None:
        with self._lock:
            if self._pending.size == 0:
                return
            frame = np.zeros(AEC_FRAME_SIZE, dtype=np.float32)
            frame[: self._pending.size] = self._pending
            self._frames.append(frame)
            while len(self._frames) > self._max_frames:
                self._frames.popleft()
            self._pending = np.zeros(0, dtype=np.float32)

    def read_frame(self, frame_size: int = AEC_FRAME_SIZE) -> np.ndarray:
        with self._lock:
            if frame_size == AEC_FRAME_SIZE and self._frames:
                return self._frames.popleft()
        return np.zeros(frame_size, dtype=np.float32)

    def read_frame_at(self, start_time: float, frame_size: int = AEC_FRAME_SIZE) -> np.ndarray | None:
        with self._lock:
            segments = list(self._timed_segments)
        if not segments:
            return None
        output = np.zeros(frame_size, dtype=np.float32)
        end_time = start_time + frame_size / SAMPLE_RATE_IN
        for segment_start, samples in segments:
            segment_end = segment_start + samples.size / SAMPLE_RATE_IN
            if segment_end <= start_time:
                continue
            if segment_start >= end_time:
                break
            dst_start = max(0, int(round((segment_start - start_time) * SAMPLE_RATE_IN)))
            src_start = max(0, int(round((start_time - segment_start) * SAMPLE_RATE_IN)))
            count = min(frame_size - dst_start, samples.size - src_start)
            if count > 0:
                output[dst_start: dst_start + count] = samples[src_start: src_start + count]
        return output

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._timed_segments.clear()
            self._pending = np.zeros(0, dtype=np.float32)


class WebRtcAec:
    def __init__(
        self,
        playback_reference: PlaybackReference,
        delay_ms: int,
        enable_ns: bool,
        enable_agc: bool,
        reference_delay_ms: int = 0,
    ) -> None:
        if AudioProcessor is None:
            raise RuntimeError(
                "aec-audio-processing is not installed; run "
                ".venv/bin/pip install aec-audio-processing"
        )
        self.playback_reference = playback_reference
        self.reference_delay_seconds = max(0, int(reference_delay_ms)) / 1000.0
        self._lock = threading.Lock()
        self.processor = AudioProcessor(
            enable_aec=True,
            enable_ns=enable_ns,
            enable_agc=enable_agc,
            enable_vad=False,
        )
        self.processor.set_stream_format(SAMPLE_RATE_IN, 1, SAMPLE_RATE_IN, 1)
        self.processor.set_reverse_stream_format(SAMPLE_RATE_IN, 1)
        self.processor.set_stream_delay(delay_ms)
        frame_size = self.processor.get_frame_size()
        if frame_size != AEC_FRAME_SIZE:
            raise RuntimeError(f"Unexpected WebRTC AEC frame size {frame_size}; expected {AEC_FRAME_SIZE}")

    def process(self, capture_frame: np.ndarray, frame_time: float) -> np.ndarray:
        capture_frame = np.asarray(capture_frame, dtype=np.float32)
        if capture_frame.size != AEC_FRAME_SIZE:
            raise ValueError(f"AEC capture frame must be {AEC_FRAME_SIZE} samples, got {capture_frame.size}")
        with self._lock:
            reverse_frame = self.playback_reference.read_frame_at(
                frame_time - self.reference_delay_seconds,
                AEC_FRAME_SIZE,
            )
            if reverse_frame is None:
                reverse_frame = self.playback_reference.read_frame(AEC_FRAME_SIZE)
            self.processor.process_reverse_stream(float_to_pcm16_bytes(reverse_frame))
            output = self.processor.process_stream(float_to_pcm16_bytes(capture_frame))
        return pcm16_bytes_to_float(output)


# ── Queued playback ──────────────────────────────────────────────────────────

@dataclass
class QueuedPlaybackItem:
    chunk: AudioChunk
    done: threading.Event = field(default_factory=threading.Event)


# One CoreAudio render stream stays open for the app's whole lifetime —
# open/close churn (a stream per narration plus cue streams) triggers AUHAL
# teardown races that wedge CoreAudio and kill audio entirely. Cancelling a
# narration just drops its queued items (chunks carry an owner; items whose
# owner is no longer active are skipped instantly).
class QueuedPlaybackHandle:
    def __init__(
        self,
        output_device,
        channels: int,
        output_sample_rate: int,
        blocksize: int,
        latency,
        playback_reference: PlaybackReference | None = None,
        pause_event: threading.Event | None = None,
        on_chunk_start: Callable[[AudioChunk], None] | None = None,
        on_chunk_done: Callable[[AudioChunk], None] | None = None,
        notify: Callable[[str], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.output_device = output_device
        self.channels = max(1, channels)
        self.output_sample_rate = output_sample_rate
        self.blocksize = blocksize
        self.latency = latency
        self.playback_reference = playback_reference
        self.pause_event = pause_event
        self.on_chunk_start = on_chunk_start
        self.on_chunk_done = on_chunk_done
        self.notify = notify or (lambda message: print(message, file=sys.stderr))
        self.log = log
        self.callbacks = 0  # render callbacks seen; 0 after open = dead stream
        self.active_owner: object = None  # only this owner's (or ownerless) items play
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.error: BaseException | None = None
        self._items: queue.Queue[QueuedPlaybackItem | None] = queue.Queue()
        self._events: queue.Queue[tuple[str, QueuedPlaybackItem]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="if-engine-playback", daemon=True)
        self._position_lock = threading.Lock()
        self._played_position = 0
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, chunk: AudioChunk) -> threading.Event:
        item = QueuedPlaybackItem(chunk)
        self._items.put(item)
        return item.done

    def set_active_owner(self, owner: object) -> None:
        """Items whose chunk.owner is neither None nor this owner are dropped
        the moment the render callback reaches (or is playing) them."""
        self.active_owner = owner

    def finish(self) -> None:
        self._items.put(None)

    def stop(self) -> None:
        self.stop_event.set()

    def pause(self) -> None:
        if self.pause_event is not None:
            self.pause_event.set()
        if self.playback_reference is not None:
            self.playback_reference.flush()

    def resume(self) -> None:
        if self.pause_event is not None:
            self.pause_event.clear()

    def wait(self, timeout: float | None = None) -> bool:
        """Join the render thread. Returns False if it is still alive after
        timeout (a wedged CoreAudio close) — callers must abandon it rather
        than block the narration pipeline."""
        self._thread.join(timeout)
        if self._thread.is_alive():
            return False
        self._print_statuses()
        if self.error is not None:
            raise self.error
        return True

    def played_seconds(self) -> float:
        """Seconds played of the chunk currently being rendered."""
        with self._position_lock:
            return self._played_position / self.output_sample_rate

    def _run(self) -> None:
        current_item: QueuedPlaybackItem | None = None
        current_output = np.zeros(0, dtype=np.float32)
        current_position = 0
        reference_resampler = (
            StreamingLinearResampler(self.output_sample_rate, SAMPLE_RATE_IN)
            if self.playback_reference is not None
            else None
        )

        def owner_ok(item: QueuedPlaybackItem) -> bool:
            owner = item.chunk.owner
            return owner is None or owner is self.active_owner

        def start_item(item: QueuedPlaybackItem) -> None:
            nonlocal current_item, current_output, current_position
            current_item = item
            current_output = resample_linear(item.chunk.samples, SAMPLE_RATE_TTS, self.output_sample_rate)
            current_position = 0
            with self._position_lock:
                self._played_position = 0
            self._events.put(("start", item))

        def load_next_item() -> bool:
            while not self._closed:
                try:
                    item = self._items.get_nowait()
                except queue.Empty:
                    return False
                if item is None:
                    self._closed = True
                    return False
                if not owner_ok(item):
                    self._events.put(("drop", item))
                    continue
                start_item(item)
                return True
            return False

        def callback(outdata, frames, time_info, status):
            nonlocal current_item, current_output, current_position
            self.callbacks += 1
            if status:
                try:
                    self.status_queue.put_nowait(str(status))
                except queue.Full:
                    pass
            rendered = np.zeros(frames, dtype=np.float32)
            output_time = callback_time_seconds(time_info, "outputBufferDacTime")
            if self.stop_event.is_set():
                outdata.fill(0)
                raise sd.CallbackStop
            if self.pause_event is not None and self.pause_event.is_set():
                outdata.fill(0)
                if self.playback_reference is not None and reference_resampler is not None:
                    self.playback_reference.push_samples(reference_resampler.process(rendered), output_time)
                return
            offset = 0
            while offset < frames:
                if current_item is not None and not owner_ok(current_item):
                    # Owner was cancelled mid-chunk: cut its audio immediately.
                    self._events.put(("drop", current_item))
                    current_item = None
                    current_output = np.zeros(0, dtype=np.float32)
                    current_position = 0
                if current_item is None and not load_next_item():
                    break
                if current_item is None:
                    break
                remaining = current_output.size - current_position
                if remaining <= 0:
                    self._events.put(("done", current_item))
                    current_item = None
                    current_output = np.zeros(0, dtype=np.float32)
                    current_position = 0
                    continue
                take = min(frames - offset, remaining)
                if current_item.chunk.on_render is not None:
                    render_time = (
                        output_time + offset / self.output_sample_rate
                        if output_time is not None
                        else time.monotonic()
                    )
                    current_item.chunk.on_render(
                        render_time,
                        current_position,
                        take,
                        self.output_sample_rate,
                    )
                rendered[offset: offset + take] = current_output[current_position: current_position + take]
                current_position += take
                with self._position_lock:
                    self._played_position = current_position
                offset += take
                if current_position >= current_output.size:
                    self._events.put(("done", current_item))
                    current_item = None
                    current_output = np.zeros(0, dtype=np.float32)
                    current_position = 0
            outdata.fill(0)
            if rendered.size:
                outdata[:, :] = rendered.reshape(-1, 1)
                if self.playback_reference is not None and reference_resampler is not None:
                    self.playback_reference.push_samples(reference_resampler.process(rendered), output_time)
            if self._closed and current_item is None:
                raise sd.CallbackStop

        try:
            with sd.OutputStream(
                samplerate=self.output_sample_rate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype="float32",
                device=self.output_device,
                latency=self.latency,
                callback=callback,
                finished_callback=self.done_event.set,
            ):
                if self.log is not None:
                    self.log(
                        f"output stream open (rate={self.output_sample_rate}, "
                        f"ch={self.channels})"
                    )
                while not self.done_event.is_set() and not self.stop_event.is_set():
                    self._print_statuses()
                    self._drain_events()
                    time.sleep(0.01)
                self._drain_events()
        except BaseException as exc:
            self.error = exc
        finally:
            if self.playback_reference is not None:
                self.playback_reference.flush()
            if current_item is not None:
                current_item.done.set()
            self._finish_pending_items()
            self.done_event.set()
            self._print_statuses()
            self._drain_events()
            if self.log is not None:
                self.log(f"output stream closed (callbacks={self.callbacks})")

    def _drain_events(self) -> None:
        while True:
            try:
                kind, item = self._events.get_nowait()
            except queue.Empty:
                return
            if kind == "start":
                if self.on_chunk_start is not None:
                    self.on_chunk_start(item.chunk)
            elif kind == "done":
                if self.on_chunk_done is not None:
                    self.on_chunk_done(item.chunk)
                item.done.set()
            elif kind == "drop":
                item.done.set()  # cancelled owner: unblock waiters, no callbacks

    def _finish_pending_items(self) -> None:
        while True:
            try:
                item = self._items.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                item.done.set()

    def _print_statuses(self) -> None:
        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                return
            self.notify(f"voice playback status: {status}")


# ── Mic capture + processing ─────────────────────────────────────────────────

@dataclass
class TurnDetectorConfig:
    threshold: float
    min_silence_ms: int
    speech_pad_ms: int
    min_turn_seconds: float
    max_turn_seconds: float
    preroll_ms: int
    postroll_ms: int
    wake_preprocess: bool = False


class MicLoop:
    def __init__(
        self,
        input_device,
        detector_config: TurnDetectorConfig,
        barge_ignore_ms: int,
        barge_rms_multiplier: float,
        barge_min_rms: float,
        barge_frames: int,
        enable_aec: bool,
        aec_delay_ms: int,
        aec_noise_suppression: bool,
        aec_agc: bool,
        aec_reference_delay_ms: int = 0,
        silero_model: Any | None = None,
        load_silero_model: bool = True,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.input_device = input_device
        self.detector_config = detector_config
        self.barge_ignore_ms = barge_ignore_ms
        self.barge_rms_multiplier = barge_rms_multiplier
        self.barge_min_rms = barge_min_rms
        self.barge_frames = barge_frames
        self.enable_aec = enable_aec
        self.notify = notify or (lambda message: print(message, file=sys.stderr))
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self.raw_audio_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=1000)
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.silero_model = silero_model if silero_model is not None else (
            load_silero_vad() if load_silero_model else None
        )
        self.playback_reference = PlaybackReference()
        self._playback_lock = threading.Lock()
        self._playback_active = False
        self._playback_started_at = 0.0
        self._gate_lock = threading.Lock()
        self._microphone_blocked = False
        self._wake_only = False
        self._wake_playback_suppressed = False
        self._manual_capture_active = False
        self._manual_capture_frames: list[np.ndarray] = []
        self._vad_reset_requested = False
        self._vad_suppress_until = 0.0
        # A missing/unbuildable AEC wheel must only cost echo cancellation, not
        # the whole voice pipeline: the processing loop already tolerates aec=None.
        self.aec = None
        if enable_aec:
            try:
                self.aec = WebRtcAec(
                    playback_reference=self.playback_reference,
                    delay_ms=aec_delay_ms,
                    enable_ns=aec_noise_suppression,
                    enable_agc=aec_agc,
                    reference_delay_ms=aec_reference_delay_ms,
                )
            except Exception as exc:
                self.aec = None
                self.notify(f"voice: echo cancellation unavailable; continuing without AEC ({exc})")
        self._processing_stop = threading.Event()
        self._processing_thread: threading.Thread | None = None
        self._last_input_callback = time.monotonic()

    def input_callback(self, indata, _frames, time_info, status) -> None:
        # Liveness marker: a live mic (even silent, even while intentionally
        # blocked) keeps firing this callback; cessation means the stream died.
        self._last_input_callback = time.monotonic()
        if status:
            try:
                self.status_queue.put_nowait(str(status))
            except queue.Full:
                pass
        block = indata[:, 0].copy()
        frame_time = callback_time_seconds(time_info, "inputBufferAdcTime")
        if frame_time is None:
            frame_time = time.monotonic()
        with self._gate_lock:
            blocked = self._microphone_blocked
        if blocked:
            return
        try:
            self.raw_audio_queue.put_nowait((block, frame_time))
        except queue.Full:
            try:
                self.raw_audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.raw_audio_queue.put_nowait((block, frame_time))

    def stream(self) -> sd.InputStream:
        return sd.InputStream(
            samplerate=SAMPLE_RATE_IN,
            blocksize=AEC_FRAME_SIZE,
            channels=1,
            dtype="float32",
            device=self.input_device,
            callback=self.input_callback,
        )

    def input_callback_age(self) -> float:
        """Seconds since the input stream last delivered a callback. A live mic
        fires callbacks continuously regardless of audio level, so a large age
        means the stream itself died (device unplugged), not a quiet room."""
        return time.monotonic() - self._last_input_callback

    def set_playback_active(self, active: bool) -> None:
        with self._playback_lock:
            self._playback_active = active
            if active:
                self._playback_started_at = time.monotonic()
        if not active:
            self.set_wake_playback_suppressed(False)

    def is_playback_active(self) -> bool:
        with self._playback_lock:
            return self._playback_active

    def playback_elapsed(self) -> float:
        with self._playback_lock:
            if not self._playback_active:
                return 0.0
            return time.monotonic() - self._playback_started_at

    def set_microphone_blocked(self, blocked: bool) -> None:
        changed = False
        now = time.monotonic()
        with self._gate_lock:
            if self._microphone_blocked != blocked:
                changed = True
            self._microphone_blocked = blocked
            if changed:
                self._vad_reset_requested = True
                if not blocked:
                    self._vad_suppress_until = now + VAD_GATE_SUPPRESS_SECONDS
        if changed:
            self.drain_audio(clear_playback_reference=False)

    def set_wake_only(self, wake_only: bool) -> None:
        changed = False
        now = time.monotonic()
        with self._gate_lock:
            if self._wake_only != wake_only:
                changed = True
            self._wake_only = wake_only
            if changed:
                self._vad_reset_requested = True
                if not wake_only:
                    self._vad_suppress_until = now + VAD_GATE_SUPPRESS_SECONDS
        if changed:
            self.drain_audio(clear_playback_reference=False)

    def wake_only(self) -> bool:
        with self._gate_lock:
            return self._wake_only

    def set_wake_playback_suppressed(self, suppressed: bool) -> None:
        with self._gate_lock:
            if self._wake_playback_suppressed == suppressed:
                return
            self._wake_playback_suppressed = suppressed
            self._vad_reset_requested = True

    def wake_playback_suppressed(self) -> bool:
        with self._gate_lock:
            return self._wake_playback_suppressed

    def begin_manual_capture(self) -> None:
        self.drain_audio(clear_playback_reference=False)
        with self._gate_lock:
            self._manual_capture_frames = []
            self._manual_capture_active = True
            self._vad_reset_requested = True

    def end_manual_capture(self) -> np.ndarray:
        now = time.monotonic()
        with self._gate_lock:
            frames = list(self._manual_capture_frames)
            self._manual_capture_frames = []
            self._manual_capture_active = False
            self._vad_reset_requested = True
            if not self._microphone_blocked:
                self._vad_suppress_until = now + VAD_GATE_SUPPRESS_SECONDS
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32)

    def start_processing(self) -> None:
        if self._processing_thread is not None:
            return
        self._processing_stop.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop, name="if-engine-mic-aec", daemon=True
        )
        self._processing_thread.start()

    def stop_processing(self) -> None:
        self._processing_stop.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None

    def _processing_loop(self) -> None:
        pending = np.zeros(0, dtype=np.float32)
        while not self._processing_stop.is_set():
            try:
                raw, frame_time = self.raw_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if raw.size != AEC_FRAME_SIZE:
                raw = raw[:AEC_FRAME_SIZE] if raw.size > AEC_FRAME_SIZE else np.pad(raw, (0, AEC_FRAME_SIZE - raw.size))
            try:
                clean = self.aec.process(raw, frame_time) if self.aec is not None else raw.astype(np.float32, copy=False)
            except Exception as exc:
                self.aec = None
                self.notify(f"voice: AEC disabled after processing error: {exc}")
                clean = raw.astype(np.float32, copy=False)
            pending = np.concatenate((pending, clean))
            while pending.size >= VAD_FRAME_SIZE:
                frame = pending[:VAD_FRAME_SIZE].copy()
                pending = pending[VAD_FRAME_SIZE:]
                self._put_audio_frame(frame)

    def _put_audio_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self._gate_lock:
            manual_capture = self._manual_capture_active
            if self._manual_capture_active:
                self._manual_capture_frames.append(frame.copy())
            blocked = self._microphone_blocked
            suppressed = blocked or manual_capture or now < self._vad_suppress_until
        if suppressed:
            return
        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(frame)

    def drain_audio(self, clear_playback_reference: bool = True) -> None:
        while True:
            drained = False
            try:
                self.audio_queue.get_nowait()
                drained = True
            except queue.Empty:
                pass
            try:
                self.raw_audio_queue.get_nowait()
                drained = True
            except queue.Empty:
                pass
            if not drained:
                if clear_playback_reference:
                    self.playback_reference.clear()
                return

    def consume_vad_reset(self) -> bool:
        with self._gate_lock:
            reset = self._vad_reset_requested
            self._vad_reset_requested = False
        return reset

    def vad_suppressed(self) -> bool:
        now = time.monotonic()
        with self._gate_lock:
            return (
                self._microphone_blocked
                or self._manual_capture_active
                or now < self._vad_suppress_until
            )

    def calibrate_noise(self, seconds: float) -> float:
        if seconds <= 0:
            return 0.0
        self.drain_audio()
        blocks: list[np.ndarray] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                blocks.append(self.audio_queue.get(timeout=0.1))
            except queue.Empty:
                pass
        if not blocks:
            return 0.0
        return float(np.median([rms(block) for block in blocks]))

    def new_vad(self) -> VADIterator:
        if self.silero_model is None:
            raise RuntimeError("Silero VAD model is not loaded")
        cfg = self.detector_config
        return VADIterator(
            self.silero_model,
            threshold=cfg.threshold,
            sampling_rate=SAMPLE_RATE_IN,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )

    def set_silero_model(self, model: Any) -> None:
        self.silero_model = model

    def drain_statuses(self) -> None:
        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                return
            self.notify(f"voice mic status: {status}")


# ── Continuous turn detection ────────────────────────────────────────────────

@dataclass
class SpeechEvent:
    kind: str  # "wake" | "wake_debug" | "start" | "end" | "abort"
    seq: int
    samples: np.ndarray | None = None
    wake_triggered: bool = False
    message: str = ""


@dataclass
class WakeWordDetection:
    label: str
    score: float


class RealtimeWakePreprocessor:
    """Causal wake-word preprocessing for car/Bluetooth AEC captures."""

    def __init__(
        self,
        highpass_hz: float = 120.0,
        preemphasis: float = 0.97,
        target_rms: float = 0.08,
        max_gain: float = 80.0,
    ) -> None:
        self.highpass_hz = highpass_hz
        self.preemphasis = preemphasis
        self.target_rms = target_rms
        self.max_gain = max_gain
        self._sos = None
        self._sos_zi = None
        self._hp_x_prev = 0.0
        self._hp_y_prev = 0.0
        self._pre_prev = 0.0
        self._gain = 1.0
        try:
            from scipy import signal
            self._signal = signal
            self._sos = signal.butter(
                4,
                highpass_hz / (SAMPLE_RATE_IN / 2),
                btype="highpass",
                output="sos",
            )
        except Exception:
            self._signal = None

    def reset(self) -> None:
        if self._sos is not None:
            self._sos_zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._hp_x_prev = 0.0
        self._hp_y_prev = 0.0
        self._pre_prev = 0.0
        self._gain = 1.0

    def _highpass(self, samples: np.ndarray) -> np.ndarray:
        if self._sos is not None and self._signal is not None:
            if self._sos_zi is None:
                self.reset()
            output, self._sos_zi = self._signal.sosfilt(
                self._sos,
                samples,
                zi=self._sos_zi,
            )
            return np.asarray(output, dtype=np.float32)
        # One-pole fallback, still causal and stateful.
        dt = 1.0 / SAMPLE_RATE_IN
        rc = 1.0 / (2.0 * math.pi * self.highpass_hz)
        alpha = rc / (rc + dt)
        output = np.empty_like(samples, dtype=np.float32)
        x_prev = self._hp_x_prev
        y_prev = self._hp_y_prev
        for idx, value in enumerate(samples):
            y = alpha * (y_prev + float(value) - x_prev)
            output[idx] = y
            x_prev = float(value)
            y_prev = y
        self._hp_x_prev = x_prev
        self._hp_y_prev = y_prev
        return output

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return samples.copy()
        highpassed = self._highpass(samples)

        emphasized = np.empty_like(highpassed, dtype=np.float32)
        emphasized[0] = highpassed[0] - self.preemphasis * self._pre_prev
        if highpassed.size > 1:
            emphasized[1:] = highpassed[1:] - self.preemphasis * highpassed[:-1]
        self._pre_prev = float(highpassed[-1])

        desired_gain = min(self.max_gain, self.target_rms / max(rms(emphasized), 1e-5))
        self._gain = 0.55 * self._gain + 0.45 * desired_gain
        return np.clip(emphasized * self._gain, -1.0, 1.0).astype(np.float32)


class OpenWakeWordGate:
    def __init__(
        self,
        model: Any,
        labels: list[str],
        model_paths: list[str],
        threshold: float,
        patience: int,
        debounce_seconds: float,
        activation_window_seconds: float,
        log_min_score: float,
    ) -> None:
        self.model = model
        self.labels = labels
        self.model_paths = model_paths
        self.threshold = threshold
        self.patience = max(1, patience)
        self.debounce_seconds = max(0.0, debounce_seconds)
        self.activation_window_seconds = max(0.1, activation_window_seconds)
        self.log_min_score = max(0.0, min(1.0, log_min_score))
        self._pending = np.zeros(0, dtype=np.float32)
        self._hot_counts: dict[str, int] = {}
        self._last_activation = 0.0
        self._debug_messages: deque[str] = deque()
        self._last_debug_at: dict[tuple[str, str], float] = {}
        self._score_window_started = time.monotonic()
        self._score_window_best: WakeWordDetection | None = None
        self._score_window_frames = 0

    @classmethod
    def from_config(cls, config: "VoiceConfig") -> "OpenWakeWordGate":
        try:
            import openwakeword
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord requires openwakeword and onnxruntime; run "
                ".venv/bin/pip install openwakeword onnxruntime"
            ) from exc

        cache_dir = Path(config.openwakeword_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_models: list[str] = []
        download_names: list[str] = []

        for spec in config.openwakeword_models:
            spec = str(spec).strip()
            if not spec:
                continue
            path = Path(spec).expanduser()
            if path.exists():
                resolved_models.append(str(path.resolve()))
                continue
            local_name = "_".join(re.findall(r"[A-Za-z0-9]+", spec.lower()))
            local_path = DEFAULT_WAKE_WORD_DIR / f"{local_name}.onnx"
            if local_path.exists():
                resolved_models.append(str(local_path.resolve()))
                continue
            info = openwakeword.MODELS.get(local_name)
            if info is not None:
                model_file = Path(info["download_url"].split("/")[-1]).with_suffix(
                    f".{config.openwakeword_inference_framework}"
                )
                resolved_models.append(str(cache_dir / model_file.name))
                download_names.append(local_name)
                continue
            raise RuntimeError(f"openWakeWord model not found: {spec}")

        if not resolved_models:
            raise RuntimeError("no openWakeWord models configured")

        download_models(
            model_names=download_names or ["__features_only__"],
            target_directory=str(cache_dir),
        )
        framework = config.openwakeword_inference_framework
        melspec_path = cache_dir / f"melspectrogram.{framework}"
        embedding_path = cache_dir / f"embedding_model.{framework}"
        missing = [
            str(path)
            for path in [melspec_path, embedding_path, *(Path(item) for item in resolved_models)]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError("openWakeWord assets missing: " + ", ".join(missing))

        model = Model(
            wakeword_models=resolved_models,
            inference_framework=framework,
            melspec_model_path=str(melspec_path),
            embedding_model_path=str(embedding_path),
        )
        return cls(
            model=model,
            labels=sorted(model.models.keys()),
            model_paths=resolved_models,
            threshold=config.openwakeword_threshold,
            patience=config.openwakeword_patience,
            debounce_seconds=config.openwakeword_debounce_seconds,
            activation_window_seconds=config.openwakeword_activation_window_seconds,
            log_min_score=config.openwakeword_log_min_score,
        )

    def describe(self) -> str:
        return ", ".join(self.labels) if self.labels else "unknown"

    def describe_sources(self) -> str:
        parts = []
        for model_path in self.model_paths:
            path = Path(model_path)
            try:
                stat = path.stat()
                parts.append(f"{path} ({stat.st_size} bytes)")
            except OSError:
                parts.append(str(path))
        return ", ".join(parts) if parts else "unknown"

    def reset_utterance(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._hot_counts.clear()
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def _debug(self, reason: str, label: str, score: float, detail: str = "") -> None:
        now = time.monotonic()
        key = (reason, label)
        if now - self._last_debug_at.get(key, 0.0) < 1.0:
            return
        self._last_debug_at[key] = now
        suffix = f" {detail}" if detail else ""
        self._debug_messages.append(
            f"wake rejected {reason} label={label} score={score:.3f} "
            f"threshold={self.threshold:.3f}{suffix}"
        )

    def _record_score(self, label: str, score: float) -> None:
        now = time.monotonic()
        self._score_window_frames += 1
        if (
            self._score_window_best is None
            or score > self._score_window_best.score
        ):
            self._score_window_best = WakeWordDetection(label, score)
        if now - self._score_window_started < 1.0:
            return
        best = self._score_window_best
        if best is not None:
            self._debug_messages.append(
                f"wake listening max label={best.label} score={best.score:.3f} "
                f"threshold={self.threshold:.3f} frames={self._score_window_frames}"
            )
        self._score_window_started = now
        self._score_window_best = None
        self._score_window_frames = 0

    def pop_debug_messages(self) -> list[str]:
        messages = list(self._debug_messages)
        self._debug_messages.clear()
        return messages

    def process_block(self, block: np.ndarray) -> WakeWordDetection | None:
        if block.size == 0:
            return None
        self._pending = np.concatenate((self._pending, block.astype(np.float32, copy=False)))
        best: WakeWordDetection | None = None
        while self._pending.size >= OPENWAKEWORD_FRAME_SIZE:
            frame = float_to_pcm16_array(self._pending[:OPENWAKEWORD_FRAME_SIZE])
            self._pending = self._pending[OPENWAKEWORD_FRAME_SIZE:]
            predictions = self.model.predict(frame)
            for label, raw_score in predictions.items():
                score = float(raw_score)
                self._record_score(label, score)
                if score >= self.threshold:
                    self._hot_counts[label] = self._hot_counts.get(label, 0) + 1
                    if self._hot_counts[label] < self.patience:
                        self._debug(
                            "patience",
                            label,
                            score,
                            f"hot={self._hot_counts[label]}/{self.patience}",
                        )
                else:
                    self._hot_counts[label] = 0
                    if score >= self.log_min_score:
                        self._debug("below_threshold", label, score)
                if self._hot_counts[label] >= self.patience and (
                    best is None or score > best.score
                ):
                    best = WakeWordDetection(label, score)
        if best is None:
            return None
        now = time.monotonic()
        if now - self._last_activation < self.debounce_seconds:
            self._debug("debounce", best.label, best.score)
            return None
        self._last_activation = now
        self.reset_utterance()
        return best

class ContinuousTurnDetector:
    """Always-on Silero VAD over AEC-cleaned mic audio. Speech while the game
    is silent is always accepted; speech during TTS playback must also clear an
    RMS barge-in threshold so residual echo cannot interrupt the narrator."""

    def __init__(
        self,
        mic: MicLoop,
        noise_floor: float,
        wake_gate: OpenWakeWordGate | None = None,
    ) -> None:
        self.mic = mic
        self.noise_floor = noise_floor
        self.wake_gate = wake_gate
        self.playback_residual_rms = max(noise_floor, 1e-4)
        self.events: queue.Queue[SpeechEvent] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="if-engine-turn-detector", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _hot_during_playback(self, block: np.ndarray) -> bool:
        level = rms(block)
        threshold = max(
            self.mic.barge_min_rms,
            self.noise_floor * 5.0,
            self.playback_residual_rms * self.mic.barge_rms_multiplier,
        )
        if level < threshold:
            self.playback_residual_rms = 0.98 * self.playback_residual_rms + 0.02 * max(level, 1e-4)
            return False
        return True

    def _can_accept_start(self, hot_frames: int, candidate_from_playback: bool) -> bool:
        if not self.mic.is_playback_active():
            return not candidate_from_playback
        if self.mic.playback_elapsed() < self.mic.barge_ignore_ms / 1000:
            return False
        return hot_frames >= self.mic.barge_frames

    def _run(self) -> None:
        cfg = self.mic.detector_config
        max_samples = int(cfg.max_turn_seconds * SAMPLE_RATE_IN)
        min_samples = int(cfg.min_turn_seconds * SAMPLE_RATE_IN)
        preroll_blocks = max(1, int((cfg.preroll_ms / 1000) * SAMPLE_RATE_IN / VAD_FRAME_SIZE))
        postroll_blocks = max(0, int(round((cfg.postroll_ms / 1000) * SAMPLE_RATE_IN / VAD_FRAME_SIZE)))
        wake_preroll_blocks = max(
            preroll_blocks,
            int(round(1.5 * SAMPLE_RATE_IN / VAD_FRAME_SIZE)),
        )
        preroll: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        wake_preroll: deque[np.ndarray] = deque(maxlen=wake_preroll_blocks)
        captured: list[np.ndarray] = []
        vad = self.mic.new_vad()
        wake_preprocessor = RealtimeWakePreprocessor() if cfg.wake_preprocess else None
        wake_transform = "hp120_pre0.97_agc0.08_realtime" if cfg.wake_preprocess else "raw"
        triggered = False
        accepted = False
        candidate_wake_mode = False
        candidate_from_playback = False
        hot_frames = 0
        seq = 0
        active_seq = 0
        total_samples = 0
        wake_audio_last_log = time.monotonic()
        wake_audio_frames = 0
        wake_audio_max_rms = 0.0
        wake_suppressed_last_log = 0.0

        def reset_capture() -> None:
            nonlocal triggered, accepted, candidate_from_playback, hot_frames, total_samples
            nonlocal candidate_wake_mode
            captured.clear()
            preroll.clear()
            total_samples = 0
            triggered = False
            accepted = False
            candidate_wake_mode = False
            candidate_from_playback = False
            hot_frames = 0
            wake_preroll.clear()
            vad.reset_states()
            if wake_preprocessor is not None:
                wake_preprocessor.reset()
            if self.wake_gate is not None:
                self.wake_gate.reset_utterance()

        def cancel_capture() -> None:
            if accepted:
                self.events.put(SpeechEvent("abort", active_seq))
            reset_capture()

        def append_postroll() -> None:
            nonlocal total_samples
            for _ in range(postroll_blocks):
                if total_samples >= max_samples or self.stop_event.is_set():
                    return
                try:
                    block = self.mic.audio_queue.get(timeout=0.03)
                except queue.Empty:
                    return
                captured.append(block)
                total_samples += len(block)

        def process_wake_audio(block: np.ndarray) -> bool:
            nonlocal seq, active_seq, accepted
            if not candidate_wake_mode or accepted or self.wake_gate is None:
                return False
            score_started = time.monotonic()
            processed = wake_preprocessor.process(block) if wake_preprocessor is not None else block
            detection = self.wake_gate.process_block(processed)
            score_elapsed_ms = (time.monotonic() - score_started) * 1000.0
            for message in self.wake_gate.pop_debug_messages():
                self.events.put(
                    SpeechEvent(
                        "wake_debug",
                        seq,
                        message=f"{message} transform={wake_transform}",
                    )
                )
            if detection is None:
                return False
            seq += 1
            active_seq = seq
            accepted = True
            self.events.put(SpeechEvent("start", active_seq))
            self.events.put(
                SpeechEvent(
                    "wake",
                    active_seq,
                    message=(
                        f"label={detection.label} score={detection.score:.3f} "
                        f"transform={wake_transform} "
                        f"duration={total_samples / SAMPLE_RATE_IN:.2f}s "
                        f"score_ms={score_elapsed_ms:.0f}"
                    ),
                )
            )
            return True

        while not self.stop_event.is_set():
            self.mic.drain_statuses()
            if self.mic.consume_vad_reset() or self.mic.vad_suppressed():
                cancel_capture()
                time.sleep(0.01)
                continue
            try:
                block = self.mic.audio_queue.get(timeout=0.1)
            except queue.Empty:
                if self.mic.wake_only():
                    now = time.monotonic()
                    if now - wake_audio_last_log >= 1.0:
                        self.events.put(
                            SpeechEvent(
                                "wake_debug",
                                seq,
                                message=(
                                    "wake audio "
                                    f"frames={wake_audio_frames} "
                                    f"max_rms={wake_audio_max_rms:.5f}"
                                ),
                            )
                        )
                        wake_audio_last_log = now
                        wake_audio_frames = 0
                        wake_audio_max_rms = 0.0
                continue
            if self.mic.consume_vad_reset() or self.mic.vad_suppressed():
                cancel_capture()
                continue

            wake_only = self.mic.wake_only()
            wake_playback_suppressed = wake_only and self.mic.wake_playback_suppressed()
            if wake_only:
                wake_audio_frames += 1
                wake_audio_max_rms = max(wake_audio_max_rms, rms(block))
                now = time.monotonic()
                if now - wake_audio_last_log >= 1.0:
                    self.events.put(
                        SpeechEvent(
                            "wake_debug",
                            seq,
                            message=(
                                "wake audio "
                                f"frames={wake_audio_frames} "
                                f"max_rms={wake_audio_max_rms:.5f}"
                            ),
                        )
                    )
                    wake_audio_last_log = now
                    wake_audio_frames = 0
                    wake_audio_max_rms = 0.0
            else:
                wake_audio_last_log = time.monotonic()
                wake_audio_frames = 0
                wake_audio_max_rms = 0.0

            if wake_playback_suppressed:
                if triggered:
                    reset_capture()
                now = time.monotonic()
                if now - wake_suppressed_last_log >= 1.0:
                    self.events.put(
                        SpeechEvent(
                            "wake_debug",
                            seq,
                            message="wake suppressed: playback text contains ok/okay",
                        )
                    )
                    wake_suppressed_last_log = now
                continue

            event = vad(torch.from_numpy(block))
            if not triggered:
                preroll.append(block)
                wake_preroll.append(block)
                if event and "start" in event:
                    triggered = True
                    accepted = False
                    candidate_wake_mode = wake_only and self.wake_gate is not None
                    candidate_from_playback = self.mic.is_playback_active()
                    hot_frames = 1 if candidate_from_playback and self._hot_during_playback(block) else 0
                    captured = list(wake_preroll if candidate_wake_mode else preroll)
                    total_samples = sum(len(item) for item in captured)
                    preroll.clear()
                    wake_preroll.clear()
                    if candidate_wake_mode:
                        if wake_preprocessor is not None:
                            wake_preprocessor.reset()
                        if self.wake_gate is not None:
                            self.wake_gate.reset_utterance()
                        self.events.put(
                            SpeechEvent(
                                "wake_debug",
                                seq,
                                message=(
                                    "wake candidate start "
                                    f"playback={'on' if candidate_from_playback else 'off'} "
                                    f"rms={rms(block):.5f}"
                                ),
                            )
                        )
                        for wake_block in captured:
                            if process_wake_audio(wake_block):
                                break
                        continue
                    if self._can_accept_start(hot_frames, candidate_from_playback):
                        seq += 1
                        active_seq = seq
                        accepted = True
                        self.events.put(SpeechEvent("start", active_seq))
                continue

            captured.append(block)
            total_samples += len(block)
            if candidate_wake_mode:
                if not accepted:
                    process_wake_audio(block)
            elif not accepted:
                if self.mic.is_playback_active() and self._hot_during_playback(block):
                    hot_frames += 1
                else:
                    hot_frames = max(0, hot_frames - 1)
                if self._can_accept_start(hot_frames, candidate_from_playback):
                    seq += 1
                    active_seq = seq
                    accepted = True
                    self.events.put(SpeechEvent("start", active_seq))

            end_of_turn = bool(event and "end" in event) or total_samples >= max_samples
            if not end_of_turn:
                continue

            if candidate_wake_mode:
                if total_samples < min_samples:
                    reset_capture()
                    continue
                if not accepted:
                    self.events.put(
                        SpeechEvent(
                            "wake_debug",
                            seq,
                            message=(
                                "wake candidate rejected no_detection "
                                f"duration={total_samples / SAMPLE_RATE_IN:.2f}s "
                                f"transform={wake_transform}"
                            ),
                        )
                    )
                    reset_capture()
                    continue
                append_postroll()
                samples = np.concatenate(captured).astype(np.float32)
                self.events.put(
                    SpeechEvent(
                        "end",
                        active_seq,
                        samples,
                        wake_triggered=True,
                    )
                )
            elif accepted and total_samples >= min_samples:
                append_postroll()
                samples = np.concatenate(captured).astype(np.float32)
                self.events.put(
                    SpeechEvent(
                        "end",
                        active_seq,
                        samples,
                        wake_triggered=False,
                    )
                )
            elif accepted:
                # Too short to transcribe; let the host unpause/clear state.
                self.events.put(SpeechEvent("abort", active_seq))

            reset_capture()


# ── Configuration ────────────────────────────────────────────────────────────

def _parse_device(value: str | None):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_is_set(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value != ""


def _narrator_clone_reference() -> Path | None:
    """Narrator voice override, set the same way as the audio cues: point an env
    var at a WAV (or replace assets/voice-clone-reference.wav). Prefers the
    friendly IF_ENGINE_NARRATOR_VOICE, falls back to the older
    IF_ENGINE_OMNIVOICE_CLONE_REFERENCE alias, then the bundled default. An
    explicit empty value forces OmniVoice design-prompt mode instead of cloning."""
    for name in ("IF_ENGINE_NARRATOR_VOICE", "IF_ENGINE_OMNIVOICE_CLONE_REFERENCE"):
        if name in os.environ:
            value = os.environ[name]
            break
    else:
        value = str(DEFAULT_OMNIVOICE_CLONE_REFERENCE)
    value = value.strip()
    return None if value == "" else Path(value)


DEFAULT_KOKORO_VOICE = "bm_george"
_BOOLEANISH_VALUES = {"0", "1", "true", "false", "on", "off", "yes", "no"}


def _kokoro_voice() -> str:
    """Kokoro voice name, set like the other audio overrides. Prefers
    IF_ENGINE_KOKORO_VOICE, falls back to the older IF_ENGINE_VOICE alias, and
    guards the footgun of setting it to a boolean-looking value while expecting an
    on/off toggle (use ./play --no-voice to disable voice)."""
    for name in ("IF_ENGINE_KOKORO_VOICE", "IF_ENGINE_VOICE"):
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        if value.lower() in _BOOLEANISH_VALUES:
            print(
                f"warning: {name}={value!r} sets the Kokoro voice name, not voice "
                f"on/off — use ./play --no-voice to disable voice. "
                f"Using {DEFAULT_KOKORO_VOICE!r} instead.",
                file=sys.stderr,
            )
            return DEFAULT_KOKORO_VOICE
        return value
    return DEFAULT_KOKORO_VOICE


def _env_path_list(name: str, default: list[Path]) -> list[Path]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [Path(item).expanduser() for item in value.split(os.pathsep) if item.strip()]


def default_openwakeword_models() -> list[Path]:
    existing = [path for path in DEFAULT_OPENWAKEWORD_MODEL_CANDIDATES if path.exists()]
    return existing or [DEFAULT_OPENWAKEWORD_MODEL]


def default_wake_word_threshold() -> float:
    if _env_is_set("IF_ENGINE_WAKE_WORD_THRESHOLD"):
        return _env_float("IF_ENGINE_WAKE_WORD_THRESHOLD", 0.9)
    if _env_bool("IF_ENGINE_CAR_MODE", False):
        return _env_float("IF_ENGINE_CAR_WAKE_WORD_THRESHOLD", 0.4)
    return 0.9


def default_wake_word_preprocess() -> bool:
    if _env_is_set("IF_ENGINE_WAKE_WORD_PREPROCESS"):
        return _env_bool("IF_ENGINE_WAKE_WORD_PREPROCESS", False)
    return _env_bool("IF_ENGINE_CAR_MODE", False)


@dataclass
class VoiceConfig:
    input_device: object = field(default_factory=lambda: _parse_device(os.environ.get("IF_ENGINE_VOICE_INPUT_DEVICE")))
    output_device: object = field(default_factory=lambda: _parse_device(os.environ.get("IF_ENGINE_VOICE_OUTPUT_DEVICE")))
    output_sample_rate: int | None = None
    output_blocksize: int = 2048
    output_latency: object = "high"

    whisper_cli: Path = field(default_factory=lambda: Path(
        os.environ.get("IF_ENGINE_WHISPER_CLI", DEFAULT_WHISPER_DIR / "build" / "bin" / "whisper-cli")))
    whisper_model: Path = field(default_factory=lambda: Path(
        os.environ.get("IF_ENGINE_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)))
    whisper_vad_model: Path = field(default_factory=lambda: Path(
        os.environ.get("IF_ENGINE_WHISPER_VAD_MODEL", DEFAULT_WHISPER_DIR / "models" / "ggml-silero-v6.2.0.bin")))
    whisper_prompt: str = ""
    whisper_language: str = "en"
    whisper_threads: int = 4

    tts_engine: str = os.environ.get("IF_ENGINE_TTS_ENGINE", "omnivoice").strip().lower()

    kokoro_lang: str = "a"
    kokoro_voice: str = field(default_factory=_kokoro_voice)
    kokoro_speed: float = 1.22
    kokoro_device: str = "cpu"
    omnivoice_model: str = os.environ.get("IF_ENGINE_OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    omnivoice_device: str = os.environ.get("IF_ENGINE_OMNIVOICE_DEVICE", "auto")
    omnivoice_dtype: str = os.environ.get("IF_ENGINE_OMNIVOICE_DTYPE", "float16")
    omnivoice_num_step: int = field(default_factory=lambda: _env_int("IF_ENGINE_OMNIVOICE_NUM_STEP", 32))
    omnivoice_speed: float = field(default_factory=lambda: _env_float("IF_ENGINE_OMNIVOICE_SPEED", 0.9))
    omnivoice_instruct: str = os.environ.get(
        "IF_ENGINE_OMNIVOICE_INSTRUCT",
        "male, low pitch, elderly, british accent",
    )
    omnivoice_whisper_tags: bool = field(
        default_factory=lambda: _env_bool("IF_ENGINE_OMNIVOICE_WHISPER_TAGS", False)
    )
    omnivoice_clone_reference: Path | None = field(default_factory=_narrator_clone_reference)
    omnivoice_clone_transcript: str = field(default_factory=lambda: os.environ.get(
        "IF_ENGINE_OMNIVOICE_CLONE_TRANSCRIPT",
        "",
    ))
    omnivoice_clone_transcript_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "IF_ENGINE_OMNIVOICE_CLONE_TRANSCRIPT_PATH",
        str(DEFAULT_OMNIVOICE_CLONE_TRANSCRIPT),
    )))
    omnivoice_character_voices_enabled: bool = field(
        default_factory=lambda: _env_bool("IF_ENGINE_OMNIVOICE_CHARACTER_VOICES", True)
    )
    elevenlabs_voice_cache_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "IF_ENGINE_ELEVENLABS_VOICE_CACHE_DIR",
        DEFAULT_ELEVENLABS_VOICE_CACHE_DIR,
    )))
    external_cost_recorder: Callable[[dict[str, Any]], None] | None = None
    omnivoice_character_voice_transcript_stem: str = ""
    omnivoice_character_voice_transcript_filename: str = ""
    omnivoice_character_voice_transcript_text: str = ""
    omnivoice_character_voice_game_title: str = ""
    omnivoice_whisper_clone_reference: Path | None = field(default_factory=lambda: (
        None
        if os.environ.get(
            "IF_ENGINE_OMNIVOICE_WHISPER_CLONE_REFERENCE",
            str(DEFAULT_OMNIVOICE_WHISPER_CLONE_REFERENCE),
        ).strip() == ""
        else Path(os.environ.get(
            "IF_ENGINE_OMNIVOICE_WHISPER_CLONE_REFERENCE",
            str(DEFAULT_OMNIVOICE_WHISPER_CLONE_REFERENCE),
        ))
    ))
    omnivoice_whisper_clone_transcript: str = os.environ.get(
        "IF_ENGINE_OMNIVOICE_WHISPER_CLONE_TRANSCRIPT",
        "",
    )
    omnivoice_whisper_clone_transcript_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "IF_ENGINE_OMNIVOICE_WHISPER_CLONE_TRANSCRIPT_PATH",
        str(DEFAULT_OMNIVOICE_WHISPER_CLONE_TRANSCRIPT),
    )))
    volume: float = 0.85
    sentence_pause_ms: int = 150
    paragraph_pause_ms: int = 500
    dash_pause_ms: int = 300
    ellipsis_pause_ms: int = 450
    playback_alignment: bool = True
    playback_alignment_margin_ms: int = 120
    turn_cue_path: Path = field(default_factory=lambda: Path(
        os.environ.get("IF_ENGINE_TURN_CUE", Path(__file__).resolve().parent / "assets" / "turn-cue.wav")))
    turn_cue_volume: float = 0.5
    turn_cue_delay_ms: int = 500
    confirm_cue_path: Path = field(default_factory=lambda: Path(
        os.environ.get("IF_ENGINE_CONFIRM_CUE", Path(__file__).resolve().parent / "assets" / "confirm-cue.wav")))
    confirm_cue_volume: float = 0.6

    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 550
    vad_speech_pad_ms: int = 80
    min_turn_seconds: float = 0.35
    max_turn_seconds: float = 18.0
    preroll_ms: int = 600
    postroll_ms: int = 250
    calibrate_seconds: float = 0.8

    barge_ignore_ms: int = field(default_factory=lambda: _env_int("IF_ENGINE_BARGE_IGNORE_MS", 450))
    barge_rms_multiplier: float = field(default_factory=lambda: _env_float("IF_ENGINE_BARGE_RMS_MULTIPLIER", 2.4))
    barge_min_rms: float = field(default_factory=lambda: _env_float("IF_ENGINE_BARGE_MIN_RMS", 0.006))
    barge_frames: int = field(default_factory=lambda: _env_int("IF_ENGINE_BARGE_FRAMES", 1))

    aec: bool = True
    aec_delay_ms: int = field(default_factory=lambda: _env_int("IF_ENGINE_AEC_DELAY_MS", 80))
    aec_reference_delay_ms: int = field(default_factory=lambda: _env_int("IF_ENGINE_AEC_REFERENCE_DELAY_MS", 0))
    aec_noise_suppression: bool = True
    aec_agc: bool = False

    car_mode: bool = field(default_factory=lambda: _env_bool("IF_ENGINE_CAR_MODE", False))
    wake_word_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "IF_ENGINE_WAKE_WORD",
            _env_bool("IF_ENGINE_CAR_MODE", False),
        )
    )
    openwakeword_models: list[Path] = field(default_factory=lambda: _env_path_list(
        "IF_ENGINE_WAKE_WORD_MODEL",
        default_openwakeword_models(),
    ))
    openwakeword_cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("IF_ENGINE_OPENWAKEWORD_CACHE_DIR", DEFAULT_OPENWAKEWORD_CACHE_DIR)
    ))
    openwakeword_threshold: float = field(default_factory=default_wake_word_threshold)
    wake_word_preprocess: bool = field(default_factory=default_wake_word_preprocess)
    openwakeword_log_min_score: float = field(default_factory=lambda: _env_float("IF_ENGINE_WAKE_WORD_LOG_MIN_SCORE", 0.35))
    openwakeword_patience: int = field(default_factory=lambda: _env_int("IF_ENGINE_WAKE_WORD_PATIENCE", 1))
    openwakeword_debounce_seconds: float = field(default_factory=lambda: _env_float("IF_ENGINE_WAKE_WORD_DEBOUNCE_SECONDS", 1.0))
    openwakeword_activation_window_seconds: float = field(default_factory=lambda: _env_float("IF_ENGINE_WAKE_WORD_WINDOW_SECONDS", 5.0))
    openwakeword_inference_framework: str = os.environ.get("IF_ENGINE_WAKE_WORD_FRAMEWORK", "onnx")


# ── TTS session ──────────────────────────────────────────────────────────────

class _TtsSession:
    """One narration being spoken: text chunks in, queued audio chunks out."""

    def __init__(self, voice: "VoiceLoop") -> None:
        self.voice = voice
        cfg = voice.config
        self.cancel_event = threading.Event()
        chunker_args = (
            max(0, cfg.sentence_pause_ms) / 1000,
            max(0, cfg.paragraph_pause_ms) / 1000,
            max(0, cfg.dash_pause_ms) / 1000,
            max(0, cfg.ellipsis_pause_ms) / 1000,
        )
        if isinstance(voice.speaker, OmniVoiceSpeaker):
            self.chunker = StreamingOmniVoiceChunker(
                *chunker_args,
                whisper_tags_enabled=cfg.omnivoice_whisper_tags,
                voice_tag_callback=voice.speaker.ensure_character_voice_started,
            )
        else:
            self.chunker = StreamingSentenceChunker(*chunker_args)
        self.text_queue: queue.Queue[TextChunk | None] = queue.Queue()
        self.audio_queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=4)
        self.display_queue: queue.Queue[str] = queue.Queue()
        self._playback_marked = False
        self.completed_chunks: list[str] = []
        self.current_chunk: AudioChunk | None = None
        self.displayed_parts: list[str] = []  # sentence texts shown so far
        self.fed_display: list[str] = []      # all sentence texts fed to TTS
        self.user_cancelled = False           # host/interruption cancel (vs failure)
        self.show_remaining_on_cancel = False
        self.remaining_notice = "voice: narration audio failed — showing the rest as text"
        self.finished = threading.Event()  # playback fully drained or cancelled
        self.synth_thread = threading.Thread(target=self._synth_loop, name="if-engine-synth", daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_loop, name="if-engine-playback-q", daemon=True)

    def start(self) -> None:
        self.synth_thread.start()
        self.playback_thread.start()

    def feed(self, text: str) -> None:
        if self.cancel_event.is_set():
            return
        for chunk in self.chunker.add(text):
            self._put_text(chunk)

    def finish_text(self) -> None:
        if not self.cancel_event.is_set():
            for chunk in self.chunker.flush():
                self._put_text(chunk)
        self.text_queue.put(None)

    def cancel(self) -> None:
        # The shared output stream is untouched: VoiceLoop deactivates this
        # session as owner, which makes the render callback drop its audio.
        self.cancel_event.set()
        self.text_queue.put(None)

    def is_speaking(self) -> bool:
        return not self.finished.is_set() and not self.cancel_event.is_set()

    def _display_part(self, text: str) -> None:
        if not text or self.cancel_event.is_set():
            return
        voice = self.voice
        with voice.state_lock:
            self.displayed_parts.append(text)
        if voice.on_display is not None:
            voice.on_display(text)

    def _drain_display_queue(self) -> None:
        while not self.cancel_event.is_set():
            try:
                text = self.display_queue.get_nowait()
            except queue.Empty:
                return
            self._display_part(text)

    def handle_chunk_start(self, chunk: AudioChunk) -> None:
        voice = self.voice
        # A queued start event can drain after cancellation; showing its
        # text then would glue stale narration after the player's input.
        if self.cancel_event.is_set():
            return
        with voice.state_lock:
            self.current_chunk = chunk
        if not self._playback_marked:
            voice.mic.set_playback_active(True)
            self._playback_marked = True
        suppress_wake = contains_wake_word_text(chunk.spoken_text)
        voice.mic.set_wake_playback_suppressed(suppress_wake)
        if suppress_wake:
            voice.log("wake word suppressed during playback chunk containing ok/okay")
        if chunk.display_text:
            self._display_part(chunk.display_text)
        voice.log(f"chunk start {chunk.spoken_text[:60]!r}")

    def handle_chunk_done(self, chunk: AudioChunk) -> None:
        with self.voice.state_lock:
            if chunk.spoken_text:
                self.completed_chunks.append(chunk.spoken_text)
            if self.current_chunk is chunk:
                self.current_chunk = None
                self.voice.mic.set_wake_playback_suppressed(False)

    def _put_text(self, chunk: TextChunk) -> None:
        if chunk.text:
            # Color named-voice dialogue (and its wrapping quotes) in each
            # character's voice-design color before the text reaches the
            # display pipeline; speech synthesis keeps the plain text/spans.
            display_source = display_markup_from_spans(
                chunk.text, chunk.style_spans, self.voice.character_display_color
            )
            display_text = strip_audio_cue_tags_for_display(display_source)
            if display_text:
                self.fed_display.extend(split_display_sentences(display_text + chunk.suffix))
            speech_text = chunk.speech_text or voice_text_with_dash_pauses(chunk.text)
            style_spans = chunk.style_spans
            if style_spans:
                style_spans = tuple(
                    StyledTextSpan(
                        voice_text_with_dash_pauses(span.text),
                        span.instruct_suffix,
                        span.voice_name,
                        span.voice_prompt,
                    )
                    for span in style_spans
                    if span.text
                )
            pause_after = chunk.pause_after
            if speech_text != chunk.text and chunk.text.rstrip().endswith("—"):
                pause_after = 0.0
            self.text_queue.put(
                TextChunk(
                    chunk.text,
                    pause_after,
                    chunk.suffix,
                    chunk.instruct_suffix,
                    speech_text,
                    style_spans,
                    display_text,
                )
            )

    def _put_audio(self, chunk: AudioChunk | None) -> None:
        while True:
            if self.cancel_event.is_set() and chunk is None:
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass
                return
            try:
                self.audio_queue.put(chunk, timeout=0.05)
                return
            except queue.Full:
                if self.cancel_event.is_set():
                    return

    def _synthesize_with_ellipsis_silence(self, text: str, cfg: VoiceConfig) -> np.ndarray:
        pieces: list[np.ndarray] = []
        silence = np.zeros(
            int(round(max(0, cfg.ellipsis_pause_ms) / 1000 * SAMPLE_RATE_TTS)),
            dtype=np.float32,
        )
        segments = ELLIPSIS_TOKEN_RE.split(text)
        for index, segment in enumerate(segments):
            spoken = normalize_spoken_text(segment, kokoro_markup=True)
            if spoken:
                pieces.append(self.voice.speaker.synthesize(spoken, cancel_event=self.cancel_event))
            if index < len(segments) - 1 and silence.size:
                pieces.append(silence)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def _make_sentence_display_callback(
        self,
        display_events: list[DisplayEvent],
    ) -> Callable[[float, int, int, int], None]:
        next_index = 0
        lock = threading.Lock()

        def on_render(
            _render_time: float,
            position: int,
            frames: int,
            output_rate: int,
        ) -> None:
            nonlocal next_index
            if self.cancel_event.is_set():
                return
            end_ms = int(round((position + frames) / output_rate * 1000))
            with lock:
                while (
                    next_index < len(display_events)
                    and display_events[next_index].start_ms <= end_ms
                ):
                    self.display_queue.put(display_events[next_index].text)
                    next_index += 1

        return on_render

    def _synth_loop(self) -> None:
        try:
            while not self.cancel_event.is_set():
                try:
                    chunk = self.text_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if chunk is None:
                    return
                cfg = self.voice.config
                is_kokoro = isinstance(self.voice.speaker, KokoroSpeaker)
                is_omnivoice = isinstance(self.voice.speaker, OmniVoiceSpeaker)
                tts_text = chunk.speech_text or chunk.text
                ellipsis_count = len(ELLIPSIS_TOKEN_RE.findall(tts_text)) if is_kokoro else 0
                use_cut_marker = (
                    is_kokoro
                    and ellipsis_count > 0
                    and self.voice.transcriber is not None
                )
                spoken_text = normalize_spoken_text(tts_text)
                if not spoken_text:
                    continue
                styled_segments: list[tuple[str, str, str | None, str | None]] = []
                if is_omnivoice and chunk.style_spans:
                    for span in chunk.style_spans:
                        segment_text = normalize_spoken_text(span.text)
                        if segment_text:
                            styled_segments.append(
                                (
                                    segment_text,
                                    span.instruct_suffix,
                                    span.voice_name,
                                    span.voice_prompt,
                                )
                            )
                render_text = (
                    " ".join(segment for segment, _style, _voice, _prompt in styled_segments)
                    if styled_segments
                    else normalize_spoken_text(
                        tts_text,
                        cut_ellipses=use_cut_marker,
                        kokoro_markup=is_kokoro,
                    )
                )
                with self.voice.synth_lock:
                    if self.cancel_event.is_set():
                        return
                    if is_omnivoice and styled_segments:
                        pieces = [
                            self.voice.speaker.synthesize(
                                segment_text,
                                cancel_event=self.cancel_event,
                                instruct_suffix=segment_style,
                                voice_name=voice_name,
                                voice_prompt_hint=voice_prompt,
                            )
                            for segment_text, segment_style, voice_name, voice_prompt in styled_segments
                        ]
                        audio = (
                            np.concatenate([piece for piece in pieces if piece.size]).astype(
                                np.float32,
                                copy=False,
                            )
                            if any(piece.size for piece in pieces)
                            else np.zeros(0, dtype=np.float32)
                        )
                    elif is_omnivoice:
                        audio = self.voice.speaker.synthesize(
                            render_text,
                            cancel_event=self.cancel_event,
                            instruct_suffix=chunk.instruct_suffix,
                        )
                    else:
                        audio = self.voice.speaker.synthesize(
                            render_text,
                            cancel_event=self.cancel_event,
                        )
                if self.cancel_event.is_set():
                    return
                if not audio.size:
                    self.voice.log(f"synth produced no audio for {render_text[:60]!r}")
                    continue
                style_suffixes = [
                    style
                    for _text, style, _voice, _prompt in styled_segments
                    if style
                ]
                voice_names = [
                    voice_name
                    for _text, _style, voice_name, _prompt in styled_segments
                    if voice_name
                ]
                style_parts = []
                if style_suffixes:
                    style_parts.append("+".join(sorted(set(style_suffixes))))
                if voice_names:
                    style_parts.append(
                        "voice=" + "+".join(sorted(set(voice_names)))
                    )
                if style_parts:
                    style_note = f" style={' '.join(style_parts)}"
                else:
                    style_note = (
                        f" style={chunk.instruct_suffix}"
                        if chunk.instruct_suffix
                        else ""
                    )
                self.voice.log(
                    f"synth ok{style_note} ({audio.size / SAMPLE_RATE_TTS:.2f}s) "
                    f"{render_text[:60]!r}"
                )
                alignment_words: list[TimedTextWord] = []
                cut_alignment_ok = False
                if use_cut_marker and self.voice.transcriber is not None:
                    try:
                        marker_words = self.voice.transcriber.align_tts_samples(
                            audio, render_text, cancel_event=self.cancel_event
                        )
                        audio, alignment_words, cut_count = replace_cut_markers_with_silence(
                            audio,
                            marker_words,
                            cfg.ellipsis_pause_ms,
                        )
                        cut_alignment_ok = cut_count == ellipsis_count
                        if cut_alignment_ok:
                            self.voice.log(
                                f"ellipsis cut markers replaced count={cut_count} pause_ms={cfg.ellipsis_pause_ms}"
                            )
                        else:
                            self.voice.log(
                                f"ellipsis cut marker mismatch expected={ellipsis_count} found={cut_count}; using silence fallback"
                            )
                    except CancelledError:
                        raise
                    except Exception as exc:
                        self.voice.notify(f"voice ellipsis cut alignment failed: {exc}")
                if ellipsis_count > 0 and not cut_alignment_ok:
                    with self.voice.synth_lock:
                        if self.cancel_event.is_set():
                            return
                        audio = self._synthesize_with_ellipsis_silence(tts_text, cfg)
                    alignment_words = []
                wants_alignment = (
                    (cfg.playback_alignment or is_omnivoice)
                    and self.voice.transcriber is not None
                )
                if wants_alignment and not alignment_words:
                    try:
                        alignment_words = self.voice.transcriber.align_tts_samples(
                            audio, spoken_text, cancel_event=self.cancel_event
                        )
                    except CancelledError:
                        raise
                    except Exception as exc:
                        self.voice.notify(f"voice alignment failed: {exc}")
                display_text = (
                    chunk.display_markup_text
                    or strip_audio_cue_tags_for_display(chunk.text)
                )
                display_full = display_text + chunk.suffix if display_text else ""
                display_parts = split_display_sentences(display_full)
                display_events = (
                    build_aligned_display_events(display_parts, alignment_words)
                    if is_omnivoice
                    else []
                )
                if is_omnivoice and display_full and not display_events:
                    self.voice.log(
                        "omnivoice sentence display alignment unavailable; "
                        "falling back to chunk-start display"
                    )

                self._put_audio(
                    AudioChunk(
                        samples=audio,
                        pause_after=chunk.pause_after,
                        spoken_text=spoken_text,
                        alignment_words=alignment_words,
                        display_text="" if display_events else display_full,
                        display_events=display_events,
                        owner=self,
                        on_render=(
                            self._make_sentence_display_callback(display_events)
                            if display_events
                            else None
                        ),
                    )
                )
        except CancelledError:
            self.cancel_event.set()
        except BaseException as exc:
            self.voice.notify(f"voice synthesis error: {exc}")
            self.cancel_event.set()
        finally:
            self._put_audio(None)

    def _playback_loop(self) -> None:
        voice = self.voice
        cfg = voice.config
        last_end: float | None = None
        last_pause = 0.0

        def wait_for_pause(duration: float, since: float) -> None:
            remaining = duration - (time.monotonic() - since)
            while remaining > 0 and not self.cancel_event.is_set():
                time.sleep(min(0.02, remaining))
                remaining = duration - (time.monotonic() - since)

        try:
            got_first_chunk = False
            text_seen_at: float | None = None
            while not self.cancel_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.05)
                except queue.Empty:
                    if not got_first_chunk and self.fed_display:
                        # Watchdog: sentences were fed but synthesis never
                        # produced audio — a wedged synthesizer would
                        # otherwise freeze the narration with no output.
                        if text_seen_at is None:
                            text_seen_at = time.monotonic()
                        elif time.monotonic() - text_seen_at > 30.0:
                            raise RuntimeError(
                                "no synthesized audio within 30s of narration text"
                            )
                    continue
                got_first_chunk = True
                if chunk is None:
                    # Narration finished naturally: ring the your-turn cue
                    # through the same stream so AEC still sees it.
                    cue = voice.turn_cue_samples
                    if cue is not None and last_end is not None:
                        wait_for_pause(max(0, cfg.turn_cue_delay_ms) / 1000, last_end)
                        if not self.cancel_event.is_set():
                            voice.play_and_wait(
                                AudioChunk(cue, 0.0, "", owner=self), self.cancel_event
                            )
                    return
                if last_end is not None and last_pause > 0:
                    wait_for_pause(last_pause, last_end)
                    if self.cancel_event.is_set():
                        return
                if not voice.play_and_wait(
                    chunk,
                    self.cancel_event,
                    while_waiting=self._drain_display_queue,
                ):
                    return
                self._drain_display_queue()
                last_end = time.monotonic()
                last_pause = chunk.pause_after
        except BaseException as exc:
            self.voice.notify(f"voice playback error: {exc}")
            self.cancel_event.set()
        finally:
            voice._drop_owner(self)
            if self._playback_marked:
                voice.mic.set_playback_active(False)
            self.finished.set()
            voice._session_finished(self)
            # Only explicit callers should dump remaining narration as text.
            # Transient playback errors can arrive after text has already
            # appeared sentence-by-sentence, and replaying the fallback here
            # duplicates visible narration.
            with voice.state_lock:
                fed = list(self.fed_display)
                shown_count = len(self.displayed_parts)
            undisplayed = fed[shown_count:]
            if undisplayed and self.show_remaining_on_cancel:
                voice.notify(self.remaining_notice)
                if voice.on_display is not None:
                    voice.on_display("".join(undisplayed))


# ── Orchestrator ─────────────────────────────────────────────────────────────

class VoiceLoop:
    """Owns the mic pipeline, the TTS pipeline, and the pause/interrupt policy.

    Host integration:
      - on_transcript(text, heard_text, was_speaking) is called from a worker
        thread whenever the player says something real. heard_text is the
        narration spoken so far (completed sentences) when they spoke over it.
      - begin_utterance()/feed_text()/finish_utterance()/cancel_utterance()
        drive narration speech; feed_text accepts streamed visible text.
    """

    def __init__(
        self,
        config: VoiceConfig | None = None,
        on_transcript: Callable[[str, str, str, bool], None] | None = None,
        on_notice: Callable[[str], None] | None = None,
        on_vad: Callable[[bool], None] | None = None,
        on_display: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or VoiceConfig()
        self.on_transcript = on_transcript
        self.on_notice = on_notice
        self.on_vad = on_vad
        self.on_display = on_display
        self.state_lock = threading.Lock()
        self.synth_lock = threading.Lock()
        self.pause_event = threading.Event()
        self.pause_holds: dict[int, float] = {}  # seq -> monotonic time paused
        self._pause_inconsistent_since: float | None = None
        self._log_lock = threading.Lock()
        self._log_file = None
        self.session: _TtsSession | None = None
        self.player: QueuedPlaybackHandle | None = None  # one stream, app lifetime
        self._player_lock = threading.Lock()
        self.turn_cue_samples: np.ndarray | None = None
        self.confirm_cue_samples: np.ndarray | None = None
        self._vad_seqs: set[int] = set()
        self._modifier_reader = ModifierKeyReader()
        self._keyboard_gate_stop = threading.Event()
        self._keyboard_gate_thread: threading.Thread | None = None
        self._keyboard_gate_lock = threading.Lock()
        self._keyboard_gate_state: tuple[bool, bool, bool, bool, bool] | None = None
        self._manual_vad_seq: int | None = None
        self._manual_vad_next_seq = -1
        self.stop_event = threading.Event()
        self.speaker: KokoroSpeaker | OmniVoiceSpeaker | None = None
        self.mic: MicLoop | None = None
        self.detector: ContinuousTurnDetector | None = None
        self.wake_gate: OpenWakeWordGate | None = None
        self.transcriber: WhisperCppTranscriber | None = None
        self._stream: sd.InputStream | None = None
        self._input_stream_lost = False  # one notice + one re-open per outage
        self._last_input_check = 0.0
        self._dispatch_thread: threading.Thread | None = None
        self.output_device = self.config.output_device
        self.output_channels = 1
        self.output_sample_rate = SAMPLE_RATE_TTS
        self._character_display_colors: dict[str, str] = {}

    def character_display_color(self, voice_name: str | None) -> str | None:
        """Voice-design display color for a named character voice, if cached.
        Misses are not memoized so a voice that finishes generating mid-turn
        starts coloring as soon as its cache entry lands."""
        if not voice_name:
            return None
        cache = getattr(self.speaker, "character_voice_cache", None)
        if cache is None:
            return None
        key = voice_name.casefold()
        color = self._character_display_colors.get(key)
        if color:
            return color
        try:
            cached = cache.cached_voice(voice_name)
        except Exception:
            return None
        if cached is None or not cached.display_color:
            return None
        self._character_display_colors[key] = cached.display_color
        return cached.display_color

    def notify(self, message: str) -> None:
        self.log(f"NOTICE {message}")
        if self.on_notice is not None:
            self.on_notice(message)
        else:
            print(message, file=sys.stderr)

    def log(self, message: str) -> None:
        """Append to the persistent voice log (best-effort) so silent failures
        in the audio pipeline can be diagnosed after the fact."""
        try:
            with self._log_lock:
                if self._log_file is None:
                    self._log_file = open(VOICE_LOG_PATH, "a", encoding="utf-8")
                now = time.time()
                stamp = time.strftime("%H:%M:%S", time.localtime(now))
                stamp = f"{stamp}.{int((now % 1.0) * 1000):03d}"
                self._log_file.write(f"{stamp} {message}\n")
                self._log_file.flush()
        except Exception:
            pass

    # ── the persistent output stream ─────────────────────────────────────

    def _on_chunk_start(self, chunk: AudioChunk) -> None:
        if chunk.owner is not None:
            chunk.owner.handle_chunk_start(chunk)

    def _on_chunk_done(self, chunk: AudioChunk) -> None:
        if chunk.owner is not None:
            chunk.owner.handle_chunk_done(chunk)

    def _open_player(self) -> QueuedPlaybackHandle:
        cfg = self.config
        player = QueuedPlaybackHandle(
            self.output_device,
            self.output_channels,
            self.output_sample_rate,
            cfg.output_blocksize,
            cfg.output_latency,
            playback_reference=(
                self.mic.playback_reference if (cfg.aec and self.mic is not None) else None
            ),
            pause_event=self.pause_event,
            on_chunk_start=self._on_chunk_start,
            on_chunk_done=self._on_chunk_done,
            notify=self.notify,
            log=self.log,
        )
        player.start()
        return player

    def _ensure_player(self) -> QueuedPlaybackHandle:
        with self._player_lock:
            if self.player is None:
                self.player = self._open_player()
            return self.player

    def _rebuild_player(self, reason: str) -> None:
        """Replace a dead output stream. A wedged CoreAudio close must never
        block narration: the old render thread is abandoned if it won't exit."""
        with self._player_lock:
            old = self.player
            self.notify(f"voice: playback stalled ({reason}) — resetting audio output")
            if old is not None:
                try:
                    old.stop()
                    if not old.wait(timeout=3.0):
                        self.log("abandoning wedged output stream thread")
                except BaseException as exc:
                    self.log(f"stalled player teardown: {exc}")
            time.sleep(0.5)  # let CoreAudio settle before reopening
            new = self._open_player()
            if old is not None:
                new.set_active_owner(old.active_owner)
            self.player = new

    def _drop_owner(self, session: "_TtsSession") -> None:
        player = self.player
        if player is not None and player.active_owner is session:
            player.set_active_owner(None)

    def play_and_wait(
        self,
        chunk: AudioChunk,
        cancel_event: threading.Event,
        while_waiting: Callable[[], None] | None = None,
    ) -> bool:
        """Play one chunk through the shared stream, detecting a dead or
        stalled stream (no render callbacks, closed, or chunk overdue) and
        rebuilding it. Returns False if cancelled; raises after repeated
        stalls so the caller's text fallback fires."""
        duration = chunk.samples.size / SAMPLE_RATE_TTS
        for attempt in (1, 2, 3):
            player = self._ensure_player()
            done = player.enqueue(chunk)
            baseline_callbacks = player.callbacks
            callback_deadline = time.monotonic() + 5.0
            chunk_deadline = time.monotonic() + duration + 60.0
            stalled: str | None = None
            while not done.wait(0.05):
                if while_waiting is not None:
                    while_waiting()
                if cancel_event.is_set():
                    return False  # owner deactivation discards the item
                now = time.monotonic()
                if self.pause_event.is_set():
                    callback_deadline = now + 5.0
                    chunk_deadline = now + duration + 60.0
                    continue
                if player.done_event.is_set():
                    stalled = "stream closed"
                elif player.callbacks == baseline_callbacks and now > callback_deadline:
                    stalled = "no audio callbacks"
                elif now > chunk_deadline:
                    stalled = "chunk overdue"
                if stalled is not None:
                    break
            if stalled is None:
                if while_waiting is not None:
                    while_waiting()
                return True
            if attempt == 3:
                raise RuntimeError(f"playback stalled repeatedly ({stalled})")
            self._rebuild_player(stalled)
        return True

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        cfg = self.config
        started = False

        def load_transcriber() -> WhisperCppTranscriber:
            transcriber = WhisperCppTranscriber(
                whisper_cli=cfg.whisper_cli,
                whisper_model=cfg.whisper_model,
                vad_model=cfg.whisper_vad_model,
                language=cfg.whisper_language,
                threads=cfg.whisper_threads,
                vad_threshold=cfg.vad_threshold,
                vad_min_speech_ms=cfg.vad_min_speech_ms,
                vad_min_silence_ms=cfg.vad_min_silence_ms,
                initial_prompt=cfg.whisper_prompt,
            )
            transcriber.check()
            return transcriber

        def load_speaker() -> KokoroSpeaker | OmniVoiceSpeaker:
            if cfg.tts_engine == "kokoro":
                return KokoroSpeaker(
                    lang=cfg.kokoro_lang,
                    voice=cfg.kokoro_voice,
                    speed=cfg.kokoro_speed,
                    device=cfg.kokoro_device,
                    volume=cfg.volume,
                )
            if cfg.tts_engine == "omnivoice":
                character_voice_cache = None
                if cfg.omnivoice_character_voices_enabled:
                    transcript_stem = (
                        cfg.omnivoice_character_voice_transcript_stem.strip()
                        or "default"
                    )
                    transcript_filename = (
                        cfg.omnivoice_character_voice_transcript_filename.strip()
                        or f"{transcript_stem}.txt"
                    )
                    character_voice_cache = CharacterVoiceCache(
                        cache_root=cfg.elevenlabs_voice_cache_dir,
                        transcript_filename_stem=transcript_stem,
                        transcript_filename=transcript_filename,
                        transcript_text=cfg.omnivoice_character_voice_transcript_text,
                        game_title=cfg.omnivoice_character_voice_game_title,
                        notify=self.notify,
                        log=self.log,
                        cost_recorder=cfg.external_cost_recorder,
                    )
                return OmniVoiceSpeaker(
                    model_name=cfg.omnivoice_model,
                    device=cfg.omnivoice_device,
                    dtype_name=cfg.omnivoice_dtype,
                    num_step=cfg.omnivoice_num_step,
                    instruct=cfg.omnivoice_instruct,
                    speed=cfg.omnivoice_speed,
                    clone_reference=cfg.omnivoice_clone_reference,
                    clone_transcript=cfg.omnivoice_clone_transcript,
                    clone_transcript_path=cfg.omnivoice_clone_transcript_path,
                    whisper_clone_reference=cfg.omnivoice_whisper_clone_reference,
                    whisper_clone_transcript=cfg.omnivoice_whisper_clone_transcript,
                    whisper_clone_transcript_path=cfg.omnivoice_whisper_clone_transcript_path,
                    character_voice_cache=character_voice_cache,
                    volume=cfg.volume,
                )
            raise RuntimeError(
                f"unknown TTS engine {cfg.tts_engine!r}; expected 'omnivoice' or 'kokoro'"
            )

        def load_cues() -> tuple[np.ndarray | None, np.ndarray | None]:
            turn_cue = None
            confirm_cue = None
            cue_path = Path(cfg.turn_cue_path)
            if cue_path.exists():
                try:
                    turn_cue = load_cue_samples(cue_path, cfg.turn_cue_volume)
                except Exception as exc:
                    self.notify(f"voice: turn cue disabled: {exc}")
            confirm_path = Path(cfg.confirm_cue_path)
            if confirm_path.exists():
                try:
                    confirm_cue = load_cue_samples(confirm_path, cfg.confirm_cue_volume)
                except Exception as exc:
                    self.notify(f"voice: confirm cue disabled: {exc}")
            return turn_cue, confirm_cue

        def load_wake_gate() -> OpenWakeWordGate | None:
            if not cfg.wake_word_enabled or not self._modifier_reader.available:
                return None
            try:
                wake_gate = OpenWakeWordGate.from_config(cfg)
                self.log(
                    "openWakeWord loaded "
                    f"models={wake_gate.describe()} "
                    f"sources={wake_gate.describe_sources()} "
                    f"threshold={wake_gate.threshold:.2f} "
                    f"car_mode={'on' if cfg.car_mode else 'off'} "
                    f"preprocess={'on' if cfg.wake_word_preprocess else 'off'}"
                )
                return wake_gate
            except Exception as exc:
                self.notify(f"voice: wake word disabled: {exc}")
                return None

        try:
            with ThreadPoolExecutor(
                max_workers=6,
                thread_name_prefix="if-engine-voice-load",
            ) as loader:
                transcriber_future = loader.submit(load_transcriber)
                speaker_future = loader.submit(load_speaker)
                cues_future = loader.submit(load_cues)
                silero_future = loader.submit(load_silero_vad)
                wake_future = loader.submit(load_wake_gate)

                self.output_device = choose_output_device(cfg.output_device)
                if self.output_device is not None and cfg.output_device is None:
                    try:
                        speaker_name = sd.query_devices(self.output_device, "output").get("name")
                        self.log(f"selected output device {self.output_device}: {speaker_name}")
                    except Exception:
                        pass
                self.output_channels = choose_output_channels(self.output_device)
                self.output_sample_rate = choose_output_sample_rate(
                    self.output_device, cfg.output_sample_rate
                )
                input_device = choose_input_device(cfg.input_device)
                if input_device is not None and cfg.input_device is None:
                    try:
                        mic_name = sd.query_devices(input_device, "input").get("name")
                        self.log(f"selected input device {input_device}: {mic_name}")
                    except Exception:
                        pass
                self.mic = MicLoop(
                    input_device=input_device,
                    detector_config=TurnDetectorConfig(
                        threshold=cfg.vad_threshold,
                        min_silence_ms=cfg.vad_min_silence_ms,
                        speech_pad_ms=cfg.vad_speech_pad_ms,
                        min_turn_seconds=cfg.min_turn_seconds,
                        max_turn_seconds=cfg.max_turn_seconds,
                        preroll_ms=cfg.preroll_ms,
                        postroll_ms=cfg.postroll_ms,
                        wake_preprocess=cfg.wake_word_preprocess,
                    ),
                    barge_ignore_ms=cfg.barge_ignore_ms,
                    barge_rms_multiplier=cfg.barge_rms_multiplier,
                    barge_min_rms=cfg.barge_min_rms,
                    barge_frames=cfg.barge_frames,
                    enable_aec=cfg.aec,
                    aec_delay_ms=cfg.aec_delay_ms,
                    aec_noise_suppression=cfg.aec_noise_suppression,
                    aec_agc=cfg.aec_agc,
                    aec_reference_delay_ms=cfg.aec_reference_delay_ms,
                    load_silero_model=False,
                    notify=self.notify,
                )
                self.mic.start_processing()
                self._stream = self.mic.stream()
                self._stream.start()
                self._ensure_player()

                noise_floor = self.mic.calibrate_noise(cfg.calibrate_seconds)

                self.turn_cue_samples, self.confirm_cue_samples = cues_future.result()
                self.transcriber = transcriber_future.result()
                self.speaker = speaker_future.result()
                self.wake_gate = wake_future.result()
                self.mic.set_silero_model(silero_future.result())

                self.detector = ContinuousTurnDetector(self.mic, noise_floor, wake_gate=self.wake_gate)
                self.detector.start()
                self._dispatch_thread = threading.Thread(
                    target=self._dispatch_loop, name="if-engine-voice-dispatch", daemon=True
                )
                self._dispatch_thread.start()
                self._start_keyboard_gate()
                started = True
                if isinstance(self.speaker, OmniVoiceSpeaker):
                    modes = []
                    if self.speaker.voice_clone_prompt is not None:
                        modes.append(f"clone ref={self.speaker.clone_reference}")
                    else:
                        modes.append("design")
                    if self.speaker.whisper_voice_clone_prompt is not None:
                        modes.append(f"whisper_ref={self.speaker.whisper_clone_reference}")
                    if self.speaker.character_voice_cache is not None:
                        modes.append(
                            "character_voice_cache="
                            f"{self.speaker.character_voice_cache.transcript_filename_stem}"
                        )
                    voice_mode = " ".join(modes)
                    self.log(
                        "started "
                        f"(tts=omnivoice model={cfg.omnivoice_model} "
                        f"device={self.speaker.device} dtype={cfg.omnivoice_dtype} "
                        f"steps={cfg.omnivoice_num_step} speed={cfg.omnivoice_speed:.2f} "
                        f"mode={voice_mode})"
                    )
                else:
                    self.log(f"started (tts=kokoro voice={cfg.kokoro_voice} speed={cfg.kokoro_speed})")
        finally:
            if not started:
                self.close()

    def close(self) -> None:
        self.stop_event.set()
        self._stop_keyboard_gate()
        self.cancel_utterance()
        if self.player is not None:
            try:
                self.player.stop()
                self.player.wait(timeout=2.0)
            except BaseException:
                pass
        if self.detector is not None:
            self.detector.stop()
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self.mic is not None:
            self.mic.stop_processing()
        # Close the persistent log handle last, after everything that might log.
        with self._log_lock:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

    # ── keyboard microphone gate ─────────────────────────────────────────

    def _start_keyboard_gate(self) -> None:
        if self.mic is None or not self._modifier_reader.available:
            return
        self._keyboard_gate_stop.clear()
        self._apply_keyboard_gate(self._modifier_reader.read())
        self._keyboard_gate_thread = threading.Thread(
            target=self._keyboard_gate_loop,
            name="if-engine-keyboard-gate",
            daemon=True,
        )
        self._keyboard_gate_thread.start()
        default_mode = "wake word" if self.wake_gate is not None else "Shift push-to-talk"
        self.log(
            "keyboard gate enabled "
            f"(default={default_mode}; Caps Lock=open mic; Shift=manual capture)"
        )

    def _stop_keyboard_gate(self) -> None:
        self._keyboard_gate_stop.set()
        if self._keyboard_gate_thread is not None:
            self._keyboard_gate_thread.join(timeout=1.0)
            self._keyboard_gate_thread = None
        self._apply_keyboard_gate(ModifierState(), transcribe_manual=False)

    def _keyboard_gate_loop(self) -> None:
        while not self._keyboard_gate_stop.wait(KEYBOARD_GATE_POLL_SECONDS):
            try:
                self._apply_keyboard_gate(self._modifier_reader.read())
            except Exception as exc:
                self.log(f"keyboard gate error: {exc}")

    def _apply_keyboard_gate(
        self, state: ModifierState, transcribe_manual: bool = True
    ) -> None:
        mic = self.mic
        if mic is None:
            return
        decision = keyboard_gate_decision(
            state,
            wake_gate_available=self.wake_gate is not None,
        )
        mic.set_wake_only(decision.wake_only)
        mic.set_microphone_blocked(decision.blocked)
        gate_state = (
            state.caps_lock,
            state.shift,
            decision.manual_active,
            decision.wake_only,
            decision.blocked,
        )
        with self._keyboard_gate_lock:
            if gate_state != self._keyboard_gate_state:
                self._keyboard_gate_state = gate_state
                self.log(
                    "keyboard gate state "
                    f"caps={'on' if state.caps_lock else 'off'} "
                    f"shift={'on' if state.shift else 'off'} "
                    f"manual={'on' if decision.manual_active else 'off'} "
                    f"open_mic={'on' if decision.open_mic else 'off'} "
                    f"wake_only={'on' if decision.wake_only else 'off'} "
                    f"blocked={'on' if decision.blocked else 'off'}"
                )
        with self._keyboard_gate_lock:
            current_seq = self._manual_vad_seq
            if decision.manual_active and current_seq is None:
                seq = self._manual_vad_next_seq
                self._manual_vad_next_seq -= 1
                self._manual_vad_seq = seq
            elif not decision.manual_active and current_seq is not None:
                seq = current_seq
                self._manual_vad_seq = None
            else:
                return

        if decision.manual_active:
            mic.begin_manual_capture()
            self.log(f"manual vad start seq={seq}")
            self._vad_started(seq)
            self._pause_for_candidate(seq)
            return

        samples = mic.end_manual_capture()
        min_samples = int(mic.detector_config.min_turn_seconds * SAMPLE_RATE_IN)
        self.log(f"manual vad end seq={seq} ({samples.size / SAMPLE_RATE_IN:.2f}s)")
        if transcribe_manual and samples.size >= min_samples:
            threading.Thread(
                target=self._process_candidate,
                args=(seq, samples, False),
                name=f"if-engine-manual-transcribe-{abs(seq)}",
                daemon=True,
            ).start()
        else:
            self._release_pause(seq)
            self._vad_finished(seq)

    # ── TTS API (called from the narration thread) ────────────────────────

    def begin_utterance(self) -> None:
        self.cancel_utterance()
        session = _TtsSession(self)
        with self.state_lock:
            self.session = session
            active_vad = bool(self._vad_seqs)
            if active_vad:
                now = time.monotonic()
                self.pause_holds = {seq: now for seq in self._vad_seqs}
                self.pause_event.set()
            else:
                self.pause_holds.clear()
                self.pause_event.clear()
        self._ensure_player().set_active_owner(session)
        self.log("utterance begin")
        session.start()

    def feed_text(self, text: str) -> None:
        with self.state_lock:
            session = self.session
        if session is not None:
            session.feed(text)

    def finish_utterance(self) -> None:
        with self.state_lock:
            session = self.session
        if session is not None:
            self.log("utterance text complete")
            session.finish_text()

    def cancel_utterance(
        self,
        show_remaining: bool = False,
        remaining_notice: str | None = None,
    ) -> None:
        with self.state_lock:
            session = self.session
            self.session = None
            self.pause_holds.clear()
            self.pause_event.clear()
        if session is not None:
            session.user_cancelled = True
            self.log("utterance cancelled")
            player = self.player
            if player is not None:
                # The stream stays open; deactivating the owner makes the
                # render callback drop this session's audio immediately.
                player.set_active_owner(None)
            session.show_remaining_on_cancel = show_remaining
            if remaining_notice is not None:
                session.remaining_notice = remaining_notice
            session.cancel()

    def is_speaking(self) -> bool:
        with self.state_lock:
            session = self.session
        return session is not None and session.is_speaking()

    def heard_text(self) -> str:
        with self.state_lock:
            return self._heard_text_locked()

    def displayed_text(self) -> str:
        """Narration text shown in the UI so far for the current utterance."""
        with self.state_lock:
            session = self.session
            if session is None:
                return ""
            return "".join(session.displayed_parts)

    def play_confirm_cue(self) -> None:
        """One-shot finger-click when a transcription is accepted as input.
        Played through the shared stream (ownerless items always play), so
        AEC sees it and no extra CoreAudio stream is spun up."""
        cue = self.confirm_cue_samples
        if cue is None:
            return
        try:
            self._ensure_player().enqueue(AudioChunk(cue, 0.0, ""))
            self.log("confirm cue")
        except Exception as exc:
            self.notify(f"voice confirm cue failed: {exc}")

    def _heard_text_locked(self) -> str:
        """Narration heard so far: completed sentences plus the word-aligned
        prefix of the sentence currently being played."""
        session = self.session
        if session is None:
            return ""
        parts = list(session.completed_chunks)
        chunk = session.current_chunk
        player = self.player
        if chunk is not None and player is not None:
            prefix = chunk.heard_prefix(
                player.played_seconds(), self.config.playback_alignment_margin_ms
            )
            if prefix:
                parts.append(prefix)
        return join_spoken_chunks(parts)

    def _session_finished(self, session: _TtsSession) -> None:
        with self.state_lock:
            if self.session is session:
                self.session = None
                self.pause_holds.clear()
                self.pause_event.clear()

    # ── speech events → pause / transcribe / submit ───────────────────────

    def _check_input_stream(self) -> None:
        """Watchdog for the mic capture stream. The OS silently stops delivering
        input callbacks when the device vanishes (AirPods drop, USB unplug) — no
        error fires, voice input just dies. Keying off callback cessation (a live
        or intentionally blocked mic still fires callbacks, so this never trips on
        silence or an intentional pause), surface a single notice and re-open the
        stream once per outage, mirroring the output-side rebuild."""
        if self.stop_event.is_set():
            return
        mic = self.mic
        stream = self._stream
        if mic is None or stream is None:
            return
        now = time.monotonic()
        if now - self._last_input_check < 1.0:
            return
        self._last_input_check = now
        if mic.input_callback_age() < INPUT_STREAM_STALL_SECONDS:
            self._input_stream_lost = False  # callbacks flowing; re-arm
            return
        if self._input_stream_lost:
            return  # already handled this outage — don't spam or fight it
        self._input_stream_lost = True
        self.notify("voice: microphone input lost — attempting to reconnect")
        try:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            mic.drain_audio(clear_playback_reference=False)
            new_stream = mic.stream()
            new_stream.start()
            self._stream = new_stream
            # Leave _input_stream_lost set until a real callback resets it, so a
            # re-open that opens but never delivers audio won't notice-loop.
            self.notify("voice: microphone input reconnected")
        except Exception as exc:
            self.notify(f"voice: microphone input could not be reconnected: {exc}")

    def _dispatch_loop(self) -> None:
        assert self.detector is not None
        while not self.stop_event.is_set():
            try:
                self._reconcile_pause_state()
            except Exception as exc:
                self.log(f"reconciler error: {exc}")
            try:
                self._check_input_stream()
            except Exception as exc:
                self.log(f"input watchdog error: {exc}")
            try:
                event = self.detector.events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if event.kind == "start":
                    self.log(f"vad start seq={event.seq}")
                    self._vad_started(event.seq)
                    self._pause_for_candidate(event.seq)
                elif event.kind == "wake":
                    suffix = f" {event.message}" if event.message else ""
                    self.log(f"wake word detected seq={event.seq}{suffix}")
                elif event.kind == "wake_debug":
                    self.log(event.message or f"wake debug seq={event.seq}")
                elif event.kind == "abort":
                    self.log(f"vad abort seq={event.seq}")
                    self._release_pause(event.seq)
                    self._vad_finished(event.seq)
                elif event.kind == "end" and event.samples is not None:
                    self.log(
                        f"vad end seq={event.seq} "
                        f"({event.samples.size / SAMPLE_RATE_IN:.2f}s)"
                    )
                    threading.Thread(
                        target=self._process_candidate,
                        args=(event.seq, event.samples, event.wake_triggered),
                        name=f"if-engine-transcribe-{event.seq}",
                        daemon=True,
                    ).start()
            except Exception as exc:
                self.notify(f"voice dispatch error: {exc}")

    def _vad_started(self, seq: int) -> None:
        with self.state_lock:
            was_active = bool(self._vad_seqs)
            self._vad_seqs.add(seq)
        if not was_active and self.on_vad is not None:
            self.on_vad(True)

    def _vad_finished(self, seq: int) -> None:
        with self.state_lock:
            was_active = bool(self._vad_seqs)
            self._vad_seqs.discard(seq)
            now_active = bool(self._vad_seqs)
        if was_active and not now_active and self.on_vad is not None:
            self.on_vad(False)

    def _pause_for_candidate(self, seq: int) -> None:
        with self.state_lock:
            session = self.session
            self.pause_holds[seq] = time.monotonic()
            self.pause_event.set()
        player = self.player
        self.log(
            f"pause seq={seq}"
            + ("" if session is not None and session.is_speaking() else " (pre-narration)")
        )
        if player is not None:
            player.pause()

    def _release_pause(self, seq: int) -> None:
        with self.state_lock:
            had = self.pause_holds.pop(seq, None)
            should_resume = not self.pause_holds
            if should_resume:
                self.pause_event.clear()
        player = self.player
        if had is not None:
            self.log(f"release seq={seq} resume={should_resume}")
        if should_resume and player is not None:
            player.resume()

    def _reconcile_pause_state(self) -> None:
        """Self-heal pause leaks: a hold that outlived its candidate, or a set
        pause flag with no holds, would otherwise freeze playback forever."""
        now = time.monotonic()
        resume_player = None
        with self.state_lock:
            expired = [
                seq for seq, since in self.pause_holds.items()
                if now - since > PAUSE_HOLD_MAX_SECONDS
            ]
            for seq in expired:
                self.pause_holds.pop(seq, None)
            inconsistent = self.pause_event.is_set() and not self.pause_holds
            if inconsistent:
                if self._pause_inconsistent_since is None:
                    self._pause_inconsistent_since = now
                elif now - self._pause_inconsistent_since > 1.0:
                    self.pause_event.clear()
                    resume_player = self.player
                    self._pause_inconsistent_since = None
            else:
                self._pause_inconsistent_since = None
        if expired:
            self.notify(f"voice: released stale pause (candidate {expired})")
        if resume_player is not None:
            self.log("reconciler: cleared orphaned pause flag")
            resume_player.resume()

    def _process_candidate(
        self,
        seq: int,
        samples: np.ndarray,
        strip_wake_word: bool = False,
    ) -> None:
        try:
            try:
                text = self.transcriber.transcribe_samples(samples, cancel_event=self.stop_event)
            except CancelledError:
                self._release_pause(seq)
                return
            except Exception as exc:
                self.notify(f"voice transcription error: {exc}")
                self._release_pause(seq)
                return
            if not text:
                # False alarm (breath, noise, echo): resume the narration.
                detail = " wake_triggered" if strip_wake_word else ""
                self.log(f"candidate seq={seq}: empty transcript{detail}")
                self._release_pause(seq)
                return
            if strip_wake_word:
                self.log(f"candidate seq={seq}: raw wake transcript {text[:120]!r}")
                stripped = strip_leading_wake_word(text)
                if stripped is None:
                    self.log(
                        f"candidate seq={seq}: rejected wake transcript without ok/okay prefix {text[:120]!r}"
                    )
                    self._release_pause(seq)
                    return
                self.log(f"candidate seq={seq}: stripped wake word -> {stripped[:120]!r}")
                text = stripped
                if not text:
                    self.log(f"candidate seq={seq}: wake word only")
                    self._release_pause(seq)
                    return
            text = capitalize_transcript_start(text)
            self.log(f"candidate seq={seq}: transcript {text[:80]!r}")
            with self.state_lock:
                session = self.session
                was_speaking = session is not None and session.is_speaking()
                heard = self._heard_text_locked() if was_speaking else ""
                displayed = "".join(session.displayed_parts) if was_speaking else ""
            self.cancel_utterance()
            # The confirm click is played by the host once the transcript is
            # actually accepted and submitted as player input.
            handler = self.on_transcript
            if handler is not None:
                handler(text, heard, displayed, was_speaking)
        finally:
            self._vad_finished(seq)
