#!/usr/bin/env python3
"""Render the README demo video: replay the captured demo session through the
real Ferrytale terminal UI and voice pipeline, record the terminal, and mux in
the exact audio the engine played.

Two phases:

  outer (default)   Prepares a truncated render session, opens a Terminal.app
                    window running the inner phase, screen-records it with
                    ffmpeg, then trims/crops/muxes into the final mp4.

  --inner           Runs inside the spawned terminal: the real engine UI +
                    OmniVoice, with Gemini replaced by a replayer that streams
                    the captured narration (so "thinking" takes ~1.5s instead
                    of ~10s), scripted player lines played aloud through the
                    engine's own output stream (heard by the audio tap), and a
                    flash+beep clapperboard for audio/video alignment.

No Gemini or ElevenLabs calls happen during rendering. Recreate from scratch:

    .venv/bin/python scripts/demo/capture_session.py --new --turn ...   # story
    .venv/bin/python scripts/demo/player_voice.py --script scripts/demo/demo_script.json
    .venv/bin/python scripts/demo/render_demo.py                        # video

Manual dependencies (not in requirements*.txt): ffmpeg, macOS Terminal.app,
screen-recording permission for the terminal running this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEMO_DIR = BASE_DIR / ".cache" / "demo"
SESSIONS_DIR = DEMO_DIR / "sessions"
OUT_DIR = DEMO_DIR / "out"
DEFAULT_SCRIPT = BASE_DIR / "scripts" / "demo" / "demo_script.json"

READY_FLAG = OUT_DIR / "inner-ready.flag"
GO_FLAG = OUT_DIR / "inner-go.flag"
DONE_FLAG = OUT_DIR / "inner-done.flag"
TAP_WAV = OUT_DIR / "engine-audio.wav"
TIMING_JSON = OUT_DIR / "timing.json"
CAPTURE_MOV = OUT_DIR / "capture.mov"

THINK_SECONDS = 1.4          # artificial "thinking" before the reply streams
STREAM_WORDS_PER_CHUNK = 7
STREAM_CHUNK_SECONDS = 0.09
CLAP_SECONDS = 0.25          # white flash + beep length
TERMINAL_COLUMNS = 84
TERMINAL_ROWS = 22
TERMINAL_ORIGIN = (60, 80)   # points; window position on screen


def load_script(path: Path) -> dict:
    script = json.loads(path.read_text(encoding="utf-8"))
    script.setdefault("name", "demo")
    return script


def render_session_name(script: dict) -> str:
    return f"render-{script['name']}"


def turns_path(script: dict) -> Path:
    return OUT_DIR / f"turns-{script['name']}.json"


def final_mp4(script: dict) -> Path:
    return OUT_DIR / f"{script['name']}.mp4"


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "ferrytale_engine", BASE_DIR / "ferrytale.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── session preparation (outer) ──────────────────────────────────────────────

def prepare_render_session(script: dict) -> list[tuple[str, str]]:
    """Truncate the captured session before the demo window and return the
    window's (player_text, narrator_text) turns."""
    captured = SESSIONS_DIR / f"{script['session']}.jsonl"
    events = [json.loads(line) for line in captured.read_text(encoding="utf-8").splitlines()]
    player_indexes = [
        i for i, e in enumerate(events) if e.get("type") in ("player", "interruption")
    ]
    window_turns = int(script.get("window_turns", len(script["steps"])))
    if len(player_indexes) < window_turns:
        raise RuntimeError("captured session has fewer turns than the demo window")
    start = player_indexes[-window_turns]

    turns: list[tuple[str, str]] = []
    for i in player_indexes[-window_turns:]:
        narrator = next(
            (e for e in events[i + 1:] if e.get("type") == "narrator"), None
        )
        if narrator is None:
            raise RuntimeError(f"no narrator reply for player event {i}")
        turns.append((events[i]["text"], narrator["text"]))

    for step, (player_text, _) in zip(script["steps"], turns):
        if step["spoken"].strip() != player_text.strip():
            raise RuntimeError(
                "demo_script.json step does not match captured player text:\n"
                f"  script:   {step['spoken']!r}\n  captured: {player_text!r}"
            )

    render_path = SESSIONS_DIR / f"{render_session_name(script)}.jsonl"
    with open(render_path, "w", encoding="utf-8") as f:
        for e in events[:start]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return turns


# ── audio tap (inner) ────────────────────────────────────────────────────────

class AudioTap:
    """Tees every rendered output-stream buffer into a mono WAV whose
    timeline starts when the engine's output stream opens."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chunks: list = []
        self.samplerate: int | None = None

    def install(self) -> None:
        import numpy as np
        import sounddevice as sd

        tap = self
        real_stream = sd.OutputStream

        class TappedOutputStream(real_stream):
            def __init__(self, *args, **kwargs):
                inner_callback = kwargs.get("callback")
                tap.samplerate = int(kwargs.get("samplerate") or 48000)

                def wrapped(outdata, frames, time_info, status):
                    try:
                        inner_callback(outdata, frames, time_info, status)
                    finally:
                        with tap.lock:
                            tap.chunks.append(np.copy(outdata[:, 0]))

                kwargs["callback"] = wrapped
                super().__init__(*args, **kwargs)

        sd.OutputStream = TappedOutputStream

    def seconds(self) -> float:
        with self.lock:
            frames = sum(chunk.size for chunk in self.chunks)
        return frames / (self.samplerate or 48000)

    def save(self, path: Path) -> None:
        import numpy as np

        with self.lock:
            samples = (
                np.concatenate(self.chunks) if self.chunks else np.zeros(1, "float32")
            )
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype("<i2")
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.samplerate or 48000)
            f.writeframes(pcm16.tobytes())


# ── inner phase ──────────────────────────────────────────────────────────────

def decode_audio(path: Path, samplerate: int):
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
         "-ar", str(samplerate), "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def run_inner(script_path: Path) -> int:
    script = load_script(script_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for flag in (READY_FLAG, GO_FLAG, DONE_FLAG):
        flag.unlink(missing_ok=True)

    tap = AudioTap()
    tap.install()

    engine = load_engine()
    engine.SESSIONS_DIR = SESSIONS_DIR

    import numpy as np
    from voice import SAMPLE_RATE_TTS, AudioChunk

    turns = json.loads(turns_path(script).read_text(encoding="utf-8"))
    narrations = [n for _, n in turns]
    timing: dict = {"events": []}

    def mark(name: str) -> None:
        timing["events"].append({"name": name, "tap_seconds": tap.seconds()})

    # Capture the TerminalUI instance the run loop creates.
    hooks: dict = {}
    real_ui = engine.TerminalUI

    class DemoUI(real_ui):
        def __init__(self):
            super().__init__()
            hooks["ui"] = self

    engine.TerminalUI = DemoUI

    # A fake Gemini stream that replays the captured narration.
    class FakeStream:
        def __init__(self, text: str):
            words = text.split(" ")
            self.parts = [
                " ".join(words[i:i + STREAM_WORDS_PER_CHUNK])
                + (" " if i + STREAM_WORDS_PER_CHUNK < len(words) else "")
                for i in range(0, len(words), STREAM_WORDS_PER_CHUNK)
            ]

        def close(self):
            pass

        def __iter__(self):
            time.sleep(THINK_SECONDS)
            for part in self.parts:
                chunk = type("Chunk", (), {})()
                chunk.text = part
                chunk.usage_metadata = None
                yield chunk
                time.sleep(STREAM_CHUNK_SECONDS)

    class FakeModels:
        def generate_content_stream(self, model=None, contents=None, config=None):
            if not narrations:
                raise RuntimeError("demo replay ran out of captured narration")
            mark("narration_start")
            return FakeStream(narrations.pop(0))

    session = engine.Session(render_session_name(script))
    game = engine.Game(
        session,
        engine.load_transcript_text(script["game"]),
        voice_enabled=True,
        game_name=script["game"],
        game_title=engine.game_title_for(script["game"]),
        voice_options={
            "car_mode": False,
            "wake_word_enabled": False,
            "openwakeword_threshold": 0.9,
            "wake_word_preprocess": False,
            "omnivoice_whisper_tags": False,
        },
        resumed_session=True,
    )
    game.client = type("FakeClient", (), {"models": FakeModels()})()

    sys.path.insert(0, str(BASE_DIR / "scripts" / "demo"))
    import player_voice as pv

    line_paths = [pv.line_audio_path(step["spoken"]) for step in script["steps"]]
    for path in line_paths:
        if not path.exists():
            raise RuntimeError(f"missing player line audio: {path} — run player_voice.py")

    def wait_quiet(voice, initial: float) -> None:
        """Wait until narration has finished and stayed quiet for a second."""
        time.sleep(initial)
        quiet_since = None
        while True:
            if voice.is_speaking():
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= 1.0:
                return
            time.sleep(0.1)

    def driver() -> None:
        try:
            # Wait for voice startup (the run loop sets game.voice when ready).
            while game.voice is None or hooks.get("ui") is None:
                if not game.voice_enabled:
                    raise RuntimeError("voice startup failed; cannot render demo")
                time.sleep(0.1)
            voice = game.voice
            ui = hooks["ui"]
            voice.on_transcript = lambda *a, **k: None  # ignore the real mic
            voice.on_vad = lambda active: None
            time.sleep(1.0)

            READY_FLAG.touch()
            while not GO_FLAG.exists():  # outer starts ffmpeg, then signals
                time.sleep(0.1)

            # Compose the opening screen as a freshly displayed live page —
            # the previous turn's input pinned to the top with its narration
            # below (optionally trimmed to start mid-page) — rather than the
            # resumed-session scrollback.
            def compose_opening() -> str | None:
                opening = script.get("opening")
                if not opening:
                    return None
                prev_player = next(
                    (e for e in reversed(game.session.events)
                     if e.get("type") in ("player", "interruption")), None
                )
                prev_narrator = next(
                    (e for e in reversed(game.session.events)
                     if e.get("type") == "narrator"), None
                )
                if prev_player is None or prev_narrator is None:
                    raise RuntimeError("opening view needs a previous turn in the session")
                shown = prev_player.get("shown") or prev_player["text"]
                markup = engine.visible_markup_text(
                    prev_narrator["text"],
                    color_resolver=game.character_display_color,
                )
                marker = opening.get("narration_from") or ""
                cut = markup.find(marker) if marker else -1
                if cut >= 0:
                    markup = markup[cut:]
                return f"\n❯ {shown}\n\n{markup.rstrip()}\n"

            opening_text = compose_opening()

            # Clapperboard: white flash on screen + beep through the tap.
            restore = opening_text if opening_text is not None else ui.raw_output_text
            beep_t = np.arange(int(SAMPLE_RATE_TTS * CLAP_SECONDS)) / SAMPLE_RATE_TTS
            beep = (0.6 * np.sin(2 * math.pi * 1000 * beep_t)).astype(np.float32)
            mark("clap")
            ui.set_text(("█" * 80 + "\n") * 40)
            voice.player.enqueue(AudioChunk(samples=beep, pause_after=0.0))
            time.sleep(CLAP_SECONDS)
            # Show the opening pinned to its input echo, like live play (the
            # plain set_text bottom-follow would leave blank rows on top).
            with ui.lock:
                ui.raw_output_text = restore
                ui.spinner_active = False
                ui.spinner_pos = None
                ui.spinner_suffix = ""
                ui.scroll_mode = "anchor"
                ui.scroll_pending = "anchor"
            ui._schedule_render()
            time.sleep(1.2)

            for i, step in enumerate(script["steps"]):
                time.sleep(float(step.get("pause_before_seconds", 1.5)))
                samples = decode_audio(line_paths[i], SAMPLE_RATE_TTS)
                # Natural beat of silence before the player speaks.
                lead = np.zeros(int(SAMPLE_RATE_TTS * 0.25), dtype=np.float32)
                mark(f"player_line_{i}")
                done = voice.player.enqueue(
                    AudioChunk(samples=np.concatenate([lead, samples]), pause_after=0.0)
                )
                done.wait(timeout=30)
                time.sleep(0.15)
                mark(f"submit_{i}")
                ui.app.loop.call_soon_threadsafe(
                    ui.submit_callback,
                    engine.VoiceUtterance(step["spoken"], "", "", was_speaking=False),
                )
                # Wait for the narration turn to finish speaking.
                wait_quiet(voice, initial=THINK_SECONDS + 4.0)
                mark(f"turn_done_{i}")

            time.sleep(float(script.get("outro_hold_seconds", 3.0)))
            mark("end")
            tap.save(TAP_WAV)
            TIMING_JSON.write_text(json.dumps(timing, indent=2), encoding="utf-8")
            DONE_FLAG.touch()
            time.sleep(1.0)
        finally:
            try:
                ui = hooks["ui"]
                ui.app.loop.call_soon_threadsafe(ui.exit)
            except Exception:
                pass

    threading.Thread(target=driver, daemon=True, name="demo-driver").start()
    game.run()
    return 0


# ── outer phase ──────────────────────────────────────────────────────────────

def screen_capture_index() -> str:
    listing = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    match = re.search(r"\[(\d+)\] Capture screen 0", listing)
    if not match:
        raise RuntimeError("no avfoundation screen-capture device found")
    return match.group(1)


DEMO_PROFILE = "FerrytaleDemo"


def open_terminal_window(command: str) -> None:
    x, y = TERMINAL_ORIGIN
    osa = f'''
    tell application "Terminal"
        activate
        if not (exists settings set "{DEMO_PROFILE}") then
            make new settings set with properties {{name:"{DEMO_PROFILE}"}}
        end if
        tell settings set "{DEMO_PROFILE}"
            set background color to {{2000, 2200, 2800}}
            set normal text color to {{55000, 56000, 58000}}
            set bold text color to {{65000, 65000, 65000}}
            set cursor color to {{30000, 32000, 38000}}
            set font name to "Menlo"
            set font size to 18
        end tell
        set demoTab to do script "{command}"
        set current settings of demoTab to settings set "{DEMO_PROFILE}"
        set number of columns of demoTab to {TERMINAL_COLUMNS}
        set number of rows of demoTab to {TERMINAL_ROWS}
        set position of front window to {{{x}, {y}}}
    end tell
    '''
    subprocess.run(["osascript", "-e", osa], check=True)


def front_terminal_bounds() -> tuple[int, int, int, int]:
    osa = 'tell application "Terminal" to get bounds of front window'
    out = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True, check=True)
    x1, y1, x2, y2 = [int(v.strip()) for v in out.stdout.strip().split(",")]
    return x1, y1, x2, y2


def park_cursor_away_from(window_right: int, window_bottom: int) -> None:
    """Move the mouse pointer below/right of the demo window so it cannot be
    composited into the window-layer recording."""
    import ctypes
    import ctypes.util

    try:
        path = ctypes.util.find_library("ApplicationServices")
        app_services = ctypes.CDLL(path)

        class CGPoint(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

        app_services.CGWarpMouseCursorPosition.argtypes = [CGPoint]
        app_services.CGWarpMouseCursorPosition(
            CGPoint(float(window_right + 200), float(window_bottom + 200))
        )
    except Exception as exc:
        print(f"warning: could not move the cursor away: {exc}", file=sys.stderr)


def front_terminal_window_id() -> str:
    osa = 'tell application "Terminal" to get id of front window'
    out = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def video_dimensions(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def wait_for(flag: Path, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    while not flag.exists():
        if time.time() > deadline:
            raise RuntimeError(f"timed out waiting for {label}")
        time.sleep(0.2)


def detect_flash_seconds(video: Path, crop: tuple[int, int, int, int]) -> float:
    """First moment the terminal region goes bright white (the clapperboard)."""
    w, h, x, y = crop
    out = subprocess.run(
        ["ffprobe", "-f", "lavfi",
         f"movie={video},crop={w}:{h}:{x}:{y},signalstats", "-show_entries",
         "frame=pts_time:frame_tags=lavfi.signalstats.YAVG",
         "-of", "csv=p=0", "-v", "error"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows: list[tuple[float, float]] = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[0] and parts[1]:
            rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise RuntimeError("could not analyze the screen capture")
    # The flash is a large brightness spike over the terminal's baseline
    # (striped by line spacing, so use a relative threshold, not pure white).
    baseline = sorted(v for _, v in rows[: max(10, len(rows) // 10)])[0]
    for t, v in rows:
        if v > baseline + 60.0 and v > 60.0:
            return t
    raise RuntimeError(
        f"clapperboard flash not found in capture (baseline {baseline:.1f}, "
        f"max {max(v for _, v in rows):.1f})"
    )


def detect_beep_seconds(wav: Path) -> float:
    import numpy as np

    with wave.open(str(wav), "rb") as f:
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    loud = np.flatnonzero(np.abs(samples.astype(np.float32) / 32768.0) > 0.1)
    if loud.size == 0:
        raise RuntimeError("clapperboard beep not found in engine audio")
    return float(loud[0]) / rate


def run_outer(script_path: Path, keep_capture: bool) -> int:
    script = load_script(script_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    turns = prepare_render_session(script)
    turns_path(script).write_text(
        json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for flag in (READY_FLAG, GO_FLAG, DONE_FLAG):
        flag.unlink(missing_ok=True)

    python = BASE_DIR / ".venv" / "bin" / "python"
    inner = (
        f"cd {BASE_DIR} && IF_ENGINE_BARGE_MIN_RMS=9 IF_ENGINE_WAKE_WORD=0 "
        f"{python} scripts/demo/render_demo.py --inner --script {script_path}; exit"
    )
    print("opening demo terminal window…")
    open_terminal_window(inner.replace('"', '\\"'))

    print("waiting for the engine + voice models to load…")
    wait_for(READY_FLAG, 600, "inner readiness")

    x1, y1, x2, y2 = front_terminal_bounds()
    window_id = front_terminal_window_id()
    scale = 2  # Retina
    title_bar_pt = 28

    park_cursor_away_from(x2, y2)

    # Record the terminal window's own composited layer — other windows can
    # overlap it without appearing in the recording.
    print(f"starting window capture (window id {window_id})…")
    recorder = subprocess.Popen(
        ["screencapture", "-v", "-l", window_id, str(CAPTURE_MOV)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    GO_FLAG.touch()

    print("recording; waiting for the demo to finish…")
    wait_for(DONE_FLAG, 900, "demo completion")
    time.sleep(0.5)
    import signal

    recorder.send_signal(signal.SIGINT)
    recorder.wait(timeout=30)

    print("post-processing…")
    # The capture includes a shadow margin around the window (not perfectly
    # symmetric), so first crop generously, find the flash, then measure the
    # content's top edge from the flash frame itself.
    cap_w, cap_h = video_dimensions(CAPTURE_MOV)
    win_w, win_h = (x2 - x1) * scale, (y2 - y1) * scale
    margin_x = max(0, (cap_w - win_w) // 2)
    margin_y = max(0, (cap_h - win_h) // 2)
    rough = (win_w // 2 * 2, (win_h - title_bar_pt * scale) // 2 * 2,
             margin_x, margin_y + title_bar_pt * scale)
    flash = detect_flash_seconds(CAPTURE_MOV, rough)

    # The flash fills the content area with bright rows; the first bright row
    # below the title bar is where terminal content begins.
    import numpy as np

    frame_png = OUT_DIR / "flash-frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{flash + 0.06:.3f}",
         "-i", str(CAPTURE_MOV), "-frames:v", "1", str(frame_png)],
        check=True,
    )
    from PIL import Image

    frame = np.asarray(Image.open(frame_png).convert("L")).astype(np.float32)
    band = frame[:, margin_x + 40: margin_x + win_w - 40]
    row_means = band.mean(axis=1)
    content_top = None
    search_from = margin_y + int(title_bar_pt * scale * 0.6)
    for yy in range(search_from, min(frame.shape[0], margin_y + win_h)):
        if row_means[yy] > 60.0:
            content_top = yy
            break
    if content_top is None:
        content_top = margin_y + title_bar_pt * scale
    crop_x = margin_x
    crop_y = max(margin_y, content_top - 6)
    crop_w = win_w // 2 * 2
    crop_h = (margin_y + win_h - crop_y) // 2 * 2
    beep = detect_beep_seconds(TAP_WAV)
    video_start = flash + CLAP_SECONDS + 0.9   # after the flash clears
    audio_start = beep + CLAP_SECONDS + 0.9
    print(f"  flash at {flash:.2f}s (video), beep at {beep:.2f}s (audio)")

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", f"{video_start:.3f}", "-i", str(CAPTURE_MOV),
         "-ss", f"{audio_start:.3f}", "-i", str(TAP_WAV),
         "-filter:v",
         f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1280:-2,fps=30",
         "-c:v", "libx264", "-preset", "slow", "-crf", "23",
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k",
         "-shortest", "-movflags", "+faststart",
         str(final_mp4(script))],
        check=True,
    )
    if not keep_capture:
        CAPTURE_MOV.unlink(missing_ok=True)
    print(f"demo video: {final_mp4(script)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the README demo video.")
    parser.add_argument("--inner", action="store_true", help="run the in-terminal phase")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT,
                        help="demo script JSON (default: demo_script.json)")
    parser.add_argument("--keep-capture", action="store_true",
                        help="keep the raw screen capture for debugging")
    args = parser.parse_args()
    if args.inner:
        return run_inner(args.script)
    return run_outer(args.script, keep_capture=args.keep_capture)


if __name__ == "__main__":
    raise SystemExit(main())
