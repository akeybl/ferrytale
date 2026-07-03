# README demo video pipeline

Everything needed to (re)create the demo video embedded at the top of the main
README: Ferrytale playing *Make It Good*, with real character voices, a fake
ElevenLabs "player" speaking the commands, and the exact audio the engine
played muxed under a recording of the live terminal UI.

## Manual dependencies (deliberately not in requirements*.txt)

- `ffmpeg` (with `ffprobe`) — `brew install ffmpeg`
- macOS **Terminal.app** (the render phase opens and records a window in it)
- Screen-recording permission for the terminal/IDE that runs `render_demo.py`
  (System Settings → Privacy & Security → Screen Recording)
- The normal Ferrytale voice stack (`scripts/install --voice`)
- `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` in `.env` (only for the one-time
  capture/design steps; rendering itself makes no API calls)

## One-time content creation (cached under .cache/demo/, gitignored)

```sh
# 1. Character voices for the game (clear first for a truly fresh set)
rm -rf .cache/elevenlabs-voices/make-it-good
.venv/bin/python scripts/pregenerate_character_voices.py make-it-good

# 2. The story: real Gemini narration, captured turn by turn.
#    Iterate freely — --show to review, --undo to drop the last turn,
#    --note to (invisibly) ask for tighter narration going forward.
.venv/bin/python scripts/demo/capture_session.py --new
.venv/bin/python scripts/demo/capture_session.py --turn "..."

# 3. The fake player's voice + one mp3 per scripted line
.venv/bin/python scripts/demo/player_voice.py --script scripts/demo/demo_script.json
```

`demo_script.json` defines the on-camera window: the last `window_turns`
player turns of the captured session, each step's `spoken` line matching the
captured player input exactly. Everything earlier appears as scrollback.

## Rendering (no API calls, fully repeatable)

```sh
.venv/bin/python scripts/demo/render_demo.py            # → .cache/demo/out/demo.mp4
.venv/bin/python scripts/demo/render_demo.py --keep-capture   # keep raw capture
```

What it does:

1. Truncates the captured session to just before the demo window and opens a
   Terminal.app window (dark "Pro" profile) running the engine's real live UI.
2. Replaces the Gemini client with a replayer that streams the captured
   narration after a short artificial "thinking" pause (`THINK_SECONDS`), so
   responses feel snappy on camera.
3. Plays each scripted player line aloud through the engine's own output
   stream (so it is heard in the recording), then submits it as voice input —
   the engine plays its own confirm click, echoes the input, and speaks the
   narration with the cached character voices and colors.
4. Tees every audio buffer the engine renders into a WAV (an in-process tap;
   no loopback driver needed) and screen-records the terminal with ffmpeg.
5. Aligns audio and video with a flash+beep clapperboard, trims it off, crops
   away the window title bar, and muxes the final `demo.mp4`.

Timing marks land in `.cache/demo/out/timing.json` for debugging.

## Iterating

- Different story: recapture turns (step 2) and update `demo_script.json`.
- Different player read: delete `.cache/demo/player-lines/` (or edit a line —
  changed lines re-synthesize automatically) and rerun `player_voice.py`.
- Different pacing: `THINK_SECONDS`, per-step `pause_before_seconds`, and
  `outro_hold_seconds` in `demo_script.json` / `render_demo.py`.
- The whole player voice: delete `.cache/demo/player-voice/` (re-designs and
  re-persists a new ElevenLabs voice).

## Embedding in the main README

GitHub only renders inline video players for files uploaded to its CDN, not
for committed files — so the README embeds `assets/demo.gif` (silent,
autoplays) linked to `assets/demo.mp4`, whose blob page plays with sound. For
a true inline player, drag `demo.mp4` into the README editor on github.com
once and replace the GIF block with the generated user-attachments URL.
