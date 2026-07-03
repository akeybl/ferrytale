#!/usr/bin/env python3
"""VoiceLoop session/policy test: real Kokoro + whisper + output stream, but
near-silent volume and no microphone. Exercises begin/feed/finish, queued
sentence playback, the turn cue, pause-on-VAD, resume-on-empty-transcript,
interrupt-on-real-transcript, word-aligned heard text, and VAD indicator
callbacks."""

import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voice as v

cfg = v.VoiceConfig()
cfg.volume = 1e-4  # inaudible, but nonzero so chunks survive synthesis

vl = v.VoiceLoop(config=cfg)
vl.transcriber = v.WhisperCppTranscriber(
    cfg.whisper_cli, cfg.whisper_model, cfg.whisper_vad_model,
    "en", 4, 0.5, 250, 550,
)
vl.transcriber.check()

# The test plays at volume 1e-4 so it is inaudible; whisper cannot align audio
# that quiet, so boost it back to normal level for alignment only.
_orig_align = vl.transcriber.align_tts_samples


def _boosted_align(samples, prompt_text, cancel_event=None):
    boosted = np.clip(np.asarray(samples, dtype=np.float32) * 8000.0, -1.0, 1.0)
    return _orig_align(boosted, prompt_text, cancel_event=cancel_event)


vl.transcriber.align_tts_samples = _boosted_align
vl.speaker = v.KokoroSpeaker(
    lang="a", voice="am_michael", speed=1.3, device="cpu", volume=cfg.volume,
)
vl.mic = v.MicLoop(
    input_device=None,
    detector_config=v.TurnDetectorConfig(0.5, 550, 80, 0.35, 18.0, 600, 250),
    barge_ignore_ms=450, barge_rms_multiplier=2.4, barge_min_rms=0.018, barge_frames=4,
    enable_aec=False, aec_delay_ms=80, aec_noise_suppression=True, aec_agc=False,
)
vl.turn_cue_samples = v.load_cue_samples(Path("assets/turn-cue.wav"), 1e-4)
vl.confirm_cue_samples = v.load_cue_samples(Path("assets/confirm-cue.wav"), 1e-4)
vl.output_channels = v.choose_output_channels(None)
vl.output_sample_rate = v.choose_output_sample_rate(None, None)

transcripts = []
vad_flips = []
displays = []
vl.on_transcript = lambda text, heard, displayed, speaking: transcripts.append(
    (text, heard, displayed, speaking)
)
vl.on_vad = vad_flips.append
vl.on_display = displays.append

# 1. Full utterance drains on its own (including the turn cue); completed
#    chunks are recorded in order and carry alignment words.
vl.begin_utterance()
session = vl.session
vl.feed_text("The fog thickens over the bay. ")
vl.feed_text("Somewhere a bell tolls. ")
vl.finish_utterance()
deadline = time.time() + 90
while vl.is_speaking() and time.time() < deadline:
    time.sleep(0.1)
assert not vl.is_speaking(), "utterance did not finish draining"
assert session.completed_chunks == [
    "The fog thickens over the bay.",
    "Somewhere a bell tolls.",
], session.completed_chunks
assert displays == [
    "The fog thickens over the bay. ",
    "Somewhere a bell tolls. ",
], displays
assert vl.session is None
print("utterance drain ok (with turn cue + synced display)")

# 2. Dash/ellipsis splitting produces separate playback chunks.
vl.begin_utterance()
session = vl.session
vl.feed_text("He hesitates — the door is open... barely. ")
vl.finish_utterance()
deadline = time.time() + 90
while vl.is_speaking() and time.time() < deadline:
    time.sleep(0.1)
assert session.completed_chunks == [
    "He hesitates", "the door is open", "barely.",
], session.completed_chunks
print("dash/ellipsis chunks ok")

# 3. Pause on VAD start; word-aligned heard text mid-sentence; resume when
#    the candidate transcribes to nothing.
vl.begin_utterance()
session = vl.session
vl.feed_text(
    "The first sentence of this narration is deliberately long so that "
    "playback remains busy for quite a while as the words roll onward. "
    "The second sentence follows it. And then a third sentence arrives. "
)
vl.finish_utterance()
deadline = time.time() + 90
while time.time() < deadline:
    with vl.state_lock:
        chunk = session.current_chunk
    player = vl.player
    if chunk is not None and player is not None and player.played_seconds() > 1.6:
        break
    time.sleep(0.05)
assert vl.is_speaking()
vl._vad_started(1)
vl._pause_for_candidate(1)
assert vl.pause_event.is_set(), "playback did not pause on VAD start"
assert vad_flips == [True], vad_flips
heard_mid = vl.heard_text()
assert heard_mid, "expected word-aligned heard text mid-sentence"
assert "deliberately" not in heard_mid or len(heard_mid) < 200
displayed_mid = vl.displayed_text()
assert displayed_mid.startswith("The first sentence"), displayed_mid
print(f"pause-on-VAD ok; heard mid-sentence: {heard_mid!r}")
silence = np.zeros(v.SAMPLE_RATE_IN, dtype=np.float32)
vl._process_candidate(1, silence)
assert not vl.pause_event.is_set(), "playback did not resume after empty transcript"
assert not transcripts, transcripts
assert vl.is_speaking()
assert vad_flips == [True, False], vad_flips
print("resume-on-empty ok")

# 4. A real transcript interrupts: playback stops, transcript is delivered
#    with the heard-so-far text (including the partial sentence).
audio = v.KokoroSpeaker(
    lang="a", voice="am_michael", speed=1.1, device="cpu", volume=0.85,
).synthesize("I open the iron gate.")
player_speech = v.resample_linear(audio, v.SAMPLE_RATE_TTS, v.SAMPLE_RATE_IN)
vl._vad_started(2)
vl._pause_for_candidate(2)
vl._process_candidate(2, player_speech)
assert len(transcripts) == 1, transcripts
text, heard, displayed, speaking = transcripts[0]
assert "gate" in text.lower(), text
assert speaking is True
assert heard, "expected heard text on interrupt"
assert displayed.startswith("The first sentence"), displayed
assert not vl.is_speaking(), "TTS session not cancelled after real transcript"
assert session.cancel_event.is_set()
assert vad_flips == [True, False, True, False], vad_flips
print(f"interrupt-on-transcript ok: text={text!r} heard={heard!r}")
print(f"displayed at interrupt: {displayed!r}")

print("voice live-test ok")
