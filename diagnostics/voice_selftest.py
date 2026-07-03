#!/usr/bin/env python3
"""Offline self-test for voice.py: no microphone or speaker is opened."""

import base64
import importlib
import json
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
engine = importlib.import_module("ferrytale")
import elevenlabs_voices as ev
import voice as v

# 0. generated-audio cleanup
raw = np.array([0.0, 2.0, -2.0, 0.0], dtype=np.float32)
limited = v.postprocess_tts_audio(raw, volume=1.0, fade_ms=0.0, remove_dc=False)
assert float(np.max(np.abs(limited))) <= v.TTS_PEAK_HEADROOM + 1e-6, limited
faded = v.apply_edge_fade(np.ones(1000, dtype=np.float32), 1000, 10.0)
assert faded[0] == 0.0, faded[:3]
assert faded[-1] == 0.0, faded[-3:]
assert faded[20] > 0.99, faded[20]
left = v.postprocess_tts_audio(
    np.linspace(0.4, 1.2, 1000, dtype=np.float32),
    volume=1.0,
    sample_rate=1000,
)
right = v.postprocess_tts_audio(
    np.linspace(-1.2, -0.4, 1000, dtype=np.float32),
    volume=1.0,
    sample_rate=1000,
)
assert abs(float(left[-1] - right[0])) < 1e-6
assert v.keyboard_gate_decision(
    v.ModifierState(caps_lock=False, shift=False),
    wake_gate_available=False,
) == v.KeyboardGateDecision(
    manual_active=False,
    wake_only=False,
    blocked=True,
    open_mic=False,
)
assert v.keyboard_gate_decision(
    v.ModifierState(caps_lock=False, shift=False),
    wake_gate_available=True,
) == v.KeyboardGateDecision(
    manual_active=False,
    wake_only=True,
    blocked=False,
    open_mic=False,
)
assert v.keyboard_gate_decision(
    v.ModifierState(caps_lock=False, shift=True),
    wake_gate_available=False,
) == v.KeyboardGateDecision(
    manual_active=True,
    wake_only=False,
    blocked=False,
    open_mic=False,
)
assert v.keyboard_gate_decision(
    v.ModifierState(caps_lock=True, shift=False),
    wake_gate_available=True,
) == v.KeyboardGateDecision(
    manual_active=False,
    wake_only=False,
    blocked=False,
    open_mic=True,
)
prompt_stub = type("PromptStub", (), {})()
prompt_stub.voice_enabled = False
assert engine.Game.live_prompt_instruction_lines(prompt_stub) == [
    "No voice: type at any time and press Enter to send; /quit to leave.",
]
prompt_stub.voice_enabled = True
prompt_stub.prompt_wake_word_enabled = False
assert engine.Game.live_prompt_instruction_lines(prompt_stub) == [
    "Type at any time and press Enter to send; /quit to leave.",
    "Voice input: hold Shift while speaking.",
    "Caps Lock toggles open mic until you turn it off.",
]
prompt_stub.prompt_wake_word_enabled = True
assert engine.Game.live_prompt_instruction_lines(prompt_stub) == [
    "Type at any time and press Enter to send; /quit to leave.",
    'Voice input: say "Okay" and then speak, or hold Shift while speaking.',
    "Caps Lock toggles open mic until you turn it off.",
]
assert engine.is_session_instruction_line("Caps Lock toggles open mic until you turn it off.")

# 1. sentence chunker
ch = v.StreamingSentenceChunker(0.03, 0.07)
chunks = ch.add("Hello there. This is a tes")
assert [c.text for c in chunks] == ["Hello there."], chunks
chunks = ch.add("t of chunking! And more")
assert [c.text for c in chunks] == ["This is a test of chunking!"], chunks
chunks = ch.flush()
assert [c.text for c in chunks] == ["And more"], chunks
ch = v.StreamingSentenceChunker(0.03, 0.07)
chunks = ch.add("Mr. Verlac is gone.\n\nThe rain continues. ")
# the second sentence is held until the next word (or flush) shows it isn't
# a lowercase continuation like '" she cried'
assert [c.text for c in chunks] == ["Mr. Verlac is gone."], chunks
chunks = ch.flush()
assert [c.text for c in chunks] == ["The rain continues."], chunks
ch = v.StreamingSentenceChunker(0.03, 0.07)
chunks = ch.add("It was H. P. Lovecraft. ") + ch.add('"Stop!" she cried. Gone.')
assert [c.text for c in chunks] == ["It was H. P. Lovecraft.", '"Stop!" she cried.'], chunks
assert [c.text for c in ch.flush()] == ["Gone."]
ch = v.StreamingSentenceChunker(0.03, 0.20, 0.11, 0.13)
chunks = ch.add("First... second — third")
assert [(c.text, c.pause_after) for c in chunks] == [("First... second —", 0.11)], chunks
assert [c.text for c in ch.flush()] == ["third"], chunks
assert v.voice_text_with_dash_pauses("First—second") == "First ... — ... second"
assert v.voice_text_with_dash_pauses("First — second") == "First ... — ... second"
ch = v.StreamingSentenceChunker(0.03, 0.20, 0.11, 0.13)
chunks = ch.add("First...\n\nSecond")
assert [(c.text, c.pause_after, c.suffix) for c in chunks] == [("First...", 0.20, "\n\n")], chunks
assert [c.text for c in ch.flush()] == ["Second"], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13)
text = (
    "First sentence. Second sentence of the first paragraph.\n\n"
    "Second paragraph first sentence. Second paragraph second sentence.\n\n"
    "Final paragraph."
)
chunks = ch.add(text[:30]) + ch.add(text[30:90]) + ch.add(text[90:]) + ch.flush()
assert [(c.text, c.pause_after, c.suffix) for c in chunks] == [
    ("First sentence.", 0.03, " "),
    ("Second sentence of the first paragraph.", 0.20, "\n\n"),
    ("Second paragraph first sentence. Second paragraph second sentence.", 0.20, "\n\n"),
    ("Final paragraph.", 0.0, " "),
], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13)
chunks = ch.add("He says <whisper>quietly</whisper> and waits.") + ch.flush()
assert [c.text for c in chunks] == ["He says quietly and waits."], chunks
assert [[(s.text, s.instruct_suffix) for s in c.style_spans] for c in chunks] == [[
    ("He says quietly and waits.", ""),
]], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13, whisper_tags_enabled=True)
chunks = ch.add("He says <whisper>quietly</whisper> and waits.") + ch.flush()
assert [c.text for c in chunks] == ["He says quietly and waits."], chunks
assert [[(s.text, s.instruct_suffix) for s in c.style_spans] for c in chunks] == [[
    ("He says ", ""),
    ("quietly", "whisper"),
    (" and waits.", ""),
]], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13, whisper_tags_enabled=True)
chunks = ch.add("The door opens. <whisper>One. Two.</whisper> Three.") + ch.flush()
assert [c.text for c in chunks] == ["The door opens.", "One. Two. Three."], chunks
assert [[(s.text, s.instruct_suffix) for s in c.style_spans] for c in chunks] == [
    [("The door opens.", "")],
    [("One. Two.", "whisper"), (" Three.", "")],
], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13, whisper_tags_enabled=True)
chunks = ch.add('He says <voice name="Doctor Watson">hello</voice> and waits.') + ch.flush()
assert [c.text for c in chunks] == ["He says hello and waits."], chunks
assert [[(s.text, s.instruct_suffix, s.voice_name, s.voice_prompt) for s in c.style_spans] for c in chunks] == [[
    ("He says ", "", None, None),
    ("hello", "", "Doctor Watson", None),
    (" and waits.", "", None, None),
]], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13, whisper_tags_enabled=True)
chunks = ch.add(
    '<voice name="Doctor Watson">Keep <whisper>quiet</whisper>.</voice>'
) + ch.flush()
assert [c.text for c in chunks] == ["Keep quiet."], chunks
assert [[(s.text, s.instruct_suffix, s.voice_name, s.voice_prompt) for s in c.style_spans] for c in chunks] == [[
    ("Keep ", "", "Doctor Watson", None),
    ("quiet", "whisper", "Doctor Watson", None),
    (".", "", "Doctor Watson", None),
]], chunks
ch = v.StreamingOmniVoiceChunker(0.03, 0.20, 0.11, 0.13, whisper_tags_enabled=True)
chunks = ch.add('"<voice name="Doctor Watson">Hello, Holmes.</voice>"') + ch.flush()
assert [c.text for c in chunks] == ['"Hello, Holmes."'], chunks
assert [[(s.text, s.voice_name) for s in c.style_spans] for c in chunks] == [[
    ('"', None),
    ("Hello, Holmes.", "Doctor Watson"),
    ('"', None),
]], chunks
assert v.strip_audio_cue_tags_for_display(
    '<voice name="Doctor Watson">Hello</voice>, he says.'
) == "Hello, he says."
assert v.normalize_spoken_text(
    '<voice name="Doctor Watson"><whisper>Hello</whisper></voice>'
) == "Hello"

with tempfile.TemporaryDirectory() as tmp:
    prompt_block = ev.build_omnivoice_voice_prompt_block("sherlock", cache_root=tmp)
    assert "After that voice has been used once" in prompt_block
    assert "dialogue attribution words" in prompt_block
    assert "does not mean removing ordinary quotation marks" in prompt_block
    assert "Do not add prompt attributes" in prompt_block
    assert "prompt=" not in prompt_block
    assert "performance direction" in prompt_block
    assert "any speaker" in prompt_block
    assert "unnamed role speakers" in prompt_block
    assert "hotel bellboy" in prompt_block
    assert "hotel manager" in prompt_block
    assert "Halliday" not in prompt_block

with tempfile.TemporaryDirectory() as tmp:
    description_calls = []
    design_calls = []
    cost_events = []
    voice_notices = []
    voice_logs = []

    def fake_description_generator(**kwargs):
        description_calls.append(kwargs)
        return ev.GeneratedVoiceDescriptions(
            elevenlabs_voice_description=(
                "Native English, British. Male, middle-aged. Studio quality. "
                "Persona: loyal doctor. Emotion: warm, practical."
            ),
            omnivoice_description="male, middle-aged, low pitch, british accent",
            gemini_usage={"prompt": 1000, "cached": 100, "output": 50, "thoughts": 25},
            gemini_model="gemini-test",
        )

    def fake_voice_designer(**kwargs):
        design_calls.append(kwargs)
        return {
            "previews": [
                {
                    "audio_base_64": base64.b64encode(b"narrow mp3").decode("ascii"),
                    "generated_voice_id": "generated-voice-narrow",
                    "media_type": "audio/mpeg",
                },
                {
                    "audio_base_64": base64.b64encode(b"wide mp3").decode("ascii"),
                    "generated_voice_id": "generated-voice-wide",
                    "media_type": "audio/mpeg",
                },
                {
                    "audio_base_64": base64.b64encode(b"medium mp3").decode("ascii"),
                    "generated_voice_id": "generated-voice-medium",
                    "media_type": "audio/mpeg",
                }
            ],
            "text": ev.ELEVENLABS_PREVIEW_TEXT,
        }

    def fake_preview_scorer(audio):
        scores = {
            b"narrow mp3": 1000.0,
            b"wide mp3": 3000.0,
            b"medium mp3": 2000.0,
        }
        return ev.PreviewAudioScore(
            score=scores[audio],
            spectral_centroid_hz=scores[audio],
            rolloff_95_hz=scores[audio],
            rolloff_99_hz=scores[audio],
            low_band_db=-12.0,
            speech_band_db=-1.0,
            high_band_db=-8.0,
        )

    cache = ev.CharacterVoiceCache(
        cache_root=tmp,
        transcript_filename_stem="sherlock",
        transcript_filename="sherlock.txt",
        transcript_text="Doctor Watson follows Holmes through the fog.",
        game_title="Sherlock",
        description_generator=fake_description_generator,
        voice_designer=fake_voice_designer,
        preview_scorer=fake_preview_scorer,
        gemini_api_key="fake-gemini",
        elevenlabs_api_key="fake-eleven",
        notify=voice_notices.append,
        log=voice_logs.append,
        cost_recorder=cost_events.append,
    )
    metadata = cache.wait_for_ready("Doctor Watson", "warm and loyal")
    assert metadata is not None
    assert metadata.voice_json_path.exists()
    assert metadata.preview_path.exists()
    assert metadata.preview_path.read_bytes() == b"wide mp3"
    assert metadata.generated_voice_id == "generated-voice-wide"
    assert "broadcast quality" in metadata.voice_description
    assert metadata.omnivoice_description == "male, middle-aged, low pitch, british accent"
    assert len(description_calls) == 1
    assert description_calls[0]["prompt_hint"] == ""
    assert len(design_calls) == 1
    assert design_calls[0]["quality"] == 1
    assert len(cost_events) == 2
    assert cost_events[0]["service"] == "gemini"
    assert cost_events[0]["category"] == "character_voice_description"
    assert cost_events[0]["model"] == "gemini-test"
    assert cost_events[0]["usage"] == {
        "prompt": 1000,
        "cached": 100,
        "output": 50,
        "thoughts": 25,
    }
    assert cost_events[1]["service"] == "elevenlabs"
    assert cost_events[1]["category"] == "voice_design"
    assert cost_events[1]["characters"] == len(ev.ELEVENLABS_PREVIEW_TEXT)
    assert cost_events[1]["credits"] == len(ev.ELEVENLABS_PREVIEW_TEXT)
    assert cost_events[1]["preview_count"] == 3
    assert cost_events[1]["quality"] == 1
    assert voice_notices == []
    assert voice_logs == [
        "voice: cached ElevenLabs character voice 'Doctor Watson' from preview 2/3 score=3000.0"
    ]
    saved_metadata = json.loads(metadata.voice_json_path.read_text(encoding="utf-8"))
    assert saved_metadata["prompt_hint"] == ""
    assert saved_metadata["omnivoice_description"] == "male, middle-aged, low pitch, british accent"
    assert saved_metadata["selected_preview_index"] == 1
    assert saved_metadata["preview_count"] == 3
    assert saved_metadata["preview_audio_score"]["score"] == 3000.0
    same_path = cache.cache_dir_for("Doctor Watson")
    assert same_path == ev.character_voice_cache_dir(tmp, "sherlock", "Doctor Watson")

    cached = ev.CharacterVoiceCache(
        cache_root=tmp,
        transcript_filename_stem="sherlock",
        transcript_filename="sherlock.txt",
        transcript_text="unused",
        game_title="Sherlock",
        description_generator=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network")),
        voice_designer=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network")),
        cost_recorder=lambda _event: (_ for _ in ()).throw(AssertionError("cost")),
        gemini_api_key="fake-gemini",
        elevenlabs_api_key="fake-eleven",
    )
    cached_metadata = cached.wait_for_ready("Doctor Watson", "different hint")
    assert cached_metadata is not None
    assert cached_metadata.voice_json_path == metadata.voice_json_path

with tempfile.TemporaryDirectory() as tmp:
    description_calls = []
    design_calls = []
    first_description_started = threading.Event()
    release_first_description = threading.Event()

    def serialized_description_generator(**kwargs):
        character_name = kwargs["character_name"]
        existing_names = [
            voice.character_name for voice in kwargs["existing_voices"]
        ]
        description_calls.append((character_name, existing_names))
        if character_name == "First Voice":
            first_description_started.set()
            assert release_first_description.wait(5.0)
        return ev.GeneratedVoiceDescriptions(
            elevenlabs_voice_description=(
                f"Native English, British. {character_name}. Studio quality. "
                "Distinct persona and delivery."
            ),
            omnivoice_description=(
                "male, elderly, low pitch, british accent"
                if character_name == "First Voice"
                else "female, young adult, high pitch, american accent"
            ),
        )

    def serialized_voice_designer(**kwargs):
        design_calls.append(kwargs["voice_description"])
        return {
            "previews": [
                {
                    "audio_base_64": base64.b64encode(
                        f"{len(design_calls)} mp3".encode("ascii")
                    ).decode("ascii"),
                    "generated_voice_id": f"generated-{len(design_calls)}",
                    "media_type": "audio/mpeg",
                }
            ],
            "text": ev.ELEVENLABS_PREVIEW_TEXT,
        }

    serialized_cache = ev.CharacterVoiceCache(
        cache_root=tmp,
        transcript_filename_stem="serialized",
        transcript_filename="serialized.txt",
        transcript_text="First Voice and Second Voice speak in sequence.",
        game_title="Serialized",
        description_generator=serialized_description_generator,
        voice_designer=serialized_voice_designer,
        preview_scorer=lambda _audio: ev.PreviewAudioScore(
            score=1.0,
            spectral_centroid_hz=1.0,
            rolloff_95_hz=1.0,
            rolloff_99_hz=1.0,
            low_band_db=-1.0,
            speech_band_db=-1.0,
            high_band_db=-1.0,
        ),
        gemini_api_key="fake-gemini",
        elevenlabs_api_key="fake-eleven",
        max_workers=2,
    )
    first_future = serialized_cache.ensure_started("First Voice")
    assert first_description_started.wait(5.0)
    second_future = serialized_cache.ensure_started("Second Voice")
    time.sleep(0.2)
    assert description_calls == [("First Voice", [])], description_calls
    release_first_description.set()
    assert first_future.result(timeout=5.0) is not None
    assert second_future.result(timeout=5.0) is not None
    assert description_calls[1][0] == "Second Voice", description_calls
    assert "First Voice" in description_calls[1][1], description_calls
    assert len(design_calls) == 2

section = engine.format_cost_breakdown_section(
    "Entire session",
    [
        {
            "type": "narrator",
            "usage": {"prompt": 1000, "cached": 100, "output": 20, "thoughts": 10},
            "cost": 0.01,
        },
        {
            "type": "external_cost",
            "service": "gemini",
            "category": "character_voice_description",
            "usage": {"prompt": 500, "cached": 0, "output": 30, "thoughts": 5},
            "cost": 0.002,
        },
        {
            "type": "external_cost",
            "service": "elevenlabs",
            "category": "voice_design",
            "character_name": "Doctor Watson",
            "characters": 101,
            "credits": 101,
            "cost": 0.0101,
            "estimated": True,
        },
    ],
)
assert "Gemini narration" in section
assert "Gemini character voice descriptions" in section
assert "ElevenLabs voice design" in section
assert "101 credits" in section
assert "1 voice" in section
assert "estimated" in section

with tempfile.TemporaryDirectory() as tmp:
    started = threading.Event()
    release = threading.Event()
    cancelled_wait = threading.Event()
    voice_notices = []

    def slow_description_generator(**_kwargs):
        started.set()
        assert release.wait(5.0)
        return {
            "elevenlabs_voice_description": (
                "British accent, middle-aged male, low pitch, perfect audio quality, "
                "studio-quality recording, broadcast quality, full-band microphone "
                "capture with rich low end and open high frequencies."
            ),
            "omnivoice_description": "male, middle-aged, low pitch, british accent",
        }

    def slow_voice_designer(**_kwargs):
        return {
            "previews": [
                {
                    "audio_base_64": base64.b64encode(b"slow wide mp3").decode("ascii"),
                    "generated_voice_id": "generated-after-interrupt",
                    "media_type": "audio/mpeg",
                }
            ],
            "text": ev.ELEVENLABS_PREVIEW_TEXT,
        }

    interrupted_cache = ev.CharacterVoiceCache(
        cache_root=tmp,
        transcript_filename_stem="sherlock",
        transcript_filename="sherlock.txt",
        transcript_text="Doctor Watson follows Holmes through the fog.",
        game_title="Sherlock",
        description_generator=slow_description_generator,
        voice_designer=slow_voice_designer,
        preview_scorer=lambda _audio: ev.PreviewAudioScore(
            score=4000.0,
            spectral_centroid_hz=2500.0,
            rolloff_95_hz=4000.0,
            rolloff_99_hz=6000.0,
            low_band_db=-8.0,
            speech_band_db=-1.0,
            high_band_db=-6.0,
        ),
        gemini_api_key="fake-gemini",
        elevenlabs_api_key="fake-eleven",
        notify=voice_notices.append,
    )
    cancel_event = threading.Event()
    errors = []

    def interrupted_waiter():
        try:
            interrupted_cache.wait_for_ready(
                "Doctor Watson",
                cancel_event=cancel_event,
            )
        except ev.CharacterVoiceCancelled:
            cancelled_wait.set()
        except BaseException as exc:
            errors.append(exc)

    wait_thread = threading.Thread(target=interrupted_waiter)
    wait_thread.start()
    assert started.wait(5.0)
    cancel_event.set()
    wait_thread.join(5.0)
    assert cancelled_wait.is_set()
    assert not errors
    assert interrupted_cache.cached_voice("Doctor Watson") is None

    release.set()
    metadata_after_interrupt = interrupted_cache.wait_for_ready("Doctor Watson")
    assert metadata_after_interrupt is not None
    assert metadata_after_interrupt.generated_voice_id == "generated-after-interrupt"
    assert metadata_after_interrupt.preview_path.read_bytes() == b"slow wide mp3"
    assert voice_notices == []

sp = object.__new__(v.OmniVoiceSpeaker)
sp.instruct = "base voice"


class FakeCharacterVoiceCache:
    def __init__(self):
        self.started = []

    def ensure_started(self, voice_name, prompt_hint=None):
        self.started.append((voice_name, prompt_hint))

    def wait_for_ready(self, voice_name, prompt_hint=None, cancel_event=None):
        assert voice_name == "Doctor Watson"
        assert prompt_hint is None
        return ev.CachedCharacterVoice(
            character_name="Doctor Watson",
            normalized_name="doctor_watson",
            prompt_hint="",
            voice_description="Native English, British. Male doctor.",
            omnivoice_description="male, middle-aged, low pitch, british accent",
            generated_voice_id="generated-voice-1",
            preview_text=ev.ELEVENLABS_PREVIEW_TEXT,
            transcript_filename="sherlock.txt",
            transcript_filename_stem="sherlock",
            created_at="now",
            updated_at="now",
            cache_dir=Path("cache"),
            voice_json_path=Path("cache/voice.json"),
            preview_path=Path("cache/preview.mp3"),
        )


sp.character_voice_cache = FakeCharacterVoiceCache()
sp.character_voice_clone_prompts = {
    "doctor_watson": (
        "clone-prompt",
        Path("test.mp3"),
        "male, middle-aged, low pitch, british accent",
    )
}
sp.character_voice_clone_prompt_misses = set()
instruct, clone_prompt, character_voice, whisper = sp._resolve_style_instruct(
    "whisper",
    voice_name="Doctor Watson",
)
assert instruct == "male, middle-aged, low pitch, british accent, whisper", instruct
assert clone_prompt == "clone-prompt", clone_prompt
assert character_voice is True
assert whisper is True
calls = []


class FakeOmniVoiceModel:
    sampling_rate = v.SAMPLE_RATE_TTS

    def generate(self, **kwargs):
        calls.append(kwargs)
        return [np.zeros(v.SAMPLE_RATE_TTS, dtype=np.float32)]


sp.model = FakeOmniVoiceModel()
sp.config = object()
sp.speed = 1.0
sp.volume = 1.0
sp.voice_clone_prompt = "narrator-prompt"
sp.whisper_voice_clone_prompt = None
audio = sp.synthesize("hello", voice_name="Doctor Watson")
assert audio.size == v.SAMPLE_RATE_TTS
assert calls[-1]["voice_clone_prompt"] == "clone-prompt", calls[-1]
assert calls[-1]["instruct"] == "male, middle-aged, low pitch, british accent", calls[-1]
audio = sp.synthesize("narration")
assert audio.size == v.SAMPLE_RATE_TTS
assert calls[-1]["voice_clone_prompt"] == "narrator-prompt", calls[-1]
assert "instruct" not in calls[-1], calls[-1]
print("chunker ok")

# 2. Kokoro synthesis
sp = v.KokoroSpeaker(lang="a", voice="am_michael", speed=1.1, device="cpu", volume=0.85)
audio = sp.synthesize("The lantern lights glimmer across the bay.")
assert audio.size > v.SAMPLE_RATE_TTS, audio.size
print(f"kokoro ok ({audio.size / v.SAMPLE_RATE_TTS:.2f}s)")

# 3. whisper.cpp round trip on the synthesized audio
cfg = v.VoiceConfig()
tr = v.WhisperCppTranscriber(
    cfg.whisper_cli, cfg.whisper_model, cfg.whisper_vad_model,
    "en", 4, 0.5, 250, 550,
)
tr.check()
s16 = v.resample_linear(audio, v.SAMPLE_RATE_TTS, v.SAMPLE_RATE_IN)
text = tr.transcribe_samples(s16)
print("whisper:", repr(text))
assert "lantern" in text.lower().replace(" ", ""), text

# 4. WebRTC AEC
ref = v.PlaybackReference()
aec = v.WebRtcAec(ref, 80, True, False)
out = aec.process(np.zeros(v.AEC_FRAME_SIZE, dtype=np.float32), 0.0)
assert out.size == v.AEC_FRAME_SIZE
print("aec ok")

# 5. continuous turn detector on a synthetic feed (no real mic)
mic = v.MicLoop(
    input_device=None,
    detector_config=v.TurnDetectorConfig(0.5, 550, 80, 0.35, 18.0, 600, 250),
    barge_ignore_ms=450, barge_rms_multiplier=2.4, barge_min_rms=0.018, barge_frames=4,
    enable_aec=False, aec_delay_ms=80, aec_noise_suppression=True, aec_agc=False,
)
det = v.ContinuousTurnDetector(mic, noise_floor=0.0005)
det.start()
sig = np.concatenate([
    np.zeros(v.SAMPLE_RATE_IN // 2, np.float32),
    s16,
    np.zeros(v.SAMPLE_RATE_IN, np.float32),
])
for i in range(0, len(sig) - v.VAD_FRAME_SIZE, v.VAD_FRAME_SIZE):
    mic.audio_queue.put(sig[i:i + v.VAD_FRAME_SIZE].copy())
events = []
deadline = time.time() + 15
end_samples = None
while time.time() < deadline:
    try:
        e = det.events.get(timeout=0.2)
    except queue.Empty:
        continue
    events.append((e.kind, e.seq))
    if e.kind == "end":
        end_samples = e.samples
        break
det.stop()
print("detector events:", events)
assert ("start", 1) in events and ("end", 1) in events, events
assert end_samples is not None and end_samples.size > v.SAMPLE_RATE_IN, end_samples

# 6. transcribe the detected turn, as the live pipeline would
text = tr.transcribe_samples(end_samples)
print("detected turn transcript:", repr(text))
assert "lantern" in text.lower().replace(" ", ""), text

print("voice self-test ok")
