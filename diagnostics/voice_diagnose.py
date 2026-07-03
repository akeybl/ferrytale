#!/usr/bin/env python3
"""Live microphone/AEC/VAD/wake-word diagnostic for IF engine voice input."""

from __future__ import annotations

import argparse
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad

import voice


DEFAULT_PLAYBACK_TEXT = (
    "This is IF engine diagnostic narration. The microphone should still hear "
    "you while this voice is playing. Speak over this sentence as if you were "
    "interrupting the game."
)


def parse_device(value: str | None):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def device_name(device, kind: str) -> str:
    try:
        info = sd.query_devices(device, kind)
        return str(info.get("name", device))
    except Exception:
        return str(device)


def print_devices(selected_input=None, selected_output=None) -> None:
    try:
        default_input, default_output = sd.default.device
    except Exception:
        default_input, default_output = None, None
    print(f"default input={default_input} output={default_output}")
    devices = sd.query_devices()
    for index, info in enumerate(devices):
        in_ch = int(info.get("max_input_channels") or 0)
        out_ch = int(info.get("max_output_channels") or 0)
        rate = info.get("default_samplerate")
        marks = []
        if selected_input == index:
            marks.append("selected-input")
        if selected_output == index:
            marks.append("selected-output")
        suffix = f" [{' '.join(marks)}]" if marks else ""
        print(
            f"{index}: in={in_ch} out={out_ch} rate={rate} "
            f"name={info.get('name')}{suffix}"
        )


def repeated_samples(unit: np.ndarray, seconds: float) -> np.ndarray:
    if unit.size == 0:
        return np.zeros(0, dtype=np.float32)
    target = max(unit.size, int(math.ceil((seconds + 1.0) * voice.SAMPLE_RATE_TTS)))
    repeats = int(math.ceil(target / unit.size))
    return np.tile(unit, repeats)[:target].astype(np.float32)


def make_fallback_tone(seconds: float, volume: float) -> np.ndarray:
    target = max(1, int(round(seconds * voice.SAMPLE_RATE_TTS)))
    t = np.arange(target, dtype=np.float32) / voice.SAMPLE_RATE_TTS
    carrier = (
        0.55 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 330.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
    )
    envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.2 * t) ** 2
    return np.clip(carrier * envelope * volume, -1.0, 1.0).astype(np.float32)


def make_playback_samples(config: voice.VoiceConfig, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.playback_mode == "tone":
        return make_fallback_tone(args.playback_seconds + 1.0, args.playback_volume), "tone"
    try:
        speaker = voice.KokoroSpeaker(
            lang=config.kokoro_lang,
            voice=config.kokoro_voice,
            speed=config.kokoro_speed,
            device=config.kokoro_device,
            volume=config.volume,
        )
        spoken = speaker.synthesize(args.playback_text)
        gap = np.zeros(int(round(0.25 * voice.SAMPLE_RATE_TTS)), dtype=np.float32)
        unit = np.concatenate((spoken * args.playback_volume, gap)).astype(np.float32)
        return repeated_samples(unit, args.playback_seconds), "kokoro"
    except Exception as exc:
        print(f"{stamp()} playback synth failed ({exc}); falling back to tone", file=sys.stderr)
        return make_fallback_tone(args.playback_seconds + 1.0, args.playback_volume), "tone"


@dataclass
class AudioStats:
    frames: int = 0
    samples: int = 0
    rms_sum: float = 0.0
    max_rms: float = 0.0
    peak: float = 0.0

    def add(self, block: np.ndarray) -> None:
        level = voice.rms(block)
        self.frames += 1
        self.samples += int(block.size)
        self.rms_sum += level
        self.max_rms = max(self.max_rms, level)
        if block.size:
            self.peak = max(self.peak, float(np.max(np.abs(block))))

    def summary(self) -> str:
        avg = self.rms_sum / self.frames if self.frames else 0.0
        return (
            f"frames={self.frames} avg_rms={avg:.5f} "
            f"max_rms={self.max_rms:.5f} peak={self.peak:.5f}"
        )


@dataclass
class PhaseState:
    name: str
    playback: bool
    started_at: float
    vad_triggered: bool = False
    vad_accepted: bool = False
    vad_started_at: float = 0.0
    vad_count: int = 0
    raw_wake_count: int = 0
    clean_wake_count: int = 0
    processed_wake_count: int = 0
    barge_count: int = 0
    hot_frames: int = 0
    playback_residual_rms: float = 1e-4


class AudioDiagnostic:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = voice.VoiceConfig()
        self.config.input_device = parse_device(args.input_device)
        self.config.output_device = parse_device(args.output_device)
        self.config.vad_threshold = args.vad_threshold
        self.config.vad_min_silence_ms = args.vad_min_silence_ms
        self.config.vad_speech_pad_ms = args.vad_speech_pad_ms
        if args.wake_model:
            self.config.openwakeword_models = [Path(item).expanduser() for item in args.wake_model]
        self.config.openwakeword_threshold = args.wake_threshold
        self.config.openwakeword_log_min_score = args.wake_log_min_score
        if args.wake_patience is not None:
            self.config.openwakeword_patience = args.wake_patience
        if args.barge_ignore_ms is not None:
            self.config.barge_ignore_ms = args.barge_ignore_ms
        if args.barge_min_rms is not None:
            self.config.barge_min_rms = args.barge_min_rms
        if args.barge_rms_multiplier is not None:
            self.config.barge_rms_multiplier = args.barge_rms_multiplier
        if args.barge_frames is not None:
            self.config.barge_frames = args.barge_frames
        self.input_device = voice.choose_input_device(self.config.input_device)
        self.output_device = voice.choose_output_device(self.config.output_device)
        self.output_channels = voice.choose_output_channels(self.output_device)
        self.output_sample_rate = voice.choose_output_sample_rate(
            self.output_device, self.config.output_sample_rate
        )
        self.raw_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=1000)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.playback_reference = voice.PlaybackReference()
        self.player: voice.QueuedPlaybackHandle | None = None
        self.aec = None
        self.vad_model = None
        self.raw_wake_gate: voice.OpenWakeWordGate | None = None
        self.clean_wake_gate: voice.OpenWakeWordGate | None = None
        self.processed_wake_gate: voice.OpenWakeWordGate | None = None
        self.raw_capture: list[np.ndarray] = []
        self.clean_capture: list[np.ndarray] = []
        self.queue_drops = 0
        self.noise_floor = 1e-4

    def setup(self) -> None:
        print(
            f"selected input {self.input_device}: "
            f"{device_name(self.input_device, 'input')}"
        )
        print(
            f"selected output {self.output_device}: "
            f"{device_name(self.output_device, 'output')}"
        )
        if self.args.list_devices:
            print_devices(self.input_device, self.output_device)
        if self.args.dry_run or self.args.seconds_only:
            return
        if self.args.no_aec:
            print("AEC disabled by --no-aec")
        elif voice.AudioProcessor is None:
            print("AEC unavailable: aec-audio-processing is not installed")
        else:
            self.aec = voice.WebRtcAec(
                playback_reference=self.playback_reference,
                delay_ms=self.config.aec_delay_ms,
                enable_ns=self.config.aec_noise_suppression,
                enable_agc=self.config.aec_agc,
                reference_delay_ms=self.config.aec_reference_delay_ms,
            )
            print(
                "AEC enabled "
                f"stream_delay_ms={self.config.aec_delay_ms} "
                f"reference_delay_ms={self.config.aec_reference_delay_ms} "
                f"noise_suppression={self.config.aec_noise_suppression}"
            )
        if not self.args.no_vad:
            self.vad_model = load_silero_vad()
            print(
                "Silero VAD loaded "
                f"threshold={self.config.vad_threshold:.2f} "
                f"min_silence_ms={self.config.vad_min_silence_ms}"
            )
        print(
            "Barge gate "
            f"ignore_ms={self.config.barge_ignore_ms} "
            f"min_rms={self.config.barge_min_rms:.5f} "
            f"frames={self.config.barge_frames} "
            f"rms_multiplier={self.config.barge_rms_multiplier:.2f}"
        )
        if not self.args.no_wake:
            self.raw_wake_gate = voice.OpenWakeWordGate.from_config(self.config)
            self.clean_wake_gate = voice.OpenWakeWordGate.from_config(self.config)
            self.processed_wake_gate = voice.OpenWakeWordGate.from_config(self.config)
            print(
                "openWakeWord loaded "
                f"models={self.clean_wake_gate.describe()} "
                f"threshold={self.clean_wake_gate.threshold:.2f} "
                f"sources={self.clean_wake_gate.describe_sources()} "
                "(scoring raw, post-AEC, and realtime post-AEC preprocessed audio)"
            )

    def new_vad(self) -> VADIterator | None:
        if self.vad_model is None:
            return None
        return VADIterator(
            self.vad_model,
            threshold=self.config.vad_threshold,
            sampling_rate=voice.SAMPLE_RATE_IN,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
        )

    def input_callback(self, indata, frames, time_info, status) -> None:
        if status:
            try:
                self.status_queue.put_nowait(str(status))
            except queue.Full:
                pass
        block = indata[:, 0].astype(np.float32).copy()
        frame_time = voice.callback_time_seconds(time_info, "inputBufferAdcTime")
        if frame_time is None:
            frame_time = time.monotonic()
        try:
            self.raw_queue.put_nowait((block, frame_time))
        except queue.Full:
            self.queue_drops += 1
            try:
                self.raw_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.raw_queue.put_nowait((block, frame_time))
            except queue.Full:
                pass

    def open_input_stream(self) -> sd.InputStream:
        return sd.InputStream(
            samplerate=voice.SAMPLE_RATE_IN,
            blocksize=voice.AEC_FRAME_SIZE,
            channels=1,
            dtype="float32",
            device=self.input_device,
            callback=self.input_callback,
        )

    def open_player(self) -> voice.QueuedPlaybackHandle:
        if self.player is not None:
            return self.player
        player = voice.QueuedPlaybackHandle(
            self.output_device,
            self.output_channels,
            self.output_sample_rate,
            self.config.output_blocksize,
            self.config.output_latency,
            playback_reference=self.playback_reference,
            notify=lambda message: print(f"{stamp()} {message}"),
            log=lambda message: print(f"{stamp()} playback {message}"),
        )
        player.start()
        self.player = player
        return player

    def close(self) -> None:
        self.stop_event.set()
        if self.player is not None:
            self.player.stop()
            self.player.wait(timeout=3.0)
            self.player = None

    def process_raw_frame(self, raw: np.ndarray, frame_time: float) -> np.ndarray:
        if raw.size != voice.AEC_FRAME_SIZE:
            raw = raw[: voice.AEC_FRAME_SIZE] if raw.size > voice.AEC_FRAME_SIZE else np.pad(
                raw, (0, voice.AEC_FRAME_SIZE - raw.size)
            )
        if self.aec is None:
            return raw.astype(np.float32, copy=False)
        return self.aec.process(raw, frame_time)

    def print_statuses(self) -> None:
        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                return
            print(f"{stamp()} INPUT STATUS {status}")

    def barge_hot(self, phase: PhaseState, frame: np.ndarray) -> tuple[bool, float, float]:
        level = voice.rms(frame)
        threshold = max(
            self.config.barge_min_rms,
            self.noise_floor * 5.0,
            phase.playback_residual_rms * self.config.barge_rms_multiplier,
        )
        hot = level >= threshold
        if not hot:
            phase.playback_residual_rms = (
                0.98 * phase.playback_residual_rms + 0.02 * max(level, 1e-4)
            )
        return hot, level, threshold

    def print_second_stats(
        self,
        phase: PhaseState,
        raw_stats: AudioStats,
        clean_stats: AudioStats,
    ) -> tuple[AudioStats, AudioStats]:
        print(
            f"{stamp()} AUDIO phase={phase.name} "
            f"raw[{raw_stats.summary()}] clean[{clean_stats.summary()}] "
            f"queue_drops={self.queue_drops}"
        )
        return AudioStats(), AudioStats()

    def process_wake_block(
        self,
        phase: PhaseState,
        source: str,
        gate: voice.OpenWakeWordGate | None,
        block: np.ndarray,
    ) -> None:
        if gate is None:
            return
        detection = gate.process_block(block)
        for message in gate.pop_debug_messages():
            print(f"{stamp()} WAKE {source} {message} phase={phase.name}")
        if detection is None:
            return
        if source == "raw":
            phase.raw_wake_count += 1
        elif source == "clean":
            phase.clean_wake_count += 1
        else:
            phase.processed_wake_count += 1
        print(
            f"{stamp()} WAKE {source} DETECTED phase={phase.name} "
            f"label={detection.label} score={detection.score:.3f}"
        )

    def process_vad_frame(
        self,
        phase: PhaseState,
        frame: np.ndarray,
        vad: VADIterator | None,
    ) -> None:
        if vad is None:
            return
        event = vad(torch.from_numpy(frame))
        if not phase.vad_triggered:
            if event and "start" in event:
                phase.vad_triggered = True
                phase.vad_started_at = time.monotonic()
                phase.vad_count += 1
                if phase.playback:
                    hot, level, threshold = self.barge_hot(phase, frame)
                    phase.hot_frames = 1 if hot else 0
                    elapsed_ms = int(round((time.monotonic() - phase.started_at) * 1000))
                    print(
                        f"{stamp()} VAD START phase={phase.name} playback_elapsed_ms={elapsed_ms} "
                        f"rms={level:.5f} hot={hot} hot_frames={phase.hot_frames}/"
                        f"{self.config.barge_frames} threshold={threshold:.5f}"
                    )
                    if (
                        elapsed_ms >= self.config.barge_ignore_ms
                        and phase.hot_frames >= self.config.barge_frames
                    ):
                        phase.vad_accepted = True
                        phase.barge_count += 1
                        print(
                            f"{stamp()} BARGE ACCEPT phase={phase.name} "
                            f"hot_frames={phase.hot_frames}/{self.config.barge_frames}"
                        )
                else:
                    phase.vad_accepted = True
                    print(f"{stamp()} VAD START phase={phase.name} accepted=true")
            return

        if phase.playback and not phase.vad_accepted:
            hot, level, threshold = self.barge_hot(phase, frame)
            if hot:
                phase.hot_frames += 1
            else:
                phase.hot_frames = max(0, phase.hot_frames - 1)
            elapsed_ms = int(round((time.monotonic() - phase.started_at) * 1000))
            if (
                elapsed_ms >= self.config.barge_ignore_ms
                and phase.hot_frames >= self.config.barge_frames
            ):
                phase.vad_accepted = True
                phase.barge_count += 1
                print(
                    f"{stamp()} BARGE ACCEPT phase={phase.name} "
                    f"rms={level:.5f} threshold={threshold:.5f} "
                    f"hot_frames={phase.hot_frames}/{self.config.barge_frames}"
                )

        if event and "end" in event:
            duration = time.monotonic() - phase.vad_started_at
            if phase.playback and not phase.vad_accepted:
                print(
                    f"{stamp()} BARGE REJECT phase={phase.name} reason=end_before_accept "
                    f"hot_frames={phase.hot_frames}/{self.config.barge_frames}"
                )
            print(
                f"{stamp()} VAD END phase={phase.name} "
                f"duration={duration:.2f}s accepted={phase.vad_accepted}"
            )
            phase.vad_triggered = False
            phase.vad_accepted = False
            phase.hot_frames = 0

    def collect_baseline(self, stream: sd.InputStream) -> None:
        seconds = max(0.0, self.args.calibrate_seconds)
        if seconds <= 0:
            return
        print(f"{stamp()} CALIBRATE stay quiet for {seconds:.1f}s")
        raw_stats = AudioStats()
        clean_stats = AudioStats()
        deadline = time.monotonic() + seconds
        clean_rms_values: list[float] = []
        while time.monotonic() < deadline and not self.stop_event.is_set():
            timeout = max(0.01, min(0.1, deadline - time.monotonic()))
            try:
                raw, frame_time = self.raw_queue.get(timeout=timeout)
            except queue.Empty:
                self.print_statuses()
                continue
            clean = self.process_raw_frame(raw, frame_time)
            raw_stats.add(raw)
            clean_stats.add(clean)
            clean_rms_values.append(voice.rms(clean))
            self.capture(raw, clean)
        if clean_rms_values:
            self.noise_floor = max(1e-4, float(np.median(clean_rms_values)))
        print(
            f"{stamp()} CALIBRATE done noise_floor={self.noise_floor:.5f} "
            f"raw[{raw_stats.summary()}] clean[{clean_stats.summary()}]"
        )

    def capture(self, raw: np.ndarray, clean: np.ndarray) -> None:
        if not self.args.save_prefix:
            return
        self.raw_capture.append(raw.copy())
        self.clean_capture.append(clean.copy())

    def run_phase(
        self,
        stream: sd.InputStream,
        name: str,
        seconds: float,
        playback: bool,
        playback_samples: np.ndarray | None = None,
        playback_source: str = "",
    ) -> PhaseState:
        phase = PhaseState(
            name=name,
            playback=playback,
            started_at=time.monotonic(),
            playback_residual_rms=max(self.noise_floor, 1e-4),
        )
        vad = self.new_vad()
        wake_preprocessor = voice.RealtimeWakePreprocessor()
        pending_clean = np.zeros(0, dtype=np.float32)
        raw_stats = AudioStats()
        clean_stats = AudioStats()
        next_report = time.monotonic() + 1.0
        deadline = time.monotonic() + max(0.0, seconds)
        if playback and playback_samples is not None and playback_samples.size:
            print(
                f"{stamp()} PHASE {name}: speak over playback now "
                f"({seconds:.1f}s, source={playback_source})"
            )
            self.open_player().enqueue(voice.AudioChunk(playback_samples, 0.0, "diagnostic playback"))
        else:
            print(f"{stamp()} PHASE {name}: speak now ({seconds:.1f}s)")

        while time.monotonic() < deadline and not self.stop_event.is_set():
            timeout = max(0.01, min(0.1, deadline - time.monotonic()))
            try:
                raw, frame_time = self.raw_queue.get(timeout=timeout)
            except queue.Empty:
                self.print_statuses()
                if time.monotonic() >= next_report:
                    raw_stats, clean_stats = self.print_second_stats(phase, raw_stats, clean_stats)
                    next_report = time.monotonic() + 1.0
                continue
            clean = self.process_raw_frame(raw, frame_time)
            raw_stats.add(raw)
            clean_stats.add(clean)
            self.capture(raw, clean)
            self.process_wake_block(phase, "raw", self.raw_wake_gate, raw)
            self.process_wake_block(phase, "clean", self.clean_wake_gate, clean)
            processed = wake_preprocessor.process(clean)
            self.process_wake_block(
                phase,
                "clean-rt",
                self.processed_wake_gate,
                processed,
            )
            pending_clean = np.concatenate((pending_clean, clean))
            while pending_clean.size >= voice.VAD_FRAME_SIZE:
                frame = pending_clean[: voice.VAD_FRAME_SIZE].copy()
                pending_clean = pending_clean[voice.VAD_FRAME_SIZE:]
                self.process_vad_frame(phase, frame, vad)
            self.print_statuses()
            if time.monotonic() >= next_report:
                raw_stats, clean_stats = self.print_second_stats(phase, raw_stats, clean_stats)
                next_report = time.monotonic() + 1.0
        if raw_stats.frames or clean_stats.frames:
            self.print_second_stats(phase, raw_stats, clean_stats)
        if phase.vad_triggered:
            print(
                f"{stamp()} VAD OPEN phase={phase.name} "
                f"accepted={phase.vad_accepted} ended=false"
            )
        print(
            f"{stamp()} SUMMARY phase={phase.name} vad_starts={phase.vad_count} "
            f"wake_raw={phase.raw_wake_count} wake_clean={phase.clean_wake_count} "
            f"wake_clean_rt={phase.processed_wake_count} "
            f"barge_accepts={phase.barge_count}"
        )
        return phase

    def save_wavs(self) -> None:
        if not self.args.save_prefix:
            return
        prefix = Path(self.args.save_prefix).expanduser()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        raw_path = prefix.with_name(prefix.name + "-raw.wav")
        clean_path = prefix.with_name(prefix.name + "-clean.wav")
        raw = np.concatenate(self.raw_capture).astype(np.float32) if self.raw_capture else np.zeros(0, dtype=np.float32)
        clean = (
            np.concatenate(self.clean_capture).astype(np.float32)
            if self.clean_capture
            else np.zeros(0, dtype=np.float32)
        )
        voice.write_wav(raw_path, raw, voice.SAMPLE_RATE_IN)
        voice.write_wav(clean_path, clean, voice.SAMPLE_RATE_IN)
        print(f"{stamp()} saved {raw_path}")
        print(f"{stamp()} saved {clean_path}")

    def run(self) -> int:
        self.setup()
        if self.args.dry_run or self.args.seconds_only:
            return 0
        playback_samples = None
        playback_source = ""
        if self.args.playback_seconds > 0 and not self.args.no_playback:
            playback_samples, playback_source = make_playback_samples(self.config, self.args)
            print(
                f"{stamp()} playback sample ready source={playback_source} "
                f"duration={playback_samples.size / voice.SAMPLE_RATE_TTS:.2f}s"
            )
        with self.open_input_stream() as stream:
            self.collect_baseline(stream)
            if self.args.silence_seconds > 0:
                self.run_phase(stream, "silence", self.args.silence_seconds, playback=False)
            if self.args.playback_seconds > 0 and not self.args.no_playback:
                self.run_phase(
                    stream,
                    "playback",
                    self.args.playback_seconds,
                    playback=True,
                    playback_samples=playback_samples,
                    playback_source=playback_source,
                )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose IF engine microphone routing, WebRTC AEC, Silero VAD, "
            "openWakeWord, and playback barge-in."
        )
    )
    parser.add_argument("--input-device", help="input device index/name; default uses IF engine priority")
    parser.add_argument("--output-device", help="output device index/name; default uses IF engine priority")
    parser.add_argument("--list-devices", action="store_true", help="print CoreAudio devices before the test")
    parser.add_argument("--dry-run", action="store_true", help="list setup choices, then exit")
    parser.add_argument("--seconds-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--calibrate-seconds", type=float, default=1.0, help="quiet baseline duration")
    parser.add_argument("--silence-seconds", type=float, default=12.0, help="listen without playback")
    parser.add_argument("--playback-seconds", type=float, default=12.0, help="listen while diagnostic playback runs")
    parser.add_argument("--no-playback", action="store_true", help="skip the playback/barge-in phase")
    parser.add_argument("--playback-mode", choices=["kokoro", "tone"], default="kokoro")
    parser.add_argument("--playback-text", default=DEFAULT_PLAYBACK_TEXT)
    parser.add_argument("--playback-volume", type=float, default=1.0)
    parser.add_argument("--no-aec", action="store_true")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--no-wake", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence-ms", type=int, default=550)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=80)
    parser.add_argument("--wake-threshold", type=float, default=0.4)
    parser.add_argument("--wake-log-min-score", type=float, default=0.05)
    parser.add_argument(
        "--wake-model",
        action="append",
        help="openWakeWord model path/name; repeat for multiple models",
    )
    parser.add_argument("--wake-patience", type=int)
    parser.add_argument("--barge-ignore-ms", type=int)
    parser.add_argument("--barge-min-rms", type=float)
    parser.add_argument("--barge-rms-multiplier", type=float)
    parser.add_argument("--barge-frames", type=int)
    parser.add_argument(
        "--save-prefix",
        help="write <prefix>-raw.wav and <prefix>-clean.wav for later inspection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diagnostic = AudioDiagnostic(args)

    def stop(_signum, _frame) -> None:
        diagnostic.stop_event.set()

    old_handler = signal.signal(signal.SIGINT, stop)
    try:
        return diagnostic.run()
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGINT, old_handler)
        diagnostic.save_wavs()
        diagnostic.close()


if __name__ == "__main__":
    raise SystemExit(main())
