#!/usr/bin/env python3
"""Dump Ferrytale model prompts for review.

This script mirrors ferrytale.py's prompt composition without starting a game,
creating a session, requiring API keys, or calling Gemini.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types as pytypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parents[1]
ENGINE_PATH = BASE_DIR / "ferrytale.py"

if sys.version_info < (3, 12):
    sys.exit(
        "scripts/dump_prompts.py requires Python 3.12+. "
        "Run it with .venv/bin/python after ./play has bootstrapped the repo."
    )


class _PromptDumpGenAIObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def install_google_genai_prompt_stub() -> None:
    try:
        from google import genai as _unused_genai  # noqa: F401
        from google.genai import types as _unused_types  # noqa: F401

        return
    except ImportError:
        pass

    google_module = sys.modules.get("google")
    if google_module is None:
        google_module = pytypes.ModuleType("google")
        google_module.__path__ = []
        sys.modules["google"] = google_module

    genai_module = pytypes.ModuleType("google.genai")
    types_module = pytypes.ModuleType("google.genai.types")

    class _ServiceTierValue:
        def __init__(self, value: str):
            self.value = value

    class _ServiceTier:
        PRIORITY = _ServiceTierValue("priority")

    types_module.ServiceTier = _ServiceTier
    types_module.Content = _PromptDumpGenAIObject
    types_module.Part = _PromptDumpGenAIObject
    types_module.GenerateContentConfig = _PromptDumpGenAIObject
    types_module.ThinkingConfig = _PromptDumpGenAIObject

    genai_module.types = types_module
    genai_module.Client = _PromptDumpGenAIObject

    google_module.genai = genai_module
    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.types"] = types_module


def load_engine_module():
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    install_google_genai_prompt_stub()
    spec = importlib.util.spec_from_file_location("ferrytale_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load engine module from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine_module()


@dataclass(frozen=True)
class PromptSection:
    name: str
    role: str
    source: str
    text: str
    status: str = "active"


@dataclass(frozen=True)
class PromptOptions:
    game: str
    title: str
    voice_enabled: bool
    tts_engine: str
    fast_mode: bool
    car_mode: bool
    wake_word_enabled: bool
    wake_word_threshold: float
    wake_word_preprocess: bool
    omnivoice_whisper_tags: bool
    omnivoice_character_voices: bool
    include_transcript: bool


def add_bool_arg(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=None,
        **kwargs,
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dump the effective Gemini system prompt and engine-injected "
            "system-like messages for review."
        )
    )
    parser.add_argument(
        "--game",
        default="anchorhead",
        help="catalog slug used for the title and character voice cache "
        "(default: anchorhead)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="override the display title used in prompt templates",
    )
    add_bool_arg(
        parser,
        "--voice",
        help="render prompts as voice mode enabled/disabled "
        "(default: enabled, matching live play intent)",
    )
    parser.add_argument(
        "--tts-engine",
        choices=("omnivoice", "kokoro", "none"),
        default=None,
        help="TTS engine for the prompt (default: IF_ENGINE_TTS_ENGINE or omnivoice)",
    )
    add_bool_arg(
        parser,
        "--fast-mode",
        help="render prompts with Gemini priority service tier enabled/disabled "
        "(default: IF_ENGINE_FAST_MODE)",
    )
    add_bool_arg(
        parser,
        "--wake-word",
        help="render prompts with wake word enabled/disabled "
        "(default: IF_ENGINE_WAKE_WORD, or car-mode default)",
    )
    add_bool_arg(
        parser,
        "--car-mode",
        help="render prompts with car/Bluetooth wake-word defaults enabled/disabled "
        "(default: IF_ENGINE_CAR_MODE)",
    )
    parser.add_argument(
        "--wake-word-threshold",
        type=float,
        default=None,
        help="wake-word threshold (default: IF_ENGINE_WAKE_WORD_THRESHOLD, "
        "or IF_ENGINE_CAR_WAKE_WORD_THRESHOLD in car mode, or 0.9)",
    )
    add_bool_arg(
        parser,
        "--wake-word-preprocess",
        help="render prompts with wake-word preprocessing enabled/disabled "
        "(default: IF_ENGINE_WAKE_WORD_PREPROCESS, or car-mode default)",
    )
    parser.add_argument(
        "--wake-word-model",
        action="append",
        default=[],
        help="accepted for parity with play; listed in effective options only",
    )
    add_bool_arg(
        parser,
        "--whisper-tags",
        help="render prompts with OmniVoice whisper tags enabled/disabled "
        "(default: IF_ENGINE_OMNIVOICE_WHISPER_TAGS)",
    )
    add_bool_arg(
        parser,
        "--character-voices",
        help="render prompts with OmniVoice character voice tags enabled/disabled "
        "(default: IF_ENGINE_OMNIVOICE_CHARACTER_VOICES, enabled)",
    )
    parser.add_argument(
        "--include-transcript",
        action="store_true",
        help="include the full transcript message body. This may download the "
        "catalog transcript if it is missing and downloads are not disabled. "
        "Ignored when --placeholders is set.",
    )
    parser.add_argument(
        "--placeholders",
        action="store_true",
        help="replace dynamic prompt values with named placeholders while still "
        "using the supplied options to mark conditional sections active/inactive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the dump to a file instead of stdout",
    )
    return parser


def resolved_tts_engine(args: argparse.Namespace) -> str:
    if args.tts_engine:
        return args.tts_engine
    return engine.configured_tts_engine_for_prompt()


def resolve_options(args: argparse.Namespace) -> PromptOptions:
    title = args.title or engine.game_title_for(args.game)
    voice_enabled = True if args.voice is None else bool(args.voice)
    fast_mode = engine.DEFAULT_FAST_MODE if args.fast_mode is None else bool(args.fast_mode)

    voice_runtime = engine.resolve_voice_runtime_options(args)
    tts_engine = resolved_tts_engine(args)
    if not voice_enabled:
        tts_engine = "none"

    character_voices = (
        bool(args.character_voices)
        if args.character_voices is not None
        else engine.env_bool("IF_ENGINE_OMNIVOICE_CHARACTER_VOICES", True)
    )

    return PromptOptions(
        game=args.game,
        title=title,
        voice_enabled=voice_enabled,
        tts_engine=tts_engine,
        fast_mode=fast_mode,
        car_mode=bool(voice_runtime["car_mode"]),
        wake_word_enabled=bool(voice_runtime["wake_word_enabled"]),
        wake_word_threshold=float(voice_runtime["openwakeword_threshold"]),
        wake_word_preprocess=bool(voice_runtime["wake_word_preprocess"]),
        omnivoice_whisper_tags=bool(voice_runtime["omnivoice_whisper_tags"]),
        omnivoice_character_voices=character_voices,
        include_transcript=bool(args.include_transcript),
    )


def runtime_config_text(options: PromptOptions) -> str:
    voice_mode = "enabled" if options.voice_enabled else "disabled"
    tts_engine = options.tts_engine if options.voice_enabled else "none"
    return engine.RUNTIME_CONFIG_PROMPT_TEMPLATE.format(
        voice_mode=voice_mode,
        tts_engine=tts_engine,
        latency_tier="low latency" if options.fast_mode else "standard latency",
        wake_word=(
            "enabled" if options.voice_enabled and options.wake_word_enabled else "disabled"
        ),
        car_mode="on" if options.voice_enabled and options.car_mode else "off",
        wake_threshold=(
            f"{options.wake_word_threshold:.2f}" if options.voice_enabled else "n/a"
        ),
        wake_preprocess=(
            "on" if options.voice_enabled and options.wake_word_preprocess else "off"
        ),
        whisper_tags=(
            "enabled"
            if options.voice_enabled and options.omnivoice_whisper_tags
            else "disabled"
        ),
        character_voices=(
            "enabled"
            if (
                options.voice_enabled
                and options.tts_engine == "omnivoice"
                and options.omnivoice_character_voices
            )
            else "disabled"
        ),
    )


def runtime_config_placeholder_text() -> str:
    return engine.RUNTIME_CONFIG_PROMPT_TEMPLATE.format(
        voice_mode="{voice_mode}",
        tts_engine="{tts_engine}",
        latency_tier="{latency_tier}",
        wake_word="{wake_word}",
        car_mode="{car_mode}",
        wake_threshold="{wake_threshold}",
        wake_preprocess="{wake_preprocess}",
        whisper_tags="{whisper_tags}",
        character_voices="{character_voices}",
    )


def character_voice_placeholder_block() -> str:
    cache_header = "Cached OmniVoice character voices for this transcript:\n"
    prompt_block = engine.build_omnivoice_voice_prompt_block(
        "__prompt_review_placeholder__"
    )
    if cache_header not in prompt_block:
        return prompt_block
    prefix = prompt_block.split(cache_header, 1)[0]
    return (
        prefix
        + cache_header
        + '<voice name="{character_name}">{voice_description}</voice>\n'
        + "...\n"
    )


def system_prompt_sections(options: PromptOptions, *, placeholders: bool) -> list[PromptSection]:
    base_prompt = engine.SYSTEM_PROMPT_TEMPLATE.format(
        title="{game_title}" if placeholders else options.title
    )
    runtime_prompt = (
        runtime_config_placeholder_text() if placeholders else runtime_config_text(options)
    )
    character_voice_block = (
        character_voice_placeholder_block()
        if placeholders
        else engine.build_omnivoice_voice_prompt_block(options.game)
    )

    components: list[tuple[str, str, bool]] = [
        ("gemini.system_instruction.component.base", base_prompt, True),
        ("gemini.system_instruction.component.runtime_config", runtime_prompt, True),
        (
            "gemini.system_instruction.component.voice_mode",
            engine.VOICE_SYSTEM_PROMPT_ADDENDUM,
            options.voice_enabled,
        ),
        (
            "gemini.system_instruction.component.omnivoice",
            engine.OMNIVOICE_SYSTEM_PROMPT_ADDENDUM,
            options.voice_enabled and options.tts_engine == "omnivoice",
        ),
        (
            "gemini.system_instruction.component.omnivoice_character_voices",
            character_voice_block,
            (
                options.voice_enabled
                and options.tts_engine == "omnivoice"
                and options.omnivoice_character_voices
            ),
        ),
        (
            "gemini.system_instruction.component.omnivoice_whisper_tags",
            engine.OMNIVOICE_WHISPER_SYSTEM_PROMPT_ADDENDUM,
            (
                options.voice_enabled
                and options.tts_engine == "omnivoice"
                and options.omnivoice_whisper_tags
            ),
        ),
    ]
    full_prompt = "".join(text for _, text, active in components if active)

    sections = [
        PromptSection(
            name="gemini.system_instruction.full_active",
            role="system_instruction",
            source="GenerateContentConfig(system_instruction=...)",
            text=full_prompt,
        )
    ]
    for name, text, active in components:
        sections.append(
            PromptSection(
                name=name,
                role="system_instruction component",
                source="ferrytale.py prompt template",
                text=text,
                status="active" if active else "inactive for resolved options",
            )
        )
    return sections


def transcript_section(options: PromptOptions, *, placeholders: bool) -> PromptSection:
    preamble = engine.TRANSCRIPT_PREAMBLE_TEMPLATE.format(
        title="{game_title}" if placeholders else options.title
    )
    if placeholders:
        transcript_body = "\n{original_transcript_text}\n"
    elif options.include_transcript:
        transcript_body = engine.load_transcript_text(options.game)
    else:
        transcript_body = (
            "\n[TRANSCRIPT BODY OMITTED. Pass --include-transcript to print the "
            "full first user message.]\n"
        )
    return PromptSection(
        name="user.transcript_message.initial_history_item",
        role="user",
        source="build_history(session, transcript_message)",
        text=preamble + transcript_body + engine.TRANSCRIPT_EPILOGUE,
        status="possible / included at the start of model history",
    )


def system_like_user_sections(
    options: PromptOptions, *, placeholders: bool
) -> Iterable[PromptSection]:
    yield transcript_section(options, placeholders=placeholders)
    yield PromptSection(
        name="user.session_memory.wrapper_template",
        role="user",
        source="build_history(... SUMMARY_WRAPPER.format(summary=...))",
        text=engine.SUMMARY_WRAPPER,
        status="possible when a session has compacted history",
    )
    yield PromptSection(
        name="user.compaction_request",
        role="user",
        source="compact_history()",
        text=engine.COMPACTION_PROMPT,
        status="possible when the context reaches the compaction threshold",
    )
    yield PromptSection(
        name="user.interruption.typed_preamble",
        role="user",
        source="stream_narration() interruption handling",
        text=engine.INTERRUPTION_PREAMBLE,
        status="possible when typed input interrupts streaming narration",
    )
    yield PromptSection(
        name="user.interruption.voice_heard_template",
        role="user",
        source="voice_interruption_preamble(heard_text)",
        text=engine.VOICE_INTERRUPTION_HEARD,
        status="possible when speech interrupts audible narration after some text was heard",
    )
    yield PromptSection(
        name="user.interruption.voice_unheard",
        role="user",
        source="voice_interruption_preamble('')",
        text=engine.VOICE_INTERRUPTION_UNHEARD,
        status="possible when speech interrupts audible narration before text was heard",
    )
    yield PromptSection(
        name="user.engine_reminder.cardinal_direction",
        role="user",
        source="post-response guardrail retry",
        text=engine.CARDINAL_DIRECTION_REMINDER,
        status="possible when visible narration uses a cardinal direction",
    )
    yield PromptSection(
        name="user.engine_reminder.question_ending",
        role="user",
        source="post-response guardrail retry",
        text=engine.QUESTION_ENDING_REMINDER,
        status="possible when visible narration ends with a direct question",
    )
    yield PromptSection(
        name="user.engine_reminder.empty_reply",
        role="user",
        source="post-response guardrail retry",
        text=engine.EMPTY_REPLY_REMINDER,
        status="possible when a response contains no visible narration",
    )
    yield PromptSection(
        name="user.engine_reminder.story_progress",
        role="user",
        source="story progress nudge",
        text=engine.STORY_PROGRESS_REMINDER,
        status="possible after multiple pages without story progress",
    )


def options_section(options: PromptOptions, args: argparse.Namespace) -> PromptSection:
    wake_models = ", ".join(str(Path(item).expanduser()) for item in args.wake_word_model)
    lines = [
        f"repo: {BASE_DIR}",
        f"game: {options.game}",
        f"title: {options.title}",
        f"model: {engine.MODEL}",
        f"thinking_level: {engine.THINKING_LEVEL}",
        f"fast_mode: {str(options.fast_mode).lower()}",
        f"voice_enabled: {str(options.voice_enabled).lower()}",
        f"tts_engine: {options.tts_engine}",
        f"car_mode: {str(options.car_mode).lower()}",
        f"wake_word_enabled: {str(options.wake_word_enabled).lower()}",
        f"wake_word_threshold: {options.wake_word_threshold:.2f}",
        f"wake_word_preprocess: {str(options.wake_word_preprocess).lower()}",
        f"wake_word_models: {wake_models or '(none)'}",
        f"omnivoice_whisper_tags: {str(options.omnivoice_whisper_tags).lower()}",
        f"omnivoice_character_voices: {str(options.omnivoice_character_voices).lower()}",
        f"include_transcript: {str(options.include_transcript).lower()}",
        f"placeholder_mode: {str(args.placeholders).lower()}",
        "",
        "Environment variables consulted:",
        f"IF_ENGINE_TTS_ENGINE={os.environ.get('IF_ENGINE_TTS_ENGINE', '')}",
        f"IF_ENGINE_FAST_MODE={os.environ.get('IF_ENGINE_FAST_MODE', '')}",
        f"IF_ENGINE_CAR_MODE={os.environ.get('IF_ENGINE_CAR_MODE', '')}",
        f"IF_ENGINE_WAKE_WORD={os.environ.get('IF_ENGINE_WAKE_WORD', '')}",
        f"IF_ENGINE_WAKE_WORD_THRESHOLD={os.environ.get('IF_ENGINE_WAKE_WORD_THRESHOLD', '')}",
        f"IF_ENGINE_CAR_WAKE_WORD_THRESHOLD={os.environ.get('IF_ENGINE_CAR_WAKE_WORD_THRESHOLD', '')}",
        f"IF_ENGINE_WAKE_WORD_PREPROCESS={os.environ.get('IF_ENGINE_WAKE_WORD_PREPROCESS', '')}",
        f"IF_ENGINE_OMNIVOICE_WHISPER_TAGS={os.environ.get('IF_ENGINE_OMNIVOICE_WHISPER_TAGS', '')}",
        f"IF_ENGINE_OMNIVOICE_CHARACTER_VOICES={os.environ.get('IF_ENGINE_OMNIVOICE_CHARACTER_VOICES', '')}",
    ]
    return PromptSection(
        name="effective_options",
        role="metadata",
        source="scripts/dump_prompts.py",
        text="\n".join(lines),
    )


def render_section(section: PromptSection) -> str:
    text = section.text
    if text and not text.endswith("\n"):
        text += "\n"
    return (
        f"===== {section.name} =====\n"
        f"role: {section.role}\n"
        f"source: {section.source}\n"
        f"status: {section.status}\n"
        "----- BEGIN -----\n"
        f"{text}"
        "----- END -----\n"
    )


def build_dump(options: PromptOptions, args: argparse.Namespace) -> str:
    sections = [
        options_section(options, args),
        *system_prompt_sections(options, placeholders=args.placeholders),
        *system_like_user_sections(options, placeholders=args.placeholders),
    ]
    header = (
        "# Ferrytale Prompt Review Dump\n\n"
        "Only `gemini.system_instruction.full_active` is sent as the Gemini "
        "system instruction. The `user.*` sections are engine-injected "
        "user-role messages or templates that behave like system messages in "
        "specific situations. Conditional sections are marked active or "
        "inactive for the resolved options.\n\n"
    )
    return header + "\n".join(render_section(section) for section in sections)


def main() -> None:
    args = parser().parse_args()
    options = resolve_options(args)
    output = build_dump(options, args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
