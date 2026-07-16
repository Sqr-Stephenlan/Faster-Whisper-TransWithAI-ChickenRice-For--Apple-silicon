"""Explicit option mapping for the CT2 and MLX adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import UnsupportedBackendOptionError

PROJECT_OPTIONS = {
    "segment_merge",
    "smart_split_with_vad",
    "target_chunk_duration_s",
}

CT2_OPTIONS = {
    "append_punctuations",
    "beam_size",
    "best_of",
    "chunk_length",
    "clip_timestamps",
    "compression_ratio_threshold",
    "condition_on_previous_text",
    "hallucination_silence_threshold",
    "hotwords",
    "initial_prompt",
    "language_detection_segments",
    "language_detection_threshold",
    "length_penalty",
    "log_progress",
    "log_prob_threshold",
    "max_initial_timestamp",
    "max_new_tokens",
    "multilingual",
    "no_repeat_ngram_size",
    "no_speech_threshold",
    "patience",
    "prefix",
    "prepend_punctuations",
    "prompt_reset_on_temperature",
    "repetition_penalty",
    "suppress_blank",
    "suppress_tokens",
    "temperature",
    "vad_filter",
    "vad_parameters",
    "without_timestamps",
    "word_timestamps",
}

MLX_OPTIONS = {
    "append_punctuations",
    "best_of",
    "clip_timestamps",
    "compression_ratio_threshold",
    "condition_on_previous_text",
    "fp16",
    "hallucination_silence_threshold",
    "initial_prompt",
    "length_penalty",
    "logprob_threshold",
    "max_initial_timestamp",
    "no_speech_threshold",
    "patience",
    "prefix",
    "prepend_punctuations",
    "sample_len",
    "suppress_blank",
    "suppress_tokens",
    "temperature",
    "verbose",
    "without_timestamps",
    "word_timestamps",
}

MLX_LEGACY_IGNORED_OPTIONS = {
    "no_repeat_ngram_size",
    "repetition_penalty",
    "vad_filter",
    "vad_parameters",
}


@dataclass(frozen=True)
class OptionMappingResult:
    options: dict[str, Any]
    ignored: tuple[str, ...] = ()


def _without_request_fields(options: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(options)
    mapped.pop("language", None)
    mapped.pop("task", None)
    for name in PROJECT_OPTIONS:
        mapped.pop(name, None)
    return mapped


def map_ct2_options(options: dict[str, Any]) -> dict[str, Any]:
    """Return only options accepted by Faster-Whisper 1.2.x."""
    mapped = _without_request_fields(options)
    if "logprob_threshold" in mapped and "log_prob_threshold" not in mapped:
        mapped["log_prob_threshold"] = mapped.pop("logprob_threshold")

    unknown = sorted(set(mapped).difference(CT2_OPTIONS))
    if unknown:
        names = ", ".join(unknown)
        raise UnsupportedBackendOptionError(f"Unsupported CTranslate2 option(s): {names}")
    return mapped


def map_mlx_options(
    options: dict[str, Any],
    *,
    task: str,
    strict_unknown: bool = False,
) -> OptionMappingResult:
    """Map project/CT2 configuration to mlx-whisper 0.4.3 options.

    mlx-whisper 0.4.3 has no beam-search decoder. ``beam_size=1`` is
    equivalent to its one-group greedy decoder and is therefore omitted.
    """
    mapped = _without_request_fields(options)
    ignored: set[str] = set()

    if "log_prob_threshold" in mapped and "logprob_threshold" not in mapped:
        mapped["logprob_threshold"] = mapped.pop("log_prob_threshold")

    beam_size = mapped.pop("beam_size", None)
    if beam_size not in (None, 1):
        raise UnsupportedBackendOptionError("mlx-whisper 0.4.3 does not implement beam search; use beam_size=1")

    for name in MLX_LEGACY_IGNORED_OPTIONS:
        if name in mapped:
            mapped.pop(name)
            ignored.add(name)

    if mapped.get("fp16") is False:
        raise UnsupportedBackendOptionError("The validated MLX model is FP16-only; fp16=false is not supported")
    mapped["fp16"] = True

    if task == "translate" and mapped.get("word_timestamps"):
        raise UnsupportedBackendOptionError("MLX translation does not support reliable word timestamps")
    mapped.setdefault("word_timestamps", False)
    mapped.setdefault("condition_on_previous_text", False)
    mapped.setdefault("verbose", None)

    unknown = sorted(set(mapped).difference(MLX_OPTIONS))
    if unknown and strict_unknown:
        names = ", ".join(unknown)
        raise UnsupportedBackendOptionError(f"Unsupported MLX option(s): {names}")
    for name in unknown:
        mapped.pop(name)
        ignored.add(name)

    return OptionMappingResult(options=mapped, ignored=tuple(sorted(ignored)))
