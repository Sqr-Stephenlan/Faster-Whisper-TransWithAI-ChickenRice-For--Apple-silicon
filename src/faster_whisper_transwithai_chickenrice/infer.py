#!/usr/bin/env python3
"""
Inference script with custom VAD injection support
"""

import argparse
import code
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyjson5

# Import accelerator/runtime-heavy dependencies defensively so `infer --help`
# still works when optional runtime libraries are unavailable.
_FASTER_WHISPER_IMPORT_ERROR = None
_CTRANSLATE2_IMPORT_ERROR = None

try:
    from faster_whisper.audio import decode_audio
except Exception as e:
    _FASTER_WHISPER_IMPORT_ERROR = e
    decode_audio = None

try:
    import ctranslate2
except Exception as e:
    _CTRANSLATE2_IMPORT_ERROR = e
    ctranslate2 = None

# Import our VAD injection system
# Import modern i18n module for translations
from . import i18n_modern as i18n
from . import inject_vad, uninject_vad
from .backends.base import BackendRequest, BackendResult, BackendSegment, WhisperBackend
from .backends.factory import BackendSelection, create_backend, select_backend
from .profiles import ProfileDefinition, get_profile
from .runtime_assets import (
    DEFAULT_MODELS_ROOT,
    AssetIssue,
    validate_ct2_model,
    validate_feature_extractor,
    validate_mlx_model,
    validate_vad_assets,
)
from .vad_manager import VadConfig, VadModelManager

# Convenience imports
_ = i18n._
format_duration = i18n.format_duration
format_percentage = i18n.format_percentage

WHISPER_TASKS = ("transcribe", "translate")
WHISPER_SAMPLING_RATE = 16_000
MAX_SMART_CHUNK_DURATION_S = 30.0
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_NO_INPUT = 2


def _normalize_whisper_task(task: Any) -> str:
    if not isinstance(task, str):
        raise ValueError(f"Whisper task must be one of {', '.join(WHISPER_TASKS)}")

    normalized = task.strip().lower()
    if normalized not in WHISPER_TASKS:
        raise ValueError(f"Invalid Whisper task '{task}'. Expected one of: {', '.join(WHISPER_TASKS)}")
    return normalized


def _require_ctranslate2():
    if ctranslate2 is None:
        raise RuntimeError(
            f"Failed to import ctranslate2. This build may be missing required GPU runtime libraries. "
            f"Original error: {_CTRANSLATE2_IMPORT_ERROR}"
        )
    return ctranslate2


def _require_audio_decoder():
    if decode_audio is None:
        raise RuntimeError(
            "Failed to import faster_whisper.audio. This build is missing the shared audio decoder. "
            f"Original error: {_FASTER_WHISPER_IMPORT_ERROR}"
        )
    return decode_audio


def parse_arguments():
    parser = argparse.ArgumentParser(description=_("app.description"))
    parser.add_argument("--model_name_or_path", type=str, default=None, help=_("args.model_path"))
    parser.add_argument(
        "--backend",
        choices=("auto", "ct2", "mlx"),
        default="auto",
        help=_("args.backend"),
    )
    parser.add_argument(
        "--model-variant",
        "--model_variant",
        dest="model_variant",
        default=None,
        help=_("args.model_variant"),
    )
    parser.add_argument("--device", type=str, default="auto", help=_("args.device"))
    parser.add_argument("--compute_type", type=str, default="auto", help=_("args.compute_type"))
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="CTranslate2 CPU worker threads; 0 lets CTranslate2 choose",
    )
    parser.add_argument(
        "--vad_threads",
        type=int,
        default=8,
        help="ONNX Runtime VAD intra-op threads; 0 lets ONNX Runtime choose",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help=_("args.overwrite"))
    parser.add_argument(
        "--audio_suffixes",
        type=str,
        default="wav,flac,mp3",
        help=_("args.audio_extensions"),
    )
    parser.add_argument("--sub_formats", type=str, default="lrc,vtt", help=_("args.subtitle_formats"))
    parser.add_argument("--output_dir", type=str, default=None, help=_("args.output_dir"))
    parser.add_argument(
        "--task",
        type=str,
        choices=WHISPER_TASKS,
        default=None,
        help=_("args.task"),
    )
    parser.add_argument(
        "--generation_config",
        type=str,
        default="generation_config.json5",
        help=_("args.config_file"),
    )
    parser.add_argument("--log_level", type=str, default="DEBUG", help=_("args.log_level"))

    # Subtitle post-processing options
    parser.add_argument(
        "--merge_segments",
        dest="merge_segments",
        action="store_true",
        default=None,
        help="Enable segment merge post-processing (override config file)",
    )
    parser.add_argument(
        "--no_merge_segments",
        dest="merge_segments",
        action="store_false",
        default=None,
        help="Disable segment merge post-processing (override config file)",
    )
    parser.add_argument(
        "--merge_max_gap_ms",
        type=int,
        default=None,
        help="Max allowed gap (ms) between segments for merging (override config file)",
    )
    parser.add_argument(
        "--merge_max_duration_ms",
        type=int,
        default=None,
        help="Max duration (ms) of a merged segment (override config file)",
    )
    parser.add_argument(
        "--smart_split_with_vad",
        type=str,
        default=None,
        help="Enable smart outer VAD chunking before backend inference (true/false)",
    )
    parser.add_argument(
        "--target_chunk_duration_s",
        type=float,
        default=None,
        help="Target duration for smart VAD chunks in seconds (override config file)",
    )

    # VAD parameter overrides (whisper_vad is always used)
    parser.add_argument("--vad_threshold", type=float, default=None, help=_("args.vad_threshold"))
    parser.add_argument(
        "--vad_min_speech_duration_ms",
        type=int,
        default=None,
        help=_("args.min_speech_duration"),
    )
    parser.add_argument(
        "--vad_min_silence_duration_ms",
        type=int,
        default=None,
        help=_("args.min_silence_duration"),
    )
    parser.add_argument("--vad_speech_pad_ms", type=int, default=None, help=_("args.speech_padding"))

    # Debug option for interactive console
    parser.add_argument(
        "--console",
        action="store_true",
        help="Launch interactive Python console for debugging",
    )

    # Batch inference options
    parser.add_argument(
        "--enable_batching",
        action="store_true",
        help="Enable CT2 batched inference for faster processing (requires more memory)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for batched inference (auto-detect if not specified)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size to try when auto-detecting (default: 8)",
    )

    parser.add_argument("base_dirs", nargs=argparse.REMAINDER, help=_("args.directories"))
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    """Resolve auto detection to the CTranslate2 device name."""
    ct2 = _require_ctranslate2()
    requested = (requested_device or "auto").strip().lower()
    if requested != "auto":
        return requested

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible in {"", "-1"}:
        return "cpu"
    try:
        return "cuda" if ct2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def select_best_compute_type(device: str) -> str:
    """
    Automatically select the best compute type based on device and available types.

    Preference order:
    - bfloat16 > float16 > int8 types > float32
    - Prefer int8 over float32 for better memory usage

    Args:
        device: The device to use ('cpu', 'cuda', or 'auto')

    Returns:
        The best available compute type for the device
    """
    ct2 = _require_ctranslate2()

    actual_device = resolve_device(device)
    if (device or "auto").strip().lower() == "auto":
        logger.info(_("info.auto_detected_device").format(device=actual_device))

    # Get supported compute types for the device.
    try:
        supported_types = ct2.get_supported_compute_types(actual_device)
    except Exception as e:
        logger.warning(_("warnings.compute_types_unavailable").format(device=actual_device, error=e))
        # Fallback to safe default
        return "int8" if actual_device == "cpu" else "float16"

    if actual_device == "cpu":
        preference_order = ["int8", "int8_float32", "float32"]
    else:
        preference_order = [
            "bfloat16",
            "float16",
            "int16",
            "int8_bfloat16",
            "int8_float16",
            "int8_float32",
            "int8",
            "float32",
        ]

    # Select the best available type based on preference
    for compute_type in preference_order:
        if compute_type in supported_types:
            logger.info(_("info.auto_selected_compute_type").format(compute_type=compute_type, device=actual_device))
            return compute_type

    # If nothing matched (shouldn't happen), use a safe default
    default = "int8" if actual_device == "cpu" else "float16"
    logger.warning(_("warnings.no_preferred_compute_type").format(default=default))
    return default


def require_local_runtime_assets(
    model_path: Path,
    *,
    backend: str = "ct2",
    validate_model: bool = True,
) -> None:
    """Fail before inference if any local-only runtime asset is incomplete."""
    issues = []
    model_issues: list[AssetIssue] = []
    if validate_model:
        if backend == "ct2":
            _checked, model_issues = validate_ct2_model(model_path)
        elif backend == "mlx":
            _checked, model_issues = validate_mlx_model(model_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    _checked, vad_issues, _providers = validate_vad_assets(DEFAULT_MODELS_ROOT)
    _checked, feature_issues = validate_feature_extractor(DEFAULT_MODELS_ROOT)
    issues.extend(model_issues)
    issues.extend(vad_issues)
    issues.extend(feature_issues)
    if issues:
        detail = "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)
        raise RuntimeError(
            "Local model assets are not ready; inference was not started.\n"
            f"{detail}\nRun './dev.sh python scripts/macos_doctor.py --mode all'."
        )


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


@dataclass
class Segment:
    start: int  # ms
    end: int  # ms
    text: str


@dataclass(frozen=True)
class SpeechSpan:
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SegmentMergeOptions:
    enabled: bool = True
    max_gap_ms: int = 2_000
    max_duration_ms: int = 20_000


@dataclass(frozen=True)
class SmartSplitOptions:
    enabled: bool = True
    target_chunk_duration_s: float = MAX_SMART_CHUNK_DURATION_S
    split_window_factor: float = 0.4


def _normalize_merge_text(text: str) -> str:
    return " ".join(text.strip().split())


def vad_segments_to_clip_timestamps(
    vad_segments: list[dict[str, Any]], sampling_rate: int = WHISPER_SAMPLING_RATE, *, batched: bool = False
) -> list[float] | list[dict[str, float]]:
    if batched:
        clips: list[dict[str, float]] = []
        for segment in vad_segments:
            start = float(segment["start"]) / sampling_rate
            end = float(segment["end"]) / sampling_rate
            if end > start:
                clips.append({"start": start, "end": end})
        return clips

    timestamps: list[float] = []
    for segment in vad_segments:
        start = float(segment["start"]) / sampling_rate
        end = float(segment["end"]) / sampling_rate
        if end > start:
            timestamps.extend([start, end])
    return timestamps


def vad_segments_to_speech_spans(
    vad_segments: list[dict[str, Any]], sampling_rate: int = WHISPER_SAMPLING_RATE
) -> list[SpeechSpan]:
    spans: list[SpeechSpan] = []
    for segment in vad_segments:
        start = float(segment["start"]) / sampling_rate
        end = float(segment["end"]) / sampling_rate
        if end > start:
            spans.append(SpeechSpan(start=start, end=end))
    return spans


def create_contiguous_chunks(
    segments: list[SpeechSpan],
    max_duration: float,
    total_duration: float,
    split_window_factor: float = 0.4,
) -> list[AudioChunk]:
    if max_duration <= 0:
        raise ValueError("max_duration must be greater than 0")
    if total_duration <= 0:
        return []
    if total_duration <= max_duration:
        return [AudioChunk(0, 0.0, total_duration)]

    chunks: list[AudioChunk] = []
    current_start = 0.0
    sorted_segments = sorted((span for span in segments if span.end > span.start), key=lambda span: span.start)

    while current_start < total_duration:
        potential_end = current_start + max_duration
        if potential_end >= total_duration:
            chunks.append(AudioChunk(len(chunks), current_start, total_duration))
            break

        decision_zone_start = current_start + (max_duration * (1 - split_window_factor))
        best_split: float | None = None
        best_gap_duration = 0.0

        for previous, current in zip(sorted_segments, sorted_segments[1:], strict=False):
            gap_start = previous.end
            gap_end = current.start
            if decision_zone_start <= gap_start and gap_end <= potential_end:
                gap_duration = gap_end - gap_start
                if gap_duration > 0.1 and gap_duration > best_gap_duration:
                    best_gap_duration = gap_duration
                    best_split = gap_start + (gap_duration / 2)

        split_point = best_split if best_split is not None else potential_end
        split_point = max(current_start, min(split_point, potential_end, total_duration))
        chunks.append(AudioChunk(len(chunks), current_start, split_point))
        current_start = split_point

    return chunks


def _max_segment_end(start: int, end: int, max_duration_ms: int | None) -> int:
    if max_duration_ms is None or max_duration_ms <= 0:
        return end
    return min(end, start + max_duration_ms)


def enforce_segment_timeline(segments: list[Segment], max_duration_ms: int | None = None) -> list[Segment]:
    normalized: list[Segment] = []

    for segment in sorted((s for s in segments if s.text.strip()), key=lambda s: (s.start, s.end)):
        start = max(segment.start, normalized[-1].end if normalized else segment.start)
        end = _max_segment_end(start, segment.end, max_duration_ms)
        if end <= start:
            continue
        normalized.append(Segment(start=start, end=end, text=segment.text))

    return normalized


def merge_segments(segments: list[Segment], options: SegmentMergeOptions | None = None) -> list[Segment]:
    if options is None:
        options = SegmentMergeOptions()

    segments = [s for s in segments if s.text.strip()]
    segments.sort(key=lambda s: (s.start, s.end))
    if not options.enabled:
        return segments

    merged: list[Segment] = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        gap_ms = seg.start - last.end
        if gap_ms > options.max_gap_ms:
            merged.append(seg)
            continue

        merged_duration_ms = seg.end - last.start
        if merged_duration_ms > options.max_duration_ms:
            merged.append(seg)
            continue

        last_norm = _normalize_merge_text(last.text)
        seg_norm = _normalize_merge_text(seg.text)

        if seg_norm.startswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        if last_norm.startswith(seg_norm) or last_norm.endswith(seg_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=last.text)
            continue

        if seg_norm.endswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        merged.append(seg)

    return merged


class SubWriter:
    @classmethod
    def txt(cls, segments: list[Segment], path: str):
        lines = []
        for _idx, segment in enumerate(segments):
            lines.append(f"{segment.text}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def lrc(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            start_ts = cls.lrc_timestamp(segment.start)
            end_es = cls.lrc_timestamp(segment.end)
            lines.append(f"[{start_ts}]{segment.text}\n")
            if idx != len(segments) - 1:
                next_start = segments[idx + 1].start
                if next_start is not None and end_es == cls.lrc_timestamp(next_start):
                    continue
            lines.append(f"[{end_es}]\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def lrc_timestamp(ms: int) -> str:
        m = ms // 60_000
        ms = ms - m * 60_000
        s = ms // 1_000
        ms = ms - s * 1_000
        ms = ms // 10
        return f"{m:02d}:{s:02d}.{ms:02d}"

    @classmethod
    def vtt(cls, segments: list[Segment], path: str):
        lines = ["WebVTT\n\n"]
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.vtt_timestamp(segment.start)} --> {cls.vtt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def vtt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ".")

    @classmethod
    def srt(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.srt_timestamp(segment.start)} --> {cls.srt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def srt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ",")

    @classmethod
    def _timestamp(cls, ms: int, delim: str):
        h = ms // 3600_000
        ms -= h * 3600_000
        m = ms // 60_000
        ms -= m * 60_000
        s = ms // 1_000
        ms -= s * 1_000
        return f"{h:02d}:{m:02d}:{s:02d}{delim}{ms:03d}"


@dataclass
class InferenceTask:
    audio_path: str
    sub_prefix: str
    sub_formats: list[str]


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(log_handler)


class Inference:
    sub_writers = {
        "lrc": SubWriter.lrc,
        "srt": SubWriter.srt,
        "vtt": SubWriter.vtt,
        "txt": SubWriter.txt,
    }

    def __init__(self, args):
        self.args = args
        model_path = getattr(args, "model_name_or_path", None)
        self.model_name_or_path = str(Path(model_path).expanduser().resolve()) if model_path else None
        self.requested_backend = (getattr(args, "backend", "auto") or "auto").strip().lower()
        self.model_variant = getattr(args, "model_variant", None)
        self.backend_selection: BackendSelection | None = None
        self.backend_name: str | None = None
        self.profile: ProfileDefinition | None = None
        self.vad_injected = False
        self.vad_manager = None
        self._runtime_ready = False
        self._warned_backend_options: set[str] = set()
        self.requested_device = (args.device or "auto").strip().lower()
        self.device = "gpu" if self.requested_backend == "mlx" else resolve_device(self.requested_device)
        self.cpu_threads = args.cpu_threads
        self.vad_threads = args.vad_threads
        if self.cpu_threads < 0 or self.vad_threads < 0:
            raise ValueError("cpu_threads and vad_threads must be non-negative")
        # Auto-select compute type if 'auto' or 'default' is specified
        if self.requested_backend == "mlx":
            self.compute_type = "float16"
        elif args.compute_type in ["auto", "default"]:
            self.compute_type = select_best_compute_type(self.device)
        else:
            self.compute_type = args.compute_type

        # Batch inference settings
        self.enable_batching = args.enable_batching
        self.batch_size = args.batch_size if args.batch_size else 0
        self.max_batch_size = args.max_batch_size

        self.overwrite = args.overwrite
        self.output_dir = args.output_dir
        if self.output_dir:
            if not os.path.isabs(self.output_dir):
                self.output_dir = os.path.join(os.getcwd(), self.output_dir)
            logger.info(_("info.output_dir", output_dir=self.output_dir))
        self.audio_suffixes = {k: True for k in args.audio_suffixes.split(",")}
        self.sub_formats = []
        for k in args.sub_formats.split(","):
            if k not in self.sub_writers:
                raise ValueError(_("warnings.unknown_format", format=k))
            self.sub_formats.append(k)

        # Load generation config
        self.generation_config, self.segment_merge_options, self.smart_split_options = self._load_generation_config(
            args
        )
        self.profile = get_profile(self.generation_config["task"])

        logger.info(_("info.generation_config", config=self.generation_config))
        logger.info(
            "Segment merge: enabled=%s, max_gap_ms=%s, max_duration_ms=%s",
            self.segment_merge_options.enabled,
            self.segment_merge_options.max_gap_ms,
            self.segment_merge_options.max_duration_ms,
        )
        logger.info(
            "Smart VAD split: enabled=%s, target_chunk_duration_s=%s",
            self.smart_split_options.enabled,
            self.smart_split_options.target_chunk_duration_s,
        )

    def _ensure_runtime_ready(self) -> None:
        if self._runtime_ready:
            return
        if self.profile is None:
            raise RuntimeError("Inference profile was not initialized")
        started = time.perf_counter()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.backend_selection = select_backend(
            self.requested_backend,
            self.profile,
            variant=self.model_variant,
            model_path=self.model_name_or_path,
        )
        self.backend_name = self.backend_selection.selected
        self.model_name_or_path = str(self.backend_selection.descriptor.path)
        if self.backend_name == "ct2":
            self.device = resolve_device(self.requested_device)
            if self.args.compute_type in ["auto", "default"]:
                self.compute_type = select_best_compute_type(self.device)
            else:
                self.compute_type = self.args.compute_type
        else:
            self.device = "gpu"
            self.compute_type = "float16"
        require_local_runtime_assets(
            Path(self.model_name_or_path),
            backend=self.backend_name,
            validate_model=False,
        )
        self._setup_vad_injection(self.args)
        self._runtime_ready = True
        logger.info(_("info.stage_timing", stage="runtime_preflight", seconds=time.perf_counter() - started))

    def _load_generation_config(self, args) -> tuple[dict[str, Any], SegmentMergeOptions, SmartSplitOptions]:
        """Load and process generation configuration"""
        # Default config
        config: dict[str, Any] = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "beam_size": 1,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }

        segment_merge_options = SegmentMergeOptions()
        smart_split_options = SmartSplitOptions()

        # Load from file if exists
        if os.path.exists(args.generation_config):
            with open(args.generation_config, encoding="utf-8") as f:
                file_config = pyjson5.decode_io(f)
                file_segment_merge = file_config.pop("segment_merge", None)
                if isinstance(file_segment_merge, dict):
                    segment_merge_options = SegmentMergeOptions(
                        enabled=bool(file_segment_merge.get("enabled", segment_merge_options.enabled)),
                        max_gap_ms=int(file_segment_merge.get("max_gap_ms", segment_merge_options.max_gap_ms)),
                        max_duration_ms=int(
                            file_segment_merge.get("max_duration_ms", segment_merge_options.max_duration_ms)
                        ),
                    )
                file_smart_split = file_config.pop("smart_split_with_vad", None)
                file_target_chunk_duration = file_config.pop("target_chunk_duration_s", None)
                smart_split_options = SmartSplitOptions(
                    enabled=_coerce_bool(file_smart_split, default=smart_split_options.enabled),
                    target_chunk_duration_s=(
                        float(file_target_chunk_duration)
                        if file_target_chunk_duration is not None
                        else smart_split_options.target_chunk_duration_s
                    ),
                    split_window_factor=smart_split_options.split_window_factor,
                )
                config = dict(**ChainMap(file_config, config))

        config["task"] = _normalize_whisper_task(args.task if args.task is not None else config.get("task"))

        # Process VAD parameters from config file
        if "vad_parameters" in config:
            vad_params = config.pop("vad_parameters")

            # Convert to VadOptions format
            vad_options = {}

            # Map common parameters
            if "threshold" in vad_params:
                vad_options["threshold"] = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_options["neg_threshold"] = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_options["min_speech_duration_ms"] = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_options["max_speech_duration_s"] = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_options["min_silence_duration_ms"] = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_options["speech_pad_ms"] = vad_params["speech_pad_ms"]

            config["vad_parameters"] = vad_options

        # Override with command line arguments
        if args.vad_threshold is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["threshold"] = args.vad_threshold

        if args.vad_min_speech_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_speech_duration_ms"] = args.vad_min_speech_duration_ms

        if args.vad_min_silence_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_silence_duration_ms"] = args.vad_min_silence_duration_ms

        if args.vad_speech_pad_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["speech_pad_ms"] = args.vad_speech_pad_ms

        # Override segment merge options with command line arguments
        segment_merge_options = SegmentMergeOptions(
            enabled=args.merge_segments if args.merge_segments is not None else segment_merge_options.enabled,
            max_gap_ms=args.merge_max_gap_ms if args.merge_max_gap_ms is not None else segment_merge_options.max_gap_ms,
            max_duration_ms=(
                args.merge_max_duration_ms
                if args.merge_max_duration_ms is not None
                else segment_merge_options.max_duration_ms
            ),
        )

        smart_split_options = SmartSplitOptions(
            enabled=(
                _coerce_bool(args.smart_split_with_vad, default=smart_split_options.enabled)
                if args.smart_split_with_vad is not None
                else smart_split_options.enabled
            ),
            target_chunk_duration_s=(
                args.target_chunk_duration_s
                if args.target_chunk_duration_s is not None
                else smart_split_options.target_chunk_duration_s
            ),
            split_window_factor=smart_split_options.split_window_factor,
        )
        if smart_split_options.target_chunk_duration_s <= 0:
            raise ValueError("target_chunk_duration_s must be greater than 0")
        if smart_split_options.target_chunk_duration_s > MAX_SMART_CHUNK_DURATION_S:
            smart_split_options = SmartSplitOptions(
                enabled=smart_split_options.enabled,
                target_chunk_duration_s=MAX_SMART_CHUNK_DURATION_S,
                split_window_factor=smart_split_options.split_window_factor,
            )

        return config, segment_merge_options, smart_split_options

    def _vad_progress_callback(self, chunk_idx, total_chunks, device):
        """Progress callback for VAD processing."""
        progress_pct = (chunk_idx / total_chunks) * 100
        # Use carriage return to update the same line
        print(
            "\r  "
            + _(
                "progress.vad",
                current=chunk_idx,
                total=total_chunks,
                percent=progress_pct,
                device=device,
            ),
            end="",
            flush=True,
        )
        if chunk_idx == total_chunks:
            print()  # New line when done

    def _setup_vad_injection(self, args):
        """Setup whisper_vad injection - always enforced"""
        # Always use whisper_vad model
        vad_model = "whisper_vad"

        logger.info(_("info.initializing_vad"))

        # Create VAD config with progress callback
        vad_config = VadConfig(default_model=vad_model)

        # Apply VAD parameters from generation config
        if "vad_parameters" in self.generation_config:
            vad_params = self.generation_config["vad_parameters"]
            if "threshold" in vad_params:
                vad_config.threshold = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_config.neg_threshold = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_config.min_speech_duration_ms = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_config.max_speech_duration_s = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_config.min_silence_duration_ms = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_config.speech_pad_ms = vad_params["speech_pad_ms"]

        # Load ONNX VAD configuration from metadata
        vad_metadata_path = DEFAULT_MODELS_ROOT / "whisper_vad_metadata.json"
        vad_config.onnx_model_path = str((DEFAULT_MODELS_ROOT / "whisper_vad.onnx").resolve())
        vad_config.onnx_metadata_path = str(vad_metadata_path.resolve())

        # Read model configuration from metadata JSON if it exists
        if vad_metadata_path.exists():
            try:
                with vad_metadata_path.open(encoding="utf-8") as f:
                    metadata = json.load(f)

                # Load model configuration from metadata
                vad_config.whisper_model_name = metadata.get("whisper_model_name", "openai/whisper-base")
                vad_config.frame_duration_ms = metadata.get("frame_duration_ms", 20)
                vad_config.chunk_duration_ms = metadata.get("total_duration_ms", 30000)

                logger.info(_("warnings.loaded_vad_config", path=vad_metadata_path))
            except Exception as e:
                logger.warning(_("warnings.failed_load_vad", path=vad_metadata_path, error=e))
                logger.warning(_("warnings.using_default_vad"))
                # Fallback to defaults
                vad_config.whisper_model_name = "openai/whisper-base"
                vad_config.frame_duration_ms = 20
                vad_config.chunk_duration_ms = 30000
        else:
            # Use defaults if metadata file doesn't exist
            logger.warning(_("warnings.vad_file_not_found", path=vad_metadata_path))
            logger.warning(_("warnings.using_default_vad"))
            vad_config.whisper_model_name = "openai/whisper-base"
            vad_config.frame_duration_ms = 20
            vad_config.chunk_duration_ms = 30000

        # VAD remains CPU-only for every Whisper backend.
        vad_config.force_cpu = True
        vad_config.num_threads = self.vad_threads

        # The manager is shared by both backends. Faster-Whisper additionally
        # receives the injection so its optional inner VAD keeps existing CT2
        # behavior; MLX consumes only the outer VAD/chunk plan.
        self.vad_manager = VadModelManager(
            config=vad_config,
            ttl=vad_config.ttl,
            progress_callback=self._vad_progress_callback,
        )
        if self.backend_name == "ct2":
            inject_vad(
                model_id=vad_model,
                config=vad_config,
                progress_callback=self._vad_progress_callback,
            )
            self.vad_injected = True
        logger.info(_("info.vad_activated", threshold=vad_config.threshold))

    def _prepare_transcription(self, audio_path: str, *, batched: bool) -> tuple[Any, dict[str, Any], float | None]:
        config = dict(self.generation_config)

        if self.smart_split_options.enabled or not config.get("vad_filter") or "clip_timestamps" in config:
            return audio_path, config, None

        if self.vad_manager is None:
            return audio_path, config, None

        decode_started = time.perf_counter()
        audio = _require_audio_decoder()(audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        logger.info(_("info.stage_timing", stage="audio_decode", seconds=time.perf_counter() - decode_started))
        vad_parameters = config.get("vad_parameters") or {}
        vad_started = time.perf_counter()
        vad_segments = self.vad_manager.get_speech_timestamps(
            model_id="whisper_vad",
            audio=audio,
            sampling_rate=WHISPER_SAMPLING_RATE,
            **vad_parameters,
        )
        logger.info(_("info.stage_timing", stage="outer_vad", seconds=time.perf_counter() - vad_started))
        duration_after_vad = sum(segment["end"] - segment["start"] for segment in vad_segments) / WHISPER_SAMPLING_RATE

        config["vad_filter"] = False
        config["clip_timestamps"] = vad_segments_to_clip_timestamps(
            vad_segments,
            WHISPER_SAMPLING_RATE,
            batched=batched,
        )
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)

        return audio, config, duration_after_vad

    def _plan_smart_chunks(
        self,
        audio_path: str,
    ) -> tuple[Any, list[AudioChunk], float | None, list[SpeechSpan]]:
        decode_started = time.perf_counter()
        audio = _require_audio_decoder()(audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        logger.info(_("info.stage_timing", stage="audio_decode", seconds=time.perf_counter() - decode_started))
        duration = len(audio) / WHISPER_SAMPLING_RATE

        if not self.smart_split_options.enabled or not self.generation_config.get("vad_filter"):
            return audio, [AudioChunk(0, 0.0, duration)], None, []

        if self.vad_manager is None:
            return audio, [AudioChunk(0, 0.0, duration)], None, []

        vad_parameters = self.generation_config.get("vad_parameters") or {}
        vad_started = time.perf_counter()
        vad_segments = self.vad_manager.get_speech_timestamps(
            model_id="whisper_vad",
            audio=audio,
            sampling_rate=WHISPER_SAMPLING_RATE,
            **vad_parameters,
        )
        logger.info(_("info.stage_timing", stage="outer_vad", seconds=time.perf_counter() - vad_started))
        duration_after_vad = sum(segment["end"] - segment["start"] for segment in vad_segments) / WHISPER_SAMPLING_RATE

        split_started = time.perf_counter()
        spans = vad_segments_to_speech_spans(vad_segments, WHISPER_SAMPLING_RATE)
        chunks = create_contiguous_chunks(
            spans,
            min(self.smart_split_options.target_chunk_duration_s, MAX_SMART_CHUNK_DURATION_S),
            duration,
            self.smart_split_options.split_window_factor,
        )
        if not chunks:
            chunks = [AudioChunk(0, 0.0, duration)]
        logger.info(_("info.stage_timing", stage="chunk_planning", seconds=time.perf_counter() - split_started))
        logger.info("Smart VAD split planned %s chunk(s)", len(chunks))

        return audio, chunks, duration_after_vad, spans

    def _backend_request(self, audio: Any, config: dict[str, Any]) -> BackendRequest:
        profile = getattr(self, "profile", None)
        return BackendRequest(
            audio=audio,
            language=str(config.get("language", profile.language if profile else "ja")),
            task=str(config.get("task", profile.task if profile else "translate")),
            options=config,
        )

    @staticmethod
    def _convert_backend_segments(
        backend_segments: list[BackendSegment],
        *,
        offset_ms: int = 0,
        limit_ms: int | None = None,
    ) -> list[Segment]:
        segments: list[Segment] = []
        for backend_segment in backend_segments:
            segment = Segment(
                start=offset_ms + int(round(backend_segment.start * 1_000)),
                end=offset_ms + int(round(backend_segment.end * 1_000)),
                text=backend_segment.text.strip(),
            )
            if limit_ms is not None:
                if segment.start >= limit_ms:
                    continue
                segment = Segment(segment.start, min(segment.end, limit_ms), segment.text)
            if segment.text and segment.end > segment.start:
                segments.append(segment)
        return segments

    def _warn_ignored_backend_options(self, backend: WhisperBackend) -> None:
        warned: set[str] = set(getattr(self, "_warned_backend_options", set()))
        ignored = set(getattr(backend, "ignored_options", ())).difference(warned)
        if not ignored:
            return
        warned.update(ignored)
        self._warned_backend_options = warned
        logger.warning(
            _(
                "warnings.backend_options_ignored",
                backend=getattr(self, "backend_name", None),
                options=", ".join(sorted(ignored)),
            )
        )

    def _transcribe_smart_chunks(
        self,
        backend: WhisperBackend,
        task: InferenceTask,
    ) -> tuple[list[Segment], BackendResult]:
        audio, chunks, outer_duration_after_vad, speech_spans = self._plan_smart_chunks(task.audio_path)
        duration = len(audio) / WHISPER_SAMPLING_RATE
        config = dict(self.generation_config)
        config.pop("clip_timestamps", None)
        config["vad_filter"] = bool(config.get("vad_filter", True))
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)

        if outer_duration_after_vad == 0:
            logger.info(_("info.no_speech_detected", path=task.audio_path))
            return [], BackendResult(
                segments=[],
                duration=duration,
                duration_after_vad=0,
                language=config.get("language"),
                backend=getattr(self, "backend_name", None) or "unknown",
                metrics={"chunk_count": 0.0},
            )

        segments: list[Segment] = []
        normalized_backend_segments: list[BackendSegment] = []
        inner_duration_after_vad = 0.0
        has_inner_duration = False
        metrics: dict[str, float] = {"chunk_count": 0.0}

        for chunk in chunks:
            start_sample = max(0, min(len(audio), int(round(chunk.start * WHISPER_SAMPLING_RATE))))
            end_sample = max(start_sample, min(len(audio), int(round(chunk.end * WHISPER_SAMPLING_RATE))))
            if end_sample <= start_sample:
                continue
            chunk_audio = audio[start_sample:end_sample]
            chunk_config = dict(config)
            if getattr(self, "backend_name", None) == "mlx" and speech_spans:
                clip_timestamps: list[float] = []
                for span in speech_spans:
                    overlap_start = max(chunk.start, span.start)
                    overlap_end = min(chunk.end, span.end)
                    if overlap_end > overlap_start:
                        clip_timestamps.extend(
                            [
                                max(0.0, overlap_start - chunk.start),
                                max(0.0, overlap_end - chunk.start),
                            ]
                        )
                if not clip_timestamps:
                    logger.debug("Skipping MLX chunk %s with no outer-VAD speech", chunk.index + 1)
                    continue
                chunk_config["clip_timestamps"] = clip_timestamps
            logger.debug(
                "Smart VAD chunk %s/%s: %s --> %s",
                chunk.index + 1,
                len(chunks),
                SubWriter.srt_timestamp(int(round(chunk.start * 1_000))),
                SubWriter.srt_timestamp(int(round(chunk.end * 1_000))),
            )
            metrics["chunk_count"] += 1.0
            chunk_result = backend.transcribe(self._backend_request(chunk_audio, chunk_config))
            self._warn_ignored_backend_options(backend)
            if chunk_result.duration_after_vad is not None:
                inner_duration_after_vad += chunk_result.duration_after_vad
                has_inner_duration = True
            for name, value in chunk_result.metrics.items():
                if name in {"batch_size", "ignored_option_count", "peak_memory_bytes"}:
                    metrics[name] = max(metrics.get(name, 0.0), value)
                else:
                    metrics[name] = metrics.get(name, 0.0) + value
            chunk_offset_ms = int(round(chunk.start * 1_000))
            chunk_end_ms = int(round(chunk.end * 1_000))
            converted = self._convert_backend_segments(
                chunk_result.segments,
                offset_ms=chunk_offset_ms,
                limit_ms=chunk_end_ms,
            )
            segments.extend(converted)
            for segment in converted:
                normalized_backend_segments.append(
                    BackendSegment(
                        start=segment.start / 1_000,
                        end=segment.end / 1_000,
                        text=segment.text,
                    )
                )
                logger.debug(
                    f"[{SubWriter.lrc_timestamp(segment.start)} --> "
                    f"{SubWriter.lrc_timestamp(segment.end)}] {segment.text}"
                )

        duration_after_vad = inner_duration_after_vad if has_inner_duration else outer_duration_after_vad
        return segments, BackendResult(
            segments=normalized_backend_segments,
            duration=duration,
            duration_after_vad=duration_after_vad,
            language=config.get("language"),
            backend=getattr(self, "backend_name", None) or "unknown",
            metrics=metrics,
        )

    def _log_duration(self, duration: float, duration_after_vad: float) -> None:
        if duration <= 0 or duration == duration_after_vad or duration_after_vad == 0:
            logger.info(_("info.duration", duration=format_duration(duration)))
            return

        rate = duration_after_vad / duration
        logger.info(
            _(
                "info.duration_filtered",
                original=format_duration(duration),
                filtered=format_duration(duration_after_vad),
                percent=format_percentage(rate),
            )
        )

    def _should_use_smart_split(self) -> bool:
        return bool(self.smart_split_options.enabled and self.generation_config.get("vad_filter", True))

    def generates(self, base_dirs):
        if len(base_dirs) == 0:
            logger.warning(_("warnings.provide_directories"))
            return EXIT_NO_INPUT

        scan_started = time.perf_counter()
        tasks = self._scan(base_dirs)
        logger.info(_("info.stage_timing", stage="file_scan", seconds=time.perf_counter() - scan_started))
        if len(tasks) == 0:
            logger.info(_("info.no_files_found"))
            return EXIT_OK if self.last_scan_supported_count > 0 else EXIT_NO_INPUT

        self._ensure_runtime_ready()

        logger.info(_("tasks.processing", count=len(tasks), task=self.generation_config["task"]))
        logger.info(_("info.loading_whisper"))

        backend: WhisperBackend | None = None
        try:
            if self.backend_selection is None:
                raise RuntimeError("Backend selection was not initialized")
            backend = create_backend(
                self.backend_selection,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                enable_batching=self.enable_batching,
                batch_size=self.batch_size,
                max_batch_size=self.max_batch_size,
            )
            if self.backend_selection.fallback_reason:
                logger.warning(
                    _(
                        "warnings.backend_auto_fallback",
                        backend=self.backend_selection.selected,
                        reason=self.backend_selection.fallback_reason,
                    )
                )
            logger.info(
                _(
                    "info.backend_selected",
                    backend=self.backend_selection.selected,
                    profile=self.profile.name if self.profile else self.generation_config["task"],
                    model=self.backend_selection.descriptor.path,
                )
            )
            logger.info(_("info.model_precision").format(precision=self.compute_type, device=self.device))

            for i, task in enumerate(tasks):
                file_started = time.perf_counter()
                logger.info(
                    _(
                        "info.processing",
                        task=self.generation_config["task"],
                        current=i + 1,
                        total=len(tasks),
                        path=task.audio_path,
                    )
                )

                if self._should_use_smart_split():
                    segments, result = self._transcribe_smart_chunks(backend, task)
                else:
                    audio_input, transcription_config, manual_duration_after_vad = self._prepare_transcription(
                        task.audio_path,
                        batched=bool(getattr(backend, "batching_enabled", False)),
                    )
                    if manual_duration_after_vad == 0:
                        logger.info(_("info.no_speech_detected", path=task.audio_path))
                        duration = len(audio_input) / WHISPER_SAMPLING_RATE if not isinstance(audio_input, str) else 0.0
                        result = BackendResult(
                            segments=[],
                            duration=duration,
                            duration_after_vad=0,
                            language=transcription_config.get("language"),
                            backend=self.backend_name or "unknown",
                        )
                    else:
                        result = backend.transcribe(self._backend_request(audio_input, transcription_config))
                        self._warn_ignored_backend_options(backend)
                    if manual_duration_after_vad is not None:
                        result = BackendResult(
                            segments=result.segments,
                            duration=result.duration,
                            duration_after_vad=manual_duration_after_vad,
                            language=result.language,
                            backend=result.backend,
                            metrics=result.metrics,
                        )
                    segments = self._convert_backend_segments(result.segments)
                    for segment in segments:
                        logger.debug(
                            f"[{SubWriter.lrc_timestamp(segment.start)} --> "
                            f"{SubWriter.lrc_timestamp(segment.end)}] {segment.text}"
                        )

                duration_after_vad = (
                    result.duration_after_vad if result.duration_after_vad is not None else result.duration
                )
                self._log_duration(result.duration, duration_after_vad)
                if result.metrics:
                    logger.info(_("info.backend_metrics", metrics=result.metrics))

                postprocess_started = time.perf_counter()
                segments = enforce_segment_timeline(segments, self.segment_merge_options.max_duration_ms)
                segments = merge_segments(segments, self.segment_merge_options)
                segments = enforce_segment_timeline(segments, self.segment_merge_options.max_duration_ms)
                postprocess_seconds = time.perf_counter() - postprocess_started
                os.makedirs(os.path.dirname(task.sub_prefix), exist_ok=True)
                write_started = time.perf_counter()
                for sub_suffix in task.sub_formats:
                    sub_path = f"{task.sub_prefix}.{sub_suffix}"
                    logger.info(_("info.writing", path=sub_path))
                    self.sub_writers[sub_suffix](segments, sub_path)
                write_seconds = time.perf_counter() - write_started
                logger.info(
                    _(
                        "info.file_timing",
                        total=time.perf_counter() - file_started,
                        postprocess=postprocess_seconds,
                        write=write_seconds,
                    )
                )

        finally:
            if backend is not None:
                backend.close()
            # Clean up VAD injection
            if self.vad_injected:
                uninject_vad()
                logger.info(_("info.vad_deactivated"))
        return EXIT_OK

    def _scan(self, base_dirs) -> list[InferenceTask]:
        tasks: list[InferenceTask] = []
        self.last_scan_supported_count = 0

        def process(base_path, audio_path):
            nonlocal tasks
            p = Path(audio_path)
            suffix = p.suffix.lower().lstrip(".")

            logger.debug(_("debug.processing", path=audio_path))
            logger.debug(_("debug.file_suffix", suffix=suffix))
            logger.debug(_("debug.valid_suffixes", suffixes=self.audio_suffixes))

            if suffix not in self.audio_suffixes:
                logger.debug(_("debug.skipped_suffix", suffix=suffix))
                return
            self.last_scan_supported_count += 1

            rel_path = p.relative_to(base_path)
            abs_path = Path(os.path.join(self.output_dir or base_path, rel_path))
            sub_formats = []

            for suffix in self.sub_formats:
                sub_path = abs_path.parent / f"{abs_path.stem}.{suffix}"
                if sub_path.exists() and not self.overwrite:
                    logger.debug(_("debug.subtitle_exists", path=sub_path))
                    continue
                sub_formats.append(suffix)

            if len(sub_formats) == 0:
                logger.debug(_("debug.skipped_all_exist"))
                return

            logger.debug(_("debug.added_task", formats=sub_formats))
            tasks.append(InferenceTask(audio_path, str(abs_path.parent / abs_path.stem), sub_formats))

        for base_dir in base_dirs:
            # Expand user home directory
            base_dir = os.path.expanduser(base_dir)
            logger.debug(_("debug.scanning", path=base_dir))

            parent_dir = os.path.dirname(base_dir)
            if os.path.isdir(base_dir):
                for root, _dirs, files in os.walk(base_dir, topdown=True):
                    for file in files:
                        process(parent_dir, os.path.join(root, file))
            else:
                process(parent_dir, base_dir)

        logger.info(_("files.found", count=len(tasks)))
        return tasks


def diagnose_environment():
    """Run comprehensive environment diagnostics for debugging"""
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)

    # System info
    print("\n1. System Information:")
    print(f"   Platform: {platform.system()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")
    print(f"   Executable: {sys.executable}")

    # CUDA environment
    print("\n2. CUDA Environment Variables:")
    cuda_vars = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDNN_HOME",
        "LD_LIBRARY_PATH",
        "PATH",
    ]
    for var in cuda_vars:
        value = os.environ.get(var, "Not set")
        if var == "PATH" and value != "Not set":
            # Just show cuda-related paths
            cuda_paths = [p for p in value.split(os.pathsep) if "cuda" in p.lower() or "nvidia" in p.lower()]
            value = os.pathsep.join(cuda_paths) if cuda_paths else "No CUDA paths in PATH"
        print(f"   {var}: {value}")

    # Check for nvidia-smi
    print("\n3. NVIDIA GPU Detection:")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"   GPU Info: {result.stdout.strip()}")
        else:
            print("   nvidia-smi failed")
    except FileNotFoundError:
        print("   nvidia-smi not found in PATH")
    except Exception as e:
        print(f"   Error: {e}")


def check_onnxruntime_detailed():
    """Detailed ONNX Runtime check for debugging"""
    print("\n" + "=" * 60)
    print("ONNX RUNTIME DIAGNOSTICS")
    print("=" * 60)

    try:
        import onnxruntime as ort

        print("\n✓ onnxruntime imported successfully")
        print(f"  Version: {ort.__version__}")
        print(f"  Location: {ort.__file__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"\n  Available providers: {providers}")

        # Check for GPU support
        has_cuda = "CUDAExecutionProvider" in providers
        has_tensorrt = "TensorrtExecutionProvider" in providers
        has_directml = "DmlExecutionProvider" in providers

        print("\n  GPU Support:")
        print(f"    CUDA: {'✓ Available' if has_cuda else '✗ Not Available'}")
        print(f"    TensorRT: {'✓ Available' if has_tensorrt else '✗ Not Available'}")
        print(f"    DirectML: {'✓ Available' if has_directml else '✗ Not Available'}")

        if not has_cuda and sys.platform != "darwin":
            print("\n  ⚠️ CUDA not available. This might be because:")
            print("    1. onnxruntime (CPU) is installed instead of onnxruntime-gpu")
            print("    2. CUDA libraries are missing or not in PATH")
            print("    3. Incompatible CUDA/cuDNN versions")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import onnxruntime: {e}")
        print("\nSuggestions:")
        print("  1. Install onnxruntime-gpu for GPU support")
        print("  2. Check that the active environment contains compatible runtime libraries")
        return False
    except Exception as e:
        print(f"\n✗ Error during ONNX Runtime check: {e}")
        traceback.print_exc()
        return False


def test_vad_initialization():
    """Test VAD model initialization for debugging"""
    print("\n" + "=" * 60)
    print("VAD MODEL TEST")
    print("=" * 60)

    try:
        from .vad_manager import VadModelManager, WhisperVADOnnxWrapper  # noqa: F401

        print("✓ VAD modules imported successfully")

        # Check for model files
        model_paths = [
            "models/whisper_vad.onnx",
            "models/vad/whisper_vad.onnx",
        ]

        model_path = None
        print("\nSearching for VAD model:")
        for path in model_paths:
            exists = os.path.exists(path)
            print(f"  {path}: {'Found' if exists else 'Not found'}")
            if exists and model_path is None:
                model_path = path

        if model_path:
            print(f"\n✓ Using model: {model_path}")

            # Try to initialize
            print("\nTesting VAD initialization (GPU if available):")
            try:
                wrapper = WhisperVADOnnxWrapper(model_path=model_path, force_cpu=False, num_threads=1)
                print(f"  ✓ Device: {wrapper.device}")
                print(f"  ✓ Providers: {wrapper.session.get_providers()}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Test with forced CPU for comparison
            print("\nTesting VAD initialization (Force CPU):")
            try:
                wrapper_cpu = WhisperVADOnnxWrapper(model_path=model_path, force_cpu=True, num_threads=1)
                print(f"  ✓ Device: {wrapper_cpu.device}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print("\n✗ No VAD model file found")
            print("  Download the model using download_models.py")

    except ImportError as e:
        print(f"✗ Failed to import VAD modules: {e}")
    except Exception as e:
        print(f"✗ Error during VAD test: {e}")
        traceback.print_exc()


def launch_debug_console():
    """Launch interactive Python console for debugging"""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEBUG CONSOLE")
    print("=" * 60)
    print("\nYou now have access to an interactive Python console.")
    print("\nAvailable commands:")
    print("  diagnose()       - Run environment diagnostics")
    print("  check_onnx()     - Check ONNX Runtime status")
    print("  test_vad()       - Test VAD initialization")
    print("  import X         - Try importing any module")
    print("  exit() or Ctrl+D - Exit console and continue")
    print("\nUseful variables:")
    print("  sys.path         - Python module search paths")
    print("  os.environ       - Environment variables")
    print("=" * 60 + "\n")

    # Create namespace with useful functions
    namespace = {
        "diagnose": diagnose_environment,
        "check_onnx": check_onnxruntime_detailed,
        "test_vad": test_vad_initialization,
        "sys": sys,
        "os": os,
        "platform": platform,
    }

    # Launch interactive console
    code.InteractiveConsole(locals=namespace).interact(banner="")


def main():
    """Main entry point for the script"""
    args = parse_arguments()

    # Keep redirected logs and terminal output consistently UTF-8 encoded.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    # Display open-source notice
    print("=" * 70)
    print("⚠️  重要声明 / IMPORTANT NOTICE")
    print("=" * 70)
    print("本软件开源于: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice")
    print("开发团队: AI汉化组 (https://t.me/transWithAI)")
    print("任何第三方非免费下载均为智商税")
    print("=" * 70)
    print()

    # Check if console mode requested
    if args.console:
        # Run diagnostics first
        diagnose_environment()
        check_onnxruntime_detailed()
        test_vad_initialization()

        # Launch interactive console
        launch_debug_console()

        # After console exits, ask if user wants to continue with normal operation
        print("\nDebug console exited.")
        try:
            response = input("Continue with normal inference? (y/n): ").strip().lower()
            if response != "y":
                print("Exiting...")
                return EXIT_OK
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return EXIT_OK

    # Normal operation
    # Set logger to DEBUG so file handler captures everything;
    # console handler stays at the user-requested level.
    logger.setLevel(logging.DEBUG)
    log_handler.setLevel(args.log_level)

    # Add file logging to latest.log in current working directory
    # This helps users report issues by providing a log file
    log_file_path = os.path.join(os.getcwd(), "latest.log")
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    # Add file handler to the module logger
    logger.addHandler(file_handler)

    logger.info(_("info.logging_to_file").format(path=log_file_path))
    logger.info(_("info.program_version").format(version="v1.10"))
    logger.info(_("info.python_version").format(version=sys.version))
    logger.info(_("info.platform").format(platform=platform.platform()))
    logger.info(_("info.arguments").format(args=vars(args)))

    if len(args.base_dirs) == 0:
        logger.warning(_("warnings.drag_files"))
        return EXIT_NO_INPUT

    inference = Inference(args)
    return inference.generates(args.base_dirs)


if __name__ == "__main__":
    # When run directly as a script
    import os

    os.chdir(os.path.dirname(__file__))
    raise SystemExit(main())
