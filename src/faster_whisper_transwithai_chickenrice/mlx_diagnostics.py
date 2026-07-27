"""Pure-Python diagnostics for MLX Whisper runtime results."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backends.base import BackendResult

ADJACENT_SEGMENT_SIMILARITY = 0.88
ADJACENT_SOFT_MAX_NORMALIZED_LENGTH = 4
ADJACENT_SOFT_MIN_GAP_SECONDS = 1.0
REPEATED_SEGMENT_WINDOW_SECONDS = 45.0
MLX_DIAGNOSTIC_SCHEMA_VERSION = 2
WHISPER_SAMPLE_RATE = 16_000
MLX_DEBUG_MAX_WINDOWS = 8
MLX_DEBUG_MAX_ATTEMPTS = 6
MLX_DEBUG_MAX_BYTES = 64 * 1024
MLX_LOOP_FINDING_LIMIT = 8
MLX_EVENT_REQUIRED_FIELDS = (
    "schema_version",
    "event",
    "backend",
    "clip_mode",
    "sample_count",
    "duration",
    "attempt_count",
    "fallback_exhausted",
    "loop_detected",
    "soft_review_required",
    "runtime_call_count",
    "safe_retry_count",
    "unresolved_anomaly",
    "segment_count",
    "inference_seconds",
    "peak_memory_bytes",
)
_FAILURE_REASONS = frozenset({"compression_ratio", "logprob", "no_speech_override"})
_ANOMALY_REASONS = frozenset(
    {
        "both",
        "diagnostics_incomplete",
        "empty_output",
        "fallback_exhausted",
        "high_confidence_loop",
        "invalid_segment",
        "multiple",
        "output_truncated",
        "repeated_frame_range",
        "seek_invariant",
        "timestamp_violation",
    }
)
_LOOP_RULES = frozenset(
    {
        "consecutive_unit",
        "adjacent_segment_similarity",
        "repeated_segment_within_45s",
    }
)


def normalize_loop_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if unicodedata.category(character)[:1] in {"L", "N"})


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repeat_threshold(unit_length: int) -> int:
    if unit_length == 1:
        return 4
    if unit_length <= 3:
        return 3
    return 2


def _is_primitive_unit(unit: str) -> bool:
    for divisor in range(1, len(unit)):
        if len(unit) % divisor == 0 and unit == unit[:divisor] * (len(unit) // divisor):
            return False
    return True


def _within_segment_findings(text: str, segment_index: int) -> list[dict[str, Any]]:
    normalized = normalize_loop_text(text)
    findings: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for unit_length in range(1, len(normalized) // 2 + 1):
        threshold = _repeat_threshold(unit_length)
        last_start = len(normalized) - unit_length * threshold
        for start in range(last_start + 1):
            unit = normalized[start : start + unit_length]
            if start >= unit_length and normalized[start - unit_length : start] == unit:
                continue
            if not _is_primitive_unit(unit):
                continue
            repeat_count = 1
            cursor = start + unit_length
            while normalized[cursor : cursor + unit_length] == unit:
                repeat_count += 1
                cursor += unit_length
            key = (start, unit_length, repeat_count)
            if repeat_count >= threshold and key not in seen:
                seen.add(key)
                findings.append(
                    {
                        "rule": "consecutive_unit",
                        "segment_index": segment_index,
                        "unit_length": unit_length,
                        "repeat_count": repeat_count,
                        "text_sha256": text_sha256(text),
                        "text_length": len(text),
                        "normalized_start": start,
                        "severity": "hard",
                    }
                )
    return findings


def _segment_time(segment: Mapping[str, Any], name: str, default: float) -> float:
    try:
        return float(segment.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LoopDetectionResult:
    findings: tuple[dict[str, Any], ...]
    segment_count: int

    @property
    def loop_detected(self) -> bool:
        return bool(self.hard_findings)

    @property
    def hard_findings(self) -> tuple[dict[str, Any], ...]:
        return tuple(finding for finding in self.findings if finding.get("severity") == "hard")

    @property
    def soft_findings(self) -> tuple[dict[str, Any], ...]:
        return tuple(finding for finding in self.findings if finding.get("severity") == "soft")

    @property
    def review_required(self) -> bool:
        return bool(self.soft_findings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loop_detected": self.loop_detected,
            "soft_review_required": self.review_required,
            "hard_finding_count": len(self.hard_findings),
            "soft_finding_count": len(self.soft_findings),
            "segment_count": self.segment_count,
            "findings": [dict(finding) for finding in self.findings],
        }


def detect_high_confidence_loops(
    raw_segments: Sequence[Mapping[str, Any]],
    *,
    duration: float | None = None,
) -> LoopDetectionResult:
    texts = [str(segment.get("text", "")) for segment in raw_segments]
    normalized_texts = [normalize_loop_text(text) for text in texts]
    findings: list[dict[str, Any]] = []

    for index, text in enumerate(texts):
        findings.extend(_within_segment_findings(text, index))

    for index in range(1, len(raw_segments)):
        left_text = normalized_texts[index - 1]
        right_text = normalized_texts[index]
        if not left_text or not right_text:
            continue
        similarity = difflib.SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
        if left_text == right_text or similarity >= ADJACENT_SEGMENT_SIMILARITY:
            left_start = _segment_time(raw_segments[index - 1], "start", math.nan)
            left_end = _segment_time(raw_segments[index - 1], "end", math.nan)
            right_start = _segment_time(raw_segments[index], "start", math.nan)
            right_end = _segment_time(raw_segments[index], "end", math.nan)
            bounds_valid = (
                all(math.isfinite(value) for value in (left_start, left_end, right_start, right_end))
                and 0.0 <= left_start < left_end
                and 0.0 <= right_start < right_end
                and (duration is None or max(left_end, right_end) <= duration + 1 / WHISPER_SAMPLE_RATE)
            )
            gap_seconds = right_start - left_end if bounds_valid else 0.0
            severity = (
                "soft"
                if (
                    left_text == right_text
                    and max(len(left_text), len(right_text)) <= ADJACENT_SOFT_MAX_NORMALIZED_LENGTH
                    and gap_seconds >= ADJACENT_SOFT_MIN_GAP_SECONDS
                )
                else "hard"
            )
            findings.append(
                {
                    "rule": "adjacent_segment_similarity",
                    "segment_index": index,
                    "unit_length": min(len(left_text), len(right_text)),
                    "repeat_count": 2,
                    "text_sha256": text_sha256(texts[index]),
                    "text_length": len(texts[index]),
                    "related_segment_index": index - 1,
                    "related_text_sha256": text_sha256(texts[index - 1]),
                    "similarity": similarity,
                    "gap_seconds": max(0.0, gap_seconds),
                    "bounds_valid": bounds_valid,
                    "severity": severity,
                }
            )

    indices_by_text: dict[str, list[int]] = {}
    for index, normalized in enumerate(normalized_texts):
        if normalized:
            indices_by_text.setdefault(normalized, []).append(index)
    for normalized, indices in indices_by_text.items():
        if len(indices) < 3:
            continue
        window_start = 0
        for window_end in range(len(indices)):
            right_index = indices[window_end]
            right_end = _segment_time(raw_segments[right_index], "end", 0.0)
            while window_start < window_end:
                left_index = indices[window_start]
                left_start = _segment_time(raw_segments[left_index], "start", 0.0)
                if right_end - left_start <= REPEATED_SEGMENT_WINDOW_SECONDS:
                    break
                window_start += 1
            repeat_count = window_end - window_start + 1
            if repeat_count >= 3:
                first_index = indices[window_start]
                first_start = _segment_time(raw_segments[first_index], "start", 0.0)
                findings.append(
                    {
                        "rule": "repeated_segment_within_45s",
                        "segment_index": first_index,
                        "unit_length": len(normalized),
                        "repeat_count": repeat_count,
                        "text_sha256": text_sha256(texts[first_index]),
                        "text_length": len(texts[first_index]),
                        "related_segment_indices": indices[window_start : window_end + 1],
                        "window_seconds": max(0.0, right_end - first_start),
                        "severity": "hard",
                    }
                )
                break

    return LoopDetectionResult(findings=tuple(findings), segment_count=len(raw_segments))


def runtime_fallback_exhausted(runtime_diagnostics: Any) -> bool:
    if not isinstance(runtime_diagnostics, Mapping):
        return False
    if bool(runtime_diagnostics.get("fallback_exhausted", False)):
        return True
    windows = runtime_diagnostics.get("windows", [])
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        return False
    return any(isinstance(window, Mapping) and bool(window.get("fallback_exhausted", False)) for window in windows)


_REQUIRED_RUNTIME_WINDOW_FIELDS = frozenset(
    {
        "attempts",
        "content_frames",
        "fallback_exhausted",
        "final_failure_reasons",
        "padding_frames",
        "seek_after",
        "seek_before",
        "seek_clip_end",
        "seek_clip_start",
        "timestamp_overrun_seconds",
        "timestamp_violation_count",
        "total_content_frames",
    }
)


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and value != number:
        return None
    return number


def _runtime_safety_findings(
    result: Mapping[str, Any],
    *,
    duration: float | None,
) -> dict[str, int | bool]:
    tolerance = 1 / WHISPER_SAMPLE_RATE
    raw_segments_value = result.get("segments", [])
    raw_segments_sequence = (
        raw_segments_value
        if isinstance(raw_segments_value, Sequence) and not isinstance(raw_segments_value, (str, bytes))
        else ()
    )
    invalid_segment_count = int(raw_segments_sequence is not raw_segments_value)
    raw_timestamp_violation_count = 0
    raw_overrun_samples = 0
    nonempty_segment_count = 0
    for segment in raw_segments_sequence:
        if not isinstance(segment, Mapping):
            invalid_segment_count += 1
            continue
        if str(segment.get("text", "")).strip():
            nonempty_segment_count += 1
        start = _segment_time(segment, "start", math.nan)
        end = _segment_time(segment, "end", math.nan)
        finite = math.isfinite(start) and math.isfinite(end)
        overrun = bool(duration is not None and duration > 0 and finite and end > duration + tolerance)
        invalid_shape = not finite or start < 0.0 or end <= start
        invalid_segment_count += int(invalid_shape)
        raw_timestamp_violation_count += int(invalid_shape or overrun)
        if overrun and duration is not None:
            raw_overrun_samples += max(
                0,
                int(math.ceil((end - duration - tolerance) * WHISPER_SAMPLE_RATE)),
            )

    runtime_diagnostics = result.get("runtime_diagnostics")
    diagnostics = runtime_diagnostics if isinstance(runtime_diagnostics, Mapping) else None
    diagnostics_incomplete = diagnostics is None or diagnostics.get("schema_version") != 2
    windows_value = diagnostics.get("windows") if diagnostics is not None else None
    windows = (
        [window for window in windows_value if isinstance(window, Mapping)]
        if isinstance(windows_value, Sequence) and not isinstance(windows_value, (str, bytes))
        else []
    )
    if (
        not isinstance(windows_value, Sequence)
        or isinstance(windows_value, (str, bytes))
        or len(windows) != len(windows_value)
    ):
        diagnostics_incomplete = True
    if duration is not None and duration > 0 and not windows:
        diagnostics_incomplete = True

    seek_nonprogress_count = 0
    repeated_frame_range_count = 0
    seen_frame_ranges: set[tuple[int, int]] = set()
    window_timestamp_violation_count = 0
    window_overrun_seconds = 0.0
    for window in windows:
        if not set(window) >= _REQUIRED_RUNTIME_WINDOW_FIELDS:
            diagnostics_incomplete = True
        seek_before = _strict_int(window.get("seek_before"))
        seek_after = _strict_int(window.get("seek_after"))
        clip_start = _strict_int(window.get("seek_clip_start"))
        clip_end = _strict_int(window.get("seek_clip_end"))
        total_content_frames = _strict_int(window.get("total_content_frames"))
        content_frames = _strict_int(window.get("content_frames"))
        padding_frames = _strict_int(window.get("padding_frames"))
        frame_fields = (
            seek_before,
            seek_after,
            clip_start,
            clip_end,
            total_content_frames,
            content_frames,
            padding_frames,
        )
        if any(value is None for value in frame_fields):
            diagnostics_incomplete = True
        else:
            assert seek_before is not None
            assert seek_after is not None
            assert clip_start is not None
            assert clip_end is not None
            assert total_content_frames is not None
            assert content_frames is not None
            assert padding_frames is not None
            terminal = min(clip_end, total_content_frames)
            frame_end = min(terminal, seek_before + max(0, content_frames))
            invariant_failed = (
                seek_before < clip_start
                or seek_before >= terminal
                or seek_after <= seek_before
                or seek_after > terminal
                or content_frames <= 0
                or padding_frames < 0
                or frame_end <= seek_before
            )
            seek_nonprogress_count += int(invariant_failed)
            frame_range = (seek_before, frame_end)
            if frame_range in seen_frame_ranges:
                repeated_frame_range_count += 1
            seen_frame_ranges.add(frame_range)
        window_timestamp_violation_count += max(
            0,
            _strict_int(window.get("timestamp_violation_count")) or 0,
        )
        window_overrun_seconds = max(
            window_overrun_seconds,
            _safe_float(window.get("timestamp_overrun_seconds")),
        )

    top_timestamp_violation_count = (
        max(0, _strict_int(diagnostics.get("timestamp_violation_count")) or 0) if diagnostics is not None else 0
    )
    top_overrun_seconds = (
        max(0.0, _safe_float(diagnostics.get("timestamp_overrun_seconds"))) if diagnostics is not None else 0.0
    )
    timestamp_violation_count = max(
        raw_timestamp_violation_count,
        top_timestamp_violation_count,
        window_timestamp_violation_count,
    )
    timestamp_overrun_samples = max(
        raw_overrun_samples,
        int(math.ceil(max(top_overrun_seconds, window_overrun_seconds) * WHISPER_SAMPLE_RATE)),
    )
    output_truncated = bool(
        result.get("output_truncated", result.get("truncated", False))
        or (diagnostics is not None and diagnostics.get("output_truncated", False))
    )
    return {
        "diagnostics_incomplete": diagnostics_incomplete,
        "empty_output": nonempty_segment_count == 0,
        "invalid_segment_count": invalid_segment_count,
        "output_truncated": output_truncated,
        "repeated_frame_range_count": repeated_frame_range_count,
        "seek_nonprogress_count": seek_nonprogress_count,
        "timestamp_overrun_samples": timestamp_overrun_samples,
        "timestamp_violation_count": timestamp_violation_count,
    }


@dataclass(frozen=True)
class RuntimeAnomaly:
    fallback_exhausted: bool
    loop_detection: LoopDetectionResult
    diagnostics_incomplete: bool
    invalid_segment_count: int
    timestamp_violation_count: int
    timestamp_overrun_samples: int
    seek_nonprogress_count: int
    repeated_frame_range_count: int
    empty_output: bool
    output_truncated: bool

    @property
    def repair_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.loop_detection.loop_detected:
            reasons.append("high_confidence_loop")
        if self.timestamp_violation_count or self.timestamp_overrun_samples:
            reasons.append("timestamp_violation")
        if self.invalid_segment_count:
            reasons.append("invalid_segment")
        if self.seek_nonprogress_count:
            reasons.append("seek_invariant")
        if self.repeated_frame_range_count:
            reasons.append("repeated_frame_range")
        if self.empty_output:
            reasons.append("empty_output")
        if self.output_truncated:
            reasons.append("output_truncated")
        if self.fallback_exhausted:
            reasons.append("fallback_exhausted")
        return tuple(dict.fromkeys(reasons))

    @property
    def anomaly_reasons(self) -> tuple[str, ...]:
        reasons = list(self.repair_reasons)
        if self.diagnostics_incomplete:
            reasons.append("diagnostics_incomplete")
        return tuple(dict.fromkeys(reasons))

    @property
    def anomaly_detected(self) -> bool:
        return bool(self.anomaly_reasons)

    @property
    def repair_required(self) -> bool:
        return bool(self.repair_reasons)

    @property
    def review_required(self) -> bool:
        return self.loop_detection.review_required

    @property
    def anomaly_reason(self) -> str | None:
        if self.fallback_exhausted and self.loop_detection.loop_detected:
            return "both"
        if self.repair_reasons:
            return self.repair_reasons[0]
        return "diagnostics_incomplete" if self.diagnostics_incomplete else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "anomaly_detected": self.anomaly_detected,
            "anomaly_reason": self.anomaly_reason,
            "anomaly_reasons": list(self.anomaly_reasons),
            "repair_required": self.repair_required,
            "repair_reasons": list(self.repair_reasons),
            "fallback_exhausted": self.fallback_exhausted,
            "diagnostics_incomplete": self.diagnostics_incomplete,
            "invalid_segment_count": self.invalid_segment_count,
            "timestamp_violation_count": self.timestamp_violation_count,
            "timestamp_overrun_samples": self.timestamp_overrun_samples,
            "seek_nonprogress_count": self.seek_nonprogress_count,
            "repeated_frame_range_count": self.repeated_frame_range_count,
            "empty_output": self.empty_output,
            "output_truncated": self.output_truncated,
            "review_required": self.review_required,
            **self.loop_detection.as_dict(),
        }


def analyze_runtime_result(
    result: Mapping[str, Any],
    *,
    duration: float | None = None,
) -> RuntimeAnomaly:
    raw_segments_value = result.get("segments", [])
    raw_segments: list[Mapping[str, Any]] = []
    if isinstance(raw_segments_value, Sequence) and not isinstance(raw_segments_value, (str, bytes)):
        raw_segments = [segment for segment in raw_segments_value if isinstance(segment, Mapping)]
    safety = _runtime_safety_findings(result, duration=duration)
    return RuntimeAnomaly(
        fallback_exhausted=runtime_fallback_exhausted(result.get("runtime_diagnostics")),
        loop_detection=detect_high_confidence_loops(raw_segments, duration=duration),
        diagnostics_incomplete=bool(safety["diagnostics_incomplete"]),
        invalid_segment_count=int(safety["invalid_segment_count"]),
        timestamp_violation_count=int(safety["timestamp_violation_count"]),
        timestamp_overrun_samples=int(safety["timestamp_overrun_samples"]),
        seek_nonprogress_count=int(safety["seek_nonprogress_count"]),
        repeated_frame_range_count=int(safety["repeated_frame_range_count"]),
        empty_output=bool(safety["empty_output"]),
        output_truncated=bool(safety["output_truncated"]),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


@dataclass(frozen=True)
class CandidateQuality:
    """Lexicographic candidate quality; every component is lower-is-better."""

    diagnostics_incomplete_count: int
    hard_loop_severity: int
    unsupported_hard_loop_count: int
    hard_finding_count: int
    timestamp_overrun_samples: int
    invalid_timestamp_count: int
    invalid_segment_count: int
    seek_nonprogress_count: int
    repeated_frame_range_count: int
    empty_output_count: int
    output_truncated_count: int
    fallback_exhausted_window_count: int
    terminal_failure_reason_count: int
    coverage_deficit_samples: int
    empty_text_ratio_ppm: int
    negative_nonempty_segment_count: int
    soft_finding_count: int

    @property
    def comparison_key(self) -> tuple[int, ...]:
        return (
            self.hard_loop_severity,
            self.unsupported_hard_loop_count,
            self.hard_finding_count,
            self.timestamp_overrun_samples,
            self.invalid_timestamp_count,
            self.invalid_segment_count,
            self.seek_nonprogress_count,
            self.repeated_frame_range_count,
            self.empty_output_count,
            self.output_truncated_count,
            self.fallback_exhausted_window_count,
            self.terminal_failure_reason_count,
            self.diagnostics_incomplete_count,
            self.coverage_deficit_samples,
            self.empty_text_ratio_ppm,
            self.negative_nonempty_segment_count,
            self.soft_finding_count,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnostics_incomplete_count": self.diagnostics_incomplete_count,
            "hard_loop_severity": self.hard_loop_severity,
            "unsupported_hard_loop_count": self.unsupported_hard_loop_count,
            "hard_finding_count": self.hard_finding_count,
            "timestamp_overrun_samples": self.timestamp_overrun_samples,
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "invalid_segment_count": self.invalid_segment_count,
            "seek_nonprogress_count": self.seek_nonprogress_count,
            "repeated_frame_range_count": self.repeated_frame_range_count,
            "empty_output_count": self.empty_output_count,
            "output_truncated_count": self.output_truncated_count,
            "fallback_exhausted_window_count": self.fallback_exhausted_window_count,
            "terminal_failure_reason_count": self.terminal_failure_reason_count,
            "coverage_deficit_samples": self.coverage_deficit_samples,
            "empty_text_ratio_ppm": self.empty_text_ratio_ppm,
            "negative_nonempty_segment_count": self.negative_nonempty_segment_count,
            "soft_finding_count": self.soft_finding_count,
            "comparison_key": list(self.comparison_key),
        }


def _quality_runtime_windows(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics = _mapping(result.get("runtime_diagnostics"))
    if diagnostics is None:
        return []
    return [window for window in _sequence(diagnostics.get("windows")) if isinstance(window, Mapping)]


def _covered_samples(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    covered = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    return covered + current_end - current_start


def _hard_loop_severity(loop_detection: LoopDetectionResult) -> int:
    severity = 0
    for finding in loop_detection.hard_findings:
        repeat_count = max(1, _safe_int(finding.get("repeat_count"), 1))
        unit_length = max(1, _safe_int(finding.get("unit_length"), 1))
        if finding.get("rule") == "consecutive_unit":
            severity += max(1, repeat_count - _repeat_threshold(unit_length) + 1)
        elif finding.get("rule") == "repeated_segment_within_45s":
            severity += max(1, repeat_count - 2)
        else:
            severity += 1
    return severity


def evaluate_candidate_quality(
    result: Mapping[str, Any],
    *,
    sample_count: int,
    anomaly: RuntimeAnomaly | None = None,
) -> CandidateQuality:
    """Evaluate a runtime candidate without changing its text or timestamps."""
    bounded_sample_count = max(0, int(sample_count))
    duration = bounded_sample_count / WHISPER_SAMPLE_RATE
    anomaly = anomaly or analyze_runtime_result(result, duration=duration)
    raw_segments = [segment for segment in _sequence(result.get("segments")) if isinstance(segment, Mapping)]
    empty_text_count = 0
    nonempty_segment_count = 0
    intervals: list[tuple[int, int]] = []
    for segment in raw_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            empty_text_count += 1
        else:
            nonempty_segment_count += 1
        start = _segment_time(segment, "start", math.nan)
        end = _segment_time(segment, "end", math.nan)
        finite = math.isfinite(start) and math.isfinite(end)
        if finite and text:
            start_sample = max(0, min(bounded_sample_count, int(round(start * WHISPER_SAMPLE_RATE))))
            end_sample = max(start_sample, min(bounded_sample_count, int(round(end * WHISPER_SAMPLE_RATE))))
            if end_sample > start_sample:
                intervals.append((start_sample, end_sample))

    runtime_diagnostics = _mapping(result.get("runtime_diagnostics")) or {}
    windows = _quality_runtime_windows(result)
    exhausted_windows = sum(bool(window.get("fallback_exhausted", False)) for window in windows)
    if not windows and bool(runtime_diagnostics.get("fallback_exhausted", False)):
        exhausted_windows = 1
    terminal_failure_reason_count = sum(
        len(_safe_failure_reasons(window.get("final_failure_reasons")))
        for window in windows
        if bool(window.get("fallback_exhausted", False))
    )
    total_segments = len(raw_segments)
    empty_text_ratio_ppm = int(round(empty_text_count * 1_000_000 / total_segments)) if total_segments else 1_000_000
    coverage_deficit_samples = max(0, bounded_sample_count - _covered_samples(intervals))
    hard_finding_count = len(anomaly.loop_detection.hard_findings)
    return CandidateQuality(
        diagnostics_incomplete_count=int(anomaly.diagnostics_incomplete),
        hard_loop_severity=_hard_loop_severity(anomaly.loop_detection),
        unsupported_hard_loop_count=hard_finding_count,
        hard_finding_count=hard_finding_count,
        timestamp_overrun_samples=anomaly.timestamp_overrun_samples,
        invalid_timestamp_count=anomaly.timestamp_violation_count,
        invalid_segment_count=anomaly.invalid_segment_count,
        seek_nonprogress_count=anomaly.seek_nonprogress_count,
        repeated_frame_range_count=anomaly.repeated_frame_range_count,
        empty_output_count=int(anomaly.empty_output),
        output_truncated_count=int(anomaly.output_truncated),
        fallback_exhausted_window_count=exhausted_windows,
        terminal_failure_reason_count=terminal_failure_reason_count,
        coverage_deficit_samples=coverage_deficit_samples,
        empty_text_ratio_ppm=empty_text_ratio_ppm,
        negative_nonempty_segment_count=-nonempty_segment_count,
        soft_finding_count=len(anomaly.loop_detection.soft_findings),
    )


@dataclass(frozen=True)
class CT2RescueDecision:
    """Pure decision returned by the MLX-to-CT2 rescue safety gate."""

    should_rescue: bool
    reason: str


_CT2_RESCUE_RECORD_BOOL_FIELDS = (
    "anomaly_detected",
    "repair_required",
    "fallback_exhausted",
    "diagnostics_incomplete",
    "empty_output",
    "output_truncated",
    "loop_detected",
    "soft_review_required",
)
_CT2_RESCUE_RECORD_COUNT_FIELDS = (
    "invalid_segment_count",
    "timestamp_violation_count",
    "timestamp_overrun_samples",
    "seek_nonprogress_count",
    "repeated_frame_range_count",
    "hard_finding_count",
    "soft_finding_count",
)
_CT2_RESCUE_ADAPTER_COUNT_FIELDS = (
    "raw_segment_count",
    "adapter_timestamp_clamp_count",
    "adapter_invalid_timestamp_count",
    "adapter_segment_drop_count",
)
_CT2_RESCUE_INITIAL_SELECTION_REASONS = frozenset(
    {
        "repair_evidence_incomplete",
        "repair_fallback_without_hard_loop_improvement",
        "repair_not_strictly_better",
    }
)


def _diagnostic_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _metric_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None
    return int(number)


def _ct2_rescue_record_is_complete(record: Mapping[str, Any]) -> bool:
    if any(not isinstance(record.get(name), bool) for name in _CT2_RESCUE_RECORD_BOOL_FIELDS):
        return False
    if any(_diagnostic_nonnegative_int(record.get(name)) is None for name in _CT2_RESCUE_RECORD_COUNT_FIELDS):
        return False

    quality = _mapping(record.get("quality"))
    if quality is None:
        return False
    diagnostics_incomplete_count = _diagnostic_nonnegative_int(quality.get("diagnostics_incomplete_count"))
    hard_loop_severity = _diagnostic_nonnegative_int(quality.get("hard_loop_severity"))
    hard_finding_count = _diagnostic_nonnegative_int(quality.get("hard_finding_count"))
    if diagnostics_incomplete_count is None or hard_loop_severity is None or hard_finding_count is None:
        return False
    if bool(diagnostics_incomplete_count) != record["diagnostics_incomplete"]:
        return False
    if hard_finding_count != record["hard_finding_count"]:
        return False
    if bool(hard_loop_severity) != bool(hard_finding_count):
        return False
    return bool(record["diagnostics_incomplete"]) or bool(record["anomaly_detected"]) == bool(record["repair_required"])


def _ct2_rescue_adapter_is_complete(adapter: Mapping[str, Any]) -> bool:
    return all(_diagnostic_nonnegative_int(adapter.get(name)) is not None for name in _CT2_RESCUE_ADAPTER_COUNT_FIELDS)


def _backend_result_has_text(result: BackendResult) -> bool:
    segments = result.segments
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        return False
    return any(isinstance(text := getattr(segment, "text", None), str) and bool(text.strip()) for segment in segments)


def _backend_result_has_invalid_timeline(result: BackendResult) -> bool:
    if isinstance(result.duration, bool) or not isinstance(result.duration, (int, float)):
        return True
    duration = float(result.duration)
    if not math.isfinite(duration) or duration < 0.0:
        return True

    segments = result.segments
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        return True
    tolerance = 1 / WHISPER_SAMPLE_RATE
    previous_start = -math.inf
    previous_end = -math.inf
    for segment in segments:
        text = getattr(segment, "text", None)
        start_value = getattr(segment, "start", None)
        end_value = getattr(segment, "end", None)
        if not isinstance(text, str) or not text.strip():
            return True
        if isinstance(start_value, bool) or isinstance(end_value, bool):
            return True
        if not isinstance(start_value, (int, float)) or not isinstance(end_value, (int, float)):
            return True
        start = float(start_value)
        end = float(end_value)
        if not math.isfinite(start) or not math.isfinite(end):
            return True
        if start < 0.0 or end <= start or end > duration + tolerance:
            return True
        if start + tolerance < previous_start or end + tolerance < previous_end:
            return True
        previous_start = start
        previous_end = end
    return False


def decide_ct2_rescue(
    result: BackendResult,
    diagnostics: Mapping[str, Any] | None = None,
    *,
    outer_vad_has_speech: bool,
    ct2_rescue_already_attempted: bool = False,
) -> CT2RescueDecision:
    """Decide whether one CT2 rescue is justified without running any backend."""
    if result.backend != "mlx":
        return CT2RescueDecision(False, "not_mlx_backend")
    if ct2_rescue_already_attempted:
        return CT2RescueDecision(False, "ct2_rescue_already_attempted")
    if not isinstance(outer_vad_has_speech, bool):
        return CT2RescueDecision(False, "outer_vad_evidence_incomplete")

    evidence = diagnostics if diagnostics is not None else result.diagnostics
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != MLX_DIAGNOSTIC_SCHEMA_VERSION:
        return CT2RescueDecision(False, "diagnostics_incomplete")

    initial = _mapping(evidence.get("initial"))
    if initial is None:
        return CT2RescueDecision(False, "mlx_production_not_run")

    repair = _mapping(evidence.get("repair"))
    safe_retry_count = _diagnostic_nonnegative_int(evidence.get("safe_retry_count"))
    repair_mode = evidence.get("repair_mode")
    if repair is None or safe_retry_count == 0 or repair_mode is None:
        return CT2RescueDecision(False, "fixed_t0_repair_not_run")
    if safe_retry_count != 1 or repair_mode != "full_t0":
        return CT2RescueDecision(False, "diagnostics_incomplete")

    if not isinstance(result.metrics, Mapping):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    runtime_call_count = _metric_count(result.metrics.get("runtime_call_count"))
    metric_retry_count = _metric_count(result.metrics.get("safe_retry_count"))
    if runtime_call_count is None or metric_retry_count is None:
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if runtime_call_count != 2:
        return CT2RescueDecision(False, "mlx_runtime_call_count_not_two")
    if metric_retry_count != 1:
        return CT2RescueDecision(False, "diagnostics_incomplete")

    selected_name = evidence.get("selected_result")
    if selected_name not in {"initial", "repair"}:
        return CT2RescueDecision(False, "invalid_selected_result")
    repair_reason = evidence.get("repair_reason")
    selection_reason = evidence.get("selection_reason")
    repair_rejected = evidence.get("repair_rejected")
    if not isinstance(repair_reason, str) or not repair_reason:
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if not isinstance(selection_reason, str) or not selection_reason:
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if not isinstance(repair_rejected, bool) or repair_rejected != (selected_name == "initial"):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if selected_name == "repair" and selection_reason != "repair_strictly_better":
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if selected_name == "initial" and selection_reason not in _CT2_RESCUE_INITIAL_SELECTION_REASONS:
        return CT2RescueDecision(False, "diagnostics_incomplete")
    selected = initial if selected_name == "initial" else repair
    if not _ct2_rescue_record_is_complete(initial) or not _ct2_rescue_record_is_complete(repair):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if initial["diagnostics_incomplete"] or repair["diagnostics_incomplete"]:
        return CT2RescueDecision(False, "diagnostics_incomplete")

    adapter = _mapping(evidence.get("adapter"))
    selected_quality = _mapping(evidence.get("selected_quality"))
    if adapter is None or not _ct2_rescue_adapter_is_complete(adapter):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if selected_quality is None or dict(selected_quality) != dict(_mapping(selected.get("quality")) or {}):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if evidence.get("selected_not_worse") is not True:
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if not isinstance(evidence.get("unresolved_anomaly"), bool):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if evidence["unresolved_anomaly"] != selected["anomaly_detected"]:
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if not isinstance(evidence.get("soft_review_required"), bool):
        return CT2RescueDecision(False, "diagnostics_incomplete")
    if evidence["soft_review_required"] != selected["soft_review_required"]:
        return CT2RescueDecision(False, "diagnostics_incomplete")

    if not repair["anomaly_detected"] or not repair["repair_required"]:
        return CT2RescueDecision(False, "repair_reliable")

    if outer_vad_has_speech and not _backend_result_has_text(result):
        return CT2RescueDecision(True, "voiced_empty_output")
    if selected["output_truncated"]:
        return CT2RescueDecision(True, "output_truncated")
    if selected["invalid_segment_count"] > 0:
        return CT2RescueDecision(True, "invalid_segment_structure")
    if selected["timestamp_violation_count"] > 0 or selected["timestamp_overrun_samples"] > 0:
        return CT2RescueDecision(True, "invalid_timestamp_structure")
    if selected["seek_nonprogress_count"] > 0:
        return CT2RescueDecision(True, "seek_nonprogress")
    if selected["repeated_frame_range_count"] > 0:
        return CT2RescueDecision(True, "repeated_frame_range")
    if _backend_result_has_invalid_timeline(result):
        return CT2RescueDecision(True, "invalid_final_timeline")

    selected_quality_hard_severity = _diagnostic_nonnegative_int(selected_quality.get("hard_loop_severity")) or 0
    repair_quality = _mapping(repair.get("quality")) or {}
    repair_hard_severity = _diagnostic_nonnegative_int(repair_quality.get("hard_loop_severity")) or 0
    if (
        selected["fallback_exhausted"]
        and selected_quality_hard_severity > 0
        and repair["fallback_exhausted"]
        and repair_hard_severity > 0
    ):
        return CT2RescueDecision(True, "hard_loop_with_fallback_exhausted")
    return CT2RescueDecision(False, "no_objective_high_risk_anomaly")


def _safe_failure_reasons(value: Any) -> list[str]:
    return [reason for reason in _sequence(value) if isinstance(reason, str) and reason in _FAILURE_REASONS]


def _safe_anomaly_reason(value: Any) -> str | None:
    return value if isinstance(value, str) and value in _ANOMALY_REASONS else None


def _runtime_windows(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runtime_diagnostics = _mapping(record.get("runtime_diagnostics"))
    if runtime_diagnostics is None:
        return []
    return [window for window in _sequence(runtime_diagnostics.get("windows")) if isinstance(window, Mapping)]


def _safe_token_ids(value: Any, *, limit: int) -> list[int]:
    return [_safe_int(token) for token in _sequence(value)[:limit]]


def _safe_attempt(attempt: Mapping[str, Any], *, include_details: bool) -> dict[str, Any]:
    sanitized = {
        "temperature": _safe_float(attempt.get("temperature")),
        "compression_ratio": _safe_float(attempt.get("compression_ratio")),
        "avg_logprob": _safe_float(attempt.get("avg_logprob")),
        "no_speech_prob": _safe_float(attempt.get("no_speech_prob")),
        "failure_reasons": _safe_failure_reasons(attempt.get("failure_reasons")),
        "accepted": bool(attempt.get("accepted", False)),
        "text_sha256": _safe_text_hash(attempt.get("text_sha256")),
        "text_length": max(0, _safe_int(attempt.get("text_length"))),
        "token_sha256": _safe_text_hash(attempt.get("token_sha256")),
        "token_count": max(0, _safe_int(attempt.get("token_count"))),
        "timestamp_token_sha256": _safe_text_hash(attempt.get("timestamp_token_sha256")),
        "timestamp_token_count": max(0, _safe_int(attempt.get("timestamp_token_count"))),
    }
    if include_details:
        text_excerpt = attempt.get("text_excerpt")
        sanitized.update(
            {
                "text_excerpt": text_excerpt[:160] if isinstance(text_excerpt, str) else "",
                "text_truncated": bool(attempt.get("text_truncated", False)),
                "token_ids": _safe_token_ids(attempt.get("token_ids"), limit=128),
                "tokens_truncated": bool(attempt.get("tokens_truncated", False)),
                "timestamp_token_ids": _safe_token_ids(attempt.get("timestamp_token_ids"), limit=64),
                "timestamp_tokens_truncated": bool(attempt.get("timestamp_tokens_truncated", False)),
            }
        )
    return sanitized


def _safe_window(
    window: Mapping[str, Any],
    *,
    runtime_phase: str,
    include_details: bool = False,
) -> dict[str, Any]:
    raw_attempts = [attempt for attempt in _sequence(window.get("attempts")) if isinstance(attempt, Mapping)]
    return {
        "runtime_phase": runtime_phase,
        "seek": _safe_int(window.get("seek")),
        "seek_before": _safe_int(window.get("seek_before", window.get("seek"))),
        "seek_after": _safe_int(window.get("seek_after")),
        "seek_update_branch": str(window.get("seek_update_branch", ""))[:64],
        "seek_forced_progress": bool(window.get("seek_forced_progress", False)),
        "seek_clamped": bool(window.get("seek_clamped", False)),
        "seek_clip_start": _safe_int(window.get("seek_clip_start", window.get("clip_start"))),
        "seek_clip_end": _safe_int(window.get("seek_clip_end", window.get("clip_end"))),
        "total_content_frames": _safe_int(window.get("total_content_frames")),
        "remaining_content_frames": _safe_int(window.get("remaining_content_frames")),
        "content_frames": _safe_int(window.get("content_frames", window.get("segment_size"))),
        "padding_frames": _safe_int(window.get("padding_frames")),
        "segment_size": _safe_int(window.get("segment_size")),
        "segment_duration": _safe_float(window.get("segment_duration")),
        "timestamp_adjustment_count": _safe_int(window.get("timestamp_adjustment_count")),
        "timestamp_violation_count": _safe_int(window.get("timestamp_violation_count")),
        "timestamp_overrun_seconds": _safe_float(window.get("timestamp_overrun_seconds")),
        "attempt_count": len(raw_attempts),
        "attempts": [
            _safe_attempt(attempt, include_details=include_details) for attempt in raw_attempts[:MLX_DEBUG_MAX_ATTEMPTS]
        ],
        "attempts_truncated": len(raw_attempts) > MLX_DEBUG_MAX_ATTEMPTS,
        "fallback_exhausted": bool(window.get("fallback_exhausted", False)),
        "final_failure_reasons": _safe_failure_reasons(window.get("final_failure_reasons")),
    }


def _record_attempt_count(record: Mapping[str, Any]) -> int:
    return sum(
        len([attempt for attempt in _sequence(window.get("attempts")) if isinstance(attempt, Mapping)])
        for window in _runtime_windows(record)
    )


def _safe_text_hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        return ""
    normalized = value.lower()
    return normalized if all(character in "0123456789abcdef" for character in normalized) else ""


def _safe_loop_findings(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_findings = [finding for finding in _sequence(record.get("findings")) if isinstance(finding, Mapping)]
    findings: list[dict[str, Any]] = []
    for finding in raw_findings[:MLX_LOOP_FINDING_LIMIT]:
        rule = finding.get("rule")
        if not isinstance(rule, str) or rule not in _LOOP_RULES:
            continue
        findings.append(
            {
                "rule": rule,
                "severity": "soft" if finding.get("severity") == "soft" else "hard",
                "segment_index": _safe_int(finding.get("segment_index")),
                "unit_length": _safe_int(finding.get("unit_length")),
                "repeat_count": _safe_int(finding.get("repeat_count")),
                "text_length": _safe_int(finding.get("text_length")),
                "text_sha256": _safe_text_hash(finding.get("text_sha256")),
            }
        )
    return findings, len(raw_findings)


def _diagnostic_record(diagnostics: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    return _mapping(diagnostics.get(phase)) or {}


def _anomaly_event(
    base: Mapping[str, Any],
    *,
    runtime_phase: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    raw_windows = [
        window
        for window in _runtime_windows(record)
        if bool(window.get("fallback_exhausted", False))
        or bool(_safe_failure_reasons(window.get("final_failure_reasons")))
        or _safe_int(window.get("timestamp_violation_count")) > 0
        or _safe_float(window.get("timestamp_overrun_seconds")) > 0.0
        or (
            "seek_after" in window
            and "seek_before" in window
            and _safe_int(window.get("seek_after")) <= _safe_int(window.get("seek_before"))
        )
    ]
    findings, finding_count = _safe_loop_findings(record)
    event = dict(base)
    event.update(
        {
            "event": "mlx_runtime_anomaly",
            "runtime_phase": runtime_phase,
            "anomaly_reason": _safe_anomaly_reason(record.get("anomaly_reason")),
            "fallback_exhausted": bool(record.get("fallback_exhausted", False)),
            "loop_detected": bool(record.get("loop_detected", False)),
            "soft_review_required": bool(record.get("soft_review_required", False)),
            "hard_finding_count": _safe_int(record.get("hard_finding_count")),
            "soft_finding_count": _safe_int(record.get("soft_finding_count")),
            "anomalous_windows": [
                _safe_window(window, runtime_phase=runtime_phase) for window in raw_windows[:MLX_DEBUG_MAX_WINDOWS]
            ],
            "anomalous_window_count": len(raw_windows),
            "windows_truncated": len(raw_windows) > MLX_DEBUG_MAX_WINDOWS,
            "loop_findings": findings,
            "loop_finding_count": finding_count,
            "findings_truncated": finding_count > len(findings),
        }
    )
    return event


def serialize_mlx_diagnostic_event(event: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(event),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def mlx_diagnostic_events_size(events: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(serialize_mlx_diagnostic_event(event).encode("utf-8")) + 1 for event in events)


def limit_mlx_diagnostic_events(
    events: Sequence[Mapping[str, Any]],
    *,
    max_bytes: int = MLX_DEBUG_MAX_BYTES,
) -> tuple[dict[str, Any], ...]:
    copied_events = [dict(event) for event in events]
    if mlx_diagnostic_events_size(copied_events) <= max_bytes:
        return tuple(copied_events)
    if not copied_events:
        return ()

    source = copied_events[-1]
    summary = {name: source.get(name) for name in MLX_EVENT_REQUIRED_FIELDS}
    summary.update(
        {
            "schema_version": MLX_DIAGNOSTIC_SCHEMA_VERSION,
            "event": "mlx_diagnostics_truncated",
            "backend": "mlx",
            "truncated": True,
            "emitted_event_count": 0,
            "dropped_event_count": len(copied_events),
        }
    )
    if mlx_diagnostic_events_size([summary]) > max_bytes:
        raise ValueError("max_bytes is too small for the required truncated diagnostic summary")

    kept: list[dict[str, Any]] = []
    for event in copied_events:
        candidate_summary = dict(summary)
        candidate_summary["emitted_event_count"] = len(kept) + 1
        candidate_summary["dropped_event_count"] = len(copied_events) - len(kept) - 1
        if mlx_diagnostic_events_size([*kept, event, candidate_summary]) > max_bytes:
            continue
        kept.append(event)

    summary["emitted_event_count"] = len(kept)
    summary["dropped_event_count"] = len(copied_events) - len(kept)
    while kept and mlx_diagnostic_events_size([*kept, summary]) > max_bytes:
        kept.pop()
        summary["emitted_event_count"] = len(kept)
        summary["dropped_event_count"] = len(copied_events) - len(kept)
    return (*kept, summary)


def build_mlx_diagnostic_events(
    *,
    diagnostics: Mapping[str, Any],
    metrics: Mapping[str, float],
    duration: float,
    segment_count: int,
    debug: bool,
) -> tuple[dict[str, Any], ...]:
    initial = _diagnostic_record(diagnostics, "initial")
    repair = _diagnostic_record(diagnostics, "repair") or _diagnostic_record(diagnostics, "retry")
    selected_name = "repair" if diagnostics.get("selected_result") in {"repair", "retry"} else "initial"
    selected = repair if selected_name == "repair" else initial
    initial_attempt_count = _record_attempt_count(initial)
    repair_attempt_count = _record_attempt_count(repair)
    base: dict[str, Any] = {
        "schema_version": MLX_DIAGNOSTIC_SCHEMA_VERSION,
        "event": "mlx_request_summary",
        "backend": "mlx",
        "clip_mode": "clips" if diagnostics.get("initial_clip_mode") == "clips" else "full_chunk",
        "sample_count": _safe_int(diagnostics.get("sample_count")),
        "duration": _safe_float(duration),
        "attempt_count": initial_attempt_count + repair_attempt_count,
        "fallback_exhausted": bool(selected.get("fallback_exhausted", False)),
        "loop_detected": bool(selected.get("loop_detected", False)),
        "soft_review_required": bool(selected.get("soft_review_required", False)),
        "runtime_call_count": max(
            1,
            _safe_int(
                metrics.get("runtime_call_count"),
                1 + _safe_int(diagnostics.get("safe_retry_count")),
            ),
        ),
        "safe_retry_count": _safe_int(diagnostics.get("safe_retry_count")),
        "unresolved_anomaly": bool(diagnostics.get("unresolved_anomaly", False)),
        "segment_count": max(0, int(segment_count)),
        "inference_seconds": _safe_float(metrics.get("inference_seconds")),
        "peak_memory_bytes": _safe_float(metrics.get("peak_memory_bytes")),
    }

    request_event = dict(base)
    request_event.update(
        {
            "attempt_count": initial_attempt_count,
            "fallback_exhausted": bool(initial.get("fallback_exhausted", False)),
            "loop_detected": bool(initial.get("loop_detected", False)),
            "soft_review_required": bool(initial.get("soft_review_required", False)),
        }
    )
    events: list[dict[str, Any]] = [request_event]

    if bool(initial.get("anomaly_detected", False)):
        events.append(_anomaly_event(base, runtime_phase="initial", record=initial))
    elif bool(initial.get("soft_review_required", False)):
        findings, finding_count = _safe_loop_findings(initial)
        review_event = dict(base)
        review_event.update(
            {
                "event": "mlx_soft_review",
                "runtime_phase": "initial",
                "loop_findings": findings,
                "loop_finding_count": finding_count,
            }
        )
        events.append(review_event)
    if _safe_int(diagnostics.get("safe_retry_count")) > 0:
        repair_event = dict(base)
        repair_event.update(
            {
                "event": "mlx_repair_candidate",
                "repair_reason": _safe_anomaly_reason(
                    diagnostics.get("repair_reason", diagnostics.get("retry_reason"))
                ),
                "fallback_exhausted": bool(initial.get("fallback_exhausted", False)),
                "loop_detected": bool(initial.get("loop_detected", False)),
            }
        )
        events.append(repair_event)
    if repair and bool(repair.get("anomaly_detected", False)):
        events.append(_anomaly_event(base, runtime_phase="repair", record=repair))

    result_event = dict(base)
    result_event.update(
        {
            "event": "mlx_result_summary",
            "selected_result": selected_name,
            "selection_reason": str(diagnostics.get("selection_reason", ""))[:80],
        }
    )
    events.append(result_event)

    if debug:
        phase_windows = [
            (runtime_phase, window)
            for runtime_phase, record in (("initial", initial), ("repair", repair))
            for window in _runtime_windows(record)
        ]
        debug_event = dict(base)
        debug_event.update(
            {
                "event": "mlx_debug_windows",
                "windows": [
                    _safe_window(
                        window,
                        runtime_phase=runtime_phase,
                        include_details=True,
                    )
                    for runtime_phase, window in phase_windows[:MLX_DEBUG_MAX_WINDOWS]
                ],
                "window_count": len(phase_windows),
                "windows_truncated": len(phase_windows) > MLX_DEBUG_MAX_WINDOWS,
            }
        )
        events.append(debug_event)

    return limit_mlx_diagnostic_events(events)
