"""MLX/Metal Whisper backend adapter."""

from __future__ import annotations

import importlib
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

from ..mlx_diagnostics import (
    CandidateQuality,
    RuntimeAnomaly,
    analyze_runtime_result,
    evaluate_candidate_quality,
)
from .base import (
    BackendCapabilities,
    BackendConfigurationError,
    BackendError,
    BackendRequest,
    BackendResult,
    BackendSegment,
    BackendUnavailableError,
    ModelDescriptor,
)
from .option_mapping import map_mlx_options


class MLXBackend:
    capabilities = BackendCapabilities(
        backend="mlx",
        supports_translate=True,
        supports_transcribe=True,
        supports_word_timestamps=False,
        supports_batching=False,
    )

    def __init__(
        self,
        descriptor: ModelDescriptor,
        *,
        transcribe_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.descriptor = descriptor
        self._transcribe_fn = transcribe_fn
        self._mx: Any | None = None
        self._transcribe_module: Any | None = None
        self._warned_ignored_options: set[str] = set()

    @property
    def batching_enabled(self) -> bool:
        return False

    @property
    def ignored_options(self) -> tuple[str, ...]:
        return tuple(sorted(self._warned_ignored_options))

    def _ensure_runtime(self) -> None:
        if self._transcribe_fn is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            self._mx = importlib.import_module("mlx.core")
            self._mx.set_default_device(self._mx.gpu)
            self._transcribe_module = importlib.import_module("mlx_whisper.transcribe")
            self._transcribe_fn = self._transcribe_module.transcribe
        except Exception as exc:
            raise BackendUnavailableError(f"Failed to initialize the MLX Metal runtime: {exc}") from exc

    @staticmethod
    def _audio_duration(audio: Any, segments: list[BackendSegment]) -> float:
        if not isinstance(audio, (str, bytes)) and hasattr(audio, "__len__"):
            try:
                return float(len(audio)) / 16_000
            except (TypeError, ValueError):
                pass
        return max((segment.end for segment in segments), default=0.0)

    @staticmethod
    def _audio_sample_count(audio: Any) -> int:
        if isinstance(audio, (str, bytes)) or not hasattr(audio, "__len__"):
            return 0
        try:
            return max(0, int(len(audio)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _coerce_bool_option(value: Any, *, default: bool, name: str) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise BackendConfigurationError(f"{name} must be a boolean")

    @staticmethod
    def _coerce_sampling_seed(value: Any, *, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            raise BackendConfigurationError("mlx_sampling_seed must be an integer")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise BackendConfigurationError("mlx_sampling_seed must be an integer") from exc
        if isinstance(value, float) and value != seed:
            raise BackendConfigurationError("mlx_sampling_seed must be an integer")
        if isinstance(value, str) and value.strip() != str(seed):
            raise BackendConfigurationError("mlx_sampling_seed must be an integer")
        if not 0 <= seed <= 2**32 - 1:
            raise BackendConfigurationError("mlx_sampling_seed must be between 0 and 2**32 - 1")
        return seed

    @staticmethod
    def _parse_clip_timestamps(value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_values: list[Any] = value.split(",") if value.strip() else []
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw_values = list(value)
        else:
            raw_values = [value]

        parsed: list[float] = []
        for raw_value in raw_values:
            try:
                number = float(raw_value)
            except (TypeError, ValueError):
                return []
            if not math.isfinite(number):
                return []
            parsed.append(number)
        return parsed

    @staticmethod
    def _is_fixed_t0(value: Any) -> bool:
        values = list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
        if not values or any(isinstance(item, bool) for item in values):
            return False
        try:
            temperatures = [float(item) for item in values]
        except (TypeError, ValueError):
            return False
        return all(math.isfinite(temperature) and temperature == 0.0 for temperature in temperatures)

    @classmethod
    def _uses_non_full_clips(cls, value: Any, *, duration: float) -> bool:
        clip_timestamps = cls._parse_clip_timestamps(value)
        if not clip_timestamps:
            return False
        tolerance = 1.0 / 16_000
        if len(clip_timestamps) == 1 and abs(clip_timestamps[0]) <= tolerance:
            return False
        return not (
            len(clip_timestamps) == 2
            and duration > 0
            and abs(clip_timestamps[0]) <= tolerance
            and clip_timestamps[1] >= duration - tolerance
        )

    def _run_runtime(self, audio: Any, options: dict[str, Any]) -> tuple[dict[str, Any], float]:
        transcribe_fn = self._transcribe_fn
        if transcribe_fn is None:
            raise BackendUnavailableError("MLX transcribe function was not initialized")
        started = time.perf_counter()
        result = transcribe_fn(
            audio,
            path_or_hf_repo=str(self.descriptor.path),
            **options,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(result, dict):
            raise BackendError("MLX runtime returned a non-dictionary result")
        return result, elapsed

    @staticmethod
    def _anomaly_record(result: Mapping[str, Any], anomaly: RuntimeAnomaly) -> dict[str, Any]:
        record = anomaly.as_dict()
        runtime_diagnostics = result.get("runtime_diagnostics")
        record["runtime_diagnostics"] = dict(runtime_diagnostics) if isinstance(runtime_diagnostics, Mapping) else {}
        return record

    @staticmethod
    def _adapt_segments(
        result: Mapping[str, Any],
        *,
        duration: float,
    ) -> tuple[list[BackendSegment], dict[str, int]]:
        segments: list[BackendSegment] = []
        stats = {
            "raw_segment_count": 0,
            "adapter_timestamp_clamp_count": 0,
            "adapter_invalid_timestamp_count": 0,
            "adapter_segment_drop_count": 0,
        }
        raw_segments = result.get("segments", [])
        if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
            return segments, stats
        for item in raw_segments:
            if not isinstance(item, Mapping):
                stats["adapter_segment_drop_count"] += 1
                continue
            stats["raw_segment_count"] += 1
            try:
                raw_start = float(item.get("start", 0.0))
                raw_end = float(item.get("end", raw_start))
            except (TypeError, ValueError):
                stats["adapter_invalid_timestamp_count"] += 1
                stats["adapter_segment_drop_count"] += 1
                continue
            if not math.isfinite(raw_start) or not math.isfinite(raw_end):
                stats["adapter_invalid_timestamp_count"] += 1
                stats["adapter_segment_drop_count"] += 1
                continue
            start = max(0.0, raw_start)
            end = raw_end
            if duration > 0:
                start = min(duration, start)
                end = min(duration, end)
            if start != raw_start or end != raw_end:
                stats["adapter_timestamp_clamp_count"] += 1
            if raw_end < raw_start:
                stats["adapter_invalid_timestamp_count"] += 1
            text = str(item.get("text", "")).strip()
            if not text or end <= start:
                stats["adapter_segment_drop_count"] += 1
                continue
            segments.append(BackendSegment(start=start, end=end, text=text))
        return segments, stats

    def transcribe(self, request: BackendRequest) -> BackendResult:
        self._ensure_runtime()
        request_options = dict(request.options)
        safe_retry_without_clips = self._coerce_bool_option(
            request_options.pop("mlx_safe_retry_without_clips", None),
            default=True,
            name="mlx_safe_retry_without_clips",
        )
        diagnostic_details = self._coerce_bool_option(
            request_options.pop("mlx_debug_diagnostics", None),
            default=False,
            name="mlx_debug_diagnostics",
        )
        sampling_seed = self._coerce_sampling_seed(
            request_options.pop("mlx_sampling_seed", None),
        )
        mapping = map_mlx_options(request_options, task=request.task)
        self._warned_ignored_options.update(mapping.ignored)
        options = dict(mapping.options)
        options["language"] = request.language
        options["task"] = request.task
        options["sampling_seed"] = sampling_seed
        options["diagnostic_details"] = diagnostic_details
        sample_count = self._audio_sample_count(request.audio)
        duration_hint = self._audio_duration(request.audio, [])
        initial_uses_clips = self._uses_non_full_clips(
            options.get("clip_timestamps"),
            duration=duration_hint,
        )

        peak_before = 0
        if self._mx is not None:
            try:
                get_peak_memory = getattr(self._mx, "get_peak_memory", None)
                if get_peak_memory is None:
                    get_peak_memory = self._mx.metal.get_peak_memory
                peak_before = int(get_peak_memory())
            except Exception:
                peak_before = 0

        initial_result, initial_elapsed = self._run_runtime(request.audio, options)
        if duration_hint <= 0:
            preview_segments, _preview_stats = self._adapt_segments(initial_result, duration=0.0)
            duration_hint = self._audio_duration(request.audio, preview_segments)
            initial_uses_clips = self._uses_non_full_clips(
                options.get("clip_timestamps"),
                duration=duration_hint,
            )
        quality_sample_count = sample_count or int(round(duration_hint * 16_000))
        initial_anomaly = analyze_runtime_result(initial_result, duration=duration_hint)
        initial_quality = evaluate_candidate_quality(
            initial_result,
            sample_count=quality_sample_count,
            anomaly=initial_anomaly,
        )
        selected_result = initial_result
        selected_anomaly = initial_anomaly
        selected_quality = initial_quality
        selected_result_name = "initial"
        repair_result: dict[str, Any] | None = None
        repair_anomaly: RuntimeAnomaly | None = None
        repair_quality: CandidateQuality | None = None
        repair_elapsed = 0.0
        repair_reason: str | None = None

        if safe_retry_without_clips and initial_anomaly.repair_required:
            repair_reason = initial_anomaly.anomaly_reason

        selection_reason = "initial_acceptable"
        repair_is_distinct = initial_uses_clips or not self._is_fixed_t0(options.get("temperature"))
        if repair_reason is not None and not repair_is_distinct:
            selection_reason = "repair_not_distinct"
        elif repair_reason is not None:
            repair_options = dict(options)
            repair_options.pop("clip_timestamps", None)
            repair_options["temperature"] = 0.0
            repair_result, repair_elapsed = self._run_runtime(request.audio, repair_options)
            repair_anomaly = analyze_runtime_result(repair_result, duration=duration_hint)
            repair_quality = evaluate_candidate_quality(
                repair_result,
                sample_count=quality_sample_count,
                anomaly=repair_anomaly,
            )
            repair_evidence_complete = repair_quality.diagnostics_incomplete_count == 0
            repair_is_strictly_better = repair_quality.comparison_key < initial_quality.comparison_key
            repair_fallback_without_hard_loop_improvement = (
                repair_anomaly.fallback_exhausted
                and repair_quality.hard_loop_severity >= initial_quality.hard_loop_severity
            )
            if (
                repair_evidence_complete
                and repair_is_strictly_better
                and not repair_fallback_without_hard_loop_improvement
            ):
                selected_result = repair_result
                selected_anomaly = repair_anomaly
                selected_quality = repair_quality
                selected_result_name = "repair"
                selection_reason = "repair_strictly_better"
            elif not repair_evidence_complete:
                selection_reason = "repair_evidence_incomplete"
            elif repair_fallback_without_hard_loop_improvement:
                selection_reason = "repair_fallback_without_hard_loop_improvement"
            else:
                selection_reason = "repair_not_strictly_better"
        elif not safe_retry_without_clips and initial_anomaly.anomaly_detected:
            selection_reason = "repair_disabled"
        elif initial_anomaly.diagnostics_incomplete:
            selection_reason = "evidence_incomplete"
        elif initial_anomaly.review_required:
            selection_reason = "soft_review_only"

        safe_retry_count = 1.0 if repair_result is not None else 0.0
        if 1 + int(safe_retry_count) > 2:
            raise AssertionError("MLX candidate policy exceeded two runtime calls")
        selected_not_worse = selected_quality.comparison_key <= initial_quality.comparison_key
        if not selected_not_worse:
            raise AssertionError("selected MLX candidate is worse than initial")

        initial_record = self._anomaly_record(initial_result, initial_anomaly)
        initial_record["quality"] = initial_quality.as_dict()
        initial_record["candidate_disposition"] = (
            "selected_degraded"
            if selected_result_name == "initial" and initial_anomaly.anomaly_detected
            else "selected_review"
            if selected_result_name == "initial" and initial_anomaly.review_required
            else "selected_clean"
            if selected_result_name == "initial"
            else "alternative_selected"
        )
        repair_record: dict[str, Any] | None = None
        if repair_result is not None and repair_anomaly is not None and repair_quality is not None:
            repair_record = self._anomaly_record(repair_result, repair_anomaly)
            repair_record["quality"] = repair_quality.as_dict()
            repair_record["candidate_disposition"] = (
                "selected_degraded"
                if selected_result_name == "repair" and repair_anomaly.anomaly_detected
                else "selected_review"
                if selected_result_name == "repair" and repair_anomaly.review_required
                else "selected_clean"
                if selected_result_name == "repair"
                else "initial_retained"
            )
        result_status = (
            "degraded_unresolved"
            if selected_anomaly.anomaly_detected
            else "review_required"
            if selected_anomaly.review_required
            else "clean"
        )
        diagnostics: dict[str, Any] = {
            "schema_version": 2,
            "sample_count": sample_count,
            "quality_sample_count": quality_sample_count,
            "sampling_seed": sampling_seed,
            "initial_clip_mode": "clips" if initial_uses_clips else "full_chunk",
            "initial": initial_record,
            "repair": repair_record,
            "repair_reason": repair_reason,
            "repair_mode": "full_t0" if repair_result is not None else None,
            "repair_rejected": repair_result is not None and selected_result_name == "initial",
            "safe_retry_count": int(safe_retry_count),
            "selected_result": selected_result_name,
            "selection_reason": selection_reason,
            "selected_quality": selected_quality.as_dict(),
            "selected_not_worse": selected_not_worse,
            "unresolved_anomaly": selected_anomaly.anomaly_detected,
            "soft_review_required": selected_anomaly.review_required,
            "result_status": result_status,
        }

        segments, adapter_stats = self._adapt_segments(selected_result, duration=duration_hint)
        diagnostics["adapter"] = dict(adapter_stats)

        metrics: dict[str, float] = {
            "inference_seconds": initial_elapsed + repair_elapsed,
            "runtime_call_count": 1.0 + safe_retry_count,
            "safe_retry_count": safe_retry_count,
            "initial_fallback_exhausted": float(initial_anomaly.fallback_exhausted),
            "initial_loop_detected": float(initial_anomaly.loop_detection.loop_detected),
            "initial_timestamp_violation_count": float(initial_anomaly.timestamp_violation_count),
            "initial_seek_invariant_failure_count": float(
                initial_anomaly.seek_nonprogress_count + initial_anomaly.repeated_frame_range_count
            ),
            "initial_diagnostics_incomplete": float(initial_anomaly.diagnostics_incomplete),
            "fallback_exhausted": float(selected_anomaly.fallback_exhausted),
            "loop_detected": float(selected_anomaly.loop_detection.loop_detected),
            "timestamp_violation_count": float(selected_anomaly.timestamp_violation_count),
            "seek_invariant_failure_count": float(
                selected_anomaly.seek_nonprogress_count + selected_anomaly.repeated_frame_range_count
            ),
            "diagnostics_incomplete": float(selected_anomaly.diagnostics_incomplete),
            "unresolved_anomaly": float(selected_anomaly.anomaly_detected),
            "soft_review_required": float(selected_anomaly.review_required),
            "selected_not_worse": float(selected_not_worse),
            **{name: float(value) for name, value in adapter_stats.items()},
        }
        if self._mx is not None:
            try:
                get_peak_memory = getattr(self._mx, "get_peak_memory", None)
                if get_peak_memory is None:
                    get_peak_memory = self._mx.metal.get_peak_memory
                peak_after = int(get_peak_memory())
                metrics["peak_memory_bytes"] = float(max(peak_before, peak_after))
            except Exception:
                pass
        if self._warned_ignored_options:
            metrics["ignored_option_count"] = float(len(self._warned_ignored_options))

        return BackendResult(
            segments=segments,
            duration=duration_hint or self._audio_duration(request.audio, segments),
            duration_after_vad=None,
            language=(
                str(selected_result.get("language"))
                if selected_result.get("language") is not None
                else request.language
            ),
            backend="mlx",
            metrics=metrics,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        if self._transcribe_module is not None:
            holder = getattr(self._transcribe_module, "ModelHolder", None)
            if holder is not None:
                holder.model = None
                holder.model_path = None
        if self._mx is not None:
            with suppress(Exception):
                self._mx.clear_cache()
        self._transcribe_fn = None
        self._transcribe_module = None
        self._mx = None
