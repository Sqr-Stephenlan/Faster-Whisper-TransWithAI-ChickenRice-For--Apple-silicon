"""MLX/Metal Whisper backend adapter."""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from .base import (
    BackendCapabilities,
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

    def transcribe(self, request: BackendRequest) -> BackendResult:
        self._ensure_runtime()
        mapping = map_mlx_options(request.options, task=request.task)
        self._warned_ignored_options.update(mapping.ignored)
        options = dict(mapping.options)
        options["language"] = request.language
        options["task"] = request.task
        transcribe_fn = self._transcribe_fn
        if transcribe_fn is None:
            raise BackendUnavailableError("MLX transcribe function was not initialized")

        peak_before = 0
        if self._mx is not None:
            try:
                get_peak_memory = getattr(self._mx, "get_peak_memory", None)
                if get_peak_memory is None:
                    get_peak_memory = self._mx.metal.get_peak_memory
                peak_before = int(get_peak_memory())
            except Exception:
                peak_before = 0

        started = time.perf_counter()
        result = transcribe_fn(
            request.audio,
            path_or_hf_repo=str(self.descriptor.path),
            **options,
        )
        elapsed = time.perf_counter() - started

        duration_hint = self._audio_duration(request.audio, [])
        segments: list[BackendSegment] = []
        for item in result.get("segments", []):
            text = str(item.get("text", "")).strip()
            start = max(0.0, float(item.get("start", 0.0)))
            end = max(start, float(item.get("end", start)))
            if duration_hint > 0:
                end = min(duration_hint, end)
            if text and end > start:
                segments.append(BackendSegment(start=start, end=end, text=text))

        metrics = {"inference_seconds": elapsed}
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
            language=str(result.get("language")) if result.get("language") is not None else request.language,
            backend="mlx",
            metrics=metrics,
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
