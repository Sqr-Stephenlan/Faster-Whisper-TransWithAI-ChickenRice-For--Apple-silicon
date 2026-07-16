"""Faster-Whisper/CTranslate2 backend adapter."""

from __future__ import annotations

import time
from typing import Any

from .base import (
    BackendCapabilities,
    BackendRequest,
    BackendResult,
    BackendSegment,
    BackendUnavailableError,
    ModelDescriptor,
)
from .option_mapping import map_ct2_options


class CTranslate2Backend:
    capabilities = BackendCapabilities(
        backend="ct2",
        supports_translate=True,
        supports_transcribe=True,
        supports_word_timestamps=True,
        supports_batching=True,
    )

    def __init__(
        self,
        descriptor: ModelDescriptor,
        *,
        device: str,
        compute_type: str,
        cpu_threads: int = 0,
        enable_batching: bool = False,
        batch_size: int = 0,
        max_batch_size: int = 8,
        model_class: Any | None = None,
        batched_pipeline_class: Any | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self._model_class = model_class
        self._batched_pipeline_class = batched_pipeline_class
        self._model: Any | None = None
        self._batched_model: Any | None = None
        self._model_load_seconds = 0.0

    @property
    def batching_enabled(self) -> bool:
        return self.enable_batching

    def _load_classes(self) -> None:
        if self._model_class is not None and self._batched_pipeline_class is not None:
            return
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except Exception as exc:
            raise BackendUnavailableError(f"Failed to import faster-whisper: {exc}") from exc
        self._model_class = WhisperModel
        self._batched_pipeline_class = BatchedInferencePipeline

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._load_classes()
        assert self._model_class is not None
        started = time.perf_counter()
        self._model = self._model_class(
            str(self.descriptor.path),
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        self._model_load_seconds = time.perf_counter() - started
        if self.enable_batching:
            assert self._batched_pipeline_class is not None
            self._batched_model = self._batched_pipeline_class(model=self._model)

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "oom" in message

    def _run_batched(self, audio: Any, options: dict[str, Any]) -> tuple[Any, Any, int]:
        assert self._batched_model is not None
        batch_size = self.batch_size or self.max_batch_size
        while batch_size >= 1:
            try:
                segments, info = self._batched_model.transcribe(audio, batch_size=batch_size, **options)
                return segments, info, batch_size
            except RuntimeError as exc:
                if not self._is_oom(exc) or batch_size == 1:
                    raise
                batch_size = max(1, int(batch_size * 0.8))
        raise RuntimeError("Failed to find an executable CTranslate2 batch size")

    def transcribe(self, request: BackendRequest) -> BackendResult:
        self._ensure_model()
        assert self._model is not None
        options = map_ct2_options(request.options)
        options["language"] = request.language
        options["task"] = request.task

        started = time.perf_counter()
        actual_batch_size = 0
        if self._batched_model is not None:
            segments_iter, info, actual_batch_size = self._run_batched(request.audio, options)
        else:
            segments_iter, info = self._model.transcribe(request.audio, **options)

        raw_segments = list(segments_iter)
        elapsed = time.perf_counter() - started
        duration = float(
            getattr(
                info,
                "duration",
                max((float(segment.end) for segment in raw_segments), default=0.0),
            )
        )
        segments: list[BackendSegment] = []
        for segment in raw_segments:
            text = str(segment.text).strip()
            start = max(0.0, float(segment.start))
            end = min(duration, max(start, float(segment.end)))
            if text and end > start:
                segments.append(BackendSegment(start=start, end=end, text=text))
        duration_after_vad_raw = getattr(info, "duration_after_vad", None)
        duration_after_vad = float(duration_after_vad_raw) if duration_after_vad_raw is not None else None
        language = getattr(info, "language", request.language)
        metrics = {
            "inference_seconds": elapsed,
            "model_load_seconds": self._model_load_seconds,
        }
        if actual_batch_size:
            metrics["batch_size"] = float(actual_batch_size)
        self._model_load_seconds = 0.0
        return BackendResult(
            segments=segments,
            duration=duration,
            duration_after_vad=duration_after_vad,
            language=str(language) if language is not None else None,
            backend="ct2",
            metrics=metrics,
        )

    def close(self) -> None:
        self._batched_model = None
        self._model = None
