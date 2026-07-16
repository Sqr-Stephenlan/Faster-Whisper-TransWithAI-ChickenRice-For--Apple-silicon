"""Project-owned backend contracts and standard error types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class BackendError(RuntimeError):
    """Base class for project-owned inference backend failures."""


class BackendUnavailableError(BackendError):
    """Raised when a requested backend cannot run before inference starts."""


class ModelNotInstalledError(BackendUnavailableError):
    """Raised when the selected profile does not have a complete local model."""


class BackendConfigurationError(BackendError):
    """Raised when backend options are invalid or incompatible."""


class UnsupportedBackendOptionError(BackendConfigurationError):
    """Raised when an option cannot be represented by the selected backend."""


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    supports_translate: bool
    supports_transcribe: bool
    supports_word_timestamps: bool
    supports_batching: bool


@dataclass(frozen=True)
class ModelDescriptor:
    backend: str
    profile: str
    variant: str
    path: Path
    source_repo: str | None = None
    source_revision: str | None = None


@dataclass(frozen=True)
class BackendRequest:
    audio: str | Any
    language: str
    task: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class BackendResult:
    segments: list[BackendSegment]
    duration: float
    duration_after_vad: float | None
    language: str | None
    backend: str
    metrics: dict[str, float] = field(default_factory=dict)


class WhisperBackend(Protocol):
    capabilities: BackendCapabilities
    descriptor: ModelDescriptor

    def transcribe(self, request: BackendRequest) -> BackendResult: ...

    def close(self) -> None: ...
