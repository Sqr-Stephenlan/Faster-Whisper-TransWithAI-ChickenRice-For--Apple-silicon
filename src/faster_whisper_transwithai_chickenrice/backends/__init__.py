"""Backend-neutral Whisper inference adapters."""

from .base import (
    BackendCapabilities,
    BackendConfigurationError,
    BackendError,
    BackendRequest,
    BackendResult,
    BackendSegment,
    BackendUnavailableError,
    ModelDescriptor,
    ModelNotInstalledError,
    UnsupportedBackendOptionError,
    WhisperBackend,
)

__all__ = [
    "BackendCapabilities",
    "BackendConfigurationError",
    "BackendError",
    "BackendRequest",
    "BackendResult",
    "BackendSegment",
    "BackendUnavailableError",
    "ModelDescriptor",
    "ModelNotInstalledError",
    "UnsupportedBackendOptionError",
    "WhisperBackend",
]
