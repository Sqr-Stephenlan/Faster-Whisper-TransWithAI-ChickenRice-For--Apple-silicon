"""Backend preflight, automatic selection, and lifecycle factory."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..profiles import ProfileDefinition
from ..runtime_assets import validate_ct2_model, validate_mlx_model
from .base import BackendUnavailableError, ModelDescriptor, ModelNotInstalledError, WhisperBackend
from .ct2 import CTranslate2Backend
from .mlx import MLXBackend

BACKENDS = ("ct2", "mlx")


@dataclass(frozen=True)
class BackendAvailability:
    backend: str
    available: bool
    descriptor: ModelDescriptor
    reasons: tuple[str, ...] = ()
    device: str | None = None


@dataclass(frozen=True)
class BackendSelection:
    requested: str
    selected: str
    descriptor: ModelDescriptor
    fallback_reason: str | None = None
    device: str | None = None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _metal_preflight(timeout: float = 30.0) -> tuple[bool, str]:
    code = "import mlx.core as mx; mx.set_default_device(mx.gpu); mx.eval(mx.array([1.0])); print(mx.default_device())"
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (completed.stdout.strip() or completed.stderr.strip()).splitlines()
    message = detail[-1] if detail else f"process exited with {completed.returncode}"
    return completed.returncode == 0 and "gpu" in message.lower(), message


def _looks_like_backend(path: Path) -> str | None:
    if (path / "model.bin").is_file():
        return "ct2"
    if any((path / name).is_file() for name in ("model.safetensors", "weights.safetensors", "weights.npz")):
        return "mlx"
    return None


def probe_backend(
    profile: ProfileDefinition,
    backend: str,
    *,
    variant: str | None = None,
    model_path: str | Path | None = None,
    check_runtime: bool = True,
    verify_hashes: bool = True,
) -> BackendAvailability:
    descriptor = profile.descriptor(backend, variant=variant, path_override=model_path)
    reasons: list[str] = []
    device: str | None = None

    if backend == "ct2":
        _checked, issues = validate_ct2_model(descriptor.path)
        reasons.extend(f"{issue.path}: {issue.message}" for issue in issues)
        if check_runtime:
            if not _module_available("ctranslate2"):
                reasons.append("ctranslate2 is not installed")
            if not _module_available("faster_whisper"):
                reasons.append("faster-whisper is not installed")
        device = "cpu"
    elif backend == "mlx":
        if platform.machine() != "arm64":
            reasons.append(f"MLX requires Apple Silicon arm64, got {platform.machine()}")
        _checked, issues = validate_mlx_model(
            descriptor.path,
            verify_hashes=verify_hashes,
            expected_task=profile.task,
            expected_language=profile.language,
            expected_config=profile.expected_mlx_config,
        )
        reasons.extend(f"{issue.path}: {issue.message}" for issue in issues)
        if check_runtime:
            if not _module_available("mlx"):
                reasons.append("mlx is not installed")
            if not _module_available("mlx_whisper"):
                reasons.append("mlx-whisper runtime is not installed")
            if not _module_available("faster_whisper.audio"):
                reasons.append("faster-whisper shared audio decoder is not installed")
            if not reasons:
                metal_ok, metal_detail = _metal_preflight()
                if not metal_ok:
                    reasons.append(f"Metal GPU preflight failed: {metal_detail}")
                else:
                    device = metal_detail
        elif not reasons:
            device = "gpu"
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return BackendAvailability(
        backend=backend,
        available=not reasons,
        descriptor=descriptor,
        reasons=tuple(reasons),
        device=device,
    )


def select_backend(
    requested: str,
    profile: ProfileDefinition,
    *,
    variant: str | None = None,
    model_path: str | Path | None = None,
    check_runtime: bool = True,
    verify_hashes: bool = True,
) -> BackendSelection:
    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", *BACKENDS}:
        raise ValueError(f"Unsupported backend '{requested}'")

    override_backend = _looks_like_backend(Path(model_path).expanduser()) if model_path else None
    if requested == "auto":
        candidates = [override_backend] if override_backend else ["mlx", "ct2"]
    else:
        candidates = [requested]

    failures: list[str] = []
    for backend in candidates:
        backend_variant = variant
        if variant is None or (backend == "ct2" and variant == "fp16") or (backend == "mlx" and variant == "int8"):
            backend_variant = None
        backend_model_path = model_path if override_backend in {None, backend} else None
        availability = probe_backend(
            profile,
            backend,
            variant=backend_variant,
            model_path=backend_model_path,
            check_runtime=check_runtime,
            verify_hashes=verify_hashes,
        )
        if availability.available:
            return BackendSelection(
                requested=requested,
                selected=backend,
                descriptor=availability.descriptor,
                fallback_reason="; ".join(failures) if failures else None,
                device=availability.device,
            )
        failures.append(f"{backend}: {'; '.join(availability.reasons)}")

    message = " | ".join(failures) or "no backend candidates were available"
    if any("model directory is missing" in failure for failure in failures):
        raise ModelNotInstalledError(
            f"Model for profile '{profile.name}' is not installed for backend '{requested}': {message}"
        )
    raise BackendUnavailableError(f"Backend preflight failed before inference: {message}")


def create_backend(
    selection: BackendSelection,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 0,
    enable_batching: bool = False,
    batch_size: int = 0,
    max_batch_size: int = 8,
    backend_overrides: dict[str, Any] | None = None,
) -> WhisperBackend:
    overrides = dict(backend_overrides or {})
    if selection.selected == "ct2":
        return CTranslate2Backend(
            selection.descriptor,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            enable_batching=enable_batching,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            **overrides,
        )
    if selection.selected == "mlx":
        return MLXBackend(selection.descriptor, **overrides)
    raise ValueError(f"Unsupported selected backend: {selection.selected}")
