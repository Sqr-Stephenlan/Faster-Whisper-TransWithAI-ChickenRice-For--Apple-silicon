"""Translate/transcribe profiles and backend-specific local model paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends.base import ModelDescriptor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ProfileDefinition:
    name: str
    language: str
    task: str
    models: dict[str, dict[str, Path]]
    source_repositories: dict[str, str]
    expected_mlx_config: dict[str, object]

    def descriptor(
        self,
        backend: str,
        *,
        variant: str | None = None,
        path_override: str | Path | None = None,
    ) -> ModelDescriptor:
        variants = self.models[backend]
        selected_variant = variant or ("fp16" if backend == "mlx" else "int8")
        if selected_variant not in variants:
            supported = ", ".join(sorted(variants))
            raise ValueError(
                f"Unsupported {backend} model variant '{selected_variant}' for profile "
                f"'{self.name}'. Expected one of: {supported}"
            )
        path = Path(path_override).expanduser().resolve() if path_override else variants[selected_variant]
        return ModelDescriptor(
            backend=backend,
            profile=self.name,
            variant=selected_variant,
            path=path.resolve(),
            source_repo=self.source_repositories.get(backend),
        )


PROFILES = {
    "translate": ProfileDefinition(
        name="translate",
        language="ja",
        task="translate",
        models={
            "ct2": {"int8": MODELS_ROOT / "translate"},
            "mlx": {"fp16": MODELS_ROOT / "mlx" / "translate" / "fp16"},
        },
        source_repositories={
            "ct2": "chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2",
            "mlx": "chickenrice0721/whisper-large-v2-translate-zh-v0.2-st",
        },
        expected_mlx_config={
            "model_type": "whisper",
            "n_audio_state": 1280,
            "n_audio_layer": 32,
            "n_text_layer": 32,
            "n_audio_head": 20,
            "n_text_head": 20,
            "n_vocab": 51865,
        },
    ),
    "transcribe": ProfileDefinition(
        name="transcribe",
        language="ja",
        task="transcribe",
        models={
            "ct2": {"int8": MODELS_ROOT / "transcribe"},
            "mlx": {"fp16": MODELS_ROOT / "mlx" / "transcribe" / "fp16"},
        },
        source_repositories={
            "ct2": "TransWithAI/whisper-ja-1.5B-ct2",
            "mlx": "TransWithAI/whisper-ja-1.5B",
        },
        expected_mlx_config={
            "model_type": "whisper",
            "n_audio_state": 1280,
            "n_audio_layer": 32,
            "n_text_layer": 32,
            "n_audio_head": 20,
            "n_text_head": 20,
            "n_vocab": 51865,
        },
    ),
}


def get_profile(name: str) -> ProfileDefinition:
    try:
        return PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unsupported inference profile '{name}'. Expected one of: {supported}") from exc
