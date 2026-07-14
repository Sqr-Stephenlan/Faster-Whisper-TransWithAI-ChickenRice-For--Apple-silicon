"""Strict local runtime asset validation shared by inference and macOS tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models"
MODEL_DIRECTORIES = {
    "translate": "translate",
    "transcribe": "transcribe",
}
PROFILE_REPOSITORIES = {
    "translate": "chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2",
    "transcribe": "TransWithAI/whisper-ja-1.5B-ct2",
}
VOCABULARY_FILES = ("tokenizer.json", "vocabulary.json", "vocab.json")


@dataclass(frozen=True)
class AssetIssue:
    path: str
    message: str


@dataclass(frozen=True)
class AssetReport:
    profile: str
    ok: bool
    checked_paths: tuple[str, ...]
    issues: tuple[AssetIssue, ...]
    vad_providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json_object(path: Path, issues: list[AssetIssue]) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(AssetIssue(str(path), "missing required JSON file"))
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(AssetIssue(str(path), f"invalid JSON: {exc}"))
        return None
    if not isinstance(value, dict):
        issues.append(AssetIssue(str(path), "JSON root must be an object"))
        return None
    return value


def validate_ct2_model(model_dir: Path) -> tuple[list[str], list[AssetIssue]]:
    checked = [str(model_dir)]
    issues: list[AssetIssue] = []
    if not model_dir.is_dir():
        return checked, [AssetIssue(str(model_dir), "model directory is missing")]

    model_bin = model_dir / "model.bin"
    checked.append(str(model_bin))
    if not model_bin.is_file():
        issues.append(AssetIssue(str(model_bin), "missing CTranslate2 model weights"))
    elif model_bin.stat().st_size == 0:
        issues.append(AssetIssue(str(model_bin), "model weights file is empty"))

    config_path = model_dir / "config.json"
    checked.append(str(config_path))
    _read_json_object(config_path, issues)

    vocabulary_paths = [model_dir / name for name in VOCABULARY_FILES]
    checked.extend(str(path) for path in vocabulary_paths)
    if not any(path.is_file() and path.stat().st_size > 0 for path in vocabulary_paths):
        issues.append(
            AssetIssue(
                str(model_dir),
                "missing non-empty tokenizer.json, vocabulary.json, or vocab.json",
            )
        )
    return checked, issues


def validate_vad_assets(models_root: Path, *, create_session: bool = True) -> tuple[list[str], list[AssetIssue], list[str]]:
    model_path = models_root / "whisper_vad.onnx"
    metadata_path = models_root / "whisper_vad_metadata.json"
    checked = [str(model_path), str(metadata_path)]
    issues: list[AssetIssue] = []
    providers: list[str] = []

    if not model_path.is_file():
        issues.append(AssetIssue(str(model_path), "missing VAD ONNX model"))
    elif model_path.stat().st_size == 0:
        issues.append(AssetIssue(str(model_path), "VAD ONNX model is empty"))
    _read_json_object(metadata_path, issues)

    if create_session and not issues:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            providers = list(session.get_providers())
            if providers != ["CPUExecutionProvider"]:
                issues.append(
                    AssetIssue(str(model_path), f"VAD session must use CPUExecutionProvider only, got {providers}")
                )
        except Exception as exc:
            issues.append(AssetIssue(str(model_path), f"could not create CPU ONNX session: {exc}"))
    return checked, issues, providers


def validate_feature_extractor(models_root: Path, *, load: bool = True) -> tuple[list[str], list[AssetIssue]]:
    model_dir = models_root / "whisper-base"
    preprocessor_path = model_dir / "preprocessor_config.json"
    config_path = model_dir / "config.json"
    checked = [str(model_dir), str(preprocessor_path), str(config_path)]
    issues: list[AssetIssue] = []

    _read_json_object(preprocessor_path, issues)
    _read_json_object(config_path, issues)
    if load and not issues:
        try:
            from transformers import WhisperFeatureExtractor

            WhisperFeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
        except Exception as exc:
            issues.append(AssetIssue(str(model_dir), f"could not load local WhisperFeatureExtractor: {exc}"))
    return checked, issues


def validate_profile(
    profile: str,
    *,
    models_root: Path = DEFAULT_MODELS_ROOT,
    load_runtime: bool = True,
) -> AssetReport:
    if profile not in {"vad", "translate", "transcribe", "all"}:
        raise ValueError(f"Unsupported model profile: {profile}")

    checked: list[str] = []
    issues: list[AssetIssue] = []
    vad_checked, vad_issues, providers = validate_vad_assets(models_root, create_session=load_runtime)
    feature_checked, feature_issues = validate_feature_extractor(models_root, load=load_runtime)
    checked.extend(vad_checked)
    checked.extend(feature_checked)
    issues.extend(vad_issues)
    issues.extend(feature_issues)

    modes = ("translate", "transcribe") if profile == "all" else ((profile,) if profile != "vad" else ())
    for mode in modes:
        model_checked, model_issues = validate_ct2_model(models_root / MODEL_DIRECTORIES[mode])
        checked.extend(model_checked)
        issues.extend(model_issues)

    return AssetReport(
        profile=profile,
        ok=not issues,
        checked_paths=tuple(checked),
        issues=tuple(issues),
        vad_providers=tuple(providers),
    )


def format_report(report: AssetReport) -> str:
    if report.ok:
        return f"Model profile '{report.profile}' is ready."
    lines = [f"Model profile '{report.profile}' is not ready:"]
    lines.extend(f"- {issue.path}: {issue.message}" for issue in report.issues)
    return "\n".join(lines)
