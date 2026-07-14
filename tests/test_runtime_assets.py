import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.runtime_assets import (
    validate_ct2_model,
    validate_profile,
    validate_vad_assets,
)


def write_json(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value)
    path.write_text(text, encoding="utf-8")


def make_common_assets(root: Path) -> None:
    (root / "whisper_vad.onnx").write_bytes(b"onnx")
    write_json(root / "whisper_vad_metadata.json", {"model": "vad"})
    write_json(root / "whisper-base" / "preprocessor_config.json", {"feature_size": 80})
    write_json(root / "whisper-base" / "config.json", {"model_type": "whisper"})


def make_ct2_model(root: Path, mode: str) -> None:
    model_dir = root / mode
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"weights")
    write_json(model_dir / "config.json", {"model_type": "Whisper"})
    write_json(model_dir / "tokenizer.json", {"version": "1.0"})


class CTranslate2AssetTests(unittest.TestCase):
    def test_missing_model_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _checked, issues = validate_ct2_model(Path(tmp_dir) / "missing")
        self.assertIn("model directory is missing", issues[0].message)

    def test_config_only_is_not_a_complete_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            write_json(model_dir / "config.json", {})
            _checked, issues = validate_ct2_model(model_dir)
        self.assertTrue(any("model weights" in issue.message for issue in issues))
        self.assertTrue(any("tokenizer" in issue.message for issue in issues))

    def test_empty_model_bin_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "model.bin").touch()
            write_json(model_dir / "config.json", {})
            write_json(model_dir / "vocabulary.json", {})
            _checked, issues = validate_ct2_model(model_dir)
        self.assertTrue(any("empty" in issue.message for issue in issues))


class VadAssetTests(unittest.TestCase):
    def test_invalid_metadata_json_fails_before_session_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "whisper_vad.onnx").write_bytes(b"onnx")
            write_json(root / "whisper_vad_metadata.json", "not json")
            _checked, issues, providers = validate_vad_assets(root)
        self.assertFalse(providers)
        self.assertTrue(any("invalid JSON" in issue.message for issue in issues))

    def test_onnx_session_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "whisper_vad.onnx").write_bytes(b"onnx")
            write_json(root / "whisper_vad_metadata.json", {})
            fake_ort = types.SimpleNamespace(InferenceSession=mock.Mock(side_effect=RuntimeError("bad graph")))
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                _checked, issues, providers = validate_vad_assets(root)
        self.assertFalse(providers)
        self.assertTrue(any("bad graph" in issue.message for issue in issues))


class ProfileAssetTests(unittest.TestCase):
    def test_profiles_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            make_common_assets(root)
            make_ct2_model(root, "translate")

            translate = validate_profile("translate", models_root=root, load_runtime=False)
            transcribe = validate_profile("transcribe", models_root=root, load_runtime=False)

        self.assertTrue(translate.ok)
        self.assertFalse(transcribe.ok)
        self.assertTrue(any("transcribe" in issue.path for issue in transcribe.issues))


if __name__ == "__main__":
    unittest.main()
