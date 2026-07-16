import hashlib
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
    validate_mlx_model,
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


def make_mlx_model(root: Path, mode: str = "translate") -> Path:
    model_dir = root / "mlx" / mode / "fp16"
    model_dir.mkdir(parents=True)
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights")
    config = {
        "model_type": "whisper",
        "n_audio_state": 1280,
        "n_audio_layer": 32,
        "n_text_layer": 32,
        "n_audio_head": 20,
        "n_text_head": 20,
        "n_vocab": 51865,
    }
    write_json(model_dir / "config.json", config)
    files = {}
    for name in ("model.safetensors", "config.json"):
        digest = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        files[name] = f"sha256:{digest}"
    write_json(
        model_dir / "conversion-manifest.json",
        {
            "source_repo": "owner/source",
            "source_revision": "source-sha",
            "converter_repo": "owner/converter",
            "converter_revision": "converter-sha",
            "mlx_whisper_version": "0.4.3",
            "mlx_version": "0.32.0",
            "dtype": "float16",
            "quantization": None,
            "language": "ja",
            "task": mode,
            "files": files,
        },
    )
    return model_dir


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


class MLXAssetTests(unittest.TestCase):
    def test_valid_fp16_model_and_manifest_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = make_mlx_model(Path(tmp_dir))
            _checked, issues = validate_mlx_model(model_dir)

        self.assertEqual(issues, [])

    def test_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = make_mlx_model(Path(tmp_dir))
            (model_dir / "model.safetensors").write_bytes(b"changed")
            _checked, issues = validate_mlx_model(model_dir)

        self.assertTrue(any("SHA-256 mismatch" in issue.message for issue in issues))

    def test_quantization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = make_mlx_model(Path(tmp_dir))
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            config["quantization"] = {"bits": 8}
            write_json(model_dir / "config.json", config)
            _checked, issues = validate_mlx_model(model_dir, verify_hashes=False)

        self.assertTrue(any("quantization" in issue.message for issue in issues))


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
