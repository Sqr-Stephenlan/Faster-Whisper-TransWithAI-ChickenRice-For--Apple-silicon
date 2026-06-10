import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path

sys.modules.setdefault("pyjson5", types.SimpleNamespace(decode_io=json.load))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.infer import Inference


def make_args(config_path: Path, task: str | None = None) -> Namespace:
    return Namespace(
        generation_config=str(config_path),
        task=task,
        vad_threshold=None,
        vad_min_speech_duration_ms=None,
        vad_min_silence_duration_ms=None,
        vad_speech_pad_ms=None,
        merge_segments=None,
        merge_max_gap_ms=None,
        merge_max_duration_ms=None,
    )


class WhisperTaskConfigTests(unittest.TestCase):
    def load_config(self, config_text: str, task: str | None = None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "generation_config.json5"
            config_path.write_text(config_text, encoding="utf-8")

            inference = Inference.__new__(Inference)
            config, _segment_merge_options = inference._load_generation_config(make_args(config_path, task=task))
            return config

    def test_uses_task_from_generation_config(self) -> None:
        config = self.load_config('{"task": "transcribe"}')

        self.assertEqual(config["task"], "transcribe")

    def test_cli_task_overrides_generation_config(self) -> None:
        config = self.load_config('{"task": "transcribe"}', task="translate")

        self.assertEqual(config["task"], "translate")

    def test_cli_task_takes_precedence_over_invalid_config_task(self) -> None:
        config = self.load_config('{"task": "caption"}', task="transcribe")

        self.assertEqual(config["task"], "transcribe")

    def test_invalid_config_task_fails_without_cli_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Whisper task"):
            self.load_config('{"task": "caption"}')

    def test_defaults_to_translate_without_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "missing.json5"
            inference = Inference.__new__(Inference)
            config, _segment_merge_options = inference._load_generation_config(make_args(config_path))

        self.assertEqual(config["task"], "translate")


if __name__ == "__main__":
    unittest.main()
