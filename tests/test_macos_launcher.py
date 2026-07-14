import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import macos_launcher


class MacOSLauncherTests(unittest.TestCase):
    def parse(self, mode: str, paths: list[str], extra: list[str] | None = None) -> list[str]:
        argv = ["--mode", mode, "--dry-run", *(extra or []), *paths]
        output = subprocess.run(
            [sys.executable, str(SCRIPTS / "macos_launcher.py"), *argv],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return json.loads(output)

    def test_translate_argv_preserves_unicode_spaces_parentheses_and_multiple_paths(self) -> None:
        paths = [
            "/tmp/sample.wav",
            "/tmp/a b/sample file.mp3",
            "/tmp/日语 音频（测试）/第 1 集.m4a",
        ]
        command = self.parse("translate", paths)

        self.assertEqual(command[-3:], paths)
        self.assertIn(str(ROOT / "models" / "translate"), command)
        self.assertEqual(command[command.index("--task") + 1], "translate")
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(command[command.index("--compute_type") + 1], "int8")
        self.assertEqual(command[command.index("--cpu_threads") + 1], "12")
        self.assertEqual(command[command.index("--vad_threads") + 1], "4")

    def test_transcribe_uses_separate_model_and_output_directory(self) -> None:
        output_dir = "/tmp/字幕 输出（中文）"
        command = self.parse(
            "transcribe",
            ["/tmp/input folder"],
            ["--output-dir", output_dir],
        )

        self.assertIn(str(ROOT / "models" / "transcribe"), command)
        self.assertEqual(command[command.index("--task") + 1], "transcribe")
        self.assertEqual(command[command.index("--output_dir") + 1], output_dir)
        self.assertEqual(command[-1], "/tmp/input folder")

    def test_dry_run_without_paths_does_not_open_finder(self) -> None:
        with mock.patch.object(macos_launcher, "choose_files") as chooser:
            self.assertEqual(macos_launcher.main(["--mode", "translate", "--dry-run"]), 0)
        chooser.assert_not_called()

    def test_cancelled_finder_selection_is_a_clean_exit(self) -> None:
        with mock.patch.object(macos_launcher, "choose_files", return_value=None):
            self.assertEqual(macos_launcher.main(["--mode", "translate"]), 0)

    def test_selected_paths_are_passed_as_individual_arguments(self) -> None:
        selected = ["/tmp/one file.wav", "/tmp/第二集.mp3"]
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(macos_launcher, "choose_files", return_value=selected),
            mock.patch.object(macos_launcher.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(macos_launcher.main(["--mode", "transcribe"]), 0)

        command = run.call_args.args[0]
        self.assertEqual(command[-2:], selected)
        self.assertNotIsInstance(command, str)


if __name__ == "__main__":
    unittest.main()
