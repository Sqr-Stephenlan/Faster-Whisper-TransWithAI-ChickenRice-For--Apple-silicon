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
    def test_doctor_command_checks_translation_profile(self) -> None:
        script = (ROOT / "检查Mac环境.command").read_text(encoding="utf-8")
        self.assertIn("scripts/macos_doctor.py --mode translate", script)

    def parse(self, mode: str, paths: list[str], extra: list[str] | None = None) -> list[str]:
        argv = ["--mode", mode, "--dry-run", *(extra or []), *paths]
        output = subprocess.run(
            [sys.executable, str(SCRIPTS / "macos_launcher.py"), *argv],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return json.loads(output)

    def parse_completed(
        self,
        paths: list[str],
        extra: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = ["--mode", "translate", "--dry-run", *(extra or []), *paths]
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "macos_launcher.py"), *argv],
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_sub_formats(self, command: list[str], expected: str) -> None:
        index = command.index("--sub_formats")
        self.assertEqual(command[index + 1], expected)

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

    def test_default_subtitle_formats_preserve_existing_behavior(self) -> None:
        command = self.parse("translate", ["/tmp/input.mp3"])

        self.assert_sub_formats(command, "srt,vtt,lrc")

    def test_all_non_empty_subtitle_format_combinations(self) -> None:
        combinations = (
            "srt",
            "vtt",
            "lrc",
            "srt,vtt",
            "srt,lrc",
            "vtt,lrc",
            "srt,vtt,lrc",
        )

        for formats in combinations:
            with self.subTest(formats=formats):
                command = self.parse(
                    "translate",
                    ["/tmp/input.mp3"],
                    ["--sub-formats", formats],
                )
                self.assert_sub_formats(command, formats)

    def test_subtitle_formats_are_canonicalized_to_fixed_order(self) -> None:
        command = self.parse(
            "translate",
            ["/tmp/input.mp3"],
            ["--sub-formats", "lrc,srt"],
        )

        self.assert_sub_formats(command, "srt,lrc")

    def test_subtitle_formats_are_case_insensitive_and_trimmed(self) -> None:
        command = self.parse(
            "translate",
            ["/tmp/input.mp3"],
            ["--sub-formats", " SRT, vtt "],
        )

        self.assert_sub_formats(command, "srt,vtt")

    def test_duplicate_subtitle_formats_are_removed(self) -> None:
        command = self.parse(
            "translate",
            ["/tmp/input.mp3"],
            ["--sub-formats", "lrc,srt,lrc"],
        )

        self.assert_sub_formats(command, "srt,lrc")

    def test_sub_formats_underscore_alias_is_supported(self) -> None:
        command = self.parse(
            "translate",
            ["/tmp/input.mp3"],
            ["--sub_formats", "vtt"],
        )

        self.assert_sub_formats(command, "vtt")

    def test_empty_subtitle_format_value_is_rejected(self) -> None:
        completed = self.parse_completed(["/tmp/input.mp3"], ["--sub-formats", ""])

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("non-empty comma-separated list", completed.stderr)

    def test_empty_subtitle_format_item_is_rejected(self) -> None:
        completed = self.parse_completed(
            ["/tmp/input.mp3"],
            ["--sub-formats", "srt,,lrc"],
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("non-empty comma-separated list", completed.stderr)

    def test_unsupported_subtitle_format_is_rejected(self) -> None:
        completed = self.parse_completed(
            ["/tmp/input.mp3"],
            ["--sub-formats", "txt"],
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unsupported subtitle format(s): txt", completed.stderr)

    def test_subtitle_formats_do_not_change_path_argument_boundaries(self) -> None:
        paths = [
            "/tmp/日语 音频（测试）/第一集.mp3",
            "/tmp/含 单引号'与$符号/第二集.m4a",
        ]
        command = self.parse(
            "translate",
            paths,
            ["--sub-formats", "lrc,srt"],
        )

        self.assert_sub_formats(command, "srt,lrc")
        self.assertEqual(command[-2:], paths)
        self.assertLess(command.index("--sub_formats"), len(command) - len(paths))

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
