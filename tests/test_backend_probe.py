import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import backend_probe


def make_availability(
    backend: str,
    *,
    available: bool,
    reasons: tuple[str, ...] = (),
) -> SimpleNamespace:
    variant = "int8" if backend == "ct2" else "fp16"
    device = "cpu" if backend == "ct2" else "gpu"
    return SimpleNamespace(
        available=available,
        device=device,
        reasons=reasons,
        descriptor=SimpleNamespace(
            path=ROOT / "models" / backend / variant,
            profile="translate",
            variant=variant,
        ),
    )


class BackendProbeReportTests(unittest.TestCase):
    def test_translate_report_has_schema_one_and_complete_backend_fields(self) -> None:
        profile = SimpleNamespace(language="ja", task="translate")
        common_assets = SimpleNamespace(ok=True, issues=[])

        with (
            mock.patch.object(backend_probe, "get_profile", return_value=profile),
            mock.patch.object(backend_probe, "validate_profile", return_value=common_assets),
            mock.patch.object(
                backend_probe,
                "probe_backend",
                side_effect=[
                    make_availability("ct2", available=True),
                    make_availability("mlx", available=True),
                ],
            ),
        ):
            report = backend_probe.build_report("translate")

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["profile"], "translate")
        self.assertEqual(report["task"], "translate")
        self.assertEqual(set(report["backends"]), {"ct2", "mlx"})
        for backend_name, item in report["backends"].items():
            with self.subTest(backend=backend_name):
                self.assertEqual(
                    set(item),
                    {"available", "capabilities", "device", "model", "reasons"},
                )
                self.assertEqual(
                    set(item["model"]),
                    {"path", "profile", "variant"},
                )
                self.assertEqual(
                    set(item["capabilities"]),
                    {"translate", "transcribe", "word_timestamps", "batching"},
                )

    def test_all_unavailable_still_prints_valid_json_and_returns_one(self) -> None:
        report = {
            "schema_version": 1,
            "profile": "translate",
            "language": "ja",
            "task": "translate",
            "backends": {
                "ct2": {"available": False},
                "mlx": {"available": False},
            },
        }
        output = io.StringIO()

        with (
            mock.patch.object(backend_probe, "build_report", return_value=report),
            redirect_stdout(output),
        ):
            exit_code = backend_probe.main(["--profile", "translate"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_one_available_backend_returns_zero(self) -> None:
        report = {
            "schema_version": 1,
            "profile": "translate",
            "language": "ja",
            "task": "translate",
            "backends": {
                "ct2": {"available": True},
                "mlx": {"available": False},
            },
        }

        with (
            mock.patch.object(backend_probe, "build_report", return_value=report),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = backend_probe.main(["--profile", "translate"])

        self.assertEqual(exit_code, 0)

    def test_common_vad_issue_makes_both_backends_unavailable(self) -> None:
        issue = SimpleNamespace(path="models/whisper_vad.onnx", message="missing")
        common_assets = SimpleNamespace(ok=False, issues=[issue])
        profile = SimpleNamespace(language="ja", task="translate")

        with (
            mock.patch.object(backend_probe, "get_profile", return_value=profile),
            mock.patch.object(backend_probe, "validate_profile", return_value=common_assets),
            mock.patch.object(
                backend_probe,
                "probe_backend",
                side_effect=[
                    make_availability("ct2", available=True),
                    make_availability("mlx", available=True),
                ],
            ),
        ):
            report = backend_probe.build_report("translate")

        expected_reason = "models/whisper_vad.onnx: missing"
        for backend_name, item in report["backends"].items():
            with self.subTest(backend=backend_name):
                self.assertFalse(item["available"])
                self.assertIn(expected_reason, item["reasons"])


if __name__ == "__main__":
    unittest.main()
