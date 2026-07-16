import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from faster_whisper_transwithai_chickenrice.backends.base import BackendResult, BackendSegment
from scripts.benchmark_backends import validate_timeline


class BenchmarkTimelineTests(unittest.TestCase):
    def test_accepts_monotonic_segments(self) -> None:
        result = BackendResult(
            segments=[
                BackendSegment(0.0, 1.0, "a"),
                BackendSegment(1.0, 2.0, "b"),
            ],
            duration=2.0,
            duration_after_vad=None,
            language="ja",
            backend="fake",
        )

        self.assertEqual(validate_timeline(result, 2.0), [])

    def test_reports_overlap_and_out_of_bounds(self) -> None:
        result = BackendResult(
            segments=[
                BackendSegment(0.0, 1.5, "a"),
                BackendSegment(1.0, 3.0, "b"),
            ],
            duration=2.0,
            duration_after_vad=None,
            language="ja",
            backend="fake",
        )

        issues = validate_timeline(result, 2.0)

        self.assertTrue(any("overlaps" in issue for issue in issues))
        self.assertTrue(any("exceeds" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
