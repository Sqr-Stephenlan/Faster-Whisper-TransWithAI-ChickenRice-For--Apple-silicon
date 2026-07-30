import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    sys.platform == "darwin" and shutil.which("xcrun"),
    "Swift launcher logic tests require macOS and Xcode Command Line Tools",
)
class MacOSSwiftLogicTests(unittest.TestCase):
    def test_launcher_logic_suite(self) -> None:
        completed = subprocess.run(
            [str(ROOT / "scripts" / "test_macos_app_logic.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("LauncherLogicTests: 8 tests passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
