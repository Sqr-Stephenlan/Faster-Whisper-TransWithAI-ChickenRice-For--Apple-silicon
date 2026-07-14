#!/usr/bin/env python3
"""Non-interactive macOS environment and local model diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faster_whisper_transwithai_chickenrice.runtime_assets import format_report, validate_profile


def build_result(mode: str) -> dict[str, object]:
    report = validate_profile(mode)
    packages: dict[str, str | None] = {}
    for name in ("numpy", "ctranslate2", "faster-whisper", "onnxruntime", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "ok": report.ok,
        "mode": mode,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.executable,
        "packages": packages,
        "assets": report.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("vad", "translate", "transcribe", "all"), default="all")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result = build_result(args.mode)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Platform: {result['platform']}")
        print(f"Architecture: {result['architecture']}")
        print(f"Python: {result['python']}")
        report = validate_profile(args.mode)
        print(format_report(report))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
