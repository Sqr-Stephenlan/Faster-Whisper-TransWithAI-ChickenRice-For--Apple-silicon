#!/usr/bin/env python3
"""Report backend/profile availability for CLI and macOS UI consumers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faster_whisper_transwithai_chickenrice.backends.factory import probe_backend
from faster_whisper_transwithai_chickenrice.profiles import get_profile
from faster_whisper_transwithai_chickenrice.runtime_assets import validate_profile


def build_report(
    profile_name: str,
    *,
    check_runtime: bool = True,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    profile = get_profile(profile_name)
    backends: dict[str, object] = {}
    common_assets = validate_profile("vad", load_runtime=check_runtime)
    for backend in ("ct2", "mlx"):
        availability = probe_backend(
            profile,
            backend,
            check_runtime=check_runtime,
            verify_hashes=verify_hashes,
        )
        reasons = list(availability.reasons)
        reasons.extend(f"{issue.path}: {issue.message}" for issue in common_assets.issues)
        deduplicated_reasons = list(dict.fromkeys(reasons))
        backends[backend] = {
            "available": availability.available and common_assets.ok,
            "capabilities": {
                "translate": True,
                "transcribe": True,
                "word_timestamps": backend == "ct2",
                "batching": backend == "ct2",
            },
            "device": availability.device,
            "model": {
                "path": str(availability.descriptor.path),
                "profile": availability.descriptor.profile,
                "variant": availability.descriptor.variant,
            },
            "reasons": deduplicated_reasons,
        }
    return {
        "schema_version": 1,
        "profile": profile_name,
        "language": profile.language,
        "task": profile.task,
        "backends": backends,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("translate", "transcribe"), default="translate")
    parser.add_argument("--no-runtime-check", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        args.profile,
        check_runtime=not args.no_runtime_check,
        verify_hashes=not args.skip_hashes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if any(item["available"] for item in report["backends"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
