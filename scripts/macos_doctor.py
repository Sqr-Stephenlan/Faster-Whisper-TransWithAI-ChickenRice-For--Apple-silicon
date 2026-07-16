#!/usr/bin/env python3
"""Non-interactive macOS environment and local model diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faster_whisper_transwithai_chickenrice.backends.factory import probe_backend
from faster_whisper_transwithai_chickenrice.profiles import get_profile
from faster_whisper_transwithai_chickenrice.runtime_assets import format_report, validate_profile


def build_result(mode: str, backend: str = "ct2") -> dict[str, Any]:
    report = validate_profile(mode, backend=backend)
    packages: dict[str, str | None] = {}
    package_names = ["numpy", "onnxruntime", "transformers"]
    package_names.extend(
        ("mlx", "mlx-whisper-runtime-local") if backend == "mlx" else ("ctranslate2", "faster-whisper")
    )
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    backend_probe: dict[str, Any] | None = None
    if mode in {"translate", "transcribe"}:
        availability = probe_backend(get_profile(mode), backend, verify_hashes=False)
        backend_probe = {
            "available": availability.available,
            "device": availability.device,
            "model": str(availability.descriptor.path),
            "reasons": list(availability.reasons),
        }
    return {
        "ok": report.ok and (backend_probe is None or bool(backend_probe["available"])),
        "mode": mode,
        "backend": backend,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.executable,
        "packages": packages,
        "assets": report.to_dict(),
        "backend_probe": backend_probe,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("vad", "translate", "transcribe", "all"), default="all")
    parser.add_argument("--backend", choices=("ct2", "mlx"), default="ct2")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result = build_result(args.mode, args.backend)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Platform: {result['platform']}")
        print(f"Architecture: {result['architecture']}")
        print(f"Python: {result['python']}")
        print(f"Backend: {result['backend']}")
        report = validate_profile(args.mode, backend=args.backend)
        print(format_report(report))
        if result["backend_probe"] is not None:
            probe = result["backend_probe"]
            if probe["available"]:
                print(f"Backend device: {probe['device']}")
            else:
                print("Backend preflight failed:")
                for reason in probe["reasons"]:
                    print(f"- {reason}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
