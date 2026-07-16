#!/usr/bin/env python3
"""Benchmark a local Whisper backend in one process with cold/warm runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faster_whisper.audio import decode_audio

from faster_whisper_transwithai_chickenrice.backends.base import BackendRequest, BackendResult
from faster_whisper_transwithai_chickenrice.backends.factory import create_backend, select_backend
from faster_whisper_transwithai_chickenrice.profiles import get_profile

SAMPLE_RATE = 16_000


def validate_timeline(result: BackendResult, duration: float) -> list[str]:
    issues: list[str] = []
    previous_end = 0.0
    for index, segment in enumerate(result.segments):
        if segment.start < 0:
            issues.append(f"segment {index} starts before zero")
        if segment.end <= segment.start:
            issues.append(f"segment {index} has a non-positive duration")
        if segment.start < previous_end:
            issues.append(f"segment {index} overlaps the previous segment")
        if segment.end > duration + 0.05:
            issues.append(f"segment {index} exceeds the audio duration")
        previous_end = max(previous_end, segment.end)
    return issues


def benchmark(
    audio_path: Path,
    *,
    backend_name: str,
    profile_name: str,
    repeats: int,
    verify_hashes: bool,
) -> dict[str, Any]:
    profile = get_profile(profile_name)
    selection = select_backend(
        backend_name,
        profile,
        verify_hashes=verify_hashes,
    )

    decode_started = time.perf_counter()
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    decode_seconds = time.perf_counter() - decode_started
    duration = len(audio) / SAMPLE_RATE

    backend = create_backend(
        selection,
        device="cpu",
        compute_type="int8",
        cpu_threads=12,
    )
    runs: list[dict[str, Any]] = []
    try:
        for run_index in range(repeats):
            started = time.perf_counter()
            result = backend.transcribe(
                BackendRequest(
                    audio=audio,
                    language=profile.language,
                    task=profile.task,
                    options={
                        "beam_size": 1,
                        "condition_on_previous_text": False,
                        "word_timestamps": False,
                        "vad_filter": False,
                    },
                )
            )
            elapsed = time.perf_counter() - started
            timeline_issues = validate_timeline(result, duration)
            runs.append(
                {
                    "index": run_index + 1,
                    "kind": "cold" if run_index == 0 else "warm",
                    "elapsed_seconds": elapsed,
                    "realtime_factor": elapsed / duration if duration else None,
                    "segment_count": len(result.segments),
                    "text_characters": sum(len(segment.text) for segment in result.segments),
                    "timeline_issues": timeline_issues,
                    "backend_metrics": result.metrics,
                }
            )
    finally:
        backend.close()

    warm_times = [run["elapsed_seconds"] for run in runs[1:]]
    return {
        "schema_version": 1,
        "scope": "backend-only; FFmpeg decode is measured separately; outer VAD and subtitle writing are excluded",
        "audio": {
            "path": str(audio_path.resolve()),
            "duration_seconds": duration,
            "decode_seconds": decode_seconds,
        },
        "backend": selection.selected,
        "profile": profile.name,
        "model": {
            "path": str(selection.descriptor.path),
            "variant": selection.descriptor.variant,
        },
        "runs": runs,
        "summary": {
            "cold_seconds": runs[0]["elapsed_seconds"],
            "warm_median_seconds": statistics.median(warm_times) if warm_times else None,
            "warm_max_seconds": max(warm_times) if warm_times else None,
            "warm_min_seconds": min(warm_times) if warm_times else None,
            "all_timelines_valid": all(not run["timeline_issues"] for run in runs),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--backend", choices=("ct2", "mlx"), required=True)
    parser.add_argument("--profile", choices=("translate", "transcribe"), default="translate")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")

    report = benchmark(
        args.audio,
        backend_name=args.backend,
        profile_name=args.profile,
        repeats=args.repeats,
        verify_hashes=not args.skip_hashes,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["summary"]["all_timelines_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
