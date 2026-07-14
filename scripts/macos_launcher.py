#!/usr/bin/env python3
"""Build and run the fixed Apple Silicon CPU inference command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = ROOT / "infer.py"
MODEL_DIRS = {
    "translate": ROOT / "models" / "translate",
    "transcribe": ROOT / "models" / "transcribe",
}
DEFAULT_AUDIO_SUFFIXES = "mp3,wav,flac,m4a,aac,ogg,wma,mp4,mkv,avi,mov,webm,flv,wmv"
DEFAULT_SUB_FORMATS = "srt,vtt,lrc"


def choose_files() -> list[str] | None:
    """Open Finder's multi-file chooser; return None when the user cancels."""
    script = """
try
    set selectedFiles to choose file with prompt "选择要翻译或转录的音视频文件" with multiple selections allowed
    set output to ""
    repeat with selectedFile in selectedFiles
        set output to output & POSIX path of selectedFile & linefeed
    end repeat
    return output
on error number -128
    return "__CANCELLED__"
end try
"""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"macOS file chooser failed: {result.stderr.strip() or result.stdout.strip()}")
    output = result.stdout.strip()
    if output == "__CANCELLED__":
        return None
    return [line for line in output.splitlines() if line]


def build_infer_argv(args: argparse.Namespace, paths: list[str]) -> list[str]:
    argv = [
        sys.executable,
        str(INFER_SCRIPT),
        "--model_name_or_path",
        str(MODEL_DIRS[args.mode]),
        "--task",
        args.mode,
        "--device",
        "cpu",
        "--compute_type",
        "int8",
        "--cpu_threads",
        str(args.cpu_threads),
        "--vad_threads",
        str(args.vad_threads),
        "--audio_suffixes",
        DEFAULT_AUDIO_SUFFIXES,
        "--sub_formats",
        DEFAULT_SUB_FORMATS,
    ]
    if args.output_dir:
        argv.extend(["--output_dir", str(Path(args.output_dir).expanduser())])
    if args.overwrite:
        argv.append("--overwrite")
    argv.extend(paths)
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("translate", "transcribe"), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the inference argv as JSON without running it")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--vad-threads", type=int, default=4)
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    if args.cpu_threads < 0 or args.vad_threads < 0:
        parser.error("thread counts must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = list(args.paths)
    if not paths and not args.dry_run:
        selected = choose_files()
        if selected is None:
            print("已取消文件选择，未启动推理。")
            return 0
        paths = selected

    command = build_infer_argv(args, paths)
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False))
        return 0

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
