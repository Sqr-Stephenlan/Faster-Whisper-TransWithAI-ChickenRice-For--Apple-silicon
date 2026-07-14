#!/usr/bin/env python3
"""
Download required model files from Hugging Face repositories
ONNX VAD model is always downloaded, additional models can be specified via HuggingFace repo path
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faster_whisper_transwithai_chickenrice.runtime_assets import (
    DEFAULT_MODELS_ROOT,
    MODEL_DIRECTORIES,
    PROFILE_REPOSITORIES,
    format_report,
    validate_ct2_model,
    validate_feature_extractor,
    validate_profile,
    validate_vad_assets,
)


# Detect if the environment supports Unicode/emoji
def can_use_unicode():
    """Check if the current environment supports Unicode output"""
    # If we're in a CI environment, be conservative and use ASCII
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        # Captured CI logs may not advertise their actual output encoding.
        return False

    # Check if UTF-8 is explicitly set
    if os.environ.get("PYTHONIOENCODING", "").lower().startswith("utf"):
        return True

    try:
        "✓".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Define symbols based on Unicode support
USE_UNICODE = can_use_unicode()

if USE_UNICODE:
    # Unicode/emoji symbols
    CHECKMARK = "✓"
    CROSS = "✗"
    DOWNLOAD = "⬇"
    PACKAGE = "📦"
    SEARCH = "🔍"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠"
else:
    # ASCII fallback symbols
    CHECKMARK = "[OK]"
    CROSS = "[X]"
    DOWNLOAD = "[DOWNLOADING]"
    PACKAGE = "[PACKAGE]"
    SEARCH = "[SEARCH]"
    SUCCESS = "[SUCCESS]"
    ERROR = "[ERROR]"
    WARNING = "[WARNING]"

DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_MAX_RETRY_DELAY_SECONDS = 30.0

HF_HOST = "https://huggingface.co/"
HF_MIRROR_HOST = "https://hf-mirror.com/"
MODELS_ROOT = DEFAULT_MODELS_ROOT


def _mirror_url(url: str) -> str | None:
    """Return the hf-mirror.com equivalent of a huggingface.co URL, if applicable."""
    if url.startswith(HF_HOST):
        return HF_MIRROR_HOST + url[len(HF_HOST) :]
    return None


def _get_429_retry_delay(response: requests.Response, retry_index: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, IndexError, OverflowError):
                delay = 0.0

        if delay > 0:
            return min(delay, DOWNLOAD_MAX_RETRY_DELAY_SECONDS)

    exponential_delay = DOWNLOAD_RETRY_BACKOFF_SECONDS * (2**retry_index)
    return min(exponential_delay, DOWNLOAD_MAX_RETRY_DELAY_SECONDS)


def _download_from_url(url: str, dest_path: Path, session: requests.Session) -> bool:
    """Download a single file from one URL, retrying on HTTP 429. Returns True on success."""
    try:
        print(f"  {DOWNLOAD} Downloading {dest_path.name}...", end=" ")

        response = None
        for retry_index in range(DOWNLOAD_MAX_RETRIES + 1):
            response = session.get(url, stream=True, timeout=30)
            if response.status_code != 429 or retry_index == DOWNLOAD_MAX_RETRIES:
                response.raise_for_status()
                break

            retry_delay = _get_429_retry_delay(response, retry_index)
            response.close()
            print(
                f"\r  {WARNING} Rate limited while downloading {dest_path.name}; "
                f"retrying in {retry_delay:.1f}s ({retry_index + 1}/{DOWNLOAD_MAX_RETRIES})"
            )
            time.sleep(retry_delay)
            print(f"  {DOWNLOAD} Downloading {dest_path.name}...", end=" ")

        if response is None:
            raise RuntimeError("Download did not start")

        # Get file size
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress
        downloaded = 0
        last_reported_progress = -1
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        # Only update display every 10%
                        progress_milestone = int(progress // 10) * 10
                        if progress_milestone > last_reported_progress:
                            print(f"\r  {DOWNLOAD} Downloading {dest_path.name}... {progress_milestone}%", end="")
                            last_reported_progress = progress_milestone

        print(f"\r  {CHECKMARK} Downloaded {dest_path.name} ({downloaded / (1024 * 1024):.1f} MB)")
        return True

    except Exception as e:
        print(f"\r  {CROSS} Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_file(
    url: str,
    dest_path: Path,
    session: requests.Session | None = None,
    *,
    force: bool = False,
) -> bool:
    """Download a file with progress indicator.

    Tries the given URL first. If it is a huggingface.co URL and the download
    fails (including after exhausting the HTTP 429 retries), the whole download
    is retried against the hf-mirror.com mirror for another round of attempts.
    """
    if session is None:
        session = requests.Session()

    # Create parent directory if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip only complete-looking files. Empty files are always replaced.
    if dest_path.exists() and dest_path.stat().st_size > 0 and not force:
        print(f"  {CHECKMARK} {dest_path.name} already exists")
        return True
    if dest_path.exists():
        dest_path.unlink()

    # Try the primary host first, then fall back to the hf-mirror.com mirror.
    candidate_urls = [url]
    mirror_url = _mirror_url(url)
    if mirror_url:
        candidate_urls.append(mirror_url)

    for attempt_index, candidate_url in enumerate(candidate_urls):
        if attempt_index > 0:
            print(f"  {WARNING} Falling back to hf-mirror.com for {dest_path.name}...")
        if _download_from_url(candidate_url, dest_path, session):
            return True

    return False


def get_hf_api_files(repo_id: str) -> list[str]:
    """Get list of files from Hugging Face repo using API"""
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        files = response.json()
        return [f["path"] for f in files if f["type"] == "file"]
    except Exception as e:
        print(f"Warning: Could not fetch file list from API: {e}")
        return []


def download_hf_model(repo_id: str, target_dir: str | None = None, *, force: bool = False):
    """Download model files from any HuggingFace repository"""
    base_url = f"https://huggingface.co/{repo_id}/resolve/main/"

    # Determine target directory
    if target_dir:
        models_dir = MODELS_ROOT / target_dir
    else:
        # Use repository name as default subdirectory
        repo_name = repo_id.split("/")[-1]
        models_dir = MODELS_ROOT / repo_name

    print(f"\n{PACKAGE} Downloading model from {repo_id}")
    print(f"   Target directory: {models_dir}")

    # Essential file extensions to download for transformer/whisper models
    essential_extensions = [".json", ".bin", ".txt", ".onnx", ".safetensors", ".model"]

    # Try to get full file list from API
    api_files = get_hf_api_files(repo_id)
    if api_files:
        # Filter for essential files
        files_to_download = [f for f in api_files if any(f.endswith(ext) for ext in essential_extensions)]
        print(f"  Found {len(files_to_download)} files in repository")
    else:
        print(f"  {CROSS} Could not obtain the repository file manifest; refusing a partial download")
        return False

    session = requests.Session()
    session.headers.update({"User-Agent": "Faster-Whisper-TransWithAI-macOS/1.0"})

    success_count = 0
    for filename in files_to_download:
        url = urljoin(base_url, filename)
        dest_path = models_dir / filename
        if download_file(url, dest_path, session, force=force):
            success_count += 1

    print(f"  {CHECKMARK} Downloaded {success_count}/{len(files_to_download)} files")
    return bool(files_to_download) and success_count == len(files_to_download)


def download_vad_model(*, force: bool = False):
    """Download VAD ONNX model files (always required)"""
    repo_id = "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main/"
    models_dir = MODELS_ROOT

    print(f"\n{PACKAGE} Downloading VAD ONNX model from {repo_id}")

    # Files to download (renamed to match existing structure)
    files = [
        ("model.onnx", "whisper_vad.onnx"),  # Download as model.onnx, save as whisper_vad.onnx
        (
            "model_metadata.json",
            "whisper_vad_metadata.json",
        ),  # Download as model_metadata.json, save as whisper_vad_metadata.json
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Faster-Whisper-TransWithAI-macOS/1.0"})

    success_count = 0
    for source_name, dest_name in files:
        url = urljoin(base_url, source_name)
        dest_path = models_dir / dest_name
        if download_file(url, dest_path, session, force=force):
            success_count += 1

    print(f"  {CHECKMARK} Downloaded {success_count}/{len(files)} files")
    return success_count == len(files)


def download_whisper_base_for_feature_extractor(*, force: bool = False):
    """Download whisper-base model files specifically for feature extractor (offline usage)"""
    repo_id = "openai/whisper-base"
    models_dir = MODELS_ROOT / "whisper-base"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main/"

    print(f"\n{PACKAGE} Downloading whisper-base for feature extractor (offline usage)")

    # Check if files already exist from main models folder
    existing_models_dir = MODELS_ROOT
    if existing_models_dir.exists():
        # Files we can copy from existing models folder if available
        files_to_copy = [
            "preprocessor_config.json",
            "config.json",
            "tokenizer.json",
            "vocab.json",
        ]

        copied = 0
        models_dir.mkdir(parents=True, exist_ok=True)
        for filename in files_to_copy:
            src = existing_models_dir / filename
            dest = models_dir / filename
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)
                print(f"  {CHECKMARK} Copied {filename} from existing models folder")
                copied += 1
            elif dest.exists():
                print(f"  {CHECKMARK} {filename} already exists")
                copied += 1

        if copied >= 2:  # At minimum we need preprocessor_config.json and config.json
            print(f"  {CHECKMARK} Used existing files for whisper-base")
            return True

    # Download ONLY the specific files needed for feature extractor
    # We don't need model weights (.bin, .safetensors) for feature extraction
    required_files = [
        "preprocessor_config.json",  # Required for feature extractor
        "config.json",  # Required for configuration
        "tokenizer.json",  # Optional but useful for tokenization
        "vocab.json",  # Optional but useful for vocabulary
    ]

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading feature extractor files from {repo_id}...")

    session = requests.Session()
    session.headers.update({"User-Agent": "Faster-Whisper-TransWithAI-macOS/1.0"})

    success_count = 0
    for filename in required_files:
        url = urljoin(base_url, filename)
        dest_path = models_dir / filename
        if download_file(url, dest_path, session, force=force):
            success_count += 1

    print(f"  {CHECKMARK} Downloaded {success_count}/{len(required_files)} feature extractor files")
    return success_count >= 2  # At minimum we need the two required files


def verify_whisper_base_feature_extractor():
    """Strictly verify local whisper-base feature extractor files."""
    print(f"\n{SEARCH} Verifying whisper-base feature extractor files...")
    _checked, issues = validate_feature_extractor(MODELS_ROOT)
    for issue in issues:
        print(f"  {CROSS} {issue.path}: {issue.message}")
    return not issues


def verify_vad_model():
    """Strictly verify VAD JSON and a CPU-only ONNX Runtime session."""
    print(f"\n{SEARCH} Verifying VAD model files...")
    _checked, issues, providers = validate_vad_assets(MODELS_ROOT)
    for issue in issues:
        print(f"  {CROSS} {issue.path}: {issue.message}")
    if providers:
        print(f"  {CHECKMARK} ONNX providers: {','.join(providers)}")
    return not issues


def verify_hf_model(repo_id: str, target_dir: str | None = None):
    """Strictly verify a local CTranslate2 model directory."""
    if target_dir:
        models_dir = MODELS_ROOT / target_dir
    else:
        repo_name = repo_id.split("/")[-1]
        models_dir = MODELS_ROOT / repo_name

    print(f"\n{SEARCH} Verifying CTranslate2 model files in {models_dir}...")
    _checked, issues = validate_ct2_model(models_dir)
    for issue in issues:
        print(f"  {CROSS} {issue.path}: {issue.message}")
    return not issues


def main():
    """Main download function"""
    parser = argparse.ArgumentParser(
        description="Model Downloader for Faster Whisper Custom VAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
    # Download VAD model and whisper-base (both required for offline usage)

  %(prog)s --skip-whisper-base
    # Download only VAD model, skip whisper-base (not recommended)

  %(prog)s --hf-model chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2
    # Download VAD, whisper-base, and Chickenrice Whisper model

  %(prog)s --hf-model TransWithAI/whisper-ja-1.5B-ct2
    # Download VAD, whisper-base, and TransWithAI Japanese transcribe bf16 model

  %(prog)s --hf-model TransWithAI/whisper-ja-1.5B-ct2 --target-dir .
    # Download the main model directly into models/ for source launchers.

  %(prog)s --hf-model openai/whisper-large-v3 --target-dir whisper-v3
    # Download VAD, whisper-base, and Whisper v3 to specific directory

  %(prog)s --force --hf-model myusername/my-custom-model
    # Force re-download everything including VAD, whisper-base, and custom model
        """,
    )

    parser.add_argument(
        "--hf-model", type=str, help='HuggingFace repository path to download (e.g., "openai/whisper-large-v3")'
    )

    parser.add_argument(
        "--target-dir",
        type=str,
        help="Target subdirectory name in models/ for the HuggingFace model (defaults to repo name)",
    )

    parser.add_argument("--force", action="store_true", help="Force re-download even if models already exist")

    parser.add_argument(
        "--profile",
        choices=("vad", "translate", "transcribe", "all"),
        help="Download or verify the fixed macOS model layout",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate local files without creating directories or making network requests",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; return a non-zero exit status on any failure",
    )

    parser.add_argument(
        "--skip-vad", action="store_true", help="Skip downloading VAD model (not recommended, for testing only)"
    )

    parser.add_argument(
        "--skip-whisper-base",
        action="store_true",
        help="Skip downloading whisper-base model for feature extractor (not recommended)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Model Downloader for Faster Whisper Custom VAD")
    print("=" * 60)

    if args.verify_only:
        if args.profile:
            report = validate_profile(args.profile)
            print(format_report(report))
            return 0 if report.ok else 1

        results = []
        if not args.skip_vad:
            results.append(verify_vad_model())
        if not args.skip_whisper_base:
            results.append(verify_whisper_base_feature_extractor())
        if args.hf_model:
            results.append(verify_hf_model(args.hf_model, args.target_dir))
        if not results:
            parser.error("--verify-only requires --profile or at least one asset selection")
        return 0 if all(results) else 1

    MODELS_ROOT.mkdir(exist_ok=True)

    if args.profile:
        modes = (
            ("translate", "transcribe") if args.profile == "all" else ((args.profile,) if args.profile != "vad" else ())
        )
        downloads_ok = download_vad_model(force=args.force)
        downloads_ok = download_whisper_base_for_feature_extractor(force=args.force) and downloads_ok
        for mode in modes:
            downloads_ok = (
                download_hf_model(
                    PROFILE_REPOSITORIES[mode],
                    MODEL_DIRECTORIES[mode],
                    force=args.force,
                )
                and downloads_ok
            )
        report = validate_profile(args.profile)
        print("\n" + format_report(report))
        return 0 if downloads_ok and report.ok else 1

    # Check if VAD model already exists
    if not args.force and not args.skip_vad and verify_vad_model():
        print(f"\n{CHECKMARK} VAD model files already present")
        vad_exists = True
    else:
        vad_exists = False

    # Check if whisper-base feature extractor already exists
    whisper_base_exists = False
    if not args.skip_whisper_base and not args.force and verify_whisper_base_feature_extractor():
        print(f"\n{CHECKMARK} Whisper-base feature extractor files already present")
        whisper_base_exists = True

    # Check if HF model already exists (if specified)
    hf_exists = False
    if args.hf_model and not args.force and verify_hf_model(args.hf_model, args.target_dir):
        print(f"\n{CHECKMARK} Model {args.hf_model} already present")
        hf_exists = True

    # If everything exists and no force flag, ask user
    all_exists = vad_exists and (not args.hf_model or hf_exists) and (args.skip_whisper_base or whisper_base_exists)
    if all_exists and not args.force:
        if args.non_interactive:
            print("Skipping download.")
            return 0
        response = input("\nAll required models are present. Re-download? (y/N): ").strip().lower()
        if response != "y":
            print("Skipping download.")
            return 0

    # Download models

    # Always download VAD model (unless explicitly skipped)
    if not args.skip_vad:
        if not download_vad_model(force=args.force):
            print(f"{ERROR} Error: VAD model is required and could not be downloaded")
            return 1
    else:
        print(f"\n{WARNING} Skipping VAD model download (not recommended)")

    # Download whisper-base feature extractor (unless explicitly skipped)
    if not args.skip_whisper_base:
        if not download_whisper_base_for_feature_extractor(force=args.force):
            print(f"{ERROR} Whisper-base feature extractor could not be downloaded completely")
            return 1
    else:
        print(f"\n{WARNING} Skipping whisper-base download (not recommended for offline usage)")

    # Download HuggingFace model if specified
    if args.hf_model and not download_hf_model(args.hf_model, args.target_dir, force=args.force):
        print(f"{ERROR} Model {args.hf_model} could not be downloaded completely")
        return 1

    # Final verification
    print("\n" + "=" * 60)

    # Verify VAD model
    if not args.skip_vad:
        if verify_vad_model():
            print(f"\n{SUCCESS} VAD model downloaded successfully!")
        else:
            print(f"\n{ERROR} Critical: VAD model is missing. Cannot proceed without it.")
            return 1

    # Verify whisper-base feature extractor (unless skipped)
    if not args.skip_whisper_base:
        if verify_whisper_base_feature_extractor():
            print(f"\n{SUCCESS} Whisper-base feature extractor downloaded successfully!")
        else:
            print(f"\n{ERROR} Whisper-base feature extractor is incomplete.")
            return 1

    # Verify HF model if specified
    if args.hf_model:
        if verify_hf_model(args.hf_model, args.target_dir):
            print(f"\n{SUCCESS} Model {args.hf_model} downloaded successfully!")
        else:
            print(f"\n{ERROR} Model {args.hf_model} is incomplete.")
            return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{ERROR} Error: {e}")
        sys.exit(1)
