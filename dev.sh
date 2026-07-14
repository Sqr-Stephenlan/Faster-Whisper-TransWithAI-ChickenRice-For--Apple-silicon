#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PY="$VENV_DIR/bin/python"

find_system_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No system Python found. Install Python 3 before bootstrapping." >&2
  return 1
}

doctor() {
  echo "Project root: $ROOT"
  echo "Venv dir: $VENV_DIR"
  if [ -x "$PY" ]; then
    "$PY" - <<'PY'
import importlib.metadata
import platform
import sys

import ctranslate2
import onnxruntime

packages = (
    "numpy",
    "librosa",
    "ctranslate2",
    "faster-whisper",
    "onnxruntime",
    "transformers",
    "pyjson5",
)

print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"Python: {sys.executable}")
print(f"Version: {platform.python_version()}")
print(f"Prefix: {sys.prefix}")
for package in packages:
    print(f"Package {package}: {importlib.metadata.version(package)}")
print("CTranslate2 CPU compute types: " + ",".join(ctranslate2.get_supported_compute_types("cpu")))
print("ONNX Runtime providers: " + ",".join(onnxruntime.get_available_providers()))
PY
  elif [ -d "$VENV_DIR" ]; then
    echo "Venv status: present but $PY is not executable"
    return 1
  else
    echo "Venv status: missing"
    return 1
  fi

  for marker in uv.lock requirements-macos.txt requirements.txt pyproject.toml poetry.lock Pipfile; do
    if [ -f "$marker" ]; then
      echo "Found: $marker"
    fi
  done
}

bootstrap() {
  if [ -x "$PY" ]; then
    echo "Existing virtual environment found at $VENV_DIR"
  elif [ -d "$VENV_DIR" ]; then
    echo "$VENV_DIR exists but $PY is not executable; refusing to overwrite it." >&2
    exit 1
  elif [ -f "uv.lock" ] && command -v uv >/dev/null 2>&1; then
    echo "Bootstrapping with uv sync"
    uv sync
    doctor
    return 0
  elif [ -f ".python-version" ] && command -v uv >/dev/null 2>&1; then
    PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"
    echo "Creating $VENV_DIR with Python $PYTHON_VERSION via uv"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  else
    SYSTEM_PY="$(find_system_python)"
    "$SYSTEM_PY" -m venv "$VENV_DIR"
  fi

  "$PY" -m ensurepip --upgrade
  "$PY" -m pip install --upgrade pip

  if [ "$(uname -s)" = "Darwin" ] && [ -f "requirements-macos.txt" ]; then
    echo "Installing requirements-macos.txt with constraints-macos-arm64.txt"
    "$PY" -m pip install -r requirements-macos.txt -c constraints-macos-arm64.txt
  elif [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt"
    "$PY" -m pip install -r requirements.txt
  elif [ -f "pyproject.toml" ] && grep -q '^\[project\]' pyproject.toml; then
    echo "Installing the project declared in pyproject.toml"
    "$PY" -m pip install -e .
  else
    echo "No installable dependency metadata found; prepared $VENV_DIR only."
  fi

  doctor
}

ensure_venv() {
  if [ -x "$PY" ]; then
    return 0
  fi
  echo "No usable virtual environment found at $VENV_DIR." >&2
  echo "Run './dev.sh bootstrap' after confirming environment setup is desired." >&2
  exit 1
}

cmd="${1:-doctor}"
case "$cmd" in
  doctor)
    doctor
    ;;
  bootstrap)
    bootstrap
    ;;
  python)
    shift
    ensure_venv
    exec "$PY" "$@"
    ;;
  pip)
    shift
    ensure_venv
    exec "$PY" -m pip "$@"
    ;;
  pytest)
    shift
    ensure_venv
    exec "$PY" -m pytest "$@"
    ;;
  mypy)
    shift
    ensure_venv
    exec "$PY" -m mypy "$@"
    ;;
  ruff)
    shift
    ensure_venv
    exec "$PY" -m ruff "$@"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: ./dev.sh {doctor|bootstrap|python|pip|pytest|mypy|ruff} ..." >&2
    exit 2
    ;;
esac
