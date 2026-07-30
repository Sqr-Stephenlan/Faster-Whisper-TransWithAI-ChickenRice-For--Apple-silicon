#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! SWIFTC="$(xcrun --find swiftc 2>/dev/null)"; then
  echo "错误：xcrun 无法找到 swiftc。请先安装 Xcode Command Line Tools。" >&2
  exit 1
fi

SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/AITranslateLauncherTests.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT

EXECUTABLE="$BUILD_ROOT/LauncherLogicTests"

"$SWIFTC" \
  -swift-version 5 \
  -parse-as-library \
  -sdk "$SDK_PATH" \
  -target arm64-apple-macos13.0 \
  -framework Foundation \
  "$ROOT/macos_app/LauncherModels.swift" \
  "$ROOT/macos_app/BackendProbeRunner.swift" \
  "$ROOT/tests/LauncherLogicTests.swift" \
  -o "$EXECUTABLE"

"$EXECUTABLE"
