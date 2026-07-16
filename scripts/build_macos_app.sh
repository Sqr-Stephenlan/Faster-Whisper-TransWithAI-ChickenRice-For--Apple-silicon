#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_SOURCES=(
  "$ROOT/macos_app/AITranslateLauncher.swift"
  "$ROOT/macos_app/LauncherModels.swift"
  "$ROOT/macos_app/BackendProbeRunner.swift"
)
ICON_SOURCE="$ROOT/macos_app/AppIcon.svg"
ICON_RENDERER_SOURCE="$ROOT/macos_app/AppIconRenderer.swift"
APP_DIR="$ROOT/AI语音翻译.app"

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "错误：未找到 $command_name。请先安装 Xcode Command Line Tools。" >&2
    exit 1
  fi
}

require_command xcrun
require_command plutil
require_command codesign
require_command tiff2icns

if ! SWIFTC="$(xcrun --find swiftc 2>/dev/null)"; then
  echo "错误：xcrun 无法找到 swiftc。请先安装 Xcode Command Line Tools。" >&2
  exit 1
fi

for source in "${APP_SOURCES[@]}"; do
  if [[ ! -f "$source" ]]; then
    echo "错误：未找到 Swift 源码：$source" >&2
    exit 1
  fi
done

if [[ ! -f "$ICON_SOURCE" ]]; then
  echo "错误：未找到图标源文件：$ICON_SOURCE" >&2
  exit 1
fi

if [[ ! -f "$ICON_RENDERER_SOURCE" ]]; then
  echo "错误：未找到图标渲染器：$ICON_RENDERER_SOURCE" >&2
  exit 1
fi

SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
BUILD_ROOT="$(mktemp -d "$ROOT/.AITranslateLauncher.build.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT

BUILD_APP="$BUILD_ROOT/AI语音翻译.app"
CONTENTS_DIR="$BUILD_APP/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
EXECUTABLE="$MACOS_DIR/AITranslateLauncher"
ICON_RENDERER="$BUILD_ROOT/AppIconRenderer"
ICONSET_DIR="$BUILD_ROOT/AppIcon.iconset"
ICON_TIFF="$BUILD_ROOT/AppIcon.tiff"
ICON_FILE="$RESOURCES_DIR/AppIcon.icns"
PLIST="$CONTENTS_DIR/Info.plist"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

"$SWIFTC" \
  -swift-version 5 \
  -parse-as-library \
  -sdk "$SDK_PATH" \
  -target arm64-apple-macos13.0 \
  -framework AppKit \
  -framework SwiftUI \
  "${APP_SOURCES[@]}" \
  -o "$EXECUTABLE"

chmod 755 "$EXECUTABLE"

"$SWIFTC" \
  -swift-version 5 \
  -sdk "$SDK_PATH" \
  -target arm64-apple-macos13.0 \
  -framework AppKit \
  -framework ImageIO \
  -framework UniformTypeIdentifiers \
  "$ICON_RENDERER_SOURCE" \
  -o "$ICON_RENDERER"

"$ICON_RENDERER" "$ICON_SOURCE" "$ICONSET_DIR" "$ICON_TIFF"
tiff2icns "$ICON_TIFF" "$ICON_FILE"
"$ICON_RENDERER" --add-retina "$ICONSET_DIR" "$ICON_FILE"

cat >"$PLIST" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>zh_CN</string>
    <key>CFBundleDisplayName</key>
    <string>AI语音翻译</string>
    <key>CFBundleExecutable</key>
    <string>AITranslateLauncher</string>
    <key>CFBundleIdentifier</key>
    <string>local.ai-translate.launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon.icns</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>AI语音翻译</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1.0</string>
    <key>CFBundleVersion</key>
    <string>3</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>用于在 Terminal 中启动本地语音翻译任务。</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
PLIST

plutil -lint "$PLIST"
codesign --force --deep --sign - "$BUILD_APP"
codesign --verify --deep --strict --verbose=2 "$BUILD_APP"

rm -rf "$APP_DIR"
mv "$BUILD_APP" "$APP_DIR"

echo
echo "已生成：$APP_DIR"
echo "使用方法：双击 AI语音翻译.app，拖入音视频或文件夹，然后点击“开始翻译”。"
