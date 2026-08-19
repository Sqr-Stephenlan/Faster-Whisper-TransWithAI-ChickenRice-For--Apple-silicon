#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_APP="$SCRIPT_DIR/AI语音翻译.app"
SOURCE_ICON="$SOURCE_APP/Contents/Resources/AppIcon.icns"
DESTINATION_DIR="$HOME/Applications"
APP_NAME="AI语音翻译.app"
BUNDLE_IDENTIFIER="local.ai-translate.launchpad-launcher"
REFRESH_LAUNCHPAD=1

usage() {
  cat <<'EOF'
用法：
  ./创建启动台入口.command [--destination 目录] [--no-refresh]

选项：
  --destination 目录  将壳应用放到指定目录（默认：~/Applications）
  --no-refresh        不注册应用且不重启 Dock，主要用于测试
  -h, --help          显示帮助
EOF
}

fail() {
  echo "错误：$*" >&2
  exit 1
}

require_command() {
  [[ -x "$1" ]] || fail "未找到系统命令：$1"
}

escape_applescript_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --destination)
      [[ $# -ge 2 ]] || fail "--destination 后需要一个目录路径"
      DESTINATION_DIR="$2"
      shift 2
      ;;
    --no-refresh)
      REFRESH_LAUNCHPAD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "未知选项：$1（使用 --help 查看帮助）"
      ;;
  esac
done

[[ -d "$SOURCE_APP" ]] || fail "未找到原应用：$SOURCE_APP"
[[ -f "$SOURCE_APP/Contents/Info.plist" ]] || fail "原应用结构不完整：$SOURCE_APP"

require_command /usr/bin/osacompile
require_command /usr/bin/plutil
require_command /usr/bin/codesign

mkdir -p "$DESTINATION_DIR"
DESTINATION_DIR="$(cd -P "$DESTINATION_DIR" && pwd)"
DESTINATION_APP="$DESTINATION_DIR/$APP_NAME"

if [[ -e "$DESTINATION_APP" || -L "$DESTINATION_APP" ]]; then
  existing_identifier="$(
    /usr/bin/plutil -extract CFBundleIdentifier raw \
      "$DESTINATION_APP/Contents/Info.plist" 2>/dev/null || true
  )"
  if [[ "$existing_identifier" != "$BUNDLE_IDENTIFIER" ]]; then
    fail "目标已存在且不是本脚本创建的启动器：$DESTINATION_APP"
  fi
fi

BUILD_ROOT="$(mktemp -d "$DESTINATION_DIR/.AITranslateLaunchpad.XXXXXX")"
BUILD_APP="$BUILD_ROOT/$APP_NAME"
BACKUP_APP="$BUILD_ROOT/previous.app"
trap 'rm -rf "$BUILD_ROOT"' EXIT

escaped_source_app="$(escape_applescript_string "$SOURCE_APP")"
COMPILE_LOG="$BUILD_ROOT/osacompile.log"
if ! /usr/bin/osacompile \
  -o "$BUILD_APP" \
  -e "property targetApp : \"$escaped_source_app\"" \
  -e 'on run' \
  -e 'do shell script "/usr/bin/open " & quoted form of targetApp' \
  -e 'end run' >"$COMPILE_LOG" 2>&1; then
  /bin/cat "$COMPILE_LOG" >&2
  fail "无法编译 AppleScript 启动器"
fi

PLIST="$BUILD_APP/Contents/Info.plist"
/usr/bin/plutil -replace CFBundleIdentifier -string "$BUNDLE_IDENTIFIER" "$PLIST"
/usr/bin/plutil -replace CFBundleName -string "AI语音翻译" "$PLIST"
/usr/bin/plutil -replace CFBundleDisplayName -string "AI语音翻译" "$PLIST"
/usr/bin/plutil -replace CFBundleShortVersionString -string "1.0" "$PLIST"
/usr/bin/plutil -replace CFBundleVersion -string "1" "$PLIST"
/usr/bin/plutil -remove CFBundleIconName "$PLIST" 2>/dev/null || true

if [[ -f "$SOURCE_ICON" ]]; then
  /bin/cp "$SOURCE_ICON" "$BUILD_APP/Contents/Resources/AppIcon.icns"
  /usr/bin/plutil -replace CFBundleIconFile -string "AppIcon.icns" "$PLIST"
fi

/usr/bin/plutil -lint "$PLIST" >/dev/null
SIGN_LOG="$BUILD_ROOT/codesign.log"
if ! /usr/bin/codesign --force --deep --sign - "$BUILD_APP" >"$SIGN_LOG" 2>&1; then
  /bin/cat "$SIGN_LOG" >&2
  fail "无法签名 AppleScript 启动器"
fi
/usr/bin/codesign --verify --deep --strict "$BUILD_APP"

if [[ -e "$DESTINATION_APP" || -L "$DESTINATION_APP" ]]; then
  /bin/mv "$DESTINATION_APP" "$BACKUP_APP"
fi

if ! /bin/mv "$BUILD_APP" "$DESTINATION_APP"; then
  if [[ -e "$BACKUP_APP" || -L "$BACKUP_APP" ]]; then
    /bin/mv "$BACKUP_APP" "$DESTINATION_APP"
  fi
  fail "无法安装启动器：$DESTINATION_APP"
fi

if [[ "$REFRESH_LAUNCHPAD" -eq 1 ]]; then
  LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
  if [[ -x "$LSREGISTER" ]]; then
    "$LSREGISTER" -f "$DESTINATION_APP" >/dev/null 2>&1 || true
  fi
  /usr/bin/killall Dock >/dev/null 2>&1 || true
fi

echo
echo "已创建启动台入口：$DESTINATION_APP"
echo "原应用仍保留在：$SOURCE_APP"
echo "如果以后移动了整个项目目录，请重新运行本脚本更新入口。"
