#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./dev.sh python scripts/macos_launcher.py --mode translate --backend ct2 "$@"
status=$?
if [ "$status" -ne 0 ] && [ "${TRANSWITHAI_NO_PAUSE:-0}" != "1" ] && [ -t 0 ]; then
  echo
  echo "运行失败（退出码 $status）。日志：$ROOT/latest.log"
  read -r -p "按回车键关闭窗口..." _
fi
exit "$status"
