#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./dev.sh python scripts/macos_doctor.py --mode translate
status=$?
if [ "${TRANSWITHAI_NO_PAUSE:-0}" != "1" ] && [ -t 0 ]; then
  echo
  read -r -p "检查结束（退出码 $status），按回车键关闭窗口..." _
fi
exit "$status"
