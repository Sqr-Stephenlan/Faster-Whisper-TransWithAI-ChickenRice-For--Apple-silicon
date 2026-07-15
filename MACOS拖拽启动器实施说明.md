# macOS 拖拽启动器实施说明

## 实施结果

本次在不修改推理核心与现有翻译参数的前提下，新增了原生 macOS 图形启动器。App 固定放在项目根目录，负责接收文件和文件夹、预检普通文件后缀，并通过 Terminal 调用现有 `运行(翻译)(CPU).command`。

功能节点提交：`7bce9e7 feat: add macOS drag-and-drop launcher`

## 交付文件

```text
macos_app/AITranslateLauncher.swift
scripts/build_macos_app.sh
AI语音翻译.app/
├── Contents/Info.plist
├── Contents/MacOS/AITranslateLauncher
└── Contents/_CodeSignature/CodeResources
```

同时更新或新增以下文档：

```text
README.md
MACOS翻译操作说明.md
AI语音翻译App简明使用说明.md
MACOS拖拽启动器实施说明.md
```

## 实现边界

- `LauncherView` 使用 SwiftUI 构建约 520 × 360 的固定窗口。
- `DropTargetNSView` 使用 AppKit `.fileURL` pasteboard 类型接收 Finder 文件 URL。
- `NSOpenPanel` 同时支持文件、文件夹与多选。
- `LauncherViewModel` 负责去重、支持后缀过滤、环境快检和启动状态。
- `TerminalLauncher` 将项目根目录、现有 `.command` 路径和所有输入路径作为独立参数传给 `/usr/bin/osascript`。
- AppleScript 对每个参数使用 `quoted form of`，避免空格、中文、括号、单引号或 shell 元字符破坏命令。
- GUI 不重复实现推理参数、目录递归、字幕输出或覆盖策略。
- 未修改 `infer.py`、`scripts/macos_launcher.py` 和 `运行(翻译)(CPU).command`。

## 构建方式

```bash
./scripts/build_macos_app.sh
```

构建脚本会检查 `xcrun`、`swiftc`、`plutil` 和 `codesign`，使用 macOS SDK 编译 `arm64-apple-macos13.0` 二进制，生成 `Info.plist`，执行 ad-hoc 签名并验证签名。

## 自动验证记录

实施日期：2026-07-15。

| 检查 | 结果 |
| --- | --- |
| `./scripts/build_macos_app.sh` | 通过 |
| `plutil -lint AI语音翻译.app/Contents/Info.plist` | 通过 |
| `codesign --verify --deep --strict --verbose=2 AI语音翻译.app` | 通过 |
| 主二进制存在且可执行 | 通过 |
| 主二进制架构 | `arm64` |
| `CFBundleExecutable` 与二进制名一致 | 通过 |
| `bash -n scripts/build_macos_app.sh` | 通过 |
| `git diff --check` | 通过 |
| `./dev.sh pytest` | 42 项全部通过 |

## 交互验收记录

已实际验证：

- App 正常显示标题、拖拽区、选择计数和操作按钮；
- 空列表时“开始翻译”和“清除”禁用；
- 点击区域可选择单文件、多文件、文件夹及混合项目；
- 不支持文件会被忽略，并在混合选择时显示数量提示；
- 路径包含空格、中文、括号和单引号时可正常加入；
- 重复选择同一路径后数量不增加；
- “清除”恢复空列表；
- 点击“开始翻译”后 App 退出，Terminal 成功运行现有 `.command`；
- `latest.log` 中收到的带单引号完整路径未被拆分；
- 三种字幕已存在时维持跳过行为；
- App 副本放在 `/tmp` 时显示项目环境缺失提示，不会错误启动。

直接 Finder 跨应用拖拽曾通过 Computer Use 尝试，但该自动化环境无法把拖动事件投递到另一应用窗口，因此未作为已通过项。源码使用标准 AppKit 文件 URL 拖放接口，仍建议用户用实际鼠标补验一次“从 Finder 拖入文件”和“从 Finder 拖入文件夹”。

未提交私人测试媒体，也未执行真实音频模型推理。Terminal 启动验收使用 `/tmp` 中的空媒体占位文件及预先存在的 SRT、VTT、LRC，使现有流程在加载模型前安全跳过。

## 简单使用方法

1. 保持 `AI语音翻译.app` 位于项目根目录。
2. 双击 App，拖入音视频或文件夹。
3. 点击“开始翻译”。
4. 在 Terminal 中查看状态。
5. 在源文件旁查找 SRT、VTT、LRC 字幕。

更多说明见 `AI语音翻译App简明使用说明.md`。
