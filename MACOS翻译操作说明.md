# macOS 翻译操作说明

## 最简单的用法

1. 在 Finder 打开本项目文件夹。
2. 首次使用可双击 `检查Mac环境.command`，确认翻译模型就绪。
3. 双击项目根目录的 `AI语音翻译.app`。
4. 将一个或多个日语音频、视频或整个文件夹拖进窗口；也可点击拖拽区域选择。
5. 确认列表后点击“开始翻译”。
6. 在自动打开的 Terminal 中查看状态。完成后会在原文件旁生成 `.srt`、`.vtt`、`.lrc` 三种字幕。

支持单文件、多文件，以及包含空格、中文或括号的路径。已有三种字幕时会自动跳过。

> `AI语音翻译.app` 必须与项目目录一起使用，不要单独移入 `/Applications`。首次运行时 macOS 可能提示允许 App 控制 Terminal，这是正常的本地启动授权。

## 重新构建 App

如果 `AI语音翻译.app` 缺失，或 `macos_app/AITranslateLauncher.swift` 有更新，在项目根目录运行：

```bash
./scripts/build_macos_app.sh
```

脚本会重新生成根目录 App、校验 `Info.plist` 并执行本地 ad-hoc 签名。

## 原有 Finder 入口

不使用拖拽窗口时，仍可双击 `运行(翻译)(CPU).command`，然后在文件选择器中选择一个或多个音视频文件。

## 从终端运行

处理一个文件：

```bash
cd "/Users/stephenlan/Documents/AI translate"
'./运行(翻译)(CPU).command' "/完整路径/日语音频.mp3"
```

指定输出文件夹并覆盖旧字幕：

```bash
'./运行(翻译)(CPU).command' \
  --output-dir "/完整路径/字幕输出" \
  --overwrite \
  "/完整路径/日语音频.mp3"
```

处理整个文件夹：

```bash
'./运行(翻译)(CPU).command' "/完整路径/音频文件夹"
```

## 当前配置

- 主模型：`models/translate/`
- 推理设备：CPU
- 计算精度：`int8`
- 默认线程：`cpu_threads=12`、`vad_threads=4`
- 模型完整后会使用本地文件，可离线推理
- 当前个人版只验收翻译流程；未配置转录模型不影响翻译

## 遇到问题

先双击 `检查Mac环境.command`。仍失败时查看项目根目录的 `latest.log`。

- 模型错误：运行 `./dev.sh python scripts/macos_doctor.py --mode translate --json`
- 没有生成字幕：确认文件格式受支持，并检查 `latest.log`
- 纯静音：允许生成空字幕，日志会明确显示“未检测到语音”
- 需要重做：在终端命令中加入 `--overwrite`
- App 提示项目环境缺失：把 `AI语音翻译.app` 放回本项目根目录，并确认 `.venv` 和 `models/translate` 存在
- App 无法打开 Terminal：检查“系统设置 → 隐私与安全性 → 自动化”中的 Terminal 授权
