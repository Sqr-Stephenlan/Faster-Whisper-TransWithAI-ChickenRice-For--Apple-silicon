# macOS 翻译操作说明

## 最简单的用法

1. 在 Finder 打开本项目文件夹。
2. 双击 `检查Mac环境.command`。看到翻译模型就绪后关闭窗口。
3. 双击 `运行(翻译)(CPU).command`。
4. 在文件选择器中选择一个或多个日语音频/视频文件。
5. 等待完成。默认在原文件旁生成 `.srt`、`.vtt`、`.lrc` 三种字幕。

支持单文件、多文件，以及包含空格、中文或括号的路径。已有三种字幕时会自动跳过。

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
