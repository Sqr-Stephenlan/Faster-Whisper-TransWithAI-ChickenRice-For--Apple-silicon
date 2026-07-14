# macOS M5 Pro CPU 适配续接日志

更新时间：2026-07-14

## 当前状态

状态：**用户要求的 macOS 翻译范围已完整完成**。转录流程按用户要求不作为本轮完成条件。

- T0–T5 已完成并逐步提交；工作树在写入本日志前为干净状态。
- 自动回归：`42 passed`；Ruff、Mypy、compileall、pip check、四个 shell 语法检查均通过。
- 环境：arm64，Python 3.10.20，CTranslate2 4.8.1，Faster-Whisper 1.2.1，ORT 1.23.2。
- 公共 VAD 与 `whisper-base` 已存在；VAD doctor 实测 provider 只有 `CPUExecutionProvider`。
- 约 3.0 GB 的翻译 CT2 模型已原地归一化到 `models/translate/`，未复制权重；翻译 doctor、严格静态验证和 CPU `int8` 实际加载均通过。
- 已从用户提供的正片截取 `testdata/local/japanese-short.wav`（30 秒，本地忽略）并完成真实日译中。
- 翻译输出：4 段非空中文 SRT/VTT/LRC；编号连续、时间轴单调不重叠，最后时间 28.48 秒；跳过测试返回 0 且文件时间不变。
- 实测日志证明主模型为 CPU `int8`、VAD 为 CPU；翻译命令退出码 0。
- Unicode/空格/括号、中文输出目录、两文件批量、跳过、覆盖、纯静音、离线变量和字幕结构均通过。
- Finder 原生打开动作已确认启动 Terminal、launcher 与原生多文件选择器；取消分支有自动化测试。
- 300 秒样本线程双测已完成：CPU 12 线程中位数 56.56 秒；VAD 4/8/9 均在 5% 内，按规则选 4 线程。峰值 RSS 约 5.43 GiB。
- 不同运行在一个约 1 秒的短句上存在 Whisper 非确定性文本波动；其余 39 段和时间轴一致，最终短样本已人工确认生成合理中文。
- 最终默认值为 `cpu_threads=12`、`vad_threads=4`；最终入口实跑退出码 0。
- 独立转录模型未配置；这是用户明确接受的个人使用范围，不影响翻译完成状态。

## 已完成提交

```text
c284faa  T0  chore: record macOS adaptation baseline
3f7000b  T1  build: add reproducible macOS arm64 environment
221ca3b  T2  feat: add strict model verification and macOS doctor
6b0ac15  T3  fix: enforce CPU int8 inference and fail-fast VAD setup
b9f5247  T4  feat: add macOS command launchers
1b7bb52  T5  docs: add Apple Silicon CPU setup and acceptance guide
f65406c  T6  fix: report silent audio distinctly from VAD failures
78edc10  T6  perf: tune macOS CPU translation thread defaults
```

## 后续可选工作

仅当用户以后需要日文原文转录时，再下载 `transcribe` profile 并执行对应验收。当前日译中使用方法见 `MACOS翻译操作说明.md`。
