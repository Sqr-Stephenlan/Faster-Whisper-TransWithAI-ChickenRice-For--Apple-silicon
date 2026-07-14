# macOS M5 Pro CPU 适配续接日志

更新时间：2026-07-14

## 当前状态

状态：**代码与自动化阶段完成，翻译端到端已通过，等待转录/批量/性能资产**。不要标记“完整完成”。

- T0–T5 已完成并逐步提交；工作树在写入本日志前为干净状态。
- 自动回归：`40 passed`；Ruff、Mypy、compileall、pip check、四个 shell 语法检查均通过。
- 环境：arm64，Python 3.10.20，CTranslate2 4.8.1，Faster-Whisper 1.2.1，ORT 1.23.2。
- 公共 VAD 与 `whisper-base` 已存在；VAD doctor 实测 provider 只有 `CPUExecutionProvider`。
- 旧 README 下载的翻译 CT2 模型位于 `models/` 根目录（约 3.0 GB），严格静态验证和 CPU `int8` 实际加载均通过。
- 新布局的 `models/translate/`、`models/transcribe/` 尚不存在，因此 `--mode all` doctor 和双模式 launcher 仍不能按新布局通过。
- 已从用户提供的正片截取 `testdata/local/japanese-short.wav`（30 秒，本地忽略）并完成真实日译中。
- 翻译输出：4 段非空中文 SRT/VTT/LRC；编号连续、时间轴单调不重叠，最后时间 28.48 秒；跳过测试返回 0 且文件时间不变。
- 实测日志证明主模型为 CPU `int8`、VAD 为 CPU；翻译命令退出码 0。
- 缺少独立转录模型和 `japanese-5min.*` 性能样本；尚未执行转录、批量、Finder、断网和线程基准。

## 已完成提交

```text
c284faa  T0  chore: record macOS adaptation baseline
3f7000b  T1  build: add reproducible macOS arm64 environment
221ca3b  T2  feat: add strict model verification and macOS doctor
6b0ac15  T3  fix: enforce CPU int8 inference and fail-fast VAD setup
b9f5247  T4  feat: add macOS command launchers
1b7bb52  T5  docs: add Apple Silicon CPU setup and acceptance guide
```

## 新窗口续接顺序

1. 先读 `MACOS_M5PRO_CPU_ADAPTATION_TASK.md` 的 T6/T7，并运行 `git status --short --branch`。
2. 先决定如何归一化旧翻译模型布局：把根目录 CT2 文件移入 `models/translate/`，或为 launcher/doctor 增加经测试的旧布局兼容；不要复制 3 GB 权重制造重复文件。
3. 获得用户对转录模型下载的明确确认后运行：

   ```bash
   ./dev.sh python download_models.py --profile transcribe --non-interactive
   ./dev.sh python scripts/macos_doctor.py --mode all --json
   ```

4. 继续使用 `testdata/local/japanese-short.wav` 做功能验收，并准备 `japanese-5min.*` 性能样本（均不得提交）。
5. 按 T6 补做转录、批量、Unicode 路径、静音、真实断网和 Finder 双击；翻译、覆盖和跳过已有通过证据。
6. 用 5 分钟样本比较 `cpu_threads=0,8,12,18` 与 `vad_threads=4,8,9`；记录 wall time、RSS、RTF、段数和字符数，再选默认值。
7. 每个新增步骤仍需测试、暂存并单独 commit；最后完整执行 T7。

当前默认值暂为 `cpu_threads=0`、`vad_threads=8`，仅是未基准前的保守值，不是最终性能结论。
