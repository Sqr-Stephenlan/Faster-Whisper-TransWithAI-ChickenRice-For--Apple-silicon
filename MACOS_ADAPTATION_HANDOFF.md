# macOS M5 Pro CPU 适配续接日志

更新时间：2026-07-14

## 当前状态

状态：**代码与自动化阶段完成，等待端到端资产**。不要标记“完整完成”。

- T0–T5 已完成并逐步提交；工作树在写入本日志前为干净状态。
- 自动回归：`40 passed`；Ruff、Mypy、compileall、pip check、四个 shell 语法检查均通过。
- 环境：arm64，Python 3.10.20，CTranslate2 4.8.1，Faster-Whisper 1.2.1，ORT 1.23.2。
- 公共 VAD 与 `whisper-base` 已存在；VAD doctor 实测 provider 只有 `CPUExecutionProvider`。
- 缺少 `models/translate/` 与 `models/transcribe/`，doctor 按预期返回 1。
- 缺少 `testdata/local/japanese-short.*` 与 `japanese-5min.*` 真实样本。
- 未经用户确认，尚未下载数 GB 主模型，也未执行 T6 真实推理/性能基准。

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
2. 获得用户对大模型下载的明确确认后运行：

   ```bash
   ./dev.sh python download_models.py --profile all --non-interactive
   ./dev.sh python scripts/macos_doctor.py --mode all --json
   ```

3. 请用户把真实日语样本放到 `testdata/local/`（该目录已忽略）。
4. 按 T6 依次验证翻译、转录、批量、覆盖/跳过、Unicode 路径、静音、离线和 Finder 双击。
5. 用 5 分钟样本比较 `cpu_threads=0,8,12,18` 与 `vad_threads=4,8,9`；记录 wall time、RSS、RTF、段数和字符数，再选默认值。
6. 每个新增步骤仍需测试、暂存并单独 commit；最后完整执行 T7。

当前默认值暂为 `cpu_threads=0`、`vad_threads=8`，仅是未基准前的保守值，不是最终性能结论。
