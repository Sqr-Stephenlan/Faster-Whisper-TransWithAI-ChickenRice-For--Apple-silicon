# Faster Whisper TransWithAI ChickenRice for macOS

面向 Apple Silicon Mac 的日语音视频转录与日译中工具。项目从源码运行 Faster Whisper 和音声优化 VAD，主模型使用 CTranslate2 CPU `int8` 推理；本地运行不依赖 Metal、MPS、CoreML 或独立显卡。

## 功能

- 日语音视频转中文字幕，支持 SRT、VTT、LRC
- 使用独立模型执行日文原文转录
- Finder 双击选择文件，也支持终端批量传入文件或目录
- 模型完整后可离线推理
- 可选 Modal 云端 GPU 推理

## 环境要求

- Apple Silicon Mac
- Python 3.10
- 项目本地 `.venv`
- 首次安装依赖和下载模型时需要网络

先检查现有环境：

```bash
./dev.sh doctor
```

若项目尚无 `.venv`，确认后再初始化：

```bash
./dev.sh bootstrap
```

运行依赖来自 `requirements-macos.txt`，并由 `constraints-macos-arm64.txt` 锁定已验证版本。测试和静态检查依赖位于 `requirements-dev.txt`。

## 准备模型

翻译与转录使用不同的主模型：

```bash
./dev.sh python download_models.py --profile translate --non-interactive
./dev.sh python download_models.py --profile transcribe --non-interactive
```

模型保存在本地 `models/`，不提交到 Git：

```text
models/
├── whisper_vad.onnx
├── whisper_vad_metadata.json
├── whisper-base/
├── translate/
└── transcribe/
```

严格检查翻译资产：

```bash
./dev.sh python scripts/macos_doctor.py --mode translate
```

## 运行

在 Finder 中双击：

- `检查Mac环境.command`
- `运行(翻译)(CPU).command`
- `运行(转录)(CPU).command`

也可在终端运行：

```bash
'./运行(翻译)(CPU).command' "/完整路径/日语音频.mp3"
'./运行(转录)(CPU).command' "/完整路径/日语音频.mp3"
```

支持一次传入多个文件、包含空格或中文的路径，以及整个目录。默认在源文件旁生成字幕；已有全部目标字幕时会跳过。使用 `--overwrite` 可覆盖，使用 `--output-dir` 可指定输出目录。

详细操作见 [MACOS翻译操作说明.md](MACOS翻译操作说明.md) 和 [使用说明.txt](使用说明.txt)。

## Modal 云端推理（可选）

本地客户端使用独立轻量环境：

```bash
conda env create -f environment-modal.yml
conda activate faster-whisper-modal
modal token new
python modal_infer.py
```

远端容器依赖定义在 `environment-modal-gpu.yml`。Modal 会产生云端费用，运行前请确认账号额度和所选 GPU。

## 开发验证

所有 Python 命令均通过项目包装器执行：

```bash
./dev.sh pytest
./dev.sh ruff check .
./dev.sh ruff format --check .
./dev.sh mypy --config-file pyproject.toml src infer.py download_models.py modal_infer.py scripts
./dev.sh python -m compileall -q infer.py download_models.py modal_infer.py scripts src tests
./dev.sh pip check
```

本地测试媒体、字幕、基准结果、日志和缓存不得提交 Git。

## 致谢与许可

项目基于 [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)，使用 chickenrice0721 日译中模型、TransWithAI 日文转录模型及音声优化 VAD。详见项目内许可文件。

本项目采用 [MIT License](LICENSE)。
