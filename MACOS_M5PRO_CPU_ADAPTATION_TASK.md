# macOS M5 Pro CPU 适配执行任务书

> 状态：待后续 agent 执行  
> 基线审计日期：2026-07-14  
> 目标仓库：`/Users/stephenlan/Documents/AI translate`  
> 目标机器：MacBook Pro `Mac17,8`，Apple M5 Pro，18 核（6 Super + 12 Performance），48 GB，macOS 26.5.2，arm64

## 1. 最终目标与完成口径

把当前以 Windows Release/BAT 为主的项目，改造成一份可在这台 Apple Silicon Mac 上从源码稳定运行的个人版：

- 主推理固定走 CTranslate2 CPU，不依赖 CUDA、ROCm、MPS、Metal 或 Apple Neural Engine。
- Whisper 主模型固定使用 `int8` 计算，除非本机基准测试证明另一种 CPU 计算类型更优并记录证据。
- 自定义 Whisper VAD 固定使用 ONNX Runtime `CPUExecutionProvider`，不得误用 CoreML provider。
- 提供可双击的翻译、转录 `.command` 入口；命令行入口仍可直接使用。
- 支持单文件、多文件、文件夹、空格/中文/括号路径，输出 SRT/VTT/LRC。
- 完成一次日译中、一次日文转录、一次批量目录的真实端到端验证。
- 模型下载完成后可离线推理；缺模型或模型损坏时必须快速失败，不能“退出码为 0 但只写空字幕”。
- 所有自动化测试、静态检查、脚本语法检查和本机验收门全部通过，才可宣称“完整完成”。

里程碑定义：

- **MVP 完成**：翻译模型的 CPU 端到端流程通过。
- **完整完成**：翻译、转录、批量、离线和 Finder 双击流程全部通过。
- 如果用户未授权下载大模型，或没有提供真实日语测试音频，代码工作可以完成，但任务状态必须写成“等待端到端资产”，不得写成“完整完成”。

## 2. 执行约束

后续 agent 必须遵守：

1. 开始前运行 `git status --short --branch` 和 `git diff`，保留用户已有改动。
2. 当前已存在的本地改动是：
   - `.gitignore` 已加入 `.codegraph/`；
   - 未跟踪的 `AGENTS.md`；
   - 未跟踪的 `dev.sh`；
   - 未跟踪的 `requirements-macos.txt`。
   - 本次审计新增的 `MACOS_M5PRO_CPU_ADAPTATION_TASK.md` 也是应保留的交付件。
3. 不得删除或重建现有 `.venv`，不得创建第二个虚拟环境。
4. 所有 Python 命令只能经以下包装器运行：

   ```bash
   ./dev.sh python ...
   ./dev.sh pip ...
   ./dev.sh pytest ...
   ./dev.sh ruff ...
   ./dev.sh mypy ...
   ```

5. 未经用户确认，不下载数 GB 级主模型，不清理模型，不改动系统 Python。
6. 模型文件、真实测试媒体、性能日志不得提交 Git；只提交代码、脚本、文档和小型自动化测试。
7. 不删除现有 Windows `.bat`、CUDA/ROCm 环境和打包文件；Mac MVP 先以源码运行，不做 `.app` 或 PyInstaller 分发包。
8. 每个阶段先补测试，再完成实现；阶段门失败时停止进入下一阶段并修复。
9. 不通过改名把 `.bat` 伪装成 `.sh`；Mac 入口必须调用 `.venv` 中的 Python 源码。

## 3. 当前仓库审计结论

### 3.1 已验证的本机基线

| 项目 | 当前结果 |
|---|---|
| 架构 | `arm64` |
| Python | `.venv/bin/python`，3.10.20 |
| CTranslate2 | 4.8.1 |
| Faster-Whisper | 1.2.1 |
| ONNX Runtime | 1.23.2 |
| Transformers | 5.13.1 |
| NumPy | 1.26.4 |
| CTranslate2 CPU 计算类型 | `float32,int8,int8_float32` |
| ORT 可见 provider | `CoreMLExecutionProvider,AzureExecutionProvider,CPUExecutionProvider` |
| Python 可见 CPU 数 | 18 |
| 单元测试 | 19 passed |
| Ruff | 通过 |
| `compileall` | 通过 |
| `pip check` | 通过 |
| Mypy | 失败 3 处，均为 `np.mean` 返回类型未显式转为 `float` |
| `models/` | 当前不存在 |

基线复现命令：

```bash
./dev.sh doctor
./dev.sh pytest -q
./dev.sh ruff check .
./dev.sh mypy src infer.py download_models.py
./dev.sh python -m compileall -q infer.py download_models.py src tests
./dev.sh pip check
bash -n dev.sh
```

### 3.2 真实运行链路

```text
.command / CLI
  -> 根目录 infer.py（把 src 加入 sys.path，并切换到仓库根目录）
  -> parse_arguments / Inference
  -> generation_config.json5
  -> Whisper VAD（Transformers 特征提取 + ONNX Runtime）
  -> Faster-Whisper / CTranslate2 WhisperModel
  -> 字幕时间轴整理与合并
  -> SRT / VTT / LRC / TXT writer
```

已有的跨平台基础：

- 根入口 `infer.py` 本身可在 macOS 从源码运行。
- `--device`、`--compute_type`、`--task`、`--model_name_or_path` 已存在。
- 文件扫描基于 `pathlib`/`os.walk`，不是 Windows 路径 API。
- PyAV 由 Faster-Whisper 带入，当前本机可导入，常见音视频解码无需调用 Windows `ffmpeg.exe`。
- VAD provider 列表当前只主动选择 CUDA 或 CPU；在本机虽然 ORT 暴露 CoreML，现有代码仍会选择 CPU。
- 翻译/转录任务覆盖逻辑和 VAD/智能切分已有部分单元测试。

### 3.3 必须解决的缺口

按优先级排序：

1. **P0：模型缺失会静默退化。** `WhisperVadModel` 初始化失败后只保留 `wrapper=None`；后续 VAD 返回空列表。默认智能切分会把它解释成“语音时长为 0”，可能正常退出并写空字幕。
2. **P0：主模型目录没有严格验证。** `download_models.py::verify_hf_model` 只要目录中存在任意文件就可能返回成功；下载函数也只要求成功下载至少一个文件。
3. **P0：当前没有任何模型。** 现阶段只能验证导入和单元测试，不能声称已可真实翻译。
4. **P0：Mac 启动入口缺失。** 现有 CPU BAT 仍调用 Windows `infer.exe`。
5. **P1：CPU 自动精度与目标不一致。** 本机支持 `int8_float32` 和 `int8`，现有偏好顺序会先选 `int8_float32`；Mac 入口必须明确 `int8`，并让 CPU 的自动选择策略可测试。
6. **P1：VAD 线程硬编码为 8。** 主 Whisper 模型没有传 `cpu_threads`；无法针对 18 核 M5 Pro 测试和固定配置。
7. **P1：VAD/特征提取路径硬编码为相对 `models/...`。** 根入口会 `chdir`，但模块方式或外部工作目录下仍有风险。
8. **P1：翻译与转录模型没有目录隔离。** `--task` 只改任务，不会自动切换正确模型。
9. **P1：推理时仍可能访问 Hugging Face。** 本地 `models/whisper-base` 缺失或加载失败时，VAD 会尝试在线加载。
10. **P1：退出码与错误语义不足。** 缺模型、无可处理文件和成功跳过应可区分。
11. **P2：依赖文件混入 Modal、PyInstaller 和开发依赖。** CPU 个人版的运行时依赖需要与可选/开发依赖分层，并固定已验证版本。
12. **P2：现有诊断面向 CUDA。** `--console` 会检查 `nvidia-smi`，不适合作为 Mac 非交互式 doctor。
13. **P2：Mypy 有 3 个既有错误。** 位于 `vad_manager.py` 中给 `probability` 赋 `np.mean(...)` 的三处。
14. **P2：README 和使用说明几乎完全按 Windows Release 编写。**

## 4. 目标目录和交付件

建议最终结构：

```text
.
├── .venv/                              # 本机环境，不提交
├── infer.py
├── dev.sh
├── requirements-macos.txt              # 最小运行依赖
├── requirements-dev.txt                # 测试/静态检查依赖
├── constraints-macos-arm64.txt          # 本机验证过的精确版本
├── scripts/
│   ├── macos_launcher.py                # 统一构造参数、文件选择、错误展示
│   └── macos_doctor.py                  # 非交互式环境和模型检查
├── 运行(翻译)(CPU).command
├── 运行(转录)(CPU).command
├── 检查Mac环境.command
├── models/                              # 全部忽略，不提交
│   ├── whisper_vad.onnx
│   ├── whisper_vad_metadata.json
│   ├── whisper-base/
│   ├── translate/                       # 海南鸡 CT2 模型
│   └── transcribe/                      # 日文转录 CT2 模型
├── tests/
│   ├── test_macos_launcher.py
│   ├── test_macos_runtime.py
│   └── ...
└── testdata/local/                      # 真实媒体和基准数据，忽略
```

主模型映射固定为：

| 模式 | 模型仓库 | 本地目录 | CLI task |
|---|---|---|---|
| 翻译 | `chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2` | `models/translate` | `translate` |
| 转录 | `TransWithAI/whisper-ja-1.5B-ct2` | `models/transcribe` | `transcribe` |

公共模型资产固定为：

- `models/whisper_vad.onnx`
- `models/whisper_vad_metadata.json`
- `models/whisper-base/preprocessor_config.json`
- `models/whisper-base/config.json`

## 5. 分阶段执行任务

### T0 — 冻结基线并保护现有改动

工作：

- [ ] 记录当前分支、HEAD、`git status`、`git diff`。
- [ ] 确认 `.venv` 为 arm64 Python 3.10.20。
- [ ] 重跑第 3.1 节全部基线命令。
- [ ] 将 Mypy 的 3 个错误记录为“既有错误”，后续修复后不得新增忽略项。
- [ ] 不覆盖现有未跟踪 `dev.sh` 和 `requirements-macos.txt`；先审阅再纳入改动。

阶段门 G0：

- `pytest` 仍为 19 passed；Ruff、compileall、pip check 通过。
- 工作树中用户原有改动没有丢失。

### T1 — 固化 Mac 依赖与环境入口

工作：

- [ ] 将 `requirements-macos.txt` 精简为本地 CPU 推理直接依赖。
- [ ] 把 `pytest`、`ruff`、`mypy`、类型 stub 等移到 `requirements-dev.txt`。
- [ ] 把 Modal、Questionary、PyInstaller、Build 工具移出 Mac 运行时依赖；如仍需保留，放到独立可选文件。
- [ ] 添加 `constraints-macos-arm64.txt`，至少固定当前已验证核心版本：
  - `numpy==1.26.4`
  - `ctranslate2==4.8.1`
  - `faster-whisper==1.2.1`
  - `onnxruntime==1.23.2`
  - `transformers==5.13.1`
- [ ] 复核 `pyjson5`、`librosa`、`av`、`tokenizers`、`huggingface-hub` 的本机版本并一并记录。
- [ ] 保持 `dev.sh` 使用现有 `.venv`；`doctor` 输出平台、架构、Python 路径、核心包版本、CT2 CPU 计算类型和 ORT provider。
- [ ] `bootstrap` 必须显式显示将安装的文件；不得删除已存在 `.venv`。

测试：

```bash
bash -n dev.sh
./dev.sh doctor
./dev.sh pip check
./dev.sh python -c 'import platform; assert platform.machine() == "arm64"'
./dev.sh python -c 'import ctranslate2; assert "int8" in ctranslate2.get_supported_compute_types("cpu")'
./dev.sh python -c 'import onnxruntime as ort; assert "CPUExecutionProvider" in ort.get_available_providers()'
```

阶段门 G1：依赖可导入、无破损依赖，doctor 可非交互运行并以正确退出码结束。

### T2 — 建立严格的模型布局、下载和预检

工作：

- [ ] 扩展 `download_models.py`，保留现有参数兼容，同时加入：
  - `--profile {vad,translate,transcribe,all}`；
  - `--verify-only`；
  - `--non-interactive`；
  - 明确的非零失败退出码。
- [ ] `translate` 下载到 `models/translate`；`transcribe` 下载到 `models/transcribe`；公共 VAD/特征提取文件仍在固定位置。
- [ ] 主 CT2 模型的静态验证至少要求：
  - 非空 `model.bin`；
  - 可解析 `config.json`；
  - 存在 `tokenizer.json` 或 `vocabulary.json` 等模型实际需要的词表文件；
  - 下载清单中的必需文件全部成功，不能以“成功一个文件”判定成功。
- [ ] VAD 验证必须解析 metadata JSON，并实际用 `CPUExecutionProvider` 创建 ONNX session。
- [ ] 特征提取验证必须用本地 `models/whisper-base` 成功构造 `WhisperFeatureExtractor`。
- [ ] 新增 `scripts/macos_doctor.py`，支持 `--mode translate|transcribe|all` 和机器可读 `--json`。
- [ ] doctor 在模型缺失时列出每个缺失路径并返回非零；不得只打印 warning 后继续。
- [ ] 推理前调用同一套资产验证，避免 doctor 与实际运行规则漂移。
- [ ] 推理期设置/支持离线模式，禁止 VAD 在本地文件缺失时静默联网回退。
- [ ] `.gitignore` 忽略整个 `models/`、`testdata/local/` 和本地 benchmark 产物。

单元测试必须覆盖：

- [ ] 主模型目录不存在。
- [ ] 只有 `config.json`。
- [ ] `model.bin` 为 0 字节。
- [ ] VAD metadata 非法 JSON。
- [ ] ONNX session 创建失败。
- [ ] translate 完整、transcribe 缺失时，按模式返回不同结果。
- [ ] `--verify-only --non-interactive` 不产生网络请求。
- [ ] downloader 遇到部分下载、429、镜像失败时返回非零且不把残缺文件当成功。

阶段门 G2：无模型时 doctor 明确失败；伪造的完整测试资产通过；所有失败分支有自动化测试。

### T3 — 修正 CPU 运行策略和错误传播

工作：

- [ ] 把“请求设备”和“实际设备”分开解析；Mac launcher 始终传 `--device cpu`。
- [ ] CPU 的自动计算类型优先级调整为 `int8 -> int8_float32 -> float32`；GPU 顺序保持原行为。
- [ ] Mac launcher 始终显式传 `--compute_type int8`，不依赖自动选择。
- [ ] 新增 `--cpu_threads`，传给 `WhisperModel(..., cpu_threads=...)`；`0` 表示 CTranslate2 自动选择。
- [ ] 新增 `--vad_threads`；移除 `_setup_vad_injection` 中硬编码的 8。
- [ ] VAD 在 CPU 模式下设置 `force_cpu=True`，session provider 必须只含 `CPUExecutionProvider`。
- [ ] ONNX Runtime 建议先使用 `inter_op_num_threads=1`，仅调 `intra_op_num_threads`，避免两个线程池都设为 N 导致过度并行；最终值由 T6 基准决定。
- [ ] 为 VAD 模型、metadata、whisper-base 和主模型使用解析后的绝对路径，不依赖调用者当前目录。
- [ ] VAD 初始化失败时抛出可操作异常，包含失败文件和建议的 doctor/下载命令。
- [ ] 对 `sys.stdout.reconfigure` / `sys.stderr.reconfigure` 做能力检查，避免被重定向或测试替身没有该方法时崩溃。
- [ ] 明确退出码：运行时/模型错误非零；成功处理或“全部输出已存在而跳过”为 0。
- [ ] 修复 Mypy 的 3 个 `np.mean` 类型错误，使用显式 `float(...)`，不得全局关闭检查。

单元测试必须覆盖：

- [ ] mock 支持类型为 `{float32,int8,int8_float32}` 时，CPU auto 选择 `int8`。
- [ ] 显式 `--compute_type` 不被覆盖。
- [ ] `cpu_threads` 正确传给 `WhisperModel`。
- [ ] CPU 模式的 VAD session 不含 CoreML/CUDA。
- [ ] VAD 文件缺失和 session 初始化失败均阻止推理。
- [ ] 缺模型不会创建空 SRT/VTT/LRC。
- [ ] stdout/stderr 不支持 `reconfigure` 时 main 仍可运行。

阶段门 G3：

```bash
./dev.sh pytest -q
./dev.sh ruff check .
./dev.sh mypy src infer.py download_models.py scripts
./dev.sh python -m compileall -q infer.py download_models.py src scripts tests
```

以上全部为 0 退出码，Mypy 不再有既有 3 错误。

### T4 — 实现可测试的 macOS 启动体验

设计要求：

- `.command` 只做定位仓库、切换目录、调用 `./dev.sh python scripts/macos_launcher.py`，不要在 shell 中重新实现业务逻辑。
- `macos_launcher.py` 使用参数列表调用 `infer.py`，不得使用 `shell=True` 或拼接命令字符串。
- 有 CLI 路径参数时逐项原样转发；无参数双击时通过 `osascript` 打开原生多文件选择器。
- 文件选择取消时返回明确状态，不启动模型。
- 翻译和转录 launcher 必须绑定不同模型目录和 task。
- 默认参数：

  ```text
  --device cpu
  --compute_type int8
  --audio_suffixes mp3,wav,flac,m4a,aac,ogg,wma,mp4,mkv,avi,mov,webm,flv,wmv
  --sub_formats srt,vtt,lrc
  ```

- 默认不启用 batched inference；只有 T6 证明 CPU 批处理有稳定收益后才加入可选开关。
- 增加 `--dry-run` 或等价测试模式，只输出 JSON argv，不加载模型。
- 自动化运行可通过环境变量关闭“按回车退出”；Finder 交互失败时保留窗口并显示日志位置。
- 三个 `.command` 文件设置 Git 可执行位并通过 shell 语法检查。

必须添加：

- [ ] `运行(翻译)(CPU).command`
- [ ] `运行(转录)(CPU).command`
- [ ] `检查Mac环境.command`
- [ ] `scripts/macos_launcher.py`
- [ ] `tests/test_macos_launcher.py`

路径测试矩阵：

- [ ] `/tmp/sample.wav`
- [ ] `/tmp/a b/sample file.mp3`
- [ ] `/tmp/日语 音频（测试）/第 1 集.m4a`
- [ ] 同时传入两个文件。
- [ ] 传入包含支持/不支持后缀的目录。
- [ ] 输出目录包含空格和中文。

阶段门 G4：dry-run 产生的 argv 中每个路径都保持一个独立参数；shell 语法检查、单元测试通过。

### T5 — 文档和可重复测试工具

工作：

- [ ] README 首页增加“macOS Apple Silicon CPU”快速开始，并明确这是 CPU 路径，不是 Metal/MPS。
- [ ] `使用说明.txt` 增加安装、模型目录、双击入口、命令行、日志、离线模式和常见错误。
- [ ] 说明翻译模型与转录模型不能只靠 `--task` 互换。
- [ ] 说明首次安装/下载需要网络，推理可离线。
- [ ] 说明模型不进入 Git，下载前检查磁盘空间。
- [ ] 增加一条完全展开、可复制的 CLI 示例。
- [ ] 添加本地真实媒体约定：
  - `testdata/local/japanese-short.*`：10–30 秒功能样本；
  - `testdata/local/japanese-5min.*`：性能样本；
  - 两者均不得提交。
- [ ] 自动化测试所需静音/正弦 WAV 应在测试中用标准库 `wave` 临时生成，不提交二进制 fixture。
- [ ] 增加最终验收清单和故障收集说明（`latest.log`、doctor JSON、命令、退出码）。

阶段门 G5：新用户仅按 README 能完成环境检查、模型准备和第一次翻译；文档中的命令全部实际执行过。

### T6 — 本机真实模型、端到端与性能验收

#### T6.1 资产准备

如果模型仍不存在，先向用户确认大文件下载，再执行经过 T2 改造后的非交互命令。下载完成后：

```bash
./dev.sh python scripts/macos_doctor.py --mode all --json
```

必须保存 doctor 输出到本地验收记录，但不要提交模型或包含私人路径的日志。

#### T6.2 CPU-only 证明

验收日志必须同时证明：

- CTranslate2 实际设备为 `cpu`；
- compute type 为 `int8`；
- VAD session 实际 provider 只有 `CPUExecutionProvider`；
- 未加载 CUDA/ROCm；
- 未把 CoreML 当作 VAD provider；
- 推理不依赖 PyTorch/MPS。

#### T6.3 功能验收

用本地短日语样本依次验证：

1. 翻译 `.command`：生成中文内容的非空 SRT/VTT/LRC。
2. 转录 `.command`：生成日文内容的非空 SRT/VTT/LRC。
3. 同名字幕存在且无 `--overwrite`：正确跳过且退出码 0。
4. 使用 `--overwrite`：重新生成且修改时间更新。
5. 中文/空格/括号路径：成功。
6. 目录批处理：至少两个文件均生成输出。
7. 纯静音临时 WAV：允许输出空字幕，但日志必须明确“未检测到语音”，不能与 VAD 初始化失败混淆。
8. 断网/离线环境：已完整下载模型时仍能成功处理。
9. Finder 双击：三个 `.command` 均可打开；取消文件选择不会报 Python traceback。

字幕结构验证不得只检查文件存在，还必须检查：

- UTF-8 可解码；
- SRT 序号连续；
- 每段 `end > start`；
- 时间轴单调且不重叠；
- 最后时间不明显超过媒体总时长；
- 短日语样本的翻译/转录内容由人工做一次语义核对。

#### T6.4 线程与性能基准

固定同一个 5 分钟样本，先 warm-up 一次，再每组测两次。至少比较：

- CTranslate2 `cpu_threads = 0, 8, 12, 18`；
- VAD `vad_threads = 4, 8, 9`；
- 主模型保持 `int8`；
- 只有在功能完全一致后，可额外对比 `int8_float32`。

用 `/usr/bin/time -l` 记录：

- wall time；
- 最大常驻内存；
- 音频时长；
- RTF = wall time / 音频时长；
- 输出段数和非空字符数；
- 是否出现温度降频迹象或异常波动。

选择规则：

- 取两次测量中位数较优者；
- 与最快结果差距小于 5% 时，优先线程更少、内存更低的配置；
- 两次波动大于 15% 时不得直接下结论，应冷却后重测；
- 不设置未经测量的绝对速度承诺；完成条件是记录本机基线并选出稳定默认值。

阶段门 G6：翻译、转录、批量、离线均通过；性能报告包含可复现命令和选定线程值。

### T7 — 最终回归与交付

完整回归命令：

```bash
git status --short --branch
./dev.sh doctor
./dev.sh pip check
./dev.sh pytest -q
./dev.sh ruff check .
./dev.sh mypy src infer.py download_models.py scripts
./dev.sh python -m compileall -q infer.py download_models.py src scripts tests
bash -n dev.sh
bash -n '运行(翻译)(CPU).command'
bash -n '运行(转录)(CPU).command'
bash -n '检查Mac环境.command'
./dev.sh python infer.py --help
./dev.sh python scripts/macos_doctor.py --mode all --json
```

随后重跑 T6 的三个真实功能路径。交付报告必须列出：

- 修改文件；
- 实际安装的核心版本；
- 模型目录和验证结果；
- 自动测试数量及结果；
- 翻译/转录/批量/离线结果；
- 选定的 `cpu_threads`、`vad_threads` 和基准数据；
- 尚存限制；
- `git status` 中哪些是用户原改动、哪些是本次新增。

## 6. 总验收门

| 门 | 必须满足 | 失败时处理 |
|---|---|---|
| G0 基线 | 原 19 测试通过，用户改动完好 | 停止并恢复误覆盖内容 |
| G1 环境 | arm64、依赖导入、pip check、doctor 通过 | 修复依赖/包装器 |
| G2 资产 | 严格验证能识别缺失、残缺、损坏模型 | 不得进入真实推理 |
| G3 核心 | CPU/int8/VAD CPU、退出码、Mypy 全通过 | 修复并补回归测试 |
| G4 启动器 | 双模式 argv、Unicode/空格路径、可执行位通过 | 修复 launcher |
| G5 文档 | 文档命令逐条验证 | 更正文档或实现 |
| G6 端到端 | 翻译、转录、批量、离线、Finder、性能通过 | 不得宣称完整完成 |
| G7 最终回归 | 全部自动检查再次通过 | 定位回归后重跑所有门 |

## 7. 建议的提交拆分（仅在用户授权提交时）

1. `build: add reproducible macOS arm64 environment`
2. `feat: add strict local model verification and macOS doctor`
3. `fix: enforce CPU int8 inference and fail-fast VAD setup`
4. `feat: add macOS command launchers`
5. `test: cover macOS paths, models, providers, and launch arguments`
6. `docs: add Apple Silicon CPU setup and acceptance guide`

每个提交都必须在提交前至少运行与其范围对应的测试；最终提交后运行 T7 全量回归。

## 8. 明确不在本轮范围内

- CTranslate2 的 Metal/MPS/CoreML 主模型加速。
- Apple Neural Engine。
- 通用 macOS `.app`、签名、公证、DMG。
- Intel Mac/x86_64 支持。
- 修改或重新发布第三方模型权重。
- 删除 Windows/NVIDIA/AMD/Modal 代码。
- 以性能微调为由改变字幕语义或降低现有 VAD/时间轴正确性。

只有在上述完整完成口径通过后，才评估下一阶段的 `.app` 打包或其他 Apple 加速后端。
