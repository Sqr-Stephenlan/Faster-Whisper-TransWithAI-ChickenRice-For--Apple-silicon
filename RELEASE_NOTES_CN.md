# macOS 适配版说明

## 当前版本

- 完成 Apple Silicon / macOS 源码运行适配。
- 新增后端中立推理合同、profile schema 与 CT2/MLX backend factory。
- 接入 ChickenRice 日译中 MLX FP16 模型和 PyTorch-free runtime-only wheel。
- 新增 `--backend {auto,ct2,mlx}`、`--model-variant`、后端可用性 JSON 探测和 MLX GPU Finder 启动入口。
- `AI语音翻译.app` 1.2.0 新增 CPU / GPU 设备卡片、异步环境探测、不可用原因详情和“重新检测”。
- App 会恢复上次仍可用的设备；没有历史选择且 MLX 可用时默认推荐 GPU。历史设备失效时会显式提示当前 UI 切换结果。
- App 启动继续复用稳定 `.command` 与 Python CLI，CPU 显式传入 `ct2`，GPU 显式传入 `mlx`，且 Swift 不再重复检查具体模型目录。
- probe 退出码 1 但 stdout 为有效 schema version 1 JSON 时，App 会按有效报告展示“全部不可用”，不会误报为进程启动失败。
- MLX 显式选择在预检失败时直接退出；`auto` 只允许在开始处理前回退 CT2。
- 保留 CPU VAD、智能 30 秒切块、全局时间轴和字幕后处理，并使核心管线不再读取 Faster-Whisper 返回对象。
- 新增 CT2/MLX 冷热运行 benchmark 工具、模型 manifest/hash 验证和未安装模型标准错误。
- 提供 Finder 可双击的翻译、转录和环境检查入口。
- 本地翻译可使用 MLX/Metal GPU FP16 或 CTranslate2 CPU `int8`；VAD 固定使用 ONNX Runtime CPU provider。
- 增加本地模型布局、严格资产校验、离线运行和明确退出码。
- 支持多文件、目录及包含空格、中文、括号的路径。
- 固化 Apple M5 Pro 验证过的 Python 3.10 依赖版本。
- 保留可选 Modal 云端 GPU 推理，并将本地客户端与远端容器依赖分离。
- CI 改为验证 macOS 源码运行路径，包括单元测试、Ruff、Mypy 和字节码编译。
- 清除旧平台启动器、打包脚本、构建工作流、环境依赖和本地验收残留。

## 本地运行边界

当前本地 MLX 路径针对 Apple Silicon Metal GPU，仅加速 Whisper 主模型张量计算；FFmpeg 解码、ONNX VAD、切块和字幕处理仍在 CPU。CT2 CPU `int8` 路径继续作为稳定回退。Apple Neural Engine、CoreML 和 CTranslate2 Metal 后端不在本版本范围内。

详细使用方法见 [README.md](README.md)、[使用说明.txt](使用说明.txt) 和 [MACOS翻译操作说明.md](MACOS翻译操作说明.md)。
