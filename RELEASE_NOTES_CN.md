# macOS 适配版说明

## 当前版本

- 完成 Apple Silicon / macOS 源码运行适配。
- 提供 Finder 可双击的翻译、转录和环境检查入口。
- 本地推理固定使用 CTranslate2 CPU `int8`，VAD 固定使用 ONNX Runtime CPU provider。
- 增加本地模型布局、严格资产校验、离线运行和明确退出码。
- 支持多文件、目录及包含空格、中文、括号的路径。
- 固化 Apple M5 Pro 验证过的 Python 3.10 依赖版本。
- 保留可选 Modal 云端 GPU 推理，并将本地客户端与远端容器依赖分离。
- CI 改为验证 macOS 源码运行路径，包括单元测试、Ruff、Mypy 和字节码编译。
- 清除旧平台启动器、打包脚本、构建工作流、环境依赖和本地验收残留。

## 本地运行边界

当前本地路径针对 Apple Silicon CPU。它不宣称使用 Metal、MPS、CoreML 主模型加速或 Apple Neural Engine。需要云端 GPU 时可选择 Modal 流程。

详细使用方法见 [README.md](README.md)、[使用说明.txt](使用说明.txt) 和 [MACOS翻译操作说明.md](MACOS翻译操作说明.md)。
