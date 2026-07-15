# AI 语音翻译 App 简明使用说明

## 开始前

`AI语音翻译.app` 应位于 `AI translate` 项目根目录，不要单独移动到 `/Applications`。项目内还需保留 `.venv`、`models`、`dev.sh`、`scripts` 和 `运行(翻译)(CPU).command`。

首次使用建议双击 `检查Mac环境.command`，确认翻译模型已经准备完成。

## 三步开始翻译

1. 双击 `AI语音翻译.app`。
2. 把音频、视频或文件夹拖进窗口；也可以点击拖拽区域选择多个项目。
3. 检查选择列表，点击“开始翻译”。

App 会退出并在 Terminal 中启动现有翻译流程。处理状态、跳过信息和错误都显示在 Terminal 中。

## 输出位置

默认在每个源音视频旁生成：

- `.srt`
- `.vtt`
- `.lrc`

如果三种字幕都已存在，该文件会自动跳过。

## 支持的输入

可一次选择文件、文件夹，或两者混合。文件夹由现有启动器递归扫描。

支持后缀：

```text
mp3, wav, flac, m4a, aac, ogg, wma,
mp4, mkv, avi, mov, webm, flv, wmv
```

重复路径只保留一次；不支持的普通文件会被忽略并显示提示。

## 首次授权

首次点击“开始翻译”时，macOS 可能询问是否允许 `AI语音翻译` 控制 Terminal。请选择允许，否则 App 无法新建 Terminal 翻译任务。

## 常见问题

- 提示“未找到项目运行环境”：将 App 放回项目根目录，并检查 `.venv` 和 `models/translate`。
- 没有生成字幕：查看 Terminal 输出和项目根目录的 `latest.log`。
- 需要重新处理已有字幕：改用终端命令并加入 `--overwrite`。
- App 缺失或源码已更新：在项目根目录运行 `./scripts/build_macos_app.sh`。
