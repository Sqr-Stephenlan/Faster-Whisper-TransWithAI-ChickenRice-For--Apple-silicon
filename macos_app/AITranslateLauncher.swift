import AppKit
import SwiftUI

private let supportedMediaExtensions: Set<String> = [
    "mp3", "wav", "flac", "m4a", "aac", "ogg", "wma",
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv",
]

enum SubtitleFormat: String, CaseIterable, Hashable {
    case srt
    case vtt
    case lrc

    var displayName: String { rawValue.uppercased() }
}

@main
struct AITranslateLauncherApp: App {
    @StateObject private var viewModel = LauncherViewModel()

    var body: some Scene {
        WindowGroup {
            LauncherView(viewModel: viewModel)
        }
        .windowResizability(.contentSize)
    }
}

struct LauncherView: View {
    @ObservedObject var viewModel: LauncherViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("AI 语音翻译")
                    .font(.system(size: 25, weight: .semibold))
                Text("本地 Apple Silicon CPU 翻译")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(viewModel.isDropTargeted ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.06))
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(
                        viewModel.isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.45),
                        style: StrokeStyle(lineWidth: viewModel.isDropTargeted ? 2 : 1, dash: [7, 5])
                    )

                VStack(spacing: 8) {
                    Image(systemName: "arrow.down.doc")
                        .font(.system(size: 31))
                        .foregroundStyle(viewModel.isDropTargeted ? Color.accentColor : Color.secondary)
                    Text("将音频、视频或文件夹拖到这里")
                        .font(.headline)
                    Text("也可点击选择")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                DropTargetView(
                    onSelect: viewModel.openPanel,
                    onDrop: viewModel.add,
                    onTargetedChange: { viewModel.isDropTargeted = $0 }
                )
            }
            .frame(height: 132)

            HStack(alignment: .center, spacing: 10) {
                Text("输出字幕格式")
                    .font(.headline)

                Spacer(minLength: 4)

                ForEach(SubtitleFormat.allCases, id: \.self) { format in
                    subtitleFormatTag(format)
                }
            }

            VStack(alignment: .leading, spacing: 7) {
                Text("已选择 \(viewModel.selectedURLs.count) 项")
                    .font(.headline)

                if viewModel.selectedURLs.isEmpty {
                    Text("尚未选择文件或文件夹")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(viewModel.selectedURLs.prefix(5).enumerated()), id: \.element.path) { _, url in
                        HStack(spacing: 6) {
                            Image(systemName: viewModel.isDirectory(url) ? "folder" : "doc")
                                .foregroundStyle(.secondary)
                            Text(url.lastPathComponent)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    if viewModel.selectedURLs.count > 5 {
                        Text("另有 \(viewModel.selectedURLs.count - 5) 项……")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, minHeight: 58, alignment: .topLeading)

            HStack {
                Button("清除", action: viewModel.clear)
                    .disabled(viewModel.selectedURLs.isEmpty || viewModel.isLaunching)
                Spacer()
                if viewModel.isLaunching {
                    ProgressView()
                        .controlSize(.small)
                    Text("正在打开 Terminal…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Button("开始翻译", action: viewModel.startTranslation)
                    .keyboardShortcut(.defaultAction)
                    .disabled(viewModel.selectedURLs.isEmpty || viewModel.isLaunching)
            }
        }
        .padding(24)
        .frame(width: 520, height: 416)
        .alert(
            "提示",
            isPresented: Binding(
                get: { viewModel.alertMessage != nil },
                set: { if !$0 { viewModel.alertMessage = nil } }
            )
        ) {
            Button("好", role: .cancel) {
                viewModel.alertMessage = nil
            }
        } message: {
            Text(viewModel.alertMessage ?? "")
        }
    }

    private func subtitleFormatTag(_ format: SubtitleFormat) -> some View {
        let isSelected = viewModel.selectedSubtitleFormats.contains(format)

        return Button {
            viewModel.toggleSubtitleFormat(format)
        } label: {
            HStack(spacing: 5) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.caption.weight(.bold))
                }
                Text(format.displayName)
                    .font(.subheadline.weight(.semibold))
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .foregroundStyle(isSelected ? Color.white : Color.primary)
            .background {
                Capsule()
                    .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.08))
            }
            .overlay {
                Capsule()
                    .strokeBorder(
                        isSelected ? Color.accentColor : Color.secondary.opacity(0.45),
                        lineWidth: 1
                    )
            }
        }
        .buttonStyle(.plain)
        .disabled(viewModel.isLaunching)
        .accessibilityLabel("\(format.displayName) 字幕格式")
        .accessibilityValue(isSelected ? "已选择" : "未选择")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

@MainActor
final class LauncherViewModel: ObservableObject {
    @Published var selectedURLs: [URL] = []
    @Published var selectedSubtitleFormats: Set<SubtitleFormat> = Set(SubtitleFormat.allCases)
    @Published var isDropTargeted = false
    @Published var isLaunching = false
    @Published var alertMessage: String?

    private let fileManager = FileManager.default

    var subtitleFormatsArgument: String {
        SubtitleFormat.allCases
            .filter(selectedSubtitleFormats.contains)
            .map(\.rawValue)
            .joined(separator: ",")
    }

    func isDirectory(_ url: URL) -> Bool {
        var isDirectory = ObjCBool(false)
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    func openPanel() {
        guard !isLaunching else { return }

        let panel = NSOpenPanel()
        panel.title = "选择要翻译的音视频或文件夹"
        panel.prompt = "选择"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.resolvesAliases = true
        panel.begin { [weak self] response in
            guard response == .OK else { return }
            self?.add(panel.urls)
        }
    }

    func add(_ urls: [URL]) {
        guard !isLaunching else { return }

        let existingPaths = Set(selectedURLs.map(\.path))
        var seenPaths = existingPaths
        var additions: [URL] = []
        var ignoredCount = 0

        for rawURL in urls {
            let url = rawURL.standardizedFileURL
            var directoryFlag = ObjCBool(false)
            guard fileManager.fileExists(atPath: url.path, isDirectory: &directoryFlag) else {
                ignoredCount += 1
                continue
            }

            let isSupported = directoryFlag.boolValue || supportedMediaExtensions.contains(url.pathExtension.lowercased())
            guard isSupported else {
                ignoredCount += 1
                continue
            }
            guard seenPaths.insert(url.path).inserted else { continue }
            additions.append(url)
        }

        selectedURLs.append(contentsOf: additions)

        if ignoredCount > 0 {
            if additions.isEmpty {
                alertMessage = "未添加项目：所选普通文件均不是支持的音频或视频格式。"
            } else {
                alertMessage = "已忽略 \(ignoredCount) 个不支持或不存在的普通文件。"
            }
        }
    }

    func clear() {
        guard !isLaunching else { return }
        selectedURLs.removeAll()
    }

    func toggleSubtitleFormat(_ format: SubtitleFormat) {
        guard !isLaunching else { return }

        if selectedSubtitleFormats.contains(format) {
            guard selectedSubtitleFormats.count > 1 else {
                alertMessage = "至少选择一种字幕格式。"
                return
            }
            selectedSubtitleFormats.remove(format)
        } else {
            selectedSubtitleFormats.insert(format)
        }
    }

    func startTranslation() {
        guard !selectedURLs.isEmpty, !isLaunching else { return }

        let projectRoot = Bundle.main.bundleURL.deletingLastPathComponent()
        let launcherURL = projectRoot.appendingPathComponent("运行(翻译)(CPU).command")
        let requiredPaths = [
            launcherURL,
            projectRoot.appendingPathComponent("dev.sh"),
            projectRoot.appendingPathComponent(".venv/bin/python"),
            projectRoot.appendingPathComponent("scripts/macos_launcher.py"),
            projectRoot.appendingPathComponent("models/translate"),
        ]

        guard requiredPaths.allSatisfy({ fileManager.fileExists(atPath: $0.path) }) else {
            alertMessage = "未找到项目运行环境。请将“AI语音翻译.app”放回 AI translate 项目根目录，并确认 .venv 和翻译模型已经准备完成。"
            return
        }

        isLaunching = true
        TerminalLauncher.launch(
            projectRoot: projectRoot,
            launcherURL: launcherURL,
            inputURLs: selectedURLs
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                NSApplication.shared.terminate(nil)
            case .failure(let error):
                self.isLaunching = false
                self.alertMessage = "无法在 Terminal 中启动翻译：\n\(error.localizedDescription)"
            }
        }
    }
}

private enum TerminalLauncher {
    private static let script = """
    on run argv
        set projectRoot to item 1 of argv
        set launcherPath to item 2 of argv
        set inputPaths to items 3 thru (count argv) of argv

        set shellCommand to "cd " & quoted form of projectRoot
        set shellCommand to shellCommand & " && " & quoted form of launcherPath

        repeat with inputPath in inputPaths
            set shellCommand to shellCommand & " " & quoted form of (contents of inputPath)
        end repeat

        tell application "Terminal"
            activate
            do script shellCommand
        end tell
    end run
    """

    static func launch(
        projectRoot: URL,
        launcherURL: URL,
        inputURLs: [URL],
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        let process = Process()
        let errorPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script, projectRoot.path, launcherURL.path] + inputURLs.map(\.path)
        process.standardError = errorPipe

        process.terminationHandler = { completedProcess in
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorText = String(data: errorData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            DispatchQueue.main.async {
                if completedProcess.terminationStatus == 0 {
                    completion(.success(()))
                } else {
                    let detail = errorText.flatMap { $0.isEmpty ? nil : $0 }
                        ?? "osascript 退出码 \(completedProcess.terminationStatus)"
                    completion(.failure(TerminalLaunchError(message: detail)))
                }
            }
        }

        do {
            try process.run()
        } catch {
            completion(.failure(error))
        }
    }
}

private struct TerminalLaunchError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

private struct DropTargetView: NSViewRepresentable {
    let onSelect: () -> Void
    let onDrop: ([URL]) -> Void
    let onTargetedChange: (Bool) -> Void

    func makeNSView(context: Context) -> DropTargetNSView {
        DropTargetNSView(
            onSelect: onSelect,
            onDrop: onDrop,
            onTargetedChange: onTargetedChange
        )
    }

    func updateNSView(_ nsView: DropTargetNSView, context: Context) {
        nsView.onSelect = onSelect
        nsView.onDrop = onDrop
        nsView.onTargetedChange = onTargetedChange
    }
}

private final class DropTargetNSView: NSView {
    var onSelect: () -> Void
    var onDrop: ([URL]) -> Void
    var onTargetedChange: (Bool) -> Void

    init(
        onSelect: @escaping () -> Void,
        onDrop: @escaping ([URL]) -> Void,
        onTargetedChange: @escaping (Bool) -> Void
    ) {
        self.onSelect = onSelect
        self.onDrop = onDrop
        self.onTargetedChange = onTargetedChange
        super.init(frame: .zero)
        registerForDraggedTypes([.fileURL])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func mouseDown(with event: NSEvent) {
        onSelect()
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        let hasFileURLs = !fileURLs(from: sender).isEmpty
        onTargetedChange(hasFileURLs)
        return hasFileURLs ? .copy : []
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        fileURLs(from: sender).isEmpty ? [] : .copy
    }

    override func draggingExited(_ sender: NSDraggingInfo?) {
        onTargetedChange(false)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = fileURLs(from: sender)
        onTargetedChange(false)
        guard !urls.isEmpty else { return false }
        onDrop(urls)
        return true
    }

    private func fileURLs(from sender: NSDraggingInfo) -> [URL] {
        let options: [NSPasteboard.ReadingOptionKey: Any] = [.urlReadingFileURLsOnly: true]
        let objects = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: options) ?? []
        return objects.compactMap { ($0 as? NSURL) as URL? }
    }
}
