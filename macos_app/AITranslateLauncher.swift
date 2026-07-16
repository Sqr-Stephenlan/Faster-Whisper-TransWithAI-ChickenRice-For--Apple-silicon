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
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("AI 语音翻译")
                    .font(.system(size: 25, weight: .semibold))
                Text(viewModel.subtitleText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            backendSection

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
            .frame(height: 112)

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
                    ForEach(Array(viewModel.selectedURLs.prefix(4).enumerated()), id: \.element.path) { _, url in
                        HStack(spacing: 6) {
                            Image(systemName: viewModel.isDirectory(url) ? "folder" : "doc")
                                .foregroundStyle(.secondary)
                            Text(url.lastPathComponent)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    if viewModel.selectedURLs.count > 4 {
                        Text("另有 \(viewModel.selectedURLs.count - 4) 项……")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, minHeight: 52, alignment: .topLeading)

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
                Button(viewModel.startButtonTitle, action: viewModel.startInference)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!viewModel.canStart)
            }
        }
        .padding(22)
        .frame(width: 560, height: 590)
        .onAppear(perform: viewModel.probeIfNeeded)
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
        .sheet(item: $viewModel.diagnosticDetails) { details in
            DiagnosticDetailsView(details: details)
        }
    }

    private var backendSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text("运行设备")
                    .font(.headline)
                Spacer()
                if viewModel.isProbing {
                    ProgressView()
                        .controlSize(.small)
                    Text("正在检测")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Button("重新检测", action: viewModel.refreshProbe)
                    .buttonStyle(.link)
                    .disabled(viewModel.isProbing || viewModel.isLaunching)
            }

            HStack(alignment: .top, spacing: 12) {
                backendCard(.ct2)
                backendCard(.mlx)
            }

            if let failure = viewModel.probeFailure {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(failure.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("查看详情", action: viewModel.showProbeFailureDetails)
                        .buttonStyle(.link)
                        .font(.caption)
                }
            }
        }
    }

    private func backendCard(_ backend: InferenceBackend) -> some View {
        let isSelected = viewModel.selectedBackend == backend
        let isAvailable = viewModel.availability(for: backend)?.available == true

        return VStack(alignment: .leading, spacing: 7) {
            Button {
                viewModel.selectBackend(backend)
            } label: {
                VStack(alignment: .leading, spacing: 7) {
                    HStack(spacing: 8) {
                        Text(backend.displayName)
                            .font(.headline)
                        Spacer()
                        if backend == .mlx, isAvailable {
                            Text("推荐")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 3)
                                .foregroundStyle(Color.accentColor)
                                .background(Color.accentColor.opacity(0.12), in: Capsule())
                        }
                    }

                    Text(viewModel.backendTechnicalDescription(backend))
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 6) {
                        backendStatusIcon(backend)
                        Text(viewModel.backendStatusText(backend))
                            .font(.caption.weight(.medium))
                            .foregroundStyle(backendStatusColor(backend))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!viewModel.canSelectBackend(backend))

            if viewModel.shouldShowBackendDetailsButton(backend) {
                HStack {
                    Text("当前不可用")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("查看详情") {
                        viewModel.showBackendDetails(backend)
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, minHeight: 102, alignment: .topLeading)
        .background {
            RoundedRectangle(cornerRadius: 11)
                .fill(isSelected ? Color.accentColor.opacity(0.10) : Color.secondary.opacity(0.055))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 11)
                .strokeBorder(
                    isSelected ? Color.accentColor : Color.secondary.opacity(0.35),
                    lineWidth: isSelected ? 2 : 1
                )
        }
        .opacity(viewModel.isBackendReadable(backend) ? 1 : 0.72)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(backend.displayName)，\(viewModel.backendTechnicalDescription(backend))")
        .accessibilityValue(viewModel.backendStatusText(backend))
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    @ViewBuilder
    private func backendStatusIcon(_ backend: InferenceBackend) -> some View {
        if viewModel.isProbing {
            ProgressView()
                .controlSize(.mini)
        } else {
            Image(systemName: viewModel.backendStatusSymbol(backend))
                .foregroundStyle(backendStatusColor(backend))
        }
    }

    private func backendStatusColor(_ backend: InferenceBackend) -> Color {
        guard let availability = viewModel.availability(for: backend) else {
            return viewModel.probeFailure == nil ? .secondary : .orange
        }
        return availability.available ? .green : .orange
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

private struct DiagnosticDetailsView: View {
    let details: DiagnosticDetails
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(details.title)
                .font(.title2.weight(.semibold))

            ScrollView {
                Text(details.message)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .textSelection(.enabled)
            }

            HStack {
                Spacer()
                Button("关闭") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding(22)
        .frame(width: 540, height: 340)
    }
}

struct DiagnosticDetails: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

@MainActor
final class LauncherViewModel: ObservableObject {
    @Published var selectedURLs: [URL] = []
    @Published var selectedSubtitleFormats: Set<SubtitleFormat> = Set(SubtitleFormat.allCases)
    @Published var selectedBackend: InferenceBackend?
    @Published var probeState: ProbeState = .idle
    @Published var isDropTargeted = false
    @Published var isLaunching = false
    @Published var alertMessage: String?
    @Published var diagnosticDetails: DiagnosticDetails?

    private let fileManager = FileManager.default
    private let task: InferenceTask = .translate
    private let probeRunner = BackendProbeRunner()
    private let userDefaults = UserDefaults.standard

    var subtitleText: String {
        switch probeState {
        case .idle, .probing:
            return "正在检查本地推理环境…"
        case .failed:
            return "本地推理环境未就绪"
        case .ready(let report, _):
            guard InferenceBackend.allCases.contains(where: report.isAvailable) else {
                return "本地推理环境未就绪"
            }
            switch selectedBackend {
            case .ct2:
                return "日译中 · CPU 兼容模式"
            case .mlx:
                return "日译中 · MLX Metal 加速"
            case nil:
                return "日译中 · 请选择运行设备"
            }
        }
    }

    var startButtonTitle: String {
        switch selectedBackend {
        case .ct2:
            return "开始 CPU 翻译"
        case .mlx:
            return "开始 GPU 翻译"
        case nil:
            return "开始翻译"
        }
    }

    var isProbing: Bool {
        if case .probing = probeState {
            return true
        }
        return false
    }

    var probeFailure: ProbeFailure? {
        if case .failed(let failure) = probeState {
            return failure
        }
        return nil
    }

    var canStart: Bool {
        guard
            !selectedURLs.isEmpty,
            !selectedSubtitleFormats.isEmpty,
            !isLaunching,
            let selectedBackend,
            let report = readyReport
        else {
            return false
        }
        return report.isAvailable(selectedBackend)
    }

    var subtitleFormatsArgument: String {
        SubtitleFormat.allCases
            .filter(selectedSubtitleFormats.contains)
            .map(\.rawValue)
            .joined(separator: ",")
    }

    private var readyReport: BackendProbeReport? {
        if case .ready(let report, _) = probeState {
            return report
        }
        return nil
    }

    private var projectRoot: URL {
        Bundle.main.bundleURL.deletingLastPathComponent()
    }

    func probeIfNeeded() {
        guard case .idle = probeState else { return }
        refreshProbe()
    }

    func refreshProbe() {
        guard !isProbing, !isLaunching else { return }

        probeState = .probing
        probeRunner.probe(projectRoot: projectRoot, task: task) { [weak self] result in
            Task { @MainActor in
                self?.handleProbeResult(result)
            }
        }
    }

    func availability(for backend: InferenceBackend) -> BackendAvailability? {
        readyReport?.availability(for: backend)
    }

    func canSelectBackend(_ backend: InferenceBackend) -> Bool {
        !isLaunching && availability(for: backend)?.available == true
    }

    func isBackendReadable(_ backend: InferenceBackend) -> Bool {
        if isProbing {
            return true
        }
        return availability(for: backend)?.available != false
    }

    func backendTechnicalDescription(_ backend: InferenceBackend) -> String {
        switch backend {
        case .ct2:
            return "CTranslate2 · int8"
        case .mlx:
            return "MLX · Metal · FP16"
        }
    }

    func backendStatusText(_ backend: InferenceBackend) -> String {
        if isProbing {
            return "检查中…"
        }
        if let availability = availability(for: backend) {
            return availability.available ? "可用" : "当前不可用"
        }
        if probeFailure != nil {
            return "无法检测"
        }
        return "等待检测"
    }

    func backendStatusSymbol(_ backend: InferenceBackend) -> String {
        guard let availability = availability(for: backend) else {
            return probeFailure == nil ? "circle.dotted" : "exclamationmark.triangle.fill"
        }
        return availability.available ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
    }

    func shouldShowBackendDetailsButton(_ backend: InferenceBackend) -> Bool {
        guard let availability = availability(for: backend) else { return false }
        return !availability.available
    }

    func selectBackend(_ backend: InferenceBackend) {
        guard canSelectBackend(backend) else { return }
        selectedBackend = backend
        userDefaults.set(backend.rawValue, forKey: task.backendPreferenceKey)
    }

    func showBackendDetails(_ backend: InferenceBackend) {
        guard let availability = availability(for: backend) else { return }

        let reasons = availability.reasons.isEmpty
            ? "未提供额外诊断。"
            : availability.reasons.map { "• \($0)" }.joined(separator: "\n")
        let device = availability.device ?? "未报告"
        let status = availability.available ? "可用" : "当前不可用"
        diagnosticDetails = DiagnosticDetails(
            title: "\(backend.displayName) 环境详情",
            message: """
            后端：\(backend.rawValue)
            状态：\(status)
            设备：\(device)
            模型：\(availability.model.path)
            模型变体：\(availability.model.variant)

            原始诊断：
            \(reasons)
            """
        )
    }

    func showProbeFailureDetails() {
        guard let failure = probeFailure else { return }
        diagnosticDetails = DiagnosticDetails(
            title: failure.summary,
            message: failure.detail
        )
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

    func startInference() {
        guard !selectedURLs.isEmpty, !isLaunching else { return }
        guard !selectedSubtitleFormats.isEmpty else {
            alertMessage = "至少选择一种字幕格式。"
            return
        }
        guard let report = readyReport else {
            refreshProbe()
            return
        }
        guard let selectedBackend, report.isAvailable(selectedBackend) else {
            alertMessage = "请选择当前可用的运行设备。"
            return
        }
        guard let launcherFilename = launcherFilename(for: selectedBackend) else {
            alertMessage = "当前任务尚未提供 \(selectedBackend.displayName) 启动入口。"
            return
        }

        let launcherURL = projectRoot.appendingPathComponent(launcherFilename)
        let requiredPaths = [
            launcherURL,
            projectRoot.appendingPathComponent("dev.sh"),
            projectRoot.appendingPathComponent(".venv/bin/python"),
            projectRoot.appendingPathComponent("scripts/macos_launcher.py"),
            projectRoot.appendingPathComponent("scripts/backend_probe.py"),
        ]

        guard requiredPaths.allSatisfy({ fileManager.fileExists(atPath: $0.path) }) else {
            alertMessage = "未找到项目运行入口。请将“AI语音翻译.app”放回 AI translate 项目根目录，并确认 dev.sh、.venv 和 scripts 目录完整。"
            return
        }

        isLaunching = true
        TerminalLauncher.launch(
            projectRoot: projectRoot,
            launcherURL: launcherURL,
            subtitleFormats: subtitleFormatsArgument,
            inputURLs: selectedURLs
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                NSApplication.shared.terminate(nil)
            case .failure(let error):
                self.isLaunching = false
                self.alertMessage = "无法在 Terminal 中启动 \(selectedBackend.displayName) 翻译：\n\(error.localizedDescription)"
            }
        }
    }

    private func handleProbeResult(_ result: Result<BackendProbeReport, ProbeFailure>) {
        switch result {
        case .success(let report):
            probeState = .ready(report, checkedAt: Date())
            let savedBackend = userDefaults
                .string(forKey: task.backendPreferenceKey)
                .flatMap(InferenceBackend.init(rawValue:))
            let decision = BackendSelectionPolicy.decide(
                report: report,
                savedBackend: savedBackend
            )
            selectedBackend = decision.selectedBackend
            if let notice = decision.notice {
                alertMessage = notice
            }
        case .failure(let failure):
            probeState = .failed(failure)
        }
    }

    private func launcherFilename(for backend: InferenceBackend) -> String? {
        switch (task, backend) {
        case (.translate, .ct2):
            return "运行(翻译)(CPU).command"
        case (.translate, .mlx):
            return "运行(翻译)(GPU-MLX).command"
        case (.transcribe, .ct2):
            return "运行(转录)(CPU).command"
        case (.transcribe, .mlx):
            return nil
        }
    }
}

private enum TerminalLauncher {
    private static let script = """
    on run argv
        set projectRoot to item 1 of argv
        set launcherPath to item 2 of argv
        set launcherArguments to items 3 thru (count argv) of argv

        set shellCommand to "cd " & quoted form of projectRoot
        set shellCommand to shellCommand & " && " & quoted form of launcherPath

        repeat with launcherArgument in launcherArguments
            set shellCommand to shellCommand & " " & quoted form of (contents of launcherArgument)
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
        subtitleFormats: String,
        inputURLs: [URL],
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        let process = Process()
        let errorPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = [
            "-e",
            script,
            projectRoot.path,
            launcherURL.path,
            "--sub-formats",
            subtitleFormats,
        ] + inputURLs.map(\.path)
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
