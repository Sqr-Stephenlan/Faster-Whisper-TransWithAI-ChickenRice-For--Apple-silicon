import Foundation

enum InferenceTask: String, Codable, CaseIterable {
    case translate
    case transcribe

    var probeProfile: String { rawValue }

    var backendPreferenceKey: String {
        "launcher.selectedBackend.\(rawValue)"
    }
}

enum InferenceBackend: String, Codable, CaseIterable {
    case ct2
    case mlx

    var displayName: String {
        switch self {
        case .ct2:
            return "CPU"
        case .mlx:
            return "GPU"
        }
    }
}

struct BackendCapabilities: Decodable, Equatable {
    let translate: Bool
    let transcribe: Bool
    let wordTimestamps: Bool
    let batching: Bool

    private enum CodingKeys: String, CodingKey {
        case translate
        case transcribe
        case wordTimestamps = "word_timestamps"
        case batching
    }
}

struct BackendModel: Decodable, Equatable {
    let path: String
    let profile: String
    let variant: String
}

struct BackendAvailability: Decodable, Equatable {
    let available: Bool
    let capabilities: BackendCapabilities
    let device: String?
    let model: BackendModel
    let reasons: [String]
}

struct BackendProbeReport: Decodable, Equatable {
    let schemaVersion: Int
    let profile: String
    let language: String
    let task: String
    let backends: [String: BackendAvailability]

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case profile
        case language
        case task
        case backends
    }

    static func decode(from data: Data, expectedTask: InferenceTask? = nil) throws -> BackendProbeReport {
        let report = try JSONDecoder().decode(BackendProbeReport.self, from: data)
        guard report.schemaVersion == 1 else {
            throw BackendProbeContractError.unsupportedSchema(report.schemaVersion)
        }
        if let expectedTask {
            guard report.profile == expectedTask.probeProfile else {
                throw BackendProbeContractError.unexpectedProfile(
                    expected: expectedTask.probeProfile,
                    actual: report.profile
                )
            }
            guard report.task == expectedTask.rawValue else {
                throw BackendProbeContractError.unexpectedTask(
                    expected: expectedTask.rawValue,
                    actual: report.task
                )
            }
        }
        return report
    }

    func availability(for backend: InferenceBackend) -> BackendAvailability? {
        backends[backend.rawValue]
    }

    func isAvailable(_ backend: InferenceBackend) -> Bool {
        availability(for: backend)?.available == true
    }
}

enum BackendProbeContractError: LocalizedError, Equatable {
    case unsupportedSchema(Int)
    case unexpectedProfile(expected: String, actual: String)
    case unexpectedTask(expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let version):
            return "不支持环境检查 schema version \(version)。"
        case .unexpectedProfile(let expected, let actual):
            return "环境检查 profile 不匹配：期望 \(expected)，实际 \(actual)。"
        case .unexpectedTask(let expected, let actual):
            return "环境检查 task 不匹配：期望 \(expected)，实际 \(actual)。"
        }
    }
}

struct BackendSelectionDecision: Equatable {
    let selectedBackend: InferenceBackend?
    let notice: String?
}

enum BackendSelectionPolicy {
    static func decide(
        report: BackendProbeReport,
        savedBackend: InferenceBackend?
    ) -> BackendSelectionDecision {
        if let savedBackend {
            if report.isAvailable(savedBackend) {
                return BackendSelectionDecision(selectedBackend: savedBackend, notice: nil)
            }

            let fallback = preferredAvailableBackend(in: report)
            if let fallback {
                return BackendSelectionDecision(
                    selectedBackend: fallback,
                    notice: "上次选择的 \(savedBackend.displayName) 当前不可用，已切换到 \(fallback.displayName)。"
                )
            }
            return BackendSelectionDecision(
                selectedBackend: nil,
                notice: "上次选择的 \(savedBackend.displayName) 当前不可用，当前没有可用的运行设备。"
            )
        }

        return BackendSelectionDecision(
            selectedBackend: preferredAvailableBackend(in: report),
            notice: nil
        )
    }

    private static func preferredAvailableBackend(in report: BackendProbeReport) -> InferenceBackend? {
        if report.isAvailable(.mlx) {
            return .mlx
        }
        if report.isAvailable(.ct2) {
            return .ct2
        }
        return InferenceBackend.allCases.first(where: report.isAvailable)
    }
}

struct ProbeFailure: LocalizedError, Equatable {
    enum Kind: Equatable {
        case launch
        case timeout
        case invalidOutput
        case unsupportedSchema
    }

    let kind: Kind
    let detail: String

    var summary: String {
        switch kind {
        case .launch:
            return "无法运行环境检查"
        case .timeout:
            return "环境检查超时"
        case .invalidOutput:
            return "无法读取环境检查结果"
        case .unsupportedSchema:
            return "环境检查版本不兼容"
        }
    }

    var errorDescription: String? {
        detail.isEmpty ? summary : "\(summary)：\(detail)"
    }
}

enum ProbeState {
    case idle
    case probing
    case ready(BackendProbeReport, checkedAt: Date)
    case failed(ProbeFailure)
}
