import Foundation

private struct TestFailure: Error, CustomStringConvertible {
    let description: String
}

private func expect(
    _ condition: @autoclosure () -> Bool,
    _ message: String,
    file: StaticString = #filePath,
    line: UInt = #line
) throws {
    guard condition() else {
        throw TestFailure(description: "\(file):\(line): \(message)")
    }
}

private func makeReportData(
    schemaVersion: Int = 1,
    ct2Available: Bool,
    mlxAvailable: Bool,
    includeUnknownBackend: Bool = false
) throws -> Data {
    func availability(
        available: Bool,
        device: String,
        variant: String
    ) -> [String: Any] {
        [
            "available": available,
            "capabilities": [
                "translate": true,
                "transcribe": true,
                "word_timestamps": device == "cpu",
                "batching": device == "cpu",
            ],
            "device": device,
            "model": [
                "path": "/models/\(variant)",
                "profile": "translate",
                "variant": variant,
            ],
            "reasons": available ? [] : ["not available"],
        ]
    }

    var backends: [String: Any] = [
        "ct2": availability(available: ct2Available, device: "cpu", variant: "int8"),
        "mlx": availability(available: mlxAvailable, device: "gpu", variant: "fp16"),
    ]
    if includeUnknownBackend {
        backends["future_backend"] = availability(
            available: true,
            device: "accelerator",
            variant: "future"
        )
    }

    return try JSONSerialization.data(
        withJSONObject: [
            "schema_version": schemaVersion,
            "profile": "translate",
            "language": "ja",
            "task": "translate",
            "backends": backends,
        ],
        options: [.sortedKeys]
    )
}

@main
private enum LauncherLogicTests {
    static func main() throws {
        try testSchemaVersionOneDecodes()
        try testExitCodeOneWithValidJSONIsAccepted()
        try testGPUIsDefaultWithoutHistory()
        try testSavedCPUIsRestored()
        try testUnavailableSavedGPUSwitchesToCPUWithNotice()
        try testNoAvailableBackendLeavesSelectionEmpty()
        try testUnknownBackendDoesNotBreakKnownBackendDecoding()
        try testUnsupportedSchemaIsReportedClearly()
        print("LauncherLogicTests: 8 tests passed")
    }

    private static func testSchemaVersionOneDecodes() throws {
        let report = try BackendProbeReport.decode(
            from: makeReportData(ct2Available: true, mlxAvailable: true),
            expectedTask: .translate
        )
        try expect(report.schemaVersion == 1, "schema version should decode")
        try expect(report.availability(for: .ct2)?.model.variant == "int8", "CT2 model should decode")
        try expect(report.availability(for: .mlx)?.model.variant == "fp16", "MLX model should decode")
    }

    private static func testExitCodeOneWithValidJSONIsAccepted() throws {
        let result = BackendProbeRunner.interpret(
            stdout: try makeReportData(ct2Available: false, mlxAvailable: false),
            stderr: Data("runtime warning".utf8),
            terminationStatus: 1,
            expectedTask: .translate
        )
        guard case .success(let report) = result else {
            throw TestFailure(description: "valid JSON must win over exit code 1")
        }
        try expect(!report.isAvailable(.ct2), "CT2 should remain unavailable")
        try expect(!report.isAvailable(.mlx), "MLX should remain unavailable")
    }

    private static func testGPUIsDefaultWithoutHistory() throws {
        let report = try BackendProbeReport.decode(
            from: makeReportData(ct2Available: true, mlxAvailable: true)
        )
        let decision = BackendSelectionPolicy.decide(report: report, savedBackend: nil)
        try expect(decision.selectedBackend == .mlx, "GPU should be recommended by default")
        try expect(decision.notice == nil, "default recommendation should not show a warning")
    }

    private static func testSavedCPUIsRestored() throws {
        let report = try BackendProbeReport.decode(
            from: makeReportData(ct2Available: true, mlxAvailable: true)
        )
        let decision = BackendSelectionPolicy.decide(report: report, savedBackend: .ct2)
        try expect(decision.selectedBackend == .ct2, "saved CPU should be restored")
        try expect(decision.notice == nil, "restored selection should not show a warning")
    }

    private static func testUnavailableSavedGPUSwitchesToCPUWithNotice() throws {
        let report = try BackendProbeReport.decode(
            from: makeReportData(ct2Available: true, mlxAvailable: false)
        )
        let decision = BackendSelectionPolicy.decide(report: report, savedBackend: .mlx)
        try expect(decision.selectedBackend == .ct2, "unavailable saved GPU should switch to CPU")
        try expect(
            decision.notice == "上次选择的 GPU 当前不可用，已切换到 CPU。",
            "automatic fallback should be visible"
        )
    }

    private static func testNoAvailableBackendLeavesSelectionEmpty() throws {
        let report = try BackendProbeReport.decode(
            from: makeReportData(ct2Available: false, mlxAvailable: false)
        )
        let decision = BackendSelectionPolicy.decide(report: report, savedBackend: nil)
        try expect(decision.selectedBackend == nil, "no backend should be selected")
    }

    private static func testUnknownBackendDoesNotBreakKnownBackendDecoding() throws {
        let report = try BackendProbeReport.decode(
            from: makeReportData(
                ct2Available: true,
                mlxAvailable: false,
                includeUnknownBackend: true
            )
        )
        try expect(report.isAvailable(.ct2), "known CT2 field should still decode")
        try expect(report.backends["future_backend"]?.available == true, "unknown backend should be tolerated")
    }

    private static func testUnsupportedSchemaIsReportedClearly() throws {
        let result = BackendProbeRunner.interpret(
            stdout: try makeReportData(
                schemaVersion: 2,
                ct2Available: true,
                mlxAvailable: true
            ),
            stderr: Data(),
            terminationStatus: 0,
            expectedTask: .translate
        )
        guard case .failure(let failure) = result else {
            throw TestFailure(description: "unsupported schema should fail")
        }
        try expect(failure.kind == .unsupportedSchema, "failure kind should identify schema mismatch")
        try expect(failure.summary == "环境检查版本不兼容", "failure summary should be localized")
    }
}
