import Foundation

final class BackendProbeRunner {
    typealias Completion = (Result<BackendProbeReport, ProbeFailure>) -> Void

    private let timeout: TimeInterval
    private let callbackQueue: DispatchQueue

    init(timeout: TimeInterval = 35, callbackQueue: DispatchQueue = .main) {
        self.timeout = timeout
        self.callbackQueue = callbackQueue
    }

    func probe(
        projectRoot: URL,
        task: InferenceTask,
        completion: @escaping Completion
    ) {
        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stateLock = NSLock()
        var hasCompleted = false

        process.executableURL = projectRoot.appendingPathComponent("dev.sh")
        process.arguments = [
            "python",
            "scripts/backend_probe.py",
            "--profile",
            task.probeProfile,
        ]
        process.currentDirectoryURL = projectRoot
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        var environment = ProcessInfo.processInfo.environment
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        process.environment = environment

        func completeOnce(_ result: Result<BackendProbeReport, ProbeFailure>) {
            stateLock.lock()
            guard !hasCompleted else {
                stateLock.unlock()
                return
            }
            hasCompleted = true
            stateLock.unlock()
            callbackQueue.async {
                completion(result)
            }
        }

        process.terminationHandler = { completedProcess in
            let stdout = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let stderr = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let result = Self.interpret(
                stdout: stdout,
                stderr: stderr,
                terminationStatus: completedProcess.terminationStatus,
                expectedTask: task
            )
            completeOnce(result)
        }

        do {
            try process.run()
        } catch {
            completeOnce(
                .failure(
                    ProbeFailure(
                        kind: .launch,
                        detail: error.localizedDescription
                    )
                )
            )
            return
        }

        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + timeout) {
            stateLock.lock()
            let shouldTerminate = !hasCompleted
            stateLock.unlock()
            guard shouldTerminate else { return }

            completeOnce(
                .failure(
                    ProbeFailure(
                        kind: .timeout,
                        detail: "超过 \(Int(self.timeout)) 秒仍未完成。"
                    )
                )
            )
            if process.isRunning {
                process.terminate()
            }
        }
    }

    static func interpret(
        stdout: Data,
        stderr: Data,
        terminationStatus: Int32,
        expectedTask: InferenceTask
    ) -> Result<BackendProbeReport, ProbeFailure> {
        do {
            let report = try BackendProbeReport.decode(from: stdout, expectedTask: expectedTask)
            return .success(report)
        } catch let contractError as BackendProbeContractError {
            switch contractError {
            case .unsupportedSchema:
                return .failure(
                    ProbeFailure(
                        kind: .unsupportedSchema,
                        detail: contractError.localizedDescription
                    )
                )
            case .unexpectedProfile, .unexpectedTask:
                return .failure(
                    ProbeFailure(
                        kind: .invalidOutput,
                        detail: contractError.localizedDescription
                    )
                )
            }
        } catch {
            let stderrText = String(data: stderr, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let stdoutText = String(data: stdout, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var details = ["退出码 \(terminationStatus)。"]
            if let stderrText, !stderrText.isEmpty {
                details.append("stderr：\(stderrText)")
            } else if let stdoutText, !stdoutText.isEmpty {
                details.append("stdout：\(stdoutText)")
            } else {
                details.append(error.localizedDescription)
            }
            return .failure(
                ProbeFailure(
                    kind: .invalidOutput,
                    detail: details.joined(separator: "\n")
                )
            )
        }
    }
}
