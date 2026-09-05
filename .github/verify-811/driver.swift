import Foundation

// Fork-only verification driver for Layr-Labs/d-inference#811.
//
// Compiled together with the UNMODIFIED production file
// provider-swift/Sources/darkbloom/Fan/FanProcessRunner.swift, so the
// behaviour asserted below is the shipped runner's, not a copy of it.
//
// `LocalContentionSnapshot.runCapture` cannot be linked here (it lives in the
// `darkbloom` executable target, which pulls in MLX/Metal), so the two
// runCapture bodies are reproduced verbatim: `preFix` is master's, `fixed` is
// the branch's one-liner over FanProcessRunner.

private final class Box: @unchecked Sendable {
    var value: String?
}

@main
struct Probe {
    /// master's `LocalContentionSnapshot.runCapture`, byte for byte.
    static func preFixRunCapture(_ path: String, args: [String]) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
            p.waitUntilExit()
        } catch {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }

    /// The branch's `LocalContentionSnapshot.runCapture`, byte for byte.
    static func fixedRunCapture(_ path: String, args: [String]) -> String? {
        let result = FanProcessRunner.run(
            path,
            arguments: args,
            timeout: 5,
            discardStandardError: true,
            maximumOutputBytes: 1024 * 1024
        )
        return result.status == -1 ? nil : result.output
    }

    /// Runs `body` off-thread so a deadlock is a reported failure instead of a
    /// wedged job, and reports whether it returned inside `joinAfter` seconds.
    static func offThread(
        joinAfter: Double,
        _ body: @escaping @Sendable () -> String?
    ) -> (returned: Bool, value: String?) {
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            box.value = body()
            done.signal()
        }
        guard done.wait(timeout: .now() + joinAfter) == .success else {
            return (false, nil)
        }
        return (true, box.value)
    }

    static func main() {
        var failures = 0
        func check(_ ok: Bool, _ label: String) {
            print("\(ok ? "PASS" : "FAIL")  \(label)")
            if !ok { failures += 1 }
        }

        // The reporter's machine emitted 78,689 bytes here; a CI runner's
        // process table is small, so the deterministic repro below uses a
        // 200,000-byte child instead. Printed for context only.
        let psSize = fixedRunCapture("/bin/sh", args: ["-c", "/bin/ps -axo comm= | wc -c"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "?"
        print("context: /bin/ps -axo comm= | wc -c => \(psSize) bytes on this runner")

        // 200,000 bytes has to exceed the ~64 KiB Darwin pipe buffer, or the
        // RED case below passes against the unfixed implementation too.
        let chatty = ["-c", "yes darkbloom | head -c 200000"]

        // RED: the reported bug, on this machine.
        let red = offThread(joinAfter: 15) { preFixRunCapture("/bin/sh", args: chatty) }
        check(
            !red.returned,
            "RED  master's runCapture never returns for a child past the pipe buffer"
        )

        // GREEN: same child, the branch's implementation.
        let green = offThread(joinAfter: 20) { fixedRunCapture("/bin/sh", args: chatty) }
        check(green.returned, "GREEN fixed runCapture returns for the same child")
        check(
            green.value?.contains("darkbloom") == true,
            "GREEN fixed runCapture returns the child's output"
        )

        // GREEN: the probe must not lose process names past the runner's 64 KiB
        // default; `ps` was 78,689 bytes on the reporter's Mac.
        let past = offThread(joinAfter: 20) {
            fixedRunCapture("/bin/sh", args: ["-c", "yes darkbloom | head -c 200000; echo ollama"])
        }
        check(
            past.value?.contains("ollama") == true,
            "GREEN the probe keeps process names past the 64 KiB default cap"
        )

        // The nine pre-existing call sites keep the old bound.
        let capped = FanProcessRunner.run(
            "/bin/sh",
            arguments: ["-c", "yes darkbloom | head -c 200000"]
        )
        check(
            capped.output.utf8.count == 64 * 1024,
            "other callers keep the 64 KiB default bound"
        )

        // GREEN: draining alone does not bound a child that never closes
        // stdout; the deadline does.
        let bounded = offThread(joinAfter: 30) {
            fixedRunCapture("/bin/sh", args: ["-c", "sleep 120"])
        }
        check(bounded.returned, "GREEN fixed runCapture honours its 5s deadline")
        check(bounded.value == nil, "GREEN a deadline hit reports nil, not empty output")

        let discarded = FanProcessRunner.run(
            "/bin/sh",
            arguments: ["-c", "echo out; echo err >&2"],
            discardStandardError: true
        )
        check(discarded.output.contains("out"), "stdout survives discardStandardError")
        check(!discarded.output.contains("err"), "stderr is dropped under discardStandardError")

        let merged = FanProcessRunner.run(
            "/bin/sh",
            arguments: ["-c", "echo out; echo err >&2"]
        )
        check(
            merged.output.contains("out") && merged.output.contains("err"),
            "default call sites still get stderr folded into stdout"
        )

        let spawnFailure = FanProcessRunner.run("/nonexistent/darkbloom-probe", arguments: [])
        check(spawnFailure.status == -1, "a spawn failure reports status -1")
        check(
            fixedRunCapture("/nonexistent/darkbloom-probe", args: []) == nil,
            "a spawn failure stays nil, distinguishable from empty output"
        )

        print(failures == 0 ? "\nall checks passed" : "\n\(failures) check(s) failed")
        exit(failures == 0 ? 0 : 1)
    }
}
