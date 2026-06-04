// ownscribe-transcribe — on-device speech-to-text using FluidAudio (Parakeet TDT).
//
// Usage:
//   ownscribe-transcribe <audio.wav> --output <result.json> [--model v2|v3]
//
// Reads an audio file, runs Parakeet ASR (word-level timings via token merge), and
// writes the result as JSON to the --output path. Progress markers are written to
// stderr; stdout is left clean. Diarization is handled in Python via pyannote.
//
// Verified against FluidAudio v0.15.0 source (checked-out package). Uses an async
// @main entry (not a semaphore-blocked main) so CoreML/AVFoundation work that
// dispatches to the main queue cannot deadlock.

import AVFoundation
import FluidAudio
import Foundation

// MARK: - JSON output model (parsed by the Python ParakeetTranscriber)

struct WordOut: Codable {
    let word: String
    let start: TimeInterval
    let end: TimeInterval
    let confidence: Float
}

struct TranscriptionOut: Codable {
    let text: String
    let language: String
    let duration: TimeInterval
    let processingTime: TimeInterval
    let rtfx: Float
    let modelVersion: String
    let words: [WordOut]
}

// MARK: - Token -> word merge (copied verbatim from FluidAudio's CLI WordTimingMerger)

enum WordTimingMerger {
    static func mergeTokensIntoWords(_ tokenTimings: [TokenTiming]) -> [WordOut] {
        guard !tokenTimings.isEmpty else { return [] }

        var words: [WordOut] = []
        var currentWord = ""
        var currentStartTime: TimeInterval?
        var currentEndTime: TimeInterval = 0
        var currentConfidences: [Float] = []

        for timing in tokenTimings {
            let token = timing.token
            if token.hasPrefix(" ") || token.hasPrefix("\n") || token.hasPrefix("\t") {
                if !currentWord.isEmpty, let startTime = currentStartTime {
                    words.append(
                        WordOut(
                            word: currentWord, start: startTime, end: currentEndTime,
                            confidence: averageConfidence(currentConfidences)))
                }
                currentWord = token.trimmingCharacters(in: .whitespacesAndNewlines)
                currentStartTime = timing.startTime
                currentEndTime = timing.endTime
                currentConfidences = [timing.confidence]
            } else {
                if currentStartTime == nil { currentStartTime = timing.startTime }
                currentWord += token
                currentEndTime = timing.endTime
                currentConfidences.append(timing.confidence)
            }
        }

        if !currentWord.isEmpty, let startTime = currentStartTime {
            words.append(
                WordOut(
                    word: currentWord, start: startTime, end: currentEndTime,
                    confidence: averageConfidence(currentConfidences)))
        }
        return words
    }

    private static func averageConfidence(_ confidences: [Float]) -> Float {
        confidences.isEmpty ? 0.0 : confidences.reduce(0, +) / Float(confidences.count)
    }
}

// MARK: - Helpers

func progress(_ marker: String) {
    FileHandle.standardError.write(Data((marker + "\n").utf8))
}

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("[ERROR] " + message + "\n").utf8))
    exit(1)
}

func modelVersion(from name: String) -> AsrModelVersion {
    name.lowercased() == "v3" ? .v3 : .v2
}

// MARK: - Entry

@main
struct OwnscribeTranscribe {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            die("usage: ownscribe-transcribe <audio> --output <json> [--model v2|v3]")
        }

        var audioPath: String?
        var outputPath: String?
        var modelName = "v2"

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--output", "-o":
                i += 1
                guard i < args.count else { die("--output requires a path") }
                outputPath = args[i]
            case "--model", "-m":
                i += 1
                guard i < args.count else { die("--model requires v2 or v3") }
                modelName = args[i]
            default:
                if audioPath == nil { audioPath = args[i] } else { die("unexpected argument: \(args[i])") }
            }
            i += 1
        }

        guard let audioPath else { die("no audio file given") }
        guard let outputPath else { die("--output <json> is required") }

        let audioURL = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            die("audio file not found: \(audioPath)")
        }

        let version = modelVersion(from: modelName)
        let modelLabel = version == .v3 ? "v3" : "v2"

        do {
            // 1) Load Parakeet ASR models (downloads on first run, then cached).
            progress("[MODEL_LOADING]")
            let models = try await AsrModels.downloadAndLoad(version: version)

            let asrConfig = ASRConfig(
                tdtConfig: TdtConfig(blankId: version.blankId),
                encoderHiddenSize: version.encoderHiddenSize
            )
            let asr = AsrManager(config: asrConfig)
            try await asr.loadModels(models)

            // 2) Transcribe (URL overload resamples + auto-streams long files).
            progress("[TRANSCRIBING]")
            var decoderState = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
            let start = Date()
            let result = try await asr.transcribe(audioURL, decoderState: &decoderState)
            let processingTime = Date().timeIntervalSince(start)

            let words = WordTimingMerger.mergeTokensIntoWords(result.tokenTimings ?? [])

            // ASRResult.duration can be 0 on the URL path; derive the true audio
            // duration from the file (fall back to the last word's end time).
            var audioDuration = result.duration
            if audioDuration <= 0 {
                if let file = try? AVAudioFile(forReading: audioURL) {
                    audioDuration = Double(file.length) / file.processingFormat.sampleRate
                }
                if audioDuration <= 0 { audioDuration = words.last?.end ?? 0 }
            }

            // 3) Emit JSON.
            let out = TranscriptionOut(
                text: result.text,
                language: version == .v2 ? "en" : "",
                duration: audioDuration,
                processingTime: processingTime,
                rtfx: processingTime > 0 ? Float(audioDuration / processingTime) : 0,
                modelVersion: modelLabel,
                words: words
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(out)
            try data.write(to: URL(fileURLWithPath: outputPath))
            progress("[DONE]")
        } catch {
            die("\(error)")
        }
    }
}
