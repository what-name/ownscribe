// ownscribe-transcribe — on-device speech-to-text using FluidAudio (Parakeet TDT).
//
// Usage:
//   ownscribe-transcribe <audio.wav> --output <result.json> [--diarize] [--model v2|v3]
//
// Reads an audio file, runs Parakeet ASR (word-level timings via token merge) and,
// when --diarize is set, FluidAudio speaker diarization. Word->speaker labels are
// assigned by time overlap (the library provides no built-in alignment). The result
// is written as JSON to the --output path. Progress markers are written to stderr;
// stdout is left clean.
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
    var speaker: String?
}

struct TranscriptionOut: Codable {
    let text: String
    let language: String
    let duration: TimeInterval
    let processingTime: TimeInterval
    let rtfx: Float
    let modelVersion: String
    let diarized: Bool
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
                            confidence: averageConfidence(currentConfidences), speaker: nil))
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
                    confidence: averageConfidence(currentConfidences), speaker: nil))
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

func distance(_ t: Double, _ seg: TimedSpeakerSegment) -> Double {
    let lo = Double(seg.startTimeSeconds), hi = Double(seg.endTimeSeconds)
    if t < lo { return lo - t }
    if t > hi { return t - hi }
    return 0
}

/// Assign a speaker label to each word by overlapping its midpoint with diarization
/// segments. Falls back to the nearest segment when no segment contains the midpoint.
func assignSpeakers(_ words: [WordOut], segments: [TimedSpeakerSegment]) -> [WordOut] {
    guard !segments.isEmpty else { return words }
    return words.map { w in
        var w = w
        let mid = (w.start + w.end) / 2.0
        if let seg = segments.first(where: {
            Double($0.startTimeSeconds) <= mid && mid <= Double($0.endTimeSeconds)
        }) {
            w.speaker = seg.speakerId
        } else {
            w.speaker = segments.min(by: { distance(mid, $0) < distance(mid, $1) })?.speakerId
        }
        return w
    }
}

// MARK: - Entry

@main
struct OwnscribeTranscribe {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            die("usage: ownscribe-transcribe <audio> --output <json> [--diarize] [--model v2|v3]")
        }

        var audioPath: String?
        var outputPath: String?
        var doDiarize = false
        var modelName = "v2"
        var clusterThreshold: Float?

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--output", "-o":
                i += 1
                guard i < args.count else { die("--output requires a path") }
                outputPath = args[i]
            case "--diarize":
                doDiarize = true
            case "--model", "-m":
                i += 1
                guard i < args.count else { die("--model requires v2 or v3") }
                modelName = args[i]
            case "--cluster-threshold":
                i += 1
                guard i < args.count, let v = Float(args[i]) else { die("--cluster-threshold requires a number") }
                clusterThreshold = v
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

            var words = WordTimingMerger.mergeTokensIntoWords(result.tokenTimings ?? [])

            // ASRResult.duration can be 0 on the URL path; derive the true audio
            // duration from the file (fall back to the last word's end time).
            var audioDuration = result.duration
            if audioDuration <= 0 {
                if let file = try? AVAudioFile(forReading: audioURL) {
                    audioDuration = Double(file.length) / file.processingFormat.sampleRate
                }
                if audioDuration <= 0 { audioDuration = words.last?.end ?? 0 }
            }

            // 3) Optional speaker diarization + midpoint alignment.
            if doDiarize {
                progress("[DIARIZING]")
                let diarModels = try await DiarizerModels.downloadIfNeeded()
                var diarConfig = DiarizerConfig()
                if let clusterThreshold { diarConfig.clusteringThreshold = clusterThreshold }
                let diarizer = DiarizerManager(config: diarConfig)
                diarizer.initialize(models: diarModels)
                let samples = try AudioConverter().resampleAudioFile(audioURL)
                let diarization = try diarizer.performCompleteDiarization(samples)
                words = assignSpeakers(words, segments: diarization.segments)
            }

            // 4) Emit JSON.
            let out = TranscriptionOut(
                text: result.text,
                language: version == .v2 ? "en" : "",
                duration: audioDuration,
                processingTime: processingTime,
                rtfx: processingTime > 0 ? Float(audioDuration / processingTime) : 0,
                modelVersion: modelLabel,
                diarized: doDiarize,
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
