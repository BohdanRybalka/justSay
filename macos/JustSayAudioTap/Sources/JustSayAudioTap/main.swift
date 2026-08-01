// justsay-audiotap — JustSay's macOS system-audio capture helper.
//
// Contract (the other half of it is the module docstring in
// backend/app/audio/macos_tap.py — changing either alone makes the other wrong):
//
//     justsay-audiotap --block-frames <N>
//
//     stdout: {"sample_rate":48000,"channels":2,"format":"f32le","tap_stream_index":0}\n
//             then raw interleaved little-endian float32 frames, forever
//     stderr: log lines, one per line
//     SIGTERM: flush whole blocks, exit 0
//
// `tap_stream_index` is the index, inside the aggregate device's input buffer
// list, of the buffer this process reads. It is in the header because it is the
// one thing the Python half cannot recover from the byte stream: if this helper
// ever reads the wrong buffer, the audio still arrives at the right rate and
// channel count and only *sounds* wrong. The Python half requires the field, so
// a build that goes back to reading buffer 0 blindly fails loudly at startup
// instead of recording the microphone twice.
//
// It captures everything the machine plays through a Core Audio process tap
// (macOS 14.4+), aggregated with the current default output device so the tap
// runs on that device's clock. See
// docs/adr/041-macos-system-audio-comes-from-a-core-audio-tap.md.

import CoreAudio
import Darwin
import Foundation

let defaultBlockFrames = 1024

func logLine(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func fail(_ message: String) -> Never {
    logLine(message)
    exit(1)
}

func parseBlockFrames(_ arguments: [String]) -> Int {
    guard let flagIndex = arguments.firstIndex(of: "--block-frames"),
          flagIndex + 1 < arguments.count,
          let value = Int(arguments[flagIndex + 1]),
          value > 0
    else {
        return defaultBlockFrames
    }
    return value
}

func writeAll(_ bytes: UnsafeRawBufferPointer) -> Bool {
    guard let base = bytes.baseAddress else { return true }
    var offset = 0
    while offset < bytes.count {
        let written = write(STDOUT_FILENO, base.advanced(by: offset), bytes.count - offset)
        if written <= 0 {
            if errno == EINTR { continue }
            return false
        }
        offset += written
    }
    return true
}

func audioObjectProperty<T>(
    _ objectID: AudioObjectID,
    _ selector: AudioObjectPropertySelector,
    _ initial: T
) -> T? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    let status = withUnsafeMutablePointer(to: &value) { pointer in
        AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, pointer)
    }
    return status == noErr ? value : nil
}

func audioObjectIDs(
    _ objectID: AudioObjectID,
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope
) -> [AudioObjectID] {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
    var size = UInt32(0)
    guard AudioObjectGetPropertyDataSize(objectID, &address, 0, nil, &size) == noErr else {
        return []
    }
    let count = Int(size) / MemoryLayout<AudioObjectID>.size
    guard count > 0 else { return [] }
    var ids = [AudioObjectID](repeating: AudioObjectID(kAudioObjectUnknown), count: count)
    let status = ids.withUnsafeMutableBytes { raw in
        AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, raw.baseAddress!)
    }
    return status == noErr ? ids : []
}

func streamFormat(_ streamID: AudioObjectID) -> AudioStreamBasicDescription? {
    audioObjectProperty(streamID, kAudioStreamPropertyVirtualFormat, AudioStreamBasicDescription())
}

/// How many `AudioBuffer`s a run of input streams contributes to an IOProc's
/// buffer list: one per interleaved stream, one per channel for a
/// non-interleaved one.
func bufferCount(of streams: [AudioObjectID]) -> Int? {
    var count = 0
    for stream in streams {
        guard let format = streamFormat(stream) else { return nil }
        if format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0 {
            count += Int(format.mChannelsPerFrame)
        } else {
            count += 1
        }
    }
    return count
}

struct OutputDevice {
    let objectID: AudioObjectID
    let uid: String
}

func defaultOutputDevice() -> OutputDevice? {
    guard let deviceID = audioObjectProperty(
        AudioObjectID(kAudioObjectSystemObject),
        kAudioHardwarePropertyDefaultOutputDevice,
        AudioObjectID(kAudioObjectUnknown)
    ), deviceID != kAudioObjectUnknown else {
        return nil
    }
    guard let uid = audioObjectProperty(
        deviceID, kAudioDevicePropertyDeviceUID, nil as CFString?
    ), let name = uid as String? else {
        return nil
    }
    return OutputDevice(objectID: deviceID, uid: name)
}

final class SystemAudioTap {
    private let blockFrames: Int
    private let queue = DispatchQueue(label: "com.justsay.audiotap.io")
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var pending: [Float] = []
    private var channels = 2
    private var nonInterleaved = false
    private var tapBufferIndex = 0

    init(blockFrames: Int) {
        self.blockFrames = blockFrames
    }

    func start() {
        guard let outputDevice = defaultOutputDevice() else {
            fail("No default output device — there is nothing to capture")
        }
        let outputUID = outputDevice.uid

        let description = CATapDescription(
            stereoGlobalTapButExcludeProcesses: [NSNumber]()
        )
        description.name = "JustSay System Audio Tap"
        description.isPrivate = true
        description.muteBehavior = .unmuted

        let tapStatus = AudioHardwareCreateProcessTap(description, &tapID)
        guard tapStatus == noErr, tapID != kAudioObjectUnknown else {
            fail("AudioHardwareCreateProcessTap failed with status \(tapStatus)")
        }

        guard let format = audioObjectProperty(
            tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription()
        ) else {
            fail("The process tap reported no stream format")
        }
        channels = Int(format.mChannelsPerFrame)
        nonInterleaved = format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        guard format.mFormatID == kAudioFormatLinearPCM,
              format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              format.mBitsPerChannel == 32,
              channels > 0
        else {
            fail("The process tap delivers an unsupported stream format")
        }

        createAggregateDevice(outputUID: outputUID, tapUID: description.uuid.uuidString)
        tapBufferIndex = resolveTapBufferIndex(outputDevice: outputDevice)
        writeHeader(sampleRate: Int(format.mSampleRate))
        startIOProc()
        logLine(
            "Capturing \(channels) ch at \(Int(format.mSampleRate)) Hz "
                + "from input buffer \(tapBufferIndex)"
        )
    }

    /// Which buffer of the aggregate device's input list carries the tap.
    ///
    /// Not `0`. The aggregate's one sub-device is the current default *output*
    /// device, and a USB headset — the single most common meeting setup — is an
    /// output device that also publishes input streams. Those sub-device streams
    /// come before the tap's streams in the aggregate, so `buffers[0]` on such a
    /// machine is the headset microphone: the meeting would be recorded with the
    /// microphone twice and no system audio at all, and would sound plausible
    /// while doing it.
    ///
    /// So the index is derived rather than assumed — it is exactly how many
    /// buffers the sub-device's own input streams contribute — and every step
    /// that could be wrong is checked instead of guessed: the aggregate must
    /// actually have a buffer at that index, and that buffer's stream must carry
    /// the channel count the tap reported. A mismatch means the ordering
    /// assumption above no longer holds, and this process exits loudly rather
    /// than recording the wrong source.
    private func resolveTapBufferIndex(outputDevice: OutputDevice) -> Int {
        let subDeviceStreams = audioObjectIDs(
            outputDevice.objectID, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput
        )
        guard let index = bufferCount(of: subDeviceStreams) else {
            fail("An input stream of the default output device reported no format")
        }

        let aggregateStreams = audioObjectIDs(
            aggregateID, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput
        )
        guard let total = bufferCount(of: aggregateStreams), index < total else {
            fail(
                "The aggregate device exposes no input buffer at index \(index) — "
                    + "the process tap is not where the sub-device layout says it is"
            )
        }

        var offset = 0
        for stream in aggregateStreams {
            guard let format = streamFormat(stream) else {
                fail("An input stream of the aggregate device reported no format")
            }
            let width = format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
                ? Int(format.mChannelsPerFrame)
                : 1
            if offset == index {
                guard Int(format.mChannelsPerFrame) == channels else {
                    fail(
                        "Input buffer \(index) carries \(format.mChannelsPerFrame) channels but "
                            + "the process tap reported \(channels) — refusing to capture it"
                    )
                }
                return index
            }
            offset += width
        }
        fail("The aggregate device's input streams do not line up with its buffer list")
    }

    private func createAggregateDevice(outputUID: String, tapUID: String) {
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "JustSay System Audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [
                [kAudioSubDeviceUIDKey: outputUID]
            ],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUID,
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]

        let status = AudioHardwareCreateAggregateDevice(
            aggregateDescription as CFDictionary, &aggregateID
        )
        guard status == noErr, aggregateID != kAudioObjectUnknown else {
            fail("AudioHardwareCreateAggregateDevice failed with status \(status)")
        }
    }

    private func writeHeader(sampleRate: Int) {
        let header = "{\"sample_rate\":\(sampleRate),\"channels\":\(channels),"
            + "\"format\":\"f32le\",\"tap_stream_index\":\(tapBufferIndex)}\n"
        let written = Data(header.utf8).withUnsafeBytes { writeAll($0) }
        guard written else {
            fail("Could not write the stdout header")
        }
    }

    private func startIOProc() {
        let procStatus = AudioDeviceCreateIOProcIDWithBlock(
            &ioProcID, aggregateID, queue
        ) { [weak self] _, inInputData, _, _, _ in
            self?.consume(inInputData)
        }
        guard procStatus == noErr, let procID = ioProcID else {
            fail("AudioDeviceCreateIOProcIDWithBlock failed with status \(procStatus)")
        }

        let startStatus = AudioDeviceStart(aggregateID, procID)
        guard startStatus == noErr else {
            fail("AudioDeviceStart failed with status \(startStatus)")
        }
    }

    private func consume(_ bufferList: UnsafePointer<AudioBufferList>) {
        let buffers = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: bufferList)
        )
        guard buffers.count > tapBufferIndex else { return }

        if nonInterleaved {
            appendNonInterleaved(buffers)
        } else {
            let buffer = buffers[tapBufferIndex]
            guard let data = buffer.mData, Int(buffer.mNumberChannels) == channels else { return }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            pending.append(
                contentsOf: UnsafeBufferPointer(
                    start: data.assumingMemoryBound(to: Float.self), count: count
                )
            )
        }

        flushWholeBlocks()
    }

    private func appendNonInterleaved(_ buffers: UnsafeMutableAudioBufferListPointer) {
        var planes: [UnsafePointer<Float>] = []
        var frames = Int.max
        guard buffers.count >= tapBufferIndex + channels else { return }
        for index in tapBufferIndex..<(tapBufferIndex + channels) {
            guard let data = buffers[index].mData else { return }
            planes.append(UnsafePointer(data.assumingMemoryBound(to: Float.self)))
            frames = min(frames, Int(buffers[index].mDataByteSize) / MemoryLayout<Float>.size)
        }
        guard !planes.isEmpty, frames > 0, frames != Int.max else { return }

        pending.reserveCapacity(pending.count + frames * channels)
        for frame in 0..<frames {
            for channel in 0..<channels {
                pending.append(planes[min(channel, planes.count - 1)][frame])
            }
        }
    }

    private func flushWholeBlocks() {
        let blockSamples = blockFrames * channels
        guard pending.count >= blockSamples else { return }
        let complete = pending.count - (pending.count % blockSamples)
        let ready = Array(pending[0..<complete])
        pending.removeFirst(complete)

        let written = ready.withUnsafeBytes { writeAll($0) }
        if !written {
            logLine("stdout is closed — stopping")
            DispatchQueue.main.async {
                self.stop()
                exit(0)
            }
        }
    }

    /// Tear the capture down. Must not be called from `queue` — `stop()` waits
    /// on it to flush, and `AudioDeviceStop` from the IOProc's own dispatch
    /// queue is a documented deadlock.
    func stop() {
        queue.sync { flushWholeBlocks() }

        if let procID = ioProcID, aggregateID != kAudioObjectUnknown {
            AudioDeviceStop(aggregateID, procID)
            AudioDeviceDestroyIOProcID(aggregateID, procID)
            ioProcID = nil
        }
        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }
}

signal(SIGPIPE, SIG_IGN)
signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)

let tap = SystemAudioTap(
    blockFrames: parseBlockFrames(Array(CommandLine.arguments.dropFirst()))
)
tap.start()

var signalSources: [DispatchSourceSignal] = []
for signalNumber in [SIGTERM, SIGINT] {
    let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .main)
    source.setEventHandler {
        tap.stop()
        exit(0)
    }
    source.resume()
    signalSources.append(source)
}

dispatchMain()
