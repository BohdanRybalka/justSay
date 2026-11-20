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
// a build that reads a buffer it never checked fails loudly at startup instead
// of recording something else.
//
// It captures everything the machine plays through a Core Audio process tap
// (macOS 14.4+) inside an aggregate device that holds the tap alone, with no
// output device in it, as Chromium's catap_audio_input_stream.mm does. The
// stream on stdout is always `outputRate`, whatever the output device does:
//
// - On macOS 15+ the aggregate is pinned to `outputRate` and the tap converts
//   the system mix to it.
// - On macOS 14 a tap does not convert between rates ("On macOS 14, CATap does
//   not handle sample rate mismatches", Chromium), so the aggregate runs at the
//   default output device's rate and this helper converts to `outputRate`.
// - Whatever the version, the rate the aggregate's input stream really carries
//   is read back after it is built, and anything other than `outputRate` is
//   converted here. A Bluetooth headset that drops to its 16 or 24 kHz call
//   profile therefore never reaches stdout as mislabelled audio.
// - A change of default output device, of the aggregate's rate or format, or
//   (on macOS 14) of the output device's rate rebuilds the tap and aggregate.
//   Frames already buffered carry over, so stdout simply pauses for the
//   rebuild and goes on at the same rate.
//
// See docs/adr/041-macos-system-audio-comes-from-a-core-audio-tap.md and
// docs/adr/096-a-mac-capture-holds-the-tap-alone-at-a-pinned-rate.md.

import AVFoundation
import CoreAudio
import Darwin
import Foundation

let defaultBlockFrames = 1024
let outputRate: Float64 = 48000
let rebuildSettleSeconds = 0.25
let rebuildRetrySeconds = 1.0
let rebuildAttempts = 10
let steadySeconds = 5.0
let rateSettleSeconds = 1.0

func logLine(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func fail(_ message: String) -> Never {
    logLine(message)
    exit(1)
}

struct CaptureError: Error {
    let message: String
}

func failureMessage(_ error: Error) -> String {
    (error as? CaptureError)?.message ?? "\(error)"
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

func propertyAddress(
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
}

func audioObjectProperty<T>(
    _ objectID: AudioObjectID,
    _ selector: AudioObjectPropertySelector,
    _ initial: T
) -> T? {
    var address = propertyAddress(selector)
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
    var address = propertyAddress(selector, scope: scope)
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

func defaultOutputDeviceID() -> AudioObjectID? {
    guard let deviceID = audioObjectProperty(
        AudioObjectID(kAudioObjectSystemObject),
        kAudioHardwarePropertyDefaultOutputDevice,
        AudioObjectID(kAudioObjectUnknown)
    ), deviceID != kAudioObjectUnknown else {
        return nil
    }
    return deviceID
}

func nominalSampleRate(_ deviceID: AudioObjectID) -> Float64? {
    guard let rate = audioObjectProperty(
        deviceID, kAudioDevicePropertyNominalSampleRate, Float64(0)
    ), rate > 0 else {
        return nil
    }
    return rate
}

/// The tap converts between rates on macOS 15 and later only.
func tapConvertsRates() -> Bool {
    if #available(macOS 15, *) {
        return true
    }
    return false
}

/// One property listener, kept so it can be removed with the block it was added with.
struct PropertyListener {
    let objectID: AudioObjectID
    let address: AudioObjectPropertyAddress
    let block: AudioObjectPropertyListenerBlock
}

final class SystemAudioTap {
    private let blockFrames: Int
    private let queue = DispatchQueue(label: "com.justsay.audiotap.io")
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var pending: [Float] = []
    private let channels = 2
    private var streamChannels = 0
    private var nonInterleaved = false
    private var tapBufferIndex = 0
    private var captureRate = outputRate
    private var converter: AVAudioConverter?
    private var captureListeners: [PropertyListener] = []
    private var outputListener: PropertyListener?
    private var pendingRebuild: DispatchWorkItem?
    private var steadyMark: DispatchWorkItem?
    private var rebuildsSinceSteady = 0
    private var reportedConversionFailure = false

    init(blockFrames: Int) {
        self.blockFrames = blockFrames
    }

    func start() {
        do {
            try buildCapture()
            writeHeader(sampleRate: Int(outputRate))
            try startIOProc()
        } catch {
            tearDownCapture()
            fail(failureMessage(error))
        }
        listenForCaptureChanges()
        listenForOutputChanges()
    }

    /// Tap, aggregate and rate conversion, ready for an IOProc; every field the IOProc reads.
    ///
    /// Runs on the main queue with no IOProc running, so nothing reads those fields meanwhile.
    private func buildCapture() throws {
        let description = CATapDescription(
            stereoGlobalTapButExcludeProcesses: [AudioObjectID]()
        )
        description.name = "JustSay System Audio Tap"
        description.isPrivate = true
        description.muteBehavior = .unmuted

        let tapStatus = AudioHardwareCreateProcessTap(description, &tapID)
        guard tapStatus == noErr, tapID != kAudioObjectUnknown else {
            throw CaptureError(message: "AudioHardwareCreateProcessTap failed with status \(tapStatus)")
        }

        guard let format = audioObjectProperty(
            tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription()
        ) else {
            throw CaptureError(message: "The process tap reported no stream format")
        }
        let tapChannels = Int(format.mChannelsPerFrame)
        guard format.mFormatID == kAudioFormatLinearPCM,
              format.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              format.mBitsPerChannel == 32,
              tapChannels > 0
        else {
            throw CaptureError(message: "The process tap delivers an unsupported stream format")
        }
        streamChannels = tapChannels

        try createAggregateDevice(tapUID: description.uuid.uuidString)
        try pinAggregateRate()
        let tapStream = try resolveTapBufferIndex()
        tapBufferIndex = tapStream.index
        nonInterleaved = tapStream.format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        captureRate = tapStream.format.mSampleRate
        converter = try makeConverter(from: captureRate)
        reportedConversionFailure = false

        logLine(
            "Capturing \(streamChannels) ch at \(Int(captureRate)) Hz from input buffer "
                + "\(tapBufferIndex), writing \(channels) ch at \(Int(outputRate)) Hz"
        )
    }

    /// Set the rate the aggregate runs at: `outputRate`, or the output device's own on macOS 14.
    ///
    /// On macOS 15+ a refused rate is only logged: the tap converts, and the rate the
    /// stream really carries is read back and converted from. On macOS 14 the tap does
    /// not convert, so a device rate that cannot be read or is not taken up is an error.
    private func pinAggregateRate() throws {
        var rate = outputRate
        if !tapConvertsRates() {
            guard let deviceID = defaultOutputDeviceID(), let deviceRate = nominalSampleRate(deviceID)
            else {
                throw CaptureError(
                    message: "The default output device's rate cannot be read, and on macOS 14 "
                        + "the tap runs at it"
                )
            }
            rate = deviceRate
        }
        var address = propertyAddress(kAudioDevicePropertyNominalSampleRate)
        let status = AudioObjectSetPropertyData(
            aggregateID, &address, 0, nil, UInt32(MemoryLayout<Float64>.size), &rate
        )
        if status != noErr {
            guard tapConvertsRates() else {
                throw CaptureError(
                    message: "The aggregate device refused \(Int(rate)) Hz with status \(status) "
                        + "on macOS 14, where the tap does not convert"
                )
            }
            logLine("Setting the aggregate device to \(Int(rate)) Hz failed with status \(status)")
            return
        }
        let deadline = Date().addingTimeInterval(rateSettleSeconds)
        while nominalSampleRate(aggregateID) != rate, Date() < deadline {
            usleep(20_000)
        }
        if !tapConvertsRates(), nominalSampleRate(aggregateID) != rate {
            throw CaptureError(
                message: "The aggregate device did not settle at the output device's "
                    + "\(Int(rate)) Hz on macOS 14, where the tap does not convert"
            )
        }
    }

    /// Which buffer of the aggregate device's input list carries the tap, and its format.
    ///
    /// The aggregate holds the tap and nothing else, so the tap is its first input
    /// stream and buffer 0. That is checked rather than assumed: the stream must
    /// exist, report a format, and carry the channel count the tap reported. A
    /// mismatch means the aggregate holds something this helper did not put there,
    /// and it refuses to record rather than record the wrong source.
    private func resolveTapBufferIndex() throws -> (index: Int, format: AudioStreamBasicDescription) {
        let aggregateStreams = audioObjectIDs(
            aggregateID, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput
        )
        guard aggregateStreams.count == 1, let format = streamFormat(aggregateStreams[0]) else {
            throw CaptureError(
                message: "The aggregate device exposes \(aggregateStreams.count) input streams "
                    + "with a readable format, not the tap's one"
            )
        }
        guard Int(format.mChannelsPerFrame) == streamChannels else {
            throw CaptureError(
                message: "Input buffer 0 carries \(format.mChannelsPerFrame) channels but "
                    + "the process tap reported \(streamChannels) — refusing to capture it"
            )
        }
        guard format.mSampleRate > 0 else {
            throw CaptureError(message: "The aggregate device's input stream reports no sample rate")
        }
        return (0, format)
    }

    private func createAggregateDevice(tapUID: String) throws {
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "JustSay System Audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceTapAutoStartKey: true,
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
            throw CaptureError(message: "AudioHardwareCreateAggregateDevice failed with status \(status)")
        }
    }

    /// A converter from `rate` to `outputRate`, or none when the two already agree.
    private func makeConverter(from rate: Float64) throws -> AVAudioConverter? {
        guard rate != outputRate else { return nil }
        guard let source = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: rate,
            channels: AVAudioChannelCount(channels), interleaved: true
        ), let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: outputRate,
            channels: AVAudioChannelCount(channels), interleaved: true
        ), let made = AVAudioConverter(from: source, to: target) else {
            throw CaptureError(
                message: "No converter from \(Int(rate)) Hz to \(Int(outputRate)) Hz"
            )
        }
        return made
    }

    private func writeHeader(sampleRate: Int) {
        let header = "{\"sample_rate\":\(sampleRate),\"channels\":\(channels),"
            + "\"format\":\"f32le\",\"tap_stream_index\":\(tapBufferIndex)}\n"
        let written = Data(header.utf8).withUnsafeBytes { writeAll($0) }
        guard written else {
            fail("Could not write the stdout header")
        }
    }

    private func startIOProc() throws {
        let procStatus = AudioDeviceCreateIOProcIDWithBlock(
            &ioProcID, aggregateID, queue
        ) { [weak self] _, inInputData, _, _, _ in
            self?.consume(inInputData)
        }
        guard procStatus == noErr, let procID = ioProcID else {
            ioProcID = nil
            throw CaptureError(
                message: "AudioDeviceCreateIOProcIDWithBlock failed with status \(procStatus)"
            )
        }

        let startStatus = AudioDeviceStart(aggregateID, procID)
        guard startStatus == noErr else {
            throw CaptureError(message: "AudioDeviceStart failed with status \(startStatus)")
        }
    }

    private func consume(_ bufferList: UnsafePointer<AudioBufferList>) {
        let buffers = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: bufferList)
        )
        guard buffers.count > tapBufferIndex else { return }

        var frames: [Float] = []
        if nonInterleaved {
            frames = remixed(interleaved(buffers), from: streamChannels)
        } else {
            let buffer = buffers[tapBufferIndex]
            guard let data = buffer.mData else { return }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            let samples = Array(
                UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: count)
            )
            frames = remixed(samples, from: Int(buffer.mNumberChannels))
        }
        guard !frames.isEmpty else { return }

        if let converter {
            pending.append(contentsOf: convert(frames, with: converter))
        } else {
            pending.append(contentsOf: frames)
        }
        flushWholeBlocks()
    }

    /// The stream's `streamChannels` planes, interleaved.
    private func interleaved(_ buffers: UnsafeMutableAudioBufferListPointer) -> [Float] {
        var planes: [UnsafePointer<Float>] = []
        var frames = Int.max
        guard buffers.count >= tapBufferIndex + streamChannels else { return [] }
        for index in tapBufferIndex..<(tapBufferIndex + streamChannels) {
            guard let data = buffers[index].mData else { return [] }
            planes.append(UnsafePointer(data.assumingMemoryBound(to: Float.self)))
            frames = min(frames, Int(buffers[index].mDataByteSize) / MemoryLayout<Float>.size)
        }
        guard !planes.isEmpty, frames > 0, frames != Int.max else { return [] }

        var samples: [Float] = []
        samples.reserveCapacity(frames * planes.count)
        for frame in 0..<frames {
            for plane in planes {
                samples.append(plane[frame])
            }
        }
        return samples
    }

    /// Interleaved `samples` of `delivered` channels as whole frames of the header's `channels`.
    ///
    /// A mono stream is copied to every channel, from a wider one the first channels are
    /// kept, and a trailing partial frame is dropped, so every frame written has the
    /// width the reader cuts blocks at.
    private func remixed(_ samples: [Float], from delivered: Int) -> [Float] {
        guard delivered > 0 else { return [] }
        let frames = samples.count / delivered
        var output: [Float] = []
        output.reserveCapacity(frames * channels)
        for frame in 0..<frames {
            for channel in 0..<channels {
                output.append(samples[frame * delivered + min(channel, delivered - 1)])
            }
        }
        return output
    }

    /// `samples` at `captureRate`, interleaved, as the same audio at `outputRate`.
    ///
    /// The converter keeps its filter state between calls: it is handed this one
    /// buffer and then told no more input is available yet, never that the stream ended.
    private func convert(_ samples: [Float], with converter: AVAudioConverter) -> [Float] {
        let frameCount = samples.count / channels
        guard frameCount > 0,
              let input = AVAudioPCMBuffer(
                  pcmFormat: converter.inputFormat, frameCapacity: AVAudioFrameCount(frameCount)
              ),
              let output = AVAudioPCMBuffer(
                  pcmFormat: converter.outputFormat,
                  frameCapacity: AVAudioFrameCount(
                      (Double(frameCount) * outputRate / captureRate).rounded(.up)
                  ) + 64
              )
        else {
            return []
        }
        input.frameLength = AVAudioFrameCount(frameCount)
        let byteCount = frameCount * channels * MemoryLayout<Float>.size
        samples.withUnsafeBytes { source in
            if let destination = input.mutableAudioBufferList.pointee.mBuffers.mData,
               let base = source.baseAddress {
                memcpy(destination, base, byteCount)
            }
        }

        var handedOver = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            if handedOver {
                inputStatus.pointee = .noDataNow
                return nil
            }
            handedOver = true
            inputStatus.pointee = .haveData
            return input
        }
        guard status != .error, output.frameLength > 0,
              let data = output.audioBufferList.pointee.mBuffers.mData
        else {
            if status == .error, !reportedConversionFailure {
                reportedConversionFailure = true
                logLine(
                    "Converting \(Int(captureRate)) Hz to \(Int(outputRate)) Hz failed: "
                        + "\(conversionError.map { "\($0)" } ?? "no reason given")"
                )
                DispatchQueue.main.async { [weak self] in
                    self?.scheduleRebuild("a rate conversion failed", onlyIfChanged: false)
                }
            }
            return []
        }
        let produced = Int(output.frameLength) * channels
        return Array(
            UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: produced)
        )
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

    /// Rebuild when what the capture runs at changes: the aggregate's rate or its stream's format,
    /// and on macOS 14 the default output device's rate, which the aggregate was built at.
    ///
    /// A change between building the capture and listening is caught by checking once now.
    private func listenForCaptureChanges() {
        var watched: [(AudioObjectID, AudioObjectPropertyAddress)] = [
            (aggregateID, propertyAddress(kAudioDevicePropertyNominalSampleRate))
        ]
        for stream in audioObjectIDs(
            aggregateID, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput
        ) {
            watched.append((stream, propertyAddress(kAudioStreamPropertyVirtualFormat)))
        }
        if !tapConvertsRates(), let deviceID = defaultOutputDeviceID() {
            watched.append((deviceID, propertyAddress(kAudioDevicePropertyNominalSampleRate)))
        }
        for (objectID, address) in watched {
            if let listener = addListener(
                objectID, address, reason: "a capture rate or format changed", onlyIfChanged: true
            ) {
                captureListeners.append(listener)
            }
        }
        if !captureIsCurrent() {
            scheduleRebuild("a capture rate or format changed while it was built", onlyIfChanged: true)
        }
    }

    /// Whether the capture still runs at the rate it was built for.
    ///
    /// Core Audio repeats notifications and applies a rate set at build time a moment
    /// later, so a notification alone is no reason to rebuild.
    private func captureIsCurrent() -> Bool {
        let streams = audioObjectIDs(
            aggregateID, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput
        )
        guard streams.count == 1, let format = streamFormat(streams[0]),
              format.mSampleRate == captureRate,
              Int(format.mChannelsPerFrame) == streamChannels
        else {
            return false
        }
        if tapConvertsRates() {
            return true
        }
        guard let deviceID = defaultOutputDeviceID() else { return false }
        return nominalSampleRate(deviceID) == captureRate
            && nominalSampleRate(aggregateID) == captureRate
    }

    /// Rebuild on every change of default output device: a headset connecting, disconnecting,
    /// or being picked by hand. The tap holds no device, but the macOS 14 rate follows it.
    private func listenForOutputChanges() {
        outputListener = addListener(
            AudioObjectID(kAudioObjectSystemObject),
            propertyAddress(kAudioHardwarePropertyDefaultOutputDevice),
            reason: "the default output device changed",
            onlyIfChanged: false
        )
    }

    private func addListener(
        _ objectID: AudioObjectID,
        _ address: AudioObjectPropertyAddress,
        reason: String,
        onlyIfChanged: Bool
    ) -> PropertyListener? {
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.scheduleRebuild(reason, onlyIfChanged: onlyIfChanged)
        }
        var mutableAddress = address
        let status = AudioObjectAddPropertyListenerBlock(objectID, &mutableAddress, .main, block)
        guard status == noErr else {
            logLine("Listening for \(reason) failed with status \(status)")
            return nil
        }
        return PropertyListener(objectID: objectID, address: address, block: block)
    }

    private func removeListener(_ listener: PropertyListener) {
        var address = listener.address
        AudioObjectRemovePropertyListenerBlock(listener.objectID, &address, .main, listener.block)
    }

    /// One rebuild for a burst of notifications: a headset switch fires several at once.
    ///
    /// A device change always rebuilds; a rate or format notification only when the
    /// capture no longer runs at the rate it was built for once the burst has settled.
    private func scheduleRebuild(_ reason: String, onlyIfChanged: Bool) {
        if !onlyIfChanged || pendingRebuild == nil {
            pendingRebuild?.cancel()
            let work = DispatchWorkItem { [weak self] in
                guard let self else { return }
                if onlyIfChanged, self.captureIsCurrent() {
                    self.pendingRebuild = nil
                    return
                }
                self.rebuild(reason)
            }
            pendingRebuild = work
            DispatchQueue.main.asyncAfter(deadline: .now() + rebuildSettleSeconds, execute: work)
        }
    }

    /// Tear the capture down and build it again on whatever the machine now has.
    ///
    /// Frames already buffered stay, and nothing is written until the new IOProc runs. A failed
    /// build is retried every `rebuildRetrySeconds`; once `rebuildAttempts` rebuilds pass without
    /// a capture holding for `steadySeconds`, this process exits and the Python half reports it.
    private func rebuild(_ reason: String) {
        pendingRebuild = nil
        steadyMark?.cancel()
        guard rebuildsSinceSteady < rebuildAttempts else {
            fail(
                "The capture was rebuilt \(rebuildsSinceSteady) times without holding for "
                    + "\(Int(steadySeconds)) s; giving up after \(reason)"
            )
        }
        rebuildsSinceSteady += 1
        let previousRate = captureRate
        for listener in captureListeners {
            removeListener(listener)
        }
        captureListeners = []
        tearDownCapture()
        do {
            try buildCapture()
            try startIOProc()
        } catch {
            tearDownCapture()
            logLine("Rebuilding the capture after \(reason) failed, retrying: \(failureMessage(error))")
            let retry = DispatchWorkItem { [weak self] in self?.rebuild(reason) }
            pendingRebuild = retry
            DispatchQueue.main.asyncAfter(deadline: .now() + rebuildRetrySeconds, execute: retry)
            return
        }
        let steady = DispatchWorkItem { [weak self] in self?.rebuildsSinceSteady = 0 }
        steadyMark = steady
        DispatchQueue.main.asyncAfter(deadline: .now() + steadySeconds, execute: steady)
        listenForCaptureChanges()
        logLine(
            "Rebuilt the capture after \(reason): \(Int(previousRate)) Hz before, "
                + "\(Int(captureRate)) Hz now"
        )
    }

    /// Stop the IOProc, let a block already queued finish, then destroy aggregate and tap.
    ///
    /// Must not be called from `queue`: `AudioDeviceStop` from the IOProc's own
    /// dispatch queue is a documented deadlock, and so is `queue.sync` from it.
    private func tearDownCapture() {
        if let procID = ioProcID, aggregateID != kAudioObjectUnknown {
            let stopStatus = AudioDeviceStop(aggregateID, procID)
            if stopStatus != noErr {
                logLine("AudioDeviceStop failed with status \(stopStatus); destroying the device anyway")
            }
            AudioDeviceDestroyIOProcID(aggregateID, procID)
            ioProcID = nil
        }
        queue.sync {}
        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }

    /// Tear the capture down. Must not be called from `queue` — `stop()` waits
    /// on it to flush, and `AudioDeviceStop` from the IOProc's own dispatch
    /// queue is a documented deadlock.
    func stop() {
        queue.sync { flushWholeBlocks() }

        pendingRebuild?.cancel()
        pendingRebuild = nil
        if let listener = outputListener {
            removeListener(listener)
            outputListener = nil
        }
        for listener in captureListeners {
            removeListener(listener)
        }
        captureListeners = []
        tearDownCapture()
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
