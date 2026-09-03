// FORK(JuanColilla): R-56 expert streaming — offset index over the original
// safetensors shards (design "Option A": no repack).
//
// Every routed-expert tensor of a Qwen 3.5 MoE checkpoint carries the expert
// on axis 0 and is stored row-major, so expert `i` of a given tensor is the
// contiguous byte range `[i*rowBytes, (i+1)*rowBytes)` inside its shard. This
// index is the map from `(layer, projection, component)` to that range, built
// once from the shard headers and persisted next to the model.
//
// TD-050: the index is only valid for that layout. A checkpoint that stores
// experts as separate `experts.N.*` tensors, or that fuses gate and up into a
// single `gate_up_proj`, is rejected loudly here rather than degrading into a
// silently wrong read.

import Foundation

// MARK: - Geometry

/// One of the three projections of a routed expert MLP.
public enum ExpertProjection: String, Codable, Sendable, CaseIterable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"
}

/// One of the three arrays an affine-quantized projection is stored as.
public enum ExpertComponent: String, Codable, Sendable, CaseIterable {
    case weight
    case scales
    case biases
}

/// One of the nine contiguous byte ranges that make up a single expert.
///
/// The raw value is the canonical slot used by every array indexed by piece
/// (staging buffers, slot-bank pools, read batches), so it is stable and
/// serialized.
public struct ExpertPiece: Hashable, Sendable {
    public let projection: ExpertProjection
    public let component: ExpertComponent

    public init(_ projection: ExpertProjection, _ component: ExpertComponent) {
        self.projection = projection
        self.component = component
    }

    /// The nine pieces in canonical order.
    public static let all: [ExpertPiece] = ExpertProjection.allCases.flatMap { projection in
        ExpertComponent.allCases.map { ExpertPiece(projection, $0) }
    }

    public var slot: Int {
        let projectionIndex = ExpertProjection.allCases.firstIndex(of: projection)!
        let componentIndex = ExpertComponent.allCases.firstIndex(of: component)!
        return projectionIndex * ExpertComponent.allCases.count + componentIndex
    }
}

/// The subset of safetensors dtypes a quantized MoE checkpoint uses, with the
/// item sizes the index validates against.
///
/// Kept as its own enum rather than MLX's `DType` so index construction and
/// validation stay pure: no MLX array is created, which is what lets these
/// tests run where no Metal device exists.
public enum ExpertDType: String, Codable, Sendable {
    case uint32 = "U32"
    case uint8 = "U8"
    case bfloat16 = "BF16"
    case float16 = "F16"
    case float32 = "F32"

    public var itemSize: Int {
        switch self {
        case .uint8: 1
        case .bfloat16, .float16: 2
        case .uint32, .float32: 4
        }
    }
}

// MARK: - Errors

public enum ExpertOffsetIndexError: Error, CustomStringConvertible {
    case noShardsFound(URL)
    case malformedShard(String)
    case fusedGateUpLayout(String)
    case perExpertLayout(String)
    case noRoutedExperts(URL)
    case inconsistentExpertCount(layer: Int, piece: String, expected: Int, found: Int)
    case incompletePiece(layer: Int, missing: String)
    case unalignedRow(layer: Int, piece: String, rowBytes: Int)
    case byteCountMismatch(layer: Int, piece: String, declared: Int, computed: Int)
    case unsupportedDType(String)
    case fingerprintMismatch(expected: String, found: String)
    case unsupportedFormatVersion(Int)
    case ambiguousKey(layer: Int, piece: String, first: String, second: String)
    case quantizationMismatch(
        projection: String, bits: Int, groupSize: Int, impliedInputDims: Int, scaleGroups: Int)

    public var description: String {
        switch self {
        case .noShardsFound(let url):
            "no safetensors shards under \(url.path)"
        case .malformedShard(let name):
            "malformed safetensors header in \(name)"
        case .fusedGateUpLayout(let key):
            """
            expert streaming does not support fused gate_up_proj checkpoints \
            (TD-050); offending tensor: \(key)
            """
        case .perExpertLayout(let key):
            """
            expert streaming requires stacked switch_mlp tensors, not per-expert \
            tensors (TD-050); offending tensor: \(key)
            """
        case .noRoutedExperts(let url):
            "no switch_mlp routed-expert tensors under \(url.path)"
        case .inconsistentExpertCount(let layer, let piece, let expected, let found):
            "layer \(layer) piece \(piece): expected \(expected) experts, found \(found)"
        case .incompletePiece(let layer, let missing):
            "layer \(layer) is missing \(missing)"
        case .unalignedRow(let layer, let piece, let rowBytes):
            """
            layer \(layer) piece \(piece): row of \(rowBytes) bytes is not a multiple \
            of the 16 KiB page, so the zero-copy delivery path cannot be used
            """
        case .byteCountMismatch(let layer, let piece, let declared, let computed):
            "layer \(layer) piece \(piece): header declares \(declared) bytes, shape implies \(computed)"
        case .unsupportedDType(let dtype):
            "unsupported safetensors dtype \(dtype)"
        case .fingerprintMismatch(let expected, let found):
            "expert offset index was built for checkpoint \(expected), found \(found)"
        case .unsupportedFormatVersion(let version):
            "expert offset index format version \(version) is not readable"
        case .ambiguousKey(let layer, let piece, let first, let second):
            """
            two tensors claim layer \(layer) piece \(piece): \(first) and \(second); \
            the index cannot tell which stack the decoder uses
            """
        case .quantizationMismatch(
            let projection, let bits, let groupSize, let impliedInputDims, let scaleGroups):
            """
            \(projection) does not match \(bits)-bit / group \(groupSize): the packed \
            weight implies \(impliedInputDims) input dims, which needs \
            \(impliedInputDims / groupSize) scale groups, but the checkpoint has \
            \(scaleGroups)
            """
        }
    }
}

// MARK: - Records

/// Where one tensor's expert rows live.
public struct ExpertTensorRecord: Codable, Sendable, Equatable {
    /// Index into ``ExpertOffsetIndex/shardFiles``.
    public let shard: Int
    /// Absolute byte offset of expert 0 inside the shard file. Already includes
    /// the 8-byte header length field and the JSON header.
    public let baseOffset: Int64
    /// Bytes occupied by a single expert.
    public let rowBytes: Int
    /// Shape of a single expert's row (the tensor's shape without axis 0).
    public let rowShape: [Int]
    public let dtype: ExpertDType

    public func offset(ofExpert expert: Int) -> Int64 {
        baseOffset + Int64(expert) * Int64(rowBytes)
    }

    /// Offset of a run of consecutive experts, for coalesced prefill reads.
    public func offset(ofRunStartingAt expert: Int) -> Int64 { offset(ofExpert: expert) }
}

/// One layer's nine records, in ``ExpertPiece/slot`` order.
public struct ExpertLayerRecords: Codable, Sendable, Equatable {
    public let layer: Int
    public let pieces: [ExpertTensorRecord]

    public subscript(piece: ExpertPiece) -> ExpertTensorRecord { pieces[piece.slot] }
}

// MARK: - Index

public struct ExpertOffsetIndex: Codable, Sendable, Equatable {
    public static let currentFormatVersion = 1

    /// The 16 KiB page of Apple silicon. A row that is not a multiple of this
    /// cannot be handed to MLX without a copy: `MetalAllocator::make_buffer`
    /// returns null for a misaligned pointer or length and `array.cpp` then
    /// falls back to `malloc` + `std::copy` *silently*.
    public static let pageSize = 16384

    public let formatVersion: Int
    /// Fingerprint of the checkpoint's weight map, for invalidation.
    public let fingerprint: String
    /// Shard file names, relative to the model directory.
    public let shardFiles: [String]
    public let expertCount: Int
    public let layers: [ExpertLayerRecords]

    public init(
        formatVersion: Int = ExpertOffsetIndex.currentFormatVersion,
        fingerprint: String,
        shardFiles: [String],
        expertCount: Int,
        layers: [ExpertLayerRecords]
    ) {
        self.formatVersion = formatVersion
        self.fingerprint = fingerprint
        self.shardFiles = shardFiles
        self.expertCount = expertCount
        self.layers = layers
    }

    public var layerCount: Int { layers.count }

    /// Bytes of one expert across its nine pieces.
    public var bytesPerExpert: Int {
        guard let first = layers.first else { return 0 }
        return first.pieces.reduce(0) { $0 + $1.rowBytes }
    }

    public var routedBytes: Int {
        layers.reduce(0) { total, layer in
            total + layer.pieces.reduce(0) { $0 + $1.rowBytes * expertCount }
        }
    }

    public func records(forLayer layer: Int) -> ExpertLayerRecords? {
        layers.first { $0.layer == layer }
    }

    public func shardURL(_ shard: Int, relativeTo modelDirectory: URL) -> URL {
        modelDirectory.appending(path: shardFiles[shard])
    }

    // MARK: Key classification

    /// Whether a checkpoint key names a stacked routed-expert tensor, i.e. one
    /// this index owns and the module tree must not load.
    public static func isRoutedExpertKey(_ key: String) -> Bool {
        key.contains(".switch_mlp.") || key.hasPrefix("switch_mlp.")
    }

    /// `(layer, piece)` for a stacked routed-expert key, or `nil`.
    ///
    /// Prefix-agnostic on purpose: the same checkpoint is read as
    /// `model.layers.N…` by the text model and `model.language_model.layers.N…`
    /// by the multimodal wrapper.
    static func parse(key: String) throws -> (layer: Int, piece: ExpertPiece)? {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false)

        // The multi-token-prediction head is a decoder layer of its own, with
        // its own `layers.N.mlp.switch_mlp.*` tensors, and `Qwen35TextModel`
        // drops every `mtp.` key in `sanitize`. Left in, `mtp.layers.0` would
        // collide with the real layer 0 and the streamed model would multiply
        // one layer's tokens by the wrong experts — which is exactly what it
        // did, silently, until a resident/streamed logit comparison caught it.
        if parts.contains("mtp") { return nil }

        if let expertsIndex = parts.firstIndex(of: "experts"),
            expertsIndex + 1 < parts.count, Int(parts[expertsIndex + 1]) != nil
        {
            throw ExpertOffsetIndexError.perExpertLayout(key)
        }
        guard let switchIndex = parts.firstIndex(of: "switch_mlp") else { return nil }
        if parts.contains("gate_up_proj") {
            throw ExpertOffsetIndexError.fusedGateUpLayout(key)
        }
        guard switchIndex + 2 < parts.count,
            let projection = ExpertProjection(rawValue: String(parts[switchIndex + 1])),
            let component = ExpertComponent(rawValue: String(parts[switchIndex + 2]))
        else {
            return nil
        }

        guard let layersIndex = parts.firstIndex(of: "layers"),
            layersIndex + 1 < parts.count,
            let layer = Int(parts[layersIndex + 1])
        else {
            return nil
        }

        return (layer, ExpertPiece(projection, component))
    }

    /// Check the checkpoint's own geometry against the quantization the caller
    /// intends to run with.
    ///
    /// A `bits` or `groupSize` that does not match the checkpoint produces
    /// fluent nonsense rather than an error: the gather matmul happily decodes
    /// 4-bit payloads as 8-bit. The shapes already contain the answer —
    /// a packed weight row of `[out, in * bits / 32]` and a scales row of
    /// `[out, in / groupSize]` have to agree on `in` — so this is checkable
    /// and therefore must be checked (TD-050's lesson, applied to the numbers
    /// instead of the layout).
    public func validateQuantization(groupSize: Int, bits: Int) throws {
        guard let layer = layers.first, bits > 0, groupSize > 0 else { return }
        for projection in ExpertProjection.allCases {
            let weight = layer[ExpertPiece(projection, .weight)]
            let scales = layer[ExpertPiece(projection, .scales)]
            guard weight.rowShape.count == 2, scales.rowShape.count == 2 else { continue }

            let bitsPerContainer = weight.dtype.itemSize * 8
            let impliedInputDims = weight.rowShape[1] * bitsPerContainer / bits
            let scaleGroups = scales.rowShape[1]
            guard impliedInputDims % groupSize == 0,
                impliedInputDims / groupSize == scaleGroups
            else {
                throw ExpertOffsetIndexError.quantizationMismatch(
                    projection: projection.rawValue, bits: bits, groupSize: groupSize,
                    impliedInputDims: impliedInputDims, scaleGroups: scaleGroups)
            }
        }
    }

    // MARK: Persistence

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }

    public static func decoded(from data: Data) throws -> ExpertOffsetIndex {
        let index = try JSONDecoder().decode(ExpertOffsetIndex.self, from: data)
        guard index.formatVersion == currentFormatVersion else {
            throw ExpertOffsetIndexError.unsupportedFormatVersion(index.formatVersion)
        }
        return index
    }

    public func write(to url: URL) throws {
        try encoded().write(to: url, options: .atomic)
    }

    /// Load a persisted index and refuse it if the checkpoint changed under it.
    public static func load(from url: URL, expectingFingerprint fingerprint: String) throws
        -> ExpertOffsetIndex
    {
        let index = try decoded(from: Data(contentsOf: url))
        guard index.fingerprint == fingerprint else {
            throw ExpertOffsetIndexError.fingerprintMismatch(
                expected: index.fingerprint, found: fingerprint)
        }
        return index
    }
}

// MARK: - Construction

extension ExpertOffsetIndex {

    /// Build the index for a model directory by reading only the shard headers.
    ///
    /// Reads at most a few hundred KB: the 8-byte length field plus the JSON
    /// header of each shard. No tensor data is touched.
    public static func build(modelDirectory: URL) throws -> ExpertOffsetIndex {
        let shardURLs = try routedWeightShards(in: modelDirectory)
        guard !shardURLs.isEmpty else {
            throw ExpertOffsetIndexError.noShardsFound(modelDirectory)
        }

        var perLayer = [Int: [Int: ExpertTensorRecord]]()
        var expertCount: Int?

        for (shardIndex, url) in shardURLs.enumerated() {
            let header = try SafetensorsHeader.read(url: url)
            for entry in header.entries {
                guard let parsed = try parse(key: entry.name) else { continue }
                guard let dtype = ExpertDType(rawValue: entry.dtype) else {
                    throw ExpertOffsetIndexError.unsupportedDType(entry.dtype)
                }
                guard entry.shape.count >= 2 else {
                    throw ExpertOffsetIndexError.malformedShard(url.lastPathComponent)
                }

                let experts = entry.shape[0]
                if let expertCount, expertCount != experts {
                    throw ExpertOffsetIndexError.inconsistentExpertCount(
                        layer: parsed.layer, piece: entry.name,
                        expected: expertCount, found: experts)
                }
                expertCount = experts

                let rowShape = Array(entry.shape.dropFirst())
                let computed = rowShape.reduce(1, *) * dtype.itemSize * experts
                guard computed == entry.byteCount else {
                    throw ExpertOffsetIndexError.byteCountMismatch(
                        layer: parsed.layer, piece: entry.name,
                        declared: entry.byteCount, computed: computed)
                }
                let rowBytes = entry.byteCount / experts
                guard rowBytes % pageSize == 0 else {
                    throw ExpertOffsetIndexError.unalignedRow(
                        layer: parsed.layer, piece: entry.name, rowBytes: rowBytes)
                }

                // Loud on collision (TD-050): a second tensor claiming the same
                // (layer, piece) means the key pattern does not identify the
                // decoder stack uniquely in this checkpoint.
                if perLayer[parsed.layer]?[parsed.piece.slot] != nil {
                    throw ExpertOffsetIndexError.ambiguousKey(
                        layer: parsed.layer,
                        piece: "\(parsed.piece.projection.rawValue).\(parsed.piece.component.rawValue)",
                        first: "already indexed", second: entry.name)
                }
                perLayer[parsed.layer, default: [:]][parsed.piece.slot] = ExpertTensorRecord(
                    shard: shardIndex,
                    baseOffset: header.dataStart + entry.begin,
                    rowBytes: rowBytes,
                    rowShape: rowShape,
                    dtype: dtype)
            }
        }

        guard let expertCount, !perLayer.isEmpty else {
            throw ExpertOffsetIndexError.noRoutedExperts(modelDirectory)
        }

        var layers = [ExpertLayerRecords]()
        for layer in perLayer.keys.sorted() {
            let slots = perLayer[layer]!
            var pieces = [ExpertTensorRecord]()
            for piece in ExpertPiece.all {
                guard let record = slots[piece.slot] else {
                    throw ExpertOffsetIndexError.incompletePiece(
                        layer: layer,
                        missing: "switch_mlp.\(piece.projection.rawValue).\(piece.component.rawValue)"
                    )
                }
                pieces.append(record)
            }
            layers.append(ExpertLayerRecords(layer: layer, pieces: pieces))
        }

        // Uniform geometry across layers is what lets one global slot bank
        // serve every layer; a checkpoint that breaks it would silently
        // misread rows of the wrong size.
        if let reference = layers.first {
            for layer in layers.dropFirst() {
                for piece in ExpertPiece.all {
                    let a = reference[piece]
                    let b = layer[piece]
                    guard a.rowBytes == b.rowBytes, a.rowShape == b.rowShape, a.dtype == b.dtype
                    else {
                        throw ExpertOffsetIndexError.byteCountMismatch(
                            layer: layer.layer,
                            piece: "\(piece.projection.rawValue).\(piece.component.rawValue)",
                            declared: b.rowBytes, computed: a.rowBytes)
                    }
                }
            }
        }

        return ExpertOffsetIndex(
            fingerprint: try fingerprint(modelDirectory: modelDirectory, shardURLs: shardURLs),
            shardFiles: shardURLs.map(\.lastPathComponent),
            expertCount: expertCount,
            layers: layers)
    }

    /// The checkpoint's weight files, preferring the names in
    /// `model.safetensors.index.json` so auxiliary files (`mtp.safetensors`,
    /// adapters) never enter the index.
    static func routedWeightShards(in modelDirectory: URL) throws -> [URL] {
        let indexURL = modelDirectory.appending(path: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: String]
        {
            let names = Set(weightMap.values).sorted()
            let urls = names.map { modelDirectory.appending(path: $0) }
            if urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) {
                return urls
            }
        }

        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        return
            contents
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("model") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// A cheap, deterministic fingerprint of the checkpoint's tensor layout.
    ///
    /// Built from `model.safetensors.index.json` when present — the file the
    /// design names — and otherwise from each shard's name, size and header
    /// length. Not a cryptographic digest and not used as one: its only job is
    /// to invalidate an index whose checkpoint was replaced.
    static func fingerprint(modelDirectory: URL, shardURLs: [URL]) throws -> String {
        var hash = FNV1a()
        let indexURL = modelDirectory.appending(path: "model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL) {
            hash.combine(data)
        } else {
            for url in shardURLs {
                hash.combine(Data(url.lastPathComponent.utf8))
                let size =
                    (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int)
                    ?? nil
                hash.combine(Int64(size ?? 0))
                hash.combine(try SafetensorsHeader.read(url: url).dataStart)
            }
        }
        return String(hash.value, radix: 16)
    }
}

/// 64-bit FNV-1a. Deterministic across runs and platforms; deliberately not a
/// security primitive (see ``ExpertOffsetIndex/fingerprint(modelDirectory:shardURLs:)``).
struct FNV1a {
    private(set) var value: UInt64 = 0xcbf2_9ce4_8422_2325

    mutating func combine(_ data: Data) {
        for byte in data {
            value ^= UInt64(byte)
            value = value &* 0x1000_0000_01b3
        }
    }

    mutating func combine(_ number: Int64) {
        withUnsafeBytes(of: number.littleEndian) { combine(Data($0)) }
    }
}

// MARK: - Safetensors header

/// The parsed header of a safetensors file: enough to locate any tensor's
/// bytes without reading them.
struct SafetensorsHeader {
    struct Entry {
        let name: String
        let dtype: String
        let shape: [Int]
        /// Offset relative to ``dataStart``.
        let begin: Int64
        let byteCount: Int
    }

    /// Absolute offset where the tensor data region begins: 8 + header length.
    let dataStart: Int64
    let entries: [Entry]

    static func read(url: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw ExpertOffsetIndexError.malformedShard(url.lastPathComponent)
        }
        let headerLength = lengthData.withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self)
        }.littleEndian
        guard headerLength > 0, headerLength <= 512 * 1024 * 1024,
            let headerData = try handle.read(upToCount: Int(headerLength)),
            headerData.count == headerLength,
            let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw ExpertOffsetIndexError.malformedShard(url.lastPathComponent)
        }

        var entries = [Entry]()
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                let begin = (offsets[0] as? NSNumber)?.int64Value,
                let end = (offsets[1] as? NSNumber)?.int64Value,
                end >= begin
            else {
                throw ExpertOffsetIndexError.malformedShard(url.lastPathComponent)
            }
            entries.append(
                Entry(
                    name: name, dtype: dtype, shape: shape, begin: begin,
                    byteCount: Int(end - begin)))
        }
        entries.sort { $0.begin < $1.begin }

        return SafetensorsHeader(dataStart: 8 + Int64(headerLength), entries: entries)
    }
}
