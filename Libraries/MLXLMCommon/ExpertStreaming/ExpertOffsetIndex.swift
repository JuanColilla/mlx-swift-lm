// FORK(JuanColilla): R-56 expert streaming — offset index over the original
// safetensors shards (design "Option A": no repack).
//
// Every routed-expert tensor of a stacked MoE checkpoint carries the expert
// on axis 0 and is stored row-major, so expert `i` of a given tensor is the
// contiguous byte range `[i*rowBytes, (i+1)*rowBytes)` inside its shard. This
// index is the map from `(family, layer, projection, component)` to that
// range, built once from the shard headers and persisted next to the model.
//
// Two families of routed experts exist. Every MoE checkpoint has the MLP
// family (`switch_mlp.{gate,up,down}_proj`); K2-Horizon MoVA adds a value
// family (`self_attn.switch_v`), a single projection routed per token by its
// own router. They have their own expert counts and geometry, so the index
// keeps them apart end to end: a slot bank is built per family.
//
// TD-050: the index is only valid for the stacked layout. A checkpoint that
// stores experts as separate `experts.N.*` tensors, or that fuses gate and up
// into a single `gate_up_proj`, is rejected loudly here rather than degrading
// into a silently wrong read.

import Foundation

// MARK: - Geometry

/// A group of routed experts that share a router, a count and a geometry.
public enum ExpertFamily: String, Codable, Sendable, CaseIterable, Comparable {
    /// The feed-forward experts of a MoE layer: `switch_mlp`.
    case mlp
    /// The value-projection experts of a MoVA attention layer: `switch_v`.
    case value

    /// The projections a routed expert of this family is made of.
    public var projections: [ExpertProjection] {
        switch self {
        case .mlp: [.gate, .up, .down]
        case .value: [.value]
        }
    }

    /// The pieces of one expert, in ``ExpertPiece/slot`` order.
    public var pieces: [ExpertPiece] {
        projections.flatMap { projection in
            ExpertComponent.allCases.map { ExpertPiece(projection, $0) }
        }
    }

    /// The tensor-name segment that marks a stacked expert tensor of this
    /// family: `switch_mlp` or `switch_v`.
    public var stackName: String {
        switch self {
        case .mlp: "switch_mlp"
        case .value: "switch_v"
        }
    }

    /// The order in which a decoder layer visits its families: attention runs
    /// before the feed-forward block, so the value experts of layer L come
    /// before the MLP experts of layer L. The prefetcher's notion of "the next
    /// step" depends on it.
    public static func < (lhs: ExpertFamily, rhs: ExpertFamily) -> Bool {
        lhs.forwardOrder < rhs.forwardOrder
    }

    private var forwardOrder: Int {
        switch self {
        case .value: 0
        case .mlp: 1
        }
    }
}

/// One projection of a routed expert.
public enum ExpertProjection: String, Codable, Sendable, CaseIterable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"
    /// The single projection of a MoVA value expert. `switch_v` has no
    /// projection segment in its key — the component follows it directly.
    case value = "switch_v"

    public var family: ExpertFamily {
        switch self {
        case .gate, .up, .down: .mlp
        case .value: .value
        }
    }
}

/// One of the three arrays an affine-quantized projection is stored as.
public enum ExpertComponent: String, Codable, Sendable, CaseIterable {
    case weight
    case scales
    case biases
}

/// One contiguous byte range of a single expert.
///
/// The slot is family-local: the MLP family has nine pieces (0…8) and the
/// value family three (0…2). Every array indexed by piece — staging buffers,
/// slot-bank pools, read batches — belongs to exactly one family, so the
/// numbering never has to be global.
public struct ExpertPiece: Hashable, Sendable {
    public let projection: ExpertProjection
    public let component: ExpertComponent

    public init(_ projection: ExpertProjection, _ component: ExpertComponent) {
        self.projection = projection
        self.component = component
    }

    public var family: ExpertFamily { projection.family }

    /// The nine pieces of the MLP family, in canonical order.
    public static let all: [ExpertPiece] = ExpertFamily.mlp.pieces

    public var slot: Int {
        let projectionIndex = family.projections.firstIndex(of: projection)!
        let componentIndex = ExpertComponent.allCases.firstIndex(of: component)!
        return projectionIndex * ExpertComponent.allCases.count + componentIndex
    }

    public var name: String { "\(projection.rawValue).\(component.rawValue)" }
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
    case missingFamily(ExpertFamily)

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
            layer \(layer) piece \(piece): row of \(rowBytes) bytes is not a whole \
            number of its dtype's items, so no expert row can be delivered as an array
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
        case .missingFamily(let family):
            "the index has no \(family.rawValue) expert family"
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
    ///
    /// Not necessarily a multiple of the 16 KiB page: the K2-Horizon MLP
    /// scales are 61.440 bytes per expert. The staging path pads the whole
    /// batch to a page instead — see `ExpertReadBatch.materialize`.
    public let rowBytes: Int
    /// Shape of a single expert's row (the tensor's shape without axis 0).
    public let rowShape: [Int]
    public let dtype: ExpertDType

    public func offset(ofExpert expert: Int) -> Int64 {
        baseOffset + Int64(expert) * Int64(rowBytes)
    }

    /// Offset of a run of consecutive experts, for coalesced prefill reads.
    public func offset(ofRunStartingAt expert: Int) -> Int64 { offset(ofExpert: expert) }

    /// Whether a row of this tensor can be handed to MLX without padding.
    public var isPageAligned: Bool { rowBytes % ExpertOffsetIndex.pageSize == 0 }
}

/// One layer's records for one family, in ``ExpertPiece/slot`` order.
public struct ExpertLayerRecords: Codable, Sendable, Equatable {
    public let layer: Int
    public let pieces: [ExpertTensorRecord]

    public init(layer: Int, pieces: [ExpertTensorRecord]) {
        self.layer = layer
        self.pieces = pieces
    }

    public subscript(piece: ExpertPiece) -> ExpertTensorRecord { pieces[piece.slot] }
}

/// Every layer of one expert family, plus the geometry they share.
public struct ExpertFamilyIndex: Codable, Sendable, Equatable {
    public let family: ExpertFamily
    public let expertCount: Int
    public let layers: [ExpertLayerRecords]

    public init(family: ExpertFamily, expertCount: Int, layers: [ExpertLayerRecords]) {
        self.family = family
        self.expertCount = expertCount
        self.layers = layers
    }

    public var layerCount: Int { layers.count }

    /// Bytes of one expert across its pieces.
    public var bytesPerExpert: Int {
        guard let first = layers.first else { return 0 }
        return first.pieces.reduce(0) { $0 + $1.rowBytes }
    }

    public var routedBytes: Int { bytesPerExpert * expertCount * layers.count }

    public func records(forLayer layer: Int) -> ExpertLayerRecords? {
        layers.first { $0.layer == layer }
    }

    /// The geometry every layer of the family shares; the slot bank is
    /// allocated from it.
    public var template: ExpertLayerRecords { layers[0] }
}

// MARK: - Index

public struct ExpertOffsetIndex: Codable, Sendable, Equatable {
    /// Version 2 introduced the expert family. A version-1 index has no
    /// family records and is rebuilt rather than migrated: building one costs
    /// a few hundred KB of header reads.
    public static let currentFormatVersion = 2

    /// The 16 KiB page of Apple silicon. Staging handed to MLX has to be a
    /// whole number of these, pointer and length: `MetalAllocator::make_buffer`
    /// returns null for a misaligned pointer or length and `array.cpp` then
    /// falls back to `malloc` + `std::copy` *silently*.
    public static let pageSize = 16384

    public let formatVersion: Int
    /// Fingerprint of the checkpoint's weight map, for invalidation.
    public let fingerprint: String
    /// Shard file names, relative to the model directory.
    public let shardFiles: [String]
    /// The families present, in `ExpertFamily.allCases` order. The MLP family
    /// is always there; the value family only for MoVA checkpoints.
    public let families: [ExpertFamilyIndex]

    public init(
        formatVersion: Int = ExpertOffsetIndex.currentFormatVersion,
        fingerprint: String,
        shardFiles: [String],
        families: [ExpertFamilyIndex]
    ) {
        self.formatVersion = formatVersion
        self.fingerprint = fingerprint
        self.shardFiles = shardFiles
        self.families = families
    }

    public func family(_ family: ExpertFamily) -> ExpertFamilyIndex? {
        families.first { $0.family == family }
    }

    public func requireFamily(_ family: ExpertFamily) throws -> ExpertFamilyIndex {
        guard let found = self.family(family) else {
            throw ExpertOffsetIndexError.missingFamily(family)
        }
        return found
    }

    /// The MLP family. Every streamed checkpoint has one; `build` refuses a
    /// directory without it.
    public var mlp: ExpertFamilyIndex { family(.mlp)! }

    // MARK: MLP-family conveniences
    //
    // The single-family API, kept because the host app reads it for its
    // memory profile and its logging. They describe the MLP family only;
    // `routedBytes` is the exception and sums every family, because it
    // answers "how much of the checkpoint is not resident".

    public var expertCount: Int { mlp.expertCount }
    public var layers: [ExpertLayerRecords] { mlp.layers }
    public var layerCount: Int { mlp.layerCount }
    public var bytesPerExpert: Int { mlp.bytesPerExpert }

    public var routedBytes: Int { families.reduce(0) { $0 + $1.routedBytes } }

    public func records(forLayer layer: Int) -> ExpertLayerRecords? {
        mlp.records(forLayer: layer)
    }

    public func shardURL(_ shard: Int, relativeTo modelDirectory: URL) -> URL {
        modelDirectory.appending(path: shardFiles[shard])
    }

    // MARK: Key classification

    /// Whether a checkpoint key names a stacked routed-expert tensor of any
    /// family, i.e. one this index owns and the module tree must not load.
    ///
    /// Both families have to be here: a `switch_v` left unrecognized is not an
    /// error, it is 3,8 GB of value experts loaded resident with nothing
    /// saying so.
    public static func isRoutedExpertKey(_ key: String) -> Bool {
        ExpertFamily.allCases.contains { family in
            key.contains(".\(family.stackName).") || key.hasPrefix("\(family.stackName).")
        }
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

        let piece: ExpertPiece
        if let switchIndex = parts.firstIndex(of: "switch_mlp") {
            if parts.contains("gate_up_proj") {
                throw ExpertOffsetIndexError.fusedGateUpLayout(key)
            }
            guard switchIndex + 2 < parts.count,
                let projection = ExpertProjection(rawValue: String(parts[switchIndex + 1])),
                projection.family == .mlp,
                let component = ExpertComponent(rawValue: String(parts[switchIndex + 2]))
            else {
                return nil
            }
            piece = ExpertPiece(projection, component)
        } else if let switchIndex = parts.firstIndex(of: "switch_v") {
            guard switchIndex + 1 < parts.count,
                let component = ExpertComponent(rawValue: String(parts[switchIndex + 1]))
            else {
                return nil
            }
            piece = ExpertPiece(.value, component)
        } else {
            return nil
        }

        guard let layersIndex = parts.firstIndex(of: "layers"),
            layersIndex + 1 < parts.count,
            let layer = Int(parts[layersIndex + 1])
        else {
            return nil
        }

        return (layer, piece)
    }

    /// Check the checkpoint's own geometry against the quantization the caller
    /// intends to run with, for every family.
    ///
    /// A `bits` or `groupSize` that does not match the checkpoint produces
    /// fluent nonsense rather than an error: the gather matmul happily decodes
    /// 4-bit payloads as 8-bit. The shapes already contain the answer —
    /// a packed weight row of `[out, in * bits / 32]` and a scales row of
    /// `[out, in / groupSize]` have to agree on `in` — so this is checkable
    /// and therefore must be checked (TD-050's lesson, applied to the numbers
    /// instead of the layout).
    public func validateQuantization(groupSize: Int, bits: Int) throws {
        guard bits > 0, groupSize > 0 else { return }
        for family in families {
            guard let layer = family.layers.first else { continue }
            for projection in family.family.projections {
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
    }

    // MARK: Persistence

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }

    public static func decoded(from data: Data) throws -> ExpertOffsetIndex {
        // The version is read before the rest so a version-1 file reports
        // itself as outdated instead of as a decoding failure.
        struct Version: Decodable { let formatVersion: Int }
        let version = try JSONDecoder().decode(Version.self, from: data).formatVersion
        guard version == currentFormatVersion else {
            throw ExpertOffsetIndexError.unsupportedFormatVersion(version)
        }
        return try JSONDecoder().decode(ExpertOffsetIndex.self, from: data)
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

        var perFamily = [ExpertFamily: [Int: [Int: ExpertTensorRecord]]]()
        var expertCounts = [ExpertFamily: Int]()

        for (shardIndex, url) in shardURLs.enumerated() {
            let header = try SafetensorsHeader.read(url: url)
            for entry in header.entries {
                guard let parsed = try parse(key: entry.name) else { continue }
                let family = parsed.piece.family
                guard let dtype = ExpertDType(rawValue: entry.dtype) else {
                    throw ExpertOffsetIndexError.unsupportedDType(entry.dtype)
                }
                guard entry.shape.count >= 2 else {
                    throw ExpertOffsetIndexError.malformedShard(url.lastPathComponent)
                }

                let experts = entry.shape[0]
                if let expected = expertCounts[family], expected != experts {
                    throw ExpertOffsetIndexError.inconsistentExpertCount(
                        layer: parsed.layer, piece: entry.name,
                        expected: expected, found: experts)
                }
                expertCounts[family] = experts

                let rowShape = Array(entry.shape.dropFirst())
                let computed = rowShape.reduce(1, *) * dtype.itemSize * experts
                guard computed == entry.byteCount else {
                    throw ExpertOffsetIndexError.byteCountMismatch(
                        layer: parsed.layer, piece: entry.name,
                        declared: entry.byteCount, computed: computed)
                }
                let rowBytes = entry.byteCount / experts
                // A row is allowed not to be a page multiple — the batch is
                // padded — but it has to be a whole number of items, or no
                // shape can describe it.
                guard rowBytes % dtype.itemSize == 0 else {
                    throw ExpertOffsetIndexError.unalignedRow(
                        layer: parsed.layer, piece: entry.name, rowBytes: rowBytes)
                }

                // Loud on collision (TD-050): a second tensor claiming the same
                // (layer, piece) means the key pattern does not identify the
                // decoder stack uniquely in this checkpoint.
                if perFamily[family]?[parsed.layer]?[parsed.piece.slot] != nil {
                    throw ExpertOffsetIndexError.ambiguousKey(
                        layer: parsed.layer, piece: parsed.piece.name,
                        first: "already indexed", second: entry.name)
                }
                perFamily[family, default: [:]][parsed.layer, default: [:]][parsed.piece.slot] =
                    ExpertTensorRecord(
                        shard: shardIndex,
                        baseOffset: header.dataStart + entry.begin,
                        rowBytes: rowBytes,
                        rowShape: rowShape,
                        dtype: dtype)
            }
        }

        guard perFamily[.mlp] != nil else {
            throw ExpertOffsetIndexError.noRoutedExperts(modelDirectory)
        }

        var families = [ExpertFamilyIndex]()
        for family in ExpertFamily.allCases {
            guard let perLayer = perFamily[family], let expertCount = expertCounts[family]
            else { continue }

            var layers = [ExpertLayerRecords]()
            for layer in perLayer.keys.sorted() {
                let slots = perLayer[layer]!
                var pieces = [ExpertTensorRecord]()
                for piece in family.pieces {
                    guard let record = slots[piece.slot] else {
                        throw ExpertOffsetIndexError.incompletePiece(
                            layer: layer, missing: "\(family.stackName).\(piece.name)")
                    }
                    pieces.append(record)
                }
                layers.append(ExpertLayerRecords(layer: layer, pieces: pieces))
            }

            // Uniform geometry across layers is what lets one slot bank serve
            // every layer of a family; a checkpoint that breaks it would
            // silently misread rows of the wrong size.
            if let reference = layers.first {
                for layer in layers.dropFirst() {
                    for piece in family.pieces {
                        let a = reference[piece]
                        let b = layer[piece]
                        guard a.rowBytes == b.rowBytes, a.rowShape == b.rowShape,
                            a.dtype == b.dtype
                        else {
                            throw ExpertOffsetIndexError.byteCountMismatch(
                                layer: layer.layer, piece: piece.name,
                                declared: b.rowBytes, computed: a.rowBytes)
                        }
                    }
                }
            }

            families.append(
                ExpertFamilyIndex(family: family, expertCount: expertCount, layers: layers))
        }

        return ExpertOffsetIndex(
            fingerprint: try fingerprint(modelDirectory: modelDirectory, shardURLs: shardURLs),
            shardFiles: shardURLs.map(\.lastPathComponent),
            families: families)
    }

    /// The checkpoint's weight files, preferring the names in
    /// `model.safetensors.index.json` so auxiliary files (`mtp.safetensors`,
    /// adapters, the `k2-reference-*.safetensors` diagnostic dumps) never
    /// enter the index. Without an index file, only `model*.safetensors`
    /// qualify, which excludes the same auxiliaries by name.
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
