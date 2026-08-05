import Foundation

// Plain DTOs describing how a model's layers are distributed across a pipeline-parallel
// group. No MLX dependency on purpose: these values need to be computable (and sendable
// over the network) before any model is loaded.

/// Storage footprint, in bytes.
public struct MemorySize: Codable, Hashable, Sendable {
    public var inBytes: Int

    public init(inBytes: Int = 0) {
        self.inBytes = inBytes
    }
}

/// Static facts about a model needed to plan a shard layout, independent of any
/// particular node's hardware.
public struct ModelMetadata: Codable, Hashable, Sendable {
    public let modelId: String
    public let modelType: String
    public let prettyName: String
    public let storageSize: MemorySize
    public let nLayers: Int
    public let hiddenSize: Int
    public let supportsTensor: Bool

    public init(
        modelId: String,
        modelType: String,
        prettyName: String,
        storageSize: MemorySize,
        nLayers: Int,
        hiddenSize: Int,
        supportsTensor: Bool = false
    ) {
        self.modelId = modelId
        self.modelType = modelType
        self.prettyName = prettyName
        self.storageSize = storageSize
        self.nLayers = nLayers
        self.hiddenSize = hiddenSize
        self.supportsTensor = supportsTensor
    }
}

/// Which contiguous range of layers a given rank should hold, within a pipeline-parallel
/// group of `worldSize` nodes.
///
/// `useTensorParallel` travels here (per shard, per message) rather than as global state,
/// since it needs to be Sendable across the mesh protocol, not read from `UserDefaults`.
public struct ShardMetadata: Codable, Hashable, Sendable {
    public let modelMeta: ModelMetadata
    public let deviceRank: Int
    public let worldSize: Int
    public let useTensorParallel: Bool

    public let startLayer: Int
    public let endLayer: Int
    public let nLayers: Int

    public var isFirstLayer: Bool { startLayer == 0 }
    public var isLastLayer: Bool { endLayer == nLayers }

    public init(
        modelMeta: ModelMetadata,
        deviceRank: Int,
        worldSize: Int,
        useTensorParallel: Bool = false,
        startLayer: Int,
        endLayer: Int,
        nLayers: Int
    ) {
        self.modelMeta = modelMeta
        self.deviceRank = deviceRank
        self.worldSize = worldSize
        self.useTensorParallel = useTensorParallel
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.nLayers = nLayers
    }
}
