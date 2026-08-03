// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

/// Selects and optionally renames weights while a model is loaded.
///
/// Pipeline-parallel models must be structurally sharded before their
/// weights are attached. The selection keeps the source model's global layer
/// names on disk while mapping the retained range to the compact local layer
/// indices in the sharded module tree.
public struct WeightLoadingSelection: Sendable {
    private let includes: @Sendable (String) -> Bool
    private let rewrites: @Sendable (String) -> String

    public init(
        includes: @Sendable @escaping (String) -> Bool,
        rewrites: @Sendable @escaping (String) -> String = { $0 }
    ) {
        self.includes = includes
        self.rewrites = rewrites
    }

    public func includes(_ key: String) -> Bool {
        includes(key)
    }

    public func rewrite(_ key: String) -> String {
        rewrites(key)
    }

    /// Keeps a contiguous, global transformer-layer range and remaps it to
    /// the zero-based indices of a pipeline-sharded module tree.
    public static func pipelineLayers(
        range: Range<Int>,
        sourcePrefixes: [String],
        destinationPrefix: String
    ) -> Self {
        Self(
            includes: { key in
                guard let layer = Self.layerIndex(in: key, prefixes: sourcePrefixes) else { return true }
                return range.contains(layer)
            },
            rewrites: { key in
                guard let match = Self.layerMatch(in: key, prefixes: [destinationPrefix]) else { return key }
                return "\(destinationPrefix)\(match.index - range.lowerBound)\(match.suffix)"
            }
        )
    }

    private static func layerIndex(in key: String, prefixes: [String]) -> Int? {
        layerMatch(in: key, prefixes: prefixes)?.index
    }

    private static func layerMatch(in key: String, prefixes: [String]) -> (index: Int, suffix: String)? {
        for prefix in prefixes where key.hasPrefix(prefix) {
            let remainder = key.dropFirst(prefix.count)
            let digits = remainder.prefix { $0.isNumber }
            guard !digits.isEmpty, let index = Int(digits) else { continue }
            return (index, String(remainder.dropFirst(digits.count)))
        }
        return nil
    }
}

package func safetensorWeightURLs(
    in modelDirectory: URL,
    selection: WeightLoadingSelection? = nil
) throws -> [URL] {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        return Set(index.weightMap.compactMap { key, filename in
            (selection?.includes(key) ?? true) ? filename : nil
        })
            .sorted()
            .map { modelDirectory.appendingPathComponent($0) }
    }

    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    return enumerator.compactMap { item -> URL? in
        guard let url = item as? URL, url.pathExtension == "safetensors" else {
            return nil
        }
        return url
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    lazy: Bool = false,
    selection: WeightLoadingSelection? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for url in try safetensorWeightURLs(in: modelDirectory, selection: selection) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            if selection?.includes(key) ?? true {
                weights[key] = value
            }
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    if let selection {
        weights = Dictionary(uniqueKeysWithValues: weights.map { key, value in
            (selection.rewrite(key), value)
        })
    }

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    if !lazy {
        eval(model)
    }
}
