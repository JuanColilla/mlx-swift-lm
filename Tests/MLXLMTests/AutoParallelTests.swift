import MLXLMCommon
import XCTest

@testable import MLXLLM

final class AutoParallelTests: XCTestCase {
    func testSetLayersKeepsLlamaCacheMetadataAlignedWithLocalShard() {
        let configuration = LlamaConfiguration(
            hiddenSize: 32,
            hiddenLayers: 4,
            intermediateSize: 64,
            attentionHeads: 4,
            rmsNormEps: 0.00001,
            vocabularySize: 128,
            kvHeads: 2
        )
        let model = LlamaModel(configuration)
        let localLayers = Array(model.model.layers[1 ..< 3])

        setLayers(on: model, newLayers: localLayers, shardOffset: 1)

        XCTAssertEqual(model.model.layers.count, 2)
        XCTAssertEqual(model.kvHeads.count, 2)
        XCTAssertEqual(model.newCache(parameters: nil).count, 2)
    }
}
