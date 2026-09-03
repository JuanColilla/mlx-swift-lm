// FORK(JuanColilla): R-56 task 1.7 — P-iso acceptance gate.
//
// SwiftLM#84 is the precedent this guards against: a per-layer synchronization
// gate needed only by the streaming path leaked into the resident path and
// degraded it 3.4x. The resident MoE decode path (`Qwen35SparseMoeBlock`, its
// compiled trace, and `SwitchGLU`) must measure the same before and after the
// expert-streaming module exists in the binary.
//
// Run with:
//
// ```sh
// MLX_R56_BENCHMARKS=1 swift test --filter ExpertStreamingIsolationBenchmark
// ```
//
// The absolute number is machine-specific and deliberately not asserted; the
// comparison is between two runs of this same file on the same machine.
//
// XCTest, not Swift Testing: the swift-testing runner in this checkout fails
// to locate MLX's default metallib, which takes down every MLX-touching test
// under it (reproduced on the untouched tree with MixedPrecisionQuantLoadTests).

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class ExpertStreamingIsolationBenchmark: XCTestCase {

    /// Qwen 3.5 MoE geometry scaled down so the block fits a benchmark: same
    /// top-8 routing, same shared expert, same quantization group size.
    static let tinyMoEConfigJSON = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": 256,
          "num_hidden_layers": 4,
          "intermediate_size": 512,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 64,
          "linear_num_value_heads": 8,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "vocab_size": 512,
          "full_attention_interval": 4,
          "num_experts": 32,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 128,
          "shared_expert_intermediate_size": 128,
          "norm_topk_prob": true,
          "tie_word_embeddings": false
        }
        """

    static func tinyMoEConfiguration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(tinyMoEConfigJSON.utf8))
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    /// Decode-shaped steps per second through the resident MoE block.
    ///
    /// One `eval` per step, matching a real decode loop: the point is to
    /// include the host/GPU round trip, since that is what a streaming
    /// regression would inflate.
    static func residentDecodeStepsPerSecond(
        steps: Int = 200, repetitions: Int = 5
    ) throws -> Double {
        MLXRandom.seed(56)
        let configuration = try tinyMoEConfiguration()
        let block = Qwen35SparseMoeBlock(configuration)
        quantize(model: block, groupSize: 64, bits: 4)
        eval(block)

        let x = MLXRandom.normal([1, 1, configuration.hiddenSize]).asType(.bfloat16)
        eval(x)

        // Warm-up: builds the compiled trace and the Metal pipelines.
        for _ in 0 ..< 20 {
            eval(block(x))
        }

        var rates = [Double]()
        for _ in 0 ..< repetitions {
            let start = Date.timeIntervalSinceReferenceDate
            for _ in 0 ..< steps {
                eval(block(x))
            }
            let elapsed = Date.timeIntervalSinceReferenceDate - start
            rates.append(Double(steps) / elapsed)
        }
        return median(rates)
    }

    func testResidentMoEDecodeThroughput() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_R56_BENCHMARKS"] == "1",
            "opt-in benchmark")

        let rate = try Self.residentDecodeStepsPerSecond()
        print("P-iso resident MoE decode: \(String(format: "%.1f", rate)) block-steps/s")
    }

    /// Correctness companion to the throughput number: the resident block must
    /// keep producing the same values, so a "no regression" claim cannot be
    /// satisfied by a path that quietly stopped computing.
    func testResidentMoEDecodeIsDeterministic() throws {
        MLXRandom.seed(56)
        let configuration = try Self.tinyMoEConfiguration()
        let block = Qwen35SparseMoeBlock(configuration)
        quantize(model: block, groupSize: 64, bits: 4)
        eval(block)

        let x = MLXRandom.normal([1, 1, configuration.hiddenSize]).asType(.bfloat16)
        let first = block(x)
        let second = block(x)
        eval(first, second)

        XCTAssertEqual(first.shape, [1, 1, configuration.hiddenSize])
        XCTAssertTrue(allClose(first, second).item(Bool.self))
    }
}
