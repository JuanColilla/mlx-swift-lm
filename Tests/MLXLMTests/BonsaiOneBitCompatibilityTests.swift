import MLX
import XCTest

final class BonsaiOneBitCompatibilityTests: XCTestCase {
    func testAffineOneBitGroup128RoundTripAndMatmul() throws {
        try Device.withDefaultDevice(.cpu) {
            let values: [Double] = (0 ..< 256).map { index in
                index.isMultiple(of: 2) ? -0.5 : 0.5
            }
            let weight = MLXArray(converting: values, [2, 128])
            let (quantizedWeight, scales, optionalBiases) = MLX.quantized(
                weight, groupSize: 128, bits: 1)
            let biases = try XCTUnwrap(optionalBiases)
            let reconstructed = dequantized(
                quantizedWeight, scales: scales, biases: biases,
                groupSize: 128, bits: 1)
            let input = MLXArray.ones([1, 128])
            let expected = matmul(input, reconstructed.T)
            let actual = quantizedMM(
                input, quantizedWeight, scales: scales, biases: biases,
                transpose: true, groupSize: 128, bits: 1)

            eval(quantizedWeight, scales, biases, reconstructed, expected, actual)

            XCTAssertEqual(quantizedWeight.dtype, .uint32)
            XCTAssertEqual(quantizedWeight.shape, [2, 4])
            XCTAssertTrue(allClose(reconstructed, weight, rtol: 0, atol: 0).item(Bool.self))
            XCTAssertTrue(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }
}
