// Copyright © 2026 Apple Inc.
//
// Golden tests for the LFM2-VL image preprocessing port.
//
// The expected values are produced by running the pure-math parts of the Python reference
// (`Lfm2VlImageProcessor.smart_resize` / `_is_image_too_large` / `_get_grid_layout` and
// `Lfm2VlProcessor._compute_tokens_*` / `_build_image_tokens`, transformers `models/lfm2_vl`)
// over the size grid below, for the three processor configurations that ship in the wild:
//
//   .mlx      LiquidAI/LFM2.5-VL-3B-MLX-4bit  (do_image_splitting false, use_thumbnail false)
//   .legacy   LiquidAI/LFM2-VL-1.6B           (splitting on, min_tiles 2)
//   .original LiquidAI/LFM2.5-VL-3B           (splitting on, min_tiles 1)
//
// Pixel values are out of scope: torchvision's resampling cannot be compared against
// CoreImage's without running torch. Geometry, token counts, patch order and the
// tokens-to-features invariant are.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class LFM2VLProcessorTests: XCTestCase {

    // MARK: - Fixtures

    /// Deterministic stand-in for the LFM2 vocabulary.
    ///
    /// Mirrors `BPETokenizer.convertTokenToId`, which returns the unknown id — never nil — for a
    /// token that is not in the vocabulary.
    private struct StubTokenizer: MLXLMCommon.Tokenizer {
        static let image = 900
        static let start = 901
        static let end = 902
        static let thumbnail = 903
        static let unknown = 999

        /// When false, the row/column markers resolve to the unknown id, as they would in a
        /// vocabulary that does not define them.
        let hasRowColumnTokens: Bool

        init(hasRowColumnTokens: Bool = true) {
            self.hasRowColumnTokens = hasRowColumnTokens
        }

        var bosToken: String? { "<|startoftext|>" }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { "<unk>" }

        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<unk>": return Self.unknown
            case "<image>": return Self.image
            case "<|image_start|>": return Self.start
            case "<|image_end|>": return Self.end
            case "<|img_thumbnail|>": return Self.thumbnail
            default: break
            }
            if hasRowColumnTokens, let (row, column) = Self.rowColumn(token) {
                return 1000 + (row - 1) * 10 + (column - 1)
            }
            return Self.unknown
        }

        func convertIdToToken(_ id: Int) -> String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

        /// Stands in for the LFM2 chat template, which emits exactly one `<image>` per image.
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens = [1]
            for message in messages {
                guard let content = message["content"] as? [[String: String]] else { continue }
                for item in content where item["type"] == "image" {
                    tokens.append(Self.image)
                }
            }
            tokens.append(2)
            return tokens
        }

        static func rowColumn(_ token: String) -> (Int, Int)? {
            guard token.hasPrefix("<|img_row_"), token.hasSuffix("|>") else { return nil }
            let body = token.dropFirst("<|img_row_".count).dropLast(2)
            let parts = body.components(separatedBy: "_col_")
            guard parts.count == 2, let row = Int(parts[0]), let column = Int(parts[1]) else {
                return nil
            }
            return (row, column)
        }

        /// Run-length description of an expansion, matching the golden `sequence` format.
        static func describe(_ tokens: [Int]) -> String {
            var parts = [String]()
            var run = 0
            func flush() {
                if run > 0 {
                    parts.append("IMG*\(run)")
                    run = 0
                }
            }
            for token in tokens {
                if token == image {
                    run += 1
                    continue
                }
                flush()
                switch token {
                case start: parts.append("START")
                case end: parts.append("END")
                case thumbnail: parts.append("THUMB")
                case 1000 ... 1099:
                    parts.append("R\((token - 1000) / 10 + 1)C\((token - 1000) % 10 + 1)")
                default: parts.append("?\(token)")
                }
            }
            flush()
            return parts.joined(separator: "|")
        }
    }

    private enum Checkpoint: String {
        /// `processor_config.json`, settings nested under `image_processor`.
        case mlx = """
            {
              "processor_class": "Lfm2VlProcessor",
              "image_processor": {
                "do_image_splitting": false,
                "downsample_factor": 2,
                "encoder_patch_size": 16,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "max_image_tokens": 256,
                "max_num_patches": 1024,
                "max_pixels_tolerance": 2.0,
                "min_image_tokens": 64,
                "tile_size": 512,
                "use_thumbnail": false
              }
            }
            """

        /// `preprocessor_config.json`, flat.
        case legacy = """
            {
              "processor_class": "Lfm2VlProcessor",
              "do_image_splitting": true,
              "downsample_factor": 2,
              "encoder_patch_size": 16,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "max_image_tokens": 256,
              "max_pixels_tolerance": 2.0,
              "max_tiles": 10,
              "min_image_tokens": 64,
              "min_tiles": 2,
              "tile_size": 512,
              "use_thumbnail": true
            }
            """

        case original = """
            {
              "processor_class": "Lfm2VlProcessor",
              "image_processor": {
                "do_image_splitting": true,
                "downsample_factor": 2,
                "encoder_patch_size": 16,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "max_image_tokens": 256,
                "max_pixels_tolerance": 2.0,
                "max_tiles": 10,
                "min_image_tokens": 64,
                "min_tiles": 1,
                "tile_size": 512,
                "use_thumbnail": true
              }
            }
            """
    }

    private func makeProcessor(
        _ checkpoint: Checkpoint, hasRowColumnTokens: Bool = true
    ) throws -> LFM2VLProcessor {
        let config = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: Data(checkpoint.rawValue.utf8))
        return LFM2VLProcessor(
            config, tokenizer: StubTokenizer(hasRowColumnTokens: hasRowColumnTokens))
    }

    private func makeImage(width: CGFloat, height: CGFloat, level: CGFloat = 0.5) -> CIImage {
        CIImage(color: CIColor(red: level, green: level, blue: level))
            .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    // MARK: - Golden table

    private struct Golden {
        let config: Checkpoint
        let height: Int
        let width: Int
        let isLarge: Bool
        let rows: Int
        let columns: Int
        let newHeight: Int
        let newWidth: Int
        /// Patch counts `[h, w]` of every unit fed to the encoder, in order.
        let units: [[Int]]
        let imageTokens: Int
        let sequence: String
    }

    private static let goldens: [Golden] = [
        Golden(
            config: .mlx, height: 768, width: 1024, isLarge: true,
            rows: 1, columns: 1, newHeight: 416, newWidth: 576,
            units: [[26, 36]], imageTokens: 234,
            sequence: "START|IMG*234|END"),
        Golden(
            config: .mlx, height: 1024, width: 768, isLarge: true,
            rows: 1, columns: 1, newHeight: 576, newWidth: 416,
            units: [[36, 26]], imageTokens: 234,
            sequence: "START|IMG*234|END"),
        Golden(
            config: .mlx, height: 512, width: 512, isLarge: false,
            rows: 1, columns: 1, newHeight: 512, newWidth: 512,
            units: [[32, 32]], imageTokens: 256,
            sequence: "START|IMG*256|END"),
        Golden(
            config: .mlx, height: 256, width: 256, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .mlx, height: 64, width: 64, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .mlx, height: 80, width: 80, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .mlx, height: 400, width: 512, isLarge: false,
            rows: 1, columns: 1, newHeight: 384, newWidth: 512,
            units: [[24, 32]], imageTokens: 192,
            sequence: "START|IMG*192|END"),
        Golden(
            config: .mlx, height: 720, width: 736, isLarge: false,
            rows: 1, columns: 1, newHeight: 480, newWidth: 512,
            units: [[30, 32]], imageTokens: 240,
            sequence: "START|IMG*240|END"),
        Golden(
            config: .mlx, height: 1200, width: 800, isLarge: true,
            rows: 1, columns: 1, newHeight: 608, newWidth: 416,
            units: [[38, 26]], imageTokens: 247,
            sequence: "START|IMG*247|END"),
        Golden(
            config: .mlx, height: 2000, width: 1000, isLarge: true,
            rows: 1, columns: 1, newHeight: 704, newWidth: 352,
            units: [[44, 22]], imageTokens: 242,
            sequence: "START|IMG*242|END"),
        Golden(
            config: .mlx, height: 100, width: 1600, isLarge: false,
            rows: 1, columns: 1, newHeight: 96, newWidth: 1600,
            units: [[6, 100]], imageTokens: 150,
            sequence: "START|IMG*150|END"),
        Golden(
            config: .mlx, height: 1600, width: 100, isLarge: false,
            rows: 1, columns: 1, newHeight: 1600, newWidth: 96,
            units: [[100, 6]], imageTokens: 150,
            sequence: "START|IMG*150|END"),
        Golden(
            config: .mlx, height: 33, width: 47, isLarge: false,
            rows: 1, columns: 1, newHeight: 224, newWidth: 320,
            units: [[14, 20]], imageTokens: 70,
            sequence: "START|IMG*70|END"),
        Golden(
            config: .legacy, height: 768, width: 1024, isLarge: true,
            rows: 2, columns: 3, newHeight: 416, newWidth: 576,
            units: [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [26, 36]],
            imageTokens: 1770,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R1C3|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R2C3|IMG*256|THUMB|IMG*234|END"
        ),
        Golden(
            config: .legacy, height: 1024, width: 768, isLarge: true,
            rows: 3, columns: 2, newHeight: 576, newWidth: 416,
            units: [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [36, 26]],
            imageTokens: 1770,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R3C1|IMG*256|R3C2|IMG*256|THUMB|IMG*234|END"
        ),
        Golden(
            config: .legacy, height: 512, width: 512, isLarge: false,
            rows: 1, columns: 1, newHeight: 512, newWidth: 512,
            units: [[32, 32]], imageTokens: 256,
            sequence: "START|IMG*256|END"),
        Golden(
            config: .legacy, height: 256, width: 256, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .legacy, height: 64, width: 64, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .legacy, height: 80, width: 80, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .legacy, height: 400, width: 512, isLarge: false,
            rows: 1, columns: 1, newHeight: 384, newWidth: 512,
            units: [[24, 32]], imageTokens: 192,
            sequence: "START|IMG*192|END"),
        Golden(
            config: .legacy, height: 720, width: 736, isLarge: false,
            rows: 1, columns: 1, newHeight: 480, newWidth: 512,
            units: [[30, 32]], imageTokens: 240,
            sequence: "START|IMG*240|END"),
        Golden(
            config: .legacy, height: 1200, width: 800, isLarge: true,
            rows: 3, columns: 2, newHeight: 608, newWidth: 416,
            units: [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [38, 26]],
            imageTokens: 1783,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R3C1|IMG*256|R3C2|IMG*256|THUMB|IMG*247|END"
        ),
        Golden(
            config: .legacy, height: 2000, width: 1000, isLarge: true,
            rows: 4, columns: 2, newHeight: 704, newWidth: 352,
            units: [
                [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32],
                [44, 22],
            ], imageTokens: 2290,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R3C1|IMG*256|R3C2|IMG*256|R4C1|IMG*256|R4C2|IMG*256|THUMB|IMG*242|END"
        ),
        Golden(
            config: .legacy, height: 100, width: 1600, isLarge: false,
            rows: 1, columns: 1, newHeight: 96, newWidth: 1600,
            units: [[6, 100]], imageTokens: 150,
            sequence: "START|IMG*150|END"),
        Golden(
            config: .legacy, height: 1600, width: 100, isLarge: false,
            rows: 1, columns: 1, newHeight: 1600, newWidth: 96,
            units: [[100, 6]], imageTokens: 150,
            sequence: "START|IMG*150|END"),
        Golden(
            config: .legacy, height: 33, width: 47, isLarge: false,
            rows: 1, columns: 1, newHeight: 224, newWidth: 320,
            units: [[14, 20]], imageTokens: 70,
            sequence: "START|IMG*70|END"),
        Golden(
            config: .original, height: 768, width: 1024, isLarge: true,
            rows: 2, columns: 3, newHeight: 416, newWidth: 576,
            units: [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [26, 36]],
            imageTokens: 1770,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R1C3|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R2C3|IMG*256|THUMB|IMG*234|END"
        ),
        Golden(
            config: .original, height: 1024, width: 768, isLarge: true,
            rows: 3, columns: 2, newHeight: 576, newWidth: 416,
            units: [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [36, 26]],
            imageTokens: 1770,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R3C1|IMG*256|R3C2|IMG*256|THUMB|IMG*234|END"
        ),
        Golden(
            config: .original, height: 512, width: 512, isLarge: false,
            rows: 1, columns: 1, newHeight: 512, newWidth: 512,
            units: [[32, 32]], imageTokens: 256,
            sequence: "START|IMG*256|END"),
        Golden(
            config: .original, height: 256, width: 256, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .original, height: 64, width: 64, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .original, height: 80, width: 80, isLarge: false,
            rows: 1, columns: 1, newHeight: 256, newWidth: 256,
            units: [[16, 16]], imageTokens: 64,
            sequence: "START|IMG*64|END"),
        Golden(
            config: .original, height: 400, width: 512, isLarge: false,
            rows: 1, columns: 1, newHeight: 384, newWidth: 512,
            units: [[24, 32]], imageTokens: 192,
            sequence: "START|IMG*192|END"),
        Golden(
            config: .original, height: 720, width: 736, isLarge: false,
            rows: 1, columns: 1, newHeight: 480, newWidth: 512,
            units: [[30, 32]], imageTokens: 240,
            sequence: "START|IMG*240|END"),
        Golden(
            config: .original, height: 1200, width: 800, isLarge: true,
            rows: 3, columns: 2, newHeight: 608, newWidth: 416,
            units: [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [38, 26]],
            imageTokens: 1783,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R3C1|IMG*256|R3C2|IMG*256|THUMB|IMG*247|END"
        ),
        Golden(
            config: .original, height: 2000, width: 1000, isLarge: true,
            rows: 4, columns: 2, newHeight: 704, newWidth: 352,
            units: [
                [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32],
                [44, 22],
            ], imageTokens: 2290,
            sequence:
                "START|R1C1|IMG*256|R1C2|IMG*256|R2C1|IMG*256|R2C2|IMG*256|R3C1|IMG*256|R3C2|IMG*256|R4C1|IMG*256|R4C2|IMG*256|THUMB|IMG*242|END"
        ),
        Golden(
            config: .original, height: 100, width: 1600, isLarge: false,
            rows: 1, columns: 1, newHeight: 96, newWidth: 1600,
            units: [[6, 100]], imageTokens: 150,
            sequence: "START|IMG*150|END"),
        Golden(
            config: .original, height: 1600, width: 100, isLarge: false,
            rows: 1, columns: 1, newHeight: 1600, newWidth: 96,
            units: [[100, 6]], imageTokens: 150,
            sequence: "START|IMG*150|END"),
        Golden(
            config: .original, height: 33, width: 47, isLarge: false,
            rows: 1, columns: 1, newHeight: 224, newWidth: 320,
            units: [[14, 20]], imageTokens: 70,
            sequence: "START|IMG*70|END"),
    ]

    // MARK: - Tests

    /// Geometry, token counts and placeholder sequence must match the Python reference exactly.
    func testGoldenLayoutAndTokens() throws {
        for golden in Self.goldens {
            let processor = try makeProcessor(golden.config)
            let label = "\(golden.config) \(golden.height)x\(golden.width)"

            XCTAssertEqual(
                processor.isImageTooLarge(height: golden.height, width: golden.width),
                golden.isLarge, "isImageTooLarge: \(label)")

            let layout = processor.layout(height: golden.height, width: golden.width)
            XCTAssertEqual(layout.rows, golden.rows, "rows: \(label)")
            XCTAssertEqual(layout.columns, golden.columns, "columns: \(label)")
            XCTAssertEqual(Int(layout.resized.height), golden.newHeight, "height: \(label)")
            XCTAssertEqual(Int(layout.resized.width), golden.newWidth, "width: \(label)")

            let patchSize = 16
            let units = layout.units.map {
                [Int($0.height) / patchSize, Int($0.width) / patchSize]
            }
            XCTAssertEqual(units, golden.units, "units: \(label)")

            let expanded = processor.expand(layout: layout, imageTokenId: StubTokenizer.image)
            XCTAssertEqual(
                StubTokenizer.describe(expanded.tokens), golden.sequence, "sequence: \(label)")
            XCTAssertEqual(
                expanded.imageTokenCount, golden.imageTokens, "image tokens: \(label)")
        }
    }

    /// Every emitted image token must be backed by exactly one row of encoder features:
    /// `ceil(h / downsample) · ceil(w / downsample)` per unit. This is the invariant whose
    /// violation the model reports with a `fatalError`.
    func testTokensMatchFeatureRows() throws {
        for golden in Self.goldens {
            let rows = golden.units.reduce(0) { $0 + ($1[0] + 1) / 2 * (($1[1] + 1) / 2) }
            XCTAssertEqual(
                rows, golden.imageTokens,
                "\(golden.config) \(golden.height)x\(golden.width)")
        }
    }

    /// `round_by_factor` uses Python's half-to-even `round`. With Swift's default rounding
    /// 720 would become 736 instead of 704, which both flips the tiling threshold and changes
    /// the token budget.
    func testHalfToEvenRounding() throws {
        let processor = try makeProcessor(.legacy)

        // 720 / 32 = 22.5 -> 704 (not 736), so 704*736 stays under the tolerance.
        XCTAssertFalse(processor.isImageTooLarge(height: 720, width: 736))
        XCTAssertEqual(processor.layout(height: 720, width: 736).rows, 1)

        // 400 / 32 = 12.5 -> 384 (not 416): 192 tokens, not 208.
        let resized = processor.smartResize(height: 400, width: 512)
        XCTAssertEqual(resized.height, 384)
        XCTAssertEqual(resized.width, 512)
        XCTAssertEqual(processor.tokensForImage(height: 384, width: 512), 192)
    }

    /// The image processor settings are nested under `image_processor` in LFM2.5-VL's
    /// `processor_config.json` and flat in LFM2-VL's `preprocessor_config.json`. Reading only
    /// the flat layout would leave `doImageSplitting` at its default — the opposite of what the
    /// nested config declares.
    func testProcessorConfigDecoding() throws {
        let nested = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: Data(Checkpoint.mlx.rawValue.utf8))
        XCTAssertFalse(nested.doImageSplitting)
        XCTAssertFalse(nested.useThumbnail)
        XCTAssertEqual(nested.tileSize, 512)
        XCTAssertEqual(nested.maxImageTokens, 256)

        let flat = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: Data(Checkpoint.legacy.rawValue.utf8))
        XCTAssertTrue(flat.doImageSplitting)
        XCTAssertTrue(flat.useThumbnail)
        XCTAssertEqual(flat.minTiles, 2)
        XCTAssertEqual(flat.maxTiles, 10)

        // Anything absent falls back to the Python class defaults.
        let empty = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: Data("{}".utf8))
        XCTAssertTrue(empty.doImageSplitting)
        XCTAssertTrue(empty.useThumbnail)
        XCTAssertEqual(empty.minTiles, 2)
        XCTAssertEqual(empty.maxTiles, 10)
        XCTAssertEqual(empty.minImageTokens, 64)
        XCTAssertEqual(empty.maxImageTokens, 256)
        XCTAssertEqual(empty.maxPixelsTolerance, 2.0)
        XCTAssertTrue(empty.useImageSpecialTokens)
        XCTAssertEqual(empty.imageToken, "<image>")
    }

    /// Patches must be extracted inside each tile, in row-major tile order and with the top-left
    /// tile first. Walking the resized image in global raster order instead hands the encoder its
    /// patches shuffled with respect to training.
    func testTileOrderAndOrigin() throws {
        let processor = try makeProcessor(.legacy)

        // Distinct grey levels, so each unit can be identified by its mean.
        let levels: [CGFloat] = [0.1, 0.3, 0.6, 0.9]
        let half: CGFloat = 512
        let quadrants = [
            CGRect(x: 0, y: half, width: half, height: half),  // top-left
            CGRect(x: half, y: half, width: half, height: half),  // top-right
            CGRect(x: 0, y: 0, width: half, height: half),  // bottom-left
            CGRect(x: half, y: 0, width: half, height: half),  // bottom-right
        ]
        var image = CIImage(color: CIColor(red: 0, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 1024, height: 1024))
        for (level, rect) in zip(levels, quadrants) {
            let block = CIImage(color: CIColor(red: level, green: level, blue: level))
                .cropped(to: rect)
            image = block.composited(over: image)
        }

        let layout = processor.layout(height: 1024, width: 1024)
        XCTAssertEqual(layout.rows, 2)
        XCTAssertEqual(layout.columns, 2)
        XCTAssertEqual(layout.units.count, 5, "4 tiles + thumbnail")

        let units = processor.units(from: image, layout: layout)
        XCTAssertEqual(units.count, 5)

        // Normalization maps a grey level l to (l - 0.5) / 0.5.
        for (index, level) in levels.enumerated() {
            let expected = Float((level - 0.5) / 0.5)
            let actual = mean(units[index]).item(Float.self)
            XCTAssertEqual(
                actual, expected, accuracy: 0.05,
                "tile \(index) should hold the quadrant with level \(level)")
        }

        // The thumbnail covers the whole image, so its mean is the mean of the four levels.
        let thumbnailExpected = Float(
            (levels.reduce(0, +) / CGFloat(levels.count) - 0.5) / 0.5)
        XCTAssertEqual(
            mean(units[4]).item(Float.self), thumbnailExpected, accuracy: 0.05,
            "the thumbnail is the whole image")

        // Patch 0 of the first tile must come from that tile, not from the global raster order.
        let patches = processor.patches(from: units[0])
        XCTAssertEqual(patches.dim(0), 32 * 32)
        XCTAssertEqual(patches.dim(1), 16 * 16 * 3)
        XCTAssertEqual(
            mean(patches[0]).item(Float.self), Float((levels[0] - 0.5) / 0.5), accuracy: 0.05)
    }

    /// Two images with different aspect ratios produce units of different patch counts. They must
    /// be padded to a common length before stacking; the pre-port code fed them straight to
    /// `concatenated`, which requires matching shapes.
    func testMultipleImagesWithDifferentShapes() async throws {
        let processor = try makeProcessor(.mlx)
        let input = UserInput(
            prompt: "describe",
            images: [
                // 6x100 = 600 patches, 150 tokens.
                .ciImage(makeImage(width: 1600, height: 100, level: 0.2)),
                // 32x32 = 1024 patches, 256 tokens.
                .ciImage(makeImage(width: 512, height: 512, level: 0.8)),
            ])

        let prepared = try await processor.prepare(input: input)
        let image = try XCTUnwrap(prepared.image)
        let frames = try XCTUnwrap(image.frames)

        XCTAssertEqual(frames.count, 2, "one unit per image with splitting disabled")
        XCTAssertEqual(image.pixels.dim(0), 2)
        XCTAssertEqual(image.pixels.dim(1), 1024, "padded to the larger of the two")
        XCTAssertEqual(frames.map(\.h), [6, 32])
        XCTAssertEqual(frames.map(\.w), [100, 32])

        let tokens = prepared.text.tokens.asArray(Int.self)
        let imageTokens = tokens.filter { $0 == StubTokenizer.image }.count
        XCTAssertEqual(imageTokens, 150 + 256)
        XCTAssertEqual(tokens.filter { $0 == StubTokenizer.start }.count, 2)
        XCTAssertEqual(tokens.filter { $0 == StubTokenizer.end }.count, 2)

        let featureRows = frames.reduce(0) { $0 + ($1.h + 1) / 2 * (($1.w + 1) / 2) }
        XCTAssertEqual(imageTokens, featureRows)
    }

    /// Tiles and thumbnail have different patch counts, so the stacked pixel array is padded and
    /// the frames keep each unit's real size — that difference is what the vision attention mask
    /// and the per-unit unpadding in `getInputEmbeddings` rely on.
    func testTilesAndThumbnailArePadded() async throws {
        let processor = try makeProcessor(.legacy)
        let input = UserInput(
            prompt: "describe", images: [.ciImage(makeImage(width: 1024, height: 768))])

        let prepared = try await processor.prepare(input: input)
        let image = try XCTUnwrap(prepared.image)
        let frames = try XCTUnwrap(image.frames)

        XCTAssertEqual(frames.count, 7, "6 tiles + thumbnail")
        XCTAssertEqual(image.pixels.dim(0), 7)
        XCTAssertEqual(image.pixels.dim(1), 32 * 32, "padded to the widest unit")
        XCTAssertEqual(frames.last?.h, 26)
        XCTAssertEqual(frames.last?.w, 36)

        let tokens = prepared.text.tokens.asArray(Int.self)
        XCTAssertEqual(tokens.filter { $0 == StubTokenizer.image }.count, 1770)
        XCTAssertEqual(tokens.filter { $0 == StubTokenizer.thumbnail }.count, 1)
    }

    /// Markers that the vocabulary does not define are skipped, and the image token count — the
    /// only quantity the feature merge checks — stays the same.
    func testMissingMarkersDoNotChangeTheImageTokenCount() throws {
        let withMarkers = try makeProcessor(.legacy)
        let withoutMarkers = try makeProcessor(.legacy, hasRowColumnTokens: false)
        let layout = withMarkers.layout(height: 1024, width: 768)

        let complete = withMarkers.expand(layout: layout, imageTokenId: StubTokenizer.image)
        let degraded = withoutMarkers.expand(layout: layout, imageTokenId: StubTokenizer.image)

        XCTAssertEqual(complete.imageTokenCount, degraded.imageTokenCount)
        XCTAssertEqual(complete.tokens.count - degraded.tokens.count, 6, "6 row/column markers")
        XCTAssertFalse(degraded.tokens.contains(StubTokenizer.unknown))
    }

    /// A prompt whose placeholders do not match the images must fail loudly instead of silently
    /// dropping tokens.
    func testPlaceholderMismatchThrows() async throws {
        struct NoPlaceholderTokenizer: MLXLMCommon.Tokenizer {
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { "<unk>" }
            func convertTokenToId(_ token: String) -> Int? {
                token == "<image>" ? StubTokenizer.image : StubTokenizer.unknown
            }
            func convertIdToToken(_ id: Int) -> String? { nil }
            func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                [1, 2]  // no placeholder at all
            }
        }

        let config = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: Data(Checkpoint.mlx.rawValue.utf8))
        let processor = LFM2VLProcessor(config, tokenizer: NoPlaceholderTokenizer())
        let input = UserInput(
            prompt: "describe", images: [.ciImage(makeImage(width: 512, height: 512))])

        do {
            _ = try await processor.prepare(input: input)
            XCTFail("expected a placeholder mismatch")
        } catch {
            XCTAssertTrue(
                "\(error)".contains("placeholder"), "unexpected error: \(error)")
        }
    }
}
