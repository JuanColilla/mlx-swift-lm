// FORK(JuanColilla): R-56 — shared fixtures for the expert-streaming tests.
//
// Writes real safetensors files (8-byte length field, JSON header, payload) so
// the index and the store are exercised against the same bytes a checkpoint
// would present, not against a mock.

import Foundation

enum SyntheticExpertCheckpoint {

    struct Tensor {
        let name: String
        let dtype: String
        let shape: [Int]

        var itemSize: Int {
            switch dtype {
            case "U8": 1
            case "BF16", "F16": 2
            default: 4
            }
        }

        var byteCount: Int { shape.reduce(1, *) * itemSize }
    }

    /// Deterministic payload: byte `i` of tensor `t` is a function of both, so
    /// a read that lands on the wrong tensor or the wrong offset cannot pass
    /// by accident.
    static func payloadByte(tensor: Int, offset: Int) -> UInt8 {
        UInt8(truncatingIfNeeded: (tensor &* 131 &+ offset &* 17) &+ (offset >> 8))
    }

    /// Writes the file and returns the payload of each tensor, in the order
    /// given, so a test can assert byte equality without re-deriving it.
    @discardableResult
    static func write(_ tensors: [Tensor], to url: URL) throws -> [Data] {
        var header = [String: Any]()
        var payloads = [Data]()
        var offset = 0

        for (position, tensor) in tensors.enumerated() {
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + tensor.byteCount],
            ]
            var payload = Data(count: tensor.byteCount)
            payload.withUnsafeMutableBytes { raw in
                let bytes = raw.bindMemory(to: UInt8.self)
                for i in 0 ..< tensor.byteCount {
                    bytes[i] = payloadByte(tensor: position, offset: i)
                }
            }
            payloads.append(payload)
            offset += tensor.byteCount
        }

        let headerData = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        var file = Data()
        withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(headerData)
        for payload in payloads { file.append(payload) }
        try file.write(to: url)

        return payloads
    }

    /// The nine stacked `switch_mlp` tensors of `layers` layers, sized so every
    /// expert row is exactly one 16 KiB page.
    static func wellFormedTensors(experts: Int, layers: Int) -> [Tensor] {
        var tensors = [Tensor]()
        for layer in 0 ..< layers {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let base = "model.layers.\(layer).mlp.switch_mlp.\(projection)"
                tensors.append(Tensor(name: "\(base).weight", dtype: "U32", shape: [experts, 64, 64]))
                tensors.append(
                    Tensor(name: "\(base).scales", dtype: "BF16", shape: [experts, 128, 64]))
                tensors.append(
                    Tensor(name: "\(base).biases", dtype: "BF16", shape: [experts, 128, 64]))
            }
            tensors.append(
                Tensor(name: "model.layers.\(layer).mlp.gate.weight", dtype: "BF16",
                    shape: [experts, 16]))
        }
        return tensors
    }

    @discardableResult
    static func writeWellFormed(experts: Int, layers: Int, to directory: URL) throws -> [Data] {
        try write(
            wellFormedTensors(experts: experts, layers: layers),
            to: directory.appending(path: "model.safetensors"))
    }

    /// A checkpoint shaped like K2-Horizon MoVA: dense layers first, then
    /// sparse layers carrying both `mlp.switch_mlp.*` (MLP family) and
    /// `self_attn.switch_v.*` (value family), with the routers and the shared
    /// expert alongside so the parser has something to ignore.
    ///
    /// The MLP `scales`/`biases` rows are 8 KiB and the value rows 12 KiB and
    /// 4 KiB — none a page multiple, like the real checkpoint's MLP scales —
    /// so an index over this fixture exercises the padded staging path.
    static func movaTensors(
        mlpExperts: Int, valueExperts: Int, denseLayers: Int, totalLayers: Int
    ) -> [Tensor] {
        var tensors = [Tensor]()
        for layer in 0 ..< totalLayers {
            let prefix = "model.layers.\(layer)"
            tensors.append(
                Tensor(name: "\(prefix).self_attn.q_proj.weight", dtype: "U32", shape: [64, 64]))
            guard layer >= denseLayers else {
                tensors.append(
                    Tensor(name: "\(prefix).mlp.gate_proj.weight", dtype: "U32", shape: [64, 64]))
                tensors.append(
                    Tensor(name: "\(prefix).self_attn.v_proj.weight", dtype: "U32", shape: [64, 64]))
                continue
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let base = "\(prefix).mlp.switch_mlp.\(projection)"
                tensors.append(
                    Tensor(name: "\(base).weight", dtype: "U32", shape: [mlpExperts, 64, 64]))
                tensors.append(
                    Tensor(name: "\(base).scales", dtype: "BF16", shape: [mlpExperts, 64, 64]))
                tensors.append(
                    Tensor(name: "\(base).biases", dtype: "BF16", shape: [mlpExperts, 64, 64]))
                tensors.append(
                    Tensor(
                        name: "\(prefix).mlp.shared_experts.\(projection).weight", dtype: "U32",
                        shape: [64, 64]))
            }
            tensors.append(
                Tensor(name: "\(prefix).mlp.gate.weight", dtype: "BF16", shape: [mlpExperts, 16]))
            tensors.append(
                Tensor(name: "\(prefix).mlp.gate.bias", dtype: "BF16", shape: [mlpExperts]))
            let value = "\(prefix).self_attn.switch_v"
            tensors.append(
                Tensor(name: "\(value).weight", dtype: "U32", shape: [valueExperts, 48, 64]))
            tensors.append(
                Tensor(name: "\(value).scales", dtype: "BF16", shape: [valueExperts, 32, 64]))
            tensors.append(
                Tensor(name: "\(value).biases", dtype: "BF16", shape: [valueExperts, 32, 64]))
            tensors.append(
                Tensor(
                    name: "\(prefix).self_attn.v_router.weight", dtype: "BF16",
                    shape: [valueExperts, 16]))
        }
        return tensors
    }

    /// Index into the tensor list produced by ``wellFormedTensors(experts:layers:)``.
    static func position(layer: Int, projection: Int, component: Int) -> Int {
        layer * 10 + projection * 3 + component
    }
}
