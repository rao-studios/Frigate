//
//  RegionBackboneNet.swift
//  VisionAX
//
//  WHAT: The classifier's backbone as an MLX module — ResNet-18 through layer3, with layer2
//        and layer3 fused at stride 8 (a one-level FPN).
//  IN:   weights converted from the ONNX backbone by VisionAX's `vxtrain export-mlx`
//  OUT:  MLXRegionBackbone
//  PIN:  THE ONNX GRAPH IS THE SPECIFICATION, not the PyTorch source. BatchNorm is already
//        folded into every convolution there, so every Conv2d here has a bias and there is
//        no normalization layer to get subtly wrong. Parameter keys mirror the torch module
//        paths the converter recovers from the graph's node names (`layer2.0.downsample.0`),
//        and `verify: .all` refuses any file that disagrees by a single key or shape.
//        NHWC THROUGHOUT, because that is MLX's convolution layout; the caller transposes
//        once on the way in and once on the way out.
//        THE STEM'S POOL IS PADDED HERE, NOT BY MaxPool2d. The vendored MLXNN pads a pooling
//        input with `[0, 0] + padding + [0, 0]` — six widths for a four-axis NHWC tensor —
//        which lands the padding on width and CHANNELS: measured, the first block received
//        (1, 15, 16, 66) instead of (1, 16, 16, 64) and MLX aborted. One pixel of -infinity on
//        height and width, then an unpadded 3×3/2 pool, is exactly ONNX's MaxPool (pads 1),
//        and the ×2 upsample is nearest at an integer scale — ONNX's Resize (nearest,
//        asymmetric, floor).
//

import Foundation
import MLX
import MLXNN

/// Two 3×3 convolutions and a shortcut — torchvision's BasicBlock with BatchNorm folded away.
final class RegionBackboneBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    /// Present when the block changes stride or width: a strided 1×1 projection.
    @ModuleInfo(key: "downsample") var downsample: [Conv2d]

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: 3, stride: IntOrPair(stride), padding: 1)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outputChannels, outputChannels: outputChannels,
            kernelSize: 3, stride: 1, padding: 1)
        self._downsample.wrappedValue =
            stride != 1 || inputChannels != outputChannels
            ? [
                Conv2d(
                    inputChannels: inputChannels, outputChannels: outputChannels,
                    kernelSize: 1, stride: IntOrPair(stride), padding: 0)
            ]
            : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = downsample.first.map { $0(x) } ?? x
        return relu(conv2(relu(conv1(x))) + shortcut)
    }
}

/// image [N, H, W, 3] → features [N, H/8, W/8, channels].
final class RegionBackboneNet: Module, UnaryLayer {
    @ModuleInfo(key: "stem") var stem: [Conv2d]
    @ModuleInfo(key: "layer1") var layer1: [RegionBackboneBlock]
    @ModuleInfo(key: "layer2") var layer2: [RegionBackboneBlock]
    @ModuleInfo(key: "layer3") var layer3: [RegionBackboneBlock]
    @ModuleInfo(key: "lateral8") var lateral8: Conv2d
    @ModuleInfo(key: "lateral16") var lateral16: Conv2d
    @ModuleInfo(key: "smooth") var smooth: Conv2d

    private let pool = MaxPool2d(kernelSize: 3, stride: 2)
    private let upsample = Upsample(scaleFactor: 2.0, mode: .nearest)

    init(channels: Int) {
        self._stem.wrappedValue = [
            Conv2d(inputChannels: 3, outputChannels: 64, kernelSize: 7, stride: 2, padding: 3)
        ]
        self._layer1.wrappedValue = [
            RegionBackboneBlock(inputChannels: 64, outputChannels: 64, stride: 1),
            RegionBackboneBlock(inputChannels: 64, outputChannels: 64, stride: 1),
        ]
        self._layer2.wrappedValue = [
            RegionBackboneBlock(inputChannels: 64, outputChannels: 128, stride: 2),
            RegionBackboneBlock(inputChannels: 128, outputChannels: 128, stride: 1),
        ]
        self._layer3.wrappedValue = [
            RegionBackboneBlock(inputChannels: 128, outputChannels: 256, stride: 2),
            RegionBackboneBlock(inputChannels: 256, outputChannels: 256, stride: 1),
        ]
        self._lateral8.wrappedValue = Conv2d(
            inputChannels: 128, outputChannels: channels, kernelSize: 1)
        self._lateral16.wrappedValue = Conv2d(
            inputChannels: 256, outputChannels: channels, kernelSize: 1)
        self._smooth.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let stemmed = relu(stem[0](image))
        // The pad value is built per call, not stored: MLXNN reflects every stored MLXArray
        // as a PARAMETER, and `verify: .all` would then demand it from the weights file.
        let edge = MLXArray(-Float.infinity)
        var x = pool(padded(stemmed, widths: [0, 1, 1, 0], mode: .constant, value: edge))
        for block in layer1 { x = block(x) }
        var stride8 = x
        for block in layer2 { stride8 = block(stride8) }
        var stride16 = stride8
        for block in layer3 { stride16 = block(stride16) }
        return smooth(lateral8(stride8) + upsample(lateral16(stride16)))
    }
}
